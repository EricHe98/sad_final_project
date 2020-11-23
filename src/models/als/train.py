# findspark finds the Spark home
# since Spark cant seem to do it itself
# needs to be run before importing Spark
import findspark
findspark.init()
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics, RegressionMetrics
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number, desc, collect_list
from pyspark.sql.types import IntegerType
import os
import json
import time
import datetime as dt
import mlflow
import argparse

# You can type `python3 SCRIPT_NAME --help` and argparse will auto-generate 
# documentation of what arguments the script accepts
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['small_100', 'small_all', 'full'],
    help='which dataset to train the model (small_100, small_all, all) to be included in MLFlow name')
parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
    help='which split of the dataset to train on (train, val, test)')

args = parser.parse_args()
data_path = 'data/raw'

def __main__():
    sc = SparkContext('local[*]')
    spark = SparkSession(sc)
    data = spark.read.parquet(os.path.join(data_path, args.dataset, args.split))\
        .select('user_id', 'hotel_id', 'label')\
        .where(col('user_id').isNotNull())\
        .where(col('label') > 0)

    # ALS cannot handle bigint, need to feed in int
    data = data.withColumn('user_id_int', (data['user_id'] - 10000000000).cast(IntegerType()))

    RANK = 32
    REG = 0.1
    NONNEGATIVE = False
    COLDSTART = 'drop'

    # using `with` scopes the MLFlow run to start and end in this block
    # you can alternatively just use `mlflow.start_run(blah)` to start
    # and `mlflow.end_run()` to end
    # Every model train should have its own MLFlow run
    with mlflow.start_run(run_name='als'):
        # MLFlow assigns a run_id to every model train
        # We need to hang onto it for the prediction and evaluation steps
        run_id = mlflow.active_run().info.run_id
        print('MLFlow Run ID is: {}'.format(run_id))
        
        # MLFlow is used simply because it gives us a nice UI to log params and metrics
        # I would like us to log at least three things for training: run_type, model_name,
        # and training_time. You have wide latitude to choose model_name; try to keep
        # model_name the same for models with the same hyperparameters.
        mlflow.log_param('dataset', args.dataset)
        mlflow.log_param('train_split', args.split)
        mlflow.log_param('model_name', 'ALS')
        mlflow.log_param('run_id', run_id)
        
        als = ALS()\
            .setUserCol('user_id_int')\
            .setRatingCol('label')\
            .setItemCol('hotel_id')\
            .setRank(RANK)\
            .setRegParam(REG)\
            .setNonnegative(NONNEGATIVE)\
            .setColdStartStrategy(COLDSTART)
        # If it can feasibly be used in hyperparameter optimization,
        # it should be logged to MLFlow.
        mlflow.log_param('cold_start', COLDSTART)
        mlflow.log_param('rank', RANK)
        mlflow.log_param('reg', REG)
        mlflow.log_param('nonnegative', NONNEGATIVE)

        # I calculate train time this way, probably best to keep
        # to this standard for consistency
        time_start = dt.datetime.now()
        model = als.fit(data)
        time_end = dt.datetime.now()

        train_time = (time_end - time_start).total_seconds()
        mlflow.log_metric('training_time', train_time)
        print('Model trained in {}'.format(train_time))
        # writes model file to MLFlow directory 
        # under {run_id}/artifacts/als
        # in the MLFlow model format
        # can be loaded in using mlflow.spark.load_model
        # see https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.load_model
        # for loading options; I use the "runs" method to load in the predict script
        mlflow.spark.log_model(model, 'als') 

if __name__ == '__main__':
    __main__()