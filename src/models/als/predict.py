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
parser.add_argument('run_id', type=str, 
    help='Run ID for saving models')
parser.add_argument('dataset', choices=['small_100', 'small_all', 'full'],
    help='which dataset to predict on (small_100, small_all, all)')
parser.add_argument('split', choices=['train', 'val', 'test'],
    help='which split of the dataset to predict on (train, val, test)')

args = parser.parse_args()
data_path = 'data/raw'

def __main__():
    sc = SparkContext('local[*]')
    spark = SparkSession(sc)
    data = spark.read.parquet(os.path.join(data_path, args.dataset, args.split))\
        .select('user_id', 'hotel_id', 'search_request_id')\
        .where(col('user_id').isNotNull())

    # ALS cannot handle bigint, need to feed in int
    data = data.withColumn('user_id_int', (data['user_id'] - 10000000000).cast(IntegerType()))

    with mlflow.start_run(run_id=args.run_id):    
        model= mlflow.spark.load_model('runs:/{}/model'.format(args.run_id))

        pred = model.transform(data)

        pred = pred.withColumn('score', pred['prediction'])\
            .drop('prediction', 'user_id')

        predictions_path = 'predictions/{}/{}/{}'.format(args.run_id, args.dataset, args.split)
        if not os.path.exists(predictions_path):
            os.makedirs(predictions_path)
        predictions_file = os.path.join(predictions_path, 'predictions.parquet')

        pred.write.mode('overwrite').parquet(predictions_file)

if __name__ == '__main__':
    __main__()