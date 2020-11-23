import sys
import s3
import json
import argparse
import boto3
import search_result_ranking.config as c
from connections.db_class import RedshiftDB
from connections.db_config import rs_de_conn, get_cred
import pandas as pd
import awswrangler as wr
import os
sys.path.append('src')
from src.data.adhoc_downloads import adhoc_download_modules as a

rs_de = get_cred('airflow-rs-dataengineer')
wr_con = wr.db.get_engine(db_type='redshift',
    host=rs_de['host'],
    port=rs_de['port'],
    database=rs_de['database'],
    user=rs_de['user'],
    password=rs_de['password'],
    connect_args={'sslmode':'verify-ca'})

unload_query = """
    SELECT {FEATURES}
    FROM {SCHEMA}.srr_train
    WHERE product_name = ''rocketmiles''
        AND srq_date_created BETWEEN ''{START_DATE}'' AND ''{END_DATE}''
"""

s3_prefix = 's3://{}/{}/{}/project_full/'\
    .format(c.S3_BUCKET, c.S3_FOLDER, c.ENV)

local_prefix = 'data/raw/full/'

os.makedirs(local_prefix)
os.makedirs(os.path.join(local_prefix, 'train'))
os.makedirs(os.path.join(local_prefix, 'val'))
os.makedirs(os.path.join(local_prefix, 'test'))

unload_query_train = unload_query.format(
    FEATURES=',\n '.join(a.FEATURES),
    SCHEMA=c.SCHEMA,
    START_DATE=a.train_start_date,
    END_DATE=a.train_end_date)

unload_query_val = unload_query.format(
    FEATURES=',\n '.join(a.FEATURES),
    SCHEMA=c.SCHEMA,
    START_DATE=a.val_start_date,
    END_DATE=a.val_end_date)

unload_query_test = unload_query.format(
    FEATURES=',\n '.join(a.FEATURES),
    SCHEMA=c.SCHEMA,
    START_DATE=a.test_start_date,
    END_DATE=a.test_end_date)

a.unload_and_download(unload_query_train,
    os.path.join(s3_prefix, 'train'),
    os.path.join(local_prefix, 'train'),
    wr_con)

a.unload_and_download(unload_query_val,
    os.path.join(s3_prefix, 'val'),
    os.path.join(local_prefix, 'val'),
    wr_con)

a.unload_and_download(unload_query_test,
    os.path.join(s3_prefix, 'test'),
    os.path.join(local_prefix, 'test'),
    wr_con)
