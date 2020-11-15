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

rs_de = get_cred('airflow-rs-dataengineer')
wr_con = wr.db.get_engine(db_type='redshift',
    host=rs_de['host'],
    port=rs_de['port'],
    database=rs_de['database'],
    user=rs_de['user'],
    password=rs_de['password'],
    connect_args={'sslmode':'verify-ca'})
con = RedshiftDB(rs_de_conn)

random_sample = """
    DROP TABLE IF EXISTS {schema}.srr_train_sample;

    CREATE TABLE {schema}.srr_train_sample AS (
    SELECT *
    FROM
    (
        SELECT DISTINCT search_request_id,
            DATE_TRUNC('month', srq_date_created::DATETIME) AS dt,
            ROW_NUMBER() OVER (PARTITION BY dt ORDER BY RANDOM()) AS rn
        FROM {schema}.srr_train
        WHERE product_name = 'rocketmiles'
            AND srq_date_created BETWEEN '2019-01-01' AND '2020-01-01'
        )
    WHERE rn <= 1500
    )
""".format(schema=c.SCHEMA)
con.execute(random_sample)

unload_query = """
    SELECT search_result_id,
        search_request_id,
        hotel_id,
        user_id,
        srq_date_created,
        label,
        display_rank,
        price_rank,
        rewards_rank,
        est_spread_rank,
        check_in,
        check_out,
        reward_program_hash,
        site_hash,
        region_id,
        est_ttm,
        average_published_tax_and_fees,
        average_published_price,
        hotel_cumulative_share,
        total_rewards,
        advance_purchase_days,
        number_of_nights,
        number_of_rooms,
        number_of_adults,
        number_of_children,
        normalized_rewards,
        previous_user_hotel_interaction,
        session_id,
        promotion_id,
        anonymous_user,
        srq_latitude,
        srq_longitude,
        check_in_weekday,
        check_out_weekday,
        srq_weekhour,
        travel_intent,
        weekday_travel,
        weekend_travel,
        hotel_latitude,
        hotel_longitude,
        rating,
        stars,
        number_of_reviews,
        srq_hotel_distance,
        srq_price_zscore,
        srq_rewards_zscore,
        srq_distance_zscore,
        srq_rating_zscore,
        srq_stars_zscore,
        user_preceding_clicks,
        raw_user_preferred_price,
        user_preferred_price,
        user_preferred_rewards,
        user_preferred_stars,
        user_preferred_rating,
        user_preferred_distance,
        region_preceding_bookings,
        region_meanprice_diff,
        region_rating_diff,
        region_stars_diff,
        region_centroid_distance,
        region_rewards_diff,
        region_price_sd,
        region_rating_sd,
        region_stars_sd,
        region_rewards_sd,
        hotel_cumulative_bookings,
        popawi
    FROM {}.srr_train
    WHERE product_name = ''rocketmiles''
        AND srq_date_created BETWEEN ''2019-01-01'' AND ''2020-01-01''
        AND search_request_id IN (SELECT DISTINCT search_request_id FROM public_qa.srr_train_sample)
""".format(c.SCHEMA)

unload_path = 's3://{}/{}/{}/small_all_results/'\
    .format(c.S3_BUCKET, c.S3_FOLDER, c.ENV)

print('Unloading to {} using the following query: \n{}'.format(unload_path, unload_query))
paths = wr.db.unload_redshift_to_files(
    sql=unload_query,
    path=unload_path,
    con=wr_con,
    iam_role='arn:aws:iam::301933004157:role/RedshiftCopyUnload'
)

local_prefix = 'data'
s3_client = boto3.client('s3')

for path in paths:
    bucket, key = wr._utils.parse_path(path)
    # bucket, key = path.replace("s3://", "").split("/", 1) # same outcome
    base = os.path.basename(key)
    with open(os.path.join(local_prefix, base), 'wb') as f:
        print('Downloading {} to {}'.format(path, f))
        s3_client.download_fileobj(bucket, key, f)
