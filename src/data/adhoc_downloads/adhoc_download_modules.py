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

train_start_date = '2019-01-01'
train_end_date = '2019-10-31'
val_start_date = '2019-11-01'
val_end_date = '2019-11-30'
test_start_date = '2019-12-01'
test_end_date = '2019-12-31'

FEATURES = ['search_result_id',
        'search_request_id',
        'hotel_id',
        'user_id',
        'srq_date_created',
        'label',
        'display_rank',
        'price_rank',
        'rewards_rank',
        'est_spread_rank',
        'check_in',
        'check_out',
        'reward_program_hash',
        'site_hash',
        'region_id',
        'est_ttm',
        'average_published_tax_and_fees',
        'average_published_price',
        'hotel_cumulative_share',
        'total_rewards',
        'advance_purchase_days',
        'number_of_nights',
        'number_of_rooms',
        'number_of_adults',
        'normalized_rewards',
        'previous_user_hotel_interaction',
        'session_id',
        'promotion_id',
        'anonymous_user',
        'srq_latitude',
        'srq_longitude',
        'check_in_weekday',
        'check_out_weekday',
        'srq_weekhour',
        'travel_intent',
        'weekday_travel',
        'weekend_travel',
        'hotel_latitude',
        'hotel_longitude',
        'rating',
        'stars',
        'number_of_reviews',
        'srq_hotel_distance',
        'srq_price_zscore',
        'srq_rewards_zscore',
        'srq_distance_zscore',
        'srq_rating_zscore',
        'srq_stars_zscore',
        'user_preceding_clicks',
        'raw_user_preferred_price',
        'user_preferred_price',
        'user_preferred_rewards',
        'user_preferred_stars',
        'user_preferred_rating',
        'user_preferred_distance',
        'region_preceding_bookings',
        'region_meanprice_diff',
        'region_rating_diff',
        'region_stars_diff',
        'region_centroid_distance',
        'region_rewards_diff',
        'region_price_sd',
        'region_rating_sd',
        'region_stars_sd',
        'region_rewards_sd',
        'hotel_cumulative_bookings',
        'popawi']

def unload_and_download(query, s3_path, local_path, wr_con):
    print('Unloading to {} using the following query: \n{}'.format(s3_path, query))
    paths = wr.db.unload_redshift_to_files(
        sql=query,
        path=s3_path,
        con=wr_con,
        iam_role='arn:aws:iam::301933004157:role/RedshiftCopyUnload'
    )
    print('{} files unloaded to s3'.format(len(paths)))

    s3_client = boto3.client('s3')

    for path in paths:
        bucket, key = wr._utils.parse_path(path)
        # bucket, key = path.replace("s3://", "").split("/", 1) # same outcome
        base = os.path.basename(key)
        with open(os.path.join(local_path, base), 'wb') as f:
            print('Downloading {} to {}'.format(path, f))
            s3_client.download_fileobj(bucket, key, f)

