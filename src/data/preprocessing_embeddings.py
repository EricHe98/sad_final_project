import argparse
import os
import random

from tqdm import tqdm

from pyspark import SQLContext
from pyspark import SparkContext
from pyspark.sql import functions as funct
from pyspark.sql.window import Window

import click
import datetime as dt
import json
import pickle
import sys
import traceback

from src.data.preprocessing import read_parquet

POS_PAIRS_FILENAME = "pos_pairs.csv"
NEG_PAIRS_FILENAME = "neg_pairs.csv"
HOTELS_FILENAME = "hotels.csv"

COLUMNS_TO_READ = ['hotel_id', 'srq_date_created', 'session_id', 'label', 'display_rank']

def write_set_to_file(filename, a_set):
    with open(filename, 'w') as f:
        for item in a_set:
            f.write(f"{item}\n")

def gen_neg_pairs_out_session(neg_list, pos_list, hotel_ids, num_neg_samples=5):
    neg_pairs = []
    neg_hotels = hotel_ids - set(pos_list)
    for target_i, target_word in enumerate(pos_list):
        neg_samples = random.sample(neg_hotels, num_neg_samples)
        for neg_sample in neg_samples:
            neg_pairs.append([target_word, neg_sample])
    return neg_pairs

def gen_pos_pairs(pos_list, window_size=5):
    pos_pairs = []
    pos_list_len = len(pos_list)
    for target_i, target_word in enumerate(pos_list):
        for w in range(window_size):
            # context ahead of target word by window_size words
            if target_i + 1 + w < pos_list_len:
                pos_pairs.append([target_word, pos_list[target_i + 1 + w]])
            # context before target word by window_size words
            if target_i - 1 - w >= 0:
                pos_pairs.append([target_word, pos_list[target_i - 1 - w]])
    return pos_pairs

# TODO: room for improvement in changing distribution of negative samples (recip, piece-wise)
def gen_neg_pairs_in_session(neg_list, pos_list, num_neg_samples=5):
    neg_pairs = []
    for target_i, target_word in enumerate(pos_list):
        num_samples = num_neg_samples
        if num_samples > len(neg_list):
            num_samples = len(neg_list)

        neg_samples = random.sample(neg_list, num_samples)
        for neg_sample in neg_samples:
            neg_pairs.append([target_word, neg_sample])
    return neg_pairs

def remove_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def append_pairs_to_file(my_list, filename):
    with open(filename, 'a') as f:
        for item in my_list:
            f.write(f"{item[0]}, {item[1]}\n")

def lists_to_list(nested_lists):
    flat_list = [item for sublist in nested_lists for item in sublist]
    return flat_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate hotel pos/neg pairs.')
    parser.add_argument('dataset', choices=['small_100', 'small_all', 'full'],
    help='which dataset to train the model (small_100, small_all, all) to be included in MLFlow name')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
        help='which split of the dataset to train on (train, val, test)')
    parser.add_argument('--data', type=str, default='data/raw', required=True,
        help='path to data dir')
    parset.add_argument('--out_dir', type=str, required=True, 'output directory')
    parser.add_argument('--in_samples', type=int, default=5,
        help='number of samples in context')
    parser.add_argument('--out_samples', type=int, default=5,
        help='number of samples out of context')


    args = parser.parse_args()

    sc = SparkContext.getOrCreate()
    sqlContext = SQLContext(sc)
    data = sqlContext.read.parquet(os.path.join(args.data, args.dataset, args.split)).select(COLUMNS_TO_READ)

    hotel_ids = set(data.select('hotel_id').distinct().rdd.flatMap(lambda x: x).collect())
    session_ids = data.select('session_id').distinct()

    remove_file(POS_PAIRS_FILENAME)
    remove_file(NEG_PAIRS_FILENAME)
    remove_file(HOTELS_FILENAME)

    list_pos_pairs = []
    list_neg_in_pairs = []
    list_neg_out_pairs = []

    print("Generating pos/neg pairs")
    for session_id in session_ids:
        context_list = data.filter(data.session_id == session_id).orderBy(['srq_date_created', 'display_rank'])
        pos_list = context_list.filter(context_list.label > 0).select('hotel_id').rdd.flatMap(lambda x: x).collect()
        neg_list = context_list.filter(context_list.label == 0).select('hotel_id').rdd.flatMap(lambda x: x).collect()

        list_pos_pairs.extend(gen_pos_pairs(pos_list))
        list_neg_in_pairs.extend(gen_neg_pairs_in_session(neg_list, pos_list, num_neg_samples=args.in_samples))
        list_neg_out_pairs.extend(gen_neg_pairs_out_session(neg_list, pos_list, hotel_ids, num_neg_samples=args.out_samples))

    print("Writing results to file")
    append_pairs_to_file(list_pos_pairs, os.path.join(args.out_dir, POS_PAIRS_FILENAME))
    append_pairs_to_file(list_neg_in_pairs, os.path.join(args.out_dir, NEG_PAIRS_FILENAME))
    append_pairs_to_file(list_neg_out_pairs, os.path.join(args.out_dir, NEG_PAIRS_FILENAME))

    set_hotels_in_train = set(lists_to_list(list_pos_pairs) + lists_to_list(list_neg_in_pairs) + lists_to_list(list_neg_out_pairs))
    write_set_to_file(os.path.join(args.out_dir, HOTELS_FILENAME), set_hotels_in_train)