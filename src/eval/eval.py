"""
Evaluate predictions given by a model.
Takes in paths to validation data and predictions data
	(assumes the data there is inner-joinable)
Predictions data must contain `search_request_id`, `hotel_id`, `score`, `rank` columns
Must supply `run_id` to join metrics back to MLFlow
"""
import os
import json
import sys
import datetime as dt

import numpy as np 
import pandas as pd

import mlflow

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import argparse
parser = argparse.ArgumentParser(description='Evaluate predictions on validation set')
parser.add_argument('run', type=str, 
    help='Run ID for saving models')

args = parser.parse_args()
run_id = args.run


