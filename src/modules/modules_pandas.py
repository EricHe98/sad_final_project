import os
import pandas as pd 

def read_parquet(data_path, num_partitions=None, random=False, verbose=True, columns=None):
    files = os.listdir(data_path)
    if random:
        import random
        random.shuffle(files)
    if num_partitions is None:
        num_partitions = len(files)
        
    data = []
    num_reads = 0
    for file_path in files:
        if num_reads >= num_partitions:
            break
        root, ext = os.path.splitext(file_path)
        # exclude non-parquet files (e.g. gitkeep, other folders)
        if ext == '.parquet':
            fp = os.path.join(data_path, file_path)
            if verbose:
                print('Reading in data from {}'.format(fp))
            data.append(pd.read_parquet(os.path.join(data_path, file_path), columns=columns))
            if verbose:
                print('Data of shape {}'.format(data[-1].shape))
            num_reads += 1
        else: 
            continue
    data = pd.concat(data, axis=0)
    if verbose:
        print('Total dataframe of shape {}'.format(data.shape))
    return data

def feature_label_split(data, model_features, label='label', qid='qid'):
    # assumes data of same QIDs are grouped together
    X = data[model_features]
    y = data[label]
    qid = data[qid].value_counts(sort=False).sort_index()
    return X, y, qid