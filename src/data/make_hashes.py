import os
import sys
import pandas as pd
import numpy as np
import json
import preprocessing


if __name__ == "__main__":
    data_path, output_path = sys.argv[1], sys.argv[2]
    data = preprocessing.read_parquet(data_path, 
                                      num_partitions = None,
                                      randomize = False,
                                      verbose = True,
                                      columns = ['hotel_id','user_id','label'])
    print('finish read')
    data.dropna(subset=['user_id'],inplace=True)
    print('finish drop')

    data['user_id'] = data['user_id'] - 1e10
    unique_users = data.user_id.unique()
    unique_hotels = data.hotel_id.unique() 

    user_id_indexed = dict((user_id, i) for i, user_id in enumerate(unique_users))
    hotel_id_indexed = dict((hotel_id, i) for i,hotel_id in enumerate(unique_hotels))
    print('finish making hash')
    
    with open(os.path.join(output_path, 'user_hash.json'), 'w') as fp:
        json.dump({str(key): value for key, value in user_id_indexed.items()},
            fp, 
            sort_keys=True)

    with open(os.path.join(output_path, 'hotel_hash.json'), 'w') as fp:
        json.dump({str(key): value for key, value in hotel_id_indexed.items()},
            fp,
            sort_keys=True)
    print('Finished hashing indicies')