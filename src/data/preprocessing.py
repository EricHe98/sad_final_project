import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as numpy
import models.MultVAE.multVAE_model 


def df_conversions(df):
    '''
    IMPORTANT!!
    1) Adds a hotel_index column, this is assigns each hotel_id a number from 0 to len(hotel_id)-1. This index
    is used for our interaction vector, which is a len(hotel_id)-length vector with the interaction label
    as the entries.
    
    2) Drops users with no user_id (aka, anonymous/first time users)
    3) subtracts 1e^10 from user_id (ask eric why)
    4) converts date time strings into pandas datetime objects
    '''
    hotel_id_to_hotel_index = dict((hotel_id, i) for (i, hotel_id) in enumerate(df['hotel_id'].unique()))
    df['hotel_index']= df['hotel_id'].map(hotel_id_to_hotel_index)
    df.drop(df.loc[df['user_id'].isna()].index, inplace=True)
    df['user_id'] = df['user_id'] - 1e10
    df['check_in'] = pd.to_datetime(df['check_in'],yearfirst=True)
    df['check_out'] = pd.to_datetime(df['check_out'],yearfirst=True)
    return df

def create_user_id_to_query_struct_dict(df):
    '''
    returns a dictionary of user_id -> "query_struct."
    check create_query_struct_for_user for what a "query_struct" is
    '''
    unique_user_ids = df['user_id'].unique()
    user_id_to_query_struct_dict = {user_id : create_query_struct_for_user(df,user_id)
                                    for user_id in unique_user_ids}
    return user_id_to_query_struct_dict


def create_query_struct_for_user(df,user_id):
    '''
    Returns a "query struct", which is a 2-tuple:
    
    (Assuming a given and fixed user_id):
    
    1st entry: dict of search_request_ids to a interaction_vec for that search request.
        Note: The interaction_vec is a dict of hotel_index -> label for that hotel_index.
              Importantly, this is a sparse vector format.
              Thus, the 1st entry is a dict{search_request_id -> dict{hotel_idx->label}}
              
    2nd return: user_vector containing all of the label/interactions w/user_id = user_id
    
    '''
    # Select only the entries for the user we care about
    df_user_id = df[df['user_id']==user_id]
    # get all of their searches (search_id)
    unique_search_ids_per_user = df_user_id['search_request_id'].unique()
    # Loop over each search, storing the interaction for each search query
    interaction_vecs_per_query = []
    for sr_id in unique_search_ids_per_user:
        # Select only entries for each search request
        df_sr_user_id = df_user_id[df_user_id['search_request_id']==sr_id] 
        # Create a dict of {hotel_index:label}
        interaction_sparse_vec = pd.Series(df_sr_user_id['label'].values,index=df_sr_user_id['hotel_index']).to_dict()
        # Add it to vector
        interaction_vecs_per_query.append(interaction_sparse_vec)
    
    #make a dict of search_ids to interactions_vec
    search_id_to_interaction_vec = dict(zip(unique_search_ids_per_user,interaction_vecs_per_query))
    
    # Merge all the interactions to get the user's entire interaction vec
    user_interaction_vec = merge_dicts_with_max(interaction_vecs_per_query)
    
    return search_id_to_interaction_vec,user_interaction_vec

def get_single_query_interaction_vec(user_id_to_query_struct_dict,user_id,sr_id):
    return user_id_to_query_struct_dict[user_id][0][sr_id]
def get_user_entire_interaction_vec(user_id_to_query_struct_dict,user_id):
    return user_id_to_query_struct_dict[user_id][1]

def merge_dicts_with_max(dict_list):
    ''' 
    merge a list of dictionaries
    if their keys overlap, return the max.
    e.g. {a:1,b:1}
         {b:2,c:2}
         merged into {a:1,b:2,c:2}
    
    We need this in order to merge a single users interaction vectors. Consider if a user
    does two queries, and ends up buying the same hotel twice. 
    '''
    return_dict = {}
    for dict_ in dict_list:
        for key in dict_:
            if key in return_dict:
                return_dict[key] = max(return_dict[key],dict_[key])
            else:
                return_dict[key] = dict_[key]
    return return_dict



def create_context_df_and_cat_encoder(df,cat_vars_to_use):
    '''
    Creates the context_dataframe and cat_encoder
    
    returns a 2-tuple,
    1st: return context_dataframe, which is a dataframe with !search_request_id as the index!
    2nd: returns a sklearn OneHotEncoder, which has been trained on context_df['cat_vars_to_use']
    
    '''
    # Our df contains 100~ results for each search request id. Each of the 100 results have the same query info
    # e.g, they all have the same values for reward_program_has, check_in_weekday,number_of_nights,etc
    # We only need one row out of those 100 to properly get the context. 
    # Here, we grab the first row
    sr_id_to_first_index_df = pd.DataFrame([[key,val.values[0]]
                              for key,val in df.groupby('search_request_id').groups.items()], 
                              columns=['search_request_id','first_index'])
    context_df = df.loc[sr_id_to_first_index_df['first_index'].values]
    context_df.set_index('search_request_id',inplace=True)
    #Encode the categorical variables
    cat_onehot_enc = OneHotEncoder()
    cat_onehot_enc.fit(context_df[cat_vars_to_use])    
    
    
    return context_df,cat_onehot_enc

def create_context_vec(context_df, cat_onehot_enc, cat_vars_to_use, quant_vars_to_use, sr_id):
    '''
    returns a np.vector which contains the context information.
            The categorical features have already been encoded via cat_onehot_enc
    
    '''
    #Get User id for this query
    user_id = context_df.loc[sr_id]['user_id']
    # Get and encode the categorical features for this query
    context_cat_pre_enc = context_df.loc[sr_id][cat_vars_to_use]
    #Reshape if we're only dealing with one row
    if len(context_cat_pre_enc.shape) == 1:
        context_cat_pre_enc = context_cat_pre_enc.values.reshape(1,-1)
    context_cat_enc = cat_onehot_enc.transform(context_cat_pre_enc).todense()
    
    # Get the quantitative features for this query
    context_quant = context_df.loc[sr_id][quant_vars_to_use].to_numpy()
    if len(context_quant.shape) == 1:
        context_quant = context_cat_pre_enc.reshape(1,-1)
    #stack the encoded categorical features and quantitative features
    context_vec = np.hstack((context_cat_enc,context_quant))
    # return the context
    return context_vec

def get_user_id_from_sr_id(context_df,sr_id):
    '''
    Make sure you use context df, as that is indexed by search_id (this will speed things up)
    '''
    return context_df.loc[sr_id]['user_id']

