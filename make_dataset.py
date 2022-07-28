import pandas as pd
import random
import math


def subtype_selection(subtype):
    global subtype_flag, data_path
    if subtype == 'H1N1':
        subtype_flag = 0
        # data_path = '/home/zh/codes/rnn_virus_source_code/data/raw/H1N1_cluster/'
    elif subtype == 'H3N2':
        subtype_flag = 1
        # data_path = '/home/zh/codes/rnn_virus_source_code/data/raw/H3N2_cluster/'
    elif subtype == 'H5N1':
        subtype_flag = 2
        # data_path = '/home/zh/codes/rnn_virus_source_code/data/raw/H5N1_cluster/'
    elif subtype == 'COV19':
        subtype_flag = 3

    return subtype_flag


def read_trigram_vecs(subtype):
    """
    Reads the csv file containing 100 dimensional prot vecs, the
    data_path argument indicating where it is located.
    Returns a dictionary that maps a 3gram of amino acids to its
    index and a numpy array containing the trigram vecs.
    """
    data_path = '/home/zh/codes/rnn_virus_source_code/data/raw/H1N1_cluster/'
    prot_vec_file = 'protVec_100d_3grams.csv'

    df = pd.read_csv(data_path + prot_vec_file, delimiter='\t')
    trigrams = list(df['words'])
    trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
    trigram_vecs = df.loc[:, df.columns != 'words'].values

    return trigram_to_idx, trigram_vecs


def read_strains_from(data_files, data_path):
    """
    Reads the raw strains from the data_files located by the data_path.
    Returns a pandas series for each data file, contained in a ordered list.
    """
    # _, data_path = subtype_selection(subtype)
    raw_strains = []
    for file_name in data_files:
        df = pd.read_csv(data_path + file_name)
        strains = df['seq']
        raw_strains.append(strains)

    return raw_strains


def train_test_split_strains(strains_by_year, test_split, cluster):
    """
    Shuffles the strains in each year and splits them into two disjoint sets,
    of size indicated by the test_split.
    Expects and returns pandas dataframe or series.
    """
    train_strains, test_strains = [], []
    if cluster == 'random':
        for strains in strains_by_year:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            shuffled_strains = strains.sample(frac=1).reset_index(drop=True)
            train = shuffled_strains.iloc[:num_of_training_examples].reset_index(drop=True)
            test = shuffled_strains.iloc[num_of_training_examples:].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    else:
        # change the starting index for the time-series training samples for multiple experiments
        for strains in strains_by_year:
            num_of_training_examples = int(math.floor(strains.count() * (1 - test_split)))
            train = strains.iloc[:800].reset_index(drop=True)
            test = strains.iloc[800:1000].reset_index(drop=True)
            train_strains.append(train)
            test_strains.append(test)
    return train_strains, test_strains



