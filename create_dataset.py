from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
import math
import random
import numpy as np
from math import floor
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
import time
from sklearn.neighbors import NearestNeighbors
import random

def label_encode(strains):
    amino_acids = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C', 'L', '-', 'B', 'J', 'Z', 'X']
    le = preprocessing.LabelEncoder()
    le.fit(amino_acids)

    encoded_strains = []
    for strain in strains:
        chars = list(strain)
        encoded_strains.append(le.transform(chars))

    return encoded_strains


def label_decode(encoded_strains):
    amino_acids = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C', 'L', '-', 'B', 'J', 'Z', 'X']
    le = preprocessing.LabelEncoder()
    le.fit(amino_acids)

    strains = []
    for one_year_encoded_strains in encoded_strains:
        one_year_strains = []
        for encoded_strain in one_year_encoded_strains:
            temp = le.inverse_transform(encoded_strain)
            one_year_strains.append(''.join(temp))
        strains.append(one_year_strains)

    return strains


#strains : df['Sequence']
def strain_cluster(strains,num_clusters=2):
    encoded_strains = label_encode(strains)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(encoded_strains)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    result = {'data':encoded_strains, 'labels':labels, 'centroids':centroids}
    return result


def show_cluster(cluster,save_fig_path='none'):
    encoded_strains = cluster['data']
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(encoded_strains)
    fig = plt.figure()
    colors = 10 * ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.']
    labels = cluster['labels']
    for i in range(len(reduced_data)):
        plt.plot(reduced_data[i][0], reduced_data[i][1], colors[labels[i]], markersize=10)
    plt.show()
    if save_fig_path!='none':
        plt.savefig(save_fig_path)


def link_clusters(clusters):
    no_years = len(clusters)
    neigh = NearestNeighbors(n_neighbors=2)
    for year_idx in range(no_years):
        if (year_idx == no_years - 1):  # last year doesn't link
            clusters[year_idx]['links'] = []
            break
        links = []
        current_centroids = clusters[year_idx]['centroids']
        next_year_centroids = clusters[year_idx + 1]['centroids']
        neigh.fit(next_year_centroids)
        idxs_by_centroid = neigh.kneighbors(current_centroids, return_distance=False)

        for label in clusters[year_idx]['labels']:
            links.append(idxs_by_centroid[label])  # centroid idx corresponds to label

        clusters[year_idx]['links'] = links


def sample_from_clusters(clusters_by_years, sample_size):
    sampled_strains = []
    for i in range(sample_size):
        one_sample = []
        start_idx = random.randint(0, len(clusters_by_years[0]['data'])-1)
        start_strain = clusters_by_years[0]['data'][start_idx]
        one_sample.append(start_strain)

        num_years = len(clusters_by_years)
        idx = start_idx
        for i in range(num_years - 1):
            next_nearest_label = clusters_by_years[i]['links'][idx][0]
            candidate_idx = np.where(clusters_by_years[i + 1]['labels'] == next_nearest_label)[0]
            idx = random.choice(candidate_idx)
            one_sample.append(clusters_by_years[i + 1]['data'][idx])

        sampled_strains.append(one_sample)

    return sampled_strains


# input: list of list, n_sample*n_year, each item is a strain(str)
# output: return/write a df/csv file, format
# like 'data/processed/H1N1/triplet_cluster_train.csv'
def create_dataset(strains, position, window_size=10, output_path='None'):
    # create label
    label = []
    for sample in strains:
        if sample[-1][position] == sample[-2][position]:
            label.append(0)
        else:
            label.append(1)
    # read prot embedding
    df = pd.read_csv('/home/zh/codes/rnn_virus_source_code/data/raw/H1N1/protVec_100d_3grams.csv', sep='\t')
    triple2idx = {}
    for index, row in df.iterrows():
        triple2idx[row['words']] = index
    # create data
    start_year_idx = len(strains[0]) - window_size - 1
    data = []
    for i in range(len(strains)):
        one_sample_data = []
        for year in range(start_year_idx, len(strains[0]) - 1):
            tritri = []
            if (position == 0):
                tritri.append(9047)
                tritri.append(9047)
                if strains[i][year][position:position + 3] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position:position + 3]])
                else:
                    tritri.append(9047)
            elif (position == 1):
                tritri.append(9047)
                if strains[i][year][position - 1:position + 2] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 1:position + 2]])
                else:
                    tritri.append(9047)
                if strains[i][year][position:position + 3] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position:position + 3]])
                else:
                    tritri.append(9047)
            elif (position == 1271):
                if strains[i][year][position - 2:position + 1] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 2:position + 1]])
                else:
                    tritri.append(9047)
                if strains[i][year][position - 1:position + 2] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 1:position + 2]])
                else:
                    tritri.append(9047)
                tritri.append(9047)
            elif (position == 1272):
                if strains[i][year][position - 2:position + 1] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 2:position + 1]])
                else:
                    tritri.append(9047)
                tritri.append(9047)
                tritri.append(9047)
            else:
                if strains[i][year][position - 2:position + 1] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 2:position + 1]])
                else:
                    tritri.append(9047)
                if strains[i][year][position - 1:position + 2] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position - 1:position + 2]])
                else:
                    tritri.append(9047)
                if strains[i][year][position:position + 3] in triple2idx:
                    tritri.append(triple2idx[strains[i][year][position:position + 3]])
                else:
                    tritri.append(9047)

            one_sample_data.append(str(tritri))

        data.append(one_sample_data)

    dataset = pd.DataFrame(data)
    print(dataset.shape)
    print(len(label))
    dataset.insert(0, 'y', label)
    return dataset

def main():
    path = '/data/zh/sprot_years/'
    files = os.listdir(path)
    files.sort()
    cluster_years = []
    start_time = time.time()
    for file in files:
        df = pd.read_csv(path + file)
        strains = df['Sequence'].sample(1000, replace=True)
        cluster = strain_cluster(strains, num_clusters=4)
        cluster_years.append(cluster)
        print("{:.1f}: {} processed.".format(time.time() - start_time, file))

    link_clusters(cluster_years)

    ss = sample_from_clusters(cluster_years, 10)

    dss = label_decode(ss)

    datasetdf = create_dataset(dss, 12)

