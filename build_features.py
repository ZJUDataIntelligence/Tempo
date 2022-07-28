import sys, os
import pandas as pd
import random
import numpy as np
# sys.path.append(os.path.abspath("C:/Users/yinr0002/Google Drive/Tier_2_MOE2014/2_Conference/CIKM2019/code/influenza-master"))

from trigram import Trigram
import validation


def sample_strains(strains_by_year, num_of_samples):
    """
    Randomly picks num_of_samples strains from each year,
    sampling is done with replacement.
    Returns a 2d list of strings.
    """
    sampled_strains_by_year = []

    for year_strains in strains_by_year:
        sampled_strains_by_year.append(random.choices(year_strains, k=num_of_samples))

    return sampled_strains_by_year


def sample_strains_cluster(strains_by_year, num_of_samples):
    """
    Picks num_of_samples strains from each year after clustering
    sampling is done with replacement.
    Returns a 2d list of strings.
    """
    sampled_strains_by_year = []
    # i = random.randint(0, 100)
    for year_strains in strains_by_year:
        sampled_strains_by_year.append(year_strains[:num_of_samples])

    return sampled_strains_by_year


def split_to_trigrams(strains_by_year, overlapping=True):
    """
    Splits the strains into trigrams, by default overlapping.
    If non-overlapping approach is used, the last amino acids are padded to make
    an extra trigram if the strain length is not evenly divisible by three.
    Expects a 2d [year, strain] list of strings,
    returns a 3d [year, strain, trigram] list of Trigram objects.
    """
    if overlapping:
        step_size = 1
        num_of_trigrams = len(strains_by_year[0][0]) - 2
    else:
        step_size = 3
        num_of_trigrams = len(strains_by_year[0][0]) // step_size

    trigrams_by_year = []
    for year_strains in strains_by_year:
        year_trigrams = []

        for strain in year_strains:
            strain_trigrams = []

            for i in range(num_of_trigrams):
                pos = i * step_size
                trigram = Trigram(strain[pos:pos + 3], pos)
                strain_trigrams.append(trigram)

            remainder = len(strain) % step_size
            if remainder > 0:
                padding = '-' * (3 - remainder)
                amino_acids = strain[-remainder:] + padding
                trigram = Trigram(amino_acids, len(strain) - remainder)
                strain_trigrams.append(trigram)

            year_trigrams.append(strain_trigrams)

        trigrams_by_year.append(year_trigrams)

    return trigrams_by_year


def make_triplet_strains(strains_by_year, positions):
    """
    Splits each strain into substrings of 'triplets' refering to 3 overlapping
    trigrams (5 amino acids), centered at the given positions.
    Expects and returns a 2d [year, strain] list of strings.
    """
    triplet_strains_by_year = []
    triplet_strain_margin = 2

    for strains_in_year in strains_by_year:
        triplet_strains_in_year = []
        for strain in strains_in_year:
            for p in positions:
                if p < triplet_strain_margin:
                    padding_size = triplet_strain_margin - p
                    triplet_strain = '-' * padding_size + strain[:p + triplet_strain_margin + 1]
                elif p > len(strain) - 1 - triplet_strain_margin:
                    padding_size = p - (len(strain) - 1 - triplet_strain_margin)
                    triplet_strain = strain[p - triplet_strain_margin:] + '-' * padding_size
                else:
                    triplet_strain = strain[p - triplet_strain_margin:p + triplet_strain_margin + 1]
                triplet_strains_in_year.append(triplet_strain)
        triplet_strains_by_year.append(triplet_strains_in_year)

    return triplet_strains_by_year


def make_triplet_labels(triplet_strains_by_year):
    """
    Creates labels indicating whether the center amino acid in each triplet
    mutates in the last year (1 for yes, 0 for no).
    Expects a 2d [year, triplet] list of strings and returns a list of ints.
    """
    num_of_triplets = len(triplet_strains_by_year[0])
    epitope_position = 2

    labels = []
    for i in range(num_of_triplets):
        if triplet_strains_by_year[-1][i][epitope_position] == triplet_strains_by_year[-2][i][epitope_position]:
            labels.append(0)
        else:
            labels.append(1)

    return labels


def get_majority_baselines(triplet_strains_by_year, labels):
    """
    Returns accuracy, precision, recall, f1-score and mcc for the baseline
    approach of simply predicting mutation epitope in the last year differs
    from the majority one.
    """
    epitope_position = 2

    predictions = []
    for i in range(len(labels)):
        epitopes = []
        for year in range(len(triplet_strains_by_year) - 1):
            epitopes.append(triplet_strains_by_year[year][i][epitope_position])
        majority_epitope = max(set(epitopes), key=epitopes.count)

        if triplet_strains_by_year[-2][i][epitope_position] == majority_epitope:
            predictions.append(0)
        else:
            predictions.append(1)

    conf_matrix = validation.get_confusion_matrix(np.array(labels), np.array(predictions))
    acc = validation.get_accuracy(conf_matrix)
    precision = validation.get_precision(conf_matrix)
    recall = validation.get_recall(conf_matrix)
    f1score = validation.get_f1score(conf_matrix)
    mcc = validation.get_mcc(conf_matrix)

    return acc, precision, recall, f1score, mcc


def extract_positions_by_year(positions, trigrams_by_year):
    """
    Extracts trigrams that contain an amino acid from one of the given positions.
    Expects and returns a 3d [year, strain, trigram] list of Trigram objects.
    """
    strain = trigrams_by_year[0][0]
    strain_idxs_to_extract = []
    idx = 0

    for pos in positions:
        pos_found = False
        while not pos_found:
            trigram = strain[idx]
            if trigram.contains_position(pos):
                pos_found = True
            else:
                idx += 1

        pos_extracted = False
        while not pos_extracted:
            trigram = strain[idx]
            if trigram.contains_position(pos):
                strain_idxs_to_extract.append(idx)
                idx += 1
            else:
                pos_extracted = True

    def extract_idxs(strain_trigrams):
        return [strain_trigrams[i] for i in strain_idxs_to_extract]

    extracted_by_year = []
    for year_trigrams in trigrams_by_year:
        extracted_by_year.append(list(map(extract_idxs, year_trigrams)))

    return extracted_by_year


def squeeze_trigrams(trigrams_by_year):
    """
    Takes a 3d [year, strain, trigram] list and squeezes the 2nd dimension
    to return a 2d list [year, trigram].
    """
    squeezed_trigrams_by_year = []

    for year_trigrams in trigrams_by_year:
        squeezed_trigrams = []

        for trigrams in year_trigrams:
            squeezed_trigrams += trigrams

        squeezed_trigrams_by_year.append(squeezed_trigrams)

    return squeezed_trigrams_by_year


def replace_uncertain_amino_acids(amino_acids):
    """
    Randomly selects replacements for all uncertain amino acids.
    Expects and returns a string.
    """
    replacements = {'B': 'DN',
                    'J': 'IL',
                    'Z': 'EQ',
                    'X': 'ACDEFGHIKLMNPQRSTVWY'}

    for uncertain in replacements.keys():
        amino_acids = amino_acids.replace(uncertain, random.choice(replacements[uncertain]))

    return amino_acids


def map_trigrams_to_idxs(nested_trigram_list, trigram_to_idx):
    """
    Takes a nested list containing Trigram objects and maps them to their index.
    """
    dummy_idx = len(trigram_to_idx)

    def mapping(trigram):
        if isinstance(trigram, Trigram):
            trigram.amino_acids = replace_uncertain_amino_acids(trigram.amino_acids)

            if '-' not in trigram.amino_acids:
                return trigram_to_idx[trigram.amino_acids]
            else:
                return dummy_idx

        elif isinstance(trigram, list):
            return list(map(mapping, trigram))

        else:
            raise TypeError('Expected nested list of Trigrams, but encountered {} in recursion.'.format(type(trigram)))

    return list(map(mapping, nested_trigram_list))


def map_idxs_to_vecs(nested_idx_list, idx_to_vec):
    """
    Takes a nested list of indexes and maps them to their trigram vec (np array).
    """
    # represent the 3-grams containing '-' by zero vector in ProVect
    # dummy_vec = np.array([0] * idx_to_vec.shape[1])

    # represent the 3-grams containing '-' by 'unknown' vector in ProVect
    dummy_vec = idx_to_vec[idx_to_vec.shape[0] - 1]

    def mapping(idx):
        if isinstance(idx, int):
            if idx < idx_to_vec.shape[0]:
                return idx_to_vec[idx]
            else:
                return dummy_vec

        elif isinstance(idx, list):
            return list(map(mapping, idx))

        else:
            raise TypeError('Expected nested list of ints, but encountered {} in recursion.'.format(type(idx)))

    return list(map(mapping, nested_idx_list))


def get_diff_vecs(trigram_vecs_by_year):
    """
    Calculates the elementwise difference between each consecutive trigram vec.
    Expects numpy array.
    """
    diff_vecs_by_year = np.zeros(
        (trigram_vecs_by_year.shape[0] - 1, trigram_vecs_by_year.shape[1], trigram_vecs_by_year.shape[2]))
    for i in range(diff_vecs_by_year.shape[0]):
        diff_vecs_by_year[i] = trigram_vecs_by_year[i + 1] - trigram_vecs_by_year[i]

    return diff_vecs_by_year


def indexes_to_mutations(trigram_indexes_x, trigram_indexes_y):
    """
    Creates an numpy array containing 1's in positions where trigram_indexes_x and
    trigram_indexes_y differ, corresponding to mutated sites and zeros elsewhere.
    """
    assert (len(trigram_indexes_x) == len(trigram_indexes_y))

    mutations = np.zeros(len(trigram_indexes_x))
    for i in range(len(trigram_indexes_x)):
        if trigram_indexes_x[i] != trigram_indexes_y[i]:
            mutations[i] = 1

    return mutations


def reshape_to_linear(vecs_by_year, window_size=3):
    reshaped = [[]] * len(vecs_by_year[0])

    for year_vecs in vecs_by_year[-window_size:]:
        for i, vec in enumerate(year_vecs):
            reshaped[i] = reshaped[i] + vec.tolist()

    return reshaped
