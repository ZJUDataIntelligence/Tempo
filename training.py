import os, sys

# sys.path.append(os.path.abspath("/home/zh/codes/rnn_virus_source_code"))
import models
import train_model
import make_dataset
import build_features
import utils
import torch
import numpy as np


def main():
    if subtype_flag == 0:
        # data_path = '/home/zh/codes/rnn_virus_source_code/data/raw/H1N1_cluster/'
        # data_set = '/home/zh/codes/rnn_virus_source_code/data/processed/H1N1/triplet_cluster'
        data_path = '/home/zh/codes/transformer_virus/data/raw/H1N1_cluster/'
        data_set = '/home/zh/codes/transformer_virus/data/processed/H1N1_drop_duplicates/triplet_cluster'
    elif subtype_flag == 1:
        data_path = '/home/zh/codes/transformer_virus/data/raw/H3N2_cluster/'
        data_set = '/home/zh/codes/transformer_virus/data/processed/H3N2/triplet_cluster'

    elif subtype_flag == 2:
        data_path = '/home/zh/codes/transformer_virus/data/raw/H5N1_cluster/'
        data_set = '/home/zh/codes/transformer_virus/data/processed/H5N1/triplet_cluster'

    elif subtype_flag == 3:
        data_path = '/home/zh/codes/transformer_virus/data/processed/COV19/'
        data_set = '/home/zh/codes/transformer_virus/data/processed/COV19/triplet_cluster'
        # data_path = '/home/zh/codes/rnn_virus_source_code/data/processed/COV19/'
        # data_set = '/home/zh/codes/rnn_virus_source_code/data/processed/COV19/triplet_cluster'

    parameters = {

        # Exlude _train/_test and file ending
        'data_set': data_set,

        # raw data path
        'data_path': data_path,

        # 'svm', lstm', 'gru', 'attention' (only temporal) or 'da-rnn' (input and temporal attention)
        'model': model,

        # Number of hidden units in the encoder
        'hidden_size': 512,

        # Droprate (applied at input)
        'dropout_p': 0.0001,

        # Note, no learning rate decay implemented
        'learning_rate': 0.001,

        # Size of mini batch
        'batch_size': 256,

        # Number of training iterations
        'num_of_epochs': 100
    }

    torch.manual_seed(1)
    np.random.seed(1)

    train_trigram_vecs, train_labels = utils.read_dataset(parameters['data_set'] + '_train.csv',
                                                          parameters['data_path'], concat=False)
    test_trigram_vecs, test_labels = utils.read_dataset(parameters['data_set'] + '_test.csv',
                                                        parameters['data_path'], concat=False)

    X_train = torch.tensor(train_trigram_vecs, dtype=torch.float32)
    Y_train = torch.tensor(train_labels, dtype=torch.int64)
    X_test = torch.tensor(test_trigram_vecs, dtype=torch.float32)
    Y_test = torch.tensor(test_labels, dtype=torch.int64)

    # give weights for imbalanced dataset
    _, counts = np.unique(Y_train, return_counts=True)
    train_counts = max(counts)
    train_imbalance = max(counts) / Y_train.shape[0]
    _, counts = np.unique(Y_test, return_counts=True)
    test_counts = max(counts)
    test_imbalance = max(counts) / Y_test.shape[0]

    print('Class imbalances:')
    print(' Training %.3f' % train_imbalance)
    print(' Testing  %.3f' % test_imbalance)

    if parameters['model'] == 'svm':
        window_size = 1
        train_model.svm_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels,
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)
    elif parameters['model'] == 'random forest':
        window_size = 1
        train_model.random_forest_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels,
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)
    elif parameters['model'] == 'logistic regression':
        window_size = 1
        train_model.logistic_regression_baseline(
            build_features.reshape_to_linear(train_trigram_vecs, window_size=window_size), train_labels,
            build_features.reshape_to_linear(test_trigram_vecs, window_size=window_size), test_labels)
    else:
        input_dim = X_train.shape[2]
        seq_length = X_train.shape[0]
        output_dim = 2

        if parameters['model'] == 'lstm':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'],
                                  cell_type='LSTM')
        elif parameters['model'] == 'gru':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'],
                                  cell_type='GRU')
        elif parameters['model'] == 'rnn':
            net = models.RnnModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'],
                                  cell_type='RNN')
        elif parameters['model'] == 'attention':
            net = models.AttentionModel(seq_length, input_dim, output_dim, parameters['hidden_size'],
                                        parameters['dropout_p'])
        elif parameters['model'] == 'da-rnn':
            net = models.DaRnnModel(seq_length, input_dim, output_dim, parameters['hidden_size'],
                                    parameters['dropout_p'])
        elif parameters['model'] == 'transformer':
            net = models.TransformerModel(100, 2, parameters['dropout_p'])

        # use gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = net.to(device)
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        # use gpu

        train_model.train_rnn(net, False, parameters['num_of_epochs'], parameters['learning_rate'],
                              parameters['batch_size'], X_train, Y_train, X_test, Y_test, True, parameters['model'])



if __name__ == '__main__':
    subtype = ['H1N1', 'H3N2', 'H5N1', 'COV19']
    subtype_flag = make_dataset.subtype_selection(subtype[3])

    # model = ['gru', 'lstm', 'attention', 'rnn', 'svm', 'logistic regression']
    # model = ['svm', 'logistic regression', 'transformer']
    # model = ['logistic regression','random forest','rnn','lstm']
    model = ['transformer']
    # model = ['attention', 'gru', 'lstm', 'rnn','logistic regression']
    # model = ['logistic regression', 'random forest', 'rnn']
    # model = ['rnn']
    for model in model:
        print('\n')
        print("Experimental results with model %s on subtype_flag %s:" % (model, subtype_flag))
        main()








