import datetime
import itertools
from collections import OrderedDict
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf

gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session

set_session(session=sess)

from Models.cnn_bilstm_attention import *
from transcript_gene_data import Gene_Wrapper
from keras.preprocessing.sequence import pad_sequences

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

encoding_annotation = OrderedDict([
    ('UNK', [0, 0, 0, 0, 0, 0]),  # for padding use
    ('f', [1, 0, 0, 0, 0, 0]),  # 'dangling start',
    ('t', [0, 1, 0, 0, 0, 0]),  # dangling end',
    ('i', [0, 0, 1, 0, 0, 0]),  # 'internal loop',
    ('h', [0, 0, 0, 1, 0, 0]),  # 'hairpin loop',
    ('m', [0, 0, 0, 0, 1, 0]),  # 'multi loop',
    ('s', [0, 0, 0, 0, 0, 1])  # 'stem'
])
seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))
annotation_encoding_keys = list(encoding_annotation.keys())
annotation_encoding_vectors = np.array(list(encoding_annotation.values()))


gene_ids = None

def data_stats(longest, data):
    print('average length:', np.average([len(gene.seq) for gene in data]))
    length = np.zeros(longest + 1, dtype=int)
    for gene in data:
        length[len(gene.seq)] += 1
    for i, freq in enumerate(length):
        if freq > 0:
            break
    min_len = i
    plt.ion()
    plt.figure()
    plt.title('Length Frequency')
    plt.xlabel('Length')
    plt.ylabel('Frequency')
    plt.ylim(top=50)
    plt.bar(range(1, longest + 1), length[1:longest + 1])
    # plt.bar(length,
    #         label="displaying {0} out of {1} sequences\nmin length:{2},max length:{3}".format(np.sum(length[1:10001]),
    #                                                                                           np.sum(length),
    #                                                                                           min_len, longest))
    # plt.legend()
    # plt.show()
    plt.savefig(OUTPATH + 'length_frequency.png')
    plt.pause(1)
    plt.close()

def mode_frequency(train_data, labels, name, locations):
    '''
    returns a class weight to account for the fact that mode dist is highly unbalanced
    :param train_data:
    :param labels:
    :param name:
    :return:
    '''
    mode_freq = [0, 0, 0, 0]
    length = [[], [], [], []]
    for gene, dist in zip(train_data, labels):
        mode_freq[np.argmax(dist)] += 1
        length[np.argmax(dist)].append(len(gene.seq))
    for i in range(4):
        length[i] = np.average(length[i])
    plt.figure(figsize=(12, 12))
    plt.ylabel('Transcript count / length(bp)')
    plt.ylim(0, 6000)
    plt.bar(locations, mode_freq, label='sample counts',
            color='cornflowerblue')
    plt.plot(locations, length, label='average length')
    plt.xticks(rotation=-20)
    plt.legend(loc="upper left")
    plt.subplots_adjust(left=0.15)
    plt.savefig(OUTPATH + name + '.png')

    base = np.sum(mode_freq) / 4
    class_weight = {}
    for i in range(nb_classes):
        class_weight[i] = base / mode_freq[i]
    return class_weight


batch_size = 256
nb_classes = 4


def label_dist(dist):
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)


def preprocess_data(lower_bound, upper_bound, use_annotations, dataset, max_len, randomization_test=False):
    gene_data = Gene_Wrapper.seq_data_loader(use_annotations, dataset, lower_bound, upper_bound,
                                             permute=randomization_test)

    print('padding and indexing data')
    if use_annotations:
        print('Using unified one-hot encoding for both sequence and annotation features')

        '''create unifed encoding scheme'''
        template = [0] * 24  # dim([a,c,g,t]) * dim([f,t,i,h,m,s])
        combined_encoding = OrderedDict()
        combined_encoding['UNK'] = template
        for i, (key_seq, key_ann) in enumerate(
                itertools.product(['A', 'C', 'G', 'T', 'N'], ['F', 'T', 'I', 'H', 'M', 'S'])):
            tmp = template.copy()
            if key_seq == 'N':
                for n in ['A', 'C', 'G', 'T']:
                    tmp[np.nonzero(combined_encoding[n + key_ann])[0][0]] = 0.25
                combined_encoding[key_seq + key_ann] = tmp
            else:
                tmp[i] = 1  # normal one-hot encoding as it is...
                combined_encoding[key_seq + key_ann] = tmp
        encoding_keys = list(combined_encoding.keys())
        encoding_vectors = np.array(list(combined_encoding.values()))

        print('padding and indexing data')
        X = pad_sequences(
            [[encoding_keys.index(s.upper() + a.upper()) for s, a in zip(gene.seq, gene.ann)] for gene in gene_data],
            maxlen=max_len,
            dtype=np.int8, value=encoding_keys.index('UNK'))
        y = np.array([label_dist(gene.dist) for gene in gene_data])
    else:
        encoding_keys = seq_encoding_keys
        encoding_vectors = seq_encoding_vectors
        X = pad_sequences([[encoding_keys.index(c) for c in gene.seq] for gene in gene_data],
                          maxlen=max_len,
                          dtype=np.int8, value=encoding_keys.index('UNK'))  # , truncating='post')
        y = np.array([label_dist(gene.dist) for gene in gene_data])

    global gene_ids
    gene_ids = np.array([gene.id for gene in gene_data])
    from sklearn.model_selection import KFold, StratifiedKFold

    # '''lame kfolds splitting'''
    # length = len(X)
    # fold_split_index = []
    # folds_X = []
    # folds_y = []
    # for i in range(1,10):
    #     fold_split_index.append(int(length*i/10)) # index: 0~8
    # for i in range(10):
    #     if i == 0:
    #         folds_X.append(X[:fold_split_index[0], :])
    #         folds_y.append(y[:fold_split_index[0], :])
    #     elif i == 9:
    #         folds_X.append(X[fold_split_index[8]:, :])
    #         folds_y.append(y[fold_split_index[8]:, :])
    #     else:
    #         folds_X.append(X[fold_split_index[i-1]:fold_split_index[i], :])
    #         folds_y.append(y[fold_split_index[i-1]:fold_split_index[i], :])
    #
    # return folds_X, folds_y, encoding_keys, encoding_vectors

    '''sklearn kfolds splitting'''
    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    folds = kf.split(X, y)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1234)
    # modes = []
    # for label in y:
    #     modes.append(np.argmax(label))
    # folds = kf.split(X, modes)
    return X, y, folds, encoding_keys, encoding_vectors


# starts training in CNN model
def run_model(lower_bound, upper_bound, max_len, dataset, use_annotations, use_attention, **kwargs):
    '''load data into the playground'''
    X, y, folds, encoding_keys, encoding_vectors = preprocess_data(lower_bound, upper_bound, use_annotations, dataset,
                                                                   max_len, randomization_test=kwargs['randomization'])

    # model mode maybe overridden by other parameter settings
    if use_attention:
        # print('ALL IN ONE')
        # model = RNATracker(max_len, nb_classes, OUTPATH, kfold_index=10)  # initialize
        # model.build_model_advanced_masking(nb_filters=kwargs['nb_filters'],
        #                                    filters_length=kwargs['filters_length'],
        #                                    pooling_size=kwargs['pooling_size'],
        #                                    lstm_units=kwargs['lstm_units'],
        #                                    embedding_vec=encoding_vectors)
        # model.train(X, y, batch_size)
        # exit()

        # for i in range(10):
        #     print('Evaluating KFolds {}/10'.format(i + 1))
        #     model = RNATracker(max_len, nb_classes, OUTPATH, kfold_index=i)  # initialize
        #     model.build_model_advanced_masking(nb_filters=kwargs['nb_filters'], filters_length=kwargs['filters_length'],
        #                                        pooling_size=kwargs['pooling_size'], lstm_units=kwargs['lstm_units'],
        #                                        embedding_vec=encoding_vectors)
        #     x_valid = folds_X[i]
        #     y_valid = folds_y[i]
        #     x_train = np.concatenate((folds_X[:i] + folds_X[i + 1:]), axis=0)
        #     y_train = np.concatenate((folds_y[:i] + folds_y[i + 1:]), axis=0)
        #     model.train(x_train, y_train, batch_size)
        #     model.evaluate(x_valid, y_valid)
        #     K.clear_session()
        for i, (train_indices, test_indices) in enumerate(folds):
            print('Evaluating KFolds {}/10'.format(i + 1))
            # from Models.RBPBindingModel import RBPBinder
            # model = RBPBinder(max_len, nb_classes, OUTPATH)
            model = RNATracker(max_len, nb_classes, OUTPATH, kfold_index=i)  # initialize
            if kwargs['load_pretrain']:
                model.build_model_advanced_masking(nb_filters=kwargs['nb_filters'],
                                                   filters_length=kwargs['filters_length'],
                                                   pooling_size=kwargs['pooling_size'],
                                                   lstm_units=kwargs['lstm_units'],
                                                   embedding_vec=encoding_vectors, load_weights=True,
                                                   w_par_dir=kwargs['weights_dir'])
            else:
                model.build_model_advanced_masking(nb_filters=kwargs['nb_filters'],
                                                   filters_length=kwargs['filters_length'],
                                                   pooling_size=kwargs['pooling_size'],
                                                   lstm_units=kwargs['lstm_units'],
                                                   embedding_vec=encoding_vectors)
            if kwargs['expand_isoforms']:
                x_train, y_train = [], []
                genes = {}
                flag = False
                with open(os.path.join(basedir, 'Data/Homo_sapiens.GRCh38.cdna.all.fa'), "r") as cdna:
                    for line in cdna:
                        if line[0] == '>':
                            if flag:  # only protein_coding mRNAs
                                if id in gene_ids[train_indices] and gene_biotype == 'gene_biotype:protein_coding' and transcript_biotype == 'transcript_biotype:protein_coding':
                                    if id in genes:
                                        genes[id].append(seq)
                                    else:
                                        genes[id] = [seq]
                            else:
                                flag = True

                            seq = ""
                            tokens = line.split()
                            id = tokens[3].split(':')[1].split('.')[0]
                            gene_biotype = tokens[4]
                            transcript_biotype = tokens[5]
                        else:
                            seq += line[:-1]
                print('all training fold isoforms loaded')
                for id, val in genes.items():
                    loc_ind = np.where(gene_ids[train_indices] == id)[0]
                    x_train += val
                    y_train.append(np.repeat(y[train_indices][loc_ind], len(val), axis=0))
                x_train = pad_sequences([[encoding_keys.index(c) for c in seq] for seq in x_train],
                                        maxlen=max_len,
                                        dtype=np.int8, value=encoding_keys.index('UNK'))  # , truncating='post')
                y_train = np.concatenate(y_train, axis=0)
                print('expanded training set', x_train.shape[0])
                model.train(x_train, y_train, batch_size, kwargs['epochs'])
            else:
                model.train(X[train_indices], y[train_indices], batch_size, kwargs['epochs'])
            model.evaluate(X[test_indices], y[test_indices], dataset)
            K.clear_session()
    else:
        '''load a model that does not use attention, for comparison'''
        for i, (train_indices, test_indices) in enumerate(folds):
            print('Evaluating KFolds {}/10'.format(i + 1))
            model = NoATTModel(max_len, nb_classes, OUTPATH, i)
            model.build_model(nb_filters=kwargs['nb_filters'], filters_length=kwargs['filters_length'],
                              pooling_size=kwargs['pooling_size'], lstm_units=kwargs['lstm_units'],
                              embedding_vec=encoding_vectors)
            model.train(X[train_indices], y[train_indices], batch_size, kwargs['epochs'])
            model.evaluate(X[test_indices], y[test_indices], dataset)
            K.clear_session()

    '''Always draw scatter plots for each experiment we run'''
    from Scripts.draw_scatter_plot import plot_scatter
    plot_scatter(OUTPATH, dataset, randomization_test=kwargs['randomization'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''Model parameters'''
    parser.add_argument('--lower_bound', type=int, default=0, help='set lower bound on sample sequence length')
    parser.add_argument('--upper_bound', type=int, default=4000, help='set upper bound on sample sequence length')
    parser.add_argument('--max_len', type=int, default=4000,
                        help="pad or slice sequences to a fixed length in preprocessing")
    parser.add_argument('--nb_filters', type=int, default=32, help='number of convolutional filters')
    parser.add_argument('--filters_length', type=int, default=10, help='filters length')
    parser.add_argument('--pooling_size', type=int, default=3, help='maxpooling size')
    parser.add_argument('--lstm_units', type=int, default=32, help='lstm hidden size')
    parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'apex-rip'],
                        help='choose from cefra-seq and apex-rip')
    parser.add_argument('--epochs', type=int, default=100, help='')

    '''Experiment parameters'''
    # parser.add_argument('--draw_motif_logo', action="store_true",
    #                     help='draw the motif our own filters have learned')
    # optional operation
    parser.add_argument("--use_annotations", action="store_true",
                        help="include cDNA secondary structure annotations into the model")
    parser.add_argument("--expand_isoforms", action="store_true",
                        help="")
    # parser.add_argument("--use_3UTR", action="store_true",
    #                     help="only use 3 UTR data if set to True; or only cDNA if set to False")
    # parser.add_argument("--use_classweights", action="store_true",
    #                     help='measure to account for imbalanced class samples')
    parser.add_argument("--use_attention", action="store_true",
                        help="leverage state of the art attention mechanism in this prediction problem")
    parser.add_argument("--message", type=str, default="", help="append to the dir name")
    parser.add_argument("--load_pretrain", action="store_true",
                        help="load pretrained CNN weights to the first convolutional layers")
    parser.add_argument("--weights_dir", type=str, default="",
                        help="Must specificy pretrained weights dir, if load_pretrain is set to true. Only enter the relative path respective to the root of this project.")
    parser.add_argument("--randomization", type=int, default=None,
                        help="Running randomization test with three settings - {1,2,3}.")
    # parser.add_argument("--nb_epochs", type=int, default=20, help='choose the maximum number of iterations over training samples')
    args = parser.parse_args()

    if args.randomization is not None:
        print('Randomization test is on.')
        OUTPATH = os.path.join(basedir,
                               'Results/RNATracker-randomization/'+ args.dataset+ '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-' + args.message +'/')
    else:
        OUTPATH = os.path.join(basedir,
                               'Results/RNATracker-10foldcv/' + args.dataset+ '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-' + args.message + '/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    print('OUTPATH:', OUTPATH)
    del args.message

    args.weights_dir = os.path.join(basedir, args.weights_dir)

    for k, v in vars(args).items():
        print(k, ':', v)

    run_model(**vars(args))
