import pandas as pd
import numpy as np
import datetime
import os
from sklearn import preprocessing
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import argparse
# import xgboost as xgb
import sys
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
OUTPATH = None

import tensorflow as tf
gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True
config = tf.ConfigProto(gpu_options=gpu_options)
sess = tf.Session(config=config)
from keras.backend.tensorflow_backend import set_session
set_session(session=sess)

from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import utils

plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'apex-rip'],
                    help='choose from cefra-seq and apex-rip')
parser.add_argument('--algo', type=str, default='NN', choices=['NN', 'LR'], help='')
args = parser.parse_args()

if args.dataset == "cefra-seq":
    path_to_kmer = os.path.join(basedir, 'Data/cefra-seq/kmer_feature.csv')
    locations = ['KDEL', 'Mito', 'NES', 'NLS']
elif args.dataset == "apex-rip":
    path_to_kmer = os.path.join(basedir, 'Data/apex-rip/kmer_apex.csv')
    locations = ["cytoplasm", "insoluble", "membrane", "nucleus"]
else:
    raise RuntimeError('No such dataset')
nb_classes = 4


kmers = pd.read_csv(open(path_to_kmer, 'r'))

'''IDs are not used'''
del kmers['id']

'''protein-coding and length less than 4000'''
to_drop = []
for i, row in kmers.iterrows():
    if row['gene_biotype'] != 'protein_coding':# or row['length'] > 4000:
        to_drop.append(i)
kmers = kmers.drop(to_drop)
del kmers['gene_biotype']
del kmers['length']

'''generating distribution label'''
y = [[float(c) for c in label.split('_')] for label in kmers['dist']]
y = np.array([np.array(label) / np.sum(label) for label in y])
del kmers['dist']

'''get kmer features'''
X = kmers.as_matrix()

def neural_net_baseline(dataset):
    '''outpath'''
    global OUTPATH, X, y
    OUTPATH = os.path.join(basedir, './Results/kmer-baseline-10foldcv/' + dataset + '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-NN-5Mer-' + dataset+'/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    print('OUTPATH:', OUTPATH)

    '''10 folds cross validation'''
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    folds = kf.split(X, y)

    for i, (train_indices, test_indices) in enumerate(folds):
        x_train = X[train_indices]
        y_train = y[train_indices]
        x_test = X[test_indices]
        y_test = y[test_indices]

        '''splitting train and validation set'''
        size_train = x_train.shape[0]
        x_valid = x_train[:int(0.1 * size_train), :]
        y_valid = y_train[:int(0.1 * size_train), :]
        x_train = x_train[int(0.1 * size_train):, :]
        y_train = y_train[int(0.1 * size_train):, :]

        '''standardizing'''
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

        print('training samples:', x_train.shape[0])
        print('validation samples:', x_valid.shape[0])
        print('testing samples:', x_test.shape[0])

        '''assemble a baseline model'''
        dim = X.shape[1]
        input = Input(shape=(dim,))
        input = Dense(dim)(input)
        preds = Dense(nb_classes, activation='softmax')(input)
        model = Model(inputs=[input], outputs=preds)
        model.compile(
            loss='kld',
            optimizer='nadam',
            metrics=['acc']
        )
        best_model_path = OUTPATH + 'weights_fold_{}.h5'.format(i)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True)
        model.fit(x_train, y_train, batch_size=256, nb_epoch=100, verbose=1,
                  validation_data=[x_valid, y_valid], callbacks=[model_checkpoint], shuffle=True)

        model.load_weights(best_model_path)
        score, acc = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score)
        print('Test accuracy:', acc)
        y_predict = model.predict(x_test)
        np.save(OUTPATH + 'y_label_fold_{}.npy'.format(i), y_test)
        np.save(OUTPATH + 'y_predict_fold_{}.npy'.format(i), y_predict)

        multiclass_roc_and_pr(y_test, y_predict, i)
    utils.multiclass_roc(OUTPATH, os.path.join(OUTPATH, 'multi_roc.png'))
    utils.multiclass_pr(OUTPATH, os.path.join(OUTPATH, 'multi_pr.png'))
    from Scripts.draw_scatter_plot import plot_scatter
    plot_scatter(OUTPATH, dataset, randomization_test=None)


def logistic_regression_baseline(dataset):
    '''outpath'''
    global OUTPATH, X, y
    OUTPATH = os.path.join(basedir, 'Results/kmer-baseline-10foldcv/' + dataset + '/' + str(datetime.datetime.now()).split('.')[0].replace(':', '-').replace(' ', '-') + '-LR-5Mer-' + dataset+'/')
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    print('OUTPATH:', OUTPATH)

    '''10 folds cross validation'''
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    folds = kf.split(X, y)

    for i, (train_indices, test_indices) in enumerate(folds):
        x_train = X[train_indices]
        y_train = y[train_indices]
        x_test = X[test_indices]
        y_test = y[test_indices]

        '''standardizing'''
        scaler = preprocessing.StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

        print('training samples:', x_train.shape[0])
        print('testing samples:', x_test.shape[0])
        lr_model = SGDClassifier(loss='log')
        X_ = np.vstack([x_train for i in range(nb_classes)])
        y_ = np.arange(nb_classes).repeat(x_train.shape[0])
        sample_weight = y_train.T.ravel()
        lr_model.fit(X_, y_, sample_weight=sample_weight)

        y_predict = lr_model.predict_proba(x_test)
        np.save(OUTPATH + 'y_label_fold_{}.npy'.format(i), y_test)
        np.save(OUTPATH + 'y_predict_fold_{}.npy'.format(i), y_predict)

        multiclass_roc_and_pr(y_test, y_predict, i)
    from Scripts.draw_scatter_plot import plot_scatter
    plot_scatter(OUTPATH, dataset, randomization_test=None)


# def xgboost_baseline():
#     global x_train, y_train
#     x_train = np.vstack([x_valid, x_train])
#     y_train = np.vstack([y_valid, y_train])
#     xgb_model = xgb.XGBClassifier(objective='multi:softmax', silent=False, num_class=4, nthread=8, max_depth=7, n_estimators=100)
#     X = np.vstack([x_train for i in range(nb_classes)])
#     y = np.arange(nb_classes).repeat(x_train.shape[0])
#     sample_weight = y_train.T.ravel()
#     xgb_model.fit(X, y, sample_weight=sample_weight, verbose=1)
#     y_pred = xgb_model.predict_proba(x_test)
#     RNATracker.multiclass_roc_and_pr(y_test, y_pred)
#     # global x_train, y_train
#     # x_train = np.vstack([x_valid, x_train])
#     # y_train = np.vstack([y_valid, y_train])
#     # xg_train = xgb.DMatrix(x_train, label=y_train)
#     # xg_test = xgb.DMatrix(x_test, label=y_test)
#     # # setup parameters for xgboost
#     # param = {}
#     # # scale weight of positive examples
#     # param['eta'] = 0.1
#     # param['max_depth'] = 7
#     # param['silent'] = 1
#     # param['nthread'] = 8
#     # param['num_class'] = 4
#     # param['n_estimators'] = 1000
#     # param['objective'] = 'multi:softprob'
#     # watchlist = [(xg_train, 'train'), (xg_test, 'test')]
#     # num_round = 100
#     # bst = xgb.train(param, xg_train, num_round, watchlist)
#     # # Note: this convention has been changed since xgboost-unity
#     # # get prediction, this is in 1D array, need reshape to (ndata, nclass)
#     # y_pred = bst.predict(xg_test).reshape(y_test.shape[0], 4)
#     # RNATracker.multiclass_roc_and_pr(y_test, y_pred)


def multiclass_roc_and_pr(y_label, y_predict, kfold_index):
    global locations
    ''' convert distribution density y_label to one-hot encoding '''
    y_label_ = list()
    for label in y_label:
        mode = np.argmax(label)
        fill = [0, 0, 0, 0]
        fill[mode] = 1
        y_label_.append(fill)
    y_label = np.array(y_label_)

    '''ROC curve'''
    n_classes = 4
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_label[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-averaging
    fpr["micro"], tpr["micro"], _ = roc_curve(y_label.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 7))
    from itertools import cycle
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color,
                 label='{0} (AUC = {1:0.2f})'
                       ''.format(locations[i], roc_auc[i]))
    plt.plot(fpr['micro'], tpr['micro'], color='red',
             label='micro-averaging (AUC={0:0.2f})'.format(roc_auc['micro']))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(OUTPATH + 'ROC_fold_{}.png'.format(kfold_index))

    '''PR curve'''
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_label[:, i],
                                                            y_predict[:, i])
        average_precision[i] = average_precision_score(y_label[:, i], y_predict[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_label.ravel(),
                                                                    y_predict.ravel())
    average_precision["micro"] = average_precision_score(y_label, y_predict,
                                                         average="micro")

    plt.figure(figsize=(7, 7))
    lines = []
    labels = []

    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-averaging (AUC = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('{0} (AUC = {1:0.2f})'
                      ''.format(locations[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR for mRNA localization')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))
    plt.savefig(OUTPATH + 'PR_fold_{}.png'.format(kfold_index))


# xgboost_baseline()
if args.algo == 'LR':
    logistic_regression_baseline(args.dataset)
elif args.algo == 'NN':
    neural_net_baseline(args.dataset)
else:
    raise RuntimeError('No such algorithm.')

