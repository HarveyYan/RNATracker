import datetime
from collections import OrderedDict
from six.moves import range
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import argparse
from sklearn.model_selection import KFold
from tqdm import tqdm

plt.style.use('ggplot')
matplotlib.rcParams.update(
    {'font.family': 'Times New Roman', 'font.size': 18, 'font.weight': 'light', 'figure.dpi': 350})

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

from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding, LSTM, Bidirectional, Input, Multiply, Activation, Lambda, Add
from keras.models import Model
from keras import backend as K
from keras.initializers import random_normal
from transcript_gene_data import Gene_Wrapper
from Scripts.draw_scatter_plot import plot_scatter


def label_dist(dist):
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)


batch_size = 256
nb_classes = 4
seq_dim = 4
validation_ratio = 0.1

encoding_seq = OrderedDict([
    ('UNK', [0, 0, 0, 0]),
    ('A', [1, 0, 0, 0]),
    ('C', [0, 1, 0, 0]),
    ('G', [0, 0, 1, 0]),
    ('T', [0, 0, 0, 1]),
    ('N', [0.25, 0.25, 0.25, 0.25]),  # A or C or G or T
])

encoding_keys = list(encoding_seq.keys())
encoding_vectors = np.array(list(encoding_seq.values()))

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'apex-rip'],
                    help='choose from cefra-seq and apex-rip')
parser.add_argument('--model', type=str, default='cnn_bilstm', choices=['cnn', 'cnn_bilstm', 'resnet'],
                    help='')
parser.add_argument('--message', type=str, default="", help='')
parser.add_argument('--epochs', type=int, default=100, help='')
args = parser.parse_args()

# no clipping, no padding
gene_data = Gene_Wrapper.seq_data_loader(False, args.dataset, lower_bound=0, upper_bound=np.inf)

X = np.array([[encoding_keys.index(c) for c in gene.seq] for gene in gene_data])
y = np.array([label_dist(gene.dist) for gene in gene_data])
kf = KFold(n_splits=10, shuffle=True, random_state=1234)
folds = kf.split(X, y)

if args.dataset == "cefra-seq":
    locations = ['KDEL', 'Mito', 'NES', 'NLS']
elif args.dataset == "apex-rip":
    locations = ["cytoplasm", "insoluble", "membrane", "nucleus"]
else:
    raise RuntimeError('No such dataset')

'''prepare extract path'''
OUTPATH = os.path.join(basedir,
                       'Results/SGDModel-10foldcv/' + args.dataset + '/' + str(datetime.datetime.now()).split('.')[0]
                       .replace(':', '-').replace(' ', '-') + '-' + args.model + '-' + args.message + '/')

# OUTPATH = '/mnt/mirror/data/zichao/mRNALocalization/Results/SGDModel-10foldcv/apex-rip/2019-01-04-16-36-23-cnn_bilstm-adam/'

if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)
print('OUTPATH:', OUTPATH)


def cnn_bilstm_model(pooling_size=3, nb_filters=32, filters_length=10, lstm_units=32, attention_size=50):
    '''build model'''
    input = Input(shape=(None,), dtype='int8')
    embedding_layer = Embedding(len(encoding_vectors), len(encoding_vectors[0]), weights=[encoding_vectors],
                                input_length=None, trainable=False)
    embedding_output = embedding_layer(input)
    with tf.name_scope('first_cnn'):
        # first cnn layer
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu', input_shape=(None, 24))(
                embedding_output))
            # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        )
    with tf.name_scope('Second_cnn'):
        # stack another cnn layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu')(cnn_output))
        )

    with tf.name_scope('Third_cnn'):
        # stack another cnn layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu')(cnn_output))
        )

    with tf.name_scope('Fourth_cnn'):
        # stack another cnn layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu')(cnn_output))
        )

    with tf.name_scope('bilstm_layer'):
        lstm_output = Bidirectional(LSTM(lstm_units, dropout=0.1, return_sequences=True,
                                         input_shape=(None, nb_filters)))(cnn_output)
        # output shape: (batch_size, time steps, hidden size=2*nb_filters)

    hidden_size = lstm_output.get_shape()[2].value
    print('hidden size:', hidden_size)

    with tf.name_scope('attention_module'):
        # [batch_size, time_steps, attention_size]
        context_weights = Dense(attention_size, activation='tanh',
                                kernel_initializer=random_normal(), bias_initializer=random_normal())(lstm_output)
        # [batch_size, time_steps]
        scores = Lambda(lambda x: K.batch_flatten(x))(
            Dense(1, kernel_initializer=random_normal(), use_bias=False)(context_weights))

        # softmax probability distribution, [batch_size, sequence_length]
        attention_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(Activation("softmax")(scores))

        # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
        # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
        # [batch_size, hidden]
        output = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights]))

    preds = Dense(nb_classes, activation='softmax')(output)
    model = Model(inputs=[input], outputs=preds)
    from keras import optimizers
    # optim = optimizers.rmsprop()
    optim = optimizers.adam(lr=0.0001)
    # optim = optimizers.sgd(lr=0.001)
    model.compile(
        loss='kld',
        optimizer=optim,
        metrics=['acc']
    )
    return model


def cnn_model(pooling_size=3, nb_filters=32, filters_length=10, attention_size=50):
    '''build model'''
    input = Input(shape=(None,), dtype='int8')
    embedding_layer = Embedding(len(encoding_vectors), len(encoding_vectors[0]), weights=[encoding_vectors],
                                input_length=None, trainable=False)
    embedding_output = embedding_layer(input)
    with tf.name_scope('first_cnn'):
        # first cnn layer
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu', input_shape=(None, 24))(
                embedding_output))
            # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        )
    with tf.name_scope('Second_cnn'):
        # stack another cnn layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu')(cnn_output))
        )

    with tf.name_scope('Third_cnn'):
        # stack another cnn layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu')(cnn_output))
        )

    with tf.name_scope('Fourth_cnn'):
        # stack another cnn layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu')(cnn_output))
        )

    # with tf.name_scope('Fifth'):
    #     # stack another cnn layer on top
    #     cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
    #         Convolution1D(nb_filters, filters_length, border_mode='same', activation='relu')(cnn_output))
    #     )

    with tf.name_scope('attention_module'):
        # [batch_size, time_steps, attention_size]
        context_weights = Dense(attention_size, activation='tanh', input_shape=(None, nb_filters),
                                kernel_initializer=random_normal(), bias_initializer=random_normal())(cnn_output)
        # [batch_size, time_steps]
        scores = Lambda(lambda x: K.batch_flatten(x))(
            Dense(1, kernel_initializer=random_normal(), input_shape=(None, attention_size),
                  use_bias=False)(context_weights))

        # softmax probability distribution, [batch_size, sequence_length]
        attention_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(Activation("softmax")(scores))

        # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
        # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
        # [batch_size, hidden]
        output = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([cnn_output, attention_weights]))

    preds = Dense(nb_classes, activation='softmax')(output)
    model = Model(inputs=[input], outputs=preds)
    model.compile(
        loss='kld',
        optimizer='sgd',
        metrics=['acc']
    )
    return model

def resnet_model():
    input = Input(shape=(None,), dtype='int8')
    embedding_layer = Embedding(len(encoding_vectors), len(encoding_vectors[0]), weights=[encoding_vectors],
                                input_length=None, trainable=False)
    embedding_output = embedding_layer(input)
    with tf.name_scope('first_cnn_layer'):
        cnn_output = Dropout(0.2)(
            Convolution1D(32, 10, border_mode='same', activation='relu', use_bias=False, strides=2)(
                embedding_output)
        )

    with tf.name_scope('first_residual_block'):
        # first cnn layer
        res_output_1 = Dropout(0.2)(
            Convolution1D(32, 3, border_mode='same', activation='relu', use_bias=False)(
                cnn_output)
        )

        # stack another cnn layer on top
        res_output_1 = Dropout(0.2)(
            Convolution1D(32, 3, border_mode='same', activation='relu', use_bias=False)(
                res_output_1)
            # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        )

        res_output_1 = Add()([cnn_output, res_output_1])

    # with tf.name_scope('second_residual_block'):
    #     res_output_2 = Dropout(0.2)(
    #         Convolution1D(32, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_1)
    #     )
    #
    #     # stack another cnn layer on top
    #     res_output_2 = Dropout(0.2)(
    #         Convolution1D(32, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_2)
    #         # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
    #     )
    #
    #     res_output_2 = Add()([res_output_1, res_output_2])
    #     # 2000, 32
    #
    # with tf.name_scope('third_residual_block'):
    #     res_output_3 = Dropout(0.2)(
    #         Convolution1D(32, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_2)
    #     )
    #
    #     # stack another cnn layer on top
    #     res_output_3 = Dropout(0.2)(
    #         Convolution1D(32, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_3)
    #         # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
    #     )
    #
    #     res_output_3 = Add()([res_output_2, res_output_3])
    #     # 2000, 32

    with tf.name_scope('cnn_downsampling'):
        cnn_downsamping = Dropout(0.2)(
            Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False, strides=2)(
                res_output_1)
        )

        cnn_downsamping = Dropout(0.2)(
            Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False)(
                cnn_downsamping)
        )

        downsample_shortcut = Convolution1D(64, 1, border_mode='same', activation='relu', use_bias=False, strides=2)(
                res_output_1)
        cnn_downsamping = Add()([downsample_shortcut, cnn_downsamping])
        # 1000, 64

    with tf.name_scope('fourth_residual_block'):
        res_output_4 = Dropout(0.2)(
            Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False)(
                cnn_downsamping)
        )

        # stack another cnn layer on top
        res_output_4 = Dropout(0.2)(
            Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False)(
                res_output_4)
            # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        )

        res_output_4 = Add()([cnn_downsamping, res_output_4])

    # with tf.name_scope('fifth_residual_block'):
    #     res_output_5 = Dropout(0.2)(
    #         Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_4)
    #     )
    #
    #     # stack another cnn layer on top
    #     res_output_5 = Dropout(0.2)(
    #         Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_5)
    #         # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
    #     )
    #
    #     res_output_5 = Add()([res_output_4, res_output_5])
    #
    # with tf.name_scope('sixth_residual_block'):
    #     res_output_6 = Dropout(0.2)(
    #         Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_5)
    #     )
    #
    #     # stack another cnn layer on top
    #     res_output_6 = Dropout(0.2)(
    #         Convolution1D(64, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_6)
    #         # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
    #     )
    #
    #     res_output_6 = Add()([res_output_5, res_output_6])

    with tf.name_scope('second_cnn_downsampling'):
        cnn_downsamping_2 = Dropout(0.2)(
            Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False, strides=2)(
                res_output_4)
        )

        cnn_downsamping_2 = Dropout(0.2)(
            Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False)(
                cnn_downsamping_2)
        )

        # 500, 128

        downsample_shortcut_2 = Convolution1D(128, 1, border_mode='same', activation='relu', use_bias=False, strides=2)(
                res_output_4)
        cnn_downsamping_2 = Add()([downsample_shortcut_2, cnn_downsamping_2])

    with tf.name_scope('seventh_residual_block'):
        res_output_7 = Dropout(0.2)(
            Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False)(
                cnn_downsamping_2)
        )

        # stack another cnn layer on top
        res_output_7 = Dropout(0.2)(
            Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False)(
                res_output_7)
            # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        )

        res_output_7 = Add()([cnn_downsamping_2, res_output_7])

    # with tf.name_scope('eighth_residual_block'):
    #     res_output_8 = Dropout(0.2)(
    #         Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_7)
    #     )
    #
    #     # stack another cnn layer on top
    #     res_output_8 = Dropout(0.2)(
    #         Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_8)
    #         # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
    #     )
    #
    #     res_output_8 = Add()([res_output_7, res_output_8])
    #
    # with tf.name_scope('ninth_residual_block'):
    #     res_output_9 = Dropout(0.2)(
    #         Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_8)
    #     )
    #
    #     # stack another cnn layer on top
    #     res_output_9 = Dropout(0.2)(
    #         Convolution1D(128, 3, border_mode='same', activation='relu', use_bias=False)(
    #             res_output_9)
    #         # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
    #     )
    #
    #     res_output_9 = Add()([res_output_8, res_output_9])

    with tf.name_scope('third_cnn_downsampling'):
        cnn_downsamping_3 = Dropout(0.2)(
            Convolution1D(256, 3, border_mode='same', activation='relu', use_bias=False, strides=2)(
                res_output_7)
        )

        cnn_downsamping_3 = Dropout(0.2)(
            Convolution1D(256, 3, border_mode='same', activation='relu', use_bias=False)(
                cnn_downsamping_3)
        )

        # 500, 128

        downsample_shortcut_3 = Convolution1D(256, 1, border_mode='same', activation='relu', use_bias=False, strides=2)(
                res_output_7)
        cnn_downsamping_3 = Add()([downsample_shortcut_3, cnn_downsamping_3])

    sequence_length = cnn_downsamping_3.get_shape()[1].value
    print('sequence length:', sequence_length)
    hidden_size = cnn_downsamping_3.get_shape()[2].value
    print('hidden size:', hidden_size)

    with tf.name_scope('attention_module'):
        context_weights = Dense(50, activation='tanh', input_shape=(None, hidden_size),
                                kernel_initializer=random_normal(), bias_initializer=random_normal())(cnn_downsamping_3)
        # [batch_size, time_steps]
        scores = Lambda(lambda x: K.batch_flatten(x))(
            Dense(1, kernel_initializer=random_normal(), input_shape=(None, 50),
                  use_bias=False)(context_weights))

        # softmax probability distribution, [batch_size, sequence_length]
        attention_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(Activation("softmax")(scores))

        # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
        # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
        # [batch_size, hidden]
        output = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([cnn_downsamping_3, attention_weights]))

    preds = Dense(nb_classes, activation='softmax')(output)
    model = Model(inputs=[input], outputs=preds)
    model.compile(
        loss='kld',
        optimizer='adam',
        metrics=['acc']
    )
    return model

def multiclass_roc_and_pr(y_label, y_predict, fold, locations):
    """
    draw multiclass ROC curve: 1 against all scheme
    predicting the mode of the distribution -> converting from a regression problem to classification problem
    comment: tha doesn't help much really
    :param y_label:
    :param y_predict:
    :return:
    """
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

    plt.figure(figsize=(10, 10))
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
    plt.show()
    plt.savefig(OUTPATH + 'ROC_fold_{}.png'.format(fold))

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

    plt.figure(figsize=(10, 10))
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
    plt.savefig(OUTPATH + 'PR_fold_{}.png'.format(fold))

for fold_num, (train_indices, test_indices) in enumerate(folds):
    # if fold_num <= 0:
    #     continue
    print('Evaluating fold', fold_num)
    prev_val_loss = np.inf
    x_train = X[train_indices]
    y_train = y[train_indices]
    x_test = X[test_indices]
    y_test = y[test_indices]

    len_train = len(train_indices)
    x_valid = x_train[:int(len_train * 0.1)]
    y_valid = y_train[:int(len_train * 0.1)]
    x_train = x_train[int(len_train * 0.1):]
    y_train = y_train[int(len_train * 0.1):]

    if args.model == 'cnn':
        model = cnn_model()
    elif args.model == 'cnn_bilstm':
        model = cnn_bilstm_model()
    elif args.model == 'resnet':
        model = resnet_model()
    else:
        raise RuntimeError('No such model.')

    for e in range(args.epochs):
        print('Epoch:', e + 1)
        for i in tqdm(range(len(x_train))):
            model.train_on_batch(np.reshape(x_train[i], (1, -1)), np.reshape(y_train[i], (1, -1)))
        # for i, (x, y) in enumerate(zip(x_train, y_train)):
        #     model.train_on_batch(np.reshape(x, (1, -1)), np.reshape(y, (1, -1)))

        train_loss, train_acc = [], []
        for i, (x, y_) in enumerate(zip(x_train, y_train)):
            l, a = model.test_on_batch(np.reshape(x, (1, -1)), np.reshape(y_, (1, -1)))
            train_loss.append(l)
            train_acc.append(a)
        train_loss = np.array(train_loss)
        train_acc = np.array(train_acc)
        print('training loss: {0}, acc: {1}'.format(train_loss.mean(), train_acc.mean()))

        loss, acc = [], []
        for i, (x, y_) in enumerate(zip(x_valid, y_valid)):
            l, a = model.test_on_batch(np.reshape(x, (1, -1)), np.reshape(y_, (1, -1)))
            loss.append(l)
            acc.append(a)
        loss = np.array(loss)
        acc = np.array(acc)
        print('validation loss: {0}, acc: {1}'.format(loss.mean(), acc.mean()))
        if loss.mean() < prev_val_loss:
            print('Validation loss improved. Saving model.\n')
            model.save(os.path.join(OUTPATH, 'weights_fold_{}'.format(fold_num)))
            prev_val_loss = loss.mean()
        else:
            print('\n')

    # load best weights
    model.load_weights(os.path.join(OUTPATH, 'weights_fold_{}'.format(fold_num)))

    preds = []
    for x in x_test:
        y_pred = model.predict(np.reshape(x, (1, -1)), batch_size=1)
        preds.append(y_pred)
    preds = np.concatenate(preds, axis=0)

    loss = -np.mean(np.sum(y_test * np.log(preds), axis=-1))
    acc = len(np.where(np.argmax(preds, axis=-1) == np.argmax(y_test, axis=-1))[0]) / len(x_test)
    print('Test loss:', loss)
    print('Test accuracy:', acc)

    multiclass_roc_and_pr(y_test, preds, fold_num, locations)

    np.save(OUTPATH + 'y_label_fold_{}.npy'.format(fold_num), y_test)
    np.save(OUTPATH + 'y_predict_fold_{}.npy'.format(fold_num), preds)

    K.clear_session()

plot_scatter(OUTPATH, args.dataset)
