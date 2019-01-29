from keras.preprocessing.sequence import pad_sequences
import datetime
from collections import OrderedDict
import tensorflow as tf
import os
import numpy as np
import sys
import subprocess
import csv
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import preprocessing
import seaborn as sns
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.style.use('ggplot')
matplotlib.rcParams.update(
    {'font.family': 'Times New Roman', 'font.size': 12, 'font.weight': 'light', 'figure.dpi': 350})
weblogo_opts = '-X NO --fineprint "" --resolution "350" --format "PNG"'
weblogo_opts += ' -C "#CB2026" A A'
weblogo_opts += ' -C "#34459C" C C'
weblogo_opts += ' -C "#FBB116" G G'
weblogo_opts += ' -C "#0C8040" T T'
weblogo_opts += ' -C "#0C8040" U U'

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transcript_gene_data import Gene_Wrapper
from Reference.seq_motifs import meme_intro, meme_add, make_filter_pwm, info_content

from keras.layers.core import Dense, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding, LSTM, Bidirectional, Input, Multiply, Activation, Lambda
from keras.models import Model
from keras import backend as K
from keras.initializers import random_normal

batch_size = 256
nb_classes = 4
seq_dim = 4
ann_dim = 6

gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True
config = tf.ConfigProto(gpu_options=gpu_options)

from keras.backend.tensorflow_backend import set_session

set_session(tf.Session(config=config))
np.random.seed(1234)


def label_dist(dist):
    '''
    dummy function
    :param dist:
    :return:
    '''
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)


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
reverse_mapping = {
    'A': 'A',
    'C': 'C',
    'G': 'G',
    'T': 'U'
}

gene_data = Gene_Wrapper.seq_data_loader(False, 'cefra-seq', 0, np.inf)

X = np.array([np.array([encoding_keys.index(c) for c in gene.seq]) for gene in gene_data])
Y = np.array([label_dist(gene.dist) for gene in gene_data])


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
    optim = optimizers.adam(lr=0.0001)
    # optim = optimizers.sgd(lr=0.001)
    model.compile(
        loss='kld',
        optimizer=optim,
        metrics=['acc']
    )
    return model


OUTPATH = os.path.join(basedir, 'Results', 'SGDModel-10foldcv', 'cefra-seq', str(datetime.datetime.now()).
                       split('.')[0].replace(':', '-').replace(' ', '-') + '-visualize-motif/')
if not os.path.exists(OUTPATH):
    os.makedirs(OUTPATH)
print('OUTPATH:', OUTPATH)

'''load model'''
model = cnn_bilstm_model()
model.load_weights(
    os.path.join(basedir, 'Results', 'SGDModel-10foldcv', 'cefra-seq', '2019-01-08-11-14-51-cnn_bilstm-adam',
                 'weights_fold_0'))


def new_motif_location_PCC(filter_outs, labels, locations=('cytosol', 'insoluble', 'membrane', 'nuclear')):
    assert (labels.shape == (len(filter_outs), len(locations)))
    log_file = open(OUTPATH + 'pcorr.csv', 'w')
    fieldnames = []
    for loc in locations:
        fieldnames += [loc + '_corr', loc + '_pval']
    writer = csv.DictWriter(log_file, fieldnames=fieldnames)
    writer.writeheader()

    corr_mat = []
    all_preds = []
    for j in tqdm(range(len(filter_outs))):  # per sequence with shape [1, length, 32]
        all_dum = []
        for i in range(32):  # per filter
            dum = np.zeros(filter_outs[j].shape).astype(np.float32)
            dum[:, :, i] = filter_outs[j][:, :, i]  # per seq per filter
            all_dum.append(dum)
        all_dum = np.concatenate(all_dum, axis=0)
        all_preds.append(K.function([K.learning_phase()] + [model.layers[3].input], [model.output])([0] + [all_dum])[0][None,:,:]) # [32, 4]

    # all_preds [1024, 32, 4]
    all_preds = np.concatenate(all_preds, axis=0)
    for j in range(32):
        corr_row = []
        log = {}
        for i, loc in enumerate(locations):
            corr, pval = stats.pearsonr(all_preds[:, j, i], labels[:, i])
            log[loc + '_corr'] = corr
            log[loc + '_pval'] = pval
            corr_row.append(corr)
        corr_mat.append(np.array(corr_row))
        writer.writerow(log)
        log_file.flush()

    log_file.close()

    fig = plt.figure(figsize=(18, 3))
    ax = fig.add_subplot(111)
    ax.grid()
    res = ax.imshow(np.array(corr_mat).T, cmap=plt.cm.jet,
                    interpolation='nearest')
    cb = fig.colorbar(res, orientation="horizontal")
    plt.xticks(np.arange(32), np.arange(32))
    plt.yticks(np.arange(len(locations)), locations)
    for i, row in enumerate(np.array(corr_mat).T):
        for j, c in enumerate(row):
            plt.text(j - .3, i + .1, round(c, 2), fontsize=8)

    plt.savefig(OUTPATH + 'pcorr.png')


def plot_filter_logo(filter_outs, filter_size, seqs, out_prefix, raw_t=0, maxpct_t=None):
    # acgt = 'ACGT'
    if maxpct_t:
        all_outs = np.concatenate(filter_outs, axis=0)
        all_outs_mean = all_outs.mean()
        all_outs_norm = all_outs - all_outs_mean
        raw_t = maxpct_t * all_outs_norm.max() + all_outs_mean

    # print fasta file of positive outputs
    filter_fasta_out = open('%s.fa' % out_prefix, 'w')
    filter_count = 0
    # iter over samples
    for i in range(len(filter_outs)):
        # iter ans entire sequence
        for j in range(4, len(filter_outs[i]) - 5):
            if filter_outs[i][j] > raw_t:
                # kmer = seqs[i][j:j + filter_size] # valid padding
                kmer = seqs[i][
                       j - 4:j - 4 + filter_size]  # same padding, always 4 padded to the left, 5 padded to the right
                if 'UNK' in kmer:
                    continue
                if len(kmer) < filter_size:
                    continue
                print('>%d_%d' % (i, j), file=filter_fasta_out)
                # converting back to mRNA from cDNA
                print("".join([reverse_mapping[c] for c in kmer]), file=filter_fasta_out)
                filter_count += 1
    filter_fasta_out.close()
    print('plot logo')
    # make weblogo
    if filter_count > 0:
        weblogo_cmd = 'weblogo %s < %s.fa > %s.png' % (weblogo_opts, out_prefix, out_prefix)
        subprocess.call(weblogo_cmd, shell=True)


def plot_filter_seq_heat(filter_outs, y_train, out_pdf, whiten=True, drop_dead=True):
    # compute filter output means per sequence per filter
    filter_seqs = []
    for seq in filter_outs:
        # seq: [1, length, 32]
        tmp = []
        for j in range(32):
            tmp.append(np.mean(seq[0, :, j]))
        filter_seqs.append(np.array(tmp))
    filter_seqs = np.array(filter_seqs)

    # whiten
    if whiten:
        filter_seqs = preprocessing.scale(filter_seqs)

    # transpose
    filter_seqs = np.transpose(filter_seqs)
    # shape: (32, 1024)
    if drop_dead:
        filter_stds = filter_seqs.std(axis=1)
        filter_seqs = filter_seqs[filter_stds > 0]

    # downsample sequences
    seqs_i = np.random.randint(0, filter_seqs.shape[1], 1024)

    hmin = np.percentile(filter_seqs[:, seqs_i], 0.1)
    hmax = np.percentile(filter_seqs[:, seqs_i], 99.9)

    locations = ['cytoplasm', 'insoluble', 'membrane', 'nuclear']

    '''classifiy'''
    y_train_ = list()
    for label in y_train:
        mode = np.argmax(label)
        y_train_.append(locations[mode])
    y_train = np.array(y_train_)[seqs_i]

    sns.set(font_scale=0.8, font="Times New Roman")

    plt.figure(figsize=(10, 10))

    '''column colors'''
    colors = [(0.9, 0.14799999999999996, 0.09999999999999998),
              (0.4520000000000001, 0.9, 0.09999999999999998),
              (0.09999999999999998, 0.8519999999999998, 0.9),
              (0.5479999999999997, 0.09999999999999998, 0.9)]
    lut = dict(zip(set(y_train), colors))
    col_colors = pd.DataFrame(y_train)[0].map(lut)
    g = sns.clustermap(filter_seqs[:, seqs_i], row_cluster=True, col_cluster=True, linewidths=0, figsize=(9, 9),
                       xticklabels=False, vmin=hmin, vmax=hmax, cmap='YlGnBu', col_colors=[col_colors], metric='cosine')
    '''re-ordered'''
    # print(y_train[g.dendrogram_col.reordered_ind])

    for label in locations:
        g.ax_col_dendrogram.bar(0, 0, color=lut[label],
                                label=label, linewidth=0)
        g.ax_col_dendrogram.legend(bbox_to_anchor=(0.8, 0.9), bbox_transform=plt.gcf().transFigure, ncol=4)

    g.cax.set_position([.05, .2, .03, .45])
    plt.savefig(out_pdf, dpi=350)
    # out_png = out_pdf[:-2] + 'ng'
    # plt.savefig(out_png, dpi=300)
    plt.close()


def get_motif(x_train, y_train):
    '''newer, straitified version, doesn't yield too much improvements'''
    rations = [256] * 4
    x_train_ = []
    y_train_ = []
    for x, y in zip(x_train, y_train):
        if rations[np.argmax(y)] > 0 and np.max(y) >= 0.25:
            rations[np.argmax(y)] -= 1
            x_train_.append(x)
            y_train_.append(y)
        if np.sum(rations) == 0:
            break
    print('final rations:', rations)
    x_train = np.array(x_train_)
    y_train = np.array(y_train_)

    filter_outs = []
    for seq in x_train:
        filter_outs.append(
            K.function([K.learning_phase()] + [model.inputs[0]], [model.layers[2].output])([0] + [seq.reshape(1, -1)])[
                0])

    new_motif_location_PCC(filter_outs, y_train)
    plot_filter_seq_heat(filter_outs, y_train, OUTPATH + 'clustering.png')


    motif_size = 10
    # turn x_train back into symbols
    x_train = [[encoding_keys[ind] for ind in gene] for gene in x_train]
    if not os.path.exists(OUTPATH + 'motif_logos/'):
        os.makedirs(OUTPATH + 'motif_logos/')

    meme_file = meme_intro(OUTPATH + 'filters_meme.txt', x_train)
    filters_ic = []
    for i in range(32):
        # draw motif logos learned from the filters
        per_filter_outs = []
        for out in filter_outs:  # [1, length, 32]
            per_filter_outs.append(out[0, :, i])

        plot_filter_logo(per_filter_outs, motif_size, x_train,
                         OUTPATH + 'motif_logos/filter{}_logo'.format(i), maxpct_t=0.5)

        # make pwm from aligned sequences; nsites is the number of aligned motifs
        filter_pwm, nsites = make_filter_pwm(OUTPATH + 'motif_logos/filter{}_logo.fa'.format(i))

        if nsites < 10:
            filters_ic.append(0)
        else:
            filters_ic.append(info_content(filter_pwm))

            meme_add(meme_file, i, filter_pwm, nsites, False)

    meme_file.close()

    subprocess.call('tomtom -dist pearson -thresh 0.05 -png -oc %s/tomtom %s/filters_meme.txt %s' % (
        OUTPATH, OUTPATH, basedir + '/Ray2013_rbp_RNA.meme'), shell=True)


get_motif(X, Y)
