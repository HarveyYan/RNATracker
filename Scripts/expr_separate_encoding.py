import datetime
from collections import OrderedDict
import argparse
import tensorflow as tf
import os
import matplotlib
import sys

matplotlib.rcParams.update({'font.size': 16, 'figure.dpi': 350})

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keras.backend.tensorflow_backend import set_session
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions()
gpu_options.allow_growth = True
config = tf.ConfigProto(gpu_options=gpu_options)
set_session(tf.Session(config=config))

from Models.cnn_bilstm_attention import *
from transcript_gene_data import Gene_Wrapper
from Scripts.draw_scatter_plot import plot_scatter
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

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
    ('F', [1, 0, 0, 0, 0, 0]),  # 'dangling start',
    ('T', [0, 1, 0, 0, 0, 0]),  # dangling end',
    ('I', [0, 0, 1, 0, 0, 0]),  # 'internal loop',
    ('H', [0, 0, 0, 1, 0, 0]),  # 'hairpin loop',
    ('M', [0, 0, 0, 0, 1, 0]),  # 'multi loop',
    ('S', [0, 0, 0, 0, 0, 1])  # 'stem'
])

seq_encoding_keys = list(encoding_seq.keys())
seq_encoding_vectors = np.array(list(encoding_seq.values()))
annotation_encoding_keys = list(encoding_annotation.keys())
annotation_encoding_vectors = np.array(list(encoding_annotation.values()))

batch_size = 256
nb_classes = 4
seq_dim = 4
ann_dim = 6

validation_ratio = 0.1


def label_dist(dist):
    '''
    dummy function
    :param dist:
    :return:
    '''
    assert (len(dist) == 4)
    return np.array(dist) / np.sum(dist)


def preprocess_data(dataset):
    gene_data = Gene_Wrapper.seq_data_loader(True, dataset, 0, 4000, permute=False)

    X_seq = pad_sequences([[seq_encoding_keys.index(c.upper()) for c in gene.seq] for gene in gene_data],
                          maxlen=4000,
                          dtype=np.int8, value=seq_encoding_keys.index('UNK'))  # , truncating='post')
    X_ann = pad_sequences([[annotation_encoding_keys.index(a.upper()) for a in gene.ann] for gene in gene_data],
                          maxlen=4000,
                          dtype=np.int8, value=annotation_encoding_keys.index('UNK'))  # , truncating='post')
    y = np.array([label_dist(gene.dist) for gene in gene_data])

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    folds = kf.split(X_seq, y)
    return X_seq, X_ann, y, folds


def build_model_separate_encoding(nb_filters, filters_length, pooling_size, lstm_units, seq_embedding_vec,
                                  ann_embedding_vec, attention_size=50):
    seq_input = Input(shape=(4000,), dtype='int8')
    seq_embedding_layer = Embedding(len(seq_embedding_vec), len(seq_embedding_vec[0]), weights=[seq_embedding_vec],
                                    input_length=4000,
                                    trainable=False)
    seq_embedding_output = seq_embedding_layer(seq_input)
    with tf.name_scope('seq_first_cnn'):
        # first cnn layer
        seq_cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(
                seq_embedding_output))
            # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        )
    with tf.name_scope('seq_second_cnn'):
        # stack another cnn layer on top
        seq_cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(seq_cnn_output)))

    ann_input = Input(shape=(4000,), dtype='int8')
    ann_embedding_layer = Embedding(len(ann_embedding_vec), len(ann_embedding_vec[0]), weights=[ann_embedding_vec],
                                    input_length=4000,
                                    trainable=False)
    ann_embedding_output = ann_embedding_layer(ann_input)
    with tf.name_scope('ann_first_cnn'):
        # first cnn layer
        ann_cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(
                ann_embedding_output))
            # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        )
    with tf.name_scope('ann_second_cnn'):
        # stack another cnn layer on top
        ann_cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(ann_cnn_output)))

    sequence_length = seq_cnn_output.get_shape()[1].value
    with tf.name_scope('bilstm_layer'):
        # model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
        lstm_output = Bidirectional(LSTM(lstm_units, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                                         input_shape=(sequence_length, nb_filters)))(
            Masking(mask_value=0.)(Concatenate(axis=-1)([seq_cnn_output, ann_cnn_output])))
        # output shape: (batch_size, time steps, hidden size=2*nb_filters)
        # to work with masking
        lstm_output = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_output)
    hidden_size = lstm_output.get_shape()[2].value
    print('sequence_length: ', sequence_length)
    print('hidden size:', hidden_size)

    with tf.name_scope('attention_module'):
        # [batch_size, time_steps, attention_size]
        context_weights = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh',
                                kernel_initializer=random_normal(), bias_initializer=random_normal())(lstm_output)
        # [batch_size, time_steps]
        scores = Reshape((sequence_length,))(
            Dense(1, input_shape=(sequence_length, attention_size), kernel_initializer=random_normal(),
                  use_bias=False)(
                context_weights))
        # softmax probability distribution, [batch_size, sequence_length]
        attention_weights = Reshape((sequence_length, 1))(Activation("softmax")(scores))

        # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
        # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
        # [batch_size, hidden]
        output = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights]))

    preds = Dense(4, activation='softmax')(output)
    model = Model(inputs=[seq_input, ann_input], outputs=preds)
    model.compile(
        loss='kld',
        optimizer='nadam',
        metrics=['acc']
    )
    return model


def multiclass_roc_and_pr(y_label, y_predict, kfold_index, locations):
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


# starts training in CNN model
def run_model(dataset, locations):
    '''load data into the playground'''
    X_seq, X_ann, y, folds = preprocess_data(dataset)

    for i, (train_indices, test_indices) in enumerate(folds):
        print('Evaluating KFolds {}/10'.format(i + 1))
        model = build_model_separate_encoding(32, 10, 3, 32, seq_encoding_vectors, annotation_encoding_vectors)
        x_train_seq = X_seq[train_indices]
        x_train_ann = X_ann[train_indices]
        y_train = y[train_indices]

        # early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        best_model_path = OUTPATH + 'weights_fold_{}.h5'.format(i)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True)
        hist = model.fit([x_train_seq, x_train_ann], y_train, batch_size=256, nb_epoch=100, verbose=1,
                         validation_split=0.1, callbacks=[model_checkpoint], shuffle=True)
        # load best performing model
        model.load_weights(best_model_path)
        Train_Result_Optimizer = hist.history
        Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
        Train_Loss = np.array([Train_Loss]).T
        Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
        Valid_Loss = np.asarray([Valid_Loss]).T
        Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
        Train_Acc = np.array([Train_Acc]).T
        Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
        Valid_Acc = np.asarray([Valid_Acc]).T
        np.savetxt(OUTPATH + 'Train_Loss_fold_{}.txt'.format(i), Train_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Loss_fold_{}.txt'.format(i), Valid_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Train_Acc_fold_{}.txt'.format(i), Train_Acc, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Acc_fold_{}.txt'.format(i), Valid_Acc, delimiter=',')

        '''evaluation'''
        x_test_seq = X_seq[test_indices]
        x_test_ann = X_ann[test_indices]
        y_test = y[test_indices]
        score, acc = model.evaluate([x_test_seq, x_test_ann], y_test, verbose=0)
        print('Test loss:', score)
        print('Test accuracy:', acc)

        y_predict = model.predict([x_test_seq, x_test_ann])
        # save label and predicted values for future plotting
        np.save(OUTPATH + 'y_label_fold_{}.npy'.format(i), y_test)
        np.save(OUTPATH + 'y_predict_fold_{}.npy'.format(i), y_predict)
        multiclass_roc_and_pr(y_test, y_predict, i, locations)
        K.clear_session()
    plot_scatter(OUTPATH, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cefra-seq', choices=['cefra-seq', 'apex-rip'],
                        help='choose fsgdrom cefra-seq and apex-rip')
    args = parser.parse_args()

    if args.dataset == "cefra-seq":
        locations = ['KDEL', 'Mito', 'NES', 'NLS']
    elif args.dataset == "apex-rip":
        locations = ["cytoplasm", "insoluble", "membrane", "nucleus"]
    else:
        raise RuntimeError('No such dataset')

    OUTPATH = './Results/SeparateEncoding-10foldcv/' + args.dataset + '/' + str(datetime.datetime.now()).split('.')[0].replace(':',
                                                                                                          '-').replace(
        ' ', '-') + '/'

    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)
    print('OUTPATH:', OUTPATH)

    run_model(args.dataset, locations)
