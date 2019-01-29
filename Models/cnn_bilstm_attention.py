from keras import regularizers
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import Embedding, BatchNormalization, LSTM, Bidirectional, Input, \
    Concatenate, Multiply, Dot, Reshape, Activation, Lambda, Masking
from keras.models import Model
from six.moves import range
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, TensorBoard
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from Reference.seq_motifs import plot_filter_heat, plot_filter_logo, meme_intro, meme_add, make_filter_pwm, info_content
import subprocess
from keras import backend as K
from matplotlib.ticker import MaxNLocator
from keras.initializers import random_normal
import os
import scipy.stats as stats
import csv
import sys

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

plt.style.use('ggplot')
matplotlib.rcParams.update(
    {'font.family': 'Times New Roman', 'font.size': 36, 'font.weight': 'light', 'figure.dpi': 350})

OUTPATH = None
pretrained_model_score = [0.8, 0.62, 0.88, 0.89, 0.63, 0.94, 0.97, 0.94, 0.62, 0.89, 0.92,
                          0.92, 0.95, 0.96, 0.72, 0.96, 0.98, 0.77, 0.78, 0.7, 0.83, 0.87,
                          0.96, 0.97, 0.9, 0.97, 0.93, 0.94, 0.9, 0.96, 0.95]


class FigureCallback(Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []

        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 6))
        plt.pause(0.001)
        fig.subplots_adjust(hspace=0.4)
        plt.pause(0.001)
        ax1.set_title('Loss')
        plt.pause(0.001)
        ax1.set_xlabel('Epoch')
        plt.pause(0.001)
        ax1.set_ylabel('Loss')
        plt.pause(0.001)
        ax2.set_title('Accuracy')
        plt.pause(0.001)
        ax2.set_xlabel('Epoch')
        plt.pause(0.001)
        ax2.set_ylabel('Acc')
        plt.pause(0.001)

        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2

        self.epochs = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epochs += 1
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        self.ax1.clear()
        self.ax2.clear()
        self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax1.set_title('Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax2.set_title('Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Acc')
        self.ax1.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.loss, label='Training Loss')
        self.ax1.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.val_loss, label='Validation Loss')
        self.ax2.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.acc, label='Training Accuracy')
        self.ax2.plot(np.arange(1, self.epochs + 1, 1, dtype=int), self.val_acc, label='Validation Accuracy')
        self.ax1.legend()
        self.ax2.legend()

        plt.draw()
        plt.pause(0.1)

    def on_train_end(self, logs={}):
        # save graph
        plt.savefig(OUTPATH + 'Train_Val.png')

class RNATracker:

    def __init__(self, max_len, nb_classes, save_path, kfold_index):
        self.max_len = max_len
        self.nb_classes = nb_classes
        self.is_built = False
        global OUTPATH
        OUTPATH = save_path
        self.kfold_index = kfold_index

    def build_model(self, nb_filters, filters_length, pooling_size, lstm_units, embedding_vec, attention_size=50):
        """
        Original settings with batch-normalization. In practice to alleviate the noise introduced
        via paddings, batch-normalization is avoided. This function should only be used as reference.
        """
        input = Input(shape=(self.max_len,), dtype='int8')
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = embedding_layer(input)
        with tf.name_scope('first_cnn'):
            # first cnn layer
            cnn_output = MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                # BatchNormalization(axis=-1)(
                Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu', use_bias=False)(
                    embedding_output)  # )
                # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
            )
        with tf.name_scope('Second_cnn'):
            # stack another cnn layer on top
            cnn_output = MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                # BatchNormalization(axis=-1)(
                Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu', use_bias=False)(
                    cnn_output)  # )
            )
        sequence_length = cnn_output.get_shape()[1].value
        with tf.name_scope('bilstm_layer'):
            # model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
            lstm_output = Bidirectional(LSTM(lstm_units, dropout=0.1, return_sequences=True,
                                             input_shape=(sequence_length, nb_filters)))(
                Masking(mask_value=0.)(cnn_output))
            # output shape: (batch_size, time steps, hidden size=2*nb_filters)
            # to work with masking
            lstm_output = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_output)
        hidden_size = lstm_output.get_shape()[2].value
        print('sequence_length: ', sequence_length)
        print('hidden size:', hidden_size)

        with tf.name_scope('attention_module'):
            # [batch_size, time_steps, attention_size]
            context_weights = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh')(
                lstm_output)
            # [batch_size, time_steps]
            scores = Reshape((sequence_length,))(
                Dense(1, input_shape=(sequence_length, attention_size), use_bias=False)(
                    context_weights))
            # softmax probability distribution, [batch_size, sequence_length]
            attention_weights = Reshape((sequence_length, 1))(Activation("softmax")(scores))

            # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
            # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
            # [batch_size, hidden]
            output = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights]))

        preds = Dense(self.nb_classes, activation='softmax')(output)
        self.model = Model(inputs=[input], outputs=preds)
        from keras import optimizers
        # optim = optimizers.RMSprop(lr=0.001)
        optim = optimizers.nadam()
        self.model.compile(
            loss='kld',
            optimizer=optim,
            metrics=['acc']
        )
        self.is_built = True
        self.bn = True

    def build_model_advanced_masking(self, nb_filters, filters_length, pooling_size, lstm_units, embedding_vec,
                                     attention_size=50, load_weights=False, w_par_dir=None):
        """
        The canonical version of RNATracker. Analysis of the RNATracker is done without secondary annotation features.
        :param nb_filters: number of filters (size of features), same for each CNN layer
        :param filters_length: length of context windows, also same for each CNN layer
        :param pooling_size: max pooling (down sampling) size and strides
        :param lstm_units: hidden units used in the lstm
        :param embedding_vec: one-hot encoding vectors, untrainable
        :param attention_size: size for the attention weight matrix
        :return: an assembled model
        """
        print('Advanced Masking')
        input = Input(shape=(self.max_len,), dtype='int8')
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = embedding_layer(input)
        with tf.name_scope('first_cnn'):
            first_cnn = Convolution1D(nb_filters, filters_length, #kernel_regularizer=regularizers.l2(0.0001),
                                      border_mode='valid', activation='relu', use_bias=False)
            # first cnn layer
            cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                # Lambda(lambda x: x, output_shape=lambda s: s)(BatchNormalization(axis=-1)(Masking(mask_value=0.)(
                first_cnn(embedding_output))  # )))
                # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
            )

        # cnn doesn't support masking
        with tf.name_scope('Second_cnn'):
            second_cnn = Convolution1D(nb_filters, filters_length, #kernel_regularizer=regularizers.l2(0.0001),
                                       border_mode='valid', activation='relu', use_bias=False)
            # stack another cnn layer on top
            cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                # Lambda(lambda x: x, output_shape=lambda s: s)(BatchNormalization(axis=-1)(Masking(mask_value=0.)(
                second_cnn(cnn_output))  # )))
                # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
            )

        # with tf.name_scope('Third_cnn'):
        #     second_cnn = Convolution1D(nb_filters, filters_length, #kernel_regularizer=regularizers.l2(0.0001),
        #                                border_mode='valid', activation='relu', use_bias=False)
        #     # stack another cnn layer on top
        #     cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
        #         # Lambda(lambda x: x, output_shape=lambda s: s)(BatchNormalization(axis=-1)(Masking(mask_value=0.)(
        #         second_cnn(cnn_output))  # )))
        #     )

        sequence_length = cnn_output.get_shape()[1].value
        with tf.name_scope('bilstm_layer'):
            # model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
            bilstm = Bidirectional(LSTM(lstm_units, dropout=0.1, return_sequences=True,
                                        input_shape=(sequence_length, nb_filters)))
            lstm_output = bilstm(
                Masking(mask_value=0.)(cnn_output))
            # output shape: (batch_size, time steps, hidden size=2*nb_filters)
            # to work with masking
            lstm_output = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_output)

        hidden_size = lstm_output.get_shape()[2].value
        # print('sequence_length: ', sequence_length)
        # print('hidden size:', hidden_size)

        with tf.name_scope('attention_module'):
            # [batch_size, time_steps, attention_size]
            context = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh')
            context_weights = context(lstm_output)

            attention = Dense(1, input_shape=(sequence_length, attention_size), use_bias=False)
            # [batch_size, time_steps]
            scores = Reshape((sequence_length,))(attention(context_weights))
            # softmax probability distribution, [batch_size, sequence_length]
            attention_weights = Reshape((sequence_length, 1))(Activation("softmax")(scores))

            # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
            # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
            # [batch_size, hidden]
            output = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights]))

        preds = Dense(self.nb_classes, activation='softmax')(output)
        self.model = Model(inputs=[input], outputs=preds)

        from keras import optimizers
        # optim = optimizers.RMSprop()
        optim = optimizers.nadam()
        # optim = optimizers.sgd()
        self.model.compile(
            loss='kld',
            optimizer=optim,  # todo
            metrics=['acc']
        )
        if load_weights:
            import h5py
            weights_path = os.path.join(w_par_dir, 'weights.h5')
            weights = h5py.File(weights_path)
            first_cnn.set_weights([weights['model_weights']['conv1d_1'][w.name] for w in first_cnn.trainable_weights])
            second_cnn.set_weights([weights['model_weights']['conv1d_2'][w.name] for w in second_cnn.trainable_weights])
            bilstm.set_weights([weights['model_weights']['bidirectional_1'][w.name] for w in bilstm.trainable_weights])
            # context.set_weights([weights['model_weights']['dense_1'][w.name] for w in context.weights])
            # attention.set_weights([weights['model_weights']['dense_2'][w.name] for w in attention.weights])

        self.is_built = True
        self.bn = False
        self.model.summary()

    def build_model_suggestive_new_arch(self, nb_filters, filters_length, pooling_size, lstm_units, embedding_vec,
                                        attention_size=50, load_weights=False, w_par_dir=None):
        """
        Experimental, not used in the paper.
        The canonical version of RNATracker. Analysis of the RNATracker is done without secondary annotation features.
        :param nb_filters: number of filters (size of features), same for each CNN layer
        :param filters_length: length of context windows, also same for each CNN layer
        :param pooling_size: max pooling (down sampling) size and strides
        :param lstm_units: hidden units used in the lstm
        :param embedding_vec: one-hot encoding vectors, untrainable
        :param attention_size: size for the attention weight matrix
        :return: an assembled model
        """
        print('Suggestive new architecture')
        input = Input(shape=(self.max_len,), dtype='int8')
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = embedding_layer(input)
        with tf.name_scope('first_cnn'):
            # first cnn layer
            if not load_weights:
                cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                    # Lambda(lambda x: x, output_shape=lambda s: s)(BatchNormalization(axis=-1)(Masking(mask_value=0.)(
                    Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu', use_bias=False)(
                        embedding_output))  # )))
                    # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
                )
            else:
                print('Loading weights from:', w_par_dir)
                '''loads weights into first cnn layer'''
                import h5py
                if w_par_dir is None:
                    raise RuntimeError('Must specify weights dir when loading pretrained weights.')
                weights = []
                for dir in os.listdir(w_par_dir):
                    if dir == '.DS_Store':
                        continue
                    weights_path = os.path.join(w_par_dir, dir, 'weights.h5')
                    weights.append(
                        h5py.File(weights_path)['model_weights']['conv1d_1']['first_cnn/conv1d_1/kernel:0'].value)

                weights = np.concatenate(weights, axis=2)  # with shape (10, 4, 32*31=992) 992 filters in total
                total_filters = weights.shape[-1]
                selected_filters = np.random.choice(np.arange(total_filters), 512)
                print('selected filters', selected_filters)
                frozen_cnn = Convolution1D(512, filters_length, border_mode='valid', activation='relu', use_bias=False)
                cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=10, stride=10)(
                    frozen_cnn(embedding_output))
                    # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
                )
                frozen_cnn.set_weights([weights[:, :, selected_filters]])
                frozen_cnn.trainble = False

        # # cnn doesn't support masking
        # with tf.name_scope('Second_cnn'):
        #     # stack another cnn layer on top
        #     cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
        #         # Lambda(lambda x: x, output_shape=lambda s: s)(BatchNormalization(axis=-1)(Masking(mask_value=0.)(
        #         Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu', use_bias=False)(
        #             cnn_output))  # )))
        #         # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
        #     )

        sequence_length = cnn_output.get_shape()[1].value
        with tf.name_scope('bilstm_layer'):
            # model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
            lstm_output = Bidirectional(LSTM(32, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                                             input_shape=(sequence_length, nb_filters)))(
                Masking(mask_value=0.)(cnn_output))
            # output shape: (batch_size, time steps, hidden size=2*nb_filters)
            # to work with masking
            lstm_output = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_output)
        hidden_size = lstm_output.get_shape()[2].value
        # print('sequence_length: ', sequence_length)
        # print('hidden size:', hidden_size)

        with tf.name_scope('attention_module'):
            # [batch_size, time_steps, attention_size]
            context_weights = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh',
                                    kernel_initializer=random_normal(), bias_initializer=random_normal())(lstm_output)
            # [batch_size, time_steps]
            scores = Reshape((sequence_length,))(
                Dense(1, input_shape=(sequence_length, attention_size), kernel_initializer=random_normal(),
                      use_bias=False)(context_weights))
            # softmax probability distribution, [batch_size, sequence_length]
            attention_weights = Reshape((sequence_length, 1))(Activation("softmax")(scores))

            # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
            # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
            # [batch_size, hidden]
            output = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights]))

        preds = Dense(self.nb_classes, activation='softmax')(output)
        self.model = Model(inputs=[input], outputs=preds)
        self.model.compile(
            loss='kld',
            optimizer='nadam',
            metrics=['acc']
        )
        self.is_built = True
        self.bn = False
        self.model.summary()

    def build_model_advanced_masking_with_reg(self, nb_filters, filters_length, pooling_size, lstm_units, embedding_vec,
                                              attention_size=50):
        """
        To curb overfiting we experimented l1_l2 regularization, but no gains in performance is observed (actually it deterioriates quite a bit).
        Future investifation may be made to see how we can better use regularizations.
        """
        print('Advanced Masking, plus l1_l2 regularizers')
        input = Input(shape=(self.max_len,), dtype='int8')
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = embedding_layer(input)
        with tf.name_scope('first_cnn'):
            # first cnn layer
            cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                # Lambda(lambda x: x, output_shape=lambda s: s)(BatchNormalization(axis=-1)(Masking(mask_value=0.)(
                Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu', use_bias=False,
                              kernel_regularizer=regularizers.l1(1e-3))(
                    embedding_output))  # )))
                # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
            )

        # cnn doesn't support masking
        with tf.name_scope('Second_cnn'):
            # stack another cnn layer on top
            cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                # Lambda(lambda x: x, output_shape=lambda s: s)(BatchNormalization(axis=-1)(Masking(mask_value=0.)(
                Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu', use_bias=False,
                              kernel_regularizer=regularizers.l1(1e-3))(
                    cnn_output))  # )))
                # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
            )

        sequence_length = cnn_output.get_shape()[1].value
        with tf.name_scope('bilstm_layer'):
            # model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
            lstm_output = Bidirectional(LSTM(lstm_units, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                                             input_shape=(sequence_length, nb_filters),
                                             kernel_regularizer=regularizers.l1(1e-3),
                                             recurrent_regularizer=regularizers.l1(1e-3)))(
                Masking(mask_value=0.)(cnn_output))
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

        preds = Dense(self.nb_classes, activation='softmax', kernel_regularizer=regularizers.l1(1e-5))(output)
        self.model = Model(inputs=[input], outputs=preds)
        self.model.compile(
            loss='kld',
            optimizer='nadam',
            metrics=['acc']
        )
        self.is_built = True
        self.bn = False

    def build_model_separate_attention(self, nb_filters, filters_length, pooling_size, lstm_units, embedding_vec,
                                       attention_size=50):
        """
        Attention module for each of the location. Quite redundant and computationally inefficient.
        A joint attention module is used instead of this one.
        """
        input = Input(shape=(self.max_len,), dtype='int8')
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = embedding_layer(input)
        # first cnn layer
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(embedding_output)))
        # stack another cnn layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(cnn_output)))
        sequence_length = cnn_output.get_shape()[1].value
        lstm_output = Bidirectional(LSTM(lstm_units, dropout=0.2, return_sequences=True,
                                         input_shape=(sequence_length, nb_filters)))(
            cnn_output)  # shape: (batch_size, time steps, hidden size=2*nb_filters)
        hidden_size = lstm_output.get_shape()[2].value
        print('sequence_length: ', sequence_length)
        print('hidden size:', hidden_size)

        # [batch_size, time_steps, attention_size]
        context_weights_0 = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh',
                                  kernel_initializer=random_normal(), bias_initializer=random_normal())(lstm_output)
        context_weights_1 = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh',
                                  kernel_initializer=random_normal(), bias_initializer=random_normal())(lstm_output)
        context_weights_2 = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh',
                                  kernel_initializer=random_normal(), bias_initializer=random_normal())(lstm_output)
        context_weights_3 = Dense(attention_size, input_shape=(sequence_length, hidden_size), activation='tanh',
                                  kernel_initializer=random_normal(), bias_initializer=random_normal())(lstm_output)
        # [batch_size, time_steps]
        scores_0 = Reshape((sequence_length,))(
            Dense(1, input_shape=(sequence_length, attention_size), kernel_initializer=random_normal(), use_bias=False)(
                context_weights_0))
        scores_1 = Reshape((sequence_length,))(
            Dense(1, input_shape=(sequence_length, attention_size), kernel_initializer=random_normal(), use_bias=False)(
                context_weights_1))
        scores_2 = Reshape((sequence_length,))(
            Dense(1, input_shape=(sequence_length, attention_size), kernel_initializer=random_normal(), use_bias=False)(
                context_weights_2))
        scores_3 = Reshape((sequence_length,))(
            Dense(1, input_shape=(sequence_length, attention_size), kernel_initializer=random_normal(), use_bias=False)(
                context_weights_3))

        # softmax probability distribution, [batch_size, sequence_length]
        attention_weights_0 = Reshape((sequence_length, 1))(Activation("softmax")(scores_0))
        attention_weights_1 = Reshape((sequence_length, 1))(Activation("softmax")(scores_1))
        attention_weights_2 = Reshape((sequence_length, 1))(Activation("softmax")(scores_2))
        attention_weights_3 = Reshape((sequence_length, 1))(Activation("softmax")(scores_3))

        # Multiply() behaves exactly as tf.multiply() which supports shape broadcasting, so its output_shape is [batch_size, time_steps, hidden_size]
        # Lambda(lambda x: K.sum(x, axis=1, keepdims=False)) is equivalent to tf.reduce_sum(axis=1)
        # [batch_size, hidden]
        output_0 = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights_0]))
        output_1 = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights_1]))
        output_2 = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights_2]))
        output_3 = Lambda(lambda x: K.sum(x, axis=1, keepdims=False))(Multiply()([lstm_output, attention_weights_3]))

        # output = Reshape((4,))(TimeDistributed(Dense(1), input_shape=(4, hidden_size))(Reshape((4, hidden_size))(Concatenate(axis=1)([output_0, output_1, output_2, output_3]))))
        preds = Dense(self.nb_classes, activation='softmax')(
            Concatenate(axis=1)([output_0, output_1, output_2, output_3]))
        self.model = Model(inputs=[input], outputs=preds)
        self.model.compile(
            loss='kld',
            optimizer='nadam',
            metrics=['acc']
        )
        self.is_built = True

    def legacy_attention_model(self, nb_filters, filters_length, pooling_size, lstm_units, embedding_vec):
        """
        Legacy attention model. For reference purpose only.
        """
        input = Input(shape=(self.max_len,), dtype='int8')
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = embedding_layer(input)
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(embedding_output)))
        # stack another layer on top
        cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
            Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(cnn_output)))
        lstm_output = Bidirectional(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2, return_sequences=True,
                                         input_shape=cnn_output.get_shape()))(cnn_output)
        score_0 = Flatten()(Dense(1, input_dim=2 * lstm_units, activation='relu')(lstm_output))
        score_1 = Flatten()(Dense(1, input_dim=2 * lstm_units, activation='relu')(lstm_output))
        score_2 = Flatten()(Dense(1, input_dim=2 * lstm_units, activation='relu')(lstm_output))
        score_3 = Flatten()(Dense(1, input_dim=2 * lstm_units, activation='relu')(lstm_output))
        # TimeDistributed(Dense(), lstm_output)
        # 440 for stacked 2 layers cnn, 1330 for 1 single cnn
        context_weights_0 = Dense(440, activation='softmax')(score_0)
        context_weights_1 = Dense(440, activation='softmax')(score_1)
        context_weights_2 = Dense(440, activation='softmax')(score_2)
        context_weights_3 = Dense(440, activation='softmax')(score_3)

        # context_0 = Lambda(lambda lstm_output, weights: weights[0]*lstm_output[0,:])(lstm_output, context_weights_0)
        # context_1 = Lambda(lambda x, y: x * y)(lstm_output, context_weights_1)
        # context_2 = Lambda(lambda x, y: x * y)(lstm_output, context_weights_2)
        # context_3 = Lambda(lambda x, y: x * y)(lstm_output, context_weights_3)

        context_0 = Dot(axes=1)([context_weights_0, lstm_output])  # dim of 64
        context_1 = Dot(axes=1)([context_weights_1, lstm_output])
        context_2 = Dot(axes=1)([context_weights_2, lstm_output])
        context_3 = Dot(axes=1)([context_weights_3, lstm_output])

        # context_0 = Dot(axes=1)([score_0, context_weights_0]) # dim of 64
        # context_1 = Dot(axes=1)([score_1, context_weights_1])
        # context_2 = Dot(axes=1)([score_2, context_weights_2])
        # context_3 = Dot(axes=1)([score_3, context_weights_3])
        merged = Concatenate()([context_0, context_1, context_2, context_3])

        # o0 = Dense(1, input_dim=2*lstm_units, activation='relu')(context_0)
        # o1 = Dense(1, input_dim=2*lstm_units, activation='relu')(context_1)
        # o2 = Dense(1, input_dim=2*lstm_units, activation='relu')(context_2)
        # o3 = Dense(1, input_dim=2*lstm_units, activation='relu')(context_3)
        # merged = Concatenate()([o0, o1, o2, o3])
        # score = Flatten()(Dense(1, input_dim=64, activation='relu')(lstm_output))
        # context_weights = Dense(1330, activation='softmax')(score)
        # context_vector = Multiply()([context_weights, lstm_output]) # ? - 1330 - 64
        # # dense_output = Dropout(0.5)(Dense(100, activation='relu')(Flatten()(lstm_output)))
        # # Concatenate()([])
        preds = Dense(self.nb_classes, activation='softmax')(merged)
        self.model = Model(inputs=[input], outputs=preds)
        self.model.compile(
            loss='kld',
            optimizer='nadam',
            metrics=['acc']
        )

    @classmethod
    def acc(cls, y_true, y_pred):
        '''
        soft-accuracy; should never be used.
        :param y_true: target probability mass of mRNA samples
        :param y_pred: predcited probability mass of mRNA samples
        :return: averaged accuracy
        '''
        return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

    def get_feature(self, X):
        '''
        K.learning_phase() returns a binary flag
        The learning phase flag is a bool tensor (0 = test, 1 = train)
        to be passed as input to any Keras function that
        uses a different behavior at train time and test time.
        '''
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _convout1_f = K.function(inputs, [self.model.layers[2].output])  # output of first convolutional filter
        activations = _convout1_f([0] + [X])

        return activations

    def get_attention(self, X):
        """
        Get the output of attention module, which assigns weights to different parts of sequence.
        from the Activation('softmax') layer
        :param X: samples for weights attention weights will be extracted
        :return:
        """
        if self.bn:
            layer = 16
        else:
            layer = 14
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _attention_f = K.function(inputs, [
            self.model.layers[layer].output])

        return _attention_f([0] + [X])

    def get_masking(self, X):
        if self.bn:
            layer = 14
        else:
            layer = 12
        inputs = [K.learning_phase()] + [self.model.inputs[0]]
        _attention_f = K.function(inputs, [self.model.layers[layer].output])

        return _attention_f([0] + [X])

    def new_motif_location_PCC(self, filter_outs, labels, locations=['cytoplasm', 'insoluble', 'membrane', 'nuclear']):
        '''
        Computes the correlation between final predictions affected by only one filter activation, and the corresponding true localization targets.
        Pearson correlation is used for this purporse.
        '''
        # filter_outs is the activation ndarray with the shape (batch_size, sequence_length - filter_length + 1, nb_filters)
        assert (labels.shape == (filter_outs.shape[0], len(locations)))
        if self.bn:
            out_layer = 20
        else:
            out_layer = 18
        log_file = open(OUTPATH + 'pcorr.csv', 'w')
        fieldnames = []
        for loc in locations:
            fieldnames += [loc + '_corr', loc + '_pval']
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        filter_outs = np.transpose(filter_outs, (2, 0, 1))
        corr_mat = []
        # filter_outs to be in the shape (nb_filters, batch_size, time_steps)
        for i, per_filter_out in enumerate(filter_outs):
            dum_filter_outs = np.zeros(filter_outs.shape)
            dum_filter_outs[i] = per_filter_out
            bn_inputs = [K.learning_phase()] + [self.model.layers[3].input]  # shape: (?, time_steps, nb_filters)
            _output_f = K.function(bn_inputs, [
                self.model.layers[out_layer].output])
            preds = _output_f([0] + [np.transpose(dum_filter_outs, (1, 2, 0))])[0]

            corr_row = []
            log = {}
            for i, loc in enumerate(locations):
                corr, pval = stats.pearsonr(preds[:, i], labels[:, i])
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
        plt.xticks(np.arange(len(filter_outs)), np.arange(len(filter_outs)))
        plt.yticks(np.arange(len(locations)), locations)
        for i, row in enumerate(np.array(corr_mat).T):
            for j, c in enumerate(row):
                plt.text(j - .3, i + .1, round(c, 2), fontsize=8)

        plt.savefig(OUTPATH + 'pcorr.png')

    def motif_location_PCC(self, filter_outs, labels, locations=['cytoplasm', 'insoluble', 'membrane', 'nuclear']):
        '''
        computes the correlation between the average of filter activation along the feature axis and the localization target.
        Preferably spearman correlation is more suited to this
        '''
        # filter_outs is the activation ndarray with the shape (batch_size, sequence_length - filter_length + 1, nb_filters)
        assert (labels.shape == (filter_outs.shape[0], len(locations)))
        log_file = open(OUTPATH + 'pcorr.csv', 'w')
        fieldnames = []
        for loc in locations:
            fieldnames += [loc + '_corr', loc + '_pval']
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()

        filter_outs = np.transpose(filter_outs, (2, 0, 1))
        corr_mat = []
        for per_filter_out in filter_outs:
            corr_row = []
            log = {}
            per_filter_sum_act = np.average(per_filter_out, axis=1)
            for i, loc in enumerate(locations):
                corr, pval = stats.spearmanr(per_filter_sum_act, labels[:, i])
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
        plt.xticks(np.arange(len(filter_outs)), np.arange(len(filter_outs)))
        plt.yticks(np.arange(len(locations)), locations)
        for i, row in enumerate(np.array(corr_mat).T):
            for j, c in enumerate(row):
                plt.text(j - .3, i + .1, round(c, 2), fontsize=8)
        plt.savefig(OUTPATH + 'pcorr.png')

    def get_motif(self, x_train, y_train, seq_encoding_keys):
        '''
        layers[1] is the embedding layer
        layers[2] is the (first) conv layer
        weights[0] are the convolution filters;
        weights[1] are the bias;
        :return: (guess what) motifs
        '''
        weights = self.model.layers[2].get_weights()
        assert (len(weights) == 1)  # without bias
        filters = weights[0]  # in shape: (filters_length, dim, nb_filters)
        filters = np.transpose(filters, (2, 1, 0))  # in shape: (nb_filters, dim, filters_length)\
        filter_weights = []
        for filter in filters:
            # convert to pwm matrices
            filter_weights.append(filter - filter.mean(axis=0))
        filter_weights = np.array(filter_weights)

        '''original, just get 4 batches of data'''
        # get the first four batchs
        # x_train = x_train[:1024]
        # y_train = y_train[:1024]

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

        filter_outs = self.get_feature(x_train)[0]

        from Reference.seq_motifs import plot_filter_seq_heat, plot_filter_seg_heat
        plot_filter_seq_heat(np.transpose(filter_outs, (0, 2, 1)), y_train, OUTPATH + 'clustering.png')
        # plot_filter_seg_heat(np.transpose(filter_outs, (0, 2, 1)), OUTPATH + 'clustering2.png', whiten=False)
        exit()
        self.new_motif_location_PCC(filter_outs, y_train)
        print('activations after the first convolution layers', filter_outs.shape)

        motif_size = 10

        # turn x_train into symbols again
        x_train = [[seq_encoding_keys[ind] for ind in gene] for gene in x_train]
        if not os.path.exists(OUTPATH + 'heatmap_filters/'):
            os.makedirs(OUTPATH + 'heatmap_filters/')
        if not os.path.exists(OUTPATH + 'motif_logos/'):
            os.makedirs(OUTPATH + 'motif_logos/')

        meme_file = meme_intro(OUTPATH + 'filters_meme.txt', x_train)
        filters_ic = []
        for i in range(len(filter_weights)):
            plot_filter_heat(filter_weights[i, :, :], OUTPATH + 'heatmap_filters/filter{}.pdf'.format(i))
            # draw motif logos learned from the filters
            plot_filter_logo(filter_outs[:, :, i], motif_size, x_train,
                             OUTPATH + 'motif_logos/filter{}_logo'.format(i), maxpct_t=0.5)

            # make pwm from aligned sequences; nsites is the number of aligned motifs
            filter_pwm, nsites = make_filter_pwm(OUTPATH + 'motif_logos/filter{}_logo.fa'.format(i))

            if nsites < 10:
                filters_ic.append(0)
            else:
                # TODO what's a bg_gc; what's an information content of pwm
                filters_ic.append(info_content(filter_pwm))

                meme_add(meme_file, i, filter_pwm, nsites, False)

        meme_file.close()

        #################################################################
        # annotate filters
        #################################################################
        # run tomtom #-evalue 0.01
        subprocess.call('tomtom -dist pearson -thresh 0.05 -png -oc %s/tomtom %s/filters_meme.txt %s' % (
            OUTPATH, OUTPATH, basedir + '/Ray2013_rbp_RNA.meme'), shell=True)

        #    # draw some filter heatmaps
        #     plot_filter_heat(filter_weights[i, :, :], OUTPATH + 'heatmap_filters', i, seq_encoding_keys)
        #     # draw motif logos learned from the filters
        #     plot_filter_logo(filter_outs[:, :, i], motif_size, x_train,
        #                      OUTPATH + 'motif_logos', i, maxpct_t=0.5)
        #     # make pwm from aligned sequences; nsites is the number of aligned motifs
        #     filter_pwm, nsites = make_filter_pwm(OUTPATH + 'motif_logos/seq_filter{}_logo.fa'.format(i))
        #
        #     if nsites < 10:
        #         filters_ic.append(0)
        #     else:
        #         # TODO what's a bg_gc; what's an information content of pwm
        #         filters_ic.append(info_content(filter_pwm))
        #
        #         meme_add(meme_file, i, filter_pwm, nsites, False)
        #
        # meme_file.close()
        #
        # #################################################################
        # # annotate filters
        # #################################################################
        # # run tomtom #-evalue 0.01
        # subprocess.call('tomtom -dist pearson -thresh 0.05 -eps -oc %s/tomtom %s/filters_meme.txt %s' % (
        #     OUTPATH, OUTPATH, 'Ray2013_rbp_RNA.meme'), shell=True)

    def train(self, x_train, y_train, batch_size, epochs=100):
        if not self.is_built:
            print('Run build_model() before calling train opertaion.')
            return
        size_train = len(x_train)
        x_valid = x_train[int(0.9 * size_train):]
        y_valid = y_train[int(0.9 * size_train):]
        x_train = x_train[:int(0.9 * size_train)]
        y_train = y_train[:int(0.9 * size_train)]
        # early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        best_model_path = OUTPATH + 'weights_fold_{}.h5'.format(self.kfold_index)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, verbose=1)
        print(self.model.evaluate(x_train, y_train, batch_size=batch_size))
        print(self.model.evaluate(x_valid, y_valid, batch_size=batch_size))
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                              validation_data=(x_valid, y_valid), callbacks=[model_checkpoint], shuffle=True)
        # load best performing model
        self.model.load_weights(best_model_path)
        Train_Result_Optimizer = hist.history
        Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
        Train_Loss = np.array([Train_Loss]).T
        Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
        Valid_Loss = np.asarray([Valid_Loss]).T
        Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
        Train_Acc = np.array([Train_Acc]).T
        Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
        Valid_Acc = np.asarray([Valid_Acc]).T
        np.savetxt(OUTPATH + 'Train_Loss_fold_{}.txt'.format(self.kfold_index), Train_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Loss_fold_{}.txt'.format(self.kfold_index), Valid_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Train_Acc_fold_{}.txt'.format(self.kfold_index), Train_Acc, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Acc_fold_{}.txt'.format(self.kfold_index), Valid_Acc, delimiter=',')

    def evaluate(self, x_test, y_test, dataset):
        score, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score)
        print('Test accuracy:', acc)

        y_predict = self.model.predict(x_test)
        # save label and predicted values for future plotting
        np.save(OUTPATH + 'y_label_fold_{}.npy'.format(self.kfold_index), y_test)
        np.save(OUTPATH + 'y_predict_fold_{}.npy'.format(self.kfold_index), y_predict)
        if dataset == 'apex-rip':
            locations = ['KDEL', 'Mito', 'NES', 'NLS']
        elif dataset == 'cefra-seq':
            locations = ["cytoplasm", "insoluble", "membrane", "nucleus"]
        else:
            raise RuntimeError('No such dataset.')
        self.multiclass_roc_and_pr(y_test, y_predict, locations)
        return score, acc

    def multiclass_roc_and_pr(self, y_label, y_predict, locations):
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
        plt.savefig(OUTPATH + 'ROC_fold_{}.png'.format(self.kfold_index))

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
        plt.savefig(OUTPATH + 'PR_fold_{}.png'.format(self.kfold_index))

class NoATTModel:

    def __init__(self, max_len, nb_classes, save_path, kfold_index):
        self.max_len = max_len
        self.nb_classes = nb_classes
        self.is_built = False
        global OUTPATH
        OUTPATH = save_path
        self.kfold_index = kfold_index
        np.random.seed(1234)

    def build_model(self, nb_filters, filters_length, pooling_size, lstm_units, embedding_vec):
        print('Building no attention model')
        input = Input(shape=(None,))
        embedding_layer = Embedding(len(embedding_vec), len(embedding_vec[0]), weights=[embedding_vec],
                                    input_length=self.max_len,
                                    trainable=False)
        embedding_output = embedding_layer(input)
        with tf.name_scope('first_cnn'):
            # first cnn layer
            cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(embedding_output)
                # output shape is in (batch_size, steps, filters), normalizing over the feature axis which is -1
            ))
        with tf.name_scope('Second_cnn'):
            # stack another cnn layer on top
            cnn_output = Dropout(0.2)(MaxPooling1D(pool_length=pooling_size, stride=pooling_size)(
                Convolution1D(nb_filters, filters_length, border_mode='valid', activation='relu')(cnn_output))
            )

        sequence_length = cnn_output.get_shape()[1].value
        with tf.name_scope('bilstm_layer'):
            lstm_output = Bidirectional(LSTM(lstm_units, dropout=0.1, return_sequences=True,
                                             input_shape=(sequence_length, nb_filters)))(
                Masking(mask_value=np.array([0.] * nb_filters))(cnn_output))
            # output shape: (batch_size, time steps, hidden size=2*nb_filters)
        hidden_size = lstm_output.get_shape()[2].value
        print('sequence_length: ', sequence_length)
        print('hidden size:', hidden_size)

        # to work with masking
        lstm_output = Lambda(lambda x: x, output_shape=lambda s: s)(lstm_output)
        preds = Dense(self.nb_classes, activation='softmax')(Flatten()(lstm_output))
        self.model = Model(inputs=[input], outputs=preds)
        self.model.compile(
            loss='kld',
            optimizer='nadam',
            metrics=['acc']
        )
        self.is_built = True

    def train(self, x_train, y_train, batch_size, epochs=100):
        if not self.is_built:
            print('Run build_model() before calling train opertaion.')
            return
        # early_stopping = EarlyStopping(monitor='val_loss', patience=7)
        best_model_path = OUTPATH + 'weights_fold_{}.h5'.format(self.kfold_index)
        model_checkpoint = ModelCheckpoint(best_model_path, save_best_only=True)
        hist = self.model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1,
                              validation_split=0.1, callbacks=[model_checkpoint], shuffle=True)
        # load best performing model
        self.model.load_weights(best_model_path)
        Train_Result_Optimizer = hist.history
        Train_Loss = np.asarray(Train_Result_Optimizer.get('loss'))
        Train_Loss = np.array([Train_Loss]).T
        Valid_Loss = np.asarray(Train_Result_Optimizer.get('val_loss'))
        Valid_Loss = np.asarray([Valid_Loss]).T
        Train_Acc = np.asarray(Train_Result_Optimizer.get('acc'))
        Train_Acc = np.array([Train_Acc]).T
        Valid_Acc = np.asarray(Train_Result_Optimizer.get('val_acc'))
        Valid_Acc = np.asarray([Valid_Acc]).T
        np.savetxt(OUTPATH + 'Train_Loss_fold_{}.txt'.format(self.kfold_index), Train_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Loss_fold_{}.txt'.format(self.kfold_index), Valid_Loss, delimiter=',')
        np.savetxt(OUTPATH + 'Train_Acc_fold_{}.txt'.format(self.kfold_index), Train_Acc, delimiter=',')
        np.savetxt(OUTPATH + 'Valid_Acc_fold_{}.txt'.format(self.kfold_index), Valid_Acc, delimiter=',')

    def evaluate(self, x_test, y_test, dataset):
        score, acc = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score)
        print('Test accuracy:', acc)

        y_predict = self.model.predict(x_test)
        # safve label and predicted values for future plotting
        np.save(OUTPATH + 'y_label_fold_{}.npy'.format(self.kfold_index), y_test)
        np.save(OUTPATH + 'y_predict_fold_{}.npy'.format(self.kfold_index), y_predict)
        if dataset == 'apex-rip':
            locations = ['KDEL', 'Mito', 'NES', 'NLS']
        elif dataset == 'cefra-seq':
            locations = ["cytoplasm", "insoluble", "membrane", "nucleus"]
        else:
            raise RuntimeError('No such dataset.')
        self.multiclass_roc_and_pr(y_test, y_predict, locations)
        return score, acc

    def multiclass_roc_and_pr(self, y_label, y_predict, locations):
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
        plt.savefig(OUTPATH + 'ROC_fold_{}.png'.format(self.kfold_index))

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
        plt.savefig(OUTPATH + 'PR_fold_{}.png'.format(self.kfold_index))

