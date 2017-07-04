import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from layers import *
import utils

class HAN():
    def __init__(self,
                 word_cell,
                 sentence_cell,
                 word_output_size,
                 sentence_output_size,
                 classes,
                 max_grad_norm,
                 embedding_size,
                 #embedding_matrix,
                 hidden_size,
                 learning_rate,
                 dropout_keep_proba,
                 device='/cpu:0',
                 is_training=None,
                 scope = None
                 ):
        self.word_cell = word_cell
        self.sentence_cell = sentence_cell
        self.word_output_size = word_output_size
        self.sentence_output_size = sentence_output_size
        self.classes = classes
        self.max_grad_norm = max_grad_norm
        #self.inputs = inputs
        self.embedding_size = embedding_size
        #self.embedding_matrix = embedding_matrix
        self.hidden_size = hidden_size
        self.dropout_keep_proba = dropout_keep_proba

        with tf.variable_scope(scope or 'tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if is_training is not None:
                self.is_training = is_training
            else:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            # self.sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='sample_weights')

            # [document x sentence x word]
            self.inputs = tf.placeholder(shape=(None, None, None), dtype=tf.int32, name='inputs')

            #
            self.embedding_matrix = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embedding_matrix')

            # [document x sentence]
            self.word_lengths = tf.placeholder(shape=(None, None), dtype=tf.int32, name='word_lengths')

            # [document]
            self.sentence_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

            # [document]
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')

            # (self.document_size,
            #  self.sentence_size,
            #  self.word_size) = tf.unstack(tf.shape(self.inputs))

            # self._init_embedding(scope)
            self.get_embedding(scope)
            #(self.batch_size, self.sentence_size, self.word_size, self.embedding_size) = self.inputs_embedding.shape
            (self.batch_size, self.sentence_size, self.word_size) = tf.unstack(tf.shape(self.inputs))
            # embeddings cannot be placed on GPU
            with tf.device(device):
                self.build(scope)

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            self.cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels=self.labels, logits=self.logits)

            #self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            self.loss = tf.reduce_mean(self.cross_entropy)
            tf.summary.scalar('loss', self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(learning_rate)
            #opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    def get_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.inputs_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

    def build(self, scope):
        with tf.variable_scope(scope):
            word_level_inputs = tf.reshape(self.inputs_embedding, [
                self.batch_size * self.sentence_size,
                self.word_size,self.embedding_size])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.batch_size * self.sentence_size])


            with tf.variable_scope('word') as scope:

                word_birnn = BiDynamicRNNLayer(
                    inputs = word_level_inputs,
                    cell_fn = self.word_cell,  # tf.nn.rnn_cell.LSTMCell,
                    n_hidden = self.hidden_size,
                    sequence_length = word_level_lengths,
                )

                # word_encoder_output, _ = BiDynamicRNNLayer(
                #     self.word_cell, self.word_cell,
                #     word_level_inputs, word_level_lengths,
                #     scope=scope)

                word_encoder_output = word_birnn.outputs

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(
                        word_encoder_output,
                        self.word_output_size,
                        scope=scope)

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(
                        word_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

            # sentence_level

            sentence_level_inputs = tf.reshape(
                word_level_output, [self.batch_size, self.sentence_size, self.word_output_size])

            with tf.variable_scope('sentence') as scope:

                sen_birnn = BiDynamicRNNLayer(
                    inputs=sentence_level_inputs,
                    cell_fn=self.sentence_cell,  # tf.nn.rnn_cell.LSTMCell,
                    n_hidden=self.hidden_size,
                    sequence_length=self.sentence_lengths,
                )
                sentence_encoder_output = sen_birnn.outputs

                # sentence_encoder_output, _ = bidirectional_rnn(
                #     self.sentence_cell, self.sentence_cell, sentence_inputs, self.sentence_lengths, scope=scope)

                with tf.variable_scope('attention') as scope:
                    sentence_level_output = task_specific_attention(
                        sentence_encoder_output, self.sentence_output_size, scope=scope)

                with tf.variable_scope('dropout'):
                    sentence_level_output = layers.dropout(
                        sentence_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )

            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    sentence_level_output, self.classes, activation_fn=None)

                self.prediction = tf.argmax(self.logits, axis=-1)

    def get_feed_data(self, x, e, y=None, class_weights=None, is_training=True):
        x_m, doc_sizes, sent_sizes = utils.batch(x)
        fd = {
            self.inputs: x_m,
            self.sentence_lengths: doc_sizes,
            self.word_lengths: sent_sizes,
            self.embedding_matrix:e,
        }
        if y is not None:
            fd[self.labels] = y
            # if class_weights is not None:
            #     fd[self.sample_weights] = [class_weights[yy] for yy in y]
            # else:
            #     fd[self.sample_weights] = np.ones(shape=[len(x_m)], dtype=np.float32)
        fd[self.is_training] = is_training
        return fd

class AN():
    def __init__(self,
                 word_cell,
                 word_output_size,
                 classes,
                 max_grad_norm,
                 embedding_size,
                 # embedding_matrix,
                 hidden_size,
                 learning_rate,
                 dropout_keep_proba,
                 device='/cpu:0',
                 is_training=None,
                 scope=None
                 ):
        self.word_cell = word_cell
        self.word_output_size = word_output_size
        self.classes = classes
        self.max_grad_norm = max_grad_norm
        # self.inputs = inputs
        self.embedding_size = embedding_size
        # self.embedding_matrix = embedding_matrix
        self.hidden_size = hidden_size
        self.dropout_keep_proba = dropout_keep_proba

        with tf.variable_scope(scope or 'tcm') as scope:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            if is_training is not None:
                self.is_training = is_training
            else:
                self.is_training = tf.placeholder(dtype=tf.bool, name='is_training')

            # self.sample_weights = tf.placeholder(shape=(None,), dtype=tf.float32, name='sample_weights')

            # [document x word]
            self.inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='inputs')

            #
            self.embedding_matrix = tf.placeholder(shape=(None, None), dtype=tf.float32, name='embedding_matrix')


            # [document]
            self.word_lengths = tf.placeholder(shape=(None,), dtype=tf.int32, name='sentence_lengths')

            # [document]
            self.labels = tf.placeholder(shape=(None,), dtype=tf.int32, name='labels')

            # (self.document_size,
            #  self.sentence_size,
            #  self.word_size) = tf.unstack(tf.shape(self.inputs))

            # self._init_embedding(scope)
            self.get_embedding(scope)
            # (self.batch_size, self.sentence_size, self.word_size, self.embedding_size) = self.inputs_embedding.shape
            (self.batch_size, self.word_size) = tf.unstack(tf.shape(self.inputs))
            # embeddings cannot be placed on GPU
            with tf.device(device):
                self.build(scope)

        with tf.variable_scope('train'):
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                logits=self.logits)

            # self.loss = tf.reduce_mean(tf.multiply(self.cross_entropy, self.sample_weights))
            self.loss = tf.reduce_mean(self.cross_entropy)
            tf.summary.scalar('loss', self.loss)

            self.accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

            tvars = tf.trainable_variables()

            grads, global_norm = tf.clip_by_global_norm(
                tf.gradients(self.loss, tvars),
                self.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            opt = tf.train.AdamOptimizer(learning_rate)
            # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

            self.train_op = opt.apply_gradients(
                zip(grads, tvars), name='train_op',
                global_step=self.global_step)

            self.summary_op = tf.summary.merge_all()

    def get_embedding(self, scope):
        with tf.variable_scope(scope):
            with tf.variable_scope("embedding") as scope:
                self.inputs_embedding = tf.nn.embedding_lookup(self.embedding_matrix, self.inputs)

    def build(self, scope):
        with tf.variable_scope(scope):
            word_level_inputs = tf.reshape(self.inputs_embedding, [
                self.batch_size,self.word_size, self.embedding_size])
            word_level_lengths = tf.reshape(
                self.word_lengths, [self.batch_size])

            with tf.variable_scope('word') as scope:
                word_birnn = BiDynamicRNNLayer(
                    inputs=word_level_inputs,
                    cell_fn=self.word_cell,  # tf.nn.rnn_cell.LSTMCell,
                    n_hidden=self.hidden_size,
                    sequence_length=word_level_lengths,
                )

                # word_encoder_output, _ = BiDynamicRNNLayer(
                #     self.word_cell, self.word_cell,
                #     word_level_inputs, word_level_lengths,
                #     scope=scope)

                word_encoder_output = word_birnn.outputs

                with tf.variable_scope('attention') as scope:
                    word_level_output = task_specific_attention(
                        word_encoder_output,
                        self.word_output_size,
                        scope=scope)

                with tf.variable_scope('dropout'):
                    word_level_output = layers.dropout(
                        word_level_output, keep_prob=self.dropout_keep_proba,
                        is_training=self.is_training,
                    )


            with tf.variable_scope('classifier'):
                self.logits = layers.fully_connected(
                    word_level_output, self.classes, activation_fn=None)

                self.prediction = tf.argmax(self.logits, axis=-1)

    def get_feed_data(self, x, e, y=None, class_weights=None, is_training=True):
        x_m, doc_sizes = utils.batch2(x)
        fd = {
            self.inputs: x_m,
            self.word_lengths: doc_sizes,
            self.embedding_matrix: e,
        }
        if y is not None:
            fd[self.labels] = y
            # if class_weights is not None:
            #     fd[self.sample_weights] = [class_weights[yy] for yy in y]
            # else:
            #     fd[self.sample_weights] = np.ones(shape=[len(x_m)], dtype=np.float32)
        else:
            y1 = []
            for i in range(len(x_m)):
                y1.append(0)
            fd[self.labels] = y1
        fd[self.is_training] = is_training
        return fd

# inputs = [[1, 2, 3], [4, 2, 1], [4, 5, 6], [3, 2, 5]]
# em = tf.constant(inputs)
# mat = tf.nn.embedding_lookup(em, [1, 3])
# with tf.Session() as sess:
#     res = sess.run(mat)
#     print res
