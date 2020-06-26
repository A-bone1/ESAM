#!/usr/bin/env python
"""Implementation of Joint Representation Learning .
Reference: Zhang, Yongfeng, et al. "Joint representation learning for top-n recommendation with heterogeneous information sources." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. ACM, 2017.
"""

import tensorflow as tf
import time

import random

from utils.evaluation.RankingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class JRL():
    """
    Here we do not use the side information.
    """

    def __init__(self, sess, num_user, num_item, learning_rate=0.5, reg_rate=0.01, epoch=500, batch_size=256,
                 verbose=False, T=1, display_step=1000):
        self.learning_rate = learning_rate
        self.epochs = epoch
        self.batch_size = batch_size
        self.reg_rate = reg_rate
        self.sess = sess
        self.num_user = num_user
        self.num_item = num_item
        self.verbose = verbose
        self.T = T
        self.display_step = display_step
        print("NeuMF.")

    def build_network(self, num_factor=10, num_neg_sample=20, hidden_dimension=10):
        self.num_neg_sample = num_neg_sample
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        self.P = tf.Variable(tf.random_normal([self.num_user, num_factor]), dtype=tf.float32)
        self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor]), dtype=tf.float32)

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        GMF = tf.multiply(user_latent_factor, item_latent_factor)

        layer_1 = tf.layers.dense(inputs=GMF, units=num_factor, kernel_initializer=tf.random_normal_initializer,
                                  activation=tf.sigmoid,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dimension, activation=tf.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_3 = tf.layers.dense(inputs=layer_2, units=hidden_dimension, activation=tf.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_4 = tf.layers.dense(inputs=layer_3, units=hidden_dimension, activation=tf.sigmoid,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        output = tf.layers.dense(inputs=layer_4, units=hidden_dimension, activation=tf.sigmoid,
                                 kernel_initializer=tf.random_normal_initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.pred_y = tf.nn.sigmoid(tf.reduce_sum(output, 1))

        self.loss = - tf.reduce_sum(
            self.y * tf.log(self.pred_y + 1e-10) + (1 - self.y) * tf.log(1 - self.pred_y + 1e-10)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
        tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q))

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        return self

    def prepare_data(self, train_data, test_data):
        '''
        You must prepare the data before train and test the model
        :param train_data:
        :param test_data:
        :return:
        '''
        t = train_data.tocoo()
        self.user = list(t.row.reshape(-1))
        self.item = list(t.col.reshape(-1))
        self.label = list(t.data)
        self.test_data = test_data

        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])

        print("data preparation finished.")
        return self

    def train(self):

        item_temp = self.item[:]
        user_temp = self.user[:]
        labels_temp = self.label[:]

        user_append = []
        item_append = []
        values_append = []
        for u in self.user:
            list_of_random_items = random.sample(self.neg_items[u], self.num_neg_sample)
            user_append += [u] * self.num_neg_sample
            item_append += list_of_random_items
            values_append += [0] * self.num_neg_sample

        item_temp += item_append
        user_temp += user_append
        labels_temp += values_append

        self.num_training = len(item_temp)
        self.total_batch = int(self.num_training / self.batch_size)
        # print(self.total_batch)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(np.array(user_temp)[idxs])
        item_random = list(np.array(item_temp)[idxs])
        labels_random = list(np.array(labels_temp)[idxs])

        # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_label = labels_random[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss = self.sess.run((self.optimizer, self.loss),
                                    feed_dict={self.user_id: batch_user, self.item_id: batch_item, self.y: batch_label})

            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cost= %.9f" % (i + 1, np.mean(loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))

    def test(self):
        evaluate(self)

    def execute(self, train_data, test_data):

        self.prepare_data(train_data, test_data)

        init = tf.global_variables_initializer()
        self.sess.run(init)

        for epoch in range(self.epochs):
            self.train()
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch))
                self.test()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id):
        return self.sess.run([self.pred_y], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items
