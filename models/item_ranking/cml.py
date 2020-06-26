#!/usr/bin/env python
"""Implementation of Collaborative Metric Learning.
Reference: Hsieh, Cheng-Kang, et al. "Collaborative metric learning." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
"""

import tensorflow as tf
import time
import numpy as np

from utils.evaluation.RankingMetrics import *

__author__ = "Shuai Zhang"
__copyright__ = "Copyright 2018, The DeepRec Project"

__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Shuai Zhang"
__email__ = "cheungdaven@gmail.com"
__status__ = "Development"


class CML():
    def __init__(self, sess, num_user, num_item, learning_rate=0.1, reg_rate=0.1, epoch=500, batch_size=500,
                 verbose=False, T=5, display_step=1000):
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
        print("CML.")

    def build_network(self, num_factor=100, margin=0.5, norm_clip_value=1):

        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.neg_item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_id')
        self.keep_rate = tf.placeholder(tf.float32)

        P = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev=1 / (num_factor ** 0.5)), dtype=tf.float32)
        Q = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=1 / (num_factor ** 0.5)), dtype=tf.float32)

        user_embedding = tf.nn.embedding_lookup(P, self.user_id)
        item_embedding = tf.nn.embedding_lookup(Q, self.item_id)
        neg_item_embedding = tf.nn.embedding_lookup(Q, self.neg_item_id)

        self.pred_distance = tf.reduce_sum(
            tf.nn.dropout(tf.squared_difference(user_embedding, item_embedding), self.keep_rate), 1)
        self.pred_distance_neg = tf.reduce_sum(
            tf.nn.dropout(tf.squared_difference(user_embedding, neg_item_embedding), self.keep_rate), 1)

        self.loss = tf.reduce_sum(tf.maximum(self.pred_distance - self.pred_distance_neg + margin, 0))

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, var_list=[P, Q])
        self.clip_P = tf.assign(P, tf.clip_by_norm(P, norm_clip_value, axes=[1]))
        self.clip_Q = tf.assign(Q, tf.clip_by_norm(Q, norm_clip_value, axes=[1]))

        return self

    def prepare_data(self, train_data, test_data):
        '''
        You must prepare the data before train and test the model
        :param train_data:
        :param test_data:
        :return:
        '''
        t = train_data.tocoo()
        self.user = t.row.reshape(-1)
        self.item = t.col.reshape(-1)
        self.num_training = len(self.item)
        self.test_data = test_data
        self.total_batch = int(self.num_training / self.batch_size)
        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print(self.total_batch)
        print("data preparation finished.")
        return self

    def train(self):
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        item_random_neg = []
        for u in user_random:
            neg_i = self.neg_items[u]
            s = np.random.randint(len(neg_i))
            item_random_neg.append(neg_i[s])

        # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item_neg = item_random_neg[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss, _, _ = self.sess.run((self.optimizer, self.loss, self.clip_P, self.clip_Q),
                                          feed_dict={self.user_id: batch_user,
                                                     self.item_id: batch_item,
                                                     self.neg_item_id: batch_item_neg,
                                                     self.keep_rate: 0.98})

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
        return -self.sess.run([self.pred_distance],
                              feed_dict={self.user_id: user_id, self.item_id: item_id, self.keep_rate: 1})[0]

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items


class CMLwarp():
    """
    To appear.


    """

    def __init__(self, sess, num_user, num_item, learning_rate=0.1, reg_rate=0.1, epoch=500, batch_size=500,
                 verbose=False, T=5, display_step=1000):
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
        print("CML warp loss.")

    def build_network(self, num_factor=100, margin=0.5, norm_clip_value=1):

        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.neg_item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='neg_item_id')

        P = tf.Variable(tf.random_normal([self.num_user, num_factor], stddev=1 / (num_factor ** 0.5)))
        Q = tf.Variable(tf.random_normal([self.num_item, num_factor], stddev=1 / (num_factor ** 0.5)))

        user_embedding = tf.nn.embedding_lookup(P, self.user_id)
        item_embedding = tf.nn.embedding_lookup(Q, self.item_id)
        neg_item_embedding = tf.nn.embedding_lookup(Q, self.neg_item_id)

        self.pred_distance = tf.reduce_sum(tf.squared_difference(user_embedding, item_embedding), 1)
        self.pred_distance_neg = tf.reduce_sum(tf.squared_difference(user_embedding, neg_item_embedding), 1)

        self.loss = tf.reduce_sum(tf.maximum(self.pred_distance - self.pred_distance_neg + margin, 0))

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, var_list=[P, Q])
        self.clip_P = tf.assign(P, tf.clip_by_norm(P, norm_clip_value, axes=[1]))
        self.clip_Q = tf.assign(Q, tf.clip_by_norm(Q, norm_clip_value, axes=[1]))

        return self

    def prepare_data(self, train_data, test_data):
        '''
        You must prepare the data before train and test the model
        :param train_data:
        :param test_data:
        :return:
        '''
        t = train_data.tocoo()
        self.user = t.row.reshape(-1)
        self.item = t.col.reshape(-1)
        self.num_training = len(self.item)
        self.test_data = test_data
        self.total_batch = int(self.num_training / self.batch_size)
        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
        print("data preparation finished.")
        return self

    def train(self):
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        user_random = list(self.user[idxs])
        item_random = list(self.item[idxs])
        item_random_neg = []
        for u in user_random:
            neg_i = self.neg_items[u]
            s = np.random.randint(len(neg_i))
            item_random_neg.append(neg_i[s])

        # train
        for i in range(self.total_batch):
            start_time = time.time()
            batch_user = user_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item = item_random[i * self.batch_size:(i + 1) * self.batch_size]
            batch_item_neg = item_random_neg[i * self.batch_size:(i + 1) * self.batch_size]

            _, loss, _, _ = self.sess.run((self.optimizer, self.loss, self.clip_P, self.clip_Q),
                                          feed_dict={self.user_id: batch_user,
                                                     self.item_id: batch_item,
                                                     self.neg_item_id: batch_item_neg})

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
        return - self.sess.run([self.pred_distance], feed_dict={self.user_id: user_id, self.item_id: item_id})[0]

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items
