#!/usr/bin/env python
"""Implementation of Neural Collaborative Filtering.
Reference: He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th International Conference on World Wide Web. International World Wide Web Conferences Steering Committee, 2017.
"""

import tensorflow as tf
from tensorflow.contrib import layers
import time
import numpy as np
import random
from test.KMMD import *
from test.center_loss import *
from utils.evaluation.RankingMetrics import *




class NeuMF_my_cikm_que():
    def __init__(self, sess, num_user, num_item, learning_rate=0.5, reg_rate=0.01, epoch=500, batch_size=25,
                 verbose=True, T=1, display_step=1000):
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
        print("NeuMF_cikm.")

    def build_network(self, num_factor=10, num_factor_mlp=64, hidden_dimension=10, num_neg_sample=30):
        self.num_neg_sample = num_neg_sample
        self.user_id = tf.placeholder(dtype=tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.int32, shape=[None], name='item_id')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

        self.P = tf.Variable(tf.random_normal([self.num_user, num_factor]), dtype=tf.float32)
        self.Q = tf.Variable(tf.random_normal([self.num_item, num_factor]), dtype=tf.float32)

        self.mlp_P = tf.Variable(tf.random_normal([self.num_user, num_factor_mlp]), dtype=tf.float32)
        self.mlp_Q = tf.Variable(tf.random_normal([self.num_item, num_factor_mlp]), dtype=tf.float32)

        user_latent_factor = tf.nn.embedding_lookup(self.P, self.user_id)
        item_latent_factor = tf.nn.embedding_lookup(self.Q, self.item_id)
        mlp_user_latent_factor = tf.nn.embedding_lookup(self.mlp_P, self.user_id)
        mlp_item_latent_factor = tf.nn.embedding_lookup(self.mlp_Q, self.item_id)

        GMF = tf.multiply(user_latent_factor, item_latent_factor)

        layer_1 = tf.layers.dense(inputs=tf.concat([mlp_item_latent_factor, mlp_user_latent_factor], axis=1),
                                  units=num_factor_mlp * 2, kernel_initializer=tf.random_normal_initializer,
                                  activation=tf.nn.relu,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dimension * 8, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_3 = tf.layers.dense(inputs=layer_2, units=hidden_dimension * 4, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        layer_4 = tf.layers.dense(inputs=layer_3, units=hidden_dimension * 2, activation=tf.nn.relu,
                                  kernel_initializer=tf.random_normal_initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))
        MLP = tf.layers.dense(inputs=layer_4, units=hidden_dimension, activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer,
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.pred_y = tf.nn.sigmoid(tf.reduce_sum(tf.concat([GMF, MLP], axis=1), 1))

        # self.pred_y = tf.layers.dense(inputs=tf.concat([GMF, MLP], axis=1), units=1, activation=tf.sigmoid, kernel_initializer=tf.random_normal_initializer, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        self.loss = - tf.reduce_sum(
            self.y * tf.log(self.pred_y + 1e-10) + (1 - self.y) * tf.log(1 - self.pred_y + 1e-10)) \
                    + tf.losses.get_regularization_loss() + self.reg_rate * (
        tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) + tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))

        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        return self
    def item_side(self,item_emb, num_factor_mlp=64,hidden_dimension=10,reuse=False):
        with tf.variable_scope("item_side", reuse=reuse):
            layer_1 = tf.layers.dense(inputs=item_emb,
                                      units=num_factor_mlp * 2, kernel_initializer=tf.random_normal_initializer,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='l1',reuse=reuse)
            layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dimension * 8, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='l2', reuse=reuse)
            layer_3 = tf.layers.dense(inputs=layer_2, units=hidden_dimension * 4, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='l3', reuse=reuse)
            layer_4 = tf.layers.dense(inputs=layer_3, units=128, activation=None,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='item_feature', reuse=reuse)
            output=tf.nn.l2_normalize(layer_4, dim=1)
            return output

    def user_side(self,user_emb, num_factor_mlp=64,hidden_dimension=10,reuse=False):
        with tf.variable_scope("user_side", reuse=reuse):
            layer_1 = tf.layers.dense(inputs=user_emb,
                                      units=num_factor_mlp * 2, kernel_initializer=tf.random_normal_initializer,
                                      activation=tf.nn.relu,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='l1',reuse=reuse)
            layer_2 = tf.layers.dense(inputs=layer_1, units=hidden_dimension * 8, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='l2', reuse=reuse)
            layer_3 = tf.layers.dense(inputs=layer_2, units=hidden_dimension * 4, activation=tf.nn.relu,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='l3', reuse=reuse)
            layer_4 = tf.layers.dense(inputs=layer_3, units=128, activation=None,
                                      kernel_initializer=tf.random_normal_initializer,
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.reg_rate),
                                      name='user_feature', reuse=reuse)
            output=tf.nn.l2_normalize(layer_4, dim=1)
            return output


    def build_network_my(self, num_factor=10, num_factor_mlp=64, hidden_dimension=10, num_neg_sample=30):
        print("my network query")
        self.num_neg_sample = num_neg_sample
        self.user_id = tf.placeholder(dtype=tf.string, shape=[None], name='user_id')
        self.item_id = tf.placeholder(dtype=tf.string, shape=[None], name='item_id')
        self.query_id=tf.placeholder(dtype=tf.string, shape=[None], name='query_id')
        ##########################################################################3
        self.target_item_id=tf.placeholder(dtype=tf.string,shape=[None],name='target_item_id')
        self.hot_item_id=tf.placeholder(dtype=tf.string,shape=[None],name='hot_item_id')
        self.long_item_id=tf.placeholder(dtype=tf.string,shape=[None],name='long_item_id')
        ###########################################################################
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')
        ###################################################################################
        a = {'user': self.user_id,'query':self.query_id}
        b={'item':self.item_id}
        c={'item':self.target_item_id}
        d = {'user_low': self.user_id}
        e = {'item_low': self.item_id}
        f={'item_low':self.target_item_id}
        h={'item':self.hot_item_id}
        l={'item':self.long_item_id}
        with tf.variable_scope(name_or_scope='embedding',reuse=tf.AUTO_REUSE) as scope:
            id_feature1 = layers.sparse_column_with_hash_bucket(
            column_name='user',
            hash_bucket_size=190000
            # use_hashmap=use_hashmap
        )
            #######################3
            id_feature_query = layers.sparse_column_with_hash_bucket(
                column_name='query',
                hash_bucket_size=1300
                # use_hashmap=use_hashmap
            )


            shared_embedding_columns_query = layers.embedding_column(
            id_feature_query,
            dimension=16,
            combiner="mean")
            #######################
            id_feature2 = layers.sparse_column_with_hash_bucket(
            column_name='item',
            hash_bucket_size=120000
            # use_hashmap=use_hashmap
        )

            shared_embedding_columns1 = layers.embedding_column(
            id_feature1,
            dimension=64,
            combiner="mean")
        #
        #
            shared_embedding_columns2 = layers.embedding_column(
            id_feature2,
            dimension=64,
            combiner="mean")
            a1 = []
            a1.append(shared_embedding_columns1)
            a1.append(shared_embedding_columns_query)
            b1=[]
            b1.append(shared_embedding_columns2)
        #
            mlp_user_latent_factor = layers.input_from_feature_columns(a, a1, scope='user')
            mlp_item_latent_factor = layers.input_from_feature_columns(b, b1, scope='item')
            mlp_target_item_latent_factor=layers.input_from_feature_columns(c, b1, scope='item')
#########################################################################################
            mlp_hot_item_latent_factor=layers.input_from_feature_columns(h, b1, scope='item')
            mlp_long_item_latent_factor=layers.input_from_feature_columns(l, b1, scope='item')
#########################################################################################


            id_feature3 = layers.sparse_column_with_hash_bucket(
            column_name='user_low',
            hash_bucket_size=190000
            # use_hashmap=use_hashmap
        )

            id_feature4 = layers.sparse_column_with_hash_bucket(
            column_name='item_low',
            hash_bucket_size=120000
            # use_hashmap=use_hashmap
        )

            shared_embedding_columns3 = layers.embedding_column(
            id_feature3,
            dimension=10,
            combiner="mean")
        #
        #
            shared_embedding_columns4 = layers.embedding_column(
            id_feature4,
            dimension=10,
            combiner="mean")
            d1 = []
            d1.append(shared_embedding_columns3)
            e1=[]
            e1.append(shared_embedding_columns4)
        #
            user_latent_factor = layers.input_from_feature_columns(d, d1, scope='user_low')
            item_latent_factor = layers.input_from_feature_columns(e, e1, scope='item_low')
            target_item_latent_factor = layers.input_from_feature_columns(f, e1, scope='item_low')
        ###################################################################################

###################################################################################################

###################################################################################################
        GMF = tf.multiply(user_latent_factor, item_latent_factor)
#####################################################################
        GMF_target= tf.multiply(user_latent_factor, target_item_latent_factor)
#####################################################################
        user_feature=self.user_side(mlp_user_latent_factor)
        item_feature=self.item_side(mlp_item_latent_factor)
        #########################################################
        target_item_feature=self.item_side(mlp_target_item_latent_factor,reuse=True)

        hot_item_feature=self.item_side(mlp_hot_item_latent_factor,reuse=True)
        long_item_feature=self.item_side(mlp_long_item_latent_factor,reuse=True)
        #########################################################
        self.pair_loss=0
        self.resort_item=[]
        self.resort_label=[]
        for i in range(0,self.batch_size):
            temp1=[]
            temp2=[]

            temp1.append(item_feature[i*self.batch_size:(i+1)*self.batch_size,:])
            temp2.append(self.y[i*self.batch_size:(i+1)*self.batch_size])
            self.resort_item.append(temp1)
            self.resort_label.append(temp2)
        discriminative_loss=[]

        for i in range(0,self.batch_size):
            discriminative_loss.append(get_center_loss(tf.reshape(self.resort_item[i],(-1,128)),tf.reshape(self.resort_label[i],(-1,1)),2))

        for i in range(0,self.batch_size):
            self.pair_loss=self.pair_loss+discriminative_loss[i]/self.batch_size
        #########################################################
        self.userF=user_feature
        self.itemF=item_feature
        self.pred_y = tf.nn.sigmoid(tf.reduce_sum(tf.concat([GMF,5*tf.multiply(user_feature, item_feature)], axis=1), 1))
        self.pred_long=tf.nn.sigmoid(tf.reduce_sum(tf.concat([GMF_target,5*tf.multiply(user_feature, target_item_feature)], axis=1), 1))
        # self.pred_y = tf.layers.dense(inputs=tf.concat([GMF, MLP], axis=1), units=1, activation=tf.sigmoid, kernel_initializer=tf.random_normal_initializer, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self.reg_rate))

        #Pseudo label
        self.p1 = tf.reshape(tf.gather(self.pred_long, tf.reshape(tf.where(tf.less(self.pred_long, 0.2)), [-1, ])), [-1, 1])
        self.p2 = tf.reshape(tf.gather(self.pred_long, tf.reshape(tf.where(tf.greater(self.pred_long, 0.8)), [-1, ])), [-1, 1])
        self.tar1 = tf.maximum(0.0, tf.reduce_mean(
            -self.p1 * tf.log(tf.clip_by_value(self.p1, 0.005, 1)))) #/ self.batch_size
        self.tar2 = tf.maximum(0.0,
                               tf.reduce_mean(-self.p2 * tf.log(tf.clip_by_value(self.p2, 0.005, 1)))) #/ self.batch_size
        self.pseudo_loss=self.tar1+self.tar2
        # self.loss = - tf.reduce_sum(
        #     self.y * tf.log(self.pred_y + 1e-10) + (1 - self.y) * tf.log(1 - self.pred_y + 1e-10))

        self.loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.pred_y,
                        labels=self.y) )
        self.weight_loss=0.01*tf.losses.get_regularization_loss() #+ self.reg_rate * (
        # tf.nn.l2_loss(self.P) + tf.nn.l2_loss(self.Q) + tf.nn.l2_loss(self.mlp_P) + tf.nn.l2_loss(self.mlp_Q))
        # self.DAloss=tf.maximum(0.0001,KMMD(hot_item_feature,long_item_feature))
        self.DAloss=self.coral_loss(hot_item_feature,long_item_feature)
        # self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)
        # self.total_loss=self.loss+(3)*self.DAloss
        # self.total_loss=self.loss+self.weight_loss+100*self.DAloss+self.pair_loss
        self.total_loss = self.loss + self.weight_loss+100*self.DAloss
        # self.total_loss=self.loss+self.weight_loss+100*self.DAloss+self.pseudo_loss
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.total_loss)

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
        ##########################################
        for u in range(n_users):
            temp=test_matrix_hot.getrow(u).nonzero()
            test_dict_hot[u] = test_matrix_hot.getrow(u).nonzero()[1]
        ##########################################

        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])

        print("data preparation finished.")
        return self
    def prepare_data_my(self, train_data, test_data, test_data_hot, test_data_tail,long_item,hot_item):
        '''
        You must prepare the data before train and test the model
        :param train_data:
        :param test_data:
        :return:
        '''
        # self.source_train={}
        # self.source_label={}
        t = train_data.tocoo()
        self.user = list(t.row.reshape(-1))
        self.item = list(t.col.reshape(-1))
        self.label = list(t.data)
        self.test_data = test_data
###################################################
        # for u in range(6040):
        #     # temp1=t.getrow(u).nonzero()
        #     # temp2=t.getrow(u)
        #     if len(t.getrow(u).nonzero()[1])>0:
        #         self.source_train[u] = t.getrow(u).nonzero()[1]
        #         self.source_label[u] = t.getrow(u).data
        #
        # max=0
        # for k in self.source_train:
        #     if len(self.source_train[k])>max:
        #         max=len(self.source_train[k])

##################################################
##################################################
        self.test_data_hot=test_data_hot
        self.test_data_tail=test_data_tail
##################################################
        self.neg_items = self._get_neg_items(train_data.tocsr())
        self.test_users = set([u for u in self.test_data.keys() if len(self.test_data[u]) > 0])
##################################################
        # self.target_items=self._get_target_items(train_data.tocsr(),long_item)
        # self.hot_list=self._get_hot_items(train_data.tocsr(),hot_item)
        # self.long_list=self._get_long_items(train_data.tocsr(),long_item)
        self.test_users_hot = set([u for u in self.test_data_hot.keys() if len(self.test_data_hot[u]) > 0])
        self.test_users_tail = set([u for u in self.test_data_tail.keys() if len(self.test_data_tail[u]) > 0])
##################################################
        print("data preparation finished.")
        return self

    def train(self):
        self.num_training=len(self.n_qids)
        self.total_batch = int(self.num_training / self.batch_size)
        print(self.total_batch)
        idxs = np.random.permutation(self.num_training)  # shuffled ordering
        nqid_random = list(np.array(self.n_qids)[idxs])


        # train

        for i in range(self.total_batch):
            batch_users = []
            batch_items = []
            batch_labels = []
            batch_targets = []
            batch_query=[]
            ######################################################
            batch_hot_items = []
            batch_long_items = []
            ######################################################
            start_time = time.time()
            batch_nqid = nqid_random[i * self.batch_size:(i + 1) * self.batch_size]
            for j in batch_nqid:
                # print(j)
                batch_users+=[str(self.train_data[j]['user_id'])] * 10
                batch_items+=np.array(self.train_data[j]['source']).astype(str).tolist()
                batch_labels+=np.array(self.train_data[j]['label']).astype(int).tolist()
                batch_targets+=np.array(self.train_data[j]['target']).astype(str).tolist()
                batch_query+=[str(self.train_data[j]['cate_id'])] * 10
                # batch_hot_items +=np.array(list(random.sample(self.hot_item, 10))).astype(str).tolist()
                # batch_long_items +=np.array(list(random.sample(self.long_item, 10))).astype(str).tolist()
                batch_hot_items +=np.array(list(random.sample(self.hot_dic[j], 10))).astype(str).tolist()
                batch_long_items +=np.array(list(random.sample(self.long_dic[j], 10))).astype(str).tolist()
            # _, loss,DAloss,pred_long,pseudo_loss = self.sess.run((self.optimizer, self.loss,self.DAloss, self.pred_long, self.pseudo_loss),
            #                         feed_dict={self.user_id: batch_user, self.item_id: batch_item, self.y: batch_label,self.hot_item_id:batch_hot,self.long_item_id:batch_long})
            _, loss, total_loss,pred, DAloss, pairloss, pseudo_loss = self.sess.run((self.optimizer, self.loss,self.total_loss,self.pred_y,self.DAloss,self.pair_loss,self.pseudo_loss),
                                    feed_dict={self.user_id: batch_users, self.item_id: batch_items, self.y: batch_labels,self.target_item_id:batch_targets, self.hot_item_id:batch_hot_items,self.long_item_id:batch_long_items,self.query_id:batch_query})
            if i % self.display_step == 0:
                if self.verbose:
                    print("Index: %04d; cross= %.9f" % (i + 1, np.mean(total_loss)))
                    print("Index: %04d; weight= %.9f" % (i + 1, np.mean(loss)))
                    print("Index: %04d; DA-coral= %.9f" % (i + 1, np.mean(DAloss)))
                    print("Index: %04d; pair= %.9f" % (i + 1, np.mean(pairloss)))
                    print("Index: %04d; pseudo= %.9f" % (i + 1, np.mean(pseudo_loss)))
                    print("one iteration: %s seconds." % (time.time() - start_time))
                    # print(pred)
                    # print(pseudo_loss)

    def test(self):
        evaluate_cikm_que(self)
        # evaluate(self)

    def execute(self, train_data, test_data):

        self.prepare_data(train_data, test_data)

        init = tf.global_variables_initializer()
        self.sess.run(init)
        # self.test()
        for epoch in range(self.epochs):
            self.train()
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch))
                self.test()

    def execute_my(self, train_data, test_data, n_qids, test_data_hot, test_data_long,hot_item,long_item,hot_dic,long_dic):
        self.epoch = 0
        self.train_data=train_data
        self.test_data=test_data
        self.n_qids=n_qids
        self.test_data_hot=test_data_hot
        self.test_data_long=test_data_long
        self.hot_item=hot_item
        self.long_item=long_item
        self.hot_dic=hot_dic
        self.long_dic=long_dic

        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.test()
        for epoch in range(self.epochs):
            self.epoch=epoch
            self.train()
            if (epoch) % self.T == 0:
                print("Epoch: %04d; " % (epoch))
                self.test()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def predict(self, user_id, item_id, cate_id):
        return self.sess.run([self.pred_y], feed_dict={self.user_id: user_id, self.item_id: item_id,self.query_id:cate_id})[0]

    def _get_neg_items(self, data):
        all_items = set(np.arange(self.num_item))
        neg_items = {}
        for u in range(self.num_user):
            neg_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return neg_items

    def _get_target_items(self, data,long_item):
        # all_items = set(long_item)
        all_items = set(np.arange(self.num_item))
        tag_items = {}
        for u in range(self.num_user):
            tag_items[u] = list(all_items - set(data.getrow(u).nonzero()[1]))

        return tag_items

    def _get_hot_items(self, data,hot_item):
        hot_items = set(hot_item)
        # all_items = set(np.arange(self.num_item))
        tag_items = {}
        for u in range(self.num_user):
            tag_items[u] = list(hot_items)

        return tag_items

    def _get_long_items(self, data,long_item):
        long_items = set(long_item)
        # all_items = set(np.arange(self.num_item))
        tag_items = {}
        for u in range(self.num_user):
            tag_items[u] = list(long_items )

        return tag_items


    def get_item_feature(self,item_id):
        return self.sess.run([self.itemF],feed_dict={self.item_id:item_id})[0]

    def coral_loss(self,h_src, h_trg, gamma=1e-3):
        # regularized covariances (D-Coral is not regularized actually..)
        # First: subtract the mean from the data matrix
        batch_size_s = 2
        batch_size_t = 2
        h_src = h_src - tf.reduce_mean(h_src, axis=0)
        h_trg = h_trg - tf.reduce_mean(h_trg, axis=0)
        cov_source = (1. / (batch_size_s - 1)) * tf.matmul(h_src, h_src,
                                                           transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        cov_target = (1. / (batch_size_t - 1)) * tf.matmul(h_trg, h_trg,
                                                           transpose_a=True)  # + gamma * tf.eye(self.hidden_repr_size)
        # Returns the Frobenius norm (there is an extra 1/4 in D-Coral actually)
        # The reduce_mean account for the factor 1/d^2
        return tf.reduce_mean(tf.square(tf.subtract(cov_source, cov_target)))
