#!/usr/bin/env python
"""
Evaluation Metrics for Top N Recommendation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE


import math


def conputeTSNE(step, source_images, target_images):
    target_features = target_images
    target_labels = np.zeros((len(target_images),))
    source_features = source_images
    source_labels = np.ones((len(source_images),))

    print ('Computing T-SNE.')

    model = TSNE(n_components=2, random_state=0)

    TSNE_hA = model.fit_transform(np.vstack((target_features, source_features)))
    plt.figure(1, facecolor="white")
    plt.cla()
    plt.scatter(TSNE_hA[:, 0], TSNE_hA[:, 1], c=np.hstack((target_labels, source_labels,)), s=10, cmap=mpl.cm.jet)
    plt.savefig('img_p_05/%d.png' % step)
# efficient version
def precision_recall_ndcg_at_k(k, rankedlist, test_matrix):
    idcg_k = 0
    dcg_k = 0
    n_k = k if len(test_matrix) > k else len(test_matrix)
    for i in range(n_k):
        idcg_k += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        dcg_k += 1 / math.log(hits[c][0] + 2, 2)

    return float(count / k), float(count / len(test_matrix)), float(dcg_k / idcg_k)


def map_mrr_ndcg(rankedlist, test_matrix):
    ap = 0
    map = 0
    dcg = 0
    idcg = 0
    mrr = 0
    for i in range(len(test_matrix)):
        idcg += 1 / math.log(i + 2, 2)

    b1 = rankedlist
    b2 = test_matrix
    s2 = set(b2)
    hits = [(idx, val) for idx, val in enumerate(b1) if val in s2]
    count = len(hits)

    for c in range(count):
        ap += (c + 1) / (hits[c][0] + 1)
        dcg += 1 / math.log(hits[c][0] + 2, 2)

    if count != 0:
        mrr = 1 / (hits[0][0] + 1)

    if count != 0:
        map = ap / count

    return map, mrr, float(dcg / idcg)


def evaluate(self):
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_users:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []
        # scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("------------------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))

########################################################
def evaluate_tail(self):
    #################################################################
    hot_feature=self.get_item_feature(self.hot_item)
    # print(hot_feature[0:2])
    long_feature=self.get_item_feature(self.long_item)
    # print('=======')
    # print(long_feature[0:2])
    conputeTSNE(self.epoch,hot_feature,long_feature)
    ##################################################################
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_users:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []
        # scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:20]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(20, pred_ratings_10[u], self.test_data[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("------------------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))

    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_users_hot:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []
        # scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:20]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data_hot[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(20, pred_ratings_10[u], self.test_data_hot[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data_hot[u])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("-------hot----------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))

####################################################
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_users_tail:
        user_ids = []
        user_neg_items = self.neg_items[u]
        item_ids = []
        # scores = []
        for j in user_neg_items:
            item_ids.append(j)
            user_ids.append(u)

        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:20]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data_tail[u])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(20, pred_ratings_10[u], self.test_data_tail[u])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data_tail[u])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("-------tail---------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))


def evaluate_cikm(self):
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_data:
        user_ids = []
        user_neg_items = self.test_data[u]['neg']
        item_ids = []
        # scores = []
        cate_ids=[]
        for j in user_neg_items:
            item_ids.append(str(j))
            user_ids.append(str(self.test_data[u]['user_id']))
            cate_ids.append(str(self.test_data[u]['cate_id']))
        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:12]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(12, pred_ratings_5[u], self.test_data[u]['test'])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u]['test'])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u]['test'])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("------------------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@12:" + str(np.mean(p_at_5)))
    print("recall@12:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@12:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))



    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_data_hot:
        user_ids = []
        user_neg_items = self.test_data_hot[u]['neg']
        item_ids = []
        # scores = []
        cate_ids = []
        for j in user_neg_items:
            item_ids.append(str(j))
            user_ids.append(str(self.test_data[u]['user_id']))
            cate_ids.append(str(self.test_data[u]['cate_id']))
        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:12]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(12, pred_ratings_5[u], self.test_data_hot[u]['test'])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data_hot[u]['test'])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data_hot[u]['test'])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("--------hot-------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@12:" + str(np.mean(p_at_5)))
    print("recall@12:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@12:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))


    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_data_long:
        user_ids = []
        user_neg_items = self.test_data_long[u]['neg']
        item_ids = []
        # scores = []
        cate_ids = []
        for j in user_neg_items:
            item_ids.append(str(j))
            user_ids.append(str(self.test_data[u]['user_id']))
            cate_ids.append(str(self.test_data[u]['cate_id']))
        scores = self.predict(user_ids, item_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:8]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(8, pred_ratings_5[u], self.test_data_long[u]['test'])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data_long[u]['test'])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data_long[u]['test'])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("--------long-------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@12:" + str(np.mean(p_at_5)))
    print("recall@12:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@12:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))

def evaluate_cikm_que(self):
    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_data:
        user_ids = []
        user_neg_items = self.test_data[u]['neg']
        item_ids = []
        # scores = []
        cate_ids=[]
        for j in user_neg_items:
            item_ids.append(str(j))
            user_ids.append(str(self.test_data[u]['user_id']))
            cate_ids.append(str(self.test_data[u]['cate_id']))
        scores = self.predict(user_ids, item_ids, cate_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data[u]['test'])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data[u]['test'])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data[u]['test'])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("------------------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))



    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_data_hot:
        user_ids = []
        user_neg_items = self.test_data_hot[u]['neg']
        item_ids = []
        # scores = []
        cate_ids = []
        for j in user_neg_items:
            item_ids.append(str(j))
            user_ids.append(str(self.test_data[u]['user_id']))
            cate_ids.append(str(self.test_data[u]['cate_id']))
        scores = self.predict(user_ids, item_ids, cate_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data_hot[u]['test'])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data_hot[u]['test'])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data_hot[u]['test'])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("--------hot-------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))


    pred_ratings_10 = {}
    pred_ratings_5 = {}
    pred_ratings = {}
    ranked_list = {}
    p_at_5 = []
    p_at_10 = []
    r_at_5 = []
    r_at_10 = []
    map = []
    mrr = []
    ndcg = []
    ndcg_at_5 = []
    ndcg_at_10 = []
    for u in self.test_data_long:
        user_ids = []
        user_neg_items = self.test_data_long[u]['neg']
        item_ids = []
        # scores = []
        cate_ids = []
        for j in user_neg_items:
            item_ids.append(str(j))
            user_ids.append(str(self.test_data[u]['user_id']))
            cate_ids.append(str(self.test_data[u]['cate_id']))
        scores = self.predict(user_ids, item_ids, cate_ids)
        # print(type(scores))
        # print(scores)
        # print(np.shape(scores))
        # print(ratings)
        neg_item_index = list(zip(item_ids, scores))

        ranked_list[u] = sorted(neg_item_index, key=lambda tup: tup[1], reverse=True)
        pred_ratings[u] = [r[0] for r in ranked_list[u]]
        pred_ratings_5[u] = pred_ratings[u][:5]
        pred_ratings_10[u] = pred_ratings[u][:10]

        p_5, r_5, ndcg_5 = precision_recall_ndcg_at_k(5, pred_ratings_5[u], self.test_data_long[u]['test'])
        p_at_5.append(p_5)
        r_at_5.append(r_5)
        ndcg_at_5.append(ndcg_5)

        p_10, r_10, ndcg_10 = precision_recall_ndcg_at_k(10, pred_ratings_10[u], self.test_data_long[u]['test'])
        p_at_10.append(p_10)
        r_at_10.append(r_10)
        ndcg_at_10.append(ndcg_10)
        map_u, mrr_u, ndcg_u = map_mrr_ndcg(pred_ratings[u], self.test_data_long[u]['test'])
        map.append(map_u)
        mrr.append(mrr_u)
        ndcg.append(ndcg_u)

    print("--------long-------------")
    print("precision@10:" + str(np.mean(p_at_10)))
    print("recall@10:" + str(np.mean(r_at_10)))
    print("precision@5:" + str(np.mean(p_at_5)))
    print("recall@5:" + str(np.mean(r_at_5)))
    print("map:" + str(np.mean(map)))
    print("mrr:" + str(np.mean(mrr)))
    print("ndcg:" + str(np.mean(ndcg)))
    print("ndcg@5:" + str(np.mean(ndcg_at_5)))
    print("ndcg@10:" + str(np.mean(ndcg_at_10)))

