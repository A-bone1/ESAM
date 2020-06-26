import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


def load_data_all(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'time'],
                  test_size=0.2, sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    train_dict = {}
    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_dict[(u, i)] = 1

    for u in range(n_users):
        for i in range(n_items):
            train_row.append(u)
            train_col.append(i)
            if (u, i) in train_dict.keys():
                train_rating.append(1)
            else:
                train_rating.append(0)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
    all_items = set(np.arange(n_items))

    neg_items = {}
    train_interaction_matrix = []
    for u in range(n_users):
        neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

    return train_interaction_matrix, test_dict, n_users, n_items


def load_data_neg(path="../data/ml100k/movielens_100k.dat", header=['user_id', 'item_id', 'rating', 'category'],
                  test_size=0.2, sep="\t"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = df.user_id.unique().shape[0]
    n_items = df.item_id.unique().shape[0]

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    # all_items = set(np.arange(n_items))
    # neg_items = {}
    # for u in range(n_users):
    #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items


def load_data_separately(path_train=None, path_test=None, path_val=None, header=['user_id', 'item_id', 'rating'],
                         sep=" ", n_users=0, n_items=0):
    n_users = n_users
    n_items = n_items
    print("start")
    train_matrix = None
    if path_train is not None:
        train_data = pd.read_csv(path_train, sep=sep, names=header, engine='python')
        print("Load data finished. Number of users:", n_users, "Number of items:", n_items)

        train_row = []
        train_col = []
        train_rating = []

        for line in train_data.itertuples():
            u = line[1]  # - 1
            i = line[2]  # - 1
            train_row.append(u)
            train_col.append(i)
            train_rating.append(1)

        train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    test_dict = None
    if path_test is not None:
        test_data = pd.read_csv(path_test, sep=sep, names=header, engine='python')
        test_row = []
        test_col = []
        test_rating = []
        for line in test_data.itertuples():
            test_row.append(line[1])
            i = line[2]  # - 1
            test_col.append(i)
            test_rating.append(1)

        test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

        test_dict = {}
        for u in range(n_users):
            test_dict[u] = test_matrix.getrow(u).nonzero()[1]
    all_items = set(np.arange(n_items))
    train_interaction_matrix = []
    for u in range(n_users):
        train_interaction_matrix.append(list(train_matrix.getrow(u).toarray()[0]))

    if path_val is not None:
        val_data = pd.read_csv(path_val, sep=sep, names=header, engine='python')

    print("end")
    return train_interaction_matrix, test_dict, n_users, n_items


def load_data_myneg(path="../test/source_train.txt", header=['user_id', 'item_list'],
                  test_size=0.2, sep=";;"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = 6040
    n_items = 3952

    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        # i_list=line[2].split('|')[0]
        i = int(line[2].split('|')[0]) - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    # all_items = set(np.arange(n_items))
    # neg_items = {}
    # for u in range(n_users):
    #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(int(line[2].split('|')[0]) - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items


def load_data_myneg_tail(path="../test/source_train.txt", header=['user_id', 'item_list'],
                  test_size=0.2, sep=";;"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = 6040
    n_items = 3952
    train_row_all=[]
    train_col_all=[]
    train_rating_all=[]
######################################################
    for line in df.itertuples():
        u = line[1] - 1
        # i_list=line[2].split('|')[0]
        i = int(line[2].split('|')[0]) - 1
        train_row_all.append(u)
        train_col_all.append(i)
        # print(max(train_row))
        # print(max(train_col))
        train_rating_all.append(1)
    train_matrix = csr_matrix((train_rating_all, (train_row_all, train_col_all)), shape=(n_users, n_items))
    train = train_matrix.A
    sum_m = np.sum(train, axis=0)
    cold = np.where(sum_m == 0)
    s = []
    for i, k in enumerate(sum_m):
        s.append((k, i))
    long_tail = sorted(s, reverse=True)
    hot=[x[1] for x in long_tail[0:200]]
    long_item=[x[1] for x in long_tail[200:400]]
    # hot=[x[1] for x in long_tail[0:500]]
    # long_item=[x[1] for x in long_tail[500:1000]]
######################################################
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        # i_list=line[2].split('|')[0]
        i = int(line[2].split('|')[0]) - 1
        train_row.append(u)
        train_col.append(i)
        train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    # all_items = set(np.arange(n_items))
    # neg_items = {}
    # for u in range(n_users):
    #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        test_row.append(line[1] - 1)
        test_col.append(int(line[2].split('|')[0]) - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]
############################################################################################
    test_row_hot = []
    test_col_hot = []
    test_rating_hot = []
    for line in test_data.itertuples():
        if (int(line[2].split('|')[0])-1) in hot:
            test_row_hot.append(line[1] - 1)
            test_col_hot.append(int(line[2].split('|')[0]) - 1)
            test_rating_hot.append(1)
    test_matrix_hot = csr_matrix((test_rating_hot, (test_row_hot, test_col_hot)), shape=(n_users, n_items))

    test_dict_hot = {}
    for u in range(n_users):
        test_dict_hot[u] = test_matrix_hot.getrow(u).nonzero()[1]
############################################################################################

############################################################################################
    test_row_long = []
    test_col_long = []
    test_rating_long = []
    for line in test_data.itertuples():
        if (int(line[2].split('|')[0])-1) in long_item:
            test_row_long.append(line[1] - 1)
            test_col_long.append(int(line[2].split('|')[0]) - 1)
            test_rating_long.append(1)
    test_matrix_long = csr_matrix((test_rating_long, (test_row_long, test_col_long)), shape=(n_users, n_items))

    test_dict_long = {}
    for u in range(n_users):
        test_dict_long[u] = test_matrix_long.getrow(u).nonzero()[1]
############################################################################################

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items, test_dict_hot, test_dict_long, hot, long_item


def load_data_myneg_tail_1(path="../test/source_train.txt", header=['user_id', 'item_list'],
                  test_size=0.2, sep=";;"):
    df = pd.read_csv(path, sep=sep, names=header, engine='python')

    n_users = 6040
    n_items = 3952
    train_row_all=[]
    train_col_all=[]
    train_rating_all=[]
######################################################
    for line in df.itertuples():
        u = line[1] - 1
        # i_list=line[2].split('|')[0]
        i = int(line[2].split('|')[0]) - 1
        train_row_all.append(u)
        train_col_all.append(i)

        # print(max(train_row))
        # print(max(train_col))
        train_rating_all.append(1)
    train_matrix = csr_matrix((train_rating_all, (train_row_all, train_col_all)), shape=(n_users, n_items))
    train = train_matrix.A
    sum_m = np.sum(train, axis=0)
    cold = np.where(sum_m == 0)
    s = []
    for i, k in enumerate(sum_m):
        s.append((k, i))
    long_tail = sorted(s, reverse=True)
    hot=[x[1] for x in long_tail[0:200]]
    long_item=[x[1] for x in long_tail[200:400]]

######################################################
    df = pd.read_csv("../ml-1m/ratings.dat", sep='::', names=['user_id', 'item_id','rating','time'], engine='python')
    train_data, test_data = train_test_split(df, test_size=test_size)
    train_data = pd.DataFrame(train_data)
    test_data = pd.DataFrame(test_data)

    train_row = []
    train_col = []
    train_rating = []

    for line in train_data.itertuples():
        u = line[1] - 1
        # i_list=line[2].split('|')[0]
        i = line[2] - 1
        train_row.append(u)
        train_col.append(i)
        if line[3]<4:
            train_rating.append(-1)
        else:
            train_rating.append(1)
    train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))

    # all_items = set(np.arange(n_items))
    # neg_items = {}
    # for u in range(n_users):
    #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))

    test_row = []
    test_col = []
    test_rating = []
    for line in test_data.itertuples():
        if line[3]<4:
            continue
        test_row.append(line[1] - 1)
        test_col.append(line[2] - 1)
        test_rating.append(1)
    test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))

    test_dict = {}
    for u in range(n_users):
        test_dict[u] = test_matrix.getrow(u).nonzero()[1]
############################################################################################
    test_row_hot = []
    test_col_hot = []
    test_rating_hot = []
    for line in test_data.itertuples():
        if (line[2]-1) in hot and line[3]>=4:
            test_row_hot.append(line[1] - 1)
            test_col_hot.append(line[2] - 1)
            test_rating_hot.append(1)
    test_matrix_hot = csr_matrix((test_rating_hot, (test_row_hot, test_col_hot)), shape=(n_users, n_items))

    test_dict_hot = {}
    for u in range(n_users):
        test_dict_hot[u] = test_matrix_hot.getrow(u).nonzero()[1]
############################################################################################

############################################################################################
    test_row_long = []
    test_col_long = []
    test_rating_long = []
    for line in test_data.itertuples():
        if (line[2]-1) in long_item and line[3]>=4:
            test_row_long.append(line[1] - 1)
            test_col_long.append(line[2] - 1)
            test_rating_long.append(1)
    test_matrix_long = csr_matrix((test_rating_long, (test_row_long, test_col_long)), shape=(n_users, n_items))

    test_dict_long = {}
    for u in range(n_users):
        test_dict_long[u] = test_matrix_long.getrow(u).nonzero()[1]
############################################################################################

    print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
    return train_matrix.todok(), test_dict, n_users, n_items, test_dict_hot, test_dict_long, hot, long_item



def load_data_myneg_cikm(path="../test/source_train.txt", header=['user_id', 'item_list'],
                  test_size=0.2, sep=";;"):

    df1 = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train-queries.csv', sep=';',
                      engine='python')
    df2 = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train_clicks_cikm_1.csv', sep=';',
                      engine='python')
    ###########################################################
    df_test = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/test_clicks_cikm_1.csv', sep=';',
                          engine='python')
    df_combine_temp_test = pd.merge(df1, df_test, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'test_click_items']]
    df_clicks_all = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train_clicks_all.csv',
                                sep=';', engine='python')
    df_combine_test = pd.merge(df_combine_temp_test, df2, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'test_click_items', 'click_items']]
    # filter out rows where Rating < 1
    df_combine_test = df_combine_test[~(df_combine_test['categoryId'] == 0)]
    df_combine_test = df_combine_test.dropna(axis=0, how='any')
    test_cikm = {}
    for i in df_combine_test.iterrows():
        test_cikm[i[1][0]] = {}
        test_cikm[i[1][0]]['user_id'] = int(i[1][1])
        test_cikm[i[1][0]]['cate_id'] = i[1][2]
        test = i[1][4].split(',')[0:-1]
        test_cikm[i[1][0]]['test'] = test
        test_cikm[i[1][0]]['neg'] = list(set(i[1][3].split(',')[0:-1]) - set(i[1][5].split(',')[0:-1]))
################################################################################################33
    df_test = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/test_clicks_cikm_hot.csv', sep=';',
                          engine='python')
    df_combine_temp_test = pd.merge(df1, df_test, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'test_click_items']]

    df_combine_test = pd.merge(df_combine_temp_test, df2, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'test_click_items', 'click_items']]
    # filter out rows where Rating < 1
    df_combine_test = df_combine_test[~(df_combine_test['categoryId'] == 0)]
    df_combine_test = df_combine_test.dropna(axis=0, how='any')
    test_cikm_hot = {}
    for i in df_combine_test.iterrows():
        test_cikm_hot[i[1][0]] = {}
        test_cikm_hot[i[1][0]]['user_id'] = int(i[1][1])
        test_cikm_hot[i[1][0]]['cate_id'] = i[1][2]
        test = i[1][4].split(',')[0:-1]
        test_cikm_hot[i[1][0]]['test'] = test
        test_cikm_hot[i[1][0]]['neg'] = list(set(i[1][3].split(',')[0:-1]) - set(i[1][5].split(',')[0:-1]))


    df_test = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/test_clicks_cikm_long.csv', sep=';',
                          engine='python')
    df_combine_temp_test = pd.merge(df1, df_test, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'test_click_items']]

    df_combine_test = pd.merge(df_combine_temp_test, df2, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'test_click_items', 'click_items']]
    # filter out rows where Rating < 1
    df_combine_test = df_combine_test[~(df_combine_test['categoryId'] == 0)]
    df_combine_test = df_combine_test.dropna(axis=0, how='any')
    test_cikm_long = {}
    for i in df_combine_test.iterrows():
        test_cikm_long[i[1][0]] = {}
        test_cikm_long[i[1][0]]['user_id'] = int(i[1][1])
        test_cikm_long[i[1][0]]['cate_id'] = i[1][2]
        test = i[1][4].split(',')[0:-1]
        test_cikm_long[i[1][0]]['test'] = test
        test_cikm_long[i[1][0]]['neg'] = list(set(i[1][3].split(',')[0:-1]) - set(i[1][5].split(',')[0:-1]))

    #################################################################333####3#########################

    df_combine_temp = pd.merge(df1, df2, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'click_items']]

    df_combine = pd.merge(df_combine_temp, df_clicks_all, how='inner', on='queryId')[
        ['queryId', 'userId', 'categoryId', 'items', 'click_items', 'click_items_all']]

    df_combine = df_combine[~(df_combine['categoryId'] == 0)]
    df_combine = df_combine.dropna(axis=0, how='any')

    cate_item = {}
    all_items = []
    df3 = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/product-categories.csv', sep=';',
                      engine='python')
    for i in df3.iterrows():
        all_items.append(i[1][0])
        if i[1][1] not in cate_item:
            cate_item[i[1][1]] = [i[1][0]]
        else:
            cate_item[i[1][1]].append(i[1][0])

    for k in cate_item:
        if len(cate_item[k]) < 10:
            cate_item[k] += random.sample(all_items, 10)

    df4 = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train_cate_item.csv', sep=';',
                      engine='python')
    df_combine_1 = pd.merge(df_combine, df4, how='left', on='categoryId')
    print(df_combine_1)
    m = 0
    n = 0

    train_cikm = {}
    for i in df_combine_1.iterrows():
        n += 1
        train = i[1][4].split(',')[0:-1]
        pos = i[1][5].split(',')[0:-1]
        if len(train) >= 10:
            m += 1
            print('filter' + str(len(train)))
            continue
        train_cikm[i[1][0]] = {}
        train_cikm[i[1][0]]['user_id'] = int(i[1][1])
        train_cikm[i[1][0]]['cate_id'] = i[1][2]


        label = list(np.ones((len(train),)))
        label += list(np.zeros(10 - len(train)))
        train_cikm[i[1][0]]['label'] = label
        neg = []
        temp = list(set(i[1][3].split(',')[0:-1]) - set(i[1][5].split(',')[0:-1]))
        # print(len(temp))
        if len(temp) < 10 - len(train):
            neg += random.sample(list(set(cate_item[i[1][2]]) - set(i[1][3].split(',')[0:-1])),
                                 10 - len(train) - len(temp))
            neg += list(temp)
        else:
            # print("=====")
            # print(len(temp))
            # print(len(train))
            neg += random.sample(temp, 10 - len(train))
        train += neg
        if len(train) != 10:
            print("error")
            print(i[1][0])
            break
        train_cikm[i[1][0]]['source'] = train
        target = random.sample(cate_item[i[1][2]], 10)
        train_cikm[i[1][0]]['target'] = target
    n_qids = list(train_cikm.keys())
    n_uids = df_combine_1.userId.unique()
    print('success'+str(len(n_uids)))
    print(m)
    print(n)
    ###########################load hot and item
    df_click = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train-clicks.csv', sep=';',
                           engine='python')
    click_sum = {}
    for i in df_click.iterrows():
        if i[1][2] not in click_sum:
            click_sum[i[1][2]] = 1
        else:
            click_sum[i[1][2]] += 1
    click_list = []
    for key in click_sum:
        click_list.append((click_sum[key], key))

    long_tail = sorted(click_list, reverse=True)

    hot = [x[1] for x in long_tail[0:2000]]
    long_item = [x[1] for x in long_tail[50000:52000]]

    # hot = [x[1] for x in long_tail[0:10000]]
    # long_item = [x[1] for x in long_tail[10000:]]

    hot_dic={}
    long_dic={}
    for i in n_qids:
        hot_dic[i]=hot
    for i in n_qids:
        long_dic[i]=long_item


    return train_cikm,test_cikm,n_qids, test_cikm_hot, test_cikm_long,hot,long_item,hot_dic,long_dic
