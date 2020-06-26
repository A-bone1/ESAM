import argparse
import tensorflow as tf
import sys
import os
import os.path
import random
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
n_users=6040
n_items=3952
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


path_rat="../ml-1m/ratings.dat"
path_user="../ml-1m/users.dat"
path_movies="../ml-1m/movies.dat"
df_rating = pd.read_csv(path_rat, sep="::", header=None,names=['user_id', 'item_id', 'rating', 'time'], engine='python')
df_user=pd.read_csv(path_user, sep="::", header=None,names=['user_id','Gender','Age','Occupation','Zip-code'], engine='python')
df_movie=pd.read_csv(path_movies, sep="::", header=None,names=['item_id','Title','Genres'], engine='python')
# print(df_rating.loc[0])
# print('=======')
# print(df_user.loc[0])
# print('=======')
# print(df_movie.loc[0])

# user_gender={}
# for i in df_user.iterrows():
#     if i[1][0] not in user_gender:
#         if i[1][1]=='F':
#             user_gender[i[1][0]]=0
#         else:
#             user_gender[i[1][0]]=1

# *1:  "Under 18"
# *18:  "18-24"
# *25:  "25-34"
# *35:  "35-44"
# *45:  "45-49"
# *50:  "50-55"
# *56:  "56+"
# user_age={}
# for i in df_user.iterrows():
#     if i[1][0] not in user_age:
#         if i[1][2]==1:
#             user_age[i[1][0]]=0
#         elif i[1][2] == 18:
#             user_age[i[1][0]] = 1
#         elif i[1][2] == 25:
#             user_age[i[1][0]] = 2
#         elif i[1][2] == 35:
#             user_age[i[1][0]] = 3
#         elif i[1][2] == 45:
#             user_age[i[1][0]] = 4
#         elif i[1][2] == 50:
#             user_age[i[1][0]] = 5
#         elif i[1][2] == 56:
#             user_age[i[1][0]] = 6
        # else:
        #     user_age[i[1][0]] = i[1][2]

# - Genres are pipe-separated and are selected from the following genres:
#
# 	* Action
# 	* Adventure
# 	* Animation
# 	* Children's
# 	* Comedy
# 	* Crime
# 	* Documentary
# 	* Drama
# 	* Fantasy
# 	* Film-Noir
# 	* Horror
# 	* Musical
# 	* Mystery
# 	* Romance
# 	* Sci-Fi
# 	* Thriller
# 	* War
# 	# * Western
movie_style={}
for i in df_movie.iterrows():
    if i[1][0] not in movie_style:
        # print(i[1][2].split('|'))
        m=[]
        for j in i[1][2].split('|'):
            if j=='Action':
                m.append(0)
            if j=='Adventure':
                m.append(1)
            if j=='Animation':
                m.append(2)
            if j=='Children\'s':
                m.append(3)
            if j=='Comedy':
                m.append(4)
            if j=='Crime':
                m.append(5)
            if j=='Documentary':
                m.append(6)
            if j=='Film-Noir':
                m.append(7)
            if j=='Fantasy':
                m.append(8)
            if j=='Horror':
                m.append(9)
            if j=='Musical':
                m.append(10)
            if j=='Mystery':
                m.append(11)
            if j=='Romance':
                m.append(12)
            if j=='Sci-Fi':
                m.append(13)
            if j=='Thriller':
                m.append(14)
            if j=='War':
                m.append(15)
            if j=='Western':
                m.append(16)
            if j=='Drama':
                m.append(17)
        movie_style[i[1][0]]=m

movie_group={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[],10:[],11:[],12:[],13:[],14:[],15:[],16:[],17:[]}
for i in df_movie.iterrows():

    for j in i[1][2].split('|'):
        if j=='Action':
            movie_group[0].append(i[1][0])
        if j=='Adventure':
            movie_group[1].append(i[1][0])
        if j=='Animation':
            movie_group[2].append(i[1][0])
        if j=='Children\'s':
            movie_group[3].append(i[1][0])
        if j=='Comedy':
            movie_group[4].append(i[1][0])
        if j=='Crime':
            movie_group[5].append(i[1][0])
        if j=='Documentary':
            movie_group[6].append(i[1][0])
        if j=='Film-Noir':
            movie_group[7].append(i[1][0])
        if j=='Fantasy':
            movie_group[8].append(i[1][0])
        if j=='Horror':
            movie_group[9].append(i[1][0])
        if j=='Musical':
            movie_group[10].append(i[1][0])
        if j=='Mystery':
            movie_group[11].append(i[1][0])
        if j=='Romance':
            movie_group[12].append(i[1][0])
        if j=='Sci-Fi':
            movie_group[13].append(i[1][0])
        if j=='Thriller':
            movie_group[14].append(i[1][0])
        if j=='War':
            movie_group[15].append(i[1][0])
        if j=='Western':
            movie_group[16].append(i[1][0])
        if j=='Drama':
            movie_group[17].append(i[1][0])


# mclicks={}
# labels={}
# j=0
# for i in df_rating.iterrows():
#     j=j+1
#     if i[1][0] not in clicks:
#         m=[]
#         n=[]
#         m.append(i[1][1])
#         n.append(1)
#         clicks[i[1][0]]=m
#         labels[i[1][0]]=n
#     else:
#         clicks[i[1][0]].append(i[1][1])
#         labels[i[1][0]].append(1)
#     if j==100:
#         break
## generate long-tail
train_row = []
train_col = []
train_rating = []

for line in df_rating.itertuples():
    u = line[1] - 1
    i = line[2] - 1
    train_row.append(u)
    train_col.append(i)
    # print(max(train_row))
    # print(max(train_col))
    train_rating.append(1)
train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
train=train_matrix.A
sum_m=np.sum(train,axis=0)
cold=np.where(sum_m==0)
s=[]
for i,k in enumerate(sum_m):
    s.append((k,i))
long_tail=sorted(s,reverse=True)
hot = [x[1] for x in long_tail[0:800]]
long_item = [x[1] for x in long_tail[800:]]
# # x_id=[]
# # y=[]
# # for i in long_tail:
# #     x_id.append(i[1])
# #     y.append(i[0])
# # x=list(np.arange(len(sum)))
# # plt.plot(x,y)
# # plt.show()
#
# #generate source train
#
neg_items={}
all_items = set(np.arange(n_items))-set(cold[0])
source_train={}

j=0
for i in df_rating.iterrows():
    print(i)
    item_append = []
    item_append.append(i[1][1])
    neg_items[i[1][0]-1] = list(all_items - set(train_matrix.getrow(i[1][0]-1).nonzero()[1]))
    list_of_random_items=random.sample(neg_items[i[1][0]-1], 2)
    item_append += list_of_random_items
    source_train[j]=(i[1][0],item_append)
    j=j+1
#     if j==100:
# # #
#        break
output = open('source_train.txt', 'w')
for i in source_train:
    output.write(str(source_train[i][0]))
    output.write(';;')
    for j in source_train[i][1]:
        output.write(str(j))
        output.write('|')
    output.write('\n')
output.close()
# target_train={}
# k=0
# for i in df_rating.iterrows():
#     a=[]
#     for j in movie_style[i[1][1]]:
#         a+=movie_group[j]
#     # print(i[1][0]-1)
#     # print('========')
#     # print(k)
#     s=source_train[k]
#     # a=a-set(s)
#     target_train[k]=(i[1][0],list(random.sample(a, 6)))
#     k=k+1
#     if k==100:
#         break
# output = open('target_train.txt', 'w')
# for i in target_train:
#     output.write(str(target_train[i][0]))
#     output.write(';;')
#     for j in target_train[i][1]:
#         output.write(str(j))
#         output.write('|')
#     output.write('\n')
# output.close()

print('sucess')