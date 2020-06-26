import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
#df1 names=['queryId','sessionId','userId','timeframe','duration','eventdate','searchstring.tokens','categoryId','items','is.test'],
#df2 queryId;timeframe;itemId
######################################
df_click=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train-clicks.csv',sep=';', engine='python')
df_test_click=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/test-cikm_1.csv',sep=';', engine='python')
click_sum={}
for i in df_click.iterrows():
    if i[1][2] not in click_sum:
        click_sum[i[1][2]]=1
    else:
        click_sum[i[1][2]]+=1
click_list=[]
for key in click_sum:
    click_list.append((click_sum[key], key))

long_tail = sorted(click_list, reverse=True)

hot=[x[1] for x in long_tail[0:2000]]
long_item=[x[1] for x in long_tail[50000:52000]]
test_long_click={}
test_hot_click={}
hot_i={}
long_i={}
for i in df_test_click.iterrows():
    if i[1][2] in hot:
        if i[1][0] not in test_hot_click:
            test_hot_click[i[1][0]] = [i[1][2]]
        else:
            test_hot_click[i[1][0]].append(i[1][2])
    if i[1][2] in long_item:
        if i[1][0] not in test_long_click:
            test_long_click[i[1][0]] = [i[1][2]]
        else:
            test_long_click[i[1][0]].append(i[1][2])

output = open('test_clicks_cikm_hot.csv', 'w')
for key in test_hot_click:
    output.write(str(key))
    output.write(';')
    for j in test_hot_click[key]:
        output.write(str(j))
        output.write(',')
    output.write('\n')
output.close()

output = open('test_clicks_cikm_long.csv', 'w')
for key in test_long_click:
    output.write(str(key))
    output.write(';')
    for j in test_long_click[key]:
        output.write(str(j))
        output.write(',')
    output.write('\n')
output.close()
######################################
df1 = pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train-queries.csv',sep=';', engine='python')
df2=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train_clicks_cikm_1.csv',sep=';',engine='python')
###########################################################
df_test=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/test_clicks_cikm_1.csv',sep=';',engine='python')
df_combine_temp_test=pd.merge(df1,df_test,how='inner',on='queryId')[['queryId','userId','categoryId','items','test_click_items']]
df_clicks_all=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train_clicks_all.csv',sep=';',engine='python')
df_combine_test=pd.merge(df_combine_temp_test,df2,how='inner',on='queryId')[['queryId','userId','categoryId','items','test_click_items','click_items']]
# filter out rows where Rating < 1
df_combine_test = df_combine_test[~(df_combine_test['categoryId']==0)]
df_combine_test=df_combine_test.dropna(axis=0,how='any')

n_qids = df1.queryId.unique()

test_cikm={}
for i in df_combine_test.iterrows():
    test_cikm[i[1][0]]={}
    test_cikm[i[1][0]]['user_id']=int(i[1][1])
    test_cikm[i[1][0]]['cate_id']=i[1][2]
    test=i[1][4].split(',')[0:-1]
    test_cikm[i[1][0]]['test'] = test
    test_cikm[i[1][0]]['neg']=list(set(i[1][3].split(',')[0:-1])-set(i[1][5].split(',')[0:-1]))

# test_clicks={}
# # m=0
# for i in df_test.iterrows():
#     if i[1][0] not in test_clicks:
#         test_clicks[i[1][0]]=[i[1][2]]
#     else:
#         test_clicks[i[1][0]].append(i[1][2])
#
#
# output = open('test_clicks_cikm_1.csv', 'w')
# for key in test_clicks:
#     output.write(str(key))
#     output.write(';')
#     for j in test_clicks[key]:
#         output.write(str(j))
#         output.write(',')
#     output.write('\n')
# output.close()
###########################################################
# train_data, test_data = train_test_split(df2, test_size=0.2)
# train_data = pd.DataFrame(train_data)
# test_data = pd.DataFrame(test_data)

# train_data[['queryId','timeframe','itemId']].to_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train-cikm_1.csv',sep=';',index=False)
# test_data[['queryId','timeframe','itemId']].to_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/test-cikm_1.csv',sep=';',index=False)
#
# dfcombine=df.groupby(df['queryId'])
# clicks={}
# # m=0
# for i in df2.iterrows():
#     if i[1][0] not in clicks:
#         clicks[i[1][0]]=[i[1][2]]
#     else:
#         clicks[i[1][0]].append(i[1][2])
#
#
# output = open('train_clicks_all.csv', 'w')
# for key in clicks:
#     output.write(str(key))
#     output.write(';')
#     for j in clicks[key]:
#         output.write(str(j))
#         output.write(',')
#     output.write('\n')
# output.close()
df_combine_temp=pd.merge(df1,df2,how='inner',on='queryId')[['queryId','userId','categoryId','items','click_items']]
df_clicks_all=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train_clicks_all.csv',sep=';',engine='python')
df_combine=pd.merge(df_combine_temp,df_clicks_all,how='inner',on='queryId')[['queryId','userId','categoryId','items','click_items','click_items_all']]
# filter out rows where Rating < 1
df_combine = df_combine[~(df_combine['categoryId']==0)]
df_combine=df_combine.dropna(axis=0,how='any')

# print(df_combine)
# n_users = df_combine.userId.unique().shape[0]
# print(n_users)
cate_item={}
all_items=[]
df3=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/product-categories.csv',sep=';',engine='python')
for i in df3.iterrows():
    all_items.append(i[1][0])
    if i[1][1] not in cate_item:
        cate_item[i[1][1]]=[i[1][0]]
    else:
        cate_item[i[1][1]].append(i[1][0])

for k in cate_item:
    if len(cate_item[k])<10:
        cate_item[k]+=random.sample(all_items,10)
#
#
# output = open('train_cate_item.csv', 'w')
# for key in cate_item:
#     output.write(str(key))
#     output.write(';')
#     for j in cate_item[key]:
#         output.write(str(j))
#         output.write(',')
#     output.write('\n')
# output.close()

df4=pd.read_csv('/ext/czh-190/DeepRec-master/data/dataset-train-diginetica/train_cate_item.csv',sep=';',engine='python')
df_combine_1=pd.merge(df_combine,df4,how='left',on='categoryId')
print(df_combine_1)
m=0
n=0
# output = open('train_cikm_1.csv', 'w')
# output.write('queryId;userId;categoryId;source;label;target')
# output.write('\n')

# for i in df_combine_1.iterrows():
#     n+=1
#     output.write(str(i[1][0]))
#     output.write(';')
#     output.write(str(int(i[1][1])))
#     output.write(';')
#     output.write(str(i[1][2]))
#     output.write(';')
#     train=i[1][4].split(',')[0:-1]
#
#     if len(train)>=10:
#         m+=1
#         print('filter'+str(len(train)))
#         continue
#     label=list(np.ones((len(train),)))
#     label+=list(np.zeros(10-len(train)))
#     neg=[]
#     temp=list(set(i[1][3].split(',')[0:-1]) - set(i[1][4].split(',')[0:-1]))
#     # print(len(temp))
#     if len(temp)<10-len(train):
#         neg+=random.sample(cate_item[i[1][2]],10-len(train)-len(temp))
#         neg+=list(temp)
#     else:
#         # print("=====")
#         # print(len(temp))
#         # print(len(train))
#         neg+=random.sample(temp, 10-len(train))
#     train+=neg
#     if len(train)!=10:
#         print("error")
#         print(i[1][0])
#         break
#     for j in train:
#         output.write(str(j))
#         output.write(',')
#     output.write(';')
#     for j in label:
#         output.write(str(int(j)))
#         output.write(',')
#     output.write(';')
#     target=random.sample(cate_item[i[1][2]],10)
#     for j in target:
#         output.write(str(j))
#         output.write(',')
#
#
#     output.write('\n')
# output.close()
n_qids = df_combine_1.queryId.unique()
train_cikm={}
for i in df_combine_1.iterrows():
    n+=1
    train_cikm[i[1][0]]={}
    train_cikm[i[1][0]]['user_id']=int(i[1][1])
    train_cikm[i[1][0]]['cate_id']=i[1][2]
    train=i[1][4].split(',')[0:-1]
    pos=i[1][5].split(',')[0:-1]
    if len(train)>=10:
        m+=1
        print('filter'+str(len(pos)))
        continue
    label=list(np.ones((len(train),)))
    label+=list(np.zeros(10-len(train)))
    train_cikm[i[1][0]]['label']=label
    neg=[]
    temp=list(set(i[1][3].split(',')[0:-1]) - set(i[1][5].split(',')[0:-1]))
    # print(len(temp))
    if len(temp)<10-len(train):
        neg+=random.sample(list(set(cate_item[i[1][2]])-set(i[1][3].split(',')[0:-1])),10-len(train)-len(temp))
        neg+=list(temp)
    else:
        # print("=====")
        # print(len(temp))
        # print(len(train))
        neg+=random.sample(temp, 10-len(train))
    train+=neg
    if len(train)!=10:
        print("error")
        print(i[1][0])
        break
    train_cikm[i[1][0]]['source']=train
    target=random.sample(cate_item[i[1][2]],10)
    train_cikm[i[1][0]]['target']=target


print('success')
print(m)
print(n)

# df = pd.read_csv(path, sep=sep, names=header, engine='python')
#
# n_users = 6040
# n_items = 3952
# train_row_all = []
# train_col_all = []
# train_rating_all = []
# ######################################################
# for line in df.itertuples():
#     u = line[1] - 1
#     # i_list=line[2].split('|')[0]
#     i = int(line[2].split('|')[0]) - 1
#     train_row_all.append(u)
#     train_col_all.append(i)
#     # print(max(train_row))
#     # print(max(train_col))
#     train_rating_all.append(1)
# train_matrix = csr_matrix((train_rating_all, (train_row_all, train_col_all)), shape=(n_users, n_items))
# train = train_matrix.A
# sum_m = np.sum(train, axis=0)
# cold = np.where(sum_m == 0)
# s = []
# for i, k in enumerate(sum_m):
#     s.append((k, i))
# long_tail = sorted(s, reverse=True)
# hot = [x[1] for x in long_tail[0:200]]
# long_item = [x[1] for x in long_tail[200:400]]
# # hot=[x[1] for x in long_tail[0:500]]
# # long_item=[x[1] for x in long_tail[500:1000]]
# ######################################################
# train_data, test_data = train_test_split(df, test_size=test_size)
# train_data = pd.DataFrame(train_data)
# test_data = pd.DataFrame(test_data)
#
# train_row = []
# train_col = []
# train_rating = []
#
# for line in train_data.itertuples():
#     u = line[1] - 1
#     # i_list=line[2].split('|')[0]
#     i = int(line[2].split('|')[0]) - 1
#     train_row.append(u)
#     train_col.append(i)
#     train_rating.append(1)
# train_matrix = csr_matrix((train_rating, (train_row, train_col)), shape=(n_users, n_items))
#
# # all_items = set(np.arange(n_items))
# # neg_items = {}
# # for u in range(n_users):
# #     neg_items[u] = list(all_items - set(train_matrix.getrow(u).nonzero()[1]))
#
# test_row = []
# test_col = []
# test_rating = []
# for line in test_data.itertuples():
#     test_row.append(line[1] - 1)
#     test_col.append(int(line[2].split('|')[0]) - 1)
#     test_rating.append(1)
# test_matrix = csr_matrix((test_rating, (test_row, test_col)), shape=(n_users, n_items))
#
# test_dict = {}
# for u in range(n_users):
#     test_dict[u] = test_matrix.getrow(u).nonzero()[1]
# ############################################################################################
# test_row_hot = []
# test_col_hot = []
# test_rating_hot = []
# for line in test_data.itertuples():
#     if (int(line[2].split('|')[0]) - 1) in hot:
#         test_row_hot.append(line[1] - 1)
#         test_col_hot.append(int(line[2].split('|')[0]) - 1)
#         test_rating_hot.append(1)
# test_matrix_hot = csr_matrix((test_rating_hot, (test_row_hot, test_col_hot)), shape=(n_users, n_items))
#
# test_dict_hot = {}
# for u in range(n_users):
#     test_dict_hot[u] = test_matrix_hot.getrow(u).nonzero()[1]
# ############################################################################################
#
# ############################################################################################
# test_row_long = []
# test_col_long = []
# test_rating_long = []
# for line in test_data.itertuples():
#     if (int(line[2].split('|')[0]) - 1) in long_item:
#         test_row_long.append(line[1] - 1)
#         test_col_long.append(int(line[2].split('|')[0]) - 1)
#         test_rating_long.append(1)
# test_matrix_long = csr_matrix((test_rating_long, (test_row_long, test_col_long)), shape=(n_users, n_items))
#
# test_dict_long = {}
# for u in range(n_users):
#     test_dict_long[u] = test_matrix_long.getrow(u).nonzero()[1]
# ############################################################################################
#
# print("Load data finished. Number of users:", n_users, "Number of items:", n_items)
# return train_matrix.todok(), test_dict, n_users, n_items, test_dict_hot, test_dict_long, hot, long_item
#
#
#
#
