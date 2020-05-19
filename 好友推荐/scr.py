#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

#建立透视表
df = pd.read_table('./readtime.txt', sep='\t')
df = pd.pivot_table(df, index='user_id', columns=['tag_id'], values=['readtime'],
                    aggfunc=np.mean)
readtimes = df.values
n_user, n_tag = df.shape


#皮尔斯相似度计算
def Pearson(x, y):
    mean_X, mean_Y = x.mean(), y.mean()
    Sxy = 0
    Sxx = 0
    Syy = 0
    for xi, yi in zip(x, y):
        Sxy += (xi - mean_X) * (yi - mean_Y)
        Sxx += (xi - mean_X) ** 2
        Syy += (yi - mean_Y) ** 2

    r = Sxy / np.sqrt(Sxx * Syy)
    return r


#推荐
def recommend(user_id, user_readtime, all_readtimes):
    user_id = user_id - 1
    max_r = -1
    for i in range(len(readtimes)):
        if i != user_id:
            r = Pearson(user_readtime, all_readtimes[i])
            if r > max_r:
                max_r = r
                recommend_user_id = i + 1
    return recommend_user_id, max_r

#显示推荐用户喜欢的tag
def show(recommend_user_id):
    id=recommend_user_id-1
    readtimeslist1 = np.round(readtimes[id].tolist(), 3)
    # print(readtimeslist1)
    readtimeslist2 = np.round(sorted(readtimes[id].tolist(), reverse=True), 3)
    # print(readtimeslist2)
    top = readtimeslist2[0]
    # print(top)
    for i in range(len(readtimeslist1)):
        if (top == readtimeslist1[i]):
            return i+1


#将tag加入到列表中
data = []
f = open("./tag.txt", "r", encoding="utf-8")
lines=f.readlines()
for line in lines:
    line_list=line.split('\t')
    id=line_list[0]
    name=line_list[1].replace('\n','')
    data.append(name)
f.close()

recommend_user_ids = []
matchings = []
for k, user_readtime in enumerate(readtimes):
    user_id = k + 1
    recommend_user_id, max_r = recommend(user_id, user_readtime, readtimes)
    recommend_user_ids.append(recommend_user_id)
    matchings.append(max_r)
    tagid=show(recommend_user_id)
    tagname=data[tagid]
    print('user_id:', user_id, '  recommend_user_id:', recommend_user_id)
    print('该用户喜欢读',tagname,'类的书')

# cols = ['tag_{}_readtime'.format(i) for i in range(1, n_tag + 1)]
# df = pd.DataFrame(data=readtimes, columns=cols)
# df['user_id'] = range(1, n_user + 1)
# df['recommend_user_id'] = recommend_user_ids
# df['matching'] = matchings
# df.to_csv('推荐结果.csv', index=None)


