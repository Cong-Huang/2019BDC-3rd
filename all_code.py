
# coding: utf-8

# # 1 特征提取
# * 训练数据只用了前4亿（尽管有的特征全部提取了，但实际只用了4亿）
# * 一共10种特征（有的特征分成了多个部分）
# * 除了特征10，其余的训练集特征都是4亿存在一起的，特征10由于一下提取4亿内存会炸，所以分成了前2亿和后两亿
# 
# ## 1.1 基础特征
# ### 1.1.1 长度相关按行提取特征
# 按行提取训练集长度相关特征（由于是按行提取的，所以测试集提取代码几乎完全一样所以不再给出）

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import csv
import json
from itertools import chain
from tqdm import tqdm
from multiprocessing import Pool

TRAIN_ROWS, TEST_ROWS=1000000000, 20000000
save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

assert os.path.exists(save_dir)

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']


debug = False
nrows = None
if debug:
    nrows = 20000
    
    
print('-------------- 开始提取特征 --------------')
since = time.time()
fea_names = ['query_length', 'title_length', 'WordMatchShare', 'WordMatchShare_query', 
             'WordMatchShare_title', 'LengthDiff', 'LengthDiffRate', 'LengthRatio_qt', 'LengthRatio_tq' # 这四个根据前面的计算得到
            ]
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
all_data = np.zeros((TRAIN_ROWS, len(fea_names)), dtype=np.float32)


with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        if debug and idx == nrows:
            print("debug mode. break.")
            break

        query = line[1].strip().split()
        title = line[3].strip().split()

        query_len, title_len = len(query), len(title)
        all_data[idx, name2idx['query_length']] = query_len
        all_data[idx, name2idx['title_length']] = title_len

        query_words = {}
        title_words = {}
        for word in query:  # query
            query_words[word] = query_words.get(word, 0) + 1
        for word in title:  # title
            title_words[word] = title_words.get(word, 0) + 1
        share_term = set(query_words.keys()) & set(title_words.keys())

         # -------------------- WordMatchShare --------------
        n_shared_word_in_query = sum([query_words[w] for w in share_term])
        n_shared_word_in_title = sum([title_words[w] for w in share_term])

        all_data[idx, name2idx['WordMatchShare']] = (n_shared_word_in_query + n_shared_word_in_title) / (query_len + title_len)
        all_data[idx, name2idx['WordMatchShare_query']] = n_shared_word_in_query / query_len
        all_data[idx, name2idx['WordMatchShare_title']] = n_shared_word_in_title / title_len


time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))


print("根据原始特征计算长度差等特征...")
all_data[:, name2idx['LengthDiff']] = all_data[:,name2idx['query_length']] - all_data[:, name2idx['title_length']]
all_data[:, name2idx['LengthDiffRate']] =          np.amin(all_data[:,[name2idx['query_length'],name2idx['title_length']]], axis=1) /          np.amax(all_data[:,[name2idx['query_length'],name2idx['title_length']]], axis=1)
         
all_data[:, name2idx['LengthRatio_qt']] =          all_data[:, name2idx['query_length']] / all_data[:, name2idx['title_length']]
all_data[:, name2idx['LengthRatio_tq']] =          all_data[:, name2idx['title_length']] / all_data[:, name2idx['query_length']]


print(all_data[:5, :])
print(all_data.dtype)

if not debug:
    print("saving...")
    save_path = os.path.join(save_dir, 'feature_1_0.csv.gz')
    np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
#     all_data.to_csv(save_path, compression='gzip', index=False)
    print(save_path, "save done!")


# ### 1.1.2 TFIDF相关特征
# 只用了前四亿的训练集加测试集的数据计算TFIDF, 最后提取的特征训练集和测试集分开存放

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import csv
import json
from tqdm import tqdm
from multiprocessing import Pool


E4 = 400000000
# TRAIN_ROWS = 1000000000
TEST_A_ROWS, TEST_B_ROWS = 20000000, 100000000
save_dir = "/home/kesci/work/feature_4e"
assert os.path.exists(save_dir)

train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_A_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
test_B_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'

train_columns = ['query_id','query','query_title_id','title','label']
test_columns = ['query_id','query','query_title_id','title']
print("当前进程PID:", os.getpid(), "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

debug = False
nrows = None
if debug:
    nrows = 1000000


doc_set = set()

print("train 4e...", flush=True)
train_df = pd.read_csv(train_file_path, names=train_columns, 
                        nrows=(nrows if debug else E4))[['query','title']]
print(train_df.shape)
for title in tqdm(train_df['title']):
    doc_set.add(title)
for query in tqdm(train_df['query']):
    doc_set.add(query)
del train_df
gc.collect()
    
    
print("test A...", flush=True)
test_A_df = pd.read_csv(test_A_file_path, names=test_columns, nrows=nrows)[['query','title']]
test_A_df["title"] = test_A_df["title"].apply(lambda x: x.strip()) # 去掉测试集title最后的tab
print(test_A_df.shape)
for title in tqdm(test_A_df['title']):
    doc_set.add(title)
for query in tqdm(test_A_df['query']):
    doc_set.add(query)
del test_A_df
gc.collect()


print("test B...", flush=True)
test_B_df = pd.read_csv(test_B_file_path, names=test_columns, nrows=nrows)[['query','title']]
test_B_df["title"] = test_B_df["title"].apply(lambda x: x.strip()) # 去掉测试集title最后的tab
print(test_B_df.shape)
for title in tqdm(test_B_df['title']):
    doc_set.add(title)
for query in tqdm(test_B_df['query']):
    doc_set.add(query)
del test_B_df
gc.collect()


idf = {}
doc_len = len(doc_set)
print("一共有%d个unique文档." % doc_len)
for doc in tqdm(doc_set):
    for word in set(doc.split()):
        idf[word] =  idf.get(word, 0) + 1

if not debug:
    idf_wordcount_path = os.path.join(save_dir, "train_4e_testAB_idf_wordcount.json")
    with open(idf_wordcount_path, 'w') as f:
        json.dump(idf, f)
        print(idf_wordcount_path, "saving done!")
    
    del doc_set
    gc.collect()
    
# print("----------------------加载idf---------------------")
# term_count_path = os.path.join(save_dir, "train_4e_testAB_idf_wordcount.json")
# with open(term_count_path, 'r') as f:
#     idf = json.load(f)
        
print("整个语料库term数:", len(idf))
doc_len = 149247529 + 1 # 一共有149247529个unique文档
for word in idf:
    idf[word] = np.log(doc_len / (idf[word] + 1.)) + 1
    
print('-------------- 开始提取特征 --------------')
since = time.time()
fea_names = ['TFIDFWordMatchShare', 'TFIDFWordMatchShare_query', 'TFIDFWordMatchShare_title']
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}

all_data = np.zeros((E4 + TEST_A_ROWS + TEST_B_ROWS, len(fea_names)), dtype=np.float32)

print("train:", flush=True)
since = time.time()
with open(train_file_path) as train_csvfile:
    csv_reader = csv.reader(train_csvfile)
    for index, line in tqdm(enumerate(csv_reader)):
        query = line[1].strip().split()
        title = line[3].strip().split()
        
        query_words = {}
        title_words = {}
        for word in query:  # query
            query_words[word] = query_words.get(word, 0) + 1
        for word in title:  # title
            title_words[word] = title_words.get(word, 0) + 1
        share_term = set(query_words.keys()) & set(title_words.keys())

        # --------------------- TFIDFWordMatchShare ---------------------
        sum_shared_word_in_query = sum([query_words[w] * idf.get(w, 0) for w in share_term])
        sum_shared_word_in_title = sum([title_words[w] * idf.get(w, 0) for w in share_term])
        sum_query_tol = sum(query_words[w] * idf.get(w, 0) for w in query_words)
        sum_title_tol = sum(title_words[w] * idf.get(w, 0) for w in title_words)
        sum_tol = sum_query_tol + sum_title_tol

        all_data[index, name2idx['TFIDFWordMatchShare']] = (sum_shared_word_in_query + sum_shared_word_in_title) / sum_tol
        all_data[index, name2idx['TFIDFWordMatchShare_query']] = sum_shared_word_in_query / sum_query_tol
        all_data[index, name2idx['TFIDFWordMatchShare_title']] = sum_shared_word_in_title / sum_title_tol
            
        if index == (E4 - 1):
            print("训练集前4亿提取完成!", flush=True)
            break
        if debug and index == 100000 - 1:
            print("debug mode. break")
            break
        
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (index, time_elapsed // 60, time_elapsed % 60))


offset = index + 1
print("test A (offset=%d):" % offset, flush=True)
since = time.time()
with open(test_A_file_path) as test_A_csvfile:
    csv_reader = csv.reader(test_A_csvfile)
    for index, line in tqdm(enumerate(csv_reader)):
        query = line[1].strip().split()
        title = line[3].strip().split()
        
        query_words = {}
        title_words = {}
        for word in query:  # query
            query_words[word] = query_words.get(word, 0) + 1
        for word in title:  # title
            title_words[word] = title_words.get(word, 0) + 1
        share_term = set(query_words.keys()) & set(title_words.keys())

        # --------------------- TFIDFWordMatchShare ---------------------
        sum_shared_word_in_query = sum([query_words[w] * idf.get(w, 0) for w in share_term])
        sum_shared_word_in_title = sum([title_words[w] * idf.get(w, 0) for w in share_term])
        sum_query_tol = sum(query_words[w] * idf.get(w, 0) for w in query_words)
        sum_title_tol = sum(title_words[w] * idf.get(w, 0) for w in title_words)
        sum_tol = sum_query_tol + sum_title_tol

        all_data[index+offset, name2idx['TFIDFWordMatchShare']] = (sum_shared_word_in_query + sum_shared_word_in_title) / sum_tol
        all_data[index+offset, name2idx['TFIDFWordMatchShare_query']] = sum_shared_word_in_query / sum_query_tol
        all_data[index+offset, name2idx['TFIDFWordMatchShare_title']] = sum_shared_word_in_title / sum_title_tol
        
        if debug and index == 100000 - 1:
            print("debug mode. break")
            break
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (index, time_elapsed // 60, time_elapsed % 60))


offset = offset + index + 1
print("test B:(offset=%d):" % offset, flush=True)
since = time.time()
with open(test_B_file_path) as test_B_csvfile:
    csv_reader = csv.reader(test_B_csvfile)
    for index, line in tqdm(enumerate(csv_reader)):
        query = line[1].strip().split()
        title = line[3].strip().split()
        
        query_words = {}
        title_words = {}
        for word in query:  # query
            query_words[word] = query_words.get(word, 0) + 1
        for word in title:  # title
            title_words[word] = title_words.get(word, 0) + 1
        share_term = set(query_words.keys()) & set(title_words.keys())

        # --------------------- TFIDFWordMatchShare ---------------------
        sum_shared_word_in_query = sum([query_words[w] * idf.get(w, 0) for w in share_term])
        sum_shared_word_in_title = sum([title_words[w] * idf.get(w, 0) for w in share_term])
        sum_query_tol = sum(query_words[w] * idf.get(w, 0) for w in query_words)
        sum_title_tol = sum(title_words[w] * idf.get(w, 0) for w in title_words)
        sum_tol = sum_query_tol + sum_title_tol

        all_data[index+offset, name2idx['TFIDFWordMatchShare']] = (sum_shared_word_in_query + sum_shared_word_in_title) / sum_tol
        all_data[index+offset, name2idx['TFIDFWordMatchShare_query']] = sum_shared_word_in_query / sum_query_tol
        all_data[index+offset, name2idx['TFIDFWordMatchShare_title']] = sum_shared_word_in_title / sum_title_tol
        
        if debug and index == 100000 - 1:
            print("debug mode. break")
            break
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (index, time_elapsed // 60, time_elapsed % 60))

print(all_data[:5, :])
print(all_data[-5:, :])
print(all_data.shape)

if not debug:
    train_save_path = os.path.join(save_dir, 'feature_1_1_4e.csv.gz')
    print("保存4亿训练集到%s..." % train_save_path)
    np.savetxt(train_save_path, all_data[:E4, :], fmt="%f", delimiter=",", 
                header=",".join(fea_names), comments="")
                
    test_A_save_path = os.path.join(save_dir, 'feature_1_1_4e_testA.csv.gz')
    print("保存测试集A到%s..." % test_A_save_path)
    np.savetxt(test_A_save_path, all_data[E4:E4+TEST_A_ROWS, :], fmt="%f", delimiter=",", 
                header=",".join(fea_names), comments="")
    print("save done!")
    
    test_B_save_path = os.path.join(save_dir, 'feature_1_1_4e_testB.csv.gz')
    print("保存测试集B到%s..." % test_B_save_path)
    np.savetxt(test_B_save_path, all_data[E4+TEST_A_ROWS:, :], fmt="%f", delimiter=",", 
                header=",".join(fea_names), comments="")
    print("save done!")

    del all_data


# ## 1.2 NgramJaccard特征
# 按行提取NgramJaccard特征: ['NgramJaccardCoef_1' 'NgramJaccardCoef_2' 'NgramJaccardCoef_3' 'NgramJaccardCoef_4']
# 
# （由于是按行提取的，所以测试集提取代码几乎完全一样所以不再给出）

# In[ ]:


# base feature
import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import csv
from itertools import chain
from tqdm import tqdm

import sys
sys.path.append("/home/kesci/work/code") 
from utils import *  # NgramUtil


TRAIN_ROWS, TEST_ROWS=1000000000, 20000000
save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

assert os.path.exists(save_dir)

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']


debug = False
nrows = None
if debug:
    nrows = 100000

print('-------------- 开始提取NgramJaccard特征 --------------')
since = time.time()
fea_names = ['NgramJaccardCoef_1', 'NgramJaccardCoef_2', 'NgramJaccardCoef_3', 'NgramJaccardCoef_4']
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
all_data = np.zeros((TRAIN_ROWS, len(fea_names)), dtype=np.float32)

with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        if debug and idx == nrows:
            print("debug mode, break.")
            break
        query = line[1].strip().split()
        title = line[3].strip().split()

        for n in range(1, 5):
            query_ngrams = NgramUtil.ngrams(query, n)
            title_ngrams = NgramUtil.ngrams(title, n)
            all_data[idx, name2idx['NgramJaccardCoef_%d'%n]] = DistanceUtil.jaccard_coef(query_ngrams, 
                                                                                         title_ngrams)
    
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))


print(all_data.shape)
print(all_data[:5, :])
print(all_data.dtype)

if not debug:
    print("saving...")
    save_path = os.path.join(save_dir, 'feature_2.csv.gz')
    np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
#     all_data.to_csv(save_path, compression='gzip', index=False)
    print(save_path, "save done!")

del all_data
gc.collect()


# ## 1.3 Levenshtein相关
# 特征: ["Levenshtein_ratio", "Levenshtein_distance", "query_title_common_words", "common_word_ratio"]
#     
# （由于是按行提取的，所以测试集提取代码几乎完全一样所以不再给出）

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import csv
import Levenshtein
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from itertools import chain

# import sys
# sys.path.append("/home/kesci/work/code") 
# from utils import *  # NgramUtil

TRAIN_ROWS, TEST_ROWS=1000000000, 20000000
save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

assert os.path.exists(save_dir)

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']


debug = False
nrows = None
if debug:
    nrows = 100000

# =============三个函数====================
def edit_distance(word1, word2):  # 计算编辑距离
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            delta = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta,
                           min(dp[i - 1][j] + 1, dp[i][j - 1] + 1))
    return dp[len1][len2]


# ----- 提取特征 -----
since = time.time()

fea_names = ["Levenshtein_ratio", "Levenshtein_distance_char",
# "Levenshtein_distance", # 这个跑起来太慢, 先不要了
"query_title_common_words", "common_word_ratio"]
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
all_data = np.zeros((TRAIN_ROWS, len(fea_names)), dtype=np.float32)

with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        if debug and idx == nrows:
            print("debug mode, break.")
            break
        query_str = line[1].strip()
        query = query_str.split()
        title_str = line[3].strip()
        title = title_str.split()

        # 9000 row/s
        all_data[idx, name2idx["Levenshtein_ratio"]] = Levenshtein.seqratio(query, title)
        # all_data[idx, name2idx["Levenshtein_distance"]] = edit_distance(query, title) # 很慢
        all_data[idx, name2idx["Levenshtein_distance_char"]] = Levenshtein.distance(query_str, title_str)
        common_words_len = len(set(query) & set(title))
        all_data[idx, name2idx["query_title_common_words"]] = common_words_len
        all_data[idx, name2idx["common_word_ratio"]]= common_words_len / min(len(query), len(title))

time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))

print(all_data.shape)
print(all_data[:5, :])
print(all_data.dtype)

if not debug:
    print("saving...")
    save_path = os.path.join(save_dir, 'feature_3.csv.gz')
    np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
    #     all_data.to_csv(save_path, compression='gzip', index=False)
    print(save_path, "save done!")

del (all_data)
gc.collect()


# ## 1.4 sequencematch相关
# 特征4: sequencematch ["lcsubstr_len", "lcseque_len", "longest_match_size", "longest_match_ratio"]
# 
# （由于是按行提取的，所以测试集提取代码几乎完全一样所以不再给出）

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import csv
import Levenshtein
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from itertools import chain


TRAIN_ROWS, TEST_ROWS=1000000000, 20000000
save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

assert os.path.exists(save_dir)

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']


debug = False
nrows = None
if debug:
    nrows = 100000


def lcsubstr_lens(s1, s2):  # 计算最长子串长度
    m = [[0 for i in range(len(s2) + 1)]
         for j in range(len(s1) + 1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  #最长匹配的长度
    p = 0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return mmax


def lcseque_lens(s1, s2):  # 计算最长子序列长度
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [[0 for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    # d用来记录转移方向
    d = [[None for x in range(len(s2) + 1)] for y in range(len(s1) + 1)]
    for p1 in range(len(s1)):
        for p2 in range(len(s2)):
            if s1[p1] == s2[p2]:  #字符匹配成功，则该位置的值为左上方的值加1
                m[p1 + 1][p2 + 1] = m[p1][p2] + 1
                d[p1 + 1][p2 + 1] = 'ok'
            elif m[p1 + 1][p2] > m[p1][p2 + 1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1 + 1][p2 + 1] = m[p1 + 1][p2]
                d[p1 + 1][p2 + 1] = 'left'
            else:  #上值大于左值，则该位置的值为上值，并标记方向up
                m[p1 + 1][p2 + 1] = m[p1][p2 + 1]
                d[p1 + 1][p2 + 1] = 'up'
    (p1, p2) = (len(s1), len(s2))
    s = []
    while m[p1][p2]:  #不为None时
        c = d[p1][p2]
        if c == 'ok':  #匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1 - 1])
            p1 -= 1
            p2 -= 1
        if c == 'left':  #根据标记，向左找下一个
            p2 -= 1
        if c == 'up':  #根据标记，向上找下一个
            p1 -= 1
    return len(s)


# ----- 提取特征 -----
since = time.time()
fea_names = ["lcsubstr_len", "lcseque_len", "longest_match_size", "longest_match_ratio"]
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
all_data = np.zeros((TRAIN_ROWS, len(fea_names)), dtype=np.float32)

with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        if debug and idx == nrows:
            print("debug mode, break.")
            break
        query_str = line[1].strip()
        query = query_str.split()
        title_str = line[3].strip()
        title = title_str.split()

        # part2: 22000 row/s
        all_data[idx, name2idx["lcsubstr_len"]] = lcsubstr_lens(query, title)
        all_data[idx, name2idx["lcseque_len"]] = lcseque_lens(query, title)
        sq = SequenceMatcher(a=query, b=title)
        match = sq.find_longest_match(0, len(query), 0, len(title))
        all_data[idx, name2idx["longest_match_size"]] = match.size
        all_data[idx, name2idx["longest_match_ratio"]] = match.size / min(len(query), len(title))
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))

print(all_data[:5, :])
print(all_data.dtype)

if not debug:
    print("saving...")
    save_path = os.path.join(save_dir, 'feature_4.csv.gz')
    np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
#     all_data.to_csv(save_path, compression='gzip', index=False)
    print(save_path, "save done!")

del (all_data)
gc.collect()


# ## 1.5 Fuzzy相关（第一部分）
# 特征5: Fuzzy part1 Fuzzy部分1: ["fuzz_qratio", "fuzz_partial_ratio"]
#         
# （由于是按行提取的，所以测试集提取代码几乎完全一样所以不再给出）

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import csv
import Levenshtein
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from itertools import chain


TRAIN_ROWS, TEST_ROWS=1000000000, 20000000
save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

assert os.path.exists(save_dir)

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']


debug = False
nrows = None
if debug:
    nrows = 100000


# ----- 提取特征 -----
since = time.time()


fea_names = ["fuzz_qratio", 
# "fuzz_WRatio", # 太慢, 去掉 
"fuzz_partial_ratio"]
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
all_data = np.zeros((TRAIN_ROWS, len(fea_names)), dtype=np.float32)

with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        if debug and idx == nrows:
            print("debug mode, break.")
            break
        query_str = line[1].strip()
        query = query_str.split()
        title_str = line[3].strip()
        title = title_str.split()

        # part3: 8000 row/s
        all_data[idx, name2idx["fuzz_qratio"]] = fuzz.QRatio(query_str, title_str)
        # all_data[idx, name2idx["fuzz_WRatio"]] = fuzz.WRatio(query_str, title_str) # 慢
        all_data[idx, name2idx["fuzz_partial_ratio"]] = fuzz.partial_ratio(query_str, title_str)
            

time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))

print(all_data[:5, :])
print(all_data.dtype)

if not debug:
    print("saving...")
    save_path = os.path.join(save_dir, 'feature_5.csv.gz')
    np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
#     all_data.to_csv(save_path, compression='gzip', index=False)
    print(save_path, "save done!")

del (all_data)
gc.collect()


# ## 1.6 Fuzzy相关（第二部分）
# 特征6: Fuzzy 部分2: ["fuzz_partial_token_sort_ratio","fuzz_token_set_ratio", 
#                  "fuzz_token_sort_ratio"]
#         
# （由于是按行提取的，所以测试集提取代码几乎完全一样所以不再给出）

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import csv
import Levenshtein
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
from itertools import chain


TRAIN_ROWS, TEST_ROWS=1000000000, 20000000
save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

assert os.path.exists(save_dir)

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']


debug = False
nrows = None
if debug:
    nrows = 100000


# ----- 提取特征 -----
since = time.time()


fea_names = [
    # "fuzz_partial_token_set_ratio", # 太慢, 去掉
"fuzz_partial_token_sort_ratio", 
             "fuzz_token_set_ratio", "fuzz_token_sort_ratio"]
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
all_data = np.zeros((TRAIN_ROWS, len(fea_names)), dtype=np.float32)

with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        if debug and idx == nrows:
            print("debug mode, break.")
            break
        query_str = line[1].strip()
        query = query_str.split()
        title_str = line[3].strip()
        title = title_str.split()

        # part4: 8000 row/s
        # all_data[idx, name2idx["fuzz_partial_token_set_ratio"]]  = fuzz.partial_token_set_ratio(query_str, title_str)
        all_data[idx, name2idx["fuzz_partial_token_sort_ratio"]]  = fuzz.partial_token_sort_ratio(query_str, title_str)
        all_data[idx, name2idx["fuzz_token_set_ratio"]] = fuzz.token_set_ratio(query_str, title_str)
        all_data[idx, name2idx["fuzz_token_sort_ratio"]] = fuzz.token_sort_ratio(query_str, title_str)


time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))

print(all_data[:5, :])
print(all_data.dtype)

if not debug:
    print("saving...")
    save_path = os.path.join(save_dir, 'feature_6.csv.gz')
    np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
#     all_data.to_csv(save_path, compression='gzip', index=False)
    print(save_path, "save done!")

del (all_data)
gc.collect()


# ## 1.7 熵相关
# 特征7: 信息熵 ["query_Entropy", "title_Entropy", "query_title_Entropy", "WordMatchShare_Entropy"]
# 
# （由于是按行提取的，所以测试集提取代码几乎完全一样所以不再给出）

# In[ ]:


# informationEntropy
import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import json
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
import csv
import Levenshtein
from difflib import SequenceMatcher
from itertools import chain

# TRAIN_ROWS, TEST_ROWS=20000, 20000
# save_dir = "./feature"
# train_file_path = './sample_data/train_data.sample'
# test_file_path = './sample_data/train_data_nolabel.csv'

TRAIN_ROWS, TEST_ROWS=1000000000, 20000000
save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

assert os.path.exists(save_dir)

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']


debug = False
nrows = None
if debug:
    nrows = 100000


print(" ----- 提取informationEntropy特征 -----")
since = time.time()
fea_names = ["query_Entropy", "title_Entropy", "query_title_Entropy", "WordMatchShare_Entropy"]
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
all_data = np.zeros((TRAIN_ROWS, len(fea_names)), dtype=np.float32)

with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        if debug and idx == nrows:
            print("debug mode, break.")
            break

        query_words = {}
        title_words = {}
        query_title_words = {}
        for word in line[1].strip().split():  # query
            query_words[word] = query_words.get(word, 0) + 1
            query_title_words[word] = query_title_words.get(word, 0) + 1
        for word in line[3].strip().split():  # title
            title_words[word] = title_words.get(word, 0) + 1
            query_title_words[word] = query_title_words.get(word, 0) + 1

        n_query_tol = sum(query_words.values())
        n_title_tol = sum(title_words.values())
        n_query_title_tol = sum(query_title_words.values())
        all_data[idx, name2idx["query_Entropy"]] = abs(sum(map(lambda x: x/n_query_tol *           math.log(x/n_query_tol,2),query_words.values())))
        all_data[idx, name2idx["title_Entropy"]] = abs(sum(map(lambda x: x/n_title_tol *           math.log(x/n_title_tol,2),title_words.values())))
        all_data[idx, name2idx["query_title_Entropy"]] = abs(sum(map(lambda x: x/n_query_title_tol *           math.log(x/n_query_title_tol,2),query_title_words.values())))

        query_title_words_share = {}
        for word in query_words:
            if word in title_words:
                query_title_words_share[word] = query_title_words_share.get(
                    word, 0) + query_words[word]
        for word in title_words:
            if word in query_words:
                query_title_words_share[word] = query_title_words_share.get(
                    word, 0) + title_words[word]

        all_data[idx, name2idx["WordMatchShare_Entropy"]] = abs(sum(map(lambda x: x/n_query_title_tol *           math.log(x/n_query_title_tol,2),query_title_words_share.values())))


time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))

print(all_data[:5, :])
print(all_data.dtype)

if not debug:
    print("saving...")
    save_path = os.path.join(save_dir, 'feature_7.csv.gz')
    np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
#     all_data.to_csv(save_path, compression='gzip', index=False)
    print(save_path, "save done!")

del (all_data)
gc.collect()


# ## 1.8 转换率特征
# 提取query 和 title的转换率cvr, query选取1个关键词，title选取2个关键词. 关键词通过TFIDF确定.

# In[ ]:


import pandas as pd 
import numpy as np
import gc, json, time, csv, os 
import copy
from collections import Counter
from tqdm import tqdm 
print("当前进程PID:", os.getpid(), "开始时间:", 
      time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 

save_dir = "/home/kesci/work/feature_4e"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
testA_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
testB_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'

train_columns = ['query_id','query','query_title_id','title','label']
test_columns = ['query_id','query','query_title_id','title']
assert os.path.exists(save_dir)
print("当前进程PID:", os.getpid(), "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

E4 = 400000000
TESTA_ROWS = 20000000
TESTB_ROWs = 100000000

print("----------------------加载idf---------------------")
term_count_path = '/home/kesci/work/feature_4e/train_4e_testAB_idf_wordcount.json'
with open(term_count_path, 'r') as f:
    word2count = json.load(f)

print("整个语料库term数:", len(word2count))
doc_len = 149247529 + 1 # 一共有149247530个unique文档
idf = {}
for word in word2count:
    idf[word] = np.log(doc_len / (word2count[word] + 1.)) + 1

query_term_top1_map = {}  # 映射关系（建立字典hash）
title_term_top2_map = {}

# 提取train_data 的query 和 title 关键词
train_query_term_top1 = np.zeros((E4,), dtype=np.int32)
train_title_term_top2 = np.zeros((E4,), dtype=np.int32)
with open(train_file_path) as csvfile:
    for index, line in tqdm(enumerate(csv.reader(csvfile))):
        if index >= E4: break
        query = line[1].strip().split()
        title = line[3].strip().split()
        # for query
        query_count = Counter(query)
        query_score = {word: query_count[word]/sum(query_count.values())*idf[word] for word in query_count}
        word_top1 = sorted(query_score.items(), key=lambda x: x[1], reverse=True)[0][0]
        if query_term_top1_map.get(word_top1) == None:
            query_term_top1_map[word_top1] = len(query_term_top1_map)
        train_query_term_top1[index] = query_term_top1_map[word_top1]
        
        # for title
        title_count = Counter(title)
        title_score = {word: title_count[word]/sum(title_count.values())*idf[word] for word in title_count}
        title_score = sorted(title_score.items(), key=lambda x: x[1], reverse=True)
        word_top2 = ' '.join(list(set([x[0] for x in title_score[:2]])))
        if title_term_top2_map.get(word_top2) == None:
            title_term_top2_map[word_top2] = len(title_term_top2_map)
        train_title_term_top2[index] = title_term_top2_map[word_top2]

#  提取testA_data 的query 和 title 关键词
testA_query_term_top1 = np.zeros((TESTA_ROWS,), dtype=np.int32)
testA_title_term_top2 = np.zeros((TESTA_ROWS,), dtype=np.int32)
since = time.time()
with open(testA_file_path) as csvfile:
    for index, line in enumerate(csv.reader(csvfile)):
        query = line[1].strip().split()
        title = line[3].strip().split()
        # for query
        query_count = Counter(query)
        query_score = {word: query_count[word]/sum(query_count.values())*idf[word] for word in query_count}
        word_top1 = sorted(query_score.items(), key=lambda x: x[1], reverse=True)[0][0]
        if query_term_top1_map.get(word_top1) == None:
            query_term_top1_map[word_top1] = len(query_term_top1_map)
        testA_query_term_top1[index] = query_term_top1_map[word_top1]
        
        # for title
        title_count = Counter(title)
        title_score = {word: title_count[word]/sum(title_count.values())*idf[word] for word in title_count}
        title_score = sorted(title_score.items(), key=lambda x: x[1], reverse=True)
        word_top2 = ' '.join(list(set([x[0] for x in title_score[:2]])))
        if title_term_top2_map.get(word_top2) == None:
            title_term_top2_map[word_top2] = len(title_term_top2_map)
        testA_title_term_top2[index] = title_term_top2_map[word_top2]
        if (index+1) % 4000000 == 0:
            print(index+1, 'step end. Time consumed:', time.time() - since)

# 提取testB_data 的query and title的关键词
testB_query_term_top1 = np.zeros((TESTB_ROWs,), dtype=np.int32)
testB_title_term_top2 = np.zeros((TESTB_ROWs,), dtype=np.int32)
since = time.time()
with open(testB_file_path) as csvfile:
    for index, line in enumerate(csv.reader(csvfile)):
        query = line[1].strip().split()
        title = line[3].strip().split()
        # for query
        query_count = Counter(query)
        query_score = {word: query_count[word]/sum(query_count.values())*idf[word] for word in query_count}
        word_top1 = sorted(query_score.items(), key=lambda x: x[1], reverse=True)[0][0]
        if query_term_top1_map.get(word_top1) == None:
            query_term_top1_map[word_top1] = len(query_term_top1_map)
        testB_query_term_top1[index] = query_term_top1_map[word_top1]
        
        # for title
        title_count = Counter(title)
        title_score = {word: title_count[word]/sum(title_count.values())*idf[word] for word in title_count}
        title_score = sorted(title_score.items(), key=lambda x: x[1], reverse=True)
        word_top2 = ' '.join(list(set([x[0] for x in title_score[:2]])))
        if title_term_top2_map.get(word_top2) == None:
            title_term_top2_map[word_top2] = len(title_term_top2_map)
        testB_title_term_top2[index] = title_term_top2_map[word_top2]
        if (index+1) % 4000000 == 0:
            print(index+1, 'step end. Time consumed:', time.time() - since)

# 查看与分析提取的结果
print(train_query_term_top1[:10], train_title_term_top2[:10])
print(testA_query_term_top1[:10], testA_title_term_top2[:10])
print(testB_query_term_top1[:10], testB_title_term_top2[:10])

print('train的query和title总量: ', len(train_query_term_top1), len(train_title_term_top2))
print('train的query和title不重复总量: ', len(set(train_query_term_top1)), len(set(train_title_term_top2)))   
# 之前1e(636341, 40104211)  之前4e(756051, 53963588)

print('映射最大值:', max(query_term_top1_map.values()), max(title_term_top2_map.values()))
print('映射大小：', len(query_term_top1_map), len(title_term_top2_map))

print(len(set(testA_query_term_top1)), len(testA_query_term_top1), len(set(train_query_term_top1)))
print(len(set(testA_title_term_top2)), len(testA_title_term_top2), len(set(train_title_term_top2)))
# 之前1e(285389, 20000000, 636341)  之前4e(285310, 20000000, 756051)
print(len(set(testB_query_term_top1)), len(testB_query_term_top1))
print(len(set(testB_title_term_top2)), len(testB_title_term_top2))

print('train与testA相交(query)：', len(set(testA_query_term_top1) & set(train_query_term_top1)))
print('train与testA相交(title)：', len(set(testA_title_term_top2) & set(train_title_term_top2)))
print('train与testB相交(query)：', len(set(testB_query_term_top1) & set(train_query_term_top1)))
print('train与testB相交(title)：', len(set(testB_title_term_top2) & set(train_title_term_top2)))
del title_term_top2_map
gc.collect()

# 转化为pandas数据框
all_data = pd.DataFrame()
all_data['query'] = np.concatenate((train_query_term_top1,testA_query_term_top1,testB_query_term_top1))
all_data['title'] = np.concatenate((train_title_term_top2,testA_title_term_top2,testB_title_term_top2))
label_4e=pd.read_csv("/home/kesci/work/feature/train_label.csv.gz", nrows=E4, dtype=np.int32)["label"].values
all_data['label'] = np.concatenate((label_4e, np.array([-1]*(120000000))))

del train_query_term_top1,testA_query_term_top1,testB_query_term_top1
del train_title_term_top2,testA_title_term_top2,testB_title_term_top2
gc.collect()

# 将int64 转化为 int32
for col in all_data.columns:
    if str(all_data[col].dtypes) == 'int64':
        all_data[col] = all_data[col].astype(np.int32)
all_data.to_csv('nn_related/all_data_cvr.csv.gz', compression='gzip', index=False) # 以便后续使用

# 关闭kernel,从这再开始
all_data = pd.read_csv('nn_related/all_data_cvr.csv.gz')
random_sector = np.random.randint(1, 6, size=(all_data.shape[0])).astype(np.int32)  # 1,2,3,4,5
all_data['random_sector'] = random_sector
all_data['random_sector'][400000000:] = 0
print(all_data.dtypes, all_data.shape)
print(all_data['random_sector'].value_counts())

# 导入贝叶斯平滑类
import numpy
import random
import scipy.special as special
import math
from math import log

class HyperParam(object):
    def __init__(self, alpha, beta): # 先初始化alpha和beta
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return I, C

    # 更新方式1
    def update_from_data_by_FPI(self, tries, success, iter_num, epsilon):
        '''estimate alpha, beta using fixed point iteration(类似EM估计)
        tries: 展示次数
        success: 点击次数
        iter_num: 迭代次数
        epsilon: 精度
        '''
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            # 当迭代稳定时，停止迭代
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        '''fixed point iteration'''
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        for i in range(len(tries)):
            sumfenzialpha += (special.digamma(success[i]+alpha) - special.digamma(alpha))
            sumfenzibeta += (special.digamma(tries[i]-success[i]+beta) - special.digamma(beta))
            sumfenmu += (special.digamma(tries[i]+alpha+beta) - special.digamma(alpha+beta))

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)

    # 更新方式1
    def update_from_data_by_moment(self, tries, success):
        '''estimate alpha, beta using moment estimation(矩估计)
        tries: 展示次数
        success: 点击次数
        '''
        # 样本均值和样本方差
        mean, var = self.__compute_moment(tries, success)
        #print 'mean and variance: ', mean, var
        #self.alpha = mean*(mean*(1-mean)/(var+0.000001)-1)
        self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        #self.beta = (1-mean)*(mean*(1-mean)/(var+0.000001)-1)
        self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def __compute_moment(self, tries, success):
        '''moment estimation(求样本均值和样本方差)'''
        ctr_list = []
        var = 0.0
        for i in range(len(tries)):
            ctr_list.append(float(success[i])/tries[i])
        mean = sum(ctr_list)/len(ctr_list)
        for ctr in ctr_list:
            var += pow(ctr-mean, 2)

        return mean, var/(len(ctr_list)-1)


# 计算query、title的转换率 
since = time.time()
sec_size = 5
frac_size = 0.5 
convert_feature = ['query', 'title']
for index, feature in enumerate(convert_feature):
    print('正在计算' + feature + '转换率')
    for sec in range(sec_size + 1):  # 0, 1, 2, 3, 4, 5  #0 is test， 1 is valid
        print(sec, '折')
        if sec == 1:
            temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)][[feature, 'label']]
        elif sec == 0:
            temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)][[feature, 'label']]
        else:
            temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)&(all_data.random_sector != 1)][[feature, 'label']]
            temp = temp.sample(frac = frac_size, random_state = 2019).reset_index(drop = True)
        
        temp[feature + '_all_count'] = temp.groupby(feature).label.transform('count')
        temp[feature + '_label_count'] = temp.groupby(feature).label.transform('sum')
		# 贝叶斯平滑
        HP = HyperParam(1, 1)
        HP.update_from_data_by_moment(temp[feature + '_all_count'].values, temp[feature + '_label_count'].values)
        temp[feature + '_convert'] = (temp[feature + '_label_count'] + HP.alpha) / (temp[feature + '_all_count'] + HP.alpha + HP.beta)
        print('temp before shape:', temp.shape)
        temp = temp[[feature, feature + '_convert']].drop_duplicates()
        print('temp after shape:', temp.shape)

        sec_data = all_data[all_data.random_sector == sec][[feature]]
        all_data.loc[all_data.random_sector == sec, feature + '_convert'] = pd.merge(sec_data, temp, on=feature,
                                                                                     how='left')[feature+'_convert'].values
        del temp, sec_data
        gc.collect()
    all_data[feature + '_convert'] = all_data[feature + '_convert'].astype(np.float32)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

E4 = 400000000
print(all_data[['label', 'query_convert', 'title_convert']][:E4].count())
print(all_data[['label', 'query_convert', 'title_convert']][E4:E4+20000000].count())
print(all_data[['label', 'query_convert', 'title_convert']][E4+20000000:].count())

print('==========计算交叉转换率==========')
since = time.time()
sec_size = 5 
frac_size = 0.5 
convert_feature = ['query', 'title']
first_feature, second_feature = convert_feature[0], convert_feature[1]
print('正在计算' + first_feature + '和' + second_feature + '的转换率')
for sec in range(sec_size + 1):
    print(sec, '折')
    if sec == 1:
        temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)][[first_feature, second_feature, 'label']]
    elif sec == 0:
        temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)][[first_feature, second_feature, 'label']]
    else:
        temp = all_data[(all_data.label != -1)&(all_data.random_sector != sec)&(all_data.random_sector != 1)][[first_feature, second_feature, 'label']]
        temp = temp.sample(frac = frac_size, random_state = 2019).reset_index(drop = True)

    temp['query_title_all_count'] = temp.groupby([first_feature, second_feature]).label.transform('count')
    temp['query_title_label_count'] = temp.groupby([first_feature, second_feature]).label.transform('sum')
    
    HP = HyperParam(1, 1)
    HP.update_from_data_by_moment(temp['query_title_all_count'].values, temp['query_title_label_count'].values)
    temp['query_title_convert']=(temp['query_title_label_count']+HP.alpha)/(temp['query_title_all_count']+HP.alpha+HP.beta)
    
    temp = temp[[first_feature, second_feature, 'query_title_convert']].drop_duplicates()
    sec_data = all_data[all_data.random_sector == sec][[first_feature, second_feature]]
    all_data.loc[all_data.random_sector == sec, 'query_title_convert']=pd.merge(sec_data, temp, on=[first_feature, second_feature],
                                                                                how='left')['query_title_convert'].values
    del temp, sec_data
    gc.collect()
all_data['query_title_convert'] = all_data['query_title_convert'].astype(np.float32)
time_elapsed = time.time() - since
print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

E4 = 400000000
print(all_data[['label', 'query_convert', 'title_convert', 'query_title_convert']][:E4].count())
print(all_data[['label', 'query_convert', 'title_convert', 'query_title_convert']][E4:E4+20000000].count())
print(all_data[['label', 'query_convert', 'title_convert', 'query_title_convert']][E4+20000000:].count())


# 保存cvr特征
all_data[:E4][['query_convert', 'title_convert', 
               'query_title_convert']].to_csv('feature_4e/train4e_cvr.csv.gz', compression='gzip', index=False)
all_data[E4:E4+20000000][['query_convert', 'title_convert', 
            'query_title_convert']].to_csv('feature_4e/testA_cvr.csv.gz', compression='gzip', index=False)
all_data[E4+20000000:][['query_convert', 'title_convert', 
        'query_title_convert']].to_csv('feature_4e/testB_cvr.csv.gz', compression='gzip', index=False)
print('finished!')


# ## 1.9 word2vec相似度
# 特征9: 提取前4亿和最终两个测试集的word2vec句向量距离特征:'w2v_avg_cosine', 'w2v_avg_cityblock', 'w2v_avg_minkowski', 'w2v_avg_braycurtis', 'w2v_avg_canberra'
# 
# * 实际将前两个作为特征9_0后三个作为特征9_1同时提取, 以下代码是9_0部分
# * 需提前训练好word2vec模型
# * 下面是用的100维的word2vec，实际还是用了300维的

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import json
from tqdm import tqdm
import csv
from sklearn.metrics.pairwise import paired_distances
from scipy.spatial import distance
import gensim
from gensim.models import Word2Vec
from itertools import chain


E4 = 400000000
# TRAIN_ROWS = 1000000000
TEST_A_ROWS, TEST_B_ROWS = 20000000, 100000000

save_dir = "/home/kesci/work/feature_4e"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_A_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
test_B_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'


train_columns = ['query_id','query','query_title_id','title','label']
test_columns = ['query_id','query','query_title_id','title']
assert os.path.exists(save_dir)
print("当前进程PID:", os.getpid(), "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

w2v_model_path = '/home/kesci/work/word2vec/w2v_100_cbow_4e.model'


debug=False


print("----- 从%s加载word2vec模型 -----" % w2v_model_path)
# 这里放上训练时的EpochSaver, 否则load会报错
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        
    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" % 
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)
            
        self.pre_loss = cum_loss
        self.since = time.time()
        print("\n\n\n")
w2v_model = Word2Vec.load(w2v_model_path)
embed_size = w2v_model.trainables.layer1_size
print("加载完毕. 总词典数: %d, embeding size: %d" % (len(w2v_model.wv.vocab), embed_size))


def w2v_sent2vec(sentence, model):
    """计算句子的平均word2vec向量, sentences是一个句子, 句向量最后会归一化"""
    M = []
    for word in sentence.split():
        try:
            M.append(model.wv[word])
        except KeyError: # 不在词典里
            continue
    if len(M) == 0:
        return ((-1 / np.sqrt(embed_size)) * np.ones(embed_size)).astype(np.float32)
    M = np.array(M)
    v = M.sum(axis=0)
    return (v / np.sqrt((v ** 2).sum())).astype(np.float32)

print(" ----- 开始提取均word2vec相似度特征 -----")
since = time.time()
fea_names = [
    "w2v_avg_cosine",
    "w2v_avg_cityblock",
    # "w2v_avg_minkowski",
    # "w2v_avg_braycurtis",
    # "w2v_avg_canberra"
    ]
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}
print("一共%d个特征:" % len(fea_names))
print(fea_names, flush=True)

N = (300000 if debug else (E4 + TEST_A_ROWS + TEST_B_ROWS))
all_data = np.zeros((N, len(fea_names)), dtype=np.float32)

print("train:", flush=True)
since = time.time()
with open(train_file_path) as train_csvfile:
    csv_reader = csv.reader(train_csvfile)
    for index, line in tqdm(enumerate(csv_reader)):
        query = line[1].strip()
        title = line[3].strip()
        
        query_avg_w2v = w2v_sent2vec(query, w2v_model)
        title_avg_w2v = w2v_sent2vec(title, w2v_model)
        
        all_data[index, name2idx["w2v_avg_cosine"]] = distance.cosine(query_avg_w2v, title_avg_w2v)
        all_data[index, name2idx["w2v_avg_cityblock"]] = distance.cityblock(query_avg_w2v, title_avg_w2v)
        # all_data[index, name2idx["w2v_avg_minkowski"]] = distance.minkowski(query_avg_w2v, title_avg_w2v)
        # all_data[index, name2idx["w2v_avg_braycurtis"]] = distance.braycurtis(query_avg_w2v, title_avg_w2v)
        # all_data[index, name2idx["w2v_avg_canberra"]] = distance.canberra(query_avg_w2v, title_avg_w2v)
        
        if index == (E4 - 1):
            print("训练集前4亿提取完成!", flush=True)
            break
        if debug and index == 100000 - 1:
            print("debug mode. break")
            break
        
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (index, time_elapsed // 60, time_elapsed % 60))


offset = index + 1
print("test A (offset=%d)..." % offset, flush=True)
since = time.time()
with open(test_A_file_path) as test_A_csvfile:
    csv_reader = csv.reader(test_A_csvfile)
    for index, line in enumerate(csv_reader):
        
        query = line[1].strip()
        title = line[3].strip()
        
        query_avg_w2v = w2v_sent2vec(query, w2v_model)
        title_avg_w2v = w2v_sent2vec(title, w2v_model)
        
        all_data[index + offset, name2idx["w2v_avg_cosine"]] = distance.cosine(query_avg_w2v, title_avg_w2v)
        all_data[index + offset, name2idx["w2v_avg_cityblock"]] = distance.cityblock(query_avg_w2v, title_avg_w2v)
        # all_data[index + offset, name2idx["w2v_avg_minkowski"]] = distance.minkowski(query_avg_w2v, title_avg_w2v)
        # all_data[index + offset, name2idx["w2v_avg_braycurtis"]] = distance.braycurtis(query_avg_w2v, title_avg_w2v)
        # all_data[index + offset, name2idx["w2v_avg_canberra"]] = distance.canberra(query_avg_w2v, title_avg_w2v)
        
        if debug and index == 100000 - 1:
            print("debug mode. break")
            break
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (index, time_elapsed // 60, time_elapsed % 60))


offset = offset + index + 1
print("test B (offset=%d)..." % offset, flush=True)
since = time.time()
with open(test_B_file_path) as test_B_csvfile:
    csv_reader = csv.reader(test_B_csvfile)
    for index, line in enumerate(csv_reader):
        
        query = line[1].strip()
        title = line[3].strip()
        
        query_avg_w2v = w2v_sent2vec(query, w2v_model)
        title_avg_w2v = w2v_sent2vec(title, w2v_model)
        
        all_data[index + offset, name2idx["w2v_avg_cosine"]] = distance.cosine(query_avg_w2v, title_avg_w2v)
        all_data[index + offset, name2idx["w2v_avg_cityblock"]] = distance.cityblock(query_avg_w2v, title_avg_w2v)
        # all_data[index + offset, name2idx["w2v_avg_minkowski"]] = distance.minkowski(query_avg_w2v, title_avg_w2v)
        # all_data[index + offset, name2idx["w2v_avg_braycurtis"]] = distance.braycurtis(query_avg_w2v, title_avg_w2v)
        # all_data[index + offset, name2idx["w2v_avg_canberra"]] = distance.canberra(query_avg_w2v, title_avg_w2v)
        
        if debug and index == 100000 - 1:
            print("debug mode. break")
            break
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (index, time_elapsed // 60, time_elapsed % 60))

print(all_data.shape)
print(all_data[:5, :])
print(all_data[-5:, :])
print(all_data.shape)

if not debug:
    train_save_path = os.path.join(save_dir, 'feature_9_0_4e.csv.gz')
    print("保存4亿训练集到%s..." % train_save_path)
    np.savetxt(train_save_path, all_data[:E4, :], fmt="%f", delimiter=",", 
                header=",".join(fea_names), comments="")
                
    test_A_save_path = os.path.join(save_dir, 'feature_9_0_4e_testA.csv.gz')
    print("保存测试集A到%s..." % test_A_save_path)
    np.savetxt(test_A_save_path, all_data[E4:E4+TEST_A_ROWS, :], fmt="%f", delimiter=",", 
                header=",".join(fea_names), comments="")
    print("save done!")
    
    test_B_save_path = os.path.join(save_dir, 'feature_9_0_4e_testB.csv.gz')
    print("保存测试集B到%s..." % test_B_save_path)
    np.savetxt(test_B_save_path, all_data[E4+TEST_A_ROWS:, :], fmt="%f", delimiter=",", 
                header=",".join(fea_names), comments="")
    print("save done!")

# del(all_data)
gc.collect()


# ## 1.10 全局出现频次相关
# 使用前4亿原始数据加上最后两个测试集计算特征: 特征: ['query_title_click', 'query_nunique_title', 'query_click', 'title_nunique_query', 'title_click']
# 
# * query_nunique_title: 这个query对应多少unique的title
# * title_nunique_query
# * query_click: 所有数据中这个query出现的次数
# * title_click: 所有数据中这个title出现的次数
# * query_title_click: 所有数据中这个query+title出现的次数
# 
# 一下跑内存会炸, 分成两个2亿跑, 下面是后两亿
# 

# In[ ]:


import pandas as pd
import numpy as np
import random
import math
import time
import gc
import os
import csv
import json
from tqdm import tqdm
from multiprocessing import Pool
    
E4 = 400000000

# TRAIN_ROWS = 1000000000
TEST_A_ROWS, TEST_B_ROWS = 20000000, 100000000
save_dir = "/home/kesci/work/feature_4e"
assert os.path.exists(save_dir)

train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_A_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
test_B_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'

train_columns = ['query_id','query','query_title_id','title','label']
test_columns = ['query_id','query','query_title_id','title']
print("当前进程PID:", os.getpid(), "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

debug = False
nrows = None
if debug:
    nrows = 1000000

print("加载train 2e...")
train_df = pd.read_csv(train_file_path, names=train_columns, skiprows = (E4 // 2),
                        nrows=(nrows if debug else (E4 // 2)))[['query','title']]
print(train_df.shape)
                        
print('加载testA...')
test_A_df = pd.read_csv(test_A_file_path, names=test_columns, nrows=nrows)[['query','title']]
test_A_df["title"] = test_A_df["title"].apply(lambda x: x.strip()) # 去掉测试集title最后的tab
print(test_A_df.shape)


print('加载testB...')
test_B_df = pd.read_csv(test_B_file_path, names=test_columns, nrows=nrows)[['query','title']]
test_B_df["title"] = test_B_df["title"].apply(lambda x: x.strip()) # 去掉测试集title最后的tab
print(test_B_df.shape)


all_data=pd.concat((train_df, test_A_df, test_B_df), axis = 0, ignore_index = True, sort = False, copy=False)
del train_df, test_A_df, test_B_df
gc.collect()
print('finish concat data')

print(all_data.info())
print(all_data.head())


print("------------------- 提取 nunique 特征 ----------------------")
since = time.time()
print("-- query_title_click...")
all_data['query_title_click'] = all_data.groupby(['query', 'title']).query.transform('count').astype("int32")

print("-- group by query:")
print("---- group...")
query_gb = all_data.groupby('query').title
print("---- query_nunique_title...")
all_data['query_nunique_title'] = query_gb.transform('nunique').astype("int32")
print("---- query_click...")
all_data['query_click'] = query_gb.transform('count').astype("int32")
del query_gb
gc.collect()


print("-- group by title:")
print("---- group...")
title_gb = all_data.groupby('title').query
print("---- title_nunique_query...")
all_data['title_nunique_query'] = title_gb.transform('nunique').astype("int32")
print("---- title_click...")
all_data['title_click'] = title_gb.transform('count').astype("int32")
del title_gb
gc.collect()

print(all_data.head())
print(all_data.tail())
print(all_data.info())

if not debug:
    PART = E4 // 2
    feature_names = [c for c in all_data.columns if c not in ['query', 'title']]
    
    train_save_path = os.path.join(save_dir, 'feature_10_1th_2e.csv.gz')
    # 注意loc的索引是左闭右闭的!!!
    print("保存训练集到%s..." % train_save_path)
    all_data.loc[:PART-1, feature_names].to_csv(train_save_path, compression='gzip', index=False)
    
    
    test_A_save_path = os.path.join(save_dir, 'feature_10_1th_2e_testA.csv.gz')
    print("保存测试集A到%s..." % test_A_save_path)
    all_data.loc[PART:PART+TEST_A_ROWS - 1, feature_names].to_csv(test_A_save_path, compression='gzip', index=False)
    
    test_B_save_path = os.path.join(save_dir, 'feature_10_1th_2e_testB.csv.gz')
    print("保存测试集B到%s..." % test_B_save_path)
    all_data.loc[PART+TEST_A_ROWS:, feature_names].to_csv(test_B_save_path, compression='gzip', index=False)
    
    print("save done!")
    del all_data
    gc.collect()


# ## 1.11 hc 补充特征 (4e_train + 2kw_test + 1e_test_final) 8 个特征
# (jaccard_similarity,
# qt_coword_query_ratio,
# qt_coword_title_ratio,
# qt_len_mean,
# qt_common_word_acc,
# ngram_query_title_precision,
# ngram_query_title_recall,
# ngram_query_title_acc)
# 
# 

# In[ ]:


import pandas as pd 
import numpy as np
from tqdm import tqdm as tqdm 
import csv, json
import os, gc, time, math
import gensim
from gensim.models import Word2Vec


save_dir = "/home/kesci/work/feature"
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
testA_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
testB_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'
assert os.path.exists(save_dir)
TRAIN_ROWS, TESTA_ROWS, TESTB_ROWS=400000000, 20000000, 100000000
train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']

# ----- 提取特征 -----
since = time.time()
fea_names = ["jaccard_similarity", "qt_coword_query_ratio", "qt_coword_title_ratio", 
             "qt_len_mean", "qt_common_word_acc", 
             "ngram_query_title_precision", "ngram_query_title_recall", "ngram_query_title_acc"]
print("一共%d个特征:" % len(fea_names), fea_names)
name2idx = {fea_name:fea_names.index(fea_name) for fea_name in fea_names}

# 训练集4e 
def get_ngram_rp_query_in_title(query, title):
    query = list(query.strip().split())
    title = list(title.strip().split())
    query_2gram = []
    for i in range(len(query) - 1):
        query_2gram.append(query[i]+query[i+1])
    query.extend(query_2gram)
    
    title_2gram = []
    for i in range(len(title) - 1):
        title_2gram.append(title[i]+title[i+1])
    title.extend(title_2gram)
    
    len_query = len(query)
    len_title = len(title)
    len_common = len(set(query) & set(title))
    
    recall = len_common / (len_query + 0.001)
    precision = len_common / (len_title + 0.001)
    acc = len_common / (len_query + len_title - len_common)
    return [recall, precision, acc]

with open(train_file_path) as train_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(train_csvfile))):
        query_set = set(line[1].strip().split())
        title_set = set(line[3].strip().split())
        common_words_len = len(query_set&title_set)
        query_len = len(line[1].strip().split())
        title_len = len(line[3].strip().split())
        recall, precision, acc = get_ngram_rp_query_in_title(line[1], line[3])
        
        all_data[idx, name2idx["jaccard_similarity"]] = common_words_len/len(query_set|title_set)
        all_data[idx, name2idx["qt_coword_query_ratio"]] = common_words_len/query_len
        all_data[idx, name2idx["qt_coword_title_ratio"]] = common_words_len/title_len
        all_data[idx, name2idx["qt_len_mean"]] = (query_len + title_len)/2.0
        all_data[idx, name2idx["qt_common_word_acc"]]=common_words_len/(query_len+title_len-common_words_len)
        all_data[idx, name2idx["ngram_query_title_precision"]] = precision
        all_data[idx, name2idx["ngram_query_title_recall"]] = recall
        all_data[idx, name2idx["ngram_query_title_acc"]]= acc
        if idx == TRAIN_ROWS-1: break

time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))
print("saving...")
save_path = os.path.join(save_dir, 'feature_train_4e_hc_add.csv.gz')
np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
print(save_path, "save done!")
del (all_data)
gc.collect()

# 测试集2kw
all_data = np.zeros((TESTA_ROWS, len(fea_names)), dtype=np.float32)
with open(testA_file_path) as test_csvfile:
    for idx, line in tqdm(enumerate(csv.reader(test_csvfile))):
        query_set = set(line[1].strip().split())
        title_set = set(line[3].strip().split())
        common_words_len = len(query_set&title_set)
        query_len = len(line[1].strip().split())
        title_len = len(line[3].strip().split())
        recall, precision, acc = get_ngram_rp_query_in_title(line[1], line[3])
        
        all_data[idx, name2idx["jaccard_similarity"]] = common_words_len/len(query_set|title_set)
        all_data[idx, name2idx["qt_coword_query_ratio"]] = common_words_len/query_len
        all_data[idx, name2idx["qt_coword_title_ratio"]] = common_words_len/title_len
        all_data[idx, name2idx["qt_len_mean"]] = (query_len + title_len)/2.0
        all_data[idx, name2idx["qt_common_word_acc"]] = common_words_len/(query_len+title_len-common_words_len)
        
        all_data[idx, name2idx["ngram_query_title_precision"]] = precision
        all_data[idx, name2idx["ngram_query_title_recall"]] = recall
        all_data[idx, name2idx["ngram_query_title_acc"]]= acc
time_elapsed = time.time() - since
print('complete with idx %d in %d min %d s.' % (idx, time_elapsed // 60, time_elapsed % 60))
print("saving...")
save_path = os.path.join(save_dir, 'feature_test_2kw_hc_add.csv.gz')
np.savetxt(save_path, all_data, fmt="%f", delimiter=",", header=",".join(fea_names), comments="")
print(save_path, "save done!")
del (all_data)
gc.collect()


# # 2 word2vec模型训练
# 语料库包括前4亿训练数据加上两个测试集, 下面使用300维的词向量（实际还训练了一个100d的）

# In[ ]:


import numpy as np
import pandas as pd
import time
import os
import csv
import gensim
from itertools import chain
from tqdm import tqdm
import logging
logging.basicConfig(format="%(levelname)s-%(asctime)s:%(message)s", datefmt='%H:%M:%S', level=logging.INFO)

E4 = 400000000
# TRAIN_ROWS = 1000000000
TEST_A_ROWS, TEST_B_ROWS = 20000000, 100000000

train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_A_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
test_B_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'

savedir = "/home/kesci/work/word2vec"
save_name = "w2v_300_cbow_4e.model"

vector_size=300

debug=False
nrows = None
if debug:
    nrows=5000000
    
print("当前进程PID:", os.getpid(), "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

def csv_generator(csv_file_path, have_header=False, num=None):
    if num is None:
        num = -1
    with open(csv_file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        if have_header:
            print("跳过表头!!!")
            next(csv_reader) # 跳过表头
        count = 0
        for line in csv_reader:
            if count == num:
                return 
            count += 1
            yield line

def query_generator(csv_file_path, query_idx=1, have_header=False, num=None):
    for line in csv_generator(csv_file_path, have_header=have_header, num=num):
        yield line[query_idx].strip().split() # strip去掉测试集里每个title最后的tab
        
def title_generator(csv_file_path, title_idx=3, have_header=False, num=None):
    for line in csv_generator(csv_file_path, have_header=have_header, num=num):
        yield line[title_idx].strip().split()
        
        
def query_title_generator(train_nrows, test_nrows):
    return chain(query_generator(train_file_path, have_header=False, num=train_nrows), 
                title_generator(train_file_path, have_header=False, num=train_nrows),
                query_generator(test_A_file_path, have_header=False, num=test_nrows), 
                title_generator(test_A_file_path, have_header=False, num=test_nrows),
                query_generator(test_B_file_path, have_header=False, num=test_nrows), 
                title_generator(test_B_file_path, have_header=False, num=test_nrows))

class SentenceIterator(object):
    """语料库生成器, 包含整个训练集和测试集"""
    def __init__(self, train_nrows, test_nrows):
        self.train_nrows = train_nrows
        self.test_nrows = test_nrows
        if not debug:
            assert test_nrows == None
    def __iter__(self):
        for sentence in query_title_generator(self.train_nrows, self.test_nrows):
            yield sentence
            
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        
    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" % 
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss >= epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)

        self.pre_loss = cum_loss
        self.since = time.time()
        print("\n\n\n")
        
        
# 将整个过程分成三步
# 1, 构建模型(不训练)
model_word2vec = gensim.models.Word2Vec(min_count=5, 
                                        window=5, 
                                        sg=0, # cbow
                                        size=vector_size,
                                        workers=8,
                                        seed=2019,
                                        batch_words=100000)
# 2, 遍历一遍语料库
since = time.time()
sentences = SentenceIterator((nrows if debug else E4), nrows)
model_word2vec.build_vocab(sentences, progress_per=(500000 if debug else 50000000))
time_elapsed = time.time() - since
print('Time to build vocab: {:.0f}min {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

# 3, 训练
model_word2vec.train(sentences, total_examples=model_word2vec.corpus_count, 
                        epochs=40, compute_loss=True, 
                        report_delay=(10 if debug else 60*10), # 每隔10分钟输出一下日志
                        callbacks=[EpochSaver(savedir, save_name)])


# # 3 lgb
# ## 3.1 训练
# * 使用的特征包括前面提到的10种特征加上100维的的word2vec平均句向量直接作差特征。
# * 四千万跑一次（后20%作为验证），4亿数据一共跑10次
# * 注意特征10是分成了两个2亿，所以前两亿和后两亿的代码略有不同，下面的代码是前2亿的lgb训练

# In[ ]:


# lightgbm train and predict
import pandas as pd
import numpy as np
import random
import time
import gc
from scipy import sparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import paired_distances
from sklearn import metrics
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
# from glove import Glove
import pickle
# from sklearn.externals import joblib
import os
import csv
import psutil


import lightgbm as lgb
import math

if not os.path.isdir('lgb_model_result'):
    os.mkdir('lgb_model_result')

TEST_A_ROWS, TEST_B_ROWS = 20000000, 100000000
E2 = 200000000
chunksize=40000000 #每次4kw训练数据
assert E2 % chunksize == 0
print("只用前4亿训练数据!")
iter_num = int(E2 // chunksize)

debug = False
if debug:
    chunksize = 1000000
    iter_num = 2


train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_A_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
test_B_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'

train_columns = ['query_id','query','query_title_id','title','label']
test_columns = ['query_id','query','query_title_id','title']
print("当前进程PID:", os.getpid(), "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


# 这个文件存放了训练集label
train_label_file = "/home/kesci/work/feature/train_label.csv.gz"
save_dir = "/home/kesci/work/feature"

# 之前提取好的训练集特征文件
files = [
    "/home/kesci/work/feature/feature_1_0.csv.gz",
    "/home/kesci/work/feature/feature_2.csv.gz",
    "/home/kesci/work/feature/feature_3.csv.gz",
    "/home/kesci/work/feature/feature_4.csv.gz",
    "/home/kesci/work/feature/feature_5.csv.gz",
    "/home/kesci/work/feature/feature_6.csv.gz",
    "/home/kesci/work/feature/feature_7.csv.gz",
    
    "/home/kesci/work/feature_4e/feature_1_1_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_0_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_1_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_0_300d_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_1_300d_4e.csv.gz",
    
    # cvr
    "/home/kesci/work/feature_4e/feature_8_4e.csv.gz",
 
    "/home/kesci/work/feature_4e/feature_10_0th_2e.csv.gz", # 0-2亿
    # "/home/kesci/work/feature_4e/feature_10_1th_2e.csv.gz", # 2-4亿
    
]


used_features = [
# feature 1_0
'query_length','title_length','WordMatchShare','WordMatchShare_query',
'WordMatchShare_title',
'LengthDiff', 'LengthDiffRate', 'LengthRatio_qt', 'LengthRatio_tq', # 这四个根据前面的计算得到

# feature 1_1
'TFIDFWordMatchShare','TFIDFWordMatchShare_query','TFIDFWordMatchShare_title',


# feature 2
'NgramJaccardCoef_1', 'NgramJaccardCoef_2', 'NgramJaccardCoef_3', 'NgramJaccardCoef_4',

# feature 3
'Levenshtein_ratio', 'Levenshtein_distance_char','query_title_common_words', 'common_word_ratio',

# feature 4
'lcsubstr_len', 'lcseque_len','longest_match_size', 'longest_match_ratio',

# feature 5
'fuzz_qratio', 'fuzz_partial_ratio',

#feature 6
'fuzz_partial_token_sort_ratio','fuzz_token_set_ratio','fuzz_token_sort_ratio',

#feature 7
'query_Entropy','title_Entropy','query_title_Entropy','WordMatchShare_Entropy',

#feature 8
"query_convert", "title_convert", "query_title_convert",

# fature 9_0
'w2v_avg_cosine', 'w2v_avg_cityblock',

# feature 9_1
'w2v_avg_minkowski', 'w2v_avg_braycurtis', 'w2v_avg_canberra',

# fature 9_0 300d
'w2v300_avg_cosine', 'w2v300_avg_cityblock',

# feature 9_1 300d
'w2v300_avg_minkowski', 'w2v300_avg_braycurtis', 'w2v300_avg_canberra',

# feature 10
'query_title_click', 'query_nunique_title', 'query_click', 'title_nunique_query', 'title_click',

]

# gensim feature
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        
    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" % 
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)
            
        self.pre_loss = cum_loss
        self.since = time.time()
        print("\n\n\n")


def query_title_generator(csv_file_path, skiprows=0, have_header=False, max_rows=None):
    if max_rows is None:
        max_rows = -1
    with open(csv_file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        if have_header:
            print("跳过表头!!!")
            next(csv_reader) # 跳过表头
        for _ in range(skiprows):
            next(csv_reader)
        
        count = 0
        for line in csv_reader:
            if count == max_rows:
                return
            count += 1
            yield line[1].strip().split(), line[3].strip().split() # query, title

def w2v_sent2vec(words, model):
    """计算句子的平均word2vec向量, sentences是一个句子, 句向量最后会归一化"""
    M = []
    for word in words:
        try:
            M.append(model.wv[word])
        except KeyError: # 不在词典里
            continue
    
    M = np.array(M)
    v = M.sum(axis=0)
    return (v / np.sqrt((v ** 2).sum())).astype(np.float32)


lgb_params = {
    "objective": 'binary',
    "num_leaves": 128, # 32, # 128, # 30, # 50, #64, 
    "max_depth": -1, #7, #7, #7, # 5, #-1,
    "learning_rate": 0.05, # 0.02, 
    # "n_estimators": 100000,
#   "subsample_for_bin"=200000, 
    "min_child_samples": 20,
    "min_child_weight": 0.001, 
    "min_split_gain": 0.0, 
            
    "subsample": 0.7,#0.5,    # 构造每棵树使用的样本率
    "subsample_freq": 1, 
    "colsample_bytree": 0.7, # 构造每棵树使用的特征率
            
    "reg_alpha": 2,# 5, #0, #10, 
    "reg_lambda": 2,# 5, # 0, #10,          

    "n_jobs": 12, # -1, 
    "random_state": 2018
}
print("n jobs:", lgb_params["n_jobs"])
print("lgb学习率: ", lgb_params["learning_rate"])
print("lgb num_leaves: ",  lgb_params["num_leaves"])
print("lgb max_depth: ",  lgb_params["max_depth"])

def load_feature_csv(csv_file, start_row=0, nrows=chunksize, drop=True):
    """如果是读取特征文件那么drop=True以丢弃不用的feature, 否则应该设成False, 例如读id文件"""
    since = time.time()
    base_feature_df = pd.read_csv(csv_file, dtype="float32", names=None, 
                                skiprows=start_row, nrows=nrows)
    df_len = base_feature_df.shape[0]
    print('load %d rows from %s(start from row %d)'                         % (df_len, csv_file.split("/")[-1], start_row), end=" ")
                        
    columns = pd.read_csv(csv_file, dtype="float32", nrows=1).columns
    base_feature_df.columns = columns
    print(columns.values)    
    unused_features = [fea for fea in columns if fea not in used_features]
    if drop and len(unused_features) != 0:
        print("\n丢弃掉feature:", unused_features, end="")
        base_feature_df = base_feature_df.drop(unused_features, axis=1)
    time_elapsed = time.time() - since
    print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    return base_feature_df
    

# ---- 按顺序记录特征名(必须和后面读取特征的顺序保持一致！！！) ------
feature_names = []
# word2vec 直接作差特征
feature_names.extend(["w2v_100_diff_%d"%i for i in range(100)])
print("特征文件:")
for f in files:
    print(f)
    columns = pd.read_csv(f, nrows=1).columns
    feature_names.extend([fea for fea in columns if fea in used_features])


print("一共有%d个特征:" % len(feature_names))
print(feature_names)

for i in range(iter_num):
    gbm_booster = None
    print(" +++++++++++++++ 开始%d/%d ++++++++++++++" % (i+1, iter_num))
    since = time.time()
    
    # 提前分配所有特征的内存
    train_fea = np.empty((chunksize, len(feature_names)), dtype=np.float32)
    start_dim, end_dim = 0, 0

    print("提取word2vec直接做差特征...")
    w2v_model_path = '/home/kesci/work/word2vec/w2v_100_cbow_4e.model'
    w2v_model = Word2Vec.load(w2v_model_path)
    embed_size = w2v_model.trainables.layer1_size
    print("word2vec加载完毕. 总词典数: %d, embeding size: %d" % (len(w2v_model.wv.vocab), embed_size))
    end_dim = embed_size
    print("train:")
    for idx, (query_words, title_words) in tqdm(enumerate(query_title_generator(train_file_path, 
                                        skiprows=chunksize * i, have_header=False, 
                                        max_rows=chunksize))):
        train_fea[idx, start_dim:end_dim] = w2v_sent2vec(query_words, w2v_model)-w2v_sent2vec(title_words, w2v_model)
    
    for file in files:
        # train feature
        print("train:", end="")
        fea = load_feature_csv(file, start_row = i * chunksize, 
                                nrows = chunksize).values.astype(np.float32)
        start_dim = end_dim
        end_dim = end_dim + fea.shape[1]
        train_fea[:, start_dim:end_dim] = fea
        del fea
        
    gc.collect()
    
    assert end_dim == len(feature_names)
    
    print("数值特征加载完毕, train shape:", train_fea.shape)
    label = load_feature_csv(train_label_file, start_row = i * chunksize, 
                                nrows = chunksize, 
                                drop=False)["label"].values.astype(np.int32)
    
    print("spliting dataset...", flush=True)
    valid_num = int(chunksize * 0.2)
    train_X, train_y = train_fea[:-valid_num, :], label[:-valid_num]
    valid_X, valid_y = train_fea[-valid_num:, :], label[-valid_num:]
    del train_fea, label
    
    gc.collect()
    print("构造lgb Dataset...", flush=True)
    lgb_train_set = lgb.Dataset(train_X, train_y) # , categorical_feature=["title_labelencoder"])
    lgb_valid_set = lgb.Dataset(valid_X, valid_y, # categorical_feature=["title_labelencoder"],
                                reference=lgb_train_set)
    gc.collect()

    gbm_booster = lgb.train(lgb_params,
                            lgb_train_set,
                            num_boost_round=4000,
                            valid_sets=[lgb_train_set, lgb_valid_set],
                            valid_names=["tn", "vd"],
                            init_model=None, # 若不为None就在上次的基础上接着训练
                            # feature_name=x_cols,
                            early_stopping_rounds=40,
                            # feval=feval_auc,
                            verbose_eval=20,
                            # learning_rates=lambda iter: max(0.01, lgb_params["learning_rate"]*(0.9988**iter)),  # 学习率衰减
                            ) # 不使用增量训练
    time_elapsed = time.time() - since
    print('lgb训练结束. complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    del lgb_train_set, lgb_valid_set, train_X, train_y, valid_X, valid_y
    
    if not debug:
        save_name_prefix = "/home/kesci/work/lgb_model_result/805_tss_lgb_noCVR_%d" % i
        gbm_booster.save_model(save_name_prefix + '_booster.txt')
        print("Model save done!")
        
        fea_impt = gbm_booster.feature_importance()
        fea_impt_df = pd.DataFrame(data={"feature": feature_names, "importance": fea_impt})
        fea_impt_df.to_csv(save_name_prefix + '_fea_impt.csv', index=False)
        print("feature importances df save done.")
    
    del gbm_booster
    gc.collect()


# ## 3.2 测试

# In[ ]:


import pandas as pd
import numpy as np
import random
import time
import gc
from scipy import sparse
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import paired_distances
from sklearn import metrics
import gensim
from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
# from glove import Glove
import pickle
# from sklearn.externals import joblib
import os
import csv
import psutil


import lightgbm as lgb
import math

if not os.path.isdir('lgb_model_result'):
    os.mkdir('lgb_model_result')

TEST_A_ROWS, TEST_B_ROWS = 20000000, 100000000


chunksize=None
debug = False
if debug:
    chunksize = 100000
    iter_num = 2


train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_A_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
test_B_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'

train_columns = ['query_id','query','query_title_id','title','label']
test_columns = ['query_id','query','query_title_id','title']
print("当前进程PID:", os.getpid(), "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


# 这个文件存放了训练集label
train_label_file = "/home/kesci/work/feature/train_label.csv.gz"
save_dir = "/home/kesci/work/feature"

# files = [os.path.join(save_dir, 'base_feature_v2_all_%d.csv'%i) for i in range(5)]
files = [
    "/home/kesci/work/feature/feature_1_0.csv.gz",
    "/home/kesci/work/feature/feature_2.csv.gz",
    "/home/kesci/work/feature/feature_3.csv.gz",
    "/home/kesci/work/feature/feature_4.csv.gz",
    "/home/kesci/work/feature/feature_5.csv.gz",
    "/home/kesci/work/feature/feature_6.csv.gz",
    "/home/kesci/work/feature/feature_7.csv.gz",
    
    "/home/kesci/work/feature_4e/feature_1_1_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_0_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_1_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_0_300d_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_1_300d_4e.csv.gz",
    # cvr
    "/home/kesci/work/feature_4e/feature_8_4e.csv.gz",
 
    "/home/kesci/work/feature_4e/feature_10_0th_2e.csv.gz", # 0-2亿
    # "/home/kesci/work/feature_4e/feature_10_1th_2e.csv.gz", # 2-4亿
]

used_features = [
# feature 1_0
'query_length','title_length','WordMatchShare','WordMatchShare_query',
'WordMatchShare_title',
'LengthDiff', 'LengthDiffRate', 'LengthRatio_qt', 'LengthRatio_tq', # 这四个根据前面的计算得到

# feature 1_1
'TFIDFWordMatchShare','TFIDFWordMatchShare_query','TFIDFWordMatchShare_title',


# feature 2
'NgramJaccardCoef_1', 'NgramJaccardCoef_2', 'NgramJaccardCoef_3', 'NgramJaccardCoef_4',

# feature 3
'Levenshtein_ratio', 'Levenshtein_distance_char','query_title_common_words', 'common_word_ratio',

# feature 4
'lcsubstr_len', 'lcseque_len','longest_match_size', 'longest_match_ratio',

# feature 5
'fuzz_qratio', 'fuzz_partial_ratio',

#feature 6
'fuzz_partial_token_sort_ratio','fuzz_token_set_ratio','fuzz_token_sort_ratio',

#feature 7
'query_Entropy','title_Entropy','query_title_Entropy','WordMatchShare_Entropy',

#feature 8
"query_convert", "title_convert", "query_title_convert",

# fature 9_0
'w2v_avg_cosine', 'w2v_avg_cityblock',

# feature 9_1
'w2v_avg_minkowski', 'w2v_avg_braycurtis', 'w2v_avg_canberra',

# fature 9_0 300d
'w2v300_avg_cosine', 'w2v300_avg_cityblock',

# feature 9_1 300d
'w2v300_avg_minkowski', 'w2v300_avg_braycurtis', 'w2v300_avg_canberra',

# feature 10
'query_title_click', 'query_nunique_title', 'query_click', 'title_nunique_query', 'title_click',
]

# gensim feature
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        
    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" % 
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)
            
        self.pre_loss = cum_loss
        self.since = time.time()
        print("\n\n\n")

def query_title_generator(csv_file_path, skiprows=0, have_header=False, max_rows=None):
    if max_rows is None:
        max_rows = -1
    with open(csv_file_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        if have_header:
            print("跳过表头!!!")
            next(csv_reader) # 跳过表头
        for _ in range(skiprows):
            next(csv_reader)
        
        count = 0
        for line in csv_reader:
            if count == max_rows:
                return
            count += 1
            yield line[1].strip().split(), line[3].strip().split() # query, title

def w2v_sent2vec(words, model):
    """计算句子的平均word2vec向量, sentences是一个句子, 句向量最后会归一化"""
    M = []
    for word in words:
        try:
            M.append(model.wv[word])
        except KeyError: # 不在词典里
            continue
    
    M = np.array(M)
    v = M.sum(axis=0)
    return (v / np.sqrt((v ** 2).sum())).astype(np.float32)

def load_feature_csv(csv_file, start_row=0, nrows=chunksize, drop=True):
    """如果是读取特征文件那么drop=True以丢弃不用的feature, 否则应该设成False, 例如读id文件"""
    since = time.time()
    base_feature_df = pd.read_csv(csv_file, dtype="float32", names=None, 
                                skiprows=start_row, nrows=nrows)
    df_len = base_feature_df.shape[0]
    print('load %d rows from %s(start from row %d)'                         % (df_len, csv_file.split("/")[-1], start_row), end=" ")
                        
    columns = pd.read_csv(csv_file, dtype="float32", nrows=1).columns
    base_feature_df.columns = columns
    print(columns.values)    
    unused_features = [fea for fea in columns if fea not in used_features]
    if drop and len(unused_features) != 0:
        print("\n丢弃掉feature:", unused_features, end="")
        base_feature_df = base_feature_df.drop(unused_features, axis=1)
    time_elapsed = time.time() - since
    print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)) # 打印出来时间
    return base_feature_df
    

feature_num = 151
print("一共有%d个特征(一定要和训练保持一致!!!)" % feature_num)


print(" +++++++++++++++ 开始提取特征(一定要和训练保持完全一致) ++++++++++++++")
since = time.time()

print("---- 提取word2vec直接做差特征...")
w2v_model_path = '/home/kesci/work/word2vec/w2v_100_cbow_4e.model'
w2v_model = Word2Vec.load(w2v_model_path)
embed_size = w2v_model.trainables.layer1_size
print("word2vec加载完毕. 总词典数: %d, embeding size: %d" % (len(w2v_model.wv.vocab), embed_size))

print("---- 提取其他特征:")
if debug:
    TEST_A_ROWS, TEST_B_ROWS = 100000, 100000
test_A_fea = np.zeros((TEST_A_ROWS, feature_num), dtype=np.float32)
test_B_fea = np.zeros((TEST_B_ROWS, feature_num), dtype=np.float32)
start_dim, end_dim = 0, 0

print("word2vec距离")
print("-- test A:")
end_dim = embed_size
for idx, (query_words, title_words) in tqdm(enumerate(query_title_generator(test_A_file_path, 
                                    skiprows=0, have_header=False, 
                                    max_rows=TEST_A_ROWS))):
    test_A_fea[idx, start_dim:end_dim] = w2v_sent2vec(query_words, w2v_model)                         - w2v_sent2vec(title_words, w2v_model)
print('complete with idx %d.' % idx)
print("-- test B:")                      
for idx, (query_words, title_words) in tqdm(enumerate(query_title_generator(test_B_file_path, 
                                    skiprows=0, have_header=False, 
                                    max_rows=TEST_B_ROWS))):
    test_B_fea[idx, start_dim:end_dim] = w2v_sent2vec(query_words, w2v_model)                         - w2v_sent2vec(title_words, w2v_model)
print('complete with idx %d.' % idx)
                        
del w2v_model
gc.collect()
                        
print("其他特征(包括前两亿的特征10):")
feature10_first2e_tag = True
for file in files:
    fea_A = load_feature_csv(file.replace(".csv", "_testA.csv"), start_row = 0, 
                            nrows = TEST_A_ROWS).values.astype(np.float32)
    start_dim = end_dim
    end_dim = end_dim + fea_A.shape[1]
    test_A_fea[:, start_dim:end_dim] = fea_A
    

    fea_B = load_feature_csv(file.replace(".csv", "_testB.csv"), start_row = 0, 
                            nrows = TEST_B_ROWS).values.astype(np.float32)
    test_B_fea[:, start_dim:end_dim] = fea_B  
    del fea_A, fea_B
    
assert end_dim == feature_num


for chunk_idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    save_name_prefix = "/home/kesci/work/lgb_model_result/805_tss_lgb_" + str(chunk_idx)
    print("chunk_idx =", chunk_idx)
    if chunk_idx < 5:
        assert feature10_first2e_tag
        print("注意feature10使用的是前2亿的!!!!")
    elif chunk_idx >= 5:
        if feature10_first2e_tag:
            print("提取后两亿对应的测试集的feature10...")
            fea10_file = "/home/kesci/work/feature_4e/feature_10_1th_2e.csv.gz"
            fea_A = load_feature_csv(fea10_file.replace(".csv", "_testA.csv"), start_row = 0, 
                            nrows = TEST_A_ROWS).values.astype(np.float32)
            test_A_fea[:, feature_num - 5:] = fea_A # feature10一共有5个特征
            
            fea_B = load_feature_csv(fea10_file.replace(".csv", "_testB.csv"), start_row = 0, 
                                    nrows = TEST_B_ROWS).values.astype(np.float32)
            test_B_fea[:, feature_num - 5:] = fea_B    
            del fea_A, fea_B
            feature10_first2e_tag = False
        print("注意feature10使用的是后2亿的!!!!")

    model_file = save_name_prefix + '_booster.txt'
    print("从%s加载lgb模型数据..." % model_file)
    gbm_booster = lgb.Booster(model_file = model_file)
    
    test_A_pred = gbm_booster.predict(test_A_fea)
    test_B_pred = gbm_booster.predict(test_B_fea)
    result_A_df = pd.DataFrame({"prediction": test_A_pred}, dtype=np.float32)
    result_B_df = pd.DataFrame({"prediction": test_B_pred}, dtype=np.float32)
    print("test A:")
    print(result_A_df.shape)
    print(result_A_df.head(10))
    print("testB:")
    print(result_B_df.shape)
    print(result_B_df.head(50))
    del test_A_pred
    del test_B_pred, gbm_booster
        
    if not debug:
        result_A_df.to_csv(save_name_prefix + "_testA.csv.gz", compression='gzip', index=False, header=False)
        print("testA Save done!")
        result_B_df.to_csv(save_name_prefix + "_testB.csv.gz", compression='gzip', index=False, header=False)
        print("TestB Save Done!")
    
    del result_A_df
    del result_B_df
    gc.collect()
del test_A_fea
del test_B_fea


# # 4 ESIM加外部特征(pytorch)
# * 外部特征使用了除转换率外所有特征
# * 下面代码使用的是100d的词向量，实际还用了300d的
# * 和lgb一样，由于特征10是分成了两个2亿，所以前两亿和后两亿的代码略有不同，下面的代码是前2亿的

# In[ ]:


import pandas as pd
import numpy as np
import random
import time
import gc
import os
import csv
import json
import math
import gensim
from itertools import chain
from tqdm import tqdm
from torch import nn
import torch
import torch.nn.functional as F
from gensim.models import Word2Vec

train_columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
test_columns = ['query_id', 'query', 'query_title_id', 'title']

TEST_A_ROWS, TEST_B_ROWS = 20000000, 100000000
train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_A_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'
test_B_file_path = '/home/kesci/input/bytedance/bytedance_contest.final_2.csv'

print("当前进程PID:", os.getpid(), 
        "开始时间:", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


args = {
    "w2v_model_path": "/home/kesci/work/word2vec/w2v_100_cbow_4e.model",
    "model_save_dir": "/home/kesci/work/tss_nn/tss_esim100_811",
    
    "dropout": 0.5,
    "hidden_size": 128,
    "linear_size": 256
}

# model_save_path = "/home/kesci/work/nn_related/tss_rnn_727.pth"
device = torch.device('cuda')
# device = torch.device('cpu')

MAX_QUERY_LEN = 10
MAX_TITLE_LEN = 20
PAD, OOV = 0, 1

E2 = 200000000  # 2亿
chunksize = E2
# assert E2 % chunksize == 0

debug = False
if debug:
    chunksize = 1000000
    chunks = 2

feature_files = [
    "/home/kesci/work/feature/feature_1_0.csv.gz",
    "/home/kesci/work/feature/feature_2.csv.gz",
    "/home/kesci/work/feature/feature_3.csv.gz",
    "/home/kesci/work/feature/feature_4.csv.gz",
    "/home/kesci/work/feature/feature_5.csv.gz",
    "/home/kesci/work/feature/feature_6.csv.gz",
    "/home/kesci/work/feature/feature_7.csv.gz",
    
    "/home/kesci/work/feature_4e/feature_1_1_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_0_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_1_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_0_300d_4e.csv.gz",
    "/home/kesci/work/feature_4e/feature_9_1_300d_4e.csv.gz",
    
    # "/home/kesci/work/feature_4e/feature_10_1th_2e.csv.gz", # 0-2亿
    "/home/kesci/work/feature_4e/feature_10_0th_2e.csv.gz", # 0-2亿

]
print("注意feature10使用的是前2亿的!!!!")

fea_names = []
for file in feature_files:
    fea_names.extend(pd.read_csv(file, nrows=2).columns)
print("一共%d个外部特征:" % len(fea_names), fea_names)


class ESIM(nn.Module):
    def __init__(self, manual_feature_dim, vocab_size, embed_dim, hidden_size, linear_size, dropout=0.5):
        super(ESIM, self).__init__()
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.embeds = nn.Embedding(vocab_size, self.embed_dim)
        # self.bn_embeds = nn.BatchNorm1d(self.embed_dim, momentum=0.01)
        self.lstm1 = nn.LSTM(self.embed_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True)
        
        
        self.fc = nn.Sequential(
            # nn.BatchNorm1d(self.hidden_size * 8 + manual_feature_dim, momentum=0.01),
            nn.Linear(self.hidden_size * 8 + manual_feature_dim, linear_size),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(linear_size, momentum=0.01),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, linear_size),
            nn.ELU(inplace=True),
            # nn.BatchNorm1d(linear_size, momentum=0.01),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, 1),
            nn.Sigmoid()
        )
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, float('-inf'))
        mask2 = mask2.float().masked_fill_(mask2, float('-inf'))

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, sent1, sent2, manual_feature):
        # batch_size * seq_len
        mask1, mask2 = sent1.eq(PAD), sent2.eq(PAD)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.embeds(sent1)
        x2 = self.embeds(sent2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep, manual_feature], -1)
        similarity = self.fc(x)
        return similarity
        
        
# ######################################### 2. 加载数据 ##########################################
def load_feature_csv(csv_file, start_row, nrows):
    base_feature_df = pd.read_csv(csv_file, dtype="float32", names=None, 
                                skiprows=start_row, nrows=nrows)
    df_len = base_feature_df.shape[0]
    print('load %d rows from %s(start from row %d)'                         % (df_len, csv_file.split("/")[-1], start_row), flush=True)

    return base_feature_df
    
class myDataloader(object):
    def __init__(self, file_path, fea_file_path_list, stoi, bigbatch,
                    skiprows = 0, nrows=None,  batch_size=128, shuffle=True, columns=train_columns):
        self.file_path = file_path
        self.fea_file_path_list = fea_file_path_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stoi = stoi
        self.bigbatch = bigbatch
        self.skiprows = skiprows
        self.nrows = nrows
        self.columns = columns
        self.feature_num = self._get_feature_num()


    def next_batch(self):
        """
        先按照顺序读取一个大bigbatch进内存, 再随机从大bigbatch里取一个batch(最后一个batch可能不够batch_size)
        """
        print("从%s加载原始数据..." % self.file_path)
        df_iter = pd.read_csv(self.file_path, nrows=self.nrows, header=None, 
                                skiprows=self.skiprows, chunksize=self.bigbatch)
        count = 0
        while True:
            try:
                df = df_iter.get_chunk()
            except StopIteration:
                return
            df.columns = self.columns

            if "label" in self.columns:
                label = df['label'].values.astype(np.int32)
            else:
                label = -1 * np.zeros(df.shape[0], dtype=np.int32)
            query_lens = []
            title_lens = []
            query_array = np.empty((df.shape[0], MAX_QUERY_LEN), dtype=np.int32)
            title_array = np.empty((df.shape[0], MAX_TITLE_LEN), dtype=np.int32)

            for i, query in enumerate(df["query"].apply(lambda x: x.strip().split())):
                query = [self.stoi.get(w, OOV) for w in query]
                if len(query) > MAX_QUERY_LEN:
                    query = query[:MAX_QUERY_LEN]
                q_len = len(query)
                query_lens.append(q_len)
                if q_len < MAX_QUERY_LEN:
                    query = query + [PAD]*(MAX_QUERY_LEN - q_len)
                query_array[i, :] = np.array(query)

            for i, title in enumerate(df["title"].apply(lambda x: x.strip().split())):
                title = [self.stoi.get(w, OOV) for w in title]
                if len(title) > MAX_TITLE_LEN:
                    title = title[:MAX_TITLE_LEN]
                t_len = len(title)
                title_lens.append(t_len)
                if t_len < MAX_TITLE_LEN:
                    title = title + [PAD]*(MAX_TITLE_LEN - t_len)
                title_array[i, :] = np.array(title)

            del df
            gc.collect()
            query_lens = np.array(query_lens)
            title_lens = np.array(title_lens)

            fea_array = np.empty((query_array.shape[0], self.feature_num), dtype=np.float32)
            start_dim, end_dim = 0, 0
            for file in self.fea_file_path_list:
                start_row = self.skiprows + count * self.bigbatch
                if "feature_10_1th_2e.csv.gz" in file:
                    start_row -= E2
                fea = load_feature_csv(file, start_row = start_row, 
                                        nrows = query_array.shape[0]).values.astype(np.float32)
                start_dim = end_dim
                end_dim = end_dim + fea.shape[1]
                fea_array[:, start_dim:end_dim] = fea
                del fea
            assert end_dim == self.feature_num
            count += 1

            idxes = list(range(query_array.shape[0]))
            if self.shuffle:
                random.shuffle(idxes)
            for b in range(math.ceil(query_array.shape[0] / self.batch_size)):
                start = b * self.batch_size
                end = min((b+1)*self.batch_size, query_array.shape[0])

                yield torch.tensor(query_lens[idxes[start:end]], dtype=torch.int),                        torch.tensor(title_lens[idxes[start:end]], dtype=torch.int),                        torch.tensor(query_array[idxes[start:end]], dtype=torch.long),                        torch.tensor(title_array[idxes[start:end]], dtype=torch.long),                        torch.tensor(fea_array[idxes[start:end]], dtype=torch.float32),                        torch.tensor(label[idxes[start:end]], dtype=torch.float)
            
    def _get_feature_num(self):
        feature_num = 0
        for file in self.fea_file_path_list:
            feature_num += pd.read_csv(file, nrows=2).shape[1]
        return feature_num

# ################################### 3. 训练 #####################################
def evaluate(dataloader, net, loss):
    net.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for query_lens, title_lens, query, title, fea, y in dataloader.next_batch():
            query = query.to(device)
            title = title.to(device)
            fea = fea.to(device)
            y = y.to(device)

            y_hat = net(query, title, fea)
            loss_sum += loss(y_hat.view(y.shape), y).cpu().item()
            n += 1
    net.train()
    return loss_sum / n

def train(net, loss, optimizer, scheduler, stoi, num_epochs, save_path, 
            verbose=1000, batch_size=512, bigbatch=128*10000, 
            valid_portion=0.2, valid_every=10000):
    net.train()
    net = net.to(device)
    print("training on ", device)
    # print("从之前的0.4111开始")
    best_valid_loss = 99999.9
    print("使用%.3f的数据作为验证集, 每%d步验证一次" % (valid_portion, valid_every))
    valid_num = int(chunksize * valid_portion)
    train_num = chunksize - valid_num
    
    # print("从2亿的数据开始!!!!!!")
    offset = 0
    
    for epoch in range(num_epochs):
        epoch_since = time.time()
        train_loader = myDataloader(train_file_path, feature_files, stoi, bigbatch,
                    skiprows=offset, nrows=train_num,  batch_size=batch_size, shuffle=True, columns=train_columns)
        valid_loader = myDataloader(train_file_path, feature_files, stoi, bigbatch,
                    skiprows=offset + train_num, nrows=valid_num,  batch_size=batch_size, shuffle=False, columns=train_columns)

        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        train_loss = []
        for step, (query_lens, title_lens, query, title, fea, y) in enumerate(train_loader.next_batch()):
            step_since = time.time()
            query = query.to(device)
            title = title.to(device)
            fea = fea.to(device)
            y = y.to(device)

            y_hat = net(query, title, fea)
            l = loss(y_hat.view(y.shape), y) 
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss.append(l.cpu().item())
            if (step + 1) % verbose == 0:
                train_avg_loss = sum(train_loss) / len(train_loss)
                train_loss = []
                print("epoch %d, step %d, train avg loss: %.4f, step time taken: %.2f sec"                     % (epoch+1, step+1, train_avg_loss, time.time() - step_since))
                    
            # validate
            if (step + 1) % valid_every == 0:
                print("开始验证...")
                valid_loss = evaluate(valid_loader, net, loss)
                scheduler.step()
                print("epoch %d, step %d, valid loss: %.4f, best valid loss: %.4f"                     % (epoch+1, step+1, valid_loss, best_valid_loss))
                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    if not debug:
                        print("better model! saving to %s..." % save_path, end="")
                        torch.save(net.state_dict(), save_path)
                        print("done!\n")
                
                
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
            
            
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存w2v模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        
    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" % 
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss >= epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)

        self.pre_loss = cum_loss
        self.since = time.time()
        print("\n\n\n")
        
print("加载word2vec模型")
w2v_model = Word2Vec.load(args["w2v_model_path"])
embed_size = w2v_model.trainables.layer1_size
vocab_size = len(w2v_model.wv.vocab)
print("加载完毕. 总词典数(不算PAD和OOV): %d, embeding size: %d" % (vocab_size, embed_size))
words = list(w2v_model.wv.vocab.keys())
word2id = {word:(i+2) for i, word in enumerate(words)} # 2, 3, 4..., 01留给PAD和OOV


print("构造embedding层")
embedding_matrix = np.zeros((vocab_size + 2, embed_size), dtype=np.float32)
for word, i in tqdm(word2id.items()):
    # try:
    embedding_matrix[i] = w2v_model.wv[word]
    # except:
    #     continue
print(embedding_matrix.shape)
del w2v_model


model = ESIM(manual_feature_dim = len(fea_names),
            vocab_size = vocab_size + 2,
            embed_dim = embed_size,
            hidden_size = args["hidden_size"],
            linear_size = args["linear_size"],
            dropout = args["dropout"])


model.embeds.weight.data.copy_(torch.tensor(embedding_matrix))
model.embeds.weight.requires_grad = False # 不更新embedding
del embedding_matrix
gc.collect()

print("统计参数个数:\n", get_parameter_number(model))

loss = torch.nn.BCELoss()
lr, num_epochs = 0.001, 50 
# 要过滤掉不计算梯度的embedding参数
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.8) #每过1step，学习率就乘gamma

verbose, batch_size, bigbatch = 1000, 4096, 4096*10000
if debug:
    print("debug mode!")
    verbose, batch_size, bigbatch = 100, 4096, 4096*100
    args["model_save_dir"] = "/home/kesci/work/tss_nn/esim100_debug"
    
    
    
os.makedirs(args["model_save_dir"], exist_ok=True)
model_save_path = os.path.join(args["model_save_dir"], "model.pth")
train(model, loss, optimizer, scheduler, word2id, num_epochs, model_save_path, 
                        verbose=verbose, batch_size=batch_size, bigbatch=bigbatch,
                        valid_portion=0.1, valid_every=(100 if debug else 10000))


# 注释掉上面的train函数调用，反注释下面的代码即可进行测试

# ----------------------- 下面是测试代码（1亿测试集的，2千万测试集的类似） ----------------------
# pretrained_path = "/home/kesci/work/tss_nn/tss_esim100_811/model.pth"
# print("从%s加载模型..." % pretrained_path)
# model.load_state_dict(torch.load(pretrained_path, map_location=device))

# test_loader = myDataloader(test_B_file_path, [f.replace(".csv", "_testB.csv") for f in feature_files], 
#                             word2id, bigbatch=20000000, skiprows=0, nrows=None, 
#                             batch_size=4096, 
#                             shuffle=False, columns=test_columns)
# model = model.to(device)
# print("testing on ", device)
# predicts = np.zeros((TEST_B_ROWS, ), dtype=np.float32)
# start, end = 0, 0
# model.eval()
# with torch.no_grad():
#     for query_lens, title_lens, query, title, fea, y in tqdm(test_loader.next_batch()):
#         query = query.to(device)
#         title = title.to(device)
#         fea = fea.to(device)
#         pred = model(query, title, fea).cpu().view(-1).detach().numpy()
        
#         end = start + pred.shape[0]
#         predicts[start:end] = pred
#         start = end

# assert end == predicts.shape[0]

# df = pd.DataFrame(data={"prediction": predicts})
# print(df.info())
# print(df.head(50))
# print(df.tail(50))

# df.to_csv("/home/kesci/work/nn_related/tss_esim100_811_4146_testB.csv.gz", 
#             compression='gzip', index=False, header=False)
# print("save done!")


# # 4.1 nn模型(2层lstm然后特征交互)+外部特征(keras)

# In[ ]:


##  hc nn model-复赛

import pandas as pd 
import numpy as np
from tqdm import tqdm as tqdm 
import time, csv, json, os, math, gc, random
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras.models import *
from keras.layers import *
from keras.optimizers import * 
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import gensim
from gensim.models import Word2Vec


w2v_model_path = '/home/kesci/work/word2vec/w2v_100_cbow_4e.model'

print("----- 从%s加载word2vec模型 -----" % w2v_model_path)
# 这里放上训练时的EpochSaver, 否则load会报错
class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    '''用于保存模型, 打印损失函数等等'''
    def __init__(self, savedir, save_name="word2vector.model"):
        os.makedirs(savedir, exist_ok=True)
        self.save_path = os.path.join(savedir, save_name)
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        
    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss() # 返回的是从第一个epoch累计的
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        print("Epoch %d, loss: %.2f, time: %dmin %ds" % 
                    (self.epoch, epoch_loss, time_taken//60, time_taken%60))
        if self.best_loss > epoch_loss:
            self.best_loss = epoch_loss
            print("Better model. Best loss: %.2f" % self.best_loss)
            model.save(self.save_path)
            print("Model %s save done!" % self.save_path)
            
        self.pre_loss = cum_loss
        self.since = time.time()
        print("\n\n\n")
w2v_model = Word2Vec.load(w2v_model_path)
embed_size = w2v_model.trainables.layer1_size
print("加载完毕. 总词典数: %d, embeding size: %d" % (len(w2v_model.wv.vocab), embed_size))


# 提取embedding
words = list(w2v_model.wv.vocab.keys())
word2id = {j:(i+2) for i, j in enumerate(words)}
print(len(word2id))

embedding_matrix = np.zeros((len(word2id)+2, 100), dtype=np.float32)
for word, i in tqdm(word2id.items()):
    try:
        embedding_matrix[i] = w2v_model.wv[word]
    except:
        continue
print(embedding_matrix.shape)
print(embedding_matrix[:100])
del w2v_model
gc.collect()

# 提取query和title  训练集和测试集
chunksize = 50000000
test_rows = 20000000
train_query = np.zeros((chunksize, 10), dtype=np.int32)
train_title = np.zeros((chunksize, 20), dtype=np.int32)
test_query =  np.zeros((test_rows, 10), dtype=np.int32)
test_title =  np.zeros((test_rows, 20), dtype=np.int32)

train_file_path = '/home/kesci/input/bytedance/train_final.csv'
test_file_path = '/home/kesci/input/bytedance/test_final_part1.csv'

print('训练集1e')
with open(train_file_path) as csvfile:
    for index,line in tqdm(enumerate(csv.reader(csvfile))):
        query = line[1].strip().split()
        query = [word2id.get(w, 1) for w in query]
        query = query[:10] + [0]*(10-len(query)) if len(query)<10 else query[:10]
        query = np.array(query, dtype=np.int32)
        train_query[index] = query
        
        title = line[3].strip().split()
        title = [word2id.get(w, 1) for w in title]
        title = title[:20] + [0]*(20-len(title)) if len(title)<20 else title[:20]
        title = np.array(title, dtype=np.int32)
        train_title[index] = title
        if index == chunksize - 1: break

print('测试集2kw')
since = time.time()
with open(test_file_path) as csvfile:
    for index,line in enumerate(csv.reader(csvfile)):
        query = line[1].strip().split()
        query = [word2id.get(w, 1) for w in query]
        query = query[:10] + [0]*(10-len(query)) if len(query)<10 else query[:10]
        query = np.array(query, dtype=np.int32)
        test_query[index] = query
        
        title = line[3].strip().split()
        title = [word2id.get(w, 1) for w in title]
        title = title[:20] + [0]*(20-len(title)) if len(title)<20 else title[:20]
        title = np.array(title, dtype=np.int32)
        test_title[index] = title
        if (index+1)%5000000 == 0:
            print(index+1, 'step. Time consumed:', time.time() - since)


# 抽取手工特征 for lstm 
train_files = [
    "/home/kesci/work/feature/feature_1_0.csv.gz",
    # 特征['query_length', 'title_length', 'WordMatchShare','WordMatchShare_query', 'WordMatchShare_title', 
    # 'LengthDiff', 'LengthDiffRate', 'LengthRatio_qt', 'LengthRatio_tq']
    "/home/kesci/work/feature/feature_2.csv.gz",
    # 特征['NgramJaccardCoef_1', 'NgramJaccardCoef_2', 'NgramJaccardCoef_3', 'NgramJaccardCoef_4']
    "/home/kesci/work/feature/feature_3.csv.gz",
    # 特征['Levenshtein_ratio', 'Levenshtein_distance_char','query_title_common_words', 'common_word_ratio']
    "/home/kesci/work/feature/feature_4.csv.gz",
    # 特征['lcsubstr_len', 'lcseque_len', 'longest_match_size', 'longest_match_ratio']
    "/home/kesci/work/feature/feature_5.csv.gz",
    # 特征['fuzz_qratio', 'fuzz_partial_ratio']
    "/home/kesci/work/feature/feature_6.csv.gz",
    # 特征['fuzz_partial_token_sort_ratio', 'fuzz_token_set_ratio', 'fuzz_token_sort_ratio']
    "/home/kesci/work/feature/feature_7.csv.gz",
    # 特征['query_Entropy', 'title_Entropy', 'query_title_Entropy', 'WordMatchShare_Entropy']
    "/home/kesci/work/feature_4e/feature_1_1_4e.csv.gz", #不用
    # 特征['TFIDFWordMatchShare', 'TFIDFWordMatchShare_query', 'TFIDFWordMatchShare_title']
    # "/home/kesci/work/feature_0th_2e/feature_9_0_0th_2e.csv.gz",  不用
    # 特征['w2v_avg_cosine', 'w2v_avg_cityblock']
    # "/home/kesci/work/feature_0th_2e/feature_9_1_0th_2e.csv.gz",  不用
    # 特征['w2v_avg_minkowski', 'w2v_avg_braycurtis', 'w2v_avg_canberra']
    "/home/kesci/work/feature_4e/feature_10_1th_2e.csv.gz",  
    # 特征['query_title_click', 'query_nunique_title', 'query_click', 'title_nunique_query', 'title_click']
    # "/home/kesci/work/feature_0th_2e/feature_12_0th_5kw.csv.gz",
    # 特征['d2v_cosine', 'd2v_cityblock', 'd2v_minkowski', 'd2v_braycurtis', 'd2v_canberra']
    "/home/kesci/work/feature/feature_train_4e_hc_add.csv.gz"
    # 特征 ['jaccard_similarity', 'qt_coword_query_ratio', 'qt_coword_title_ratio', 'qt_len_mean', 
    # 'qt_common_word_acc', 'ngram_query_title_precision', 'ngram_query_title_recall', 'ngram_query_title_acc']
]

testA_files = [
    "/home/kesci/work/feature/feature_1_0_testA.csv.gz",
    "/home/kesci/work/feature/feature_2_testA.csv.gz",
    "/home/kesci/work/feature/feature_3_testA.csv.gz",
    "/home/kesci/work/feature/feature_4_testA.csv.gz",
    "/home/kesci/work/feature/feature_5_testA.csv.gz",
    "/home/kesci/work/feature/feature_6_testA.csv.gz",
    "/home/kesci/work/feature/feature_7_testA.csv.gz",
    "/home/kesci/work/feature_4e/feature_1_1_4e_testA.csv.gz",
    "/home/kesci/work/feature_4e/feature_10_1th_2e_testA.csv.gz",
    "/home/kesci/work/feature/feature_test_2kw_hc_add.csv.gz"
]
cols_sum = 0
for file in testA_files:
    col_len = len(pd.read_csv(file, dtype='float32', nrows=1).columns)
    cols_sum = cols_sum + col_len
print('manual feature columns size:', cols_sum)

TRAIN_ROWS = 100000000
TEST_ROWS = 20000000
train_fea = np.empty((TRAIN_ROWS, cols_sum), dtype=np.float32)
test_fea = np.empty((TEST_ROWS, cols_sum), dtype=np.float32)
st = 0
since = time.time()
for file in train_files:
    col_len = len(pd.read_csv(file, dtype='float32', nrows=1).columns)
    if file != "/home/kesci/work/feature_4e/feature_10_1th_2e.csv.gz":
        train_fea[:,st:st+col_len]=np.array(pd.read_csv(file, dtype='float32', skiprows=2*TRAIN_ROWS, 
                                            nrows=TRAIN_ROWS), dtype=np.float32)
    else:
        train_fea[:,st:st+col_len]=np.array(pd.read_csv(file, dtype='float32', 
                                            nrows=TRAIN_ROWS), dtype=np.float32)
    st = st + col_len
    print(file, ' end. Time consumed: ', time.time() - since)
stt = 0
for file in testA_files:
    col_len = len(pd.read_csv(file, dtype='float32', nrows=1).columns)
    test_fea[:,stt:stt+col_len]=np.array(pd.read_csv(file, dtype='float32', nrows=TEST_ROWS),dtype=np.float32)
    stt = stt + col_len
    print(file, ' end')
    
train_label_file = "/home/kesci/work/feature/train_label.csv.gz"
train_y = pd.read_csv(train_label_file, dtype='float32',nrows=50000000)["label"].values.astype(np.int32)
print(train_query.shape, train_title.shape, train_fea.shape, train_y.shape)


# define nn model (based on lstm) 增大网络容量
def pool_corr(q1,q2,pool_way):
    if pool_way == 'max':
        pool = GlobalMaxPooling1D()
    elif pool_way == 'ave':
        pool = GlobalAveragePooling1D()
    else:
        raise RuntimeError("don't have this pool way")
    q1 = pool(q1)
    q2 = pool(q2)

    def norm_layer(x, axis=1):
        return (x - K.mean(x, axis=axis, keepdims=True)) / K.std(x, axis=axis, keepdims=True)
    q1 = Lambda(norm_layer)(q1)
    q2 = Lambda(norm_layer)(q2)
    
    def jaccard(x):
        return  x[0]*x[1]/(K.sum(x[0]**2,axis=1,keepdims=True)+
                           K.sum(x[1]**2,axis=1,keepdims=True)-
                           K.sum(K.abs(x[0]*x[1]),axis=1,keepdims=True))
    merged = Lambda(jaccard)([q1, q2])
    return merged

query = Input(shape=(10, ), name='query')
title = Input(shape=(20, ), name='title')
query_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(query)
title_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(title)

embedd_word = Embedding(len(word2id)+2, 300, weights=[embedding_matrix], trainable=False, name='emb')

lstm_dim1 = 200
lstm_dim2 = 150

lstm_w = Bidirectional(CuDNNLSTM(lstm_dim1,return_sequences=True), merge_mode='sum')
lstm2_w = Bidirectional(CuDNNLSTM(lstm_dim2,return_sequences=True), merge_mode='sum')

norm = BatchNormalization()
q1 = embedd_word(query)
q1 = norm(q1)
q1 = SpatialDropout1D(0.2)(q1)

q2 = embedd_word(title)
q2 = norm(q2)
q2 = SpatialDropout1D(0.2)(q2)

q1 = Lambda(lambda x: x[0] * x[1])([q1, query_mask])  # mask
q2 = Lambda(lambda x: x[0] * x[1])([q2, title_mask])

q1 = lstm_w(q1)
q2 = lstm_w(q2)
q1 = Lambda(lambda x: x[0] * x[1])([q1, query_mask])  # mask
q2 = Lambda(lambda x: x[0] * x[1])([q2, title_mask])
q1 = lstm2_w(q1)
q2 = lstm2_w(q2)

merged_max = pool_corr(q1, q2, 'max')
merged_ave = pool_corr(q1, q2, 'ave')

manual_fea = Input(shape=(46,),name='mf')
mf = BatchNormalization()(manual_fea)
mf = Dense(100, activation='relu')(mf)
mf = Dropout(0.2)(mf)

merged = concatenate([merged_ave, merged_max])
merged = Dense(200, activation='relu')(merged)
merged = concatenate([merged, mf])
merged = Dense(200, activation='relu')(merged)
output = Dense(1, activation='sigmoid')(merged)

lr=0.001
model = Model(inputs=[query, title, manual_fea], outputs=output)
model.compile(loss='binary_crossentropy',optimizer=Nadam(lr),metrics=['accuracy'])


# 模型运行 先训练前一亿得到权重，再加载该权重在一亿数据上进行微调。

from keras.callbacks import LearningRateScheduler
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=6)
weights_file="/home/kesci/work/nn_related/0805_lstm_bigsize_bn.weights"
model_checkpoint= ModelCheckpoint(weights_file, monitor="val_loss",         save_best_only=True, save_weights_only=True, mode='auto')

def schedule_steps(epoch, steps):
    for step in steps:
        if step[1] > epoch:
            print("Setting learning rate to {}".format(step[0]))
            return step[0]
    print("Setting learning rate to {}".format(steps[-1][0]))
    return steps[-1][0]
lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(0.001, 4), (0.000316228, 8), 
                                                                        (0.0001, 12), (3.16228e-05, 16),
                                                                        (1e-05, 20)]))
callbacks_list=[early_stopper, model_checkpoint, lrSchedule]

model.fit([train_query[:80000000], train_title[:80000000], train_fea[:80000000]], 
          train_y[:80000000], epochs=30, batch_size=2048,
          validation_data=([train_query[80000000:], train_title[80000000:], train_fea[80000000:]], train_y[80000000:]),
          callbacks=callbacks_list)

'''
from keras.callbacks import LearningRateScheduler
# lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=3, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=5)
weights_file="/home/kesci/work/nn_related/0808_lstm_bigsize_bn_.weights"
model_checkpoint= ModelCheckpoint(weights_file, monitor="val_loss", \
        save_best_only=True, save_weights_only=True, mode='auto')

def schedule_steps(epoch, steps):
	for step in steps:
		if step[1] > epoch:
			print("Setting learning rate to {}".format(step[0]))
			return step[0]
	print("Setting learning rate to {}".format(steps[-1][0]))
	return steps[-1][0]
lrSchedule = LearningRateScheduler(lambda epoch: schedule_steps(epoch, [(0.0005, 2), (0.000316228, 4), 
                                                                        (0.0001, 6), (3.16228e-05, 8),
                                                                        (1e-05, 10)]))
callbacks_list=[early_stopper, model_checkpoint, lrSchedule]

model.load_weights('/home/kesci/work/nn_related/0805_lstm_bigsize_bn.weights')  
model.fit([train_query[:80000000], train_title[:80000000], train_fea[:80000000]], 
          train_y[:80000000], epochs=20, batch_size=2048,
          validation_data=([train_query[80000000:], train_title[80000000:], train_fea[80000000:]], train_y[80000000:]),
          callbacks=callbacks_list)
'''

# 预测  test_data 2kw   （1e测试集与此类似）
model.load_weights('/home/kesci/work/nn_related/0808_lstm_bigsize_bn_.weights')
test_preds = model.predict([test_query, test_title, test_fea], batch_size=1024, verbose=1) 

# 保存预测文件
submission_df = pd.read_csv('0802_lgb_hc_all_fea_cvr.csv', header=None, names=['1', '2', '3'])[['1', '2']]
print(submission_df.info())
submission_df["3"] = test_preds[:,0]
print(submission_df.head(5))
save_file = '/home/kesci/work/nn_related/nn_submit/0809_1e_hc_300d_nn_pretrain.csv'
submission_df.to_csv(save_file, index=False, header=False)
print("%s Save Done!" % save_file)


# # 5 最终提交代码
# * 将得到的结果进行加权平均并按照要求保存提交文件
# * 下面代码是2千万的，1亿的类似

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pandas as pd
import os


test_file_path = "/home/kesci/input/bytedance/test_final_part1.csv"
TEST_ROWS = 20000000

# 每个文件包含一列，存放了2千万测试集的prediction
files = [
"/home/kesci/work/lgb_model_result/805_tss_lgb_testA_0-9_avg.csv.gz", # 0.60508400
"/home/kesci/work/lgb_model_result/808_hc_lgb_testA_avg.csv.gz", # 0.60487700

"/home/kesci/work/nn_related/tss_esim_806_4077.csv.gz", # 0.61537900
"/home/kesci/work/nn_related/tss_esim_809_4181.csv.gz", # 0.61660700
"/home/kesci/work/nn_related/tss_esim100_811_4146.csv.gz", # 0.62665700


"/home/kesci/work/nn_related/nn_submit/0806_1e_hc_300d_nn.csv", # 0.61092100
"/home/kesci/work/nn_related/nn_submit/0809_1e_hc_300d_nn_pretrain.csv", # 0.61357200

]

weights = [
2, 2, 5, 6, 18, 3, 4
]

if len(weights) == 0: # 如果没给权重那就设权重一样
    print("直接平均.")
    weights = list([1 for _ in range(len(files))])
else:
    print(weights)
assert len(weights) == len(files)


save_file = "/home/kesci/work/submit/wieght_225634_18.csv"


df_for_corr = pd.DataFrame()

# 加权平均
preds = np.zeros((TEST_ROWS,), dtype=np.float32)
for i in range(len(files)):
    preds_csv = pd.read_csv(files[i], header=None, 
                            names=["prediction"], dtype="float32")
    preds += weights[i] * preds_csv["prediction"].values
    
    # 用前100000行计算皮尔森相关系数
    n = 1000000
    df_for_corr["%d" % i] = preds_csv["prediction"].values[:n]
    
preds /= sum(weights)


# 相关系数可视化
import seaborn as sns
plt.subplots(figsize=(df_for_corr.shape[1], df_for_corr.shape[1]))
sns.heatmap(abs(df_for_corr.corr().values), annot=True, vmax=1, square=True, cmap="Blues", cbar=False)
plt.show()


# 加上两行: query id 和 query_title_id
test_columns = ['query_id','query','query_title_id','title']
submission_df = pd.read_csv(test_file_path, names = test_columns,
                            dtype={"query_id":np.int32, "query":str, 
                                    "query_title_id":np.int32, 
                                    "title":str})[["query_id", "query_title_id"]]
submission_df["prediction"] = preds
submission_df.to_csv(save_file, index=False, header=False)
print("%s Save Done!" % save_file)

