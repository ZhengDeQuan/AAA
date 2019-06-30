import pandas as pd
import numpy as np
from collections import Counter
test_dfn="/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/test_ins_continues.txtlineNUM_videoID"

# test_data = pd.read_csv(test_dfn, delim_whitespace=True, header=None)
# test_data[0] = (test_data[0]).astype(int)
# input_data= test_data
# X = input_data.iloc[:,1:].values.astype(np.float32)
# Y = input_data[0].values.astype(np.float32)
# print(type(Y))
# print(np.unique(Y))
# print(Counter(Y))
#test集
# <class 'numpy.ndarray'>
# [0. 1. 2.]
# Counter({1.0: 153990, 0.0: 146009, 2.0: 1})
#train集
# <class 'numpy.ndarray'>
# [0. 1. 2.]
# Counter({0.0: 1810106, 1.0: 189891, 2.0: 3})

# test_data = pd.read_csv(test_dfn, delim_whitespace=True, header=None)
# test_data[0] = (test_data[0]).astype(int)
# input_data= test_data[test_data[0].isin([0,1])]
# print(test_data.shape)
# print(input_data.shape)
# X = input_data.iloc[:,1:].values.astype(np.float32)
# Y = input_data[0].values.astype(np.float32)
# print(type(Y))
# print(np.unique(Y))
# print(Counter(Y))

test_data = pd.read_csv(test_dfn, delim_whitespace=True, header=None)
test_data[0] = (test_data[0]).astype(int)
input_data= test_data[test_data[0].isin([0,1])]
print(test_data.shape)
print(input_data.shape)
col_num = test_data.shape[1] - 1
print("col_num = ",col_num)
vocab = {}
vocab["[UNK]"] = 5
test_data[col_num] = (test_data[col_num].apply(
                    lambda x: vocab[x] if x in vocab else vocab["[UNK]"])).astype(int)
print(test_data)