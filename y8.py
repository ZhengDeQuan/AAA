'''
多离散值处理方案
tf.string_split()处理一个样例，但是我要一个batch中处理多个样例。
'''

import tensorflow as tf
import numpy as np

import pandas as pd

csv1 = [
  "harden|james|curry",
  "wrestbrook|harden|durant",
  "paul|towns",
]
csv2 = [
"curry",
  "wrestbrook|harden|durant",
  "paul|towns",
]
csv3 = [
"harden|james|curry",
  "durant",
  "paul|towns",
]

csv_s= [csv1,csv2,csv3]

'''
csv1 csv2 csv3 代表三个样本，csv_s代表一个bathc中的数据
csvi中的每一个元素，用逗号间隔的，代表一个特征
这里模拟的是一个特征有多个取值的时候的情况
'''

TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]
table = tf.contrib.lookup.index_table_from_tensor(mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##

vocab_size = 10
embedding_size = 8
embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
# for w, i in word_index.items():
#     v = embeddings.get(w)
#     if v is not None and i < vocab_size:
#         embedding_matrix[i] = v

for i in range(vocab_size):
    embedding_matrix[i] = np.ones(embedding_size) * i

def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix

params = {'embedding_initializer': my_initializer}

# education = tf.contrib.layers.sparse_column_with_integerized_feature(
#     column_name="education",
#     bucket_size=10,
#     combiner='sum',
#     default_value=-1
#     dtype=tf.dtypes.int64
# )
#
# res_of_edu = tf.contrib.layers.embedding_column(education,
#                                                 initializer=my_initializer,
#                                                 combiner="mean",
#                                                 dimension=8)

education = tf.contrib.layers.sparse_column_with_keys(
    column_name = "education",
    keys=np.array([0,1,2,3,4,5,6,7,8,9]).astype(np.int64),
    default_value=-1,
    combiner='sum',
    dtype=tf.dtypes.int64
)

res_one_hot_education = tf.contrib.layers.one_hot_column(education)

embedded_tags = []
for csv in csv_s:
    split_tags = tf.string_split(csv, "|" )
    split_tags_values = split_tags.values
    tag = tf.SparseTensor(
      indices=split_tags.indices,
      values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
      dense_shape=split_tags.dense_shape)
    for_map_education = {'education': tag}
    #res = tf.feature_column.input_layer(for_map_education,res_of_edu)
    res = tf.feature_column.input_layer(for_map_education, res_one_hot_education)
    embedded_tag = tf.reshape(res,[1,-1])
    embedded_tags.append(embedded_tag)

embedded_tags = tf.concat(embedded_tags, axis=0)









with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print(s.run(embedded_tags))
  print(s.run(embedded_tag))




