'''
多离散值处理方案
tf.string_split()处理一个样例，但是我要一个batch中处理多个样例。
'''

import tensorflow as tf

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
TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))


embedded_tags = []
for csv in csv_s:
    split_tags = tf.string_split(csv, "|" )
    split_tags_values = split_tags.values
    tag = tf.SparseTensor(
      indices=split_tags.indices,
      values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
      dense_shape=split_tags.dense_shape)
    embedded_tag = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tag, sp_weights=None)
    embedded_tag = tf.reshape(embedded_tag,[1,-1])
    embedded_tags.append(embedded_tag)

embedded_tags = tf.concat(embedded_tags,axis = 0)



with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print(s.run(embedded_tags))
  print(s.run(embedded_tag))




