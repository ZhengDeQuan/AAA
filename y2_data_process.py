'''
要处理的数据，类似于以下的情况
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

csv1 csv2 csv3 代表三个样本，csv_s代表一个bathc中的数据
csvi中的每一个元素，用逗号间隔的，代表一个特征
这里模拟的是一个特征有多个取值的时候的情况。

plan：每个样例用一个dict来装，dict中的键对应tag_id,值是一个列表，每个元素是这个tag的取值。
每个样例的键值对公有108个。
所有的样例用一个list来装。最后转换成DataFrame，每个tag_id对应的多个值也有冒号来join起来，变成字符串。
读取之后注意将第0列，第1列转化为int，
然后将第0列删除即全部为1的列删除，
将第1列中，取值不为1也不为0的删除。（第一列为label）
还要将两个信息读入，
一个是每个tag_id,最多能对应多少种取值，种类少的ont_hot,种类多的hash_bucket+embedding,中间的embedding
另一个是每个tag_id在每个样例中最多对应多少取值，这样就可以估计所有的特征concat之后有多少维度。

需要手动构造的外部文件有两类：
第一类：就是上面说的，那些tag_id对应的特征少，多，中。
第二类：每个特征都是hash之后的int64位的数值，每个int64位的值需要映射到保留到一个文件中，比如以json的形式存成列表，
在（tf.contrib.lookup.index_table_from_tensor）中需要用到。
如下所示。
TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]
table = tf.contrib.lookup.index_table_from_tensor(mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
'''

import pickle
import json
import tensorflow as tf
import  numpy as np




GLOBAL_BATCH_SIZE = 2
GLOBAL_START_INDEX = 0
GLOBAL_TOTAL_EXAMPLE_NUM = 3
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
这里要有获取数据的逻辑
'''
input_data = csv_s
tag_featureNum = pickle.load(open("aaa.pkl","rb"))
tag_featureNumOneline = pickle.load(open("bbb.pkl","rb"))
tags = []

class Tag(object):
    def __init__(self,
                 featureNum = 0,
                 featureNumOneline=0,
                 vocab_size=0,
                 embedding_size = 0,
                 tag_set=[],
                 tag_name = ""):
        self.featureNum = featureNum
        self.featureNumOneline = featureNumOneline
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.tag_set =tag_set
        self.tag_name = tag_name

'''
获得tags的逻辑,tags中的每个元素都是一个Tag对象
其中要包含得到Tag对象的 kind属性的逻辑
'''


for tag in tags:
    if tag.kind != "hash": #如果tag_set = []，则用hash_bucket
        table = tf.contrib.lookup.index_table_from_tensor(mapping=tag.tag_set, default_value=-1)  ## 这里构造了个查找表 ##
        tag.table = table
    vocab_size = tag.vocab_size
    embedding_size = tag.embedding_size
    if tag.kind == "one_hot":
        one_feature = tf.contrib.layers.sparse_column_with_keys(
            column_name=tag.tag_name,
            keys=tag.tag_set,
            #default_value=-1,
            combiner='sum',
            #dtype=tf.dtypes.int64
            dtype=tf.dtypes.string
        )
        res = tf.contrib.layers.one_hot_column(one_feature)
    elif tag.kind == "hash":
        one_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
            column_name=tag.tag_name,
            hash_bucket_size=tag.vocab_size ,
            combiner="sum",
            dtype=tf.dtypes.string
            #dtype=tf.dtypes.int32
        )

        res = tf.contrib.layers.embedding_column(one_feature,
                                                        # initializer=my_initializer,
                                                        combiner="mean",
                                                        dimension=tag.embedding_size)
    tag.embedding_res = res



if GLOBAL_START_INDEX + GLOBAL_BATCH_SIZE > GLOBAL_TOTAL_EXAMPLE_NUM:
    current_batch_data = input_data[GLOBAL_START_INDEX:GLOBAL_TOTAL_EXAMPLE_NUM]
    GLOBAL_START_INDEX = GLOBAL_START_INDEX + GLOBAL_BATCH_SIZE - GLOBAL_TOTAL_EXAMPLE_NUM
    current_batch_data += input_data[0:GLOBAL_START_INDEX]
else:
    current_batch_data = input_data[GLOBAL_START_INDEX:GLOBAL_START_INDEX+GLOBAL_BATCH_SIZE]
    GLOBAL_START_INDEX = GLOBAL_START_INDEX + GLOBAL_BATCH_SIZE


embedding_res_list = []
for one_example in current_batch_data:
    feature_embedding_res_list = []
    for one_feature,tag in zip(one_example,tags):
        split_tags = tf.string_split(one_feature, "|")
        split_tags_values = split_tags.values
        one_sparse = tf.SparseTensor(
            indices=split_tags.indices,
            values=tag.table.lookup(split_tags.values) if tag.tag_name != "hash" else split_tags.values,## 这里给出了不同值通过表查到的index ##
            dense_shape=split_tags.dense_shape
        )
        current_mapping = {tag.tag_name: one_sparse}
        one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, tag.embedding_res)
        feature_embedding_res_list.append(one_feature_embedding_res)
    feature_embedding_res = tf.concat(feature_embedding_res_list,axis = -1)
    feature_embedding_res = tf.reshape(feature_embedding_res,[1,-1])
    embedding_res_list.append(feature_embedding_res)
embedded_res = tf.concat(embedding_res_list, axis=0) #一个batch内的所有的样例的embedding表示








