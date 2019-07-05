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
from custom_tag_config import custom_tags



GLOBAL_BATCH_SIZE = 1
GLOBAL_FEATURE_NUM = 108
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
#
df_data = pickle.load(open("/home2/data/ttd/zhengquan_test.processed.csv.pkl","rb")) #一个DataFrame
# import pdb
# pdb.set_trace()
df_data = df_data.dropna(how="all", axis=0) # 0 对行进行操作，how='any'只要有一个NA就删除这一行，how='all'必须全部是NA才能删除这一行
#不能用any过滤，否则过滤完了，1000个只剩3个。
df_data['label'] = (df_data['label']).astype(int)
df_data = df_data[df_data['label'].isin([0,1])] #只保留label为0或者1的

#分离X,Y
X_data = df_data.drop(['label'],axis = 1)
X_data = X_data.applymap(str)
# X_data = X_data.values.astype(np.str)
X_data = X_data.values
Y_data = df_data['label'].values.astype(np.int32)
data_size=len(X_data)
indices=np.random.permutation(np.arange(data_size))
import pdb
pdb.set_trace()
print(type(X_data))
shufflfed_X=X_data[indices]
shufflfed_Y=Y_data[indices]
X_batch_data = shufflfed_X[0:GLOBAL_BATCH_SIZE]
# input_data = csv_s
# input_data = X_batch_data
input_data = X_batch_data


exit(5678)


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

    def cal_(self,tag_set):
        if self.featureNum <=30:
            self.kind = "one_hot"
            self.wide_or_deep_side = "wide"
            self.vocab_size = self.featureNum
            self.embedding_size = self.featureNum
            self.tag_set = tag_set

        #直接映射成embedding的情况
        elif 30 < self.featureNum <= 10000:
            self.kind = "inteEmbdding"
            # self.wide_or_deep_side = "wide+deep"
            self.wide_or_deep_side = "deep"
            self.vocab_size = self.featureNum
            self.embedding_size = 10 #写死的
            self.tag_set = tag_set
        #hash的情况
        else:
            if self.featureNum <= 100000:
                self.kind = "hash"
                self.wide_or_deep_side = "deep"
                self.vocab_size = min(self.featureNum,70000)
                self.embedding_size = 40  # 写死的
                self.tag_set = tag_set
                # self.tag_set = tag_set if self.vocab_size == self.featureNum else []
            else:
                self.kind = "hash"
                self.wide_or_deep_side = "deep"
                self.vocab_size = int( min(self.featureNum , max(100000,self.featureNum * 0.4)) )
                self.embedding_size = 40  # 写死的
                self.tag_set = tag_set
                # self.tag_set = tag_set if self.vocab_size == self.featureNum else []


tag2value = json.load(open("tag2value.json","r",encoding="utf-8"))
tag2valueOneline = json.load(open('tag2valueOneline.json',"r",encoding="utf-8"))
tag2value = sorted(tag2value.items(),key = lambda x: x[0])
tag2value = dict(tag2value)


#type = dict, key是tag，value是这个tag的所有的可能的取值组成的列表
for key in tag2value:
    tag = Tag(
        featureNum=len(tag2value[key]),
        featureNumOneline=len(tag2valueOneline),
        tag_name=key,
    )
    tag.cal_(tag2value[key])
    tags.append(tag)








vocab_size = 10
embedding_size = 8
embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
# for w, i in word_index.items():
#     v = embeddings.get(w)
#     if v is not None and i < vocab_size:
#         embedding_matrix[i] = v

for i in range(vocab_size):
    embedding_matrix[i] = np.ones(embedding_size) * i * 2

def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix

tags_to_repair = {}
for custom_tag in custom_tags:
    tags_to_repair[custom_tag['tag_name']] = custom_tag

for tag in tags:
    if tag.tag_name in tags_to_repair:
        tag.kind = "custom"
        tag.wide_or_deep_side = "deep"
        tag.embedding_size = tags_to_repair[tag.tag_name]['embedding_size']
        tag.vocab_size = tags_to_repair[tag.tag_name]['vocab_size']
        table = tf.contrib.lookup.index_table_from_tensor(mapping=tag.tag_set, default_value=-1)  ## 这里构造了个查找表 ##
        tag.table = table
        one_feature = tf.contrib.layers.sparse_column_with_keys(
            column_name=tag.tag_name,
            keys=tag.tag_set,
            default_value=0,
            combiner='sum',
            # dtype=tf.dtypes.int64
            dtype=tf.dtypes.string
        )
        res = tf.contrib.layers.embedding_column(one_feature,
                                           initializer=tags_to_repair[tag.tag_name]['initializer_function'],
                                           combiner="mean",
                                           dimension=tag.embedding_size)
        tag.embedding_res = res
        continue

    vocab_size = tag.vocab_size
    embedding_size = tag.embedding_size
    if tag.kind == "one_hot":
        one_feature = tf.contrib.layers.sparse_column_with_keys(
            column_name=tag.tag_name,
            keys=tag.tag_set,
            default_value=-1,
            combiner='sum',
            #dtype=tf.dtypes.int64
            dtype=tf.dtypes.string
        )
        res = tf.contrib.layers.one_hot_column(one_feature)
    else:
        one_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
            column_name=tag.tag_name,
            hash_bucket_size=tag.vocab_size ,
            combiner="sum",
            dtype=tf.dtypes.string
            #dtype=tf.dtypes.int64
        )

        res = tf.contrib.layers.embedding_column(one_feature,
                                                        # initializer=my_initializer,
                                                        combiner="mean",
                                                        dimension=tag.embedding_size)
    tag.embedding_res = res








wide_embedding_res_list = []
deep_embedding_res_list = []


# current_batch_data = current_batch_data.tolist()
# for one_example in current_batch_data:
#     wide_feature_embedding_res_list = []
#     deep_feature_embedding_res_list = []
#     for one_feature,tag in zip(one_example,tags):
#         print("tag_name = ",tag.tag_name)
#         split_tag = tf.string_split([one_feature], "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.kind == "custom" else split_tag.values,  ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#         current_mapping = {tag.tag_name: one_sparse}
#         one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, tag.embedding_res)
#         if tag.wide_or_deep_side == "wide":
#             wide_feature_embedding_res_list.append(one_feature_embedding_res)
#         else:
#             deep_feature_embedding_res_list.append(one_feature_embedding_res)
#
#     wide_feature_embedding_res = tf.concat(wide_feature_embedding_res_list,axis = -1)
#     deep_feature_embedding_res = tf.concat(deep_feature_embedding_res_list,axis = -1)
#
#     wide_feature_embedding_res = tf.reshape(wide_feature_embedding_res,[1,-1])
#     deep_feature_embedding_res = tf.reshape(deep_feature_embedding_res,[1,-1])
#
#     wide_embedding_res_list.append(wide_feature_embedding_res)
#     deep_embedding_res_list.append(deep_feature_embedding_res)
#
# wide_embedded_res = tf.concat(wide_embedding_res_list, axis=0) #一个batch内的所有的样例的wide side embedding表示
# deep_embedded_res = tf.concat(deep_embedding_res_list, axis=0) #一个batch内的所有的样例的deep side embedding表示



# current_batch_data = current_batch_data.tolist()
# wide_side_dimension_size = 0
# deep_side_dimension_size = 0
# for tag in tags:
#     if tag.wide_or_deep_side == "wide":
#         wide_side_dimension_size += tag.embedding_size
#     else:
#         deep_side_dimension_size += tag.embedding_size
#
#
#
# for one_example in current_batch_data:
#     wide_mappings = {}
#     wide_tensors = []
#     deep_mappings = {}
#     deep_tensors = []
#     for one_feature, tag in zip(one_example, tags):
#         if tag.wide_or_deep_side != "wide":
#             continue
#         split_tag = tf.string_split([one_feature], "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
#             ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#
#         wide_mappings[tag.tag_name] = one_sparse
#         wide_tensors.append(tag.embedding_res)
#
#     for one_feature, tag in zip(one_example, tags):
#         if tag.wide_or_deep_side == "wide":
#             continue
#         split_tag = tf.string_split([one_feature], "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
#             ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#
#         deep_mappings[tag.tag_name] = one_sparse
#         deep_tensors.append(tag.embedding_res)
#     mappings = {}
#     tensors = []
#     for key in wide_mappings:
#         mappings[key] = wide_mappings[key]
#     for key in deep_mappings:
#         mappings[key] = deep_mappings[key]
#     tensors = wide_tensors + deep_tensors
#     wide_and_deep_embedding_res = tf.feature_column.input_layer(mappings, tensors)
#     break
#
#
# wide_side_embedding , deep_side_embedding = tf.split(wide_and_deep_embedding_res,[wide_side_dimension_size,deep_side_dimension_size],axis = 1)


# current_batch_data = current_batch_data.tolist()
wide_side_dimension_size = 0
deep_side_dimension_size = 0
for tag in tags:
    if tag.wide_or_deep_side == "wide":
        wide_side_dimension_size += tag.embedding_size
    else:
        deep_side_dimension_size += tag.embedding_size

X = tf.placeholder(shape=[GLOBAL_BATCH_SIZE,GLOBAL_FEATURE_NUM],dtype=tf.string)
exp_X = tf.expand_dims(X,axis = -1)
uns_X = tf.unstack(exp_X,axis = 0)

batch_embedding_res = []
for one_example in uns_X:
    wide_mappings = {}
    wide_tensors = []
    deep_mappings = {}
    deep_tensors = []
    features  = tf.unstack(one_example,axis = 0)
    for one_feature, tag in zip(features, tags):
        if tag.wide_or_deep_side != "wide":
            continue
        split_tag = tf.string_split(one_feature, "|")
        one_sparse = tf.SparseTensor(
            indices=split_tag.indices,
            values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
            ## 这里给出了不同值通过表查到的index ##
            dense_shape=split_tag.dense_shape
        )

        wide_mappings[tag.tag_name] = one_sparse
        wide_tensors.append(tag.embedding_res)

    for one_feature, tag in zip(features, tags):
        if tag.wide_or_deep_side == "wide":
            continue
        split_tag = tf.string_split(one_feature, "|")
        one_sparse = tf.SparseTensor(
            indices=split_tag.indices,
            values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
            ## 这里给出了不同值通过表查到的index ##
            dense_shape=split_tag.dense_shape
        )

        deep_mappings[tag.tag_name] = one_sparse
        deep_tensors.append(tag.embedding_res)
    mappings = {}
    tensors = []
    for key in wide_mappings:
        mappings[key] = wide_mappings[key]
    for key in deep_mappings:
        mappings[key] = deep_mappings[key]
    tensors = wide_tensors + deep_tensors
    wide_and_deep_embedding_res = tf.feature_column.input_layer(mappings, tensors)
    batch_embedding_res.append(wide_and_deep_embedding_res)

batch_embedding_res = tf.concat(batch_embedding_res,axis = 0)
wide_side_embedding , deep_side_embedding = tf.split(batch_embedding_res,[wide_side_dimension_size,deep_side_dimension_size],axis = 1)


df_data = pickle.load(open("/home2/data/ttd/zhengquan_test.processed.csv.pkl","rb")) #一个DataFrame
# import pdb
# pdb.set_trace()
df_data = df_data.dropna(how="all", axis=0) # 0 对行进行操作，how='any'只要有一个NA就删除这一行，how='all'必须全部是NA才能删除这一行
#不能用any过滤，否则过滤完了，1000个只剩3个。
df_data['label'] = (df_data['label']).astype(int)
df_data = df_data[df_data['label'].isin([0,1])] #只保留label为0或者1的

#分离X,Y
X_data = df_data.drop(['label'],axis = 1)
X_data = X_data.values.astype(str)
Y_data = df_data['label'].values.astype(np.int32)

data_size=len(X_data)
indices=np.random.permutation(np.arange(data_size))
shufflfed_X=X_data[indices]
shufflfed_Y=Y_data[indices]

train_X=shufflfed_X[0:GLOBAL_BATCH_SIZE]
train_Y=shufflfed_Y[0:GLOBAL_BATCH_SIZE]
train_Y=np.expand_dims(train_Y,axis = -1)


with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.tables_initializer()])
    print(sess.run(wide_side_embedding,feed_dict={X:train_X}))
    print(sess.run(tf.shape(wide_side_embedding),feed_dict={X:train_X}))
    print(sess.run(deep_side_embedding,feed_dict={X:train_X}))
    print(sess.run(tf.shape(deep_side_embedding),feed_dict={X:train_X}))
    print(wide_side_embedding)
    print(deep_side_embedding)





