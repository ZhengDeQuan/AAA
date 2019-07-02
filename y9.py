'''
对那个expect dtype=string，but found dtype=int64的debug。结果是，
one_feature = tf.contrib.layers.sparse_column_with_keys(
            column_name=tag.tag_name,
            keys=tag.tag_set,
            default_value=0,
            combiner='sum',
            #dtype=tf.dtypes.int64
            dtype=tf.dtypes.string
        )
这个函数中设置了keys关键字的话，那么就不应该在调用中再用table了
table = tf.contrib.lookup.index_table_from_tensor(mapping=tag.tag_set, default_value=0)  ## 这里构造了个查找表 ##
one_sparse = tf.SparseTensor(
    indices=split_tag.indices,
    values=table.lookup(split_tag.values),
    dense_shape=split_tag.dense_shape
)

'''

import pickle
import json
import tensorflow as tf
import  numpy as np


GLOBAL_BATCH_SIZE = 2
GLOBAL_START_INDEX = 0
GLOBAL_TOTAL_EXAMPLE_NUM = 3

df_data = pickle.load(open("/home2/data/ttd/zhengquan_test.processed.csv.pkl","rb")) #一个DataFrame
# import pdb
# pdb.set_trace()
df_data = df_data.dropna(how="all", axis=0) # 0 对行进行操作，how='any'只要有一个NA就删除这一行，how='all'必须全部是NA才能删除这一行
#不能用any过滤，否则过滤完了，1000个只剩3个。
df_data['label'] = (df_data['label']).astype(int)
df_data = df_data[df_data['label'].isin([0,1])] #只保留label为0或者1的

#分离X,Y
X_data = df_data.drop(['label'],axis = 1)
# X_data = X_data.values.astype(tf.string)
X_data = X_data.applymap(str)
X_data = X_data.values.astype(np.str)
X_data = X_data.tolist()
Y_data = df_data['label'].values.astype(np.int32)





# data_size=len(X_data)
# indices=np.random.permutation(np.arange(data_size))
# shufflfed_X=X_data[indices]
# shufflfed_Y=Y_data[indices]
# X_batch_data ,Y_batch_data = shufflfed_X[0:GLOBAL_BATCH_SIZE]
# # input_data = csv_s
input_data = X_data
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

    def cal_vocabSize_embeddingSize(self,tag_set):
        if self.featureNum <=30:
            self.kind = "one_hot"
            self.wide_or_deep_side = "wide"
            self.vocab_size = self.featureNum
            self.embedding_size = self.featureNum
            self.tag_set = tag_set

        #直接映射成embedding的情况
        elif 30 < self.featureNum <= 10000:
            self.kind = "inteEmbdding"
            self.wide_or_deep_side = "wide+deep"
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
                self.tag_set = tag_set if self.vocab_size == self.featureNum else []
            else:
                self.kind = "hash"
                self.wide_or_deep_side = "deep"
                self.vocab_size = min(self.featureNum , max(100000,self.featureNum * 0.4))
                self.embedding_size = 40  # 写死的
                self.tag_set = tag_set if self.vocab_size == self.featureNum else []


tag2value = json.load(open("tag2value.json","r",encoding="utf-8"))
tag2valueOneline = json.load(open('tag2valueOneline.json',"r",encoding="utf-8"))
#type = dict, key是tag，value是这个tag的所有的可能的取值组成的列表
for key in tag2value:
    tag = Tag(
        featureNum=len(tag2value[key]),
        featureNumOneline=len(tag2valueOneline),
        tag_name=key,
    )
    tag.cal_vocabSize_embeddingSize(tag2value[key])
    tags.append(tag)


tag = tags[0]
one_feature = tf.contrib.layers.sparse_column_with_keys(
            column_name=tag.tag_name,
            keys=tag.tag_set,
            default_value=0,
            combiner='sum',
            #dtype=tf.dtypes.int64
            dtype=tf.dtypes.string
        )
res = tf.contrib.layers.one_hot_column(one_feature)


tag = tags[0]
print(tag.tag_set)
table = tf.contrib.lookup.index_table_from_tensor(mapping=tag.tag_set, default_value=0)  ## 这里构造了个查找表 ##
tag.table = table
vocab_size = tag.vocab_size
embedding_size = tag.embedding_size
one_feature = tf.contrib.layers.sparse_column_with_keys(
    column_name=tag.tag_name,
    keys=["Marry","Hello","World"],
    default_value=-1,
    combiner='sum',
    #dtype=tf.dtypes.int64
    dtype=tf.dtypes.string
)
res = tf.contrib.layers.one_hot_column(one_feature)
tag.embedding_res = res



current_batch_data = input_data[GLOBAL_START_INDEX:GLOBAL_START_INDEX+GLOBAL_BATCH_SIZE]



wide_embedding_res_list = []
deep_embedding_res_list = []
# current_batch_data = current_batch_data.tolist()
one_example = current_batch_data[0]
wide_feature_embedding_res_list = []
deep_feature_embedding_res_list = []
one_feature_of_one_example = one_example[0]
one_feature_of_one_example = "Hello|World"
print("one_feature = ",type(one_feature_of_one_example))
print(one_feature_of_one_example)
split_tag = tf.string_split([one_feature_of_one_example], "|")

split_tag_values = split_tag.values

look_up_res = table.lookup(split_tag_values)

one_sparse = tf.SparseTensor(
    indices=split_tag.indices,
    values=split_tag.values,  ## 这里给出了不同值通过表查到的index ##
    #values=look_up_res,
    dense_shape=split_tag.dense_shape
)
current_mapping = {tag.tag_name: one_sparse}
one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, res)
print("tag_name = ", tag.tag_name, "  tag_kind = ", tag.kind)
print("tag_set = ",tag.tag_set)
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.tables_initializer()])
    a,b ,c,d= sess.run([split_tag,one_sparse,look_up_res,one_feature_embedding_res])
    print("a = ",a)
    print("type_a = ",type(a[0]))
    print("b = ",b)
    print("c = ",c)
    print("tag.tag_set = ",tag.tag_set)
    print("d = ",d)




