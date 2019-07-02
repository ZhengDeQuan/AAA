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
import pandas as pd
from custom_tag_config import custom_tags


LEARNING_RETE_BASE = 0.8  # 基学习率
LEARNING_RETE_DECAY = 0.99  # 学习率的衰减率
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减系数
REGULARIZATION_RATE = 0.0001  # 正则化项的权重系数
TRAINING_STEPS = 10000  # 迭代训练次数

GLOBAL_BATCH_SIZE = 2
GLOBAL_FEATURE_NUM = 108
GLOBAL_START_INDEX = 0
GLOBAL_TOTAL_EXAMPLE_NUM = 3
GLOBAL_EPOCH_NUM = 1
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
# input_data = pd.read_csv("zhengquan_test.processed")
# input_data = csv_s
# eval_input_data = pd.read_csv("eval_ins_add.processed")
# input_data = pickle.load(open("/home2/data/ttd/zhengquan_test.processed.csv.pkl","rb"))
# tag_featureNum = pickle.load(open("aaa.pkl","rb"))
# tag_featureNumOneline = pickle.load(open("bbb.pkl","rb"))
tags = []

#决定每个tag有多少的维度和词表的大小的逻辑
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

tags_to_repair = {}
for custom_tag in custom_tags:
    tags_to_repair[custom_tag['tag_name']] = custom_tag

for tag in tags:
    if tag.tag_name in tags_to_repair:
        tag.kind = "custom"
        tag.wide_or_deep_side = "deep"
        tag.embedding_size = tags_to_repair[tag.tag_name]['embedding_size']
        tag.vocab_size = tags_to_repair[tag.tag_name]['vocab_size']
        # tag.tag_set = tags_to_repair[tag.tag_name]['vocab_fun'](tag.tag_set)
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




X = tf.placeholder(shape=[GLOBAL_BATCH_SIZE,GLOBAL_FEATURE_NUM],dtype=tf.string)
Y = tf.placeholder(shape=[GLOBAL_BATCH_SIZE,1],dtype=tf.float32)

wide_embedding_res_list = []
deep_embedding_res_list = []
# for i in range(GLOBAL_BATCH_SIZE):
#     wide_feature_embedding_res_list = []
#     deep_feature_embedding_res_list = []
#     for j in range(GLOBAL_FEATURE_NUM):
#         one_feature = X[i][j]
#         one_feature = tf.reshape(one_feature, shape=[1])





exp_X = tf.expand_dims(X,axis=-1)

# def my_wide_function(one_example):
#     wide_feature_embedding_res_list = []
#     features = tf.unstack(one_example,axis = 0)
#     for one_feature, tag in zip( features , tags):
#         split_tag = tf.string_split(one_feature, "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
#             ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#         current_mapping = {tag.tag_name: one_sparse}
#         one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, tag.embedding_res)
#         # [[ 0.08187684,  0.22063671, -0.16549297]] with the shape of [1,embedding_size]
#         if tag.wide_or_deep_side == "wide":
#             wide_feature_embedding_res_list.append(one_feature_embedding_res)
#     wide_feature_embedding_res = tf.concat(wide_feature_embedding_res_list,axis = -1) #希望是[1，xigma of embedding_size]
#     wide_feature_embedding_res = tf.reshape(wide_feature_embedding_res,[1,-1])
#     return wide_feature_embedding_res
#
# def my_deep_function(one_example):
#     deep_feature_embedding_res_list = []
#     features = tf.unstack(one_example, axis=0)
#     for one_feature, tag in zip(features, tags):
#         split_tag = tf.string_split(one_feature, "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
#             ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#         current_mapping = {tag.tag_name: one_sparse}
#         one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, tag.embedding_res)
#         # [[ 0.08187684,  0.22063671, -0.16549297]] with the shape of [1,embedding_size]
#         if tag.wide_or_deep_side == "deep":
#             deep_feature_embedding_res_list.append(one_feature_embedding_res)
#     deep_feature_embedding_res = tf.concat(deep_feature_embedding_res_list, axis=-1)
#     deep_feature_embedding_res = tf.reshape(deep_feature_embedding_res, [1, -1])
#     return deep_feature_embedding_res
#
# wide_side_dimension_size = 0
# deep_side_dimension_size = 0
# for tag in tags:
#     if tag.wide_or_deep_side == "wide":
#         wide_side_dimension_size += tag.embedding_size
#     else:
#         deep_side_dimension_size += tag.embedding_size
#
# wide_embedded_res = tf.map_fn(fn=my_wide_function,elems=exp_X,dtype=tf.float32)
# deep_embedded_res = tf.map_fn(fn=my_deep_function,elems=exp_X,dtype=tf.float32)
#
#
# wide_inputs = tf.reshape(wide_embedded_res,[-1,wide_side_dimension_size])
# deep_inputs = tf.reshape(deep_embedded_res,[-1,deep_side_dimension_size])



# def my_function(one_example):
#     features = tf.unstack(one_example, axis=0)
#     wide_mappings = {}
#     wide_tensors = []
#     deep_mappings = {}
#     deep_tensors = []
#     for one_feature, tag in zip(features, tags):
#         if tag.wide_or_deep_side != "wide":
#             continue
#         split_tag = tf.string_split(one_feature, "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
#             ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#
#         wide_mappings[tag.tag_name]=one_sparse
#         wide_tensors.append(tag.embedding_res)
#
#     for one_feature, tag in zip(features, tags):
#         if tag.wide_or_deep_side == "wide":
#             continue
#         split_tag = tf.string_split(one_feature, "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
#             ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#
#         deep_mappings[tag.tag_name]=one_sparse
#         deep_tensors.append(tag.embedding_res)
#     mappings = {}
#     tensors = []
#     for key in wide_mappings:
#         mappings[key] = wide_mappings[key]
#     for key in deep_mappings:
#         mappings[key] = deep_mappings[key]
#     tensors = wide_tensors + deep_tensors
#     final_res = tf.feature_column.input_layer(mappings,tensors)
#     return final_res
#
# final_res = tf.map_fn(fn=my_function,elems=exp_X,dtype=tf.float32) #[batch_size, wide+deep_embedding_size]
# print(final_res)

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
    features = tf.unstack(one_example,axis = 0)
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
# wide_side_embedding , deep_side_embedding = tf.split(batch_embedding_res,[wide_side_dimension_size,deep_side_dimension_size],axis = 1)

wide_inputs,deep_inputs = tf.split(batch_embedding_res,[wide_side_dimension_size,deep_side_dimension_size],1)

wide_inputs = tf.reshape(wide_inputs,[GLOBAL_BATCH_SIZE,wide_side_dimension_size])
deep_inputs = tf.reshape(deep_inputs,[GLOBAL_BATCH_SIZE,deep_side_dimension_size])




with tf.variable_op_scope([wide_inputs], None, "cb_unit", reuse=False) as scope:
    central_bias =tf.Variable(name = 'central_bias',
                            initial_value=tf.random_normal(shape=[GLOBAL_BATCH_SIZE,1], mean=0, stddev=1),
                            trainable=True)

wide_side = tf.contrib.layers.fully_connected(inputs = wide_inputs,
                               num_outputs = 700,
                               activation_fn=tf.nn.relu,
                               biases_initializer=None
                               )
wide_side = tf.reduce_sum(wide_side,1,name="reduce_sum")
wide_side = tf.reshape(wide_side, [-1,1])
w_a_d = tf.concat([wide_inputs,deep_inputs],axis = 1,name="concat")

nodes = [100,50]
for k in range(len(nodes)):
    w_a_d = tf.contrib.layers.fully_connected(w_a_d,nodes[k],activation_fn=tf.nn.relu)
    w_a_d = tf.layers.dropout(
        inputs=w_a_d,
        rate = 0.5,
        name="deep_dropout_%d" % k,
    )
deep_side = tf.contrib.layers.fully_connected(w_a_d,1,
                               activation_fn=None,
                               biases_initializer=None)
deep_side = tf.reshape(deep_side,[-1,1])
w_a_d_logit = tf.add(deep_side,wide_side)
w_a_d_logit = tf.add(w_a_d_logit, central_bias, name="wide_with_bias")
print(w_a_d_logit)
print(Y)
w_a_d_output = tf.nn.softmax(w_a_d_logit,dim=-1)
loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels = Y,
    logits=w_a_d_logit,
    name="loss_function"
)
loss_mean = tf.reduce_mean(loss)
global_step = tf.Variable(0, trainable=False)  # 定义存储当前迭代训练轮数的变量

# 定义ExponentialMovingAverage类对象
variable_averages = tf.train.ExponentialMovingAverage(
    MOVING_AVERAGE_DECAY, global_step)  # 传入当前迭代轮数参数
# 定义对所有可训练变量trainable_variables进行更新滑动平均值的操作op
trainable_vars = tf.trainable_variables()
variables_averages_op = variable_averages.apply(trainable_vars)
# regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
# regul_weights = [ v for v in trainable_vars if not v.name.endswith("b")] #只限制W不限制b
# regularization = regularizer(regul_weights)
# total_loss = loss_mean + regularization  # 总损失值
total_loss = loss_mean

# 定义指数衰减学习率
learning_rate = tf.train.exponential_decay(LEARNING_RETE_BASE, global_step,
                                           2000000 / GLOBAL_BATCH_SIZE, LEARNING_RETE_DECAY)
# 定义梯度下降操作op，global_step参数可实现自加1运算
train_step = tf.train.GradientDescentOptimizer(learning_rate) \
    .minimize(loss, global_step=global_step)
# 组合两个操作op
train_op = tf.group(train_step, variables_averages_op)
'''
# 与tf.group()等价的语句
with tf.control_dependencies([train_step, variables_averages_op]):
    train_op = tf.no_op(name='train')
'''
# 定义准确率
# 在最终预测的时候，神经网络的输出采用的是经过滑动平均的前向传播计算结果
predictions = tf.cast(tf.greater(w_a_d_output, 0), tf.int64)
correct_prediction = tf.equal(predictions, tf.cast(Y,tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




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






# 初始化回话sess并开始迭代训练
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.tables_initializer()])
    # 验证集待喂入数据
    train_steps = 0
    current_epoch = 0
    # sess.run(train_op, feed_dict={X: train_X, Y: train_Y})
    # a,b = sess.run([final_res,tf.shape(final_res)], feed_dict={X: train_X, Y: train_Y})
    # print("a = ", a)
    # print("b = ",b)


    # a,b,c = sess.run([w_a_d_logit,wide_inputs,deep_inputs], feed_dict={X: train_X, Y: train_Y})
    # print("a = ", a)
    # print("b = ", b)
    # print("c = ", c)

    # a, b = sess.run([wide_inputs, deep_inputs], feed_dict={X: train_X, Y: train_Y})
    # print("a = ", a)
    # print("b = ", b)


    a,b,c = sess.run([predictions,correct_prediction,accuracy],feed_dict={X: train_X, Y: train_Y})
    print("a = ",a)
    print("b = ",b)
    print("c = ",c)












