'''
先实现骨架
'''
import tensorflow as tf
import numpy as np
import json
import pickle
import pandas as pd
from custom_tag_config import custom_tags

LEARNING_RETE_BASE = 0.8  # 基学习率
LEARNING_RETE_DECAY = 0.99  # 学习率的衰减率
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减系数
REGULARIZATION_RATE = 0.0001  # 正则化项的权重系数
TRAINING_STEPS = 10000  # 迭代训练次数
GLOBAL_BATCH_SIZE = 50
GLOBAL_FEATURE_NUM = 108
GLOBAL_WIDE_DIMENSION = 700
GLOBAL_DEEP_DIMENSION = 700

nodes = [700, 100]
# wide_inputs = tf.random_normal(shape=[GLOBAL_BATCH_SIZE,GLOBAL_WIDE_DIMENSION], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# deep_inputs = tf.truncated_normal(shape=[GLOBAL_BATCH_SIZE,GLOBAL_DEEP_DIMENSION], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# Y = np.arange(GLOBAL_BATCH_SIZE).reshape(GLOBAL_BATCH_SIZE,1)
# Y = tf.constant(value=Y, dtype=tf.int32, shape=[GLOBAL_BATCH_SIZE, ])
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
        tf.contrib.layers.embedding_column(one_feature,
                                           initializer=tags_to_repair[tag.tag_name]['initializer_function'],
                                           combiner="mean",
                                           dimension=tag.embedding_size)
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


X = tf.placeholder(shape=[None,GLOBAL_FEATURE_NUM],dtype=tf.string)
Y = tf.placeholder(shape=[None,1],dtype=tf.float32)

exp_X = tf.expand_dims(X,axis=-1)
# example_list = tf.unstack(exp_X,axis = 0)


wide_embedding_res_list = []
deep_embedding_res_list = []

#用unstack的方式也不能拜托GLOBAL_BATCH_SIZE的限制，但是至少可以用feed_dict了
# exp_X = tf.expand_dims(X,axis=-1)
# example_list = tf.unstack(exp_X,axis = 0)
# for one_example in example_list:
#     wide_feature_embedding_res_list = []
#     deep_feature_embedding_res_list = []
#     features = tf.unstack(one_example,axis = 0)
#     feature = features[0]
#     for one_feature in features:
#         split_tag = tf.string_split(one_feature, "|")
#         tag = tags[j]
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values=tag.table.lookup(split_tag.values) if tag.tag_name != "hash" else split_tag.values,
#             ## 这里给出了不同值通过表查到的index ##
#             dense_shape=split_tag.dense_shape
#         )
#         current_mapping = {tag.tag_name: one_sparse}
#         one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, tag.embedding_res)
#         # [[ 0.08187684,  0.22063671, -0.16549297]] with the shape of [1,embedding_size]
#         if tag.wide_or_deep == "wide":
#             wide_feature_embedding_res_list.append(one_feature_embedding_res)
#         else:
#             deep_feature_embedding_res_list.append(one_feature_embedding_res)
#
#     wide_feature_embedding_res = tf.concat(wide_feature_embedding_res_list,axis = -1) #希望是[1，xigma of embedding_size]
#     deep_feature_embedding_res = tf.concat(deep_feature_embedding_res_list,axis = -1)
#
#     wide_feature_embedding_res = tf.reshape(wide_feature_embedding_res,[1,-1])
#     deep_feature_embedding_res = tf.reshape(deep_feature_embedding_res,[1,-1])
#
#     wide_embedding_res_list.append(wide_feature_embedding_res)
#     deep_embedding_res_list.append(deep_feature_embedding_res)
#
# wide_embedded_res = tf.concat(wide_embedding_res_list, axis=0) #一个batch内的所有的样例的wide side embedding表示 #[希望是batch_size, xigma of embedding_size]
# deep_embedded_res = tf.concat(deep_embedding_res_list, axis=0) #一个batch内的所有的样例的deep side embedding表示
#
# wide_inputs = wide_embedded_res
# deep_inputs = deep_embedded_res



def my_wide_function(one_example):
    wide_feature_embedding_res_list = []
    features = tf.unstack(one_example,axis = 0)
    for one_feature, tag in zip( features , tags):
        split_tag = tf.string_split(one_feature, "|")
        one_sparse = tf.SparseTensor(
            indices=split_tag.indices,
            values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
            ## 这里给出了不同值通过表查到的index ##
            dense_shape=split_tag.dense_shape
        )
        current_mapping = {tag.tag_name: one_sparse}
        one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, tag.embedding_res)
        # [[ 0.08187684,  0.22063671, -0.16549297]] with the shape of [1,embedding_size]
        if tag.wide_or_deep_side == "wide":
            wide_feature_embedding_res_list.append(one_feature_embedding_res)
    wide_feature_embedding_res = tf.concat(wide_feature_embedding_res_list,axis = -1) #希望是[1，xigma of embedding_size]
    wide_feature_embedding_res = tf.reshape(wide_feature_embedding_res,[1,-1])
    return wide_feature_embedding_res

def my_deep_function(one_example):
    deep_feature_embedding_res_list = []
    features = tf.unstack(one_example, axis=0)
    for one_feature, tag in zip(features, tags):
        split_tag = tf.string_split(one_feature, "|")
        one_sparse = tf.SparseTensor(
            indices=split_tag.indices,
            values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
            ## 这里给出了不同值通过表查到的index ##
            dense_shape=split_tag.dense_shape
        )
        current_mapping = {tag.tag_name: one_sparse}
        one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, tag.embedding_res)
        # [[ 0.08187684,  0.22063671, -0.16549297]] with the shape of [1,embedding_size]
        if tag.wide_or_deep_side == "deep":
            deep_feature_embedding_res_list.append(one_feature_embedding_res)
    deep_feature_embedding_res = tf.concat(deep_feature_embedding_res_list, axis=-1)
    deep_feature_embedding_res = tf.reshape(deep_feature_embedding_res, [1, -1])
    return deep_feature_embedding_res

wide_embedded_res = tf.map_fn(fn=my_wide_function,elems=exp_X,dtype=tf.float32)
deep_embedded_res = tf.map_fn(fn=my_deep_function,elems=exp_X,dtype=tf.float32)

wide_inputs = wide_embedded_res
deep_inputs = deep_embedded_res











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

w_a_d_output = tf.nn.softmax(w_a_d_logit,dim=-1)
loss = tf.nn.sigmoid_cross_entropy_with_logits(
    labels = Y,
    logits=w_a_d_logit,
    name="loss_function"
)
loss_mean = tf.reduce_mean(loss,)
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





data = list(zip(X_data,Y_data))
data_size=len(data)
indices=np.random.permutation(np.arange(data_size))
shufflfed_data=data[indices]
X_batch_data ,Y_batch_data = shufflfed_data[0:GLOBAL_BATCH_SIZE]
import pdb
pdb.set_trace()
train_X = X_batch_data
train_Y = Y_batch_data
eval_X = X_batch_data
eval_Y = Y_batch_data
test_X = X_batch_data
test_Y = Y_batch_data

# 初始化回话sess并开始迭代训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 验证集待喂入数据
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            validate_acc = sess.run(accuracy,feed_dict={X:eval_X,Y:eval_Y})
            print('After %d training steps, validation accuracy'
                  ' using average model is %f' % (i, validate_acc))

        sess.run(train_op,feed_dict={X:train_X,Y:train_Y})

    test_acc = sess.run(accuracy,feed_dict = {X:test_X,Y:test_Y})
    print('After %d training steps, test accuracy'
          ' using average model is %f' % (TRAINING_STEPS, test_acc))

#万事具备，只欠eval/train/test_X/Y的获得了，之后可以通过feed_dict的方式处理了。
#要获得eval/train/test_X/Y 需要搞定pandas，还要搞定获得数据的data load部分