import pandas as pd
import tensorflow as tf
import numpy as np
import json
import pickle
import logging
import time
import os
from collections import OrderedDict
import sklearn
from sklearn import metrics
import pdb

def compute_auc(labels, pred):
    if len(labels) != len(pred):
        print("error labels or pred")
        return 0
    sorted_pred = sorted(range(len(pred)), key=lambda i: pred[i])
    pos = 0.0
    neg = 0.0
    auc = 0.0
    last_pre = pred[sorted_pred[0]]
    count = 0.0
    pre_sum = 0.0  # 当前位置之前的预测值相等的rank之和，rank是从1开始的，所以在下面的代码中就是i+1
    pos_count = 0.0  # 记录预测值相等的样本中标签是正的样本的个数
    for i in range(len(sorted_pred)):
        if labels[sorted_pred[i]] > 0:
            pos += 1
        else:
            neg += 1
        if last_pre != pred[sorted_pred[i]]:  # 当前的预测概率值与前一个值不相同
            # 对于预测值相等的样本rank需要取平均值，并且对rank求和
            auc += pos_count * pre_sum / count
            count = 1
            pre_sum = i + 1  # 更新为当前的rank
            last_pre = pred[sorted_pred[i]]
            if labels[sorted_pred[i]] > 0:
                pos_count = 1  # 如果当前样本是正样本 ，则置为1
            else:
                pos_count = 0  # 反之置为0
        else:
            pre_sum += i + 1  # 记录rank的和
            count += 1  # 记录rank和对应的样本数，pre_sum / count就是平均值了
            if labels[sorted_pred[i]] > 0:  # 如果是正样本
                pos_count += 1  # 正样本数加1

    auc += pos_count * pre_sum / count  # 加上最后一个预测值相同的样本组
    auc -= pos * (pos + 1) / 2  # 减去正样本在正样本之前的情况
    if pos == 0.0 or neg == 0.0:
        return 0.0
    auc = auc / (pos * neg)  # 除以总的组合数
    return auc

features_to_exclude = [
409,410,411,412,413,414,415,416
]

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
        elif 30 < self.featureNum <= 200000:
            self.kind = "inteEmbdding"
            # self.wide_or_deep_side = "wide+deep"
            self.wide_or_deep_side = "deep"
            self.vocab_size = self.featureNum
            self.embedding_size = 25 #写死的
            self.tag_set = tag_set

        #hash的情况
        else:
            if self.featureNum <= 1000000:
                self.kind = "hash"
                self.wide_or_deep_side = "deep"
                self.vocab_size = min(self.featureNum,700000)
                self.embedding_size = 30  # 写死的
                self.tag_set = tag_set
                # self.tag_set = tag_set if self.vocab_size == self.featureNum else []
            else:
                self.kind = "hash"
                self.wide_or_deep_side = "deep"
                self.vocab_size = int( min(self.featureNum , max(100000,self.featureNum * 0.4)) )
                self.embedding_size = 30  # 写死的
                self.tag_set = tag_set
                # self.tag_set = tag_set if self.vocab_size == self.featureNum else []


class WideAndDeep(object):
    def __init__(self,
                 batch_size = 2,
                 feature_num = 1,
                 train_epoch_num = 1,
                 train_steps = 1000,
                 tag2value = None,
                 custom_tags = [],
                 wide_side_node=100,
                 deep_side_nodes=[700,100],
                 video_side_nodes=[300,100],
                 user_side_nodes=[700,100],
                 context_side_nodes=[100,100],
                 eval_freq = 1000,#每隔1000个step，evaluate一次
                 moving_average_decay = 0.99,# 滑动平均的衰减系数
                 learning_rate_base = 1e-1,# 基学习率
                 learning_rate_decay = 0.5,# 学习率的衰减率
                 learning_rate_decay_step=3000,
                 total_example_num = 2000000,
                 train_filename = "",
                 eval_filename = "",
                 test_filename = "",
                 features_to_exclude = [],
                 features_to_keep = [],
                 user_features = [],
                 video_features = [],
                 context_features = [],
                 sim_loss_a  = 0.2,
                 args=None
                 ):

        assert tag2value is not None, "tag2value should be a dict, but found None"
        self.batch_size = batch_size
        self.feature_num = feature_num
        self.train_epoch_num = train_epoch_num
        self.train_steps = train_steps
        self.args = args
        self.logger = logging.getLogger("brc")
        self.tag2value = tag2value
        self.custom_tags = custom_tags
        self.wide_side_node = wide_side_node
        self.deep_side_nodes = deep_side_nodes
        self.video_side_nodes = video_side_nodes
        self.user_side_nodes = user_side_nodes
        self.context_side_nodes = context_side_nodes
        self.sim_loss_a = sim_loss_a
        self.eval_freq = eval_freq
        self.moving_average_decay= moving_average_decay
        self.learning_rate_base = learning_rate_base
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_decay_step = learning_rate_decay_step
        self.total_example_num = total_example_num
        self.train_filename = train_filename
        self.eval_filename = eval_filename
        self.test_filename = test_filename
        self.features_to_exclude = features_to_exclude
        self.features_to_keep = features_to_keep
        self.user_features = user_features
        self.video_features = video_features#user_features我之前将这个错误的搞成了user_features了，但是也有效果的提升，那么解释起来就是user_feature跟自己匹配，这样的话，就会令自己的向量有更好的表征，这也给了我提醒，自己跟自己经过折叠之后的表征做逼近，可以得到更好的自己的表征，之后可以user-user,ad-ad,user-ad,context-context之间逼近
        self.context_features = context_features


        if len(self.features_to_keep) > 0:
            self.tag2value = OrderedDict()
            for key in self.features_to_keep:
                if key in tag2value:
                    self.tag2value[key] = tag2value[key]

        start_t = time.time()
        self.sess = tf.Session()
        self._setup_placeholder()
        self._setup_and_realize_mapping3()
        self._build_graph()
        self._UserMatchVideo()
        self._build_loss()
        self._create_train_op()
        self.saver = tf.train.Saver()
        self.sess.run([tf.global_variables_initializer(),tf.tables_initializer()])


    def _setup_placeholder(self):
        self.X = tf.placeholder(shape=[None,self.feature_num],dtype=tf.string)
        self._Y = tf.placeholder(shape=[None,],dtype=tf.float32)
        self.Y = tf.expand_dims(self._Y,axis=-1)

    def _setup_mappings(self):
        tag2value = sorted(self.tag2value.items(),key = lambda x: x[0])
        tag2value = dict(tag2value)
        # print("tag2value.keys() = ",tag2value.keys())
        # for key in tag2value:
        #     print(key)
        # import pdb
        # pdb.set_trace()

        self.tag2value = tag2value
        tags = []
        # type = dict, key是tag，value是这个tag的所有的可能的取值组成的列表
        for key in tag2value:
            print(" in setup embedding key = ",key)
            tag = Tag(
                featureNum=len(tag2value[key]),
                tag_name=key
            )
            tag.cal_(tag2value[key])
            tags.append(tag)

        tags_to_repair = OrderedDict()

        if self.custom_tags:
            for custom_tag in self.custom_tags:
                tags_to_repair[custom_tag['tag_name']] = custom_tag

        self.tags_to_repair= tags_to_repair

        for tag in tags:
            print("in setup mapping tag.tag_name",tag.tag_name)
            if tag.tag_name in tags_to_repair:
                tag.kind = "custom"
                tag.wide_or_deep_side = "deep"
                tag.embedding_size = tags_to_repair[tag.tag_name]['embedding_size']
                tag.vocab_size = tags_to_repair[tag.tag_name]['vocab_size']
                # tag.tag_set = tags_to_repair[tag.tag_name]['vocab_fun'](tag.tag_set)
                table = tf.contrib.lookup.index_table_from_tensor(mapping=tag.tag_set,
                                                                  default_value=-1)  ## 这里构造了个查找表 ##
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
                                                         initializer=tags_to_repair[tag.tag_name][
                                                             'initializer_function'],
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
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.one_hot_column(one_feature)
            elif tag.kind == "inteEmbdding":
                one_feature = tf.contrib.layers.sparse_column_with_keys(
                    column_name=tag.tag_name,
                    keys=tag.tag_set,
                    default_value=0,
                    combiner='sum',
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.embedding_column(one_feature,
                                                         combiner="mean",
                                                         dimension=tag.embedding_size)
                tag.embedding_res = res
            else:
                one_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
                    column_name=tag.tag_name,
                    hash_bucket_size=tag.vocab_size,
                    combiner="sum",
                    dtype=tf.dtypes.string
                    # dtype=tf.dtypes.int64
                )

                res = tf.contrib.layers.embedding_column(one_feature,
                                                         # initializer=my_initializer,
                                                         combiner="mean",
                                                         dimension=tag.embedding_size)
            tag.embedding_res = res

        wide_side_dimension_size = 0
        deep_side_dimension_size = 0
        for tag in tags:
            print("in setup embedding tag.name = ",tag.tag_name)
            if tag.wide_or_deep_side == "wide":
                wide_side_dimension_size += tag.embedding_size
            else:
                deep_side_dimension_size += tag.embedding_size

        self.wide_side_dimension_size = wide_side_dimension_size
        self.deep_side_dimension_size = deep_side_dimension_size
        print("self.wide_side_dimension_size = ",self.wide_side_dimension_size)
        print("self.deep_side_dimension_size = ",self.deep_side_dimension_size)
        self.tags = tags

    def _realize_mappings(self):
        # exp_X = tf.expand_dims(self.X, axis=-1)
        features = tf.unstack(self.X, axis=1) #List with Feature_NUM ele each with a shape of [batch_size]
        print("len(features) = ",len(features))
        print("shape = ",features[0].shape)
        wide_mappings = OrderedDict()
        wide_tensors = []
        deep_mappings = OrderedDict()
        deep_tensors = []

        for one_feature, tag in zip(features, self.tags):
            print("in realize embedding tag.name = ",tag.tag_name)
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


        for one_feature, tag in zip(features, self.tags):
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


        mappings = OrderedDict()
        tensors = []
        for key in wide_mappings:
            mappings[key] = wide_mappings[key]
        for key in deep_mappings:
            mappings[key] = deep_mappings[key]
        tensors = wide_tensors + deep_tensors
        wide_and_deep_embedding_res = tf.feature_column.input_layer(mappings, tensors)
        print("res.shape = ",wide_and_deep_embedding_res.shape)
        wide_inputs, deep_inputs = tf.split(wide_and_deep_embedding_res, [self.wide_side_dimension_size, self.deep_side_dimension_size],1)

        self.wide_inputs = tf.reshape(wide_inputs, [-1, self.wide_side_dimension_size])
        self.deep_inputs = tf.reshape(deep_inputs, [-1, self.deep_side_dimension_size])

    def _realize_mappings2(self):
        # exp_X = tf.expand_dims(self.X, axis=-1)
        features = tf.unstack(self.X, axis=1)  # List with Feature_NUM ele each with a shape of [batch_size]
        print("len(features) = ", len(features))
        print("shape = ", features[0].shape)
        wide_mappings = OrderedDict()
        wide_tensors = []
        deep_mappings = OrderedDict()
        deep_tensors = []
        self.one_sparse_s = []
        for one_feature, tag in zip(features, self.tags):
            print("in realize embedding tag.name = ", tag.tag_name)
            split_tag = tf.string_split(one_feature, "|")
            one_sparse = tf.SparseTensor(
                indices=split_tag.indices,
                values=tag.table.lookup(split_tag.values) if tag.tag_name == "custom" else split_tag.values,
                ## 这里给出了不同值通过表查到的index ##
                dense_shape=split_tag.dense_shape
            )
            self.one_sparse_s.append(one_sparse)

            if tag.wide_or_deep_side == "wide":
                wide_mappings[tag.tag_name] = one_sparse
                wide_tensors.append(tag.embedding_res)
            else:
                deep_mappings[tag.tag_name] = one_sparse
                deep_tensors.append(tag.embedding_res)


        mappings = OrderedDict()
        tensors = []
        for key in wide_mappings:
            mappings[key] = wide_mappings[key]
        for key in deep_mappings:
            mappings[key] = deep_mappings[key]
        tensors = wide_tensors + deep_tensors
        wide_and_deep_embedding_res = tf.feature_column.input_layer(mappings, tensors)
        print("res.shape = ", wide_and_deep_embedding_res.shape)
        wide_inputs, deep_inputs = tf.split(wide_and_deep_embedding_res,
                                            [self.wide_side_dimension_size, self.deep_side_dimension_size], 1)

        self.wide_inputs = tf.reshape(wide_inputs, [-1, self.wide_side_dimension_size])
        self.deep_inputs = tf.reshape(deep_inputs, [-1, self.deep_side_dimension_size])

    def _setup_and_realize_mapping(self):
        tag2value = sorted(self.tag2value.items(), key=lambda x: x[0])
        tag2value = dict(tag2value)
        self.tag2value = tag2value
        deep_side_dimension_size = 0
        features = tf.unstack(self.X, axis=1)  # List with Feature_NUM ele each with a shape of [batch_size]
        mappings = OrderedDict()
        tensor_s = []
        for one_feature, key in zip(features, self.tag2value):
            print("in setup and realize mapping")
            print("one_features = ",one_feature)
            print("key  = ",key)
            tag = Tag(
                featureNum=len(tag2value[key]),
                tag_name=key
            )
            tag.cal_(tag2value[key])
            split_tag = tf.string_split(one_feature, "|")
            one_sparse = tf.SparseTensor(
                indices=split_tag.indices,
                values=split_tag.values,
                ## 这里给出了不同值通过表查到的index ##
                dense_shape=split_tag.dense_shape
            )
            mappings[key]=one_sparse
            deep_side_dimension_size += tag.embedding_size
            one_feature = tf.contrib.layers.sparse_column_with_keys(
                column_name=tag.tag_name,
                keys=tag.tag_set,
                default_value=0,
                combiner='sum',
                # dtype=tf.dtypes.int64
                dtype=tf.dtypes.string
            )
            res = tf.contrib.layers.embedding_column(one_feature,
                                                     combiner="mean",
                                                     dimension=tag.embedding_size)
            tensor_s.append(res)
        self.deep_inputs = tf.feature_column.input_layer(mappings, tensor_s)

    def _setup_and_realize_mapping2(self):
        tag2value = sorted(self.tag2value.items(), key=lambda x: x[0])
        tag2value = dict(tag2value)
        self.tag2value = tag2value
        features = tf.unstack(self.X, axis=1)  # List with Feature_NUM ele each with a shape of [batch_size]
        mappings = OrderedDict()
        tensor_s = []
        deep_mappings = OrderedDict()
        deep_tensor_s = []
        deep_side_dimension_size = 0
        wide_mappings = OrderedDict()
        wide_tensor_s = []
        wide_side_dimension_size = 0

        Keys = []
        for one_feature, key in zip(features, self.tag2value):
            print("in setup and realize mapping2")
            print("one_features = ",one_feature)
            print("key  = ",key)
            Keys.append(key)
            tag = Tag(
                featureNum=len(tag2value[key]),
                tag_name=key
            )
            tag.cal_(tag2value[key])
            split_tag = tf.string_split(one_feature, "|")
            one_sparse = tf.SparseTensor(
                indices=split_tag.indices,
                values=split_tag.values,
                ## 这里给出了不同值通过表查到的index ##
                dense_shape=split_tag.dense_shape
            )

            if tag.kind == "one_hot":
                one_feature = tf.contrib.layers.sparse_column_with_keys(
                    column_name=tag.tag_name,
                    keys=tag.tag_set,
                    default_value=-1,
                    combiner='sum',
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.one_hot_column(one_feature)
            elif tag.kind == "inteEmbdding":
                one_feature = tf.contrib.layers.sparse_column_with_keys(
                    column_name=tag.tag_name,
                    keys=tag.tag_set,
                    default_value=0,
                    combiner='sum',
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.embedding_column(one_feature,
                                                         combiner="mean",
                                                         dimension=tag.embedding_size)
            else:
                one_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
                    column_name=tag.tag_name,
                    hash_bucket_size=tag.vocab_size,
                    combiner="sum",
                    dtype=tf.dtypes.string
                    # dtype=tf.dtypes.int64
                )

                res = tf.contrib.layers.embedding_column(one_feature,
                                                         # initializer=my_initializer,
                                                         combiner="mean",
                                                         dimension=tag.embedding_size)
            if tag.wide_or_deep_side == "wide":
                wide_mappings[key] = one_sparse
                wide_tensor_s.append(res)
                wide_side_dimension_size += tag.embedding_size
            else:
                deep_mappings[key] = one_sparse
                deep_tensor_s.append(res)
                deep_side_dimension_size += tag.embedding_size



        for key in wide_mappings:
            mappings[key] = wide_mappings[key]
        for key in deep_mappings:
            mappings[key] = deep_mappings[key]
        tensor_s = wide_tensor_s + deep_tensor_s
        wide_and_deep_embedding_res = tf.feature_column.input_layer(mappings, tensor_s)

        print("res.shape = ", wide_and_deep_embedding_res.shape)
        wide_inputs, deep_inputs = tf.split(wide_and_deep_embedding_res,
                                            [wide_side_dimension_size, deep_side_dimension_size], 1)

        self.wide_inputs = tf.reshape(wide_inputs, [-1, wide_side_dimension_size])
        self.deep_inputs = tf.reshape(deep_inputs, [-1, deep_side_dimension_size])

    def _setup_and_realize_mapping3(self):
        tag2value = sorted(self.tag2value.items(), key=lambda x: x[0])
        tag2value = dict(tag2value)
        self.tag2value = tag2value
        features = tf.unstack(self.X, axis=1)  # List with Feature_NUM ele each with a shape of [batch_size]
        mappings = OrderedDict()
        tensor_s = []
        deep_user_mappings = OrderedDict()
        deep_user_tensor_s = []
        deep_user_dimension_size = 0
        deep_video_mappings = OrderedDict()
        deep_video_tensor_s = []
        deep_video_dimension_size = 0
        deep_context_mappings = OrderedDict()
        deep_context_tensor_s = []
        deep_context_dimension_size = 0
        wide_user_mappings = OrderedDict()
        wide_user_tensor_s = []
        wide_user_dimension_size = 0
        wide_video_mappings = OrderedDict()
        wide_video_tensor_s = []
        wide_video_dimension_size = 0
        wide_context_mappings = OrderedDict()
        wide_context_tensor_s = []
        wide_context_dimension_size = 0

        Keys = []
        for one_feature, key in zip(features, self.tag2value):
            print("in setup and realize mapping2")
            print("one_features = ",one_feature)
            print("key  = ",key)
            Keys.append(key)
            tag = Tag(
                featureNum=len(tag2value[key]),
                tag_name=key
            )
            tag.cal_(tag2value[key])
            split_tag = tf.string_split(one_feature, "|")
            one_sparse = tf.SparseTensor(
                indices=split_tag.indices,
                values=split_tag.values,
                ## 这里给出了不同值通过表查到的index ##
                dense_shape=split_tag.dense_shape
            )

            if tag.kind == "one_hot":
                one_feature = tf.contrib.layers.sparse_column_with_keys(
                    column_name=tag.tag_name,
                    keys=tag.tag_set,
                    default_value=-1,
                    combiner='sum',
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.one_hot_column(one_feature)
            elif tag.kind == "inteEmbdding":
                one_feature = tf.contrib.layers.sparse_column_with_keys(
                    column_name=tag.tag_name,
                    keys=tag.tag_set,
                    default_value=0,
                    combiner='sum',
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.embedding_column(one_feature,
                                                         combiner="mean",
                                                         dimension=tag.embedding_size)
            else:
                one_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
                    column_name=tag.tag_name,
                    hash_bucket_size=tag.vocab_size,
                    combiner="sum",
                    dtype=tf.dtypes.string
                    # dtype=tf.dtypes.int64
                )

                res = tf.contrib.layers.embedding_column(one_feature,
                                                         # initializer=my_initializer,
                                                         combiner="mean",
                                                         dimension=tag.embedding_size)
            if tag.wide_or_deep_side == "wide":
                if tag.tag_name in self.user_features:
                    wide_user_mappings[key]=one_sparse
                    wide_user_tensor_s.append(res)
                    wide_user_dimension_size += tag.embedding_size
                elif tag.tag_name in self.video_features:
                    wide_video_mappings[key] = one_sparse
                    wide_video_tensor_s.append(res)
                    wide_video_dimension_size += tag.embedding_size
                else:
                    wide_context_mappings[key] = one_sparse
                    wide_context_tensor_s.append(res)
                    wide_context_dimension_size += tag.embedding_size
            else:
                if tag.tag_name in self.user_features:
                    deep_user_mappings[key] = one_sparse
                    deep_user_tensor_s.append(res)
                    deep_user_dimension_size += tag.embedding_size
                elif tag.tag_name in self.video_features:
                    deep_video_mappings[key] = one_sparse
                    deep_video_tensor_s.append(res)
                    deep_video_dimension_size += tag.embedding_size
                else:
                    deep_context_mappings[key] = one_sparse
                    deep_context_tensor_s.append(res)
                    deep_context_dimension_size += tag.embedding_size



        for key in wide_user_mappings:
            mappings[key] = wide_user_mappings[key]
        for key in wide_video_mappings:
            mappings[key] = wide_video_mappings[key]
        for key in wide_context_mappings:
            mappings[key] = wide_context_mappings[key]
        for key in deep_user_mappings:
            mappings[key] = deep_user_mappings[key]
        for key in deep_video_mappings:
            mappings[key] = deep_video_mappings[key]
        for key in deep_context_mappings:
            mappings[key] = deep_context_mappings[key]

        tensor_s = wide_user_tensor_s + wide_video_tensor_s + wide_context_tensor_s + deep_user_tensor_s + deep_video_tensor_s + deep_context_tensor_s
        wide_and_deep_embedding_res = tf.feature_column.input_layer(mappings, tensor_s)

        print("res.shape = ", wide_and_deep_embedding_res.shape)
        wide_user_inputs, wide_video_inputs,\
        wide_context_inputs, deep_user_inputs,\
        deep_video_inputs, deep_context_inputs = tf.split(wide_and_deep_embedding_res,
                                            [wide_user_dimension_size,
                                             wide_video_dimension_size,
                                             wide_context_dimension_size,
                                             deep_user_dimension_size,
                                             deep_video_dimension_size,
                                             deep_context_dimension_size], 1)

        self.wide_inputs = tf.concat([wide_user_inputs,wide_video_inputs,wide_context_inputs],axis = 1)
        self.deep_inputs = tf.concat([deep_user_inputs,deep_video_inputs,deep_context_inputs],axis = 1)

        self.user_inputs = tf.concat([wide_user_inputs,deep_user_inputs],axis = 1)
        self.video_inputs = tf.concat([wide_video_inputs,deep_video_inputs],axis = 1)
        self.context_inputs = tf.concat([wide_context_inputs,deep_context_inputs],axis = 1)

        self.wide_side_dimension_size = wide_user_dimension_size + wide_video_dimension_size + wide_context_dimension_size
        self.deep_side_dimension_size = deep_user_dimension_size + deep_video_dimension_size + deep_context_dimension_size

        self.user_side_dimension_size = wide_user_dimension_size + deep_user_dimension_size
        self.video_side_dimension_size = wide_video_dimension_size + deep_video_dimension_size
        self.context_side_dimension_size = wide_context_dimension_size + deep_context_dimension_size

    def _UserMatchVideo(self):
        with tf.variable_scope("UserNet"):
            user_side=self.user_inputs
            for k in range(len(self.user_side_nodes)):
                if k != len(self.user_side_nodes) - 1:
                    user_side = tf.contrib.layers.fully_connected(user_side, self.user_side_nodes[k], activation_fn=tf.nn.relu,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                else:
                    user_side = tf.contrib.layers.fully_connected(user_side, self.user_side_nodes[k], activation_fn=tf.nn.sigmoid,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                user_side = tf.layers.dropout(
                    inputs=user_side,
                    rate=0.5,
                    name="deep_dropout_%d" % k,
                )

        with tf.variable_scope("VideoNet"):
            video_side=self.video_inputs
            for k in range(len(self.video_side_nodes)):
                if k != len(self.video_side_nodes) - 1:
                    video_side = tf.contrib.layers.fully_connected(video_side, self.video_side_nodes[k], activation_fn=tf.nn.relu,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                else:
                    video_side = tf.contrib.layers.fully_connected(video_side, self.video_side_nodes[k], activation_fn=tf.nn.sigmoid,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                video_side = tf.layers.dropout(
                    inputs=video_side,
                    rate=0.5,
                    name="deep_dropout_%d" % k,
                )

        with tf.variable_scope("ContextNet"):
            context_side=self.context_inputs
            for k in range(len(self.context_side_nodes)):
                if k != len(self.context_side_nodes) - 1:
                    context_side = tf.contrib.layers.fully_connected(context_side, self.context_side_nodes[k], activation_fn=tf.nn.relu,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                else:
                    context_side = tf.contrib.layers.fully_connected(context_side, self.context_side_nodes[k], activation_fn=tf.nn.sigmoid,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                context_side = tf.layers.dropout(
                    inputs=context_side,
                    rate=0.5,
                    name="deep_dropout_%d" % k,
                )

        # user_side = tf.contrib.layers.fully_connected(user_side,self.video_side_nodes[-1],
        #                                   activation_fn=None,
        #                                   weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
        #                                   biases_initializer=None)
        self.sim_user_video = tf.reduce_sum( tf.multiply( user_side, video_side ), 1 )

    def _build_graph(self):
        central_bias = tf.Variable(name='central_bias',
                                   initial_value=tf.random_normal(shape=[2], mean=0, stddev=0),
                                   trainable=True)
        with tf.variable_scope("WideNet"):
            wide_side = tf.contrib.layers.fully_connected(inputs=self.wide_inputs,
                                                          num_outputs=2,
                                                          activation_fn=tf.nn.relu,
                                                          biases_initializer=None,
                                                          weights_regularizer=tf.contrib.layers.l1_regularizer(0.1)
                                                          )


        wide_side = tf.reshape(wide_side, [-1, 2])
        w_a_d = tf.concat([self.wide_inputs, self.deep_inputs], axis=1, name="concat")

        with tf.variable_scope("DeepNet"):
            for k in range(len(self.deep_side_nodes)):
                if k != len(self.deep_side_nodes) - 1:
                    w_a_d = tf.contrib.layers.fully_connected(w_a_d, self.deep_side_nodes[k], activation_fn=tf.nn.relu,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                else:
                    w_a_d = tf.contrib.layers.fully_connected(w_a_d, self.deep_side_nodes[k], activation_fn=tf.nn.relu,
                                                              weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                              )
                w_a_d = tf.layers.dropout(
                    inputs=w_a_d,
                    rate=0.5,
                    name="deep_dropout_%d" % k,
                )
            deep_side = tf.contrib.layers.fully_connected(w_a_d, 2,
                                                          activation_fn=None,
                                                          biases_initializer=None,
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                          )
        deep_side = tf.reshape(deep_side, [-1, 2])
        w_a_d_logit = tf.add(deep_side, wide_side)
        self.w_a_d_logit = tf.add(w_a_d_logit, central_bias, name="wide_with_bias")
        self.w_a_d_logit_after_soft = tf.nn.softmax(self.w_a_d_logit,dim=-1)
        self.w_a_d_output = tf.nn.softmax(self.w_a_d_logit, dim=-1)[:,1]
        self.predictions = tf.argmax(self.w_a_d_logit,dimension=1)
        self.correct_prediction = tf.equal(self.predictions, tf.cast(self.Y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def _build_graph_only_deep(self):
        central_bias = tf.Variable(name='central_bias',
                                   initial_value=tf.random_normal(shape=[2], mean=0, stddev=1),
                                   trainable=True)
        w_a_d = self.deep_inputs
        with tf.variable_scope("DeepNet"):
            for k in range(len(self.deep_side_nodes)):
                w_a_d = tf.contrib.layers.fully_connected(w_a_d, self.deep_side_nodes[k], activation_fn=tf.nn.sigmoid,
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                          )
                w_a_d = tf.layers.dropout(
                    inputs=w_a_d,
                    rate=0.5,
                    name="deep_dropout_%d" % k,
                )
            deep_side = tf.contrib.layers.fully_connected(w_a_d, 2,
                                                          activation_fn=None,
                                                          biases_initializer=None,
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(0.1)
                                                          )

        self.w_a_d_logit = tf.add(deep_side, central_bias, name="wide_with_bias")
        self.w_a_d_logit_after_soft = tf.nn.softmax(self.w_a_d_logit,dim=-1)
        self.w_a_d_output = tf.nn.softmax(self.w_a_d_logit, dim=-1)[:, 1]
        self.predictions = tf.argmax(self.w_a_d_logit, dimension=1)
        self.correct_prediction = tf.equal(self.predictions, tf.cast(self.Y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def _build_loss(self):
        # self.total_loss = -tf.reduce_mean(self.Y * tf.log(tf.clip_by_value(self.w_a_d_output,1e-10,1.0)))

        self.total_loss =tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.one_hot(tf.cast(self._Y,dtype=tf.int64),depth=2),
            logits=self.w_a_d_logit,
            name="loss_function"
        ))

        sim_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self._Y,
            logits=self.sim_user_video,
        ))
        self.total_loss += self.sim_loss_a * sim_loss
        # self.total_loss = tf.reduce_mean( tf.nn.weighted_cross_entropy_with_logits(logits=self.w_a_d_logit,
        #                                                            targets=tf.one_hot(tf.cast(self._Y,dtype=tf.int64),depth=2),
        #                                                             pos_weight=10) )

        # self.total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     labels = tf.one_hot(tf.cast(self.Y,dtype=tf.int32), depth=2),
        #     logits = self.w_a_d_logit,
        # ))
        # loss_mean = tf.reduce_mean(loss)
        # reg_ds = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'DeepNet')
        # reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'WideNet')
        # d_regu_loss = tf.reduce_sum(reg_ds)
        # w_regu_loss = tf.reduce_sum(reg_ws)
        # # self.total_loss = loss_mean + d_regu_loss + w_regu_loss
        # self.total_loss = loss_mean

    def _create_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)  # 定义存储当前迭代训练轮数的变量

        # 定义ExponentialMovingAverage类对象
        self.variable_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, self.global_step)  # 传入当前迭代轮数参数
        # 定义对所有可训练变量trainable_variables进行更新滑动平均值的操作op
        trainable_vars = tf.trainable_variables()
        self.variables_averages_op = self.variable_averages.apply(trainable_vars)


        # 定义指数衰减学习率
        # self.learning_rate = tf.train.exponential_decay(self.learning_rate_base, self.global_step,
        #                                    2000000 / self.batch_size, self.learning_rate_decay)
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base, self.global_step,
                                                        self.learning_rate_decay_step, self.learning_rate_decay)
        # 定义梯度下降操作op，global_step参数可实现自加1运算
        # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate) \
        #     .minimize(self.total_loss, global_step=self.global_step)
        # self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss)

        self.train_step = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.total_loss, global_step=self.global_step)
        # 组合两个操作op
        self.train_op = tf.group(self.train_step, self.variables_averages_op)

    def load_data(self,filename = "/home2/data/ttd/zhengquan_test.processed.csv.pkl"):
        df_data = pickle.load(open(filename, "rb"))  # 一个DataFrame
        print("df_data_columns = ",df_data.columns.values.tolist())
        if len(self.features_to_keep) > 0:
            print("run features_to_keep")
            print(self.features_to_keep)
            df_data = df_data[self.features_to_keep]
        elif len(self.features_to_exclude) > 0:
            print("run features_to_exclude")
            print(self.features_to_exclude)
            self.features_to_exclude = list(map(str,self.features_to_exclude))
            df_data = df_data.drop(self.features_to_exclude,axis = 1)
        df_data = df_data.dropna(how="all", axis=0)  # 0 对行进行操作，how='any'只要有一个NA就删除这一行，how='all'必须全部是NA才能删除这一行
        # 不能用any过滤，否则过滤完了，1000个只剩3个。
        df_data['label'] = (df_data['label']).astype(int)

        # df_data = df_data[df_data['label'].isin([0, 1])]  # 只保留label为0或者1的
        positive_samples = df_data[df_data['label'].isin([1])]
        positive_samples = positive_samples.sample(frac=1).reset_index(drop=True)
        positive_num = positive_samples.shape[0]
        negative_samples = df_data[df_data['label'].isin([0])]
        negative_samples = negative_samples.sample(frac=1).reset_index(drop=True)
        negative_num = negative_samples.shape[0]
        min_num = min(positive_num,negative_num)
        positive_samples = positive_samples[0:min_num]
        negative_samples = negative_samples[0:min_num]
        df_data = pd.concat([positive_samples,negative_samples],axis=0)
        df_data = df_data.sample(frac=1).reset_index(drop=True)
        # 分离X,Y
        X_data = df_data.drop(['label'], axis=1)
        # X_data = X_data.values.astype(str) #MemoryError
        X_data = X_data = X_data.applymap(str)
        print("df_data_columns = ",df_data.columns.values.tolist())
        print("self.tag2value.keys() = ",self.tag2value.keys())
        # import pdb
        # pdb.set_trace()
        X_data = X_data.values
        Y_data = df_data['label'].values.astype(np.int32)
        return X_data,Y_data

    def load_batch_data(self,data):
        data = np.array(data)
        data_size = len(data)
        num_batchs_per_epchs = int((data_size - 1) / self.batch_size) + 1
        indices = np.random.permutation(np.arange(data_size))
        # indices = np.arange(data_size)
        shufflfed_data = data[indices]
        for batch_num in range(num_batchs_per_epchs):
            start_index = batch_num * self.batch_size
            # end_index = min((batch_num + 1) * self.batch_size, data_size)
            end_index = (batch_num + 1) * self.batch_size
            if end_index > data_size:
                yield shufflfed_data[data_size - self.batch_size : data_size]
            else:
                yield shufflfed_data[start_index:end_index]

    def train(self,train_filename="",eval_filename=""):
        if train_filename:
            self.X_data, self.Y_data = self.load_data(train_filename)
        else:
            self.X_data, self.Y_data = self.load_data(self.train_filename)
        # Test_Example_Num = 20
        # self.X_data = self.X_data[:Test_Example_Num]
        # self.Y_data = self.Y_data[:Test_Example_Num]
        print("Train data with the shape : ",self.X_data.shape)
        if eval_filename:
            self.eval_X_data, self.eval_Y_data = self.load_data(eval_filename)
        else:
            self.eval_X_data, self.eval_Y_data = self.load_data(self.eval_filename)
        train_steps = 0
        history_acc = 0
        history_auc = 0
        start_t = time.time()
        # eval_acc, eval_auc, eval_auc2, eval_loss = self.evaluate(eval_filename) if eval_filename else self.evaluate()
        # print("初始时随机预测")
        # print("train_steps=%d, auc=%f, auc2=%f, acc=%f, loss=%f" % (train_steps, eval_auc, eval_auc2, eval_acc, eval_loss))

        for epoch in range( self.train_epoch_num ):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            print('Training the model for epoch {}'.format(str(epoch)))
            batch_data = self.load_batch_data(data = list(zip(self.X_data,self.Y_data)))
            idx_after_eval = 0
            for idx, current_batch_data in enumerate(batch_data):
                # print("current_batch_data.shape = ",current_batch_data.shape)
                if current_batch_data.shape[0] != self.batch_size:
                    continue
                x_feed_in , y_feed_in = zip(*current_batch_data)
                # x_feed_in = [
                #     ['1139106401876931908', '15886857682112188718',
                #      '6728750808792419899', '10377478459526382958',
                #      '11402135692967554962', '15024105355511557174',
                #      '18279728417322692022', '2671595363826300260',
                #      '6097686670438894417', '14337747038263225241',
                #      '9256938326208594408', '16178348525114398855',
                #      '17612717209987694723', '16034417821505887034',
                #      '14795403760074392295', '5505945717242741188',
                #      '16351098390554857849', '15528266426035309131',
                #      '5767447939531708429', '16991271114131988673',
                #      '6207627389625379132', '17716678521250976184',
                #      '18174542692050568972|10666997157873911510|9242694767697709935|3234171145478725144|18243827533795425813',
                #      '7546267194602827514|1257130021676035005|13246810706923501185|12073309690357165243|13110872298757764732|1764994469626041175|7739609129333920694|2395056743730592370|15492244529449170205|9963868576079343502|12373945533678155799',
                #      '16915856613620591395|6681652090518917235|5932236865381570399|12103621046530019384',
                #      '4330268761743707385|3185643772996816325|9473240722520540265|7124092986950762390|11838339773470972085|4150017948190452204|15407271740473800904|17435082456261220533|2036866169768483380|694286831697819443',
                #      '4362507278181596706|4074482704488081908|11091734370612424491|5853276300864683932|9931345680415856004|10962769516721223439',
                #      '13914985367412221773|8152940895640221691|14422655417779893512|13489914433925931089|12523448448389958802|10726272168340292362|4179621030749388969|12509561851330293329|4150503493031245684|4639278698913908722|17291063626640236520',
                #      '8121547862520414133|1373422643066879105|859089644799545581',
                #      '12800102174142460566|12367891195136022562|10659552055988159232',
                #      '10059699340734965712|12049869669426859125|8543036108439378445',
                #      '5351162946619210970|5253900460750461180|16596786958231533176|10407942323907252031|16131844784848082957|3496972008877927055|5350182168293007753',
                #      '7811488656042211780', '9843582546740847637',
                #      '10000921495557861310',
                #      '9599757397698941319|12882092791947180585|9985691954296525681|9749922333824373647|6458604640602229838|2863944244915615616|4122795543525495023|17244296364053370323|3344430547571097211|6226265894444545464|7171223261059018955|16563333699799260914|13815204553741649386|256587301150032696|13540517897365098005|4388240609884627767|11105550847475148540|13186534013092783706|9222576141942162344|12710165122445826601|16995098405332169247|8417849217224411667|16112383033954400690|9064735758079776116|8297271482419888216|11053619936926543723|14784576209109005743|13085873372114590072|14193205276994058047|13343409586525297934|1814613925210990837|846824784310137869|3131336491916387685|10108937232460993497|6932673226802345393|17572641801158488040|4729498561683101304|848104761674096381|2767684814077469931|16219198003171812605|17518003909770277608|15456700447101791260|5877920276518501476|6060536589676000783|3285531152702862108|14002351444350655349|14605238826674425709|9119312227923054543|8248440891961811759|13554457757269229472',
                #      '12772073413713612088', '4001964435720', '13035244670563100228',
                #      '12401653084618873790', '17538469681306235154',
                #      '15463729423859659656', '11193800870197892099',
                #      '12061933392540942855', '13375481503743282034',
                #      '6313134508873819159', '16568006616906105142',
                #      '9092449348627646249', '2850821598667849033',
                #      '5287679808889998677', '13181628879923256078',
                #      '4543728636134850388', '17379159481797219389',
                #      '14997531903357524035',
                #      '17078877138053630755|10004762522140879703|1788685576844260246|4201261920374413950',
                #      '9950741179107800030', '14557900884772461431',
                #      '17538744310877726682', '7442330856402383951',
                #      '10075985356981492474', '7473350022634778048',
                #      '11338586754503425561', '2183047305502583255',
                #      '1977326967590208941|5364905212705560023|640817657024447067|14080167006232581162',
                #      'nan', '1193847170372156533',
                #      '15284902483641533210|12221533502322966836|11235046423293729184|8465382941717138578|5937796759630084842|18298698624126439301',
                #      'nan',
                #      '8853786067107308264|6289046359270496723|15173563523220421811|685946250852437039|9843368054632184177|18270952862813757515|9949943340061907167|5750327010024418111|2052952327102699285|5919824838978404843|8073362412320421392|4737521332565394934',
                #      '12867317216118195630',
                #      '14313140575369553501|6306771619204665851|5600606819574543533|5600606819574543533|17760980775691980106',
                #      'nan',
                #      '3985704400231958185|12763924833142121938|5651831115566953833|5651831115566953833|7226992376149310776|2566823960511015888|9335824699591894168|6864559765696487226|6738438620259711053|10467381234105993990',
                #      'nan', '2769397328760428245', '12938188901629621628',
                #      '11663257550773569111|14136955673182458146|17165656632268531954|8290740630219577123',
                #      '2048498117375500834|1721935850813209930|9475233373340172859|13194941964409812406',
                #      '1507570432033665141|4579897228912532047|9803248088706390563|5717083530866043383',
                #      '13473257906015313553', '14179110932626403756',
                #      '7246114893777235673', '9384298211912532142',
                #      '17090763658850109906', '4803820018709242041',
                #      '2373829419393294528', '5318045735436314968',
                #      '15572697263701988209', '14394543285647607735',
                #      '921722677166023324', 'nan', 'nan', 'nan', 'nan', 'nan', 'nan',
                #      'nan', 'nan', 'nan', '8357868485411169975', '-4901807207409367903',
                #      '-7476375841979702617', '-8545962712856845286',
                #      '1040638468292150727', '3536429987980764205', '-41740490601023585',
                #      '-3070712613269130595|5703169950741268706|-4595338982389432466|-6835692887479711209|5984435686037929123|-2743526132431467218|-2701620779366995535|1808984031833209046|6787495781425891441|5812290129867201580|-5171650664547851941|5985373613494082505|-2969520433431289645|-3857111849800370413|7201309492178522175|-4811470696229060665|-2045897304710891986|-2775985688890127679|-56593676536433456|-9008834830563589132|-4491998333855381759|-636860706041434799|5863981040936635527|788314673269688711|-6775379162440603530|5132899734712512741|-3671696293618759968',
                #      '16006896213674570733'],
                #     ['1139106401876931908', '14494090190267135201',
                #      '16696715067022626927',
                #      '13611105671480456336|13846756255604062899',
                #      '14824439082751082246', '685724864613537498',
                #      '4251165612741912349', '2671595363826300260',
                #      '2809543020566612793', '14337747038263225241',
                #      '8751927998099084487', '11136140737820388483',
                #      '15034855917252918619', '16034417821505887034',
                #      '14795403760074392295', '16621533179520687294',
                #      '8721435597913118903', '17798765492018937819',
                #      '15823670350982649560', '18313957856019683958',
                #      '13312733396855342027', '9230881386301056674',
                #      '9242694767697709935|18374571413799270557|14471994071354243037|2628513590576265569',
                #      '4009450669004633885|6753380984975528460|11286069056851975991|17925196220538044612|14007625606920688152|15328246749085724110|4868760147675277805|1345456525795157806|3307174997296956342|17385740560087637394|12963139468084193138',
                #      'nan', '6784956121099692574', 'nan', 'nan', 'nan', 'nan', 'nan',
                #      'nan', '7811488656042211780', '9843582546740847637',
                #      '10001135467711829267',
                #      '556291926545000575|13810336100469131807|13797370410919396296|10799379762971157077|15614987872520255152|13292758155353783131|4238612923336311283|2933294269859425860|1499394419278583621|10560388138664624271|1993474683759564950|2493053781577006715|9535778670578504549|12763492470515435592|17394883215608057972|10852642973750327415|8386333473009113251|8864125557149075216|17757265672690284275|1104465636038562242|11285620291338985955|16904013283166065032|3592628795915754608|7227323852473205174|15563375416554331969|11827550810351423390|5167074922926304415|15239043919448762269|1824324491150352859|16497143523408477933|3623075852543453933|16431476937790674334|12609598109296112668|8051616650854517223|12835944122145309517|5411525356518502969|8536115216572765285|2644106523247636838|402837406898832403|13915299053934747968|11256510620050430940|15936479723333094724|12839357132663877518|5770860399479794488|3747239903613139452|16221016807763549030|3127080596286031988|10439665700520350792|6201716774701812077|2414703134415072560',
                #      '808603073452746582', '17543732098672673662',
                #      '17902127367658648365|6101991332976481787', '9205558503199733543',
                #      '7469549925176692699', '3291427149295744014',
                #      '8388493045216528825', '16679783394712196096',
                #      '9712580795729967372', '12056540504885013596',
                #      '2232198050115259183', '6219411346622652094',
                #      '10337476022670035005|6254132313694251057', '17346979826827917216',
                #      '3702044719496915389', '14007835644598087958',
                #      '17484939553209789146', '1615396891009521569', 'nan',
                #      '11982203661804483916', '14154332103682892876',
                #      '5921069929320453728', '5766983250939418596',
                #      '5245652291015113912', '2734458555338968650',
                #      '11338586754503425561', '7332134400676384080', 'nan', 'nan', 'nan',
                #      '9142171976169053903|2213310578074290203|14314322544560726612|185809800395964470|7967014680745744988|16966973451539237912|1737835945871842211|8465382941717138578|5937796759630084842',
                #      '10834885338915739975|2835323396247712370|3072275408300298705|13186411982248420316|2954605066296153856|8919379937287126630|14545041589237172228|4068776906656605775|1715544949675085138|1558299842412499049|14095728536345456547',
                #      '8303638342979969211|2288996264235197094|16105858993952977786|18107532300480691119|17875236985249840684|3447978581584682075|2052952327102699285|13914497029692748048',
                #      '9447129501164265773', 'nan', 'nan',
                #      '3985704400231958185|3985704400231958185|3985704400231958185|3985704400231958185|5651831115566953833|5651831115566953833|5651831115566953833|2836157336361803747|7364315006300976718|7226992376149310776|7226992376149310776|7226992376149310776|7226992376149310776|2566823960511015888|2566823960511015888|2566823960511015888|2566823960511015888|11614319480510162860|6864559765696487226|6738438620259711053',
                #      '17608587270731759045|1742645719589358136|1742645719589358136|10189428290434562479|6249809173138426562|4984826707099565251|4984826707099565251|16016356843999198851|16016356843999198851|16016356843999198851|5791817768996543691|5791817768996543691|4807796683661617305|8351210473168019268|8351210473168019268|207258209663041434|207258209663041434|18412743071571349924|18412743071571349924|18412743071571349924',
                #      '14828308200619590503', '12938188901629621628', 'nan', 'nan',
                #      'nan', '2910402673942410997', '18012435707096466599',
                #      '7246114893777235673', '611708628506443546',
                #      '17090763658850109906', '4803820018709242041',
                #      '15337821239243262395', '5183682098020051672', 'nan',
                #      '14394543285647607735', '921722677166023324', 'nan', 'nan', 'nan',
                #      'nan', 'nan', 'nan', 'nan', 'nan', '6218493924648197871',
                #      '2137428766905285608', '4519559717884557247',
                #      '4769354302259644961', '-3591347114160427991',
                #      '1040638468292150727', '5523506462945413747', '703022405956316884',
                #      '3194007223683290055|-428574572934160435|-6324852920221193347|2488664616070565408|-2783060833925245100|6164470339584946534|5126475097765232806|-1269678415037554336|-3548285716343282862|3391899272616577533|6420183120877045981|-8913680371215120376|7528064057748830922|-7514681234621477388|8651490296704511752|6983880462389746264|1842355116095442839|-1767155784319745736|8922962296530306948|-7252309211894572578|-5303641774946588715|-3996652901780635472|697787283636272338|944838277085011512|7640555975982704883|-6242115325085084595|5927526218393245506|4559590830992069460|-2588292879165980768|4983943125257717986|-8489808950375370361|-1435272844516103527|1513815002973440325|8999307921174807645|-9147182495838388998|-4328968574439195205|-7808740903836271779|-8603098472951612726|2999743338497402351|7387318967265100344|6808590278670286408|2634831093497450055|-5026305846522309366|6493673911848439400|-8080810860723864653|8179276711650732170|282665720207070471|-5860213028826349914|980339078567592417|-135502708712589743|3200480787742050935|-4575807230995903803|-4537127082018387268|4968290977681631322|-1311513358649797837|5459140314403062295|3179811452724025382|-3293489628328133746|4656426793424929207|905214033996351390|-8935986905850427452|-4271162229535350909|-1092206458930581198|-38519176735116705|-4275315803708023806|-6097829377910672750|-1321358106691257943|-5873925154908139225|4065303735504010424|-6835692887479711209|-5151298432190689037|-2103966144490514712|-3896375322115732313|2714844444812651816|-8458322637585443333|-1005156623180731559|4002730219522623739',
                #      '5704043386056474865']
                # ]
                # y_feed_in = (1, 0)
                if idx_after_eval < 10:
                    _,current_loss,current_accuracy,x,y,pred,w_a_d_logit_softmax,w_a_d_logit,deep_inputs = self.sess.run([self.train_op,self.total_loss,self.accuracy,self.X,self.Y,self.predictions,self.w_a_d_logit_after_soft,self.w_a_d_logit,self.deep_inputs],feed_dict={self.X:x_feed_in,self._Y:y_feed_in})
                    # print("x_feed_in = ",x_feed_in)
                    # print("y_feed_in = ",y_feed_in)
                    # print("x = ",x)
                    print("y = ",y)
                    print("pred= ",pred)
                    print("w_a_d_logit_softmax = ",w_a_d_logit_softmax)
                    # print("w_a_d_logit = ",w_a_d_logit)
                    # print("deep_inputs[0] = ",deep_inputs[0])
                    # pdb.set_trace()
                else:
                    _, current_loss, current_accuracy = self.sess.run([self.train_op, self.total_loss, self.accuracy],
                                                                      feed_dict={self.X: x_feed_in, self._Y: y_feed_in})
                idx_after_eval += 1
                print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()),"idx = ",idx," train_steps = ",train_steps, " current loss = ",current_loss," current accuracy = ",current_accuracy)
                train_steps += 1
                if train_steps % self.eval_freq == 0:
                    idx_after_eval = 0
                    self.logger.info('Time to run {} steps : {} s'.format(train_steps,time.time() - start_t))
                    print('Time to run {} steps : {} s'.format(train_steps,time.time() - start_t))
                    start_t = time.time()
                    eval_acc, eval_auc, eval_auc2, eval_loss = self.evaluate(eval_filename) if eval_filename else self.evaluate()
                    print('Time to run {} steps : {} s'.format(train_steps,time.time() - start_t))
                    start_t = time.time()
                    print("epoch = %d, train_steps=%d, auc=%f, auc2=%f, acc=%f, loss=%f" % (epoch, train_steps, eval_auc, eval_auc2, eval_acc, eval_loss))
                    if eval_auc > history_auc or (eval_auc == history_auc and eval_acc > history_acc) :
                        self.save_model(save_dir="/home2/data/zhengquan/batch_32_lr_01_UserId_VideoId/",prefix="auc=%f"%(eval_auc))
                        history_auc = eval_auc
                        history_acc = eval_acc
                        print("epoch = %d, train_steps=%d, auc=%f, auc2=%f, acc=%f, loss=%f, get better score"%(epoch,train_steps,eval_auc, eval_auc2, eval_acc,eval_loss))
                        self.logger.info("epoch = %d, train_steps=%d, auc=%.f, auc2=%f, acc=%.f"%(epoch,train_steps,eval_auc, eval_auc2,eval_acc))

        eval_acc, eval_auc, eval_auc2, eval_loss = self.evaluate(eval_filename) if eval_filename else self.evaluate()
        print('Time to run {} steps : {} s'.format(train_steps, time.time() - start_t))
        start_t = time.time()
        print("epoch = %d, train_steps=%d, auc=%f, acc=%f, loss=%f" % (epoch, train_steps, eval_auc, eval_acc,eval_loss))
        if eval_auc > history_auc or (eval_auc == history_auc and eval_acc > history_acc):
            self.save_model(save_dir="/home2/data/zhengquan/batch_32_lr_01/", prefix="auc=%f" % (eval_auc))
            history_auc = eval_auc
            history_acc = eval_acc
            print("epoch = %d, train_steps=%d, auc=%f, auc2=%f, acc=%f, loss=%f, get better score" % (
            epoch, train_steps, eval_auc, eval_auc2, eval_acc,eval_loss))
            self.logger.info(
                "epoch = %d, train_steps=%d, auc=%f, auc=%f, acc=%f, loss=%f" % (epoch, train_steps, eval_auc, eval_auc2, eval_acc, eval_loss))

    def evaluate(self,filename=""):
        if self.eval_X_data is None:
            if filename:
                self.eval_X_data , self.eval_Y_data = self.load_data(filename)
            else:
                self.eval_X_data , self.eval_Y_data = self.load_data(self.eval_filename)
        batch_data = self.load_batch_data(data=list(zip(self.eval_X_data, self.eval_Y_data)))
        acc_s = []
        logit_s = []
        label_s = []
        loss_s = []

        for idx, current_batch_data in enumerate(batch_data):
            if current_batch_data.shape[0] != self.batch_size:
                continue
            x_feed_in, y_feed_in = zip(*current_batch_data)
            if idx < 10:
                acc, logit, loss, x, y, pred, w_a_d_logit_after_soft, w_a_d_logit = self.sess.run(
                    [self.accuracy, self.w_a_d_output, self.total_loss, self.X, self.Y, self.predictions,
                     self.w_a_d_logit_after_soft, self.w_a_d_logit], feed_dict={self.X: x_feed_in, self._Y: y_feed_in})
                print("in eval y    = ",y.reshape(-1))
                print("in eval pred = ", pred)
                print("in eval w_a_d_after_soft = ",w_a_d_logit_after_soft)
            else:
                acc,logit,loss= self.sess.run([self.accuracy,self.w_a_d_output,self.total_loss],feed_dict={self.X:x_feed_in,self._Y:y_feed_in})
            print("in eval","idx = ",idx," acc = ",acc," loss = ", loss)

            acc_s.append(acc)
            loss_s.append(loss)
            logit_s.extend(list(logit.reshape(-1)))
            label_s.extend(list(y_feed_in))
            # a = list(logit.reshape(-1))
            # b = list(y_feed_in)
            # print("in evaluate")
            # print('a = ',a)
            # print("b = ",b)
            # import pdb
            # pdb.set_trace()


        average_acc = np.mean(acc_s)
        auc = compute_auc(label_s,logit_s)
        auc2 = sklearn.metrics.roc_auc_score(label_s, logit_s)
        average_loss = np.mean(loss_s)
        return average_acc , auc , auc2 , average_loss

    def test(self,filename=""):
        if filename:
            self.test_X_data, self.test_Y_data = self.load_data(filename)
        else:
            self.test_X_data, self.test_Y_data = self.load_data(self.test_filename)
        batch_data = self.load_batch_data(data=list(zip(self.test_X_data, self.test_Y_data)))
        acc_s = []
        logit_s = []
        label_s = []

        for idx, current_batch_data in enumerate(batch_data):
            print(current_batch_data.shape," <--> ",self.batch_size)
            if current_batch_data.shape[0] != self.batch_size:
                continue
            x_feed_in, y_feed_in = zip(*current_batch_data)
            acc, logit = self.sess.run([self.accuracy, self.w_a_d_logit],
                                       feed_dict={self.X: x_feed_in, self._Y: y_feed_in})
            print("in test acc = ", acc)
            acc_s.append(acc)
            logit_s.extend(list(logit))
            label_s.extend(list(y_feed_in))

        average_acc = np.mean(acc_s)
        auc = compute_auc(label_s, logit_s)
        return average_acc, auc

    def load_batch_data_indices(self, data_size, batch_size):
        num_batchs_per_epchs = int((data_size - 1) / batch_size) + 1
        indices = np.random.permutation(np.arange(data_size))
        for batch_num in range(num_batchs_per_epchs):
            start_index = batch_num * batch_size
            # end_index = min((batch_num + 1) * batch_size, data_size)
            end_index = (batch_num + 1) * batch_size
            if end_index > data_size:
                yield indices[data_size-self.batch_size:data_size]
            else:
                yield indices[start_index:end_index]

    def train2(self, train_filename="", eval_filename=""):
        if train_filename:
            self.X_data, self.Y_data = self.load_data(train_filename)
        else:
            self.X_data, self.Y_data = self.load_data(self.train_filename)
        # Test_Example_Num = 20
        # self.X_data = self.X_data[:Test_Example_Num]
        # self.Y_data = self.Y_data[:Test_Example_Num]
        if eval_filename:
            self.eval_X_data, self.eval_Y_data = self.load_data(eval_filename)
        else:
            self.eval_X_data, self.eval_Y_data = self.load_data(self.eval_filename)
        train_steps = 0
        history_acc = 0
        history_auc = 0
        start_t = time.time()
        for epoch in range(self.train_epoch_num):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            print('Training the model for epoch {}'.format(str(epoch)))
            batch_data_indices = self.load_batch_data_indices(len(self.X_data),batch_size=self.batch_size)
            for idx, current_batch_data_indices in enumerate(batch_data_indices):
                print("current_batch_data_indices.shape = ",current_batch_data_indices.shape)
                if current_batch_data_indices.shape[0] != self.batch_size:
                    continue
                x_feed_in, y_feed_in = self.X_data[current_batch_data_indices],self.Y_data[current_batch_data_indices]
                _, current_loss, current_accuracy = self.sess.run([self.train_op, self.total_loss, self.accuracy],
                                                                  feed_dict={self.X: x_feed_in, self._Y: y_feed_in})
                print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()),"idx = ", idx, " train_steps = ", train_steps, " current loss = ", current_loss,
                      " current accuracy = ", current_accuracy)
                train_steps += 1
                if train_steps % self.eval_freq == 0:
                    self.logger.info('Time to run {} steps : {} s'.format(train_steps, time.time() - start_t))
                    start_t = time.time()
                    eval_acc, eval_auc = self.evaluate2(eval_filename) if eval_filename else self.evaluate()
                    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()),"epoch = %d, train_steps=%d, auc=%f, acc=%f" % (epoch, train_steps, eval_auc, eval_acc))
                    if eval_auc > history_auc or (eval_auc == history_auc and eval_acc > history_acc):
                        self.save_model(save_dir="/home2/data/zhengquan/WAD/", prefix="auc=%f" % (eval_auc))
                        history_auc = eval_auc
                        history_acc = eval_acc
                        print("epoch = %d, train_steps=%d, auc=%f, acc=%f, get better score" % (
                            epoch, train_steps, eval_auc, eval_acc))
                        self.logger.info(
                            "epoch = %d, train_steps=%d, auc=%f, acc=%f" % (epoch, train_steps, eval_auc, eval_acc))
                    return

    def evaluate2(self,filename=""):
        if self.eval_X_data is None:
            if filename:
                self.eval_X_data , self.eval_Y_data = self.load_data(filename)
            else:
                self.eval_X_data , self.eval_Y_data = self.load_data(self.eval_filename)
        batch_data_indices = self.load_batch_data_indices(data_size=len(self.eval_X_data),batch_size=self.batch_size)
        acc_s = []
        logit_s = []
        label_s = []

        for idx, current_batch_data_indices in enumerate(batch_data_indices):
            if current_batch_data_indices.shape[0] != self.batch_size:
                continue
            x_feed_in, y_feed_in = self.eval_X_data[current_batch_data_indices],self.eval_Y_data[current_batch_data_indices]
            acc,logit = self.sess.run([self.accuracy,self.w_a_d_logit],feed_dict={self.X:x_feed_in,self._Y:y_feed_in})
            acc_s.append(acc)
            logit_s.extend(list(logit))
            label_s.extend(list(y_feed_in))

        average_acc = np.mean(acc_s)
        auc = compute_auc(label_s,logit_s)
        return average_acc , auc

    def test2(self,filename=""):
        self.test_X_data, self.test_Y_data = self.load_data(filename) if filename else self.load_data(self.test_filename)
        batch_data_indices = self.load_batch_data_indices(data_size=len(self.test_X_data),batch_size=self.batch_size)

        acc_s = []
        logit_s = []
        label_s = []

        for idx, current_batch_data_indices in enumerate(batch_data_indices):
            if current_batch_data_indices.shape[0] != self.batch_size:
                continue
            x_feed_in, y_feed_in = self.test_X_data[current_batch_data_indices],self.test_Y_data[current_batch_data_indices]
            acc, logit = self.sess.run([self.accuracy, self.w_a_d_logit],
                                       feed_dict={self.X: x_feed_in, self._Y: y_feed_in})
            acc_s.append(acc)
            logit_s.extend(list(logit))
            label_s.extend(list(y_feed_in))

        average_acc = np.mean(acc_s)
        auc = compute_auc(label_s, logit_s)
        return average_acc, auc

    def save_model(self,save_dir,prefix):
        temp_path = os.path.join(save_dir,prefix)
        os.makedirs(temp_path,exist_ok=True)
        save_path = self.saver.save(self.sess, os.path.join(save_dir,prefix,"model.ckpt"))
        self.logger.info('Model saved in {}'.format(save_path))

    def restore_model(self,save_dir,prefix):
        self.saver.restore(self.sess,os.path.join(save_dir,prefix,'model.ckpt'))
        self.logger.info('Model restored from {}'.format(os.path.join(save_dir,prefix,'model.ckpt')))


if __name__ == "__main__":
    tag2value = json.load(open("tag2value.json", "r", encoding="utf-8"))
    # tag2valueOneline = json.load(open('tag2valueOneline.json', "r", encoding="utf-8"))
    user_features = json.load(open("AllDeep/user_side_feature.txt.json","r",encoding="utf-8"))
    video_features = json.load(open("AllDeep/video_side_feature.txt.json","r",encoding="utf-8"))
    context_features = json.load(open("AllDeep/context_feature.txt.json","r",encoding="utf-8"))
    print("user_features = ",user_features)
    print("video_features = ",video_features)
    print("context_features = ",context_features)
    A = WideAndDeep(batch_size=64,eval_freq=1000,tag2value=tag2value,custom_tags = [],
                    train_epoch_num=2,
                    train_filename="/home2/data/ttd/train_ins_add.processed.csv.pkl",
                    eval_filename="/home2/data/ttd/sub_eval_ins_add.processed.csv.pkl",
                    test_filename="/home2/data/ttd/sub_test_ins_add.processed.csv.pkl",
                    # features_to_exclude=features_to_exclude,
                    # features_to_exclude=[],
                    features_to_exclude=['409', '410', '412', '413', '414', '415', '416'],
                    features_to_keep=[],
                    feature_num=100,
                    learning_rate_base=1e-3,
                    learning_rate_decay_step=3000,
                    user_features = user_features,
                    video_features = video_features,
                    context_features = context_features,
                    wide_side_node=100,
                    deep_side_nodes=[700, 100],
                    video_side_nodes=[300, 100],
                    user_side_nodes=[700, 100],
                    context_side_nodes=[100, 100],
                    sim_loss_a = 0.2 #也可能是5
                    )

    A.train(train_filename="/home2/data/ttd/train_ins_add.processed.csv.pkl",eval_filename="/home2/data/ttd/eval_ins_add.processed.csv.pkl")
    #A.train(train_filename="/home2/data/ttd/zhengquan_test.processed.csv.pkl",eval_filename="/home2/data/ttd/zhengquan_test.processed.csv.pkl")
    print("begin test")
    # test_acc , test_auc = A.test(filename="/home2/data/ttd/eval_ins_add.processed.csv.pkl")
    #test_acc , test_auc = A.test(filename="/home2/data/ttd/zhengquan_test.processed.csv.pkl")

    # A.restore_model(save_dir="/home2/data/zhengquan/WAD/",prefix="auc=0.693")
    # test_acc , test_auc = A.test(filename="/home2/data/ttd/sub_test_ins_add.processed.csv.pkl")
    #
    # print("test_acc = ",test_acc)
    # print("test_auc = ",test_auc)
