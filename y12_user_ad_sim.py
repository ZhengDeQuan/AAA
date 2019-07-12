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
from custom_tag_config import custom_tags

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
        self.has_sibling = False
        self.sibling = None

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
                 tag2valueOneline = None,
                 custom_tags = [],
                 wide_side_node=100,
                 deep_side_nodes=[700, 100],
                 video_side_nodes=[300, 100],
                 user_side_nodes=[700, 100],
                 context_side_nodes=[100, 100],
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
                 user_features=[],
                 video_features=[],
                 context_features=[],
                 sim_loss_a=0.2,
                 args=None
                 ):

        assert tag2value is not None, "tag2value should be a dict, but found None"
        assert tag2valueOneline is not None, "tag2valueOneline should be a dict, but found None"
        self.batch_size = batch_size
        self.feature_num = feature_num
        self.train_epoch_num = train_epoch_num
        self.train_steps = train_steps
        self.args = args
        self.logger = logging.getLogger("brc")
        self.tag2valueOneline = tag2valueOneline
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
        self.video_features = user_features#video_features,两者弄成相同的，再看看
        self.context_features = context_features


        if len(self.features_to_keep) > 0:
            self.tag2value = OrderedDict()
            self.tag2valueOneline = OrderedDict()
            for key in self.features_to_keep:
                if key in tag2value:
                    self.tag2value[key] = tag2value[key]
                if key in tag2valueOneline:
                    self.tag2valueOneline[key] = tag2valueOneline[key]

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
        tag2valueOneline = self.tag2valueOneline
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
                featureNumOneline=tag2valueOneline[key],
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
        tag2valueOneline = self.tag2valueOneline
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
                featureNumOneline=tag2valueOneline[key],
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
        tag2valueOneline = self.tag2valueOneline
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

        tags_to_repair = OrderedDict()
        if self.custom_tags:
            for custom_tag in self.custom_tags:
                tags_to_repair[custom_tag['tag_name']] = custom_tag
        self.tags_to_repair = tags_to_repair
        Keys = []
        for one_feature, key in zip(features, self.tag2value):
            print("in setup and realize mapping2")
            print("one_features = ",one_feature)
            print("key  = ",key)
            Keys.append(key)
            tag = Tag(
                featureNum=len(tag2value[key]),
                featureNumOneline=tag2valueOneline[key],
                tag_name=key
            )
            tag.cal_(tag2value[key])
            if key in tags_to_repair:
                tag.has_sibling = True
                new_tag = Tag(
                    featureNum=len(tag2value[key]),
                    featureNumOneline=tag2valueOneline[key],
                    tag_name="custom_"+key
                )
                new_tag.cal_(tag2value[key])
                tag.sibling = new_tag

            split_tag = tf.string_split(one_feature, "|")
            one_sparse = tf.SparseTensor(
                indices=split_tag.indices,
                values=split_tag.values,
                ## 这里给出了不同值通过表查到的index ##
                dense_shape=split_tag.dense_shape
            )
            if tag.sibling is not None:
                sibling = tag.sibling
                print("sibling.tag_name = ",sibling.tag_name)
                sibling.kind = "custom"
                sibling.wide_or_deep_side = "deep"
                sibling.tag_set = tags_to_repair[tag.tag_name]['vocab_fun'](sibling.tag_set)
                #注意，一定要先调用上面这句话，再调用下面两句话。
                sibling.embedding_size = tags_to_repair[tag.tag_name]['embedding_size']
                sibling.vocab_size = tags_to_repair[tag.tag_name]['vocab_size']
                print("sibling.vocab_size = ",sibling.vocab_size)
                print("sibling.embedding_size = " ,sibling.embedding_size)
                # import pdb
                # pdb.set_trace()
                table = tf.contrib.lookup.index_table_from_tensor(mapping=sibling.tag_set,
                                                                  default_value=-1)  ## 这里构造了个查找表 ##
                sibling.table = table
                one_feature = tf.contrib.layers.sparse_column_with_keys(
                    column_name=sibling.tag_name,
                    keys=sibling.tag_set,
                    default_value=0,
                    combiner='sum',
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.embedding_column(one_feature,
                                                         initializer=tags_to_repair[tag.tag_name]['initializer_function'],
                                                         combiner="mean",
                                                         dimension=sibling.embedding_size)
                deep_mappings[sibling.tag_name]= one_sparse
                deep_tensor_s.append(res)
                deep_side_dimension_size += sibling.embedding_size

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
        tag2valueOneline = self.tag2valueOneline
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

        tags_to_repair = OrderedDict()
        if self.custom_tags:
            for custom_tag in self.custom_tags:
                tags_to_repair[custom_tag['tag_name']] = custom_tag
        self.tags_to_repair = tags_to_repair
        Keys = []
        for one_feature, key in zip(features, self.tag2value):
            print("in setup and realize mapping2")
            print("one_features = ",one_feature)
            print("key  = ",key)
            Keys.append(key)
            tag = Tag(
                featureNum=len(tag2value[key]),
                featureNumOneline=tag2valueOneline[key],
                tag_name=key
            )
            tag.cal_(tag2value[key])
            if key in tags_to_repair:
                tag.has_sibling = True
                new_tag = Tag(
                    featureNum=len(tag2value[key]),
                    featureNumOneline=tag2valueOneline[key],
                    tag_name="custom_"+key
                )
                new_tag.cal_(tag2value[key])
                tag.sibling = new_tag

            split_tag = tf.string_split(one_feature, "|")
            one_sparse = tf.SparseTensor(
                indices=split_tag.indices,
                values=split_tag.values,
                ## 这里给出了不同值通过表查到的index ##
                dense_shape=split_tag.dense_shape
            )
            if tag.sibling is not None:
                sibling = tag.sibling
                print("sibling.tag_name = ",sibling.tag_name)
                sibling.kind = "custom"
                sibling.wide_or_deep_side = "deep"
                sibling.tag_set = tags_to_repair[tag.tag_name]['vocab_fun'](sibling.tag_set)
                #注意，一定要先调用上面这句话，再调用下面两句话。
                sibling.embedding_size = tags_to_repair[tag.tag_name]['embedding_size']
                sibling.vocab_size = tags_to_repair[tag.tag_name]['vocab_size']
                print("sibling.vocab_size = ",sibling.vocab_size)
                print("sibling.embedding_size = " ,sibling.embedding_size)
                # import pdb
                # pdb.set_trace()
                table = tf.contrib.lookup.index_table_from_tensor(mapping=sibling.tag_set,
                                                                  default_value=-1)  ## 这里构造了个查找表 ##
                sibling.table = table
                one_feature = tf.contrib.layers.sparse_column_with_keys(
                    column_name=sibling.tag_name,
                    keys=sibling.tag_set,
                    default_value=0,
                    combiner='sum',
                    # dtype=tf.dtypes.int64
                    dtype=tf.dtypes.string
                )
                res = tf.contrib.layers.embedding_column(one_feature,
                                                         initializer=tags_to_repair[tag.tag_name]['initializer_function'],
                                                         combiner="mean",
                                                         dimension=sibling.embedding_size)
                if tag.tag_name in self.user_features:
                    deep_user_mappings[sibling.tag_name] = one_sparse
                    deep_user_tensor_s.append(res)
                    deep_user_dimension_size += sibling.embedding_size
                elif tag.tag_name in self.video_features:
                    deep_video_mappings[sibling.tag_name] = one_sparse
                    deep_video_tensor_s.append(res)
                    deep_video_dimension_size += sibling.embedding_size
                else:
                    deep_context_mappings[sibling.tag_name] = one_sparse
                    deep_context_tensor_s.append(res)
                    deep_context_dimension_size += sibling.embedding_size

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
                    wide_user_mappings[key] = one_sparse
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
        wide_user_inputs, wide_video_inputs, \
        wide_context_inputs, deep_user_inputs, \
        deep_video_inputs, deep_context_inputs = tf.split(wide_and_deep_embedding_res,
                                                          [wide_user_dimension_size,
                                                           wide_video_dimension_size,
                                                           wide_context_dimension_size,
                                                           deep_user_dimension_size,
                                                           deep_video_dimension_size,
                                                           deep_context_dimension_size], 1)

        self.wide_inputs = tf.concat([wide_user_inputs, wide_video_inputs, wide_context_inputs], axis=1)
        self.deep_inputs = tf.concat([deep_user_inputs, deep_video_inputs, deep_context_inputs], axis=1)

        self.user_inputs = tf.concat([wide_user_inputs, deep_user_inputs], axis=1)
        self.video_inputs = tf.concat([wide_video_inputs, deep_video_inputs], axis=1)
        self.context_inputs = tf.concat([wide_context_inputs, deep_context_inputs], axis=1)

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
        self.sim_user_video = tf.reduce_sum( tf.multiply( user_side, video_side ), 1 )
        # user_side = tf.contrib.layers.fully_connected(user_side,self.video_side_nodes[-1],
        #                                   activation_fn=None,
        #                                   weights_regularizer=tf.contrib.layers.l2_regularizer(0.1),
        #                                   biases_initializer=None)
        # self.sim_user_video = tf.reduce_sum( tf.multiply( user_side, video_side ), 1 )

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
                #     ['1139106401876931908', '14251240999902296838',
                #      '8156326296122895700', '486563035044894608',
                #      '14328034882667202648', '15036919652803740552',
                #      '14042244077753449587', '10700211283520125228',
                #      '2809543020566612793', '7721097958472601184',
                #      '9256938326208594408', '11383592208638576218',
                #      '7862227347642427297', '11051057910055446882',
                #      '18138841324584158049', '4429773002561127151',
                #      '2858154398152268805', '2188711209970881255',
                #      '6909071201165497899', '12686116844730352342',
                #      '13312733396855342027', '10352974914583171274',
                #      '17216655373308165839|1297183746042897414|4548331351168289402|18374571413799270557|14471994071354243037|2628513590576265569',
                #      '4344486144888318060|5494946837539691739|6342150510384385421|11150867894224099096|16296755669046745214|6958415081942107550|7250140884553799495|9624069708253481449|11035592085314692929|5353165378964083855|9243380248444904824',
                #      '14996020257046463320|16915856613620591395|6681652090518917235|5932236865381570399|87369269970126777|4320148549793965832|1226507125928639614',
                #      '9692253798400122607|3269236643836507188|10670183168040155576|9740487927786944040|3137344945685435943|11591911419660457963|15174408778494083309|348557118936851049|13419116425387889711|14467631856090779217',
                #      '4362507278181596706|11091734370612424491|5853276300864683932|109339551198684474|11588815455707146693',
                #      '3557950714524746283|15440109019532501603|3311212693222607286|8400853433050784389|8306662223646513813|15728194835647734048|4135935307034927186|2720066006721433337|1169354622115836051',
                #      'nan', 'nan', 'nan', 'nan', '7811488656042211780',
                #      '11038530244903891459', '13212198940446403373',
                #      '15047543949812858760|14443112839117660577|16198667752460643840|4165238059196522898|17550887418304407737|13462334839024648284|3041152845486977139|13326508072962455582|10694959028498566240|15050619538451750123|9421160004204997353|10088585657133901474|17646444669360651840|12546824958872727236|13495513921883499630|14664531813610957120|3182202928379617788|4339007962356069816|17432831273777678417|16359280748754583334|8418422241532532799|15577706011217260588|10237405972504119925|10258241860967007666|2605570455625427519|931962667567729033|13481562437305211555|15258159479087173165|8180096752974054224|2048639640247734892|7065927411967818776|11888232570725137238|11432779477424855605|3372355852573054399|3528400773630569794|11945717837007359844|8042173956157990559|9565710858076165779|15936479723333094724|17457456642275017191|11364859856069838464|13499995712886517183|6633128201401631452|8410481969185991053|12385293219784555154|17819005560355588061|10801301477520673848|5532463411752736969|3441682880758088069|15443691085669775426',
                #      '15251515538588611611', '13288147893574155447',
                #      '8043229614276226839', '16068620925422622478',
                #      '17126816401314613785', '347728814626000368',
                #      '13805380568563865991', '8913016071035174941',
                #      '9042829607402409649', '10975603474365085086',
                #      '123471851312176903', '9517692237640246759', '1685038070105069615',
                #      '9585917213752811097', '1308926824548290208',
                #      '12765756940636153004', '1981827375497862370',
                #      '6887795066668969302',
                #      '4193024172164896280|3565771841518901855|14728828426041736960|17541959070606957121',
                #      '1145300960957708165', '10962750466625816877',
                #      '15561419044730699602', '13390657329729922589',
                #      '9702273921971298476', '2589231521512941579',
                #      '5650394086445605816', '7332134400676384080', 'nan', 'nan',
                #      '2458524913987183451',
                #      '9142171976169053903|2213310578074290203|4335267573905255309|12221533502322966836|16966973451539237912|5289287689681396479|12233383276019439290|1927584253162437741',
                #      '18403652446572426290|12570019107734999026|3072275408300298705|8469469022845262589|4567125991642785281|14545041589237172228|2961293259635871552|14095728536345456547',
                #      '597966533097459223|17875236985249840684|16757741048428772076|8762812589572883681|16254392423357082807|10071298949412902309',
                #      '15380449625927026102', 'nan', 'nan',
                #      '3985704400231958185|3985704400231958185|3985704400231958185|5651831115566953833|5651831115566953833|5651831115566953833|5651831115566953833|7828363789063543645|7828363789063543645|7226992376149310776|2566823960511015888|2566823960511015888|2566823960511015888|18097246579576234887|10467381234105993990|10467381234105993990|10467381234105993990|10467381234105993990|11499711293444345957|11499711293444345957',
                #      '17608587270731759045|17608587270731759045|17608587270731759045|1338201828520893568|1338201828520893568|1338201828520893568|10189428290434562479|4984826707099565251|16016356843999198851|5791817768996543691|5791817768996543691|5791817768996543691|5791817768996543691|18245228737520028799|18245228737520028799|18245228737520028799|18245228737520028799|18412743071571349924|18412743071571349924|18412743071571349924',
                #      '14904519619766778163', '12938188901629621628', 'nan', 'nan',
                #      'nan', '9309173260138316748', '8836938996338491968',
                #      '7246114893777235673', '17655959831516163419',
                #      '17090763658850109906', '4803820018709242041',
                #      '16084229495722910183', '12030757886824062096', 'nan',
                #      '14394543285647607735', '921722677166023324',
                #      '1155521055480828894|8692709813717798963|11459458593813865569|6044088236897874160|5491416649503917070|732163117012296753|2434121216357764640|3767358438443136584|7585912220764241424',
                #      '9755251724477987685|16434002679314019006|11098968456805832867|5770000118721821576|16570501233585543600|3001833081046513755|8360972913229769016|4668989311192011282|14772885225184628148|7965810342199571405|9880507707363938407|16499292093859845778|3152724320110640149|16382632601598373100|2377335301970622958|7652977301664242774|16622669313691901064|2574819821342110081|8265182534168341089|4084452208479238743|2387936357413411612|8736942389225076397|12009740467245532352|17388733625009960448|5173399495584165321',
                #      '16951803530242380182|17941293129512970563|1326159236745910520|5769056233029588850|14146443913213867785|6655930445201114565|818012633082401334|4792991097825755713|8421673125173483714',
                #      '13107304437483108506|17181776496285634278|1318604128623651|2569939827165670348|18083594344199745737|525435215572503587|11493935306995063468|7249894690305644052|913567934092578769',
                #      '4346979625878017500|6077351400311854567|11132442596171762155|3447415594747804625|10570140180988203675|5230364446960483001|8358044141816068029|1447313335555913320|5097309868884379582',
                #      '4842113398229228003|9345862366545755171|2617224355694777436|17365238347377517002|7101900494435369087|6006603110714361660|10077758512226258296|14451670513417062260|14664345666778575700|17248715472013855562|5385403276207701724|8322295001735994443|10721353100001564596|2888962317953521611|15543711293074385896|8375217179395275570|10899160094602503628|7234734285455928509|5568218404396450391|15505336575875408846|7089367010653144589|11360444461716236060|18299178965994838928|2156934850774973928|5749175949324121340',
                #      '11757666502456051546|9149657679390002925|4592170877033250528|2257036327688029695|3480994461651563806|15010211064986942064|14222174774115817607|17115649888256105279|13951521014964679050|12870599631926651101|7529348736515623986|17308024664084261102|10703995197755589448|7589686572624636316|7070758025269968763|9898081615337314087|11222071910915254343|2522014843055164673|12991181800632670421|9297076939528030057|429930400130084470|4693550156743605668|5380969020437329297|17360491332682457886|1066612263780366277',
                #      '2821410439800442491|4642336336922499412|2955001922274103359|2500075381205921329|11739804506643642483|8878722699117003181|10539523902790599544|2578273668044985684|2699784835516107799|14024468209703344055|11501149734712272543|9043855167531818141|17968112127010888995|2815229402766422540|7833132809742205579|8491039486693163103|12863242304618067225|1818668659088413967|1238578441355063885|18040837849191632861|3088410293957760137|1580663772207170676|12643204615096686042|5116168528962710236|2000870364729210344',
                #      '8357868485411169975', '15634731065619785278'],
                #     ['1139106401876931908', '745785247849726351', '9191856338692964838',
                #      '11699587169719753', '1311825981923988584', '7417301490858926583',
                #      '6744131256303148084', '10700211283520125228',
                #      '2809543020566612793', '13890394283574237965',
                #      '6635637182228579443', '2046238089089866838',
                #      '15034855917252918619', '5456741272669859853',
                #      '18057988843231594966', '5505945717242741188',
                #      '6557996544808580186', '14483380428435730133',
                #      '17728006300274365726', '7835155520823890705',
                #      '13312733396855342027', '11636397103134160771',
                #      '9242694767697709935|3906376295377258030|14061049681392393889|1387560312023783843',
                #      '7415203336816613969|9995917746477430839|2929705610976253326|13740189917040727881|9123751176989003043|2005796101783901849|11508397918588814009|5960102028162660684|8685572140556768337|16136223881403780355|17520866376544656886',
                #      'nan', 'nan', '4362507278181596706', '17571761779145756190', 'nan',
                #      'nan', 'nan', 'nan', '7811488656042211780', '4383023995824779264',
                #      '9074382149259128550',
                #      '5897671272294757971|2030203024395195051|8435948636215220608|11647246626606107969|88634036165558633|1527684610781034110|8616878340636057517|10434894882613228485|4486814023620835576|11874668018522210019|2832832899265435691|2731059736527019650|4224935177299394294|13422568039189543634|14458468676138123101|16956374178565621645|2491232975750080947|17004838669601269942|13329487484115188693|1893060949481232491|1473260601924423145|534537175903752892|4844544053792261536|4941089024526919571|12562177101916198021|17929950542064457092|17162441094359050099|17906513098977395600|2443110422019743789|3664656309553612800|5558876297260992666|2889178471084420968|4375186504336214454|11185655100749798726|6510597666801702019|1309289497184547415|16470304639434357500|846824784310137869|13152487290758123387|15392056353395532403|15434005248357528637|8899168797708242823|9151315041138244156|15716072795595059612|16045197346985349148|15860619575119729607|15905124675248868432|12442263984838403891|13577030304886712550|11775497055433918640',
                #      '4441576132110064728', '13955290530090352702',
                #      '14949745424548641532', '8976526829182982322',
                #      '13391261191399334826', '8342868706374202435',
                #      '5887456808186513399', '10567036631171767755',
                #      '16372780765213129667', '11562254063813715857',
                #      '377011176521260508', '16524077006879216085',
                #      '6410021542029097144', '174605596838664715',
                #      '10124401142404527828', '16032272561659291917',
                #      '7352622153332486501', '10559030038422583946', 'nan',
                #      '14419356597971583867', '10962750466625816877',
                #      '10803705403237549128', '3866811833490286798',
                #      '9702273921971298476', '7473350022634778048',
                #      '16925053901752961011', '7332134400676384080',
                #      '1977326967590208941|534481388260228805|9553332143704200127|14004100310311481872',
                #      '4586700795774690066|16216182162463708354', '8011140828338466336',
                #      '2213310578074290203|1050312251627752660|16966973451539237912|1015334570620604612|1737835945871842211|17621638373960890281',
                #      '12570019107734999026|8919379937287126630|1715544949675085138|9493480580858507568|5344123558707009630|7396266163579789398|17978650666151417261',
                #      '8303638342979969211|2854930708482023851|3447978581584682075|4624237757085457423',
                #      '12692862381310633550',
                #      '14313140575369553501|10881878276249440260|14247214455509068200|342500318128156201',
                #      '13601229673527954069|4287966133456876746|4287966133456876746',
                #      '3985704400231958185|3985704400231958185|5651831115566953833|5651831115566953833|5651831115566953833|5651831115566953833|4415816537655689599|2566823960511015888|2566823960511015888|2566823960511015888|2566823960511015888|16113238019957411347|11614319480510162860|18097246579576234887|9335824699591894168|11499711293444345957',
                #      '17608587270731759045|17608587270731759045|17608587270731759045|1338201828520893568|1338201828520893568|1338201828520893568|1338201828520893568|16016356843999198851|16016356843999198851|16016356843999198851|5791817768996543691|8351210473168019268|8351210473168019268|11637796944459384723|9304999839509156427|18412743071571349924|13863133986803256022',
                #      '962683575123434719', '12938188901629621628',
                #      '11582229580283823839|6871071353921082289|16793306238176136147|16827029993268730829',
                #      '2048498117375500834|7930873359356555285|15065994169582831440|4783376903954343255',
                #      '947685604720322932|15792565059371109648|7862009950486942783|11742220377861584904',
                #      '9309173260138316748', '14179110932626403756',
                #      '7246114893777235673', '17655959831516163419',
                #      '17090763658850109906', '4803820018709242041',
                #      '16084229495722910183', '5183682098020051672', 'nan',
                #      '14394543285647607735', '921722677166023324', 'nan', 'nan', 'nan',
                #      'nan', 'nan', 'nan', 'nan', 'nan', '2137428766905285608',
                #      '18225238955838146297']
                # ]
                # y_feed_in = (1, 0)
                # _, current_loss, current_accuracy, x, y, pred, w_a_d_logit_softmax, w_a_d_logit, deep_inputs = self.sess.run(
                #     [self.train_op, self.total_loss, self.accuracy, self.X, self.Y, self.predictions,
                #      self.w_a_d_logit_after_soft, self.w_a_d_logit, self.deep_inputs],
                #     feed_dict={self.X: x_feed_in, self._Y: y_feed_in})
                # print("x_feed_in = ",x_feed_in)
                # print("y_feed_in = ",y_feed_in)
                # print("x = ",x)
                # print("y = ", y)
                # print("pred= ", pred)
                # print("w_a_d_logit_softmax = ", w_a_d_logit_softmax)
                # print("w_a_d_logit = ",w_a_d_logit)
                # print("deep_inputs[0] = ",deep_inputs[0])
                # pdb.set_trace()
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
    tag2value = json.load(open("small_tag2value.json", "r", encoding="utf-8"))
    tag2valueOneline = json.load(open('small_tag2valueOneline.json', "r", encoding="utf-8"))
    user_features = json.load(open("AllDeep/user_side_feature.txt.json", "r", encoding="utf-8"))
    video_features = json.load(open("AllDeep/video_side_feature.txt.json", "r", encoding="utf-8"))
    context_features = json.load(open("AllDeep/context_feature.txt.json", "r", encoding="utf-8"))
    print("user_features = ", user_features)
    print("video_features = ", video_features)
    print("context_features = ", context_features)
    A = WideAndDeep(batch_size=64, eval_freq=1000, tag2value=tag2value, tag2valueOneline=tag2valueOneline,
                    train_epoch_num=2,
                    train_filename="/home2/data/ttd/train_ins_add.processed.csv.pkl",
                    eval_filename="/home2/data/ttd/sub_eval_ins_add.processed.csv.pkl",
                    test_filename="/home2/data/ttd/sub_test_ins_add.processed.csv.pkl",
                    # features_to_exclude=features_to_exclude,
                    features_to_exclude=[],
                    features_to_keep=[],
                    feature_num=100,
                    learning_rate_base=1e-3,
                    learning_rate_decay_step=3000,
                    custom_tags = custom_tags,
                    user_features=user_features,
                    video_features=video_features,
                    context_features=context_features,
                    wide_side_node=100,
                    deep_side_nodes=[700, 100],
                    video_side_nodes=[300, 100],
                    user_side_nodes=[700, 100],
                    context_side_nodes=[100, 100],
                    sim_loss_a=0.2  # 也可能是5
                    )

    A.train(train_filename="/home2/data/ttd/train_ins.processed.csv.pkl",eval_filename="/home2/data/ttd/eval_ins.processed.csv.pkl")
    #A.train(train_filename="/home2/data/ttd/zhengquan_test.processed.csv.pkl",eval_filename="/home2/data/ttd/zhengquan_test.processed.csv.pkl")
    print("begin test")
    # test_acc , test_auc = A.test(filename="/home2/data/ttd/eval_ins_add.processed.csv.pkl")
    #test_acc , test_auc = A.test(filename="/home2/data/ttd/zhengquan_test.processed.csv.pkl")

    # A.restore_model(save_dir="/home2/data/zhengquan/WAD/",prefix="auc=0.693")
    # test_acc , test_auc = A.test(filename="/home2/data/ttd/sub_test_ins_add.processed.csv.pkl")
    #
    # print("test_acc = ",test_acc)
    # print("test_auc = ",test_auc)
