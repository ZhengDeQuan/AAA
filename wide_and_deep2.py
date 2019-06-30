from __future__ import division, print_function

import os
import sys
import argparse
import zqtflearn
import tempfile
import urllib

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn import metrics
import pickle

# 数据格式，每行110列，第一列是label，之后的108列是连续特征值。先用wide模型跑，第110列是video_id用deep模型跑
# 2000000 train_ins_continues.txt 200w
# 300000 eval_ins_continues.txt 30w

# -----------------------------------------------------------------------------

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

LABEL_COLUMN = 1

CATEGORICAL_COLUMNS = {"video_id": [125079, 768]}

CATEGORICAL_COLUMNS_OLD = {"workclass": 10, "education": 17, "marital_status": 8,
                           "occupation": 16, "relationship": 7, "race": 6,
                           "gender": 3, "native_country": 43, "age_binned": 14}

CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]


# -----------------------------------------------------------------------------

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


class TFLearnWideAndDeep(object):
    '''
    Wide and deep model, implemented using TFLearn
    '''
    AVAILABLE_MODELS = ["wide", "deep", "wide+deep"]

    def __init__(self, model_type="wide+deep", verbose=None, name=None, tensorboard_verbose=3,
                 wide_learning_rate=0.001, deep_learning_rate=0.001,bias_learning_rate=0.001,
                 checkpoints_dir=None,tensorboard_dir=None,max_checkpoints=15,
                 vocab_path="video_ids.pkl",
                 embedding_path="video_ids_embedding.pkl"):
        '''
        model_type = `str`: wide or deep or wide+deep
        verbose = `bool`
        name = `str` used for run_id (defaults to model_type)
        tensorboard_verbose = `int`: logging level for tensorboard (0, 1, 2, or 3)
        wide_learning_rate = `float`: defaults to 0.001
        deep_learning_rate = `float`: defaults to 0.001
        checkpoints_dir = `str`: where checkpoint files will be stored (defaults to "CHECKPOINTS")
        '''
        self.model_type = model_type or "wide+deep"
        assert self.model_type in self.AVAILABLE_MODELS
        self.verbose = verbose or 0
        self.tensorboard_verbose = tensorboard_verbose
        self.name = name or self.model_type  # name is used for the run_id
        self.data_columns = COLUMNS
        self.continuous_columns = CONTINUOUS_COLUMNS
        self.categorical_columns = CATEGORICAL_COLUMNS  # dict with category_name: category_size
        self.label_column = 0
        self.checkpoints_dir = checkpoints_dir or "CHECKPOINTS"
        self.tensorboard_dir = tensorboard_dir or "tensorboard_dir"
        self.max_checkpoints = max_checkpoints
        if not os.path.exists(self.checkpoints_dir):
            os.mkdir(self.checkpoints_dir)
            print("Created checkpoints directory %s" % self.checkpoints_dir)
        self.build_model([wide_learning_rate, deep_learning_rate,bias_learning_rate])

    def read_embedding(self, embedding_path,vocab):
        # df = pd.read_csv(embedding_path, header=None, delim_whitespace=True)  # [video_id vector] 空格分隔
        pretrained_embedding = pickle.load(open(embedding_path,"rb"))
        dimension = 0
        for key in pretrained_embedding:
            temp = pretrained_embedding[key]
            dimension=len(temp["vector"])
            break
        emb = np.random.randn(len(vocab), dimension).astype(np.float32)
        for key in pretrained_embedding:
            if key not in vocab:
                continue
            emb[vocab[key]] = pretrained_embedding[key]["vector"]
        return emb

    def build_model(self, learning_rate=[0.001, 0.01, 0.001]):
        '''
        Model - wide and deep - built using zqtflearn
        '''
        n_cc = len(self.continuous_columns)
        n_cc = 108

        input_shape = [None, n_cc]
        if self.verbose:
            print("=" * 77 + " Model %s (type=%s)" % (self.name, self.model_type))
            print("  Input placeholder shape=%s" % str(input_shape))
        wide_inputs = zqtflearn.input_data(shape=input_shape, name="wide_X")
        deep_inputs = zqtflearn.input_data(shape=[None, 1], name="deep_X")
        if not isinstance(learning_rate, list):
            learning_rate = [learning_rate, learning_rate,learning_rate]  # wide, deep
        if self.verbose:
            print("  Learning rates (wide, deep)=%s" % learning_rate)

        with tf.name_scope("Y"):  # placeholder for target variable (i.e. trainY input)
            Y_in = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="Y")

        with tf.variable_op_scope([wide_inputs], None, "cb_unit", reuse=False) as scope:
            central_bias = zqtflearn.variables.variable('central_bias', shape=[1],
                                                        initializer=tf.constant_initializer(np.random.randn()),
                                                        trainable=True, restore=True)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/cb_unit', central_bias)

        if 'wide' in self.model_type:
            wide_network = self.wide_model(wide_inputs, n_cc)
            network = wide_network
            wide_network_with_bias = tf.add(wide_network, central_bias, name="wide_with_bias")

        if 'deep' in self.model_type:
            deep_network = self.deep_model(wide_inputs, deep_inputs,
                                           n_cc)  # 这里面是wide inputs,在这个函数内部wide_inputs,会和deep_model制造的输入相合并。
            deep_network_with_bias = tf.add(deep_network, central_bias, name="deep_with_bias")
            if 'wide' in self.model_type:
                network = tf.add(wide_network, deep_network)
                if self.verbose:
                    print("Wide + deep model network %s" % network)
            else:
                network = deep_network

        network = tf.add(network, central_bias, name="add_central_bias")

        # add validation monitor summaries giving confusion matrix entries
        with tf.name_scope('Monitors'):
            predictions = tf.cast(tf.greater(network, 0), tf.int64)
            print("predictions=%s" % predictions)
            Ybool = tf.cast(Y_in, tf.bool)
            print("Ybool=%s" % Ybool)
            pos = tf.boolean_mask(predictions, Ybool)
            neg = tf.boolean_mask(predictions, ~Ybool)
            psize = tf.cast(tf.shape(pos)[0], tf.int64)
            nsize = tf.cast(tf.shape(neg)[0], tf.int64)
            true_positive = tf.reduce_sum(pos, name="true_positive")
            false_negative = tf.subtract(psize, true_positive, name="false_negative")
            false_positive = tf.reduce_sum(neg, name="false_positive")
            true_negative = tf.subtract(nsize, false_positive, name="true_negative")
            overall_accuracy = tf.truediv(tf.add(true_positive, true_negative), tf.add(nsize, psize),
                                          name="overall_accuracy")
        vmset = [true_positive, true_negative, false_positive, false_negative, overall_accuracy]

        trainable_vars = tf.trainable_variables()
        tv_deep = [v for v in trainable_vars if v.name.startswith('deep_')]
        tv_wide = [v for v in trainable_vars if v.name.startswith('wide_')]

        if self.verbose:
            print("DEEP trainable_vars")
            for v in tv_deep:
                print("  Variable %s: %s" % (v.name, v))
            print("WIDE trainable_vars")
            for v in tv_wide:
                print("  Variable %s: %s" % (v.name, v))

        # if 'wide' in self.model_type:
        #     if not 'deep' in self.model_type:
        #         tv_wide.append(central_bias)
        #     zqtflearn.regression(wide_network_with_bias,
        #                          placeholder=Y_in,
        #                          optimizer='sgd',
        #                          loss='roc_auc_score',
        #                          #loss='binary_crossentropy',
        #                          metric="accuracy",
        #                          learning_rate=learning_rate[0],
        #                          validation_monitors=vmset,
        #                          trainable_vars=tv_wide,
        #                          op_name="wide_regression",
        #                          name="Y")
        #
        # if 'deep' in self.model_type:
        #     if not 'wide' in self.model_type:
        #         tv_wide.append(central_bias)
        #     zqtflearn.regression(deep_network_with_bias,
        #                          placeholder=Y_in,
        #                          optimizer='adam',
        #                          loss='roc_auc_score',
        #                          #loss='binary_crossentropy',
        #                          metric="accuracy",
        #                          learning_rate=learning_rate[1],
        #                          validation_monitors=vmset if not 'wide' in self.model_type else None,
        #                          trainable_vars=tv_deep,
        #                          op_name="deep_regression",
        #                          name="Y")

        if self.model_type == 'wide+deep':  # learn central bias separately for wide+deep
            zqtflearn.regression(network,
                                 placeholder=Y_in,
                                 optimizer='adam',
                                 loss="roc_auc_score",
                                 #loss='binary_crossentropy',
                                 metric="accuracy",
                                 validation_monitors=vmset,
                                 learning_rate=learning_rate[0],  # use wide learning rate
                                 #trainable_vars=[central_bias], #[tv_deep,tv_wide,central_bias] # None
                                 op_name="central_bias_regression",
                                 name="Y")

        self.model = zqtflearn.DNN(network,
                                   tensorboard_verbose=self.tensorboard_verbose,
                                   max_checkpoints=self.max_checkpoints,
                                   checkpoint_path="%s/%s.tfl" % (self.checkpoints_dir, self.name),
                                   tensorboard_dir=self.tensorboard_dir
                                   )
        # tensorboard_dir="/tmp/tflearn_logs/" zqtflearn.DNN 我把他改为当前目录下的了，这样也比较好规范
        if 'deep' in self.model_type:
            embeddingWeights = zqtflearn.get_layer_variables_by_name('deep_video_ids_embed')[0]
            # CUSTOM_WEIGHT = pickle.load("Haven't deal")
            # emb = np.array(CUSTOM_WEIGHT, dtype=np.float32)
            # emb = self.embedding
            new_emb_t = tf.convert_to_tensor(self.embedding)
            self.model.set_weights(embeddingWeights, new_emb_t)

        if self.verbose:
            print("Target variables:")
            for v in tf.get_collection(tf.GraphKeys.TARGETS):
                print("  variable %s: %s" % (v.name, v))

            print("=" * 77)

        print("model build finish")

    def deep_model(self, wide_inputs, deep_inputs, n_inputs, n_nodes=[100, 50], use_dropout=False):
        '''
        Model - deep, i.e. two-layer fully connected network model
        '''
        vocab = pickle.load(open("video_ids.pkl","rb"))
        vocab = list(vocab)

        self.vocab = {}
        for idx, wd in enumerate(vocab):
            self.vocab[wd] = idx
        self.vocab["[UNK]"] = len(self.vocab)
        if os.path.exists('self_embedding.pkl'):
            self.embedding = pickle.load(open("self_embedding.pkl","rb"))
            print("making new self.embedding!!!")
        else:
            self.embedding = self.read_embedding(embedding_path='zq_videoID_vectors.pkl', vocab = self.vocab)
            pickle.dump(self.embedding,open("self_embedding.pkl","wb"))

        net = zqtflearn.layers.embedding_ops.embedding(deep_inputs, len(self.vocab), self.embedding.shape[1],
                                                       trainable=False
                                                       ,name="deep_video_ids_embed")
        net = tf.squeeze(net, squeeze_dims=[1], name="video_ids_squeeze")
        net = zqtflearn.fully_connected(net, 108 , activation="relu",name="deep_fc_108" )
        net = zqtflearn.fully_connected(net, 54, activation="relu", name="deep_fc_54")
        network = tf.concat([wide_inputs, net], axis=1,
                            name="deep_concat")  # x=xigma(dim of each element in flat_vars) + wide_inputs.size(1) [?,x]

        print("n_nodes = ",n_nodes)
        for k in range(len(n_nodes)):
            network = zqtflearn.fully_connected(network, n_nodes[k], activation="relu",
                                                name="deep_fc%d" % (k + 1))  # 默认应该是用bais的。要不然下面为什么要写bias=False
            if use_dropout:
                network = zqtflearn.dropout(network, 0.5, name="deep_dropout%d" % (k + 1))
        if self.verbose:
            print("Deep model network before output %s" % network)
        network = zqtflearn.fully_connected(network, 1, activation="linear", name="deep_fc_output", bias=False)  # [?,1]

        network = tf.reshape(network, [-1,1])  # so that accuracy is binary_accuracy added by zhengquan ,不reshape不也是[?,1]的吗?可能如果最后的输出维度是1的话，结果是[?]的尺寸
        if self.verbose:
            print("Deep model network %s" % network)
        return network

    def wide_model(self, inputs, n_inputs):
        '''
        Model - wide, i.e. normal linear model (for logistic regression)
        '''
        network = inputs
        # use fully_connected (instead of single_unit) because fc works properly with batches, whereas single_unit is 1D only
        network = zqtflearn.fully_connected(network, n_inputs, activation="linear", name="wide_linear",
                                            bias=False)  # x*W (no bias) #added by zhengquan[?,8] , 写的是inputs，但是他的实际含义是输出的维度。
        network = tf.reduce_sum(network, 1, name="reduce_sum")  # batched sum, to produce logits [?]
        network = tf.reshape(network, [-1, 1])  # so that accuracy is binary_accuracy [?,1]
        if self.verbose:
            print("Wide model network %s" % network)
        return network

    # /home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/
    # def load_data(self,
    #               train_dfn="/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/train_ins_continues.txtlineNUM_videoID",
    #               eval_dfn="/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/eval_ins_continues.txtlineNUM_videoID"):
    def load_data(self,
                  train_dfn="a_train_shuf.txt",
                  eval_dfn="a_eval_shuf.txt"):
        self.train_data = pd.read_csv(train_dfn, delim_whitespace=True, header=None)
        self.eval_data = pd.read_csv(eval_dfn, delim_whitespace=True, header=None)

        # self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)  # shuffle [以100%de比例随机选择原来的数据，drop=True自动新建一列记录原来的index]
        # self.eval_data = self.eval_data.sample(frac=1).reset_index(drop=True)

        self.train_data[self.label_column] = (self.train_data[self.label_column]).astype(int)
        self.eval_data[self.label_column] = (self.eval_data[self.label_column]).astype(int)

        self.train_data = self.train_data[self.train_data[self.label_column].isin([0, 1])]  # 防止label列，即第0列，有除了0，1之外的值出现。
        self.eval_data = self.eval_data[self.eval_data[self.label_column].isin([0, 1])]

        if "deep" in self.model_type:
            col_num = self.train_data.shape[1] - 1
            # print(self.vocab)
            # print(col_num)
            # print(self.train_data[col_num])
            # temp = self.train_data[col_num]
            # len = temp.shape[0]
            # a = [temp[i] for i in list(range(len))]
            # for ele in a:
            #     if ele in self.vocab:
            #         print("self.vocab[ele] = ",self.vocab[ele])
            self.train_data[col_num] = (self.train_data[col_num].apply( lambda x: self.vocab[str(x)] if str(x) in self.vocab else self.vocab["[UNK]"])).astype(int)
            # temp = (self.train_data[col_num].apply( lambda x: self.vocab[str(x)] if str(x) in self.vocab else self.vocab["[UNK]"])).astype(int)

            self.eval_data[col_num] = (self.eval_data[col_num].apply( lambda x: self.vocab[str(x)] if str(x) in self.vocab else self.vocab["[UNK]"])).astype(int)
            print("train_unk = ",sum(self.train_data[col_num].isin([self.vocab["[UNK]"]])))
            print("eval_unk = ",sum(self.eval_data[col_num].isin([self.vocab["[UNK]"]])))


    # def load_test_data(self,
    #                    test_dfn="/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/test_ins_continues.txtlineNUM_videoID"):
    def load_test_data(self,
                       test_dfn="a_test_shuf.txt"):
        self.test_data = pd.read_csv(test_dfn, delim_whitespace=True, header=None)

        # self.test_data = self.test_data.sample(frac=1).reset_index(drop=True)

        self.test_data[self.label_column] = (self.test_data[self.label_column]).astype(int)

        self.test_data = self.test_data[self.test_data[self.label_column].isin([0, 1])]

        if "deep" in self.model_type:
            col_num = self.test_data.shape[1] - 1
            self.test_data[col_num] = (
                self.test_data[col_num].apply(
                    lambda x: self.vocab[str(x)] if str(x) in self.vocab else self.vocab["[UNK]"])).astype(int)
        print("test_unk = ",sum(self.test_data[col_num].isin([self.vocab["[UNK]"]])))

        self.testX_dict, self.testY_dict = self.prepare_input_data(self.test_data)

    def prepare_input_data(self, input_data):
        if 'deep' in self.model_type:
            X = input_data.iloc[:, 1: input_data.shape[1] - 1].values.astype(np.float32)
        else:
            X = input_data.iloc[:, 1: input_data.shape[1] - 1].values.astype(np.float32)
        Y = input_data[self.label_column].values.astype(np.float32)
        # print("X.shape = ",X.shape)
        # print("Y.shape = ",Y.shape)
        # X.shape = (50000, 108)
        # Y.shape = (50000,)
        Y = Y.reshape([-1, 1])
        if self.verbose:
            print("  Y shape=%s, X shape=%s" % (Y.shape, X.shape))
        X_dict = {"wide_X": X}
        Y_dict = {"Y": Y}
        if 'deep' in self.model_type:
            deep_X = input_data[input_data.shape[1]-1].values.astype(np.int32)
            deep_X = deep_X.reshape([-1,1])
            X_dict["deep_X"] = deep_X
        print("deep X shape = ",deep_X.shape)
        return X_dict, Y_dict

    def prepare(self):
        self.X_dict, self.Y_dict = self.prepare_input_data(self.train_data)
        self.evalX_dict, self.evalY_dict = self.prepare_input_data(self.eval_data)

    def train(self, n_epoch=1000, snapshot_step=10, batch_size=None):
        validation_batch_size = batch_size or self.evalY_dict['Y'].shape[0]
        batch_size = batch_size or self.Y_dict['Y'].shape[0]

        print("Input data shape = %s; output data shape=%s, batch_size=%s" % (str(self.X_dict['wide_X'].shape),
                                                                              str(self.Y_dict['Y'].shape),
                                                                              batch_size))
        print("Eval data shape = %s; output data shape=%s, validation_batch_size=%s" % (
        str(self.evalX_dict['wide_X'].shape),
        str(self.evalY_dict['Y'].shape),
        validation_batch_size))
        print("=" * 60 + "  Training")
        self.model.fit(self.X_dict,
                       self.Y_dict,
                       n_epoch=n_epoch,
                       validation_set=(self.evalX_dict, self.evalY_dict),
                       snapshot_step=snapshot_step,
                       batch_size=batch_size,
                       validation_batch_size=validation_batch_size,
                       show_metric=True,
                       snapshot_epoch=True,
                       shuffle=True,
                       run_id=self.name,
                       )

    def evaluate(self):
        logits = np.array(self.model.predict(self.evalX_dict)).reshape([-1])
        print ("="*60 + "  Evaluation")
        print ("  logits: %s, min=%s, max=%s" % (logits.shape, logits.min(), logits.max()))
        probs =  1.0 / (1.0 + np.exp(-logits))
        y_pred = pd.Series((probs > 0.5).astype(np.int32))
        Y = pd.Series(self.evalY_dict['Y'].astype(np.int32).reshape([-1]))
        self.confusion_matrix = self.output_confusion_matrix(Y, y_pred)
        print ("="*60)
        return probs, Y , y_pred ,logits
        return None, Y, None, logits

    def test(self):
        logits = np.array(self.model.predict(self.testX_dict)).reshape([-1])
        print("=" * 60 + "  Evaluation")
        print("  logits: %s, min=%s, max=%s" % (logits.shape, logits.min(), logits.max()))
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pred = pd.Series((probs > 0.5).astype(np.int32))
        Y = pd.Series(self.testY_dict['Y'].astype(np.int32).reshape([-1]))
        self.confusion_matrix = self.output_confusion_matrix(Y, y_pred)
        print("=" * 60)
        return probs, Y, y_pred ,logits
        return None, Y, None, logits

    def output_confusion_matrix(self, y, y_pred):
        assert y.size == y_pred.size
        print("Actual IDV")
        print(y.value_counts())
        print("Predicted IDV")
        print(y_pred.value_counts())
        print()
        print("Confusion matrix:")
        cmat = pd.crosstab(y_pred, y, rownames=['predictions'], colnames=['actual'])
        print(cmat)
        sys.stdout.flush()
        return cmat


# -----------------------------------------------------------------------------

def CommandLine(args=None):
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    try:
        flags.DEFINE_string("model_type", "wide+deep", "Valid model types: {'wide', 'deep', 'wide+deep'}.")
        flags.DEFINE_string("run_name", None, "name for this run (defaults to model type)")
        flags.DEFINE_string("load_weights", None, "filename with initial weights to load")
        flags.DEFINE_string("checkpoints_dir", "checkpoints_dir", "name of directory where checkpoints should be saved")
        flags.DEFINE_string("tensorboard_dir", "tensorboard_dir", "name of directory where checkpoints should be saved")
        flags.DEFINE_integer("max_checkpoints", 15, "Number of training epoch steps")
        flags.DEFINE_integer("n_epoch", 200, "Number of training epoch steps")

        flags.DEFINE_integer("snapshot_step", 100, "Step number when snapshot (and validation testing) is done")
        flags.DEFINE_float("wide_learning_rate", 0.001, "learning rate for the wide part of the model")
        flags.DEFINE_float("deep_learning_rate", 0.001, "learning rate for the deep part of the model")
        flags.DEFINE_float("bias_learning_rate", 0.001, "learning rate for the deep part of the model")
        flags.DEFINE_boolean("verbose", False, "Verbose output")
        flags.DEFINE_boolean("do_train", False, "do_train")
        flags.DEFINE_boolean("do_test", False, "do_test")
    except argparse.ArgumentError:
        pass  # so that CommandLine can be run more than once, for testing

    twad = TFLearnWideAndDeep(model_type=FLAGS.model_type, verbose=FLAGS.verbose,
                              name=FLAGS.run_name, wide_learning_rate=FLAGS.wide_learning_rate,
                              deep_learning_rate=FLAGS.deep_learning_rate,
                              bias_learning_rate=FLAGS.bias_learning_rate,
                              checkpoints_dir=FLAGS.checkpoints_dir,
                              tensorboard_dir=FLAGS.tensorboard_dir,
                              max_checkpoints=FLAGS.max_checkpoints)
    twad.load_data()
    twad.prepare()
    if FLAGS.load_weights:
        print("Loading initial weights from %s" % FLAGS.load_weights)
        twad.model.load(FLAGS.load_weights)
        print("Load Weight Succeed")
    if FLAGS.do_train:
        twad.train(n_epoch=FLAGS.n_epoch, snapshot_step=FLAGS.snapshot_step, batch_size=50)
        twad.model.save(FLAGS.checkpoints_dir + "aaaa")

    y_pred_prob, gtruth, y_pred, logits = twad.evaluate()
    acc = [1 if ele1 == ele2 else 0 for ele1, ele2 in zip(gtruth, y_pred)]
    print("accuracy = ", sum(acc) / len(acc))
    y_pred = pd.Series((logits > 0).astype(np.int32))
    acc = [1 if ele1 == ele2 else 0 for ele1, ele2 in zip(gtruth, y_pred)]
    print("accuracy2 = ", sum(acc) / len(acc))
    fpr, tpr, thresholds = metrics.roc_curve(y_true=gtruth, y_score=list(y_pred_prob),pos_label=1) #因为我发现居然有2，train中3个，test中两个
    auc1 = metrics.auc(fpr,tpr)
    print("eval_auc = ",auc1)
    auc2 = sklearn.metrics.roc_auc_score(gtruth, list(y_pred_prob))
    print("eval_auc = ",auc2)
    auc3 = compute_auc(gtruth, list(logits))
    print("eval_auc = ", auc3)

    if FLAGS.do_test:
        twad.load_test_data()
        y_pred_prob, gtruth, y_pred, logits = twad.test()
        acc = [1 if ele1 == ele2 else 0 for ele1,ele2 in zip(gtruth,y_pred)]
        print("accuracy = ",sum(acc)/len(acc))

        y_pred = pd.Series((logits > 0).astype(np.int32))
        acc = [1 if ele1 == ele2 else 0 for ele1,ele2 in zip(gtruth,y_pred)]
        print("accuracy2 = ",sum(acc)/len(acc))

        fpr, tpr, thresholds = metrics.roc_curve(y_true=gtruth, y_score=y_pred_prob,pos_label=1)  # 因为我发现居然有2，train中3个，test中两个
        auc1 = metrics.auc(fpr, tpr)
        print("eval_auc = ", auc1)
        auc2 = sklearn.metrics.roc_auc_score(gtruth, list(y_pred_prob))
        print("eval_auc = ", auc2)
        auc = compute_auc(gtruth, list(logits))
        print("test_auc = ", auc) #test_auc =  0.5985576923076923（小数据集，bert）
    return twad


# -----------------------------------------------------------------------------
# unit tests

def test_wide_and_deep():
    import glob
    tf.reset_default_graph()
    cdir = "test_checkpoints"
    if os.path.exists(cdir):
        os.system("rm -rf %s" % cdir)
    twad = CommandLine(args=dict(verbose=True, n_epoch=5, model_type="wide+deep", snapshot_step=5,
                                 wide_learning_rate=0.0001, checkpoints_dir=cdir))
    cfiles = glob.glob("%s/*.tfl-*" % cdir)
    print("cfiles=%s" % cfiles)
    assert (len(cfiles))
    cm = twad.confusion_matrix.values.astype(np.float32)
    assert (cm[1][1])


def test_deep():
    import glob
    tf.reset_default_graph()
    cdir = "test_checkpoints"
    if os.path.exists(cdir):
        os.system("rm -rf %s" % cdir)
    twad = CommandLine(args=dict(verbose=True, n_epoch=5, model_type="deep", snapshot_step=5,
                                 wide_learning_rate=0.0001, checkpoints_dir=cdir))
    cfiles = glob.glob("%s/*.tfl-*" % cdir)
    print("cfiles=%s" % cfiles)
    assert (len(cfiles))
    cm = twad.confusion_matrix.values.astype(np.float32)
    assert (cm[1][1])


def test_wide():
    import glob
    tf.reset_default_graph()
    cdir = "test_checkpoints"
    if os.path.exists(cdir):
        os.system("rm -rf %s" % cdir)
    twad = CommandLine(args=dict(verbose=True, n_epoch=5, model_type="wide", snapshot_step=5,
                                 wide_learning_rate=0.0001, checkpoints_dir=cdir))
    cfiles = glob.glob("%s/*.tfl-*" % cdir)
    print("cfiles=%s" % cfiles)
    assert (len(cfiles))
    cm = twad.confusion_matrix.values.astype(np.float32)
    assert (cm[1][1])


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    CommandLine()
    None
