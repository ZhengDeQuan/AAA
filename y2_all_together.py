import pandas as pd
import tensorflow as tf
import numpy as np
import json
import pickle
import logging
import time
import os


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


class WideAndDeep(object):
    def __init__(self,
                 batch_size = 2,
                 feature_num = 108,
                 train_epoch_num = 2,
                 train_steps = 1000,
                 tag2value = None,
                 tag2valueOneline = None,
                 custom_tags = [],
                 wide_side_node=100,
                 deep_side_nodes=[700,100],
                 eval_freq = 1000,#每隔1000个step，evaluate一次
                 moving_average_decay = 0.99,# 滑动平均的衰减系数
                 learning_rate_base = 0.8,# 基学习率
                 learning_rate_decay = 0.99,# 学习率的衰减率
                 total_example_num = 2000000,
                 train_filename = "",
                 eval_filename = "",
                 test_filename = "",
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

        self.eval_freq = eval_freq
        self.moving_average_decay= moving_average_decay
        self.learning_rate_base = learning_rate_base
        self.learning_rate_decay = learning_rate_decay
        self.total_example_num = total_example_num

        self.train_filename = train_filename
        self.eval_filename = eval_filename
        self.test_filename = test_filename

        start_t = time.time()
        self.sess = tf.Session()
        self._setup_placeholder()
        self._setup_mappings()
        self._realize_mappings()
        self._build_graph()
        self._build_loss()
        self._create_train_op()
        self.saver = tf.train.Saver()
        self.sess.run([tf.global_variables_initializer(),tf.tables_initializer()])


    def _setup_placeholder(self):
        self.X = tf.placeholder(shape=[self.batch_size,self.feature_num],dtype=tf.string)
        self._Y = tf.placeholder(shape=[self.batch_size,],dtype=tf.float32)
        self.Y = tf.expand_dims(self._Y,axis=-1)

    def _setup_mappings(self):
        tag2valueOneline = self.tag2valueOneline
        tag2value = sorted(self.tag2value.items(),key = lambda x: x[0])
        tag2value = dict(tag2value)
        self.tag2value = tag2value
        tags = []
        # type = dict, key是tag，value是这个tag的所有的可能的取值组成的列表
        for key in tag2value:
            tag = Tag(
                featureNum=len(tag2value[key]),
                featureNumOneline=len(tag2valueOneline),
                tag_name=key,
            )
            tag.cal_(tag2value[key])
            tags.append(tag)

        tags_to_repair = {}

        if self.custom_tags:
            for custom_tag in self.custom_tags:
                tags_to_repair[custom_tag['tag_name']] = custom_tag

        self.tags_to_repair= tags_to_repair

        for tag in tags:
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
            if tag.wide_or_deep_side == "wide":
                wide_side_dimension_size += tag.embedding_size
            else:
                deep_side_dimension_size += tag.embedding_size

        self.wide_side_dimension_size = wide_side_dimension_size
        self.deep_side_dimension_size = deep_side_dimension_size
        self.tags = tags

    def _realize_mappings(self):
        exp_X = tf.expand_dims(self.X, axis=-1)
        uns_X = tf.unstack(exp_X, axis=0)
        batch_embedding_res = []
        for one_example in uns_X:
            wide_mappings = {}
            wide_tensors = []
            deep_mappings = {}
            deep_tensors = []
            features = tf.unstack(one_example, axis=0)
            for one_feature, tag in zip(features, self.tags):
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
            mappings = {}
            tensors = []
            for key in wide_mappings:
                mappings[key] = wide_mappings[key]
            for key in deep_mappings:
                mappings[key] = deep_mappings[key]
            tensors = wide_tensors + deep_tensors
            wide_and_deep_embedding_res = tf.feature_column.input_layer(mappings, tensors)
            batch_embedding_res.append(wide_and_deep_embedding_res)

        batch_embedding_res = tf.concat(batch_embedding_res, axis=0)
        # wide_side_embedding , deep_side_embedding = tf.split(batch_embedding_res,[wide_side_dimension_size,deep_side_dimension_size],axis = 1)
        wide_inputs, deep_inputs = tf.split(batch_embedding_res, [self.wide_side_dimension_size, self.deep_side_dimension_size],1)

        self.wide_inputs = tf.reshape(wide_inputs, [self.batch_size, self.wide_side_dimension_size])
        self.deep_inputs = tf.reshape(deep_inputs, [self.batch_size, self.deep_side_dimension_size])


    def _build_graph(self):
        with tf.variable_op_scope([self.wide_inputs], None, "cb_unit", reuse=False) as scope:
            central_bias = tf.Variable(name='central_bias',
                                       initial_value=tf.random_normal(shape=[self.batch_size, 1], mean=0, stddev=1),
                                       trainable=True)

        wide_side = tf.contrib.layers.fully_connected(inputs=self.wide_inputs,
                                                      num_outputs=self.wide_side_node,
                                                      activation_fn=tf.nn.relu,
                                                      biases_initializer=None
                                                      )

        wide_side = tf.reduce_sum(wide_side, 1, name="reduce_sum")
        wide_side = tf.reshape(wide_side, [-1, 1])
        w_a_d = tf.concat([self.wide_inputs, self.deep_inputs], axis=1, name="concat")

        for k in range(len(self.deep_side_nodes)):
            w_a_d = tf.contrib.layers.fully_connected(w_a_d, self.deep_side_nodes[k], activation_fn=tf.nn.relu)
            w_a_d = tf.layers.dropout(
                inputs=w_a_d,
                rate=0.5,
                name="deep_dropout_%d" % k,
            )
        deep_side = tf.contrib.layers.fully_connected(w_a_d, 1,
                                                      activation_fn=None,
                                                      biases_initializer=None)
        deep_side = tf.reshape(deep_side, [-1, 1])
        w_a_d_logit = tf.add(deep_side, wide_side)
        self.w_a_d_logit = tf.add(w_a_d_logit, central_bias, name="wide_with_bias")
        self.w_a_d_output = tf.nn.softmax(self.w_a_d_logit, dim=-1)
        # 定义准确率
        self.predictions = tf.cast(tf.greater(self.w_a_d_output, 0), tf.int64) # 在最终预测的时候，神经网络的输出采用的是经过滑动平均的前向传播计算结果
        self.correct_prediction = tf.equal(self.predictions, tf.cast(self.Y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def _build_loss(self):
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.Y,
            logits=self.w_a_d_logit,
            name="loss_function"
        )
        loss_mean = tf.reduce_mean(loss)
        self.total_loss = loss_mean

    def _create_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)  # 定义存储当前迭代训练轮数的变量

        # 定义ExponentialMovingAverage类对象
        self.variable_averages = tf.train.ExponentialMovingAverage(
            self.moving_average_decay, self.global_step)  # 传入当前迭代轮数参数
        # 定义对所有可训练变量trainable_variables进行更新滑动平均值的操作op
        trainable_vars = tf.trainable_variables()
        self.variables_averages_op = self.variable_averages.apply(trainable_vars)


        # 定义指数衰减学习率
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base, self.global_step,
                                           2000000 / self.batch_size, self.learning_rate_decay)
        # 定义梯度下降操作op，global_step参数可实现自加1运算
        self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate) \
            .minimize(self.total_loss, global_step=self.global_step)
        # 组合两个操作op
        self.train_op = tf.group(self.train_step, self.variables_averages_op)

    def load_data(self,filename = "/home2/data/ttd/zhengquan_test.processed.csv.pkl"):
        df_data = pickle.load(open(filename, "rb"))  # 一个DataFrame
        # import pdb
        # pdb.set_trace()
        df_data = df_data.dropna(how="all", axis=0)  # 0 对行进行操作，how='any'只要有一个NA就删除这一行，how='all'必须全部是NA才能删除这一行
        # 不能用any过滤，否则过滤完了，1000个只剩3个。
        df_data['label'] = (df_data['label']).astype(int)
        df_data = df_data[df_data['label'].isin([0, 1])]  # 只保留label为0或者1的

        # 分离X,Y
        X_data = df_data.drop(['label'], axis=1)
        X_data = X_data.values.astype(str)
        Y_data = df_data['label'].values.astype(np.int32)
        return X_data,Y_data

    def load_batch_data(self,data):
        data = np.array(data)
        data_size = len(data)
        num_batchs_per_epchs = int((data_size - 1) / self.batch_size) + 1
        indices = np.random.permutation(np.arange(data_size))
        shufflfed_data = data[indices]
        for batch_num in range(num_batchs_per_epchs):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, data_size)
            yield shufflfed_data[start_index:end_index]

    def train(self,train_filename="",eval_filename=""):
        self.X_data, self.Y_data = self.load_data(train_filename) if train_filename else self.load_data()
        Test_Example_Num = 20
        self.X_data = self.X_data[:Test_Example_Num]
        self.Y_data = self.Y_data[:Test_Example_Num]
        self.eval_X_data , self.eval_Y_data = None, None
        train_steps = 0
        history_acc = 0
        history_auc = 0
        start_t = time.time()
        for epoch in range( self.train_epoch_num ):
            self.logger.info('Training the model for epoch {}'.format(epoch))
            print('Training the model for epoch {}'.format(str(epoch)))
            batch_data = self.load_batch_data(data = list(zip(self.X_data,self.Y_data)))
            for idx, current_batch_data in enumerate(batch_data):
                x_feed_in , y_feed_in = zip(*current_batch_data)
                _,current_loss,current_accuracy = self.sess.run([self.train_op,self.total_loss,self.accuracy],feed_dict={self.X:x_feed_in,self._Y:y_feed_in})
                print("idx = ",idx," train_steps = ",train_steps, " current loss = ",current_loss," current accuracy = ",current_accuracy)
                train_steps += 1
                if train_steps % self.eval_freq == 0:
                    self.logger.info('Time to run {} steps : {} s'.format(train_steps,time.time() - start_t))
                    start_t = time.time()
                    eval_acc, eval_auc = self.evaluate(eval_filename) if eval_filename else self.evaluate()
                    print("epoch = %d, train_steps=%d, auc=%.3f, acc=%.3f" % (epoch, train_steps, eval_auc, eval_acc))
                    if eval_auc > history_auc or (eval_auc == history_auc and eval_acc > history_acc) :
                        self.save_model(save_dir="/home2/data/zhengquan/WAD/",prefix="auc=%.3f"%(eval_auc))
                        history_auc = eval_auc
                        history_acc = eval_acc
                        print("epoch = %d, train_steps=%d, auc=%.3f, acc=%.3f, get better score"%(epoch,train_steps,eval_auc,eval_acc))
                        self.logger.info("epoch = %d, train_steps=%d, auc=%.3f, acc=%.3f"%(epoch,train_steps,eval_auc,eval_acc))



    def evaluate(self,filename=""):
        if self.eval_X_data is None:
            self.eval_X_data , self.eval_Y_data = self.load_data(filename) if filename else self.load_data(self.eval_filename)
        batch_data = self.load_batch_data(data=list(zip(self.eval_X_data, self.eval_Y_data)))
        acc_s = []
        logit_s = []
        label_s = []

        for idx, current_batch_data in enumerate(batch_data):
            x_feed_in, y_feed_in = zip(*current_batch_data)
            acc,logit = self.sess.run([self.accuracy,self.w_a_d_logit],feed_dict={self.X:x_feed_in,self._Y:y_feed_in})
            acc_s.append(acc)
            logit_s.extend(list(logit))
            label_s.extend(list(y_feed_in))

        average_acc = np.mean(acc_s)
        auc = compute_auc(label_s,logit_s)
        return average_acc , auc

    def test(self,filename=""):
        self.test_X_data, self.test_Y_data = self.load_data(filename) if filename else self.load_data(self.test_filename)
        batch_data = self.load_batch_data(data=list(zip(self.test_X_data, self.test_Y_data)))
        acc_s = []
        logit_s = []
        label_s = []

        for idx, current_batch_data in enumerate(batch_data):
            x_feed_in, y_feed_in = zip(*current_batch_data)
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
    tag2valueOneline = json.load(open('tag2valueOneline.json', "r", encoding="utf-8"))
    A = WideAndDeep(batch_size=15,eval_freq=4000,tag2value=tag2value,tag2valueOneline=tag2valueOneline,custom_tags = [],
                    train_epoch_num=1,
                    train_filename="/home2/data/ttd/train_ins_add.processed.csv.pkl",
                    eval_filename="/home2/data/ttd/sub_eval_ins_add.processed.csv.pkl",
                    test_filename="/home2/data/ttd/sub_test_ins_add.processed.csv.pkl")

    A.train(train_filename="/home2/data/ttd/train_ins_add.processed.csv.pkl",eval_filename="/home2/data/ttd/sub_eval_ins_add.processed.csv.pkl")
    print("begin test")
    test_acc , test_auc = A.test(filename="/home2/data/ttd/sub_test_ins_add.processed.csv.pkl")
    print("test_acc = ",test_acc)
    print("test_auc = ",test_auc)
