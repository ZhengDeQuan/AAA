#encoding=utf-8

from smnn.algorithm import ClassifyAlgorithmBase, AlgorithmBase
from smnn.feature_transform import FeatureColumnGroupConfigParser
from smnn.exception import FeatureColumnGroupConfigError, InvalidTrainOpError, InputException
from smnn.component.dnn import build_dnn_logits
from tensorflow.contrib.layers.python.layers import feature_column as feature_column_ops_lib
from tensorflow.python.saved_model import signature_constants
import tensorflow as tf
import math
import six # Six is a Python 2 and 3 compatibility library.

'''
算法使用的feature可选择的group名:
(1) 如果feature要配置在linear侧，group名配
    置为LINEAR_FEATURE_GROUP_NAMES之一
(2) 如果feature要配置在dnn侧，group名配置
    为DNN_FEATURE_GROUP_NAMES之一
(3) 如果feature要配置在fm侧，group名配置
    为FM_FEATURE_GROUP_NAMES
'''
LINEAR_FEATURE_GROUP_NAMES = ['wide', 'linear']
DNN_FEATURE_GROUP_NAMES = ['deep', 'dnn']
FM_FEATURE_GROUP_NAMES = ['fm']

'''
算法支持的模式：
(1) classify为分类模式
(2) regression为回归模式
'''
CLASSIFY = 'classify'
REGRESSION = 'regression'

class WideAndDeepAlgorithm(object):
    '''
    算法的dnn,fm,linear三侧使用的name scope名
    '''
    DNN_SCOPE_NAME='dnn'
    FM_SCOPE_NAME='fm'
    LINEAR_SCOPE_NAME='linear'

    '''
    算法的linear,dnn,fm三侧使用的优化器配置
    关键字
    '''
    LINEAR_OPTIMIZER_CONFIG_KEY='linear'
    DNN_OPTIMIZER_CONFIG_KEY='dnn'
    FM_OPTIMIZER_CONFIG_KEY='fm'

    '''
    算法的dnn,linear,fm三侧优化器使用的默认
    learning rate
    '''
    _DNN_LEARNING_RATE = 0.05
    _LINEAR_LEARNING_RATE = 0.2
    _FM_LEARNING_RATE = 0.05

    def __init__(self):
        self.linear_default_optimizer = tf.train.FtrlOptimizer(learning_rate=0.1)
        self.dnn_default_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.fm_default_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.num_class = 2
        self.linear_logits = None
        self.dnn_logits = None
        self.fm_logits = None
        self.dnn_optimzer = None
        self.linear_optimizer = None
        self.fm_optimizer = None
        self.gradient_clip_norm = None
        self.fix_global_step_increment_bug = True
        self.DNN_SCOPE_NAME = 'dnn'

    @staticmethod
    def _prepare_feature_columns(model_desc_dict):
        '''
        Function:
            为算法准备feature column的接口，返回linear,dnn,fm三
            侧的feature columns
        Args:
            model_desc_dict(dict):
                从配置model_desc的json文件中解析出的dict 
        Returns:
            linear_feature_columns(feature column set):
                配置在linear侧的feature column集合
            dnn_feature_columns(feature column set):
                配置在dnn侧的feature column集合
            fm_feature_columns(feature column set):
                配置在fm侧的feature column集合
        '''
        linear_feature_columns, dnn_feature_columns, fm_feature_columns = ([],[],[])

        parser = FeatureColumnGroupConfigParser(model_desc_dict)
        all_feature_column_group = parser.get_all_feature_column_group()

        for key, feature_column_list in all_feature_column_group.items():
            if key in LINEAR_FEATURE_GROUP_NAMES:
                linear_feature_columns = feature_column_list
                continue
            if key in DNN_FEATURE_GROUP_NAMES:
                dnn_feature_columns = feature_column_list
                continue
            if key in FM_FEATURE_GROUP_NAMES:
                fm_feature_columns = feature_column_list
                continue
        return linear_feature_columns, dnn_feature_columns, fm_feature_columns

    @staticmethod
    def _build_linear_logits(features, linear_feature_columns, output_rank,
                             num_ps_replicas, input_layer_min_slice_size, joint_linear_weights=False):
        '''
        Function:
            构建linear侧逻辑的接口，返回linear侧的logits
        Args:
            features(tensor_dic):
                包含所有特征的tensor_dict
            linear_feature_columns(feature column set):
                构成linear侧的feature column集合
            output_rank(int):
                期望的输出矩阵的秩
            num_ps_replicas(int):
                PS replicas的数量
            input_layer_min_slice_size(int):
                输入层的最小分片大小
            joint_linear_weights(bool):
                是否使用联合线性权重，默认为False。如果为True，则使用
                tf.contrib.layers.joint_weighted_sum_from_feature_colum
                生成linear侧的logits，否则，使用
                tf.contrib.layers.weighted_sum_from_feature_columns生成
                linear侧的logits。
        Returns:
            linear_logits(Tensor):
                构成算法逻辑linear侧的logits
        '''
        if not linear_feature_columns or len(linear_feature_columns) == 0:
            return None
        else:
            WideAndDeepAlgorithm._LINEAR_LEARNING_RATE = \
                WideAndDeepAlgorithm._linear_learning_rate(WideAndDeepAlgorithm._LINEAR_LEARNING_RATE, len(linear_feature_columns))
            linear_parent_scope = WideAndDeepAlgorithm.LINEAR_SCOPE_NAME
            linear_partitioner = (
                tf.min_max_variable_partitioner(
                    max_partitions=num_ps_replicas,
                    min_slice_size = input_layer_min_slice_size))
            with tf.variable_scope(
                linear_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=linear_partitioner) as scope:
                if joint_linear_weights:
                    linear_logits, _, _ = tf.contrib.layers.joint_weighted_sum_from_feature_columns(
                        columns_to_tensors=features,
                        feature_columns=linear_feature_columns,
                        num_outputs=output_rank,
                        weight_collections=[linear_parent_scope],
                        scope=scope)
                else:
                    linear_logits, _, _ = tf.contrib.layers.weighted_sum_from_feature_columns(
                        columns_to_tensors=features,
                        feature_columns=linear_feature_columns,
                        num_outputs=output_rank,
                        weight_collections=[linear_parent_scope],
                        scope=scope)
                return linear_logits

    @staticmethod
    def _build_fm_logits(features, fm_feature_columns, output_rank, num_ps_replicas, input_layer_min_slice_size):
        '''
        Function:
            构建fm侧逻辑的接口，返回fm侧的logits
        Args:
            features(tensor_dic):
                包含所有特征的tensor_dict
            fm_feature_columns(feature column set):
                构成fm侧的feature column集合
            output_rank(int):
                期望的输出矩阵的秩
            num_ps_replicas(int):
                PS replicas的数量
            input_layer_min_slice_size(int):
                输入层的最小分片大小
        Returns:
            fm_logits(Tensor):
                构成算法逻辑fm侧的logits
        '''
        if not fm_feature_columns or len(fm_feature_columns) <= 1:
            return None
        else:
            fm_parent_scope = WideAndDeepAlgorithm.FM_SCOPE_NAME
            fm_partitioner = (
                tf.min_max_variable_partitioner(
                    max_partitions=num_ps_replicas))
            with tf.variable_scope(
                fm_parent_scope,
                values=tuple(six.itervalues(features)),
                partitioner=fm_partitioner) as fm_input_scope:

                fm_feature_tensor_list = []
                transformed_tensor_dict = tf.contrib.layers.transform_features(features, fm_feature_columns)

                for column in fm_feature_columns:
                    transformed_tensor = transformed_tensor_dict[column]
                    output_tensor = None
                    with tf.variable_scope(column.name):
                        try:
                            arguments = column._deep_embedding_lookup_arguments(transformed_tensor)
                            output_tensor = feature_column_ops_lib._embeddings_from_arguments(
                                column,
                                arguments,
                                [fm_parent_scope],
                                True,
                                output_rank=output_rank+1)

                        except NotImplementedError as ee:
                            try:
                                output_tensor  = column._to_dnn_input_layer(
                                    transformed_tensor,
                                    [fm_parent_scope],
                                    True,
                                    output_rank=output_rank+1)
                            except ValueError as e:
                                raise ValueError('Error creating input layer for column: {}.\n'
                                               '{}, {}'.format(column.name, e, ee))
                    fm_feature_tensor_list.append(output_tensor)

                fm_logits = 0.0
                num_embs = len(fm_feature_columns)
                for a in range(num_embs):
                    for b in range(a + 1, num_embs):
                        fm_logits += tf.reduce_sum(tf.multiply(fm_feature_tensor_list[a], fm_feature_tensor_list[b]), output_rank, keep_dims=True)
                return fm_logits

    @staticmethod
    def _linear_learning_rate(default_learning_rate, num_linear_feature_columns):
        '''
        Function:
            获得linear侧优化器的learning rate的接口
        Args:
            default_learning_rate(float):
                默认的linear侧优化器的learning rate
            num_linear_feature_columns(int):
                配置在linear侧的feature column数量
        Returns:
            算法逻辑linear侧优化器的learning rate(learning rate)
        '''
        learning_rate = 1. / math.sqrt(num_linear_feature_columns)
        return min(learning_rate, default_learning_rate)

    def get_algorithm_default_optimizer(self):
        '''
        Function:
            获得算法的默认优化器的接口，返回一个dict，包含linear侧、
            dnn侧、fm侧的默认优化器对象
        Args:
            无
        Returns:
            一个优化器对象的dict
        '''
        return {self.LINEAR_OPTIMIZER_CONFIG_KEY: self.linear_default_optimizer,
                self.DNN_OPTIMIZER_CONFIG_KEY: self.dnn_default_optimizer,
                self.FM_OPTIMIZER_CONFIG_KEY: self.fm_default_optimizer}

    @staticmethod
    def _embedding_column_check(embedding_list):
        '''
        Function:
            检查fm的embedding column的接口，fm使用的所有embedding 
            column必须有相同dimension
        Args:
            embedding_list(list):
                fm使用的embedding column的list
        Returns:
            (1) 如果fm使用的所有embedding column具有相同维度，无返
                回值
            (2) 否则，抛出FeatureColumnGroupConfigError的异常
        '''
        dimension = None
        for embedding_feature in embedding_list:
            if not dimension:
                dimension = embedding_feature.dimension
            elif dimension != embedding_feature.dimension:
                raise FeatureColumnGroupConfigError("All embedding columns in FM part must have the same dimension.")

    @staticmethod
    def _get_fm_variables():
        '''
        Function:
            获取fm侧使用的variable的接口，包含fm logits本身使用的variable
            以及和dnn侧共享使用的variable
        Args:
            无
        Returns:
            fm侧使用的variable集合
        '''
        fm_variables = tf.get_collection(WideAndDeepAlgorithm.FM_SCOPE_NAME)
        sharing_variables = filter(lambda x: 'shared_embedding' in x.name, tf.get_collection(WideAndDeepAlgorithm.DNN_SCOPE_NAME))
        return fm_variables + sharing_variables

    def remove_none_trainable_variable(self, var_list):
        trainable_variables = tf.trainable_variables()
        ret_list = list()
        for var in var_list:
            if var in trainable_variables:
                ret_list.append(var)
        return ret_list

    def get_train_op(self, optimizer, global_step, model=CLASSIFY):
        '''
        Function:
            通过重载此接口实现算法self._loss_op以及train_op的逻辑。
            接口中先根据之前生成的linear,dnn,fm三侧的logits生成三侧
            对应的train_op，三侧的train_op均完成一步训练后，再让
            global_step加1，这整个过程构成算法使用的train_op。
        Args:
            optimizer(dict):
                算法使用的优化器的dict
            global_step(Tensor):
                算法运行时的global step
        Returns:
            (1) 如果算法优化器配置正确，返回包含了train_op,
                self._loss_op,self.accuracy,global_step的dict
            (2) 否则，则抛出InvalidTrainOpError异常
        '''
        train_ops = []
        if not (isinstance(optimizer, dict)):
            raise InvalidTrainOpError('Optimzer is not a dict.')

        linear_optimizer = optimizer.get(self.LINEAR_OPTIMIZER_CONFIG_KEY)
        dnn_optimzer = optimizer.get(self.DNN_OPTIMIZER_CONFIG_KEY)
        fm_optimizer = optimizer.get(self.FM_OPTIMIZER_CONFIG_KEY)

        if self.dnn_logits is not None:
          dnn_variables = tf.get_collection(self.DNN_SCOPE_NAME)
          dnn_variables = self.remove_none_trainable_variable(dnn_variables)
          train_ops.append(
              tf.contrib.layers.optimize_loss(
                  loss=self._loss_op,
                  global_step=global_step,
                  learning_rate=self._DNN_LEARNING_RATE,
                  optimizer=dnn_optimzer,
                  gradient_multipliers=None,
                  clip_gradients=self.gradient_clip_norm,
                  variables=dnn_variables,
                  summaries=[],
                  increment_global_step=not self.fix_global_step_increment_bug))
        if self.linear_logits is not None:
          linear_variables = tf.get_collection(self.LINEAR_SCOPE_NAME)
          linear_variables = self.remove_none_trainable_variable(linear_variables)
          train_ops.append(
              tf.contrib.layers.optimize_loss(
                  loss=self._loss_op,
                  global_step=global_step,
                  learning_rate=self._LINEAR_LEARNING_RATE,
                  optimizer=linear_optimizer,
                  clip_gradients=self.gradient_clip_norm,
                  variables=linear_variables,
                  name=self.LINEAR_SCOPE_NAME,
                  summaries=[],
                  increment_global_step=not self.fix_global_step_increment_bug))
        if self.fm_logits is not None:
          fm_variables = self._get_fm_variables()
          fm_variables = self.remove_none_trainable_variable(fm_variables)
          train_ops.append(
              tf.contrib.layers.optimize_loss(
                  loss=self._loss_op,
                  global_step=global_step,
                  learning_rate=self._FM_LEARNING_RATE,
                  optimizer=fm_optimizer,
                  clip_gradients=self.gradient_clip_norm,
                  variables=fm_variables,
                  name=self.FM_SCOPE_NAME,
                  # Empty summaries, because head already logs "loss" summary.
                  summaries=[],
                  increment_global_step=not self.fix_global_step_increment_bug))

        with tf.name_scope('train_op'):
            train_op = tf.group(*train_ops)
            if self.fix_global_step_increment_bug:
              with tf.control_dependencies([train_op]):
                with tf.colocate_with(global_step):
                  train_op = tf.assign_add(global_step, 1).op
        if model == CLASSIFY:
            return {'_':train_op, 'train_result':{'loss': self._loss_op, 'accuracy': self._accuracy_tensor}, 'global_step':global_step}
        else:
            return {'_':train_op, 'train_result':{'loss': self._loss_op}, 'global_step':global_step}

    def _build_algorithm_graph(self, tensor_dict, algorithm_args, context, model=CLASSIFY):
        '''
        Function:
            构建分类算法的Tensorflow Graph接口，此接口中完成算法逻辑
            相关的参数校验，以及实现算法的self._inference_op的逻辑
        Args:
            tensor_dict(smnn.TensorDict):
                通过smnn.io解析出用于模型训练、评估的TensorDict
            algorithm_args(smnn.param):
                用户配置的算法逻辑的参数经过自动解析后存放在此对象中
            context(smnn.context):
                用户算法运行时的context，算法运行的配置参数，以及调度
                框架自动生成的机器角色信息都会存放在context中
        Returns:
            (1) 如果算法逻辑参数校验正确，inference op逻辑实现成功，无返回值
            (2) 如果算法逻辑参数校验出错，或者inference op逻辑实现出错，则抛
                出异常
        '''
        if model == CLASSIFY:
            self._label_tensor = tf.cast(tensor_dict.declare_and_get_tensor('label', tf.int64, [-1, 1], True, 'Label字段'), tf.int64)
            self.num_class = algorithm_args.declare_and_get_arg('num_class', int, 2, False, '分类个数')
            if self.num_class == 2:
                self._label_dimension = 1
            else:
                self._label_dimension = self.num_class
        else:
            self._label_tensor = tf.cast(tensor_dict.declare_and_get_tensor('label', tf.float32, [-1, 1], True, 'Label字段'), tf.float32)
            self._label_dimension = 1

        try:
            self._weight_tensor = tensor_dict.declare_and_get_tensor('weight', tf.float32, [-1, 1], False, '权重字段')
        except InputException:
            self._weight_tensor = None

        model_desc_dict = \
                algorithm_args.declare_and_get_json_file('feature_transform_config', '', '特征变换的配置文件地址')
        dnn_hidden_units = \
                algorithm_args.declare_and_get_list('dnn_hidden_units',\
                int, '512,256,128', \
                'DNN 各隐藏层的单元个数')
        dnn_activation_fn = \
                algorithm_args.declare_and_get_activation_fn('dnn_activation_fn',\
                'relu', 'DNN 激活函数')
        dnn_dropout = \
                algorithm_args.declare_and_get_arg('dnn_dropout', float, 0, False, 'DNN 训练时 Dropout 比例，评估预测时不生效')
        self.gradient_clip_norm = None
       #        algorithm_args.declare_and_get_arg('gradient_clip_norm', float, None, False, '梯度裁剪的阈值')
       #embedding_lr_multipliers = algorithm_args.declare_and_get_arg('embedding_lr_multipliers', str)
        min_slize_size  = 64 << 20
        input_layer_min_slice_size = \
                algorithm_args.declare_and_get_arg('input_layer_min_slice_size', int, min_slize_size, False,'Variable 分片的最小字节数')

        self._LINEAR_LEARNING_RATE = \
                algorithm_args.declare_and_get_arg('linear_learning_rate', float, 0.2, False, 'linear侧优化器学习率')
        self._DNN_LEARNING_RATE = \
                algorithm_args.declare_and_get_arg('dnn_learning_rate', float, 0.05, False, 'dnn侧优化器学习率')
        self._FM_LEARNING_RATE = \
                algorithm_args.declare_and_get_arg('fm_learning_rate', float, 0.05, False, 'fm侧优化器学习率')

        #更新用户设置的optimizer参数
        if self._LINEAR_LEARNING_RATE is not None:
            self.linear_default_optimizer = tf.train.FtrlOptimizer(learning_rate=self._LINEAR_LEARNING_RATE)
        if self._DNN_LEARNING_RATE is not None:
            self.dnn_default_optimizer = tf.train.AdagradOptimizer(learning_rate=self._DNN_LEARNING_RATE)
        if self._FM_LEARNING_RATE is not None:
            self.fm_default_optimizer = tf.train.AdagradOptimizer(learning_rate=self._FM_LEARNING_RATE)

        num_ps_replicas = context.cluster_info.num_ps_replicas

        linear_feature_columns, dnn_feature_columns, fm_feature_columns = self._prepare_feature_columns(model_desc_dict)

        self._embedding_column_check(fm_feature_columns)

        features = tensor_dict.tensor_dict

        if dnn_feature_columns:
            self.dnn_logits = build_dnn_logits(features, dnn_feature_columns, self._label_dimension,
                                           dnn_hidden_units, dnn_activation_fn, self.DNN_SCOPE_NAME,
                                           dnn_dropout, num_ps_replicas, input_layer_min_slice_size,
                                           mode=context.mode)

        self.linear_logits = WideAndDeepAlgorithm._build_linear_logits(features, linear_feature_columns, self._label_dimension, num_ps_replicas, input_layer_min_slice_size)

        self.fm_logits = WideAndDeepAlgorithm._build_fm_logits(features, fm_feature_columns, self._label_dimension, num_ps_replicas, input_layer_min_slice_size)

        if self.linear_logits is None and self.dnn_logits is None and self.fm_logits is None:
            smnn.logger.error('All logits are None.')
            return False

        total_logits = 0.0
        if self.linear_logits is not None: total_logits += self.linear_logits
        if self.dnn_logits is not None: total_logits += self.dnn_logits
        if self.fm_logits is not None: total_logits += self.fm_logits
        self._inference_op = total_logits

class ClassifyWideAndDeepAlgorithm(ClassifyAlgorithmBase):
    def __init__(self):
        super(ClassifyWideAndDeepAlgorithm, self).__init__('ClassifyWideAndDeep')
        self.model = WideAndDeepAlgorithm()

    def _build_classify_algorithm_graph(self, tensor_dict, algorithm_args, context):
        ret = self.model._build_algorithm_graph(tensor_dict, algorithm_args, context, CLASSIFY)
        self._inference_op = self.model._inference_op
        self._label_tensor = self.model._label_tensor
        self.num_class = self.model.num_class
        self._weight_tensor = self.model._weight_tensor
        return ret

    def build_prediction_signature(self, tensor_dict):
        input_tensor = tensor_dict.raw_input_tensor
        inputs = {"inputs" : input_tensor}
        logistic_tensor = tf.concat([1-self._logistic_tensor, self._logistic_tensor],1)
        outputs = {"prob" : logistic_tensor}
        prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs, outputs)
        self._signature_map = {
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY : prediction_signature
        }

    def get_train_op(self, optimizer, global_step):
        self.model._accuracy_tensor = self._accuracy_tensor
        self.model._loss_op = self._loss_op
        return self.model.get_train_op(optimizer, global_step, CLASSIFY)

    def get_algorithm_default_optimizer(self):
        return self.model.get_algorithm_default_optimizer()

class RegressionWideAndDeepAlgorithm(AlgorithmBase):
    def __init__(self):
        super(RegressionWideAndDeepAlgorithm, self).__init__('RegressionWideAndDeep')
        self.model = WideAndDeepAlgorithm()

    def _get_mean_squared_loss(self, label_op, inference_op):
        with tf.name_scope('regression_loss'):
            if self._weight_tensor is None:
                self._weight_tensor = tf.constant(1.0, name='weight')
            reg_loss = tf.losses.mean_squared_error(labels=label_op, predictions=inference_op, weights=self._weight_tensor)
            return tf.reduce_mean(reg_loss, name='reduce_mean_squared_loss')

    def build_graph(self, tensor_dict, algorithm_args, context):
        self.model._build_algorithm_graph(tensor_dict, algorithm_args, context, REGRESSION)
        self._inference_op = self.model._inference_op
        self._label_tensor = self.model._label_tensor
        self._weight_tensor = self.model._weight_tensor
        self._logistic_tensor = self._inference_op
        self._predictions = self._inference_op
        self._loss_op = self._get_mean_squared_loss(self._label_tensor, self._logistic_tensor)
        tf.summary.scalar('loss', self._loss_op)
        return True

    def get_train_op(self, optimizer, global_step):
        self.model._loss_op = self._loss_op
        return self.model.get_train_op(optimizer, global_step, REGRESSION)

    def get_algorithm_default_optimizer(self):
        return self.model.get_algorithm_default_optimizer()

    def get_inference_op(self):
        return tf.concat([1-self._inference_op, self._inference_op],1)
