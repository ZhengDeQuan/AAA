'''
先实现骨架
'''
import tensorflow as tf
import numpy as np

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
wide_inputs = tf.random_normal(shape=[GLOBAL_BATCH_SIZE,GLOBAL_WIDE_DIMENSION], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
deep_inputs = tf.truncated_normal(shape=[GLOBAL_BATCH_SIZE,GLOBAL_DEEP_DIMENSION], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
# Y = np.arange(GLOBAL_BATCH_SIZE).reshape(GLOBAL_BATCH_SIZE,1)
# Y = tf.constant(value=Y, dtype=tf.int32, shape=[GLOBAL_BATCH_SIZE, ])
Y = tf.ones(shape=[GLOBAL_BATCH_SIZE,1],dtype=tf.float32)

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

# 初始化回话sess并开始迭代训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 验证集待喂入数据
    for i in range(TRAINING_STEPS):
        if i % 1000 == 0:
            validate_acc = sess.run(accuracy)
            print('After %d training steps, validation accuracy'
                  ' using average model is %f' % (i, validate_acc))

        sess.run(train_op)

    test_acc = sess.run(accuracy)
    print('After %d training steps, test accuracy'
          ' using average model is %f' % (TRAINING_STEPS, test_acc))