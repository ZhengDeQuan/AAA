import tensorflow as tf
batch_size = 4
feature_num = 3
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

csv4 = [
"wrestbrook|harden|durant",
"wrestbrook|harden|durant",
"wrestbrook|harden|durant"
]

csv_s= [csv1,csv2,csv3,csv4]
X = tf.placeholder(shape=[None,feature_num],dtype=tf.string)

one_feature = tf.contrib.layers.sparse_column_with_hash_bucket(
            column_name="zhengquan_test",
            hash_bucket_size=10,
            combiner="sum",
            dtype=tf.string
            # dtype=tf.dtypes.int32
        )

res = tf.contrib.layers.embedding_column(one_feature,
                                         # initializer=my_initializer,
                                         combiner="mean",
                                         dimension=3)
#除了有下面这种方法还有tf.unstack的方法
# for i in range(batch_size):
#     for j in range(feature_num):
#         one_feature = X[i][j]
#         one_feature = tf.reshape(one_feature,shape=[1])
#         split_tag = tf.string_split(one_feature, "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values= split_tag.values,
#             dense_shape=split_tag.dense_shape
#         )
#
#         current_mapping = {'zhengquan_test': one_sparse}
#         one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, res)
#         #[[ 0.08187684,  0.22063671, -0.16549297]]


#用unstack证明也是可行的，但是placeholder的第一个dimension不能是None，需要是一个确切的数值，不然unstack函数不能解析
# exp_X = tf.expand_dims(X,axis=-1)
# example_list = tf.unstack(exp_X,axis = 0)
# for one_example in example_list:
#     features = tf.unstack(one_example,axis = 0)
#     feature = features[0]
#     for one_feature in features:
#         # one_feature = tf.reshape(one_feature,shape=[1])
#         split_tag = tf.string_split(one_feature, "|")
#         one_sparse = tf.SparseTensor(
#             indices=split_tag.indices,
#             values= split_tag.values,
#             dense_shape=split_tag.dense_shape
#         )
#
#         current_mapping = {'zhengquan_test': one_sparse}
#         one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, res)
        #[[-0.10367388,  0.25915673, -0.00741819]]



def my_function(one_example):
    features = tf.unstack(one_example,axis = 0)
    for one_feature in features:
        split_tag = tf.string_split(one_feature, "|")
        one_sparse = tf.SparseTensor(
            indices=split_tag.indices,
            values= split_tag.values,
            dense_shape=split_tag.dense_shape
        )
        current_mapping = {'zhengquan_test': one_sparse}
        one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, res)
    return one_feature_embedding_res

exp_X = tf.expand_dims(X,axis=-1)
res = tf.map_fn(fn=my_function,elems=exp_X,dtype=tf.float32)
print(tf.shape(res))
import pdb
pdb.set_trace()
# res_seq = tf.squeeze(res,squeeze_dims=[-1])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess_res = sess.run([res],feed_dict={X:csv_s})
    print(type(sess_res))
    print(sess_res)

