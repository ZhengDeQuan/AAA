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
X = tf.placeholder(shape=[batch_size,feature_num],dtype=tf.string)

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
for i in range(batch_size):
    for j in range(feature_num):
        one_feature = X[i][j]
        one_feature = tf.reshape(one_feature,shape=[1])
        split_tag = tf.string_split(one_feature, "|")
        one_sparse = tf.SparseTensor(
            indices=split_tag.indices,
            values= split_tag.values,
            dense_shape=split_tag.dense_shape
        )

        current_mapping = {'zhengquan_test': one_sparse}
        one_feature_embedding_res = tf.feature_column.input_layer(current_mapping, res)
        #[[ 0.08187684,  0.22063671, -0.16549297]]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess_res = sess.run([one_feature_embedding_res],feed_dict={X:csv_s})
    print(type(sess_res))
    print(sess_res)

