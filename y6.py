'''
尝试在这个代码基础之上修改
https://github.com/tensorflow/tensorflow/blob/r0.12/tensorflow/examples/learn/wide_n_deep_tutorial.py#L146
并且修改.embedding_column部分，使之能够加载预训练的权重
'''
import numpy as np
import tensorflow as tf
import pandas as pd
embeddings = {}
# with open('glove.6B.50d.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         values = line.strip().split()
#         w = values[0]
#         vectors = np.asarray(values[1:], dtype='float32')
#         embeddings[w] = vectors

vocab_size = 10
embedding_size = 8
embedding_matrix = np.random.uniform(-1, 1, size=(vocab_size, embedding_size))
# for w, i in word_index.items():
#     v = embeddings.get(w)
#     if v is not None and i < vocab_size:
#         embedding_matrix[i] = v

for i in range(vocab_size):
    embedding_matrix[i] = np.ones(embedding_size) * i

def my_initializer(shape=None, dtype=tf.float32, partition_info=None):
    assert dtype is tf.float32
    return embedding_matrix

params = {'embedding_initializer': my_initializer}

# education = tf.contrib.layers.sparse_column_with_hash_bucket(
#       "education", hash_bucket_size=10 ,
#     combiner="sum",
#     # dtype=tf.dtypes.string
#     dtype=tf.dtypes.int32
# )


# education = tf.contrib.layers.sparse_column_with_integerized_feature(
#     column_name="education",
#     bucket_size=10,
#     combiner='sum',
#     dtype=tf.dtypes.int64
# )

education = tf.contrib.layers.sparse_column_with_keys(
    column_name = "education",
    keys=np.array([0,1,2,3,4,5,6,7,8,9]).astype(np.int64),
    default_value=-1,
    combiner='sum',
    dtype=tf.dtypes.int64
)

res_one_hot_education = tf.contrib.layers.one_hot_column(education)

# res_of_edu = tf.contrib.layers.embedding_column(education,
#                                                 initializer=my_initializer,
#                                                 combiner="mean",
#                                                 dimension=8)


'''
'''
import pandas as pd

data={'c':['1','2'],'one':['5 7','6 7']}
df=pd.DataFrame(data)
#df=pd.DataFrame(np.arange(16).reshape((4,4)),index=['a','b','c','d'],columns=['one','two','three','four'])


# for_map_education = {'education':tf.SparseTensor(
#     indices=[[i,0] for i in range(df['one'].size)],
#     values = df['one'].values.astype(str),
#     dense_shape = [df['one'].size,1]
# )}

for_map_education = {'education':tf.SparseTensor(
    indices=[[0,0],[0,1],[1,0],[1,1]],
    values = np.array([0,1,2,3]).astype(np.int64),
    dense_shape = [2,2]
)}


# res = tf.feature_column.input_layer(for_map_education,res_of_edu)
res = tf.feature_column.input_layer(for_map_education,res_one_hot_education)
#
#
#
# '''
# another try
# '''
#
#
# csv = [
#   "1,harden|james|curry",
#   "2,wrestbrook|harden|durant",
#   "3,|paul|towns",
# ]
#
# TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]
#
#
# ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
# table = tf.contrib.lookup.index_table_from_tensor(
#   mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
# split_tags = tf.string_split(post_tags_str, "|")
# temp_value = table.lookup(split_tags.values)
# tags = tf.SparseTensor(
#   indices=split_tags.indices,
#   values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
#   dense_shape=split_tags.dense_shape)
#
# TAG_EMBEDDING_DIM = 8
# embedding_params = tf.Variable(tf.truncated_normal([10, TAG_EMBEDDING_DIM]))
# # embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)



with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    print(sess.run(res))
    # print(sess.run(for_map_education))
    # print(sess.run(tags))
    # print(sess.run(post_tags_str))
    # print(sess.run(temp_value))
    # print(sess.run(embedded_tags))

