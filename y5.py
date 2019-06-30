'''
多离散值处理方案
'''

import tensorflow as tf

csv = [
  "1,harden|james|curry",
  "2,wrestbrook|harden|durant",
  "3,|paul|towns",
]

TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]


ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
'''
tf.io.decode_csv(
    records,
    record_defaults,
    field_delim=',',
    use_quote_delim=True,
    name=None,
    na_value='',
    select_cols=None
)
'''
print("ids = ",ids)
print("post_tags_str = ",post_tags_str)

table = tf.contrib.lookup.index_table_from_tensor(mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
split_tags = tf.string_split(post_tags_str, "|" )
split_tags_values = split_tags.values
tags = tf.SparseTensor(
  indices=split_tags.indices,
  values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
  dense_shape=split_tags.dense_shape)

TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))

embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)

with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print("ids = ")
  print(s.run(ids))
  print("post_tags_str = ")
  print(s.run(post_tags_str))
  #print(s.run(table))
  print("split_tags = ")
  print(s.run(split_tags))
  print("split_tags_values")
  print(s.run(split_tags_values))
  print("tags = ")
  print(s.run([tags]))



