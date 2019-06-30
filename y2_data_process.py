'''
要处理的数据，类似于以下的情况
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

csv_s= [csv1,csv2,csv3]

csv1 csv2 csv3 代表三个样本，csv_s代表一个bathc中的数据
csvi中的每一个元素，用逗号间隔的，代表一个特征
这里模拟的是一个特征有多个取值的时候的情况。

plan：每个样例用一个dict来装，dict中的键对应tag_id,值是一个列表，每个元素是这个tag的取值。
每个样例的键值对公有108个。
所有的样例用一个list来装。最后转换成DataFrame，每个tag_id对应的多个值也有冒号来join起来，变成字符串。
读取之后注意将第0列，第1列转化为int，
然后将第0列删除即全部为1的列删除，
将第1列中，取值不为1也不为0的删除。（第一列为label）
还要将两个信息读入，
一个是每个tag_id,最多能对应多少种取值，种类少的ont_hot,种类多的hash_bucket+embedding,中间的embedding
另一个是每个tag_id在每个样例中最多对应多少取值，这样就可以估计所有的特征concat之后有多少维度。

需要手动构造的外部文件有两类：
第一类：就是上面说的，那些tag_id对应的特征少，多，中。
第二类：每个特征都是hash之后的int64位的数值，每个int64位的值需要映射到保留到一个文件中，比如以json的形式存成列表，
在（tf.contrib.lookup.index_table_from_tensor）中需要用到。
如下所示。
TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]
table = tf.contrib.lookup.index_table_from_tensor(mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
'''

import pickle


