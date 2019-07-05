import pickle
import pandas as pd
import numpy as np
GLOBAL_BATCH_SIZE = 50

df_data = pickle.load(open("/home2/data/ttd/zhengquan_test.processed.csv.pkl","rb")) #一个DataFrame
# import pdb
# pdb.set_trace()
df_data = df_data.dropna(how="all", axis=0) # 0 对行进行操作，how='any'只要有一个NA就删除这一行，how='all'必须全部是NA才能删除这一行
#不能用any过滤，否则过滤完了，1000个只剩3个。
df_data['label'] = (df_data['label']).astype(int)
df_data = df_data[df_data['label'].isin([0,1])] #只保留label为0或者1的

#分离X,Y
X_data = df_data.drop(['label'],axis = 1)
X_data = X_data.applymap(str)
# X_data = X_data.values.astype(np.str)
X_data = X_data.values
Y_data = df_data['label'].values.astype(np.int32)

# def batch_iter(data,batch_size=2,num_epochs=5):
#     data=np.array(data)
#     data_size=len(data)
#     num_batchs_per_epchs=int((data_size-1)/batch_size)+1
#     for epoch in range(num_epochs):
#         indices=np.random.permutation(np.arange(data_size))
#         shufflfed_data=data[indices]
#         for batch_num in range(num_batchs_per_epchs):
#             start_index=batch_num*batch_size
#             end_index=min((batch_num + 1) * batch_size, data_size)
#             yield shufflfed_data[start_index:end_index]
#
# def feed_data(batch):
#     x_batch, y_batch = zip(*batch)
#     feed_dict = {
#     "input_x": x_batch,
#     "input_y": y_batch
#     }
#     return feed_dict, len(x_batch)
#
# import pdb
# batch_train = batch_iter(list(zip(X_data, Y_data)))
# for i, batch in enumerate(batch_train):
#     feed_dict, _ = feed_data(batch)
#     print(i,"--->",feed_dict)
#     pdb.set_trace()


def batch_iter2(data_size,batch_size):
    num_batchs_per_epchs = int((data_size - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_size))
    for batch_num in range(num_batchs_per_epchs):
        start_index=batch_num*batch_size
        end_index=min((batch_num + 1) * batch_size, data_size)
        yield indices[start_index:end_index]

batch_train = batch_iter2(1000,4)
for i, batch in enumerate(batch_train):
    print(i," <--> ",batch," <--> ",type(batch))
    X = X_data[batch]
    Y = Y_data[batch]
    print("X = ",X)
    print("Y = ",Y)
    import pdb
    pdb.set_trace()
