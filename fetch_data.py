import numpy as np
x=[[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7]]
y=[[0,1],[1,0],[0,1],[0,1],[0,1],[1,0],[1,0]]
"""
生成批次数据
每个batch为2 ，每次有两条数据去更新模型(当数据总数为奇数的时候，则最后只有一条)
，总共轮训5次，也就是每条数据都有5次机会去更新模型的参数
"""
def batch_iter(data,batch_size=2,num_epochs=5):
    data=np.array(data)
    data_size=len(data)
    num_batchs_per_epchs=int((data_size-1)/batch_size)+1
    for epoch in range(num_epochs):
        indices=np.random.permutation(np.arange(data_size))
        shufflfed_data=data[indices]
        for batch_num in range(num_batchs_per_epchs):
            start_index=batch_num*batch_size
            end_index=min((batch_num + 1) * batch_size, data_size)
            yield shufflfed_data[start_index:end_index]
# x=[[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],[1,7]]
# y=[[0,1],[1,0],[0,1],[0,1],[0,1],[1,0],[1,0]]
"""
准备需要喂入模型的数据
"""
def feed_data(batch):
    x_batch, y_batch = zip(*batch)
    feed_dict = {
    "input_x": x_batch,
    "input_y": y_batch
    }
    return feed_dict, len(x_batch)

batch_train = batch_iter(list(zip(x, y)))
for i, batch in enumerate(batch_train):
    feed_dict, _ = feed_data(batch)
    print(i,"--->",feed_dict)
