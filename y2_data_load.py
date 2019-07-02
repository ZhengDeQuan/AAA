import pandas as pd
import tensorflow as tf
import numpy as np
import argparse
import pickle

#在y2_data_preprocess之后利用这个y2_data_load
#进行两个方面的实验：
#1.pandas处理数据中的异常值的检验和将数据进行字符串转化的检验，并且同时要注意分割X和Y
#2.利用fetch_data.py的逻辑，将数据分批导入，最后候选的feed_dict,(利用类中的self.X，可以解决变量可见的问题，不用担心.feed_dict{self.X:input_data_X})

#总之先一边实验，一边用pandas处理数据吧

def process(filename):
    # input_data = pd.read_csv(filename,delim_whitespace=True)
    # input_data = pd.read_csv(filename,sep="\t")
    # pickle.dump(tocsv_data, open(filename + ".processed.csv.pkl", "wb"))
    df_data = pickle.load(open(filename+".processed.csv.pkl","rb")) #一个DataFrame
    # import pdb
    # pdb.set_trace()
    df_data = df_data.dropna(how="all", axis=0) # 0 对行进行操作，how='any'只要有一个NA就删除这一行，how='all'必须全部是NA才能删除这一行
    #不能用any过滤，否则过滤完了，1000个只剩3个。
    df_data['label'] = (df_data['label']).astype(int)
    df_data = df_data[df_data['label'].isin([0,1])] #只保留label为0或者1的

    #分离X,Y
    X = df_data.drop(['label'],axis = 1).values.astype(str)
    Y = df_data['label'].values.astype(np.int32)
    return X,Y




def main(args):
    for filename in args.input_files:
        process(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', default=['/home2/data/ttd/train_ins_add', '/home2/data/ttd/eval_ins_add'],
                        nargs="+")
    args = parser.parse_args(['--input_files','/home2/data/ttd/zhengquan_test'])
    #args = parser.parse_args()
    print("args = ", args)
    main(args)