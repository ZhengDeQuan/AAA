'''
将原始的数据train_ins_add,eval_ins_add
每一行一个item
每个item有不定个数的属性
每个属性有不定个数的值

统计有多少个属性
以及每个属性有多少个不同的离散值
将每个属性下的离散值归一化

2000000 train_ins_add
 300000 eval_ins_add
2300000 total

'''
import sys
import os
import argparse
import pickle
from tqdm import tqdm
import json
def cal_feature_value_kind(data_file,Dict):
    print("data_file = ",data_file)
    #data_file = "/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/temp.txt"
    with open(data_file,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            line = line.strip().split()
            line = line[1:]#第0列全部都是1，所以跳过
            if int(line[0])==2:
                continue
            label = line[0]
            line = line[1:]#把标签拿出来。
            for pair in line:
                feature_hash_value, feature_id = pair.split(":")
                if feature_id not in Dict:
                    Dict[feature_id] = set()
                Dict[feature_id].add(feature_hash_value)

def main(args):
    Dict = {}
    for data_file in args.input_data_files:
        cal_feature_value_kind(data_file,Dict)
    for feature_id in Dict:
        print("feature_id = ",feature_id, "has : ", len(Dict[feature_id]))
    print("total feature = ",len(Dict))
    pickle.dump(Dict,open("feature_value_kind_dict.pkl","wb"))



'''
统计每个样本中每个tag最多有多少个取值
'''

def process_one_line(line):
    line = line.strip().split()
    line = line[2:]#第一列一直是1，第二列是是否点击的标志
    Dict = {}
    for pair in line:
        v,v_id = pair.split(":")
        if v_id not in Dict:
            Dict[v_id] = 1
        else:
            Dict[v_id] += 1
    return Dict


def process2(filename,Dict):
    with open(filename,"r",encoding="utf-8") as fin:
        for line in fin.readlines():
            cus_Dict = process_one_line(line)
            for ele in cus_Dict:
                Dict[ele] = max(Dict[ele],cus_Dict[ele])


def main2(args):
    data = pickle.load(open("feature_value_kind_dict.pkl","rb"))
    keys = list(data.keys()) #所有的tag id
    Dict = {}
    for ele in keys:
        Dict[ele] = 0
    for filename in args.input_data_files:
        process2(filename,Dict)

    pickle.dump(Dict,open("ecch_item_each_tag_max_len.pkl","wb"))
    for ele in Dict:
        print(ele, " ", Dict[ele])




'''
统计每个特征值出现的次数，将出现次数<5的特征值去掉
'''

def cal_feature_value_kind_and_times(data_file,Dict):
    print("data_file = ",data_file)
    #data_file = "/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/temp.txt"
    with open(data_file,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            line = line.strip().split()
            line = line[1:]#第0列全部都是1，所以跳过
            if int(line[0])==2:
                continue
            label = line[0]
            line = line[1:]#把标签拿出来。
            for pair in line:
                feature_hash_value, feature_id = pair.split(":")
                if feature_id not in Dict:
                    Dict[feature_id] = {feature_hash_value:1}
                else:
                    if feature_hash_value not in Dict[feature_id]:
                        Dict[feature_id][feature_hash_value] = 0
                    Dict[feature_id][feature_hash_value] += 1


def main3(args):
    Dict = {}
    for data_file in args.input_data_files:
        cal_feature_value_kind_and_times(data_file,Dict)
    for feature_id in Dict:
        #print("feature_id = ",feature_id, "has : ", len(Dict[feature_id]))
        chosen_feature_value_num = 0
        for feature_value in Dict[feature_id]:
            #print("feature_id=",feature_id," feature_value=",feature_value," occur_times=",Dict[feature_id][feature_value])
            if Dict[feature_id][feature_value] > 5:
                chosen_feature_value_num += Dict[feature_id][feature_value]
        print("feature_id = \t",feature_id, "has : \t", chosen_feature_value_num)
    print("total feature = ",len(Dict))
    pickle.dump(Dict,open("feature_value_kind_and_times_dict.pkl","wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_data_files','--input_data_files',nargs="+",default=['/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/train_ins_add','/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/eval_ins_add'])
    args = parser.parse_args()
    print(args)
    # main(args)
    # main2(args)
    main3(args)
