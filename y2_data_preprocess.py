import pandas as pd
import json
import pickle
import argparse
from tqdm import tqdm
import os
# column_data= pd.read_csv('All_Deep/feature_column_data.txt')
# print(column_data)
# import pdb
# pdb.set_trace()
from collections import OrderedDict



def process_data(file_name,Outer_List):
    '''
    将每一行数据都标准化
    '''
    print("in process_data")
    new_lines = []
    tocsv_data = []
    with open(file_name,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            Dict = {}
            #one example
            line = line.strip().split()
            label = line[1]
            line = line[2:]
            for pair in line:
                value,key = pair.split(":")
                if key not in Dict:
                    Dict[key]=[]
                Dict[key].append(value)
            sorted_tuple = sorted(Dict.items(),key = lambda z:z[0])

            Dict = dict(sorted_tuple)#这个dict是无序的()
            for key in Dict:
                Dict[key] = '|'.join(Dict[key])
            Dict['label'] = label
            tocsv_data.append(Dict)
            Outer_List.append(Dict)
            values = ['|'.join(ele[1]) for ele in sorted_tuple]
            values.insert(0,label)
            values = ' '.join(values)
            new_lines.append(values)

    # with open(file_name+".processed","w",encoding="utf-8") as fout:
    #     for line in new_lines:
    #         fout.write(line+"\n")
    # tocsv_data = pd.DataFrame(tocsv_data)
    if not os.path.exists(filename+".processed.csv.pkl"):
        pickle.dump(tocsv_data,open(filename+".processed.csv.pkl","wb"))
    # tocsv_data.to_csv(filename+".processed.csv",sep=" ") #要是能知道详细的参数变成csv文件就好了


def get_values_for_each_tag(file_name,Outer_Dict):
    print("in get_values....")
    with open(file_name,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm( lines):
            line = line.strip().split()
            line = line[2:]
            for pair in line:
                value,key = pair.split(":")
                if key not in Outer_Dict:
                    Outer_Dict[key] = set()
                Outer_Dict[key].add(value)


def cal_valueKindNum_for_each_tag_each_line(file_name,Outer_Dict):
    print("in cal_value")
    with open(file_name,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            one_line_dict = {}
            line = line.strip().split()
            line = line[2:]
            for pair in line:
                value , key = pair.split(":")
                if key not in one_line_dict:
                    one_line_dict[key]=0
                one_line_dict[key] += 1
            for key in one_line_dict:
                if key not in Outer_Dict:
                    Outer_Dict[key] = one_line_dict[key]
                else:
                    Outer_Dict[key] = max(Outer_Dict[key],one_line_dict[key])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files',default=['/home2/data/ttd/train_ins_add','/home2/data/ttd/eval_ins_add'],nargs="+")
    #args = parser.parse_args(['--input_files','/home2/data/ttd/zhengquan_test'])
    args = parser.parse_args()
    print("args = ",args)

    Outer_List = []
    Outer_Dict_tag2values = {}
    Outer_Dict_tag2valuesOneline = {}
    for filename in args.input_files:
        print("filename = ",filename)
        process_data(filename,Outer_List)
        get_values_for_each_tag(filename,Outer_Dict_tag2values)
        #cal_valueKindNum_for_each_tag_each_line(filename,Outer_Dict_tag2valuesOneline)

    sorted_tuple = sorted(Outer_Dict_tag2values.items(), key=lambda z: z[0])
    Outer_Dict_tag2values = dict(sorted_tuple)
    for key in Outer_Dict_tag2values:
        Outer_Dict_tag2values[key] = list(Outer_Dict_tag2values[key])
    json.dump(Outer_Dict_tag2values,open('tag2value.json',"w",encoding="utf-8"),ensure_ascii=False)
    pickle.dump(Outer_Dict_tag2values , open("tag2value.pkl","wb"))

    # sorted_tuple = sorted(Outer_Dict_tag2valuesOneline.items(), key=lambda z: z[0])
    # Outer_Dict_tag2valuesOneline = dict(sorted_tuple)
    # json.dump(Outer_Dict_tag2valuesOneline,open('tag2valueOneline.json',"w",encoding="utf-8"),ensure_ascii=False)
    # pickle.dump(Outer_Dict_tag2valuesOneline,open('tag2valueOneline.pkl',"wb"))

    # data = pickle.load(open("eval_ins_add.processed.csv.pkl", "rb"))
    # tmp = data.sample(frac=1).reset_index(drop=True)
    # sub_eval = tmp[0:int(tmp.shape[0] / 2)]
    # sub_test = tmp[int(tmp.shape[0] / 2), tmp.shape[0]]
    # pickle.dump(sub_eval, open("sub_eval_ins_add.processed.csv.pkl", "wb"))
    # pickle.dump(sub_test, open("sub_test_ins_add.processed.csv.pkl", "wb"))
