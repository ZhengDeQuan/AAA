import pandas as pd
import json
import pickle

# column_data= pd.read_csv('All_Deep/feature_column_data.txt')
# print(column_data)
# import pdb
# pdb.set_trace()



def process_data(file_name,Outer_List):
    '''
    将每一行数据都标准化
    '''
    new_lines = []
    with open(file_name,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in lines:
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
            Dict = dict(sorted_tuple)
            Outer_List.append(Dict)
            values = ['|'.join(ele[1]) for ele in sorted_tuple]
            values = ' '.join(values)
            new_lines.append(values)

    with open(file_name+".processed","w",encoding="utf-8") as fout:
        for line in new_lines:
            fout.write(line+"\n")


def get_values_for_each_tag(file_name,Outer_Dict):
    with open(file_name,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.strip().split()
            for pair in line:
                value,key = pair.split(":")
                if key not in Outer_Dict:
                    Outer_Dict[key] = set()
                Outer_Dict.add(value)


if __name__ == "__main__":
    Outer_Dict = {}
    Outer_List = []
    for filename in args.filenames:
        pass

    for key in Outer_Dict:
        Outer_Dict[key] = list(Outer_Dict[key])
    json.dump(Outer_Dict,open('tag2valueList.json',"w",encoding="utf-8"),ensure_ascii=False)

