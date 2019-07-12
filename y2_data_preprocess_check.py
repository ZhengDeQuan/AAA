'''
检查y2_data_preprocess.py处理输出的数据是否有对齐错误
'''

from tqdm import tqdm

def check(origianl_file,processed_file):
    with open(origianl_file,"r",encoding="utf-8") as fin1, open(processed_file,"r",encoding="utf-8") as fin2:
        lines1 = fin1.readlines()
        lines2 = fin2.readlines()
        print("lines1 = ",len(lines1))
        print("lines2 = ",len(lines2))
        idx = -1
        import pdb
        pdb.set_trace()
        for line1,line2 in tqdm(list(zip(lines1,lines2))):
            idx += 1
            # print("idx = ",idx)
            line1 = line1.strip().split()
            line2 = line2.strip().split()
            label1 = line1[1]
            label2 = line2[0]
            if label1 != label2:
                print("idx = ",idx)

def check2(filename):
    '''
        将每一行数据都标准化
        '''
    print("in process_data")
    with open(filename, "r", encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm(lines):
            Dict = {}
            line = line.strip().split()
            print("line = ")
            print(line)
            label = line[1]
            line = line[2:]
            for pair in line:
                value, key = pair.split(":")
                if key not in Dict:
                    Dict[key] = []
                Dict[key].append(value)
            sorted_tuple = sorted(Dict.items(), key=lambda z: z[0])
            print(sorted_tuple)
            values = ['|'.join(ele[1]) for ele in sorted_tuple]
            print(values)
            values.insert(0, label)
            values = ' '.join(values)
            import pdb
            pdb.set_trace()

def check3(filename_train,filename_eval):
    import pickle
    df_train = pickle.load(open(filename_train,"rb"))
    df_eval = pickle.load(open(filename_eval,"rb"))
    userTrain = df_train[['2']].values.tolist()
    userEval = df_eval[['2']].values.tolist()
    userEval = [ele[0] for ele in userEval]
    userTrain = [ele[0] for ele in userTrain]
    userTrain = set(userTrain)
    userEval = set(userEval)
    have = 0
    havent = 0
    total = 0
    for userid in userEval:
        total += 1
        if userid in userTrain:
            have += 1
        else:
            havent += 1
    print(have)
    print(havent)
    print(total)
    print(have/total)
    # 16530
    # 300565
    # 317095
    # 0.05212948800832558

def check4(filename_train,filename_eval):
    import pickle
    df_train = pickle.load(open(filename_train,"rb"))
    df_eval = pickle.load(open(filename_eval,"rb"))
    userTrain = df_train[['7']].values.tolist()
    userEval = df_eval[['7']].values.tolist()
    userEval = [ele[0] for ele in userEval]
    userTrain = [ele[0] for ele in userTrain]
    userTrain = set(userTrain)
    userEval = set(userEval)
    have = 0
    havent = 0
    total = 0
    for userid in userEval:
        total += 1
        if userid in userTrain:
            have += 1
        else:
            havent += 1
    print(have)
    print(havent)
    print(total)
    print(have/total)
    # 24711
    # 89412
    # 114123
    # 0.21652953392392418


if __name__ == "__main__":
    # check("/home2/data/ttd/train_ins_add","/home2/data/ttd/train_ins_add.processed")
    # check2("/home2/data/ttd/train_ins_add")
    check4("/home2/data/ttd/eval_ins_add.processed.csv.pkl","/home2/data/ttd/train_ins_add.processed.csv.pkl")