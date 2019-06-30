import argparse
import pickle

def process(data_file,video_ids):
    print("data_file = ",data_file)
    lineNUM_videoID= []
    with open(data_file,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in lines:
            line = line.split()[2:]
            flag = False
            for pair in line:
                feature_value,feature_id = pair.split(":")
                if feature_id == '7':
                    video_ids.add(feature_value)
                    lineNUM_videoID.append(feature_value)
                    flag=True
                    break
            assert flag

    assert len(lineNUM_videoID) == len(lines)
    pickle.dump(lineNUM_videoID,open(data_file+"lineNUM_videoID","wb"))




def main(args):
    video_ids = set()
    for data_file in args.input_data_file:
        process(data_file,video_ids)
    print("video_ids = ",len(video_ids))#video_ids =  125079
    pickle.dump(video_ids,open("video_ids.pkl","wb"))


def process2(a,b):
    import pandas as pd
    df = pd.read_csv(b,header=None,delim_whitespace=True)
    lineNUM_videoID = pickle.load(open(a+"lineNUM_videoID",'rb'))
    df[df.shape[1]] = lineNUM_videoID
    df.to_csv(b+"lineNUM_videoID",mode="w",sep=" ",header=None,index=None)

def main2(args):
    for a , b in zip(args.input_data_file,args.input_data_file2):
        process2(a,b)



def check(args):
    for a,b in zip(args.input_data_file,args.input_data_file3):
        with open(a,"r",encoding="utf-8") as fina, open(b,"r",encoding="utf-8") as finb:
            linesa = fina.readlines()
            linesb = finb.readlines()
            for la,lb in zip(linesa,linesb):
                videoIDb = lb.split()[-1]
                la = la.split()
                la = la[2:]
                videoIDa = None
                for ele in la:
                    v,v_id = ele.split(":")
                    if v_id == "7":
                        videoIDa = v
                        break
                assert videoIDa is not None
                assert videoIDa == videoIDb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_data_file', '--input_data_file',nargs="+",
                        default=["/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/eval_ins_add",
                                 "/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/train_ins_add"])
    parser.add_argument('-input_data_file2', '--input_data_file2', nargs="+",
                        default=["/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/eval_ins_continues.txt",
                                 "/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/train_ins_continues.txt"])
    parser.add_argument('-save_file', '--save_file', default="/home4/data/zhengquan/comment_vector")
    parser.add_argument('-input_data_file3', '--input_data_file3', nargs="+",
                        default=["/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/eval_ins_continues.txtlineNUM_videoID",
                                 "/home3/data/zhengquan/from_huangbo/lr_feature_select/new_data/train_ins_continues.txtlineNUM_videoID"])
    args = parser.parse_args()
    main(args)
    print("main_finish")
    main2(args)
    print("main2_finish")
    check(args)
