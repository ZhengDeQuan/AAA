'''
删除广告，表情，过短的评论
不需要进行情感分析

处理逻辑很简单:读入处理，保存处理的结果，写出到filename.filerComment

'''

from collections import defaultdict
import re
import argparse
from tqdm import tqdm

#怀疑是广告评论的关键词，如果评论中出现了这些组合中的某一个就删除该评论。
delete_list = [
    ["job", "money"],
    ["job", "month"],
    ["job", "day"],
    ["job", "daily"],
    ["job", "apply"],
    ["per", "day"],
    ["job", "income"],
    ["work", "income"],
    ["$", "month"],
    ["$", "daily"],
    ["money", "daily"],
    ["money", "month"],
    ["refer", "id"],
    ["refer", "code"],
    ["google", "play"],
    ["donwload", "http"],
    ["pay", "app"],
    ["pay", "job"],
    ["pay", "company"],
    ["store", "app"],
    ["earn", "month"],
    ["earn", "day"],
    ["earn", "$"],
    ["earn", "job"],
    ["call", "get"],
    ["business", "whatsapp"],
    ["part", "time", "job"],
    ["vmate", "uninstall"],
    ["ragistration", "investment"],
    ["registration", "investment"],
    ["registe", "investment"],
    ["requir", "boy"],
    ["requir", "girl"],
    ["डाउनलोड", "कमायें"],
    ["कमायें", "रुपया"],
    ["कमायें", "हज़ार ाशि"],
    ["रेफर", "कोड"]]
# ["डाउनलोड", "कमायें"],  # 下载，赚
# ["कमाई"],  # 收益
# ["कमायें", "रुपया"],  # 赚，卢比
# ["कमायें", "हज़ार ाशि"],  # 赚，千卢比
# ["रेफर", "कोड"]  # 推荐码

langage = defaultdict(int) #统计所有的评论,每种语言的有多少个
commentKind2num = defaultdict(int) #统计所有的评论中，每种被刷掉的情况有多少个 adv, like_adv, small_len


# 计算表情的数目
def cal_emo(comment):
    res=defaultdict(int)
    emo_list=re.findall('\[(.*?)\]', comment)
    for item in emo_list:
        res[item]+=1
    return res

#删除所有的表情,并且长长度进行判断
def delete_emo(comment):
    res=''
    comment = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", comment)
    for item in comment:
        if ord(item) > 120000:
            continue
        res+=item
    return res

# 过滤广告
def filter(comment):
    #广告
    adv = ['Arrey! Aapne sticker nahi try kiya? Stickers lagao aur likes paao']
    adv.append('Thank You! Aapka duet video acha laga[hearteye][heart]')
    for item in adv:
        if comment.find(item) >= 0:
            return 'adv'

    for item in delete_list:
        if comment.find(item[0])>=0 and comment.find(item[1])>=0:
            #print(comment)
            return 'like_adv'

    # 字符长度小于3的并且不含表情的
    if len(comment)<3 and  len(re.findall('\[(.*?)\]', comment))<1:
         return 'small_len'
    return 'true'


def comment_process(comment):
    filter_res = filter(comment)
    commentKind2num[filter_res] += 1
    if filter_res != 'true':
        return {}, None
    res = defaultdict(int)
    # 计算表情
    res['emoji'] = cal_emo(comment)
    # 删除句子中的表情
    comment = delete_emo(comment)
    # if len(comment.strip()) < 3:
    #     comment = None
    return res , comment

def main(filename):
    new_lines = []
    len_not_2 = 0
    with open(filename,"r",encoding="utf-8") as fin:
        lines = fin.readlines()
        for line in tqdm( lines ):
            line = line.strip().split('\t')
            if len(line) != 2:
                len_not_2 += 1
                continue
            video_id, o_comment = line
            # print("video_id = ",video_id," comment = ",o_comment)
            res , comment = comment_process(o_comment.strip())
            if comment is not None:
                if len(comment.strip()) < 4  and len(comment.strip().split()) < 3:
                    # print('\t'.join([video_id,comment]))
                    continue
                new_lines.append('\t'.join([video_id,comment]))
    with open(filename+".filterComment","w",encoding="utf-8") as fout:
        for line in new_lines:
            fout.write(line+"\n")
    print("len_not_2 = ",len_not_2)
    print("keep  = ",len(new_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data',type=str,default="vmate_comment")
    args = parser.parse_args()
    main(args.input_data)

