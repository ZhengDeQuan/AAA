'''
删除广告，表情，过短的评论
不需要进行情感分析
'''
# -*- coding: utf-8 -*-
import sys
import math
import re
import langid
# import nltk
# nltk.download('movie_reviews')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
import json
from collections import defaultdict
import pandas as pd

langage = defaultdict(int)
fail = defaultdict(int)


#定义一些全局变量用来过滤
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

# 对每条评论数据进行处理，提取信息
def comment_process(comment):

    #返回值
    res = defaultdict(int)

    #过滤规则
    filter_res=filter(comment)
    if filter_res!='true':
        fail[filter_res] += 1
        return {}

    #计算表情
    res['emoji']=cal_emo(comment)
    #删除句子中的表情
    comment=delete_emo(comment)
    #情感进行预测
    if(len(comment)>2):

        #语言进行预测
        lan = langid.classify(comment)
        langage[lan[0]] += 1

        ss = nltk_classifier.polarity_scores(comment)

        ss.pop('compound')
        #这里要过滤掉一些不显著的，即必须要大于0.5
        max_key=max(ss,key=ss.get)
        if ss[max_key]>0.5:
            res[max_key+'_num']=1
            res[max_key+'_value']=ss[max_key]

    return res


#以video_id为单位进提取特征
def main_process():
    index=0
    res= defaultdict(lambda:defaultdict(int))
    source_data = open('vmate_comment')
    #res_data = open('comment_video_res.txt', 'w')
    res_dic=open('comment_res.dic','w')
    for line in source_data.readlines():
        item = line.strip('\n').replace(',', ' ').split('\t')
        video_id=item[0]
        comment=item[1]
        comm_res=comment_process(comment)
        if comm_res=={}:
            continue
        res[video_id]['total_num']+=1
        #更新字典
        for key,value in comm_res['emoji'].items():
            if value>0:
                res[video_id][key]+=value

        res[video_id]['pos_num']+=comm_res['pos_num']
        res[video_id]['neg_num'] += comm_res['neg_num']
        res[video_id]['pos_value'] += comm_res['pos_value']
        res[video_id]['neg_value'] += comm_res['neg_value']
    #开始进行输出  在这里直接保存成字典，方便后面进行映射
    for key,value in res.items():
        #tmp = [key]
        if value['total_num']!=0:
            res[key]['total_num']=str(discretization(value['total_num'],2))
        if value['pos_value']!=0:
            res[key]['pos_value'] = str(discretization(value['pos_value'] / value['pos_num'], 1))
            res[key]['pos_num']=str(discretization(value['pos_num'],2))

        if value['neg_value'] != 0:
            res[key]['neg_value'] = str(discretization(value['neg_value'] / value['neg_num'], 1))
            res[key]['neg_num']= str(discretization(value['neg_num'],2))

    res_dic.write(json.dumps(res))
    print('无效的评论数据：')
    for key ,value in fail.items():
        print(key+':'+str(value))
    print('英语所占的比例：' + str(langage['en'] / sum(langage.values())))
    print('语言总数：'+str(sum(langage.values())))