'''
from huangbo 过滤评论中的广告的脚本
'''
# -*- coding: utf-8 -*-
import sys
import math
import nltk
import math
import re
import langid
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

# 对每条评论数据进行处理，提取信息
def comment_process(comment):

    #返回值
    res = defaultdict(int)
    nltk_classifier = nltk_sentiment()

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

#给原始数据追加上新的特征
def cal_new_feature():
    #保存每个video id的新信息 聚类ID，machine_tag,
    # file_machine_tag=open('machine_tags')
    # machine_dic={}
    # for line in file_machine_tag.readlines():
    #     item=line.strip('\n').split('\t')
    #     if len(item)==2:
    #         machine_dic[item[0]]=item[1]

    #评论数据
    file_comment=open('comment_res.dic')
    comment_dic=json.loads(file_comment.readline().strip('\n'))

    # #聚类ID
    # cluster_dic={}
    # cluster_file=open('cluster_res.txt')
    # for line in cluster_file.readlines():
    #     item=line.strip('\n').split(' ')
    #     cluster_dic[item[0]]='clustre-'+str(item[2])

    #开始写入特征
    train_source_file = open('eval_ins')
    train_res_file=open('eval_ins_add','w')
    for line in train_source_file.readlines():
        line=line.strip('\n')
        add_list=[]
        feature_list=line.split('\t')[2].split(' ')
        for feature in feature_list:
            feature_item=feature.split(':')
            value=feature_item[0]
            id=feature_item[1]
            if id =='7':
                # if value in machine_dic:
                #     add_list.append(str(hash(machine_dic[value]))+':409')
                # if value in cluster_dic:
                #     add_list.append(str(hash(cluster_dic[value])) + ':410')
                if value in comment_dic:
                    value_dic=comment_dic[value]
                    if 'total_num' in value_dic:
                        add_list.append(str(hash(str(value_dic['total_num'])+'-total_num'))+':411')
                    if 'pos_num' in value_dic:
                        add_list.append(str(hash(str(value_dic['pos_num']) + '-pos_num+'))+':412')
                    if 'pos_value' in value_dic:
                        add_list.append(str(hash(str(value_dic['pos_value'])+ '-pos_value'))+':413')
                    if 'neg_num' in value_dic:
                        add_list.append(str(hash(str(value_dic['neg_num']) + '-neg_num'))+':414')
                    if 'neg_value' in value_dic:
                        add_list.append(str(hash(str(value_dic['neg_value']) + '-neg_value'))+':415')
                    for key in [item for item in value_dic.keys() if item not in ['pos_value','pos_num','neg_num','neg_value','total_num']]:
                        add_list.append(str(hash(str(comment_dic[value][key])+'-'+key))+':416')
                if len(add_list)>0:
                    print(add_list)
                    train_res_file.write(line+' '+' '.join(add_list)+'\n')
                break



#连续值进行离散化,num表示第几种离散化
#1：概率值  2：total_num
def discretization(value,num):
    if num==1:
        value=value*100-49
    try:
        return (int)((math.log(value)/math.log(2))+0.49)
    except Exception:
        print('error!'+':'+str(value))
    #return value

#删除所有的表情,并且长长度进行判断
def delete_emo(comment):
    res=''
    comment = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", comment)
    for item in comment:
        if ord(item) > 120000:
            continue
        res+=item
    return res

#获取情感模型
def nltk_sentiment():
    import nltk
    #nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    return sid
    # for sen in view:
    #     print(sen)
    #     ss = sid.polarity_scores(sen)
    #     for k in ss:
    #         print('{0}:{1},'.format(k, ss[k]), end='')



def get_emoji_list():
    data=open('emoji_list.txt')
    res=[]
    for line in data.readlines():
        tmp=line[line.index('[')+1:line.index(']')]
        res.append(tmp)
    print('表情包数量：'+str(len(res)))
    return res



if __name__ == "__main__":
    main_process()
    print('done')
