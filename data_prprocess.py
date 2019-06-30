# -*- coding: utf-8 -*-
import sys
import  math
import json
from collections import defaultdict
import pandas as pd


#计算每个特征的平均ctr
def cal_feature_avg_value():
    res={}
    file=open('./../new_data/train_ins_add')
    res_file=open('./../new_data/feature_ctr_dic.txt','w')

    # 先把词典初始化完成
    res_dic=defaultdict(dict)
    # 给字典填充值
    index=0
    for line in file.readlines():
        click=line.split('\t')[1]
        val_id_list=line.split('\t')[2].split(' ')
        for val_id in val_id_list:
            item=val_id.split(':')
            if(len(item)!=2):
                print(index)
                print(line)
            val=val_id.split(':')[0]
            id=val_id.split(':')[1].strip('\n')
            if val not in res_dic[id].keys():
                res_dic[id][val]=[1,0]
                if click=='1':
                    res_dic[id][val][1] += 1
            else:
                res_dic[id][val][0]+=1
                if click=='1':
                    res_dic[id][val][1] += 1
        index+=1
    #计算平均ctr
    for id ,value_list in res_dic.items():
        value_num=len(value_list.keys())
        sum=0
        res[id]={}
        for val,value in value_list.items():
            #res_file.write(id+':'+val+' '+str(float(value[1])/value[0])+'\n')
            res[id][val]=str(float(value[1])/value[0])
            sum+=float(value[1])/value[0]
        #print (id+":"+str(float(sum)/value_num))
        res[id]['avg_ctr']=str(float(sum)/value_num)
    res_file.write(json.dumps(res))
   #print('done')

#计算每个特征的平均信息增益率，直接以特征为单位进行计算
def cal_feature_avg_entropy():
    res_dic={}
    feature_file=open('feature_list.txt')
    file=open('./data/train_eval_ins_merge')
    res_file=open('./data/ins_feature_entroy.txt','w')
    feature_id_list=[]
    # 先把词典初始化完成
    for line in feature_file.readlines():
        id=line.strip('\n').strip()
        feature_id_list.append(id)
        res_dic[id]=[0,0,0,0]
    # 给字典填充值
    index=0
    pos=0
    neg=0
    for line in file.readlines():
        read_list=[]
        click=line.split('\t')[1]
        if click=='1':
            pos+=1
        else:
            neg+=1
        val_id_list=line.split('\t')[2].split(' ')
        for val_id in val_id_list:
            id=val_id.split(':')[1].strip('\n')
            #包含此id的正负样本数
            if id not in read_list:
                if click=='1' :
                    res_dic[id][0] += 1
                else:
                    res_dic[id][1] += 1
                read_list.append(id)
        not_read_list=[tmp for tmp in feature_id_list if tmp not in read_list]
        for item in not_read_list:
            if click == '1':
                res_dic[item][2]+=1
            else:
                res_dic[item][3]+=1

    #总体信息熵
    pos=float(pos)
    neg=float(neg)
    all_entroy=-pos/(pos+neg)*math.log(2,pos/(pos+neg))- neg/(pos+neg)*math.log(2,neg/(pos+neg))
    #计算特征的信息增益率
    for id,val_list in res_dic.items():
        num=float(sum(val_list))
        pos_num=float(val_list[0]+val_list[1])
        neg_num = float(val_list[2] + val_list[3])
        #计算固有值

        #计算信息增益率
        if pos_num==num or neg_num==num:
            res_file.write(id+' '+str(all_entroy)+'\n')
        else:
            id_entroy = -pos_num / (pos_num + neg_num) * math.log(2, pos_num / (pos_num + neg_num)) - neg_num / (
                        pos_num + neg_num) * math.log(2, neg_num / (pos_num + neg_num))
            entroy=pos_num/num*(-val_list[0]/pos_num * math.log(2,val_list[0]/pos_num)-val_list[2]/neg_num * math.log(2,val_list[2]/neg_num))
            #print (id+' '+str((all_entroy-entroy)/id_entroy))
            res_file.write(id+' '+str((all_entroy-entroy)/id_entroy)+'\n')

#计算每个样本的新的93维度离散特征，注意顺序
def create_appear_value():
    res_file=open('eval_ins_93_dispersed.txt','w')
    id_list=[]
    list_file=open('feature_list.txt')
    for line in list_file.readlines():
        id =line.strip('\n')
        id_list.append(id)
    tmp=[str(id) for id in id_list]
    res_file.write('label '+' '.join(tmp))
    file = open('eval_ins')
    for line in file.readlines():
        click=line.split('\t')[1]
        res = {}
        res_list=[]
        for id in id_list:
            if line.find(':'+id+' ')>=0 or line.find(':'+id+'\n')>=0:
                res[id]=1
            else:
                res[id]=0
        for id in id_list:
            res_list.append(res[id])
        res_list1=[str(item) for item in res_list]
        res_file.write(str(click)+' '+' '.join(res_list1)+'\n')

def create_ctr_dic():
    file = open('./feature_ctr.txt')
    res_file=open('./feature_ctr_dic.txt','w')
    res={}
    for line in file.readlines():
        id=line.split(':')[0]
        val=line.split(':')[1].split(' ')[0]
        #行尾集的去掉‘\n’
        ctr = line.split(' ')[1].strip('\n')
        if id not in res.keys():
            res[id]={}
            res[id][val]=ctr
        else:
            if val not in res[id].keys():
                res[id][val] = ctr
    res_file.write(json.dumps(res))

#计算每个样本的新的93维度连续特征,注意保持顺序
def create_ctr_value():
    #加载字典
    dic=open('./../new_data/feature_ctr_dic.txt')
    data=dic.readline()
    ctr_dic=json.loads(data)
    #保存平均ctr作为缺失值处理
    id_list = []
    list_file = open('./../new_data/feature_list.txt')
    for line in list_file.readlines():
        id = line.strip('\n')
        id_list.append(id)
    res_file=open('./../new_data/eval_ins_continues.txt','w')
    file=open('./../new_data/eval_ins')
    for line in file.readlines():
        res = defaultdict(list)
        res_list=[]
        click=line.split('\t')[1]
        val_list=line.split('\t')[2].split(' ')
        for item in val_list:
            val=item.split(':')[0]
            id=item.split(':')[1].strip('\n')
            if val in ctr_dic[id].keys():
                res[id].append(ctr_dic[id][val])
        for id in id_list:
            if len(res[id])==0:
                res_list.append(ctr_dic[id]['avg_ctr'])
            else:
                tmp=[float(val.encode('utf-8')) for val in res[id]]
                res_list.append(str(sum(tmp)/len(tmp)))
        res_file.write(str(click)+' '+' '.join(res_list)+'\n')

#去除全为1的列，并保存其下标
def delete_all_1_feature():
    id_list = []
    list_file = open('feature_list.txt')
    for line in list_file.readlines():
        id = line.strip('\n')
        id_list.append(id)
    index=0
    res=[]
    data=pd.read_csv('./data/eval_ins_93_dispersed.txt',sep=' ')
    for col in data.columns:
        if 0 in data[col].tolist():
            res.append(col)
        else:
            del data[col]
    data.to_csv('./data/eval_ins_93_dispersed_delete.txt',index=None,sep=' ')
    res_file=open('dispersed_save_id.txt','w')
    for id in res:
        res_file.write(str(id)+'\n')

def res_formate():
    id_dic = {}
    file = open('lr_feature_dict.txt')
    for line in file.readlines():
        line_item= line.strip('\n').split('\t')
        id_dic[line_item[1]]=line_item[0]
    file=open('特征筛选结果整理.csv')
    res_file=open('lr_feature_survey_res.csv','w')
    for line in file.readlines():
        tmp=[]
        line_item=line.strip('\n').split(',')
        for item in line_item:
            if item in id_dic.keys():
                tmp.append(id_dic[item])
            else:
                tmp.append(item)
        res_file.write(','.join(tmp)+'\n')

#查看结果在每个结果列表中的位置
def res_test():
    data = pd.read_csv('特征筛选结果_连续.csv',sep=',',header=None)
    set0=set(data[20].tolist()[:10])


    column=list(data)
    for col in range(0,19,2):
        res=set()
        for id in  data[len(column)-2]:
            if not pd.isna(id):
                res.add(data[col].tolist().index(int(id)))
        set1= set(data[col].tolist()[:10])
        print(len(set0&set1))
        #print(res)

#查看不同方法结果的取前n个结果的并集包含了几个结果

def get_unio_res():
    data = pd.read_csv('特征筛选结果_连续.csv', sep=',',header=None)

    column = list(data)
    #首先保存一下结果的集合
    set0=set()
    for item in data[20].values:
        if not pd.isna(item):
            set0.add(item)
    print(set0)

    #先计算RF ,GBDT ,Xgboost特征为0的部分的交集
    dic=defaultdict(list)
    flag=0
    for col in [1,3,5]:
       for index in range(len(data[col].values)):
           flag+=1
           if data[col][index]==0:
               dic[col].append(data[col-1][index])
    set1=set(dic[1])&set(dic[3])&set(dic[5])
    print(set1)
    #先取Lasso和LR正则化的结果top5的并集
    set2 = set()
    for col in[6,8]:
        for index in range(5):
            set2.add(data[col][index])
    print(set2)
    #求一下两种方法的并集
    set3=set1.union(set2)
    # 取出来其他几种方法的top5,再和重要度取交集
    set5=set()
    for col in range(8,20,2):
        for index in range(5):
            set5.add(data[col][index])


def get_not_combine_feature_id():
    data=open('lr_feature_dict.txt')
    id_list=[]
    for line in data.readlines():
        item=line.strip('\n').split('\t')
        name=item[0]
        id=item[1]
        if name.find('-')<0 and (int(id)<150 or int(id )>170):
            id_list.append(id)
            print(line)
    print(len(id_list))


#给原始数据追加上新的特征
def cal_new_feature():
    #保存每个video id的新信息 聚类ID，machine_tag,
    res = defaultdict(lambda: defaultdict(int))
    file_machine_tag=open('machine_tags')
    machine_dic={}
    for line in file_machine_tag.readlines():
        item=line.strip('\n').split('\t')
        if len(item)==2:
            machine_dic[item[0]]=item[1]

    #评论数据
    file_comment=open('./../../comment/comment_res.dic')
    comment_dic=json.loads(file_comment.readline().strip('\n'))

    #聚类ID
    cluster_dic={}
    cluster_file=open('cluster_res.txt')
    for line in cluster_file.readlines():
        item=line.strip('\n').split(' ')
        cluster_dic[item[0]]='clustre-'+str(item[2])

    #开始写入特征
    train_source_file = open('./../new_data/eval_ins')
    train_res_file=open('./../new_data/eval_ins_add','w')
    for line in train_source_file.readlines():
        line=line.strip('\n')
        add_list=[]
        feature_list=line.split('\t')[2].split(' ')
        flag=0
        for feature in feature_list:
            feature_item=feature.split(':')
            value=feature_item[0]
            id=feature_item[1]
            if id =='7':
                if value in machine_dic:
                    add_list.append(str(hash(machine_dic[value]))+':409')
                if value in cluster_dic:
                    add_list.append(str(hash(cluster_dic[value])) + ':410')
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
                    train_res_file.write(line+' '+' '.join(add_list).strip(' ')+'\n')
                else:
                    train_res_file.write(line+'\n')
                flag=1
                break
        if flag==0:
            print(line)
            train_res_file.write(line+'\n')


#获取训练集的所有videoid
def get_video_id():
    res=set()
    map_dic={}
    train_file=open('train_ins')
    eval_file=open('eval_ins')
    map_file=open('id_map.txt')
    for line in map_file.readlines():
        item =line.strip('\n').split('\t')
        if len(item) > 2:
            map_dic[item[0]]=item[1]
        else:
            print(line)
    for line in train_file.readlines():
        item=line.strip('\n').split('\t')[2].split(' ')
        for feature in item:
            id=feature.split(':')[1]
            value=feature.split(':')[0]
            if id=='7' and value in map_dic.keys():
                res.add(value+'\t'+map_dic[value])
                break
    for line in eval_file.readlines():
        item=line.strip('\n').split('\t')[2].split(' ')
        for feature in item:
            id=feature.split(':')[1]
            value=feature.split(':')[0]
            if id=='7' and value in map_dic.keys():
                res.add(value+'\t'+map_dic[value])
                break
    for item in res:
        print(item)

#先处理一下聚类ID
def cluster_process():
    # 聚类ID
    dic = {}

    cluster_file = open('cluster_save.txt')
    for line in cluster_file.readlines():
        item = line.strip('\n').split('\t')
        if len(item)!=2:
            print(item)
        dic[item[0]] = item[1]

    file=open('id_map.txt')
    res_file=open('cluster_res.txt','w')
    for line in file.readlines():
        item=line.strip('\n').split('\t')
        video_id=item[1]
        if video_id in dic:
            res_file.write(item[0]+' '+item[1]+' '+dic[item[1]]+'\n')


def test1():
    file=open('2.log')
    for line in file.readlines():
        line=line.strip('\n')
        if(line.find('not')>=0):
            continue
        else:
            print(line)

if __name__ == "__main__":
    #cal_feature_avg_value()
    #cal_feature_avg_entropy()
    #create_ctr_value()
    #create_ctr_dic():
    #delete_all_1_feature()
    #create_appear_value()
    #res_formate()
    #res_test()
    #cluster_process()
    #get_unio_res()
    #get_not_combine_feature_id()
    #get_video_id()
    #cal_new_feature()
    test1()
    print('done')
