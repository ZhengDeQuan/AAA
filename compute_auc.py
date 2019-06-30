# import numpy as np
# -*- coding: utf-8 -*-
import os
import sys

def compute_auc(labels, pred):
  if len(labels) != len(pred):
    print "error labels or pred"
    return 0
  sorted_pred = sorted(range(len(pred)), key = lambda i : pred[i])
  pos = 0.0
  neg = 0.0
  auc = 0.0
  last_pre = pred[sorted_pred[0]]
  count = 0.0
  pre_sum = 0.0  #当前位置之前的预测值相等的rank之和，rank是从1开始的，所以在下面的代码中就是i+1
  pos_count = 0.0  #记录预测值相等的样本中标签是正的样本的个数
  for i in range(len(sorted_pred)) :
    if labels[sorted_pred[i]] > 0:
      pos += 1
    else:
      neg += 1
    if last_pre != pred[sorted_pred[i]]: #当前的预测概率值与前一个值不相同
      #对于预测值相等的样本rank需要取平均值，并且对rank求和
      auc += pos_count * pre_sum / count
      count = 1
      pre_sum = i + 1     #更新为当前的rank
      last_pre = pred[sorted_pred[i]]
      if labels[sorted_pred[i]] > 0:
        pos_count = 1   #如果当前样本是正样本 ，则置为1
      else:
        pos_count = 0   #反之置为0
    else:
      pre_sum += i + 1    #记录rank的和
      count += 1          #记录rank和对应的样本数，pre_sum / count就是平均值了
      if labels[sorted_pred[i]] > 0:#如果是正样本
        pos_count += 1  #正样本数加1
  auc += pos_count * pre_sum / count #加上最后一个预测值相同的样本组
  auc -= pos *(pos + 1) / 2 #减去正样本在正样本之前的情况
  if pos == 0.0 or neg == 0.0:
    return 0.0
  auc = auc / (pos * neg)  #除以总的组合数
  return auc
