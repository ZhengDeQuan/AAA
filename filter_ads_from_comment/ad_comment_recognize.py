# -*- encoding:utf-8 -*-
# from Gao LiEn
# @Author: Wego
# @Date: 2018.10

import re,itertools
phonePattern = [
                re.compile("\d{2}\-\d{2}\-\d{2}\-\d{2}\-\d{2}"),
                re.compile("\d{2}\-\d{3}\-\d{3}\-\d{2}") ]
numberPattern = re.compile("(\d+?)")


sensitiveCombineKeys = [
    ["job","money"],
    ["job","month"],
    ["job","day"],
    ["job","daily"],
    ["job", "apply"],
    ["per","day"],
    ["job", "income"],
    ["work", "income"],
    ["$","month"],
    ["$","daily"],
    ["money","daily"],
    ["money","month"],
    ["refer","id"],
    ["refer","code"],
    ["google","play"],
    ["donwload","http"],
    ["onead"],
    ["pay", "app"],
    ["pay", "job"],
    ["pay", "company"],
    ["bonus"],
    ["helpline"],
    ["store","app"],
    ["helpline"],
    ["earn", "month"],
    ["earn", "day"],
    ["earn", "$"],
    ["earn", "job"],
    ["call","get"],
    ["business", "whatsapp"],
    ["part","time","job"],
    ["vmate","uninstall"],
    ["ragistration", "investment"],
    ["registration", "investment"],
    ["registe", "investment"],    
    ['requir', 'boy'],
    ['requir', 'girl'],
    ["डाउनलोड", "कमायें"],       # 下载，赚
    ["कमाई"],                  # 收益
    ["कमायें","रुपया"],            # 赚，卢比
    ["कमायें","हज़ार ाशि"],         # 赚，千卢比
    ["रेफर", "कोड"]            # 推荐码
]

protectKeys = [
    ["vmate", "official"]
]

sensitiveKeys = set([])
for i in itertools.chain(sensitiveCombineKeys,protectKeys):
    for j in i:
        sensitiveKeys.add(j)
sensitiveKeys = list(sensitiveKeys)
acMation = None


def CountKeysByACM(body):
    body = body.lower()
    body = body.encode("utf-8")
    D = {}
    for i in acMation.iter(body):
        key = i[1][1]
        if key in D:
            D[key] += 1
        else:
            D[key] = 1
    return D
    

def CountKeysByStr(body):
    body = body.lower()
    D = {}
    for i in sensitiveKeys:
        x = body.find(i)
        if x != -1:
            D[i] = x
    return D
    

def CountProtect(D):
    DP = {}
    for i in protectKeys:
        key = "_".join(i)
        value = 0
        for j in i:
            if D.get(j) == None:
                value = 0
                break
            else:
                value = 1
        DP[key] = value
    return DP

def CountSensitive(D):
    DS = {}
    for i in sensitiveCombineKeys:
        key = "_".join(i)
        value = 0
        for j in i:
            if D.get(j) == None:
                value = 0
                break
            else:
                value = 1
        DS[key] = value
    return DS

def IsPhone(body):
    if len(phonePattern[0].findall(body))>= 1 or \
            len(phonePattern[1].findall(body))>= 1:
        return True
    else:
        return False


def LikeId(body):
    body = body.lower()
    m = len(numberPattern.findall(body))
    if m >= 10:
        N = len(body)
        if body.find("whatsapp") != -1 or body.find("phone") != -1:
            for i in range(N-20):
                if len(numberPattern.findall(body[i:i+20])) >= 10:
                    return True
            return False
        else:
            if m <= 15:
                return True
            return False
    else:
        return False



def FeatureComment(body):
    is_phone = IsPhone(body)
    like_id = LikeId(body)
    D = CountKeys(body)
    DS = CountSensitive(D)
    DP = CountProtect(D)
    return {
        "is_phone"  : is_phone,
        "like_id"   : like_id,
        "ds"        : DS,
        "dp"        : DP,
        "word_count": len(body.split())}     


def IsAdver(body):
    result = FeatureComment(body)
    if sum(result["dp"].values())>= 1:
        return False,result
    if  ( result["is_phone"] and sum(result["ds"].values())>= 1) or \
        ( sum(result["ds"].values())>= 1 and result["word_count"]>=5 and result["like_id"]==True ) or \
        ( sum(result["ds"].values())>=2):
        return True,result
    return False,result


try:
    import ahocorasick
    acMation = ahocorasick.Automaton()
    for index,word in enumerate(sensitiveKeys):
        acMation.add_word(word,(index,word))
    acMation.make_automaton()
    CountKeys = CountKeysByACM
except:
    CountKeys = CountKeysByStr
