import re,itertools

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
'''

迭代器
chain()

实参
p, q, ...

结果
p0, p1, ... plast, q0, q1, ...


示例
chain('ABC', 'DEF') --> A B C D E F
'''
for i in itertools.chain(sensitiveCombineKeys,protectKeys):
    for j in i:
        sensitiveKeys.add(j)

