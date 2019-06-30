import pandas as pd
'''
删除只有一个不同值的数据帧列
'''
for col in df.columns:
    if len(df[col].unique()) == 1:
        df.drop(col,inplace=True,axis=1)


'''
另一种不做就地丢弃的方法
'''
res = df
for col in df.columns:
    if len(df[col].unique()) == 1:
        res = res.drop(col,axis=1)