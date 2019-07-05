import pickle
from collections import defaultdict
import re

#语言缩写及其英文全称的对照表
# lang2code = defaultdict("NoneNuLL")
# with open("3.txt","r",encoding="utf-8") as fin:
#     lines = fin.readlines()
#     for line in lines:
#         line = line.strip().split()
#         line = line[:2]
#         code , lang = line
#         lang2code[lang]=code

#bert所包含的语言
with open("bert_multilingual_allowed_language.txt","r",encoding="utf-8") as fin:
    lines = fin.readlines()
    for line in lines:
        line = line.strip()
        line = re.sub('\((.*?)\)',"", line)
        print(line)




