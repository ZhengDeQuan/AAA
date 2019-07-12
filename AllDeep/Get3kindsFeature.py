import json


userFeatures = []
videoFeatures = []
contextFeatures = []

def process(filename):
    List = []
    with open(filename,"r",encoding="utf-8") as fin:
        for line in fin.readlines():
            print(line)
            name, tag = line.strip().split()
            List.append(tag)
    print(List)
    json.dump(List,open(filename+".json","w",encoding="utf-8"),ensure_ascii=False)


if __name__ == "__main__":
    filenames = ['user_side_feature.txt','video_side_feature.txt','context_feature.txt']
    Lists = [userFeatures,videoFeatures,contextFeatures]
    for filename in filenames:
        process(filename)
