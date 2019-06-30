import pickle

data = pickle.load(open("static_about_comment.pkl","rb"))

comment_nums = list(data.values())
max_comment_num = max(comment_nums)
min_comment_num = min(comment_nums)
avg_comment_num = sum(comment_nums)/len(comment_nums)

Dict = {}
for ele in comment_nums:
    if ele not in Dict:
        Dict[ele] = 1
    else:
        Dict[ele] += 1

Dict = sorted(Dict.items(),key = lambda x:x[0])
print(Dict)
my_range = [ele[0] for ele in Dict]
my_value = [ele[1] for ele in Dict]
import matplotlib.pyplot as plt
plt.bar(my_range,my_value)
plt.show()

