import numpy as np

import util.PATH as PATH
import matplotlib.pyplot as plt
'''
画一些辅助描述数据集的图形
'''
fixed_num = {}
with open(PATH.path_bug_msg_all, 'r') as reader:
	for line in reader.readlines():
		dev = line.strip().split('\t')[1]       # 提取开发者
		if dev in fixed_num.keys():
			fixed_num[dev] += 1
		else:
			fixed_num[dev] = 0

# y = sorted(fixed_num.values())
y = fixed_num.values()
print(len(y))
# plt.hist(y,bins=100, normed=0, histtype='step')
x = np.arange(0, len(y), 1)
# plt.plot(x, y, 'b-o')
plt.xlabel("developer's id")
plt.ylabel('the number of bugs fixed')
plt.scatter(x, y, alpha=0.5, s=30)

plt.show()