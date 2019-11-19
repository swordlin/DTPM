'''
本文件负责分析预测结果,DeepTriage在预测的时候, 我把top5的标签索引和真实标签索引都保存了下来,
本文件将根据保存的这些文件,来判断每个类别被正确修复的个数, 以及哪些样本被分错了.
属于对实验的再分析
'''
import numpy as np

def _read_result_file():
	filepath = '../data/results_predict/new_/analysis_windows_0.csv'
	# filepath = '../data/results_predict/analysis_windows_0.csv'
	pre_results = []        # 开发者id top5序列 真实标签的索引 活跃度列表的索引
	with open(filepath, 'r') as reader:
		for line in reader.readlines():
			pre_results.append(line.strip().split(','))
	return pre_results

def analyze_results(pre_results):
	dever_Nbugs = {}        # key=开发者的id, value=[总数, 分派正确的数量]
	for i in range(len(pre_results)):
		dever_id = pre_results[i][2]
		top_5_id = pre_results[i][1].split(' ')
		if dever_id in dever_Nbugs.keys():
			dever_Nbugs[dever_id][0] += 1
		else:
			dever_Nbugs.setdefault(dever_id, [1,0])
		if dever_id in top_5_id:
			dever_Nbugs[dever_id][1] += 1
	print('developer_id\t开发者修复的总bug数\t预测正确的比例')
	for id, value in dever_Nbugs.items():
		print('{}\t{}\t{}'.format(id, value[0], value[1]/value[0]))
	# 尝试计算G-mean值
	caculate_G_mean(dever_Nbugs)


def caculate_G_mean(dever_Nbugs):
	'''
	计算G-mean值
	:param dever_Nbugs: 
	:return: 
	'''
	G_mean = 1
	for id, value in dever_Nbugs.items():
		G_mean *= ((value[1]+1)/value[0])   # 加1是为了防止0
	# print(G_mean)
	G_mean = np.sqrt(G_mean)        # G-mean值极低,像是e^(-45)
	print(G_mean)
	print('%.6f' % G_mean)






if __name__ == '__main__':
    pre_results = _read_result_file()
    analyze_results(pre_results)