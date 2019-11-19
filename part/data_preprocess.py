import numpy as np
import pandas as pd
import util.PATH as PATH
import time
import datetime

# 1. 从原始文件中提取元素，生成bug_msg_all文件以及将每个样本的描述文本抽取成一个文件夹。
# 2. 抽取词汇表形成文件，抽取开发者形成文件。
# 3. 抽取活跃度形成文件，数据集分割比例，8:2。验证集为从训练集中抽取出的具有一定规模的平衡数据集。

def read_raw_dataset_from_csv(dataset_path):
	df = pd.read_csv(dataset_path, header=None) # # bug_id, product, short_desc, long_desc, product, ?, 优先级, assignee, ?, 状态
	df = df.fillna('')      # 将所有的缺省值替换成指定字符
	return df

def filter_invalid_developers(datas):
	'''
	删除无效开发者修复的bug
	:param datas: 
	:return: 
	'''
	invalid_names = ['unassigned', 'issues', 'needsconfirm', 'swneedsconfirm', 'nobody', 'webmaster', 'inbox']
	invalid_bugs = []       # 记录无效bug的索引，这个索引是在DataFrame中的索引
	count = 0
	for i, data in datas.iterrows():
		developer = data[7].lower()
		for j in range(len(invalid_names)):
			if invalid_names[j] in developer:
				invalid_bugs.append(i)
				count += 1
	print('由无效开发者修复的bug总条数:{}'.format(count))
	# 删除所有无效的bug
	datas.drop(invalid_bugs, axis=0, inplace=True)  # 0表示按照索引index来删除
	return datas

def filter_inefficient_developers(datas):
	'''
	删除低效开发者修复的bug，并且统计低效开发者和高效开发者的数量，
	简单起见，我们称修复数量小于阈值的开发者为低效开发者，修复数量大于阈值的开发者为高效开发者。
	同时将高效开发者写入文件。
	:param datas: 
	:return: 
	'''
	fixed_bugs_num = {}     # key=developer,value=修复bug的bug索引
	inefficient_devs = []   # 低效开发者
	efficient_devs = []     # 高效开发者
	for index, data in datas.iterrows():        # 统计所有开发者修复的bug数量
		developer = data[7]
		if developer in fixed_bugs_num.keys():
			fixed_bugs_num[developer].append(index)           # 保存修复的bug在DataFrame中的索引
		else:
			# fixed_bugs_num.setdefault(data[7], [data[0]])
			fixed_bugs_num.setdefault(developer, [index])

	count = 0
	for key, value in fixed_bugs_num.items():       # 删掉低效开发者
		if len(value) < 10:
			inefficient_devs.append(key)
			# 删除索引对应的行
			datas.drop(value, axis=0, inplace=True)
			count += len(value)
		else:
			efficient_devs.append(key)
	# 将高效开发者写入文件保存
	with open(PATH.path_developer, 'w') as writer:
		for i in range(len(efficient_devs)):
			writer.write('{}\n'.format(efficient_devs[i]))
	print('高效开发者的数量为: {}'.format(len(efficient_devs)))
	print('低效开发者的数量为: {}'.format(len(inefficient_devs)))
	print('由低效开发者修复的bug数量:{}'.format(count))
	return datas

def create_bug_msg_all(datas):
	'''
	根据处理好的数据，创建bug_msg_all文件，注意，当前使用的数据集没有时间字段，为了拟合之前的代码，时间字段保留，统一设置成当前时间。
	bugid, developer 创建时间，最新修复时间，产品，组件。
	:param datas: 
	:return: 
	'''
	bug_msg_all = {}
	for index, data in datas.iterrows():
		bugid = int(data[0])
		developer = data[7]
		create_time = time.strftime('%Y-%m-%d %H:%M')
		delta_time = time.strftime('%Y-%m-%d %H:%M:%S')
		product = data[1]
		component = data[4]
		bug_msg_all[bugid] = [developer, create_time, delta_time, product, component]

	sorted_bug_msg_all = sorted(bug_msg_all.items(), key=lambda x:x[0])     # 按照bugid，有小到大排序。
	with open(PATH.path_bug_msg_all, 'w') as writer:
		for i in range(len(sorted_bug_msg_all)):
			writer.write('{}\t{}\n'.format(sorted_bug_msg_all[i][0], '\t'.join(sorted_bug_msg_all[i][1])))



def filter_low_and_high_frequency_words(datas):
	'''
	统计高频和低频词，保存每个样本的文本信息，保存词汇表文件。
	:param datas: 
	:return: 
	'''
	noise_words = []
	words_num = {}          # 统计出现的文档数
	for index, data in datas.iterrows():
		bugid = int(data[0])
		desc = data[2] + data[3]     # 短描述+长描述，短描述后面自带空格，所以不用在意
		words = set(desc.strip().split(' '))        # 因为要统计出现的文档数，所以重复词都不需要了
		for word in words:
			if word in words_num.keys():
				words_num[word] += 1
			else:
				words_num[word] = 1
	# 统计高频和低频个数
	thresold = len(words_num.keys()) * 0.5
	for key, value in words_num.items():
		if value < 5:                       # 少于阈值，视为低频
			noise_words.append(key)
		if value >= thresold:
			noise_words.append(key)

	for index, data in datas.iterrows():        # 之前部分行被删除，但是index并没有更新，注意
		bugid = int(data[0])
		desc = data[2]+data[3]
		words = desc.strip().split(' ')
		print(index)
		with open(PATH.path_corpus + str(bugid), 'w') as writer:        # 将每条bug对应的文本信息写成文件
			[writer.write('{}\n'.format(words[i])) for i in range(len(words)) if words[i] not in noise_words]

	print('处理前的单词数={}'.format(len(words_num.keys())))
	print('噪声单词数={}'.format(len(noise_words)))

	with open(PATH.path_vocabulary, 'w') as writer:     # 保存词汇表文件
		for word in words_num.keys():
			if word not in noise_words:
				writer.write('{}\n'.format(word))


if __name__ == '__main__':

	path = PATH.path_origin_corpus
	datas = read_raw_dataset_from_csv(path)
	datas = filter_invalid_developers(datas)
	datas = filter_inefficient_developers(datas)
	# create_bug_msg_all(datas)
	filter_low_and_high_frequency_words(datas)