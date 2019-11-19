'''
1. 统计各开发者修复的bug总数;
2. 抽取少量开发者以及由他们修复的bug,组成一个小型数据集用来验证差分进化的不平衡问题.暂定10个吧
'''
from util.data_helper import get_msg_all
import util.PATH as PATH
import numpy as np
import random

def count_bug_fixed_by_each_developer():
	bugid_each_developer = {}       # key=developers' name , value=[bug_ids]
	bug_msg_all, _ = get_msg_all()
	# bug_msg_all = {}
	for bugid, value in bug_msg_all.items():
		der = value[0]      # 获取当前样本的修复者
		if der in bugid_each_developer.keys():
			bugid_each_developer[der].append(bugid)
		else:
			bugid_each_developer.setdefault(der, [bugid])
	# with open('../data/num_of_bugs_fixed_by_each_developer.txt', 'w') as writer:
	# 	for name in bugid_each_developer.keys():
	# 		writer.write('{}\t{}\n'.format(name, len(bugid_each_developer[name])))
	return bugid_each_developer

def extract_little_bugs_by_developers(bugid_each_developer):
	# selected_names = []

	# 随机抽取指定数目的开发者。。。
	# 首先抽取指定数目的随机索引
	all_developer_names = list(bugid_each_developer.keys())
	# print(all_developer_names)
	selected_names = random.sample(all_developer_names, 30)
	'''
		selected_names = [       # 需要抽取的开发者的名字
			# 'Darin_Wright@ca.ibm.com',      # 2831
			'Tod_Creasey@ca.ibm.com',       # 2559
			'martin_aeschlimann@ch.ibm.com',    # 2314
			'markus_keller@ch.ibm.com',     # 750
			'kim_horne@ca.ibm.com',         # 702
			'veronika_irvine@ca.ibm.com',         # 688
			'guru.nagarajan@intel.com',         # 202
			'kchong@ca.ibm.com',         # 202
			'lynne_kues@us.ibm.com',         # 203
			'shaun@uvic.ca',         # 30
			'wayne@eclipse.org',         # 31
			'tkoeck@gup.jku.at',         # 31
			'jhalhead@ca.ibm.com',         # 32
			'mahutch@ca.ibm.com',         # 34
			'ohurley@iona.com',         # 20
			'ljagga@us.ibm.com',         # 21
			'aviman@ca.ibm.com',         # 21
		]'''
	print(selected_names)
	# 得到对应的开发者names列表
	# for i in indexs:
	# 	selected_names.append(all_developer_names[i])
	# 抽取指定开发者修复的所有bug的bugid
	little_bugs = {}
	train_little_bugs = {}
	eval_little_bugs = {}
	# 准备计数
	count_number_of_each_dev = {}
	for name, value in bugid_each_developer.items():
		if name in selected_names:
			little_bugs[name] = bugid_each_developer[name]
			# 统计每个开发者的修复数量
			count_number_of_each_dev[name] = len(little_bugs[name])
			# 按照8:2划分训练集和测试集
			train_little_bugs[name] = little_bugs[name][:int(0.8*len(little_bugs[name]))]
			eval_little_bugs[name] = little_bugs[name][int(0.8*len(little_bugs[name])):]
	_write_little_bug_id(train_little_bugs, PATH.path_little_train_bug_msg_all)
	_write_little_bug_id(eval_little_bugs, PATH.path_little_eval_bug_msg_all)
	print(count_number_of_each_dev)
	# 记录开发者的修复数量分布
	_write_the_number_of_each_category(count_number_of_each_dev)
	# _writer_little_bug_msg_all(train_little_bugs)



def _write_little_bug_id(little_bugids, filepath):
	'''
	将输入的bugid写入文件
	:param little_bugids: 
	:param filepath: 
	:return: 
	'''
	with open(filepath, 'w') as writer:
		for name, value in little_bugids.items():
			for bugid in value:
				writer.write('{}\n'.format(bugid))  # 记录下抽取的bug的bugid

def _read_little_bug_id(filepath):
	little_bugs = []
	with open(filepath, 'r') as reader:
		for line in reader.readlines():
			little_bugs.append(int(line.strip()))   # 注意这里要转成int, 不然后续无法处理
	return little_bugs

def _writer_little_bug_msg_all(little_bugids):
	'''
	这个文件是测试用的,是为了查看划分的训练集是否正确
	:param little_bugids: 
	:return: 
	'''
	bug_msg_all, _ = get_msg_all()
	with open('../data/eval_little_bug_msg_all.txt', 'w') as writer:
		for name, value in little_bugids.items():
			for bugid in value:
				# writer.write('{}\n'.format('\t'.join(bug_msg_all.get(bugid))))
				writer.write('{}\t{}\n'.format(bugid, bug_msg_all.get(bugid)[0]))

def _write_the_number_of_each_category(count_number_of_each_dev):
	'''
	将每个开发者修复的bug数量存入文件
	:param count_number_of_each_dev: key=the developer's name, value=the number of bug fixed by the developer
	:return: 
	'''
	with open('../data/little_developer_distribution.txt', 'w') as writer:
		for key, value in count_number_of_each_dev.items():
			writer.write('{}\t{}\n'.format(key, value))
	pass

def implement_train_and_eval_windows():
	'''
	
	:return: 之所以写成这样是为了兼容之前写的时间窗口代码,这样那部分代码就不用做太大的改动了,毕竟那块还要用
	'''
	train_little_bugs = _read_little_bug_id(PATH.path_little_train_bug_msg_all)
	eval_little_bugs = _read_little_bug_id(PATH.path_little_eval_bug_msg_all)
	np.random.shuffle(train_little_bugs)
	np.random.shuffle(eval_little_bugs)
	return [train_little_bugs, eval_little_bugs]

def implement_create_developers_list():
	'''
	兼容之前的主体代码，为开发者列表的获取提供适应于小型数据集的新的外部接口
	:return: 
	'''
	developer_list = ["_PAD", "_UNK"]
	with open('../data/little_developer_distribution.txt', 'r') as reader:
		for line in reader.readlines():
			developer_list.append(line.strip().split('\t')[0])
	return developer_list


if __name__ == '__main__':
	# bugid_each_developer = count_bug_fixed_by_each_developer()
	# extract_little_bugs_by_developers(bugid_each_developer)
	print(implement_create_developers_list())