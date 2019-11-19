
import tensorflow as tf
import re
import numpy as np
import random
import datetime
import os
# from sklearn.feature_extraction.text import CountVectorizer
import util.PATH as PATH

'''
设想的需要的功能不少, 
1. get_msg_all, 然后将其按照时间(论文里说明是按照提交时间, 那不就是按照bug_id?)排序, 平均分成11份:
第一叠数据用于训练, 下一叠用来测试, 以此类推, 10叠训练10叠测试, 取10叠执行结果的平均值来作为最总的评估指标;
	1.1. 这里按照bug_id进行排序
2. 填充文本数据至等长
3. 提供词汇表
4. 不将数据一次性写入内存, 一个一个batch写入好了, 使用tf.data.Dataset.from_generator()来实现
5. 关于assign_to开发者的索引化或者one-hot编码
	5.1. 若当前开发者不存在于开发者列表developer文件中, 置当前开发者id为-1
'''

# Processing tokens, 在词汇表中添加辅助性的token(标记)
# <GO>用来分隔问题与回复
# Token<PAD>用来补齐问题或回复
'''原来是这样做的，0代表padding，即填充值；1代表没有出现过的单词或者开发者。
但是后来发现特意设置1，是不需要的，反而会额外造成准确率的虚高。所以如果这样的单词，不予处理。
如果出现这样的开发者，这是不可能的，因为使用的开发者列表包含了全部开发者，不需要考虑这种情况。'''
_PAD = b"_PAD"  # padding?, 用_PADK来填充数据至等长
# _GO = b"_GO"    # <start>
# _EOS = b"_EOS"  # <end>, end of sentence
_UNK = b"_UNK"      # 把没有出现过的词统计为unk(unknown token)
# _START_VOCAB = [_PAD, _GO, _EOS, _UNK]
# _START_VOCAB = [_PAD, _UNK]
_START_VOCAB = [_PAD]

PAD_ID = 0
# GO_ID = 1
# EOS_ID = 2
# UNK_ID = 1

# max_doc_len =400
# mozilla <=400 99.2%, <=500, 99.4%

# 读取所有的msg, 并按照id顺序升序排列
# 原顺序为: bug_id    assign_to   resolution  dup_id  creation_ts delta_ts    product component
# 读取时删除resolution和dup_id, 改为:
# key=bug_id, value= {assign_to, creation_ts, delta_ts, product,component}
def get_msg_all():
	'''
	key=bug_id, value= {assign_to, creation_ts, delta_ts, product, component}
	:return: 
	'''
	bug_msg_all = {}
	with open(PATH.path_bug_msg_all, 'r') as reader:
		for line in reader.readlines():
			temp = line.strip().split('\t')
			# 处理时间成时间戳格式, 原始格式为'1999-03-12 17:33 -0400'
			# 观察了下数据集, creation_ts的格式始终为: 1999-03-12 17:33 -0400
			# 而delta_ts的格式为: 1999-04-21 11:23:22 -0400'
			temp[2] = ' '.join(temp[2].split(' ')[:2])
			temp[2] = datetime.datetime.strptime(temp[2], '%Y-%m-%d %H:%M').timestamp()
			temp[3] = ' '.join(temp[3].split(' ')[:2])
			temp[3] = datetime.datetime.strptime(temp[3], '%Y-%m-%d %H:%M:%S').timestamp()
			bug_msg_all[int(temp[0])] = temp[1:]        # 这个int相当重要
	# 将字典按照bug_id排序出一个列表, 由远及近
	sorted_bugs = sorted(bug_msg_all.items(), key=lambda item: item[0])
	# 这里在后续调用的时候一般只接收前面一个位置，不接受后面，是因为原始文件中，所有的bug报告都是按照bugid排好的
	return bug_msg_all, sorted_bugs

#
def calculate_doc_max_len():
	'''
	计算语料库中单个文档单词的最大长度, 用于后续的padding
	这个最好先独立计算出来, 免得每次跑着费劲, 而且还要筛选
	:return: 
	'''
	names = os.listdir(PATH.path_corpus)
	lens = {}
	for name in names:
		file = PATH.path_corpus + name
		c_len = len(open(file, 'r').readlines())
		print(c_len)
		lens[name] = c_len
	return lens

# tf.data.Dataset.from_generator()中用到的生成器
# datas_ids: 是分割出来的一个窗口的索引
# 其实该方法就是以一个窗口的数据作为数据集, 进行yield, 事实上, 一个窗口也就1万条数据, 似乎没必要写成生成器.......
def dataset_generator(vocabulary, developers_list, bugs_msg_all, datas_ids, max_doc_len, active_size, window_id):
	'''
	顺手将单词和assignee索引化了
	:param vocabulary: 
	:param developers_list: 
	:param bugs_msg_all: 
	:param datas_ids: 
	:param active_size: 
	:param cost_matrix:
	:param window_id:
	:return: 
	'''
	index = 0
	while True:
		name = datas_ids[index]       # 取出index位置的bug_id, 作为文件名
		feature = []
		#
		with open(PATH.path_corpus + str(name), 'r') as reader:
			for line in reader.readlines():
				feature.append(line.strip())
		words_to_ids, feature_actual_length = data_padding_and_to_ids(feature, vocabulary, max_doc_len)
		label = bugs_msg_all[name][0]       # assign_to
		try:
			label_id = developers_list.index(label)
		except ValueError:
			continue                # 如果开发者不存在于开发者列表，进行下次循环。
		active_vec, active_actual_length = get_active_sequence_vec_by_bug_id(name, developers_list, active_size, window_id)
		yield words_to_ids, label_id, active_vec, feature_actual_length, active_actual_length
		index += 1
		if index == len(datas_ids):
			# index = 0  # 可以实现无限循环而不会发生outrangeError, 置repeat于不顾, 只能通过for循环的次数来满足epoch这个限制
			break           # 循环完一次break, 这样虽然保证了repeat方法还有用, 但是每个epoch的最后一批都不满足batch_size了....

# 原本想根据单词出现的文档数来构建索引, 但是麻烦
# 所以直接使用前一阶段的文件顺序保存了
def create_vocabulary():

	vocabulary = [] + _START_VOCAB      # 先将前面的辅助性标记放入vocabualry中
	with open(PATH.path_vocabulary, 'r') as reader:
	# with open('../data/windows/window_0_vocabulary.txt', 'r') as reader:
		for line in reader.readlines():
			vocabulary.append(line.strip())
	return vocabulary

# 针对一个doc的数据填充, 将单词转换成数字索引
def data_padding_and_to_ids(feature, vocabulary, max_doc_len):
	# 返回单词在词汇表中的value(or索引), 如果单词不在词汇表, 返回UNK_ID=3
	# ids = [word_vocabulary.get(word, UNK_ID) for word in sentence]
	ids = []
	for word in feature:
		try:
			ids.append(vocabulary.index(word))
		except ValueError:
			continue                # 如果真的出现词汇表中没有的单词，忽略
	if len(ids) > max_doc_len:      # 长于max_doc_len的步长都切掉
		ids = ids[:max_doc_len]
	'''注意这里在ids之前填充0, 这样做来避免太长的0影响本身单词的记忆效果'''
	words_as_ids = ids+[PAD_ID] * (max_doc_len - len(ids))      # 用PAD_ID来填充数据
	return words_as_ids, len(ids)

# 将数据集按照时间窗口分割成11份
# 直观点说, 就是把bug_id集合按顺序分成等量的11份
def split_dataset_by_time_windows(bug_msg_all):
	bug_ids = sorted(bug_msg_all.keys())       # 升序排列
	# 将bug_ids分成等量的11份
	delta = int(len(bug_ids) / 11)
	# 最后一位是步长
	return [bug_ids[i:i+delta] for i in range(0, len(bug_ids), delta)]

def create_developers_list():
	developers_list = [] + _START_VOCAB     # 开发者也需要包含这两个, 0代表没有开发者带来的填充, 1代表历史记录中不存在的开发者
	with open(PATH.path_developer, 'r') as reader:
	# with open('../data/windows/developer_window_0.txt', 'r') as reader:
		for line in reader.readlines():
			developers_list.append(line.strip())
	return developers_list

# 根据当前bug_id找到相应的开发者活跃序列文件读取
# 这里我在预处理阶段, 就把每个bug对应的开发者活跃序列提取保存成了文件
# 相较于程序运行过程中处理, 前者自然更节省运行时间, 但是无疑灵活度降低, 而且前者模块分的太多, 使得项目臃肿了很多, 一饮一啄吧...
def get_active_sequence_vec_by_bug_id(current_id, developers_list, active_size, window_id):
	'''
	:param current_id:
	:param developers_list:
	:param active_size:
	:param window_id:当前10轮增量实验处于的轮数，每个轮数对应了不同的活跃度文件夹，为了方便起见实现预处理好了都。
	:return: 返回填充好的列表和填充之前的实际长度
	'''
	active_vec = []         # 替换成数字索引得到的vector向量
	with open(PATH.path_active_list + str(window_id) + '/' + str(current_id), 'r') as reader:
		for line in reader.readlines():
			w = line.strip()
			try:
				active_vec.append(developers_list.index(w))     # 将开发者替换成在developers_list中的索引
			except ValueError:
				continue            # 如果开发者不存在，忽略
	padding_active_vec =  active_vec + [PAD_ID]*(active_size - len(active_vec))
	return padding_active_vec, len(padding_active_vec)
	# return padding_active_vec, len(active_vec)

def extract_small_balance_val_set(bug_ids, bug_msg_all):
	'''
	从bug_ids代表的原始训练集中，抽取一个小型的平衡数据集，所谓的平衡数据集，是指包含所有开发者，
	且每个开发者有一定数量的样本，比如说5个？（这意味着原始训练集必须是预训练好的，最小的开发者修复bug数量=10）
	稍做一些思维发散，其实我觉得测试集也应该处理成数据平衡。原先受数据制约太严重了。
	麻烦的是，重新预处理完成后，我需要重新记录vocabulary和developers列表，还有分别对应的活跃度列表。
	这标志着我可能需要重新整合之前的代码。。。。。。
	然后返回被切割完的新训练集和小型平衡数据集。
	:param bug_ids: 
	:param bug_msg_all: key=bug_id, value= {assign_to   creation_ts delta_ts    product component}
	:return: 
	'''
	fixed_by_developers = {}
	for i in range(len(bug_ids)):       # 统计每个开发者修复的所有bug的id
		devr = bug_msg_all[bug_ids[i]][0]
		if devr not in fixed_by_developers.keys():
			fixed_by_developers[devr] = []
		fixed_by_developers[devr].append(bug_ids[i])
	# for key in fixed_by_developers.keys():
	# 	print(len(fixed_by_developers[key]))
	# 统计每个开发者修复的数量，删除修复数量少于10的开发者及他们修复的bug
	developers = [] + _START_VOCAB     # 保存预处理以后的开发者列表
	trainset_ids = []   # 保存预处理+抽取之后的训练集ids
	valset_ids = []     # 验证集，或者说是每个类别样本=5的平衡验证集
	for devr, ids in fixed_by_developers.items():
		if len(ids) >= 10:
			developers.append(devr)
			trainset_ids += ids[0:-10]
			valset_ids += ids[-10:]     # 后5个抽取作为验证集，其余为训练集
		else:
			developers.append(devr)
			trainset_ids += ids
			valset_ids += ids

	# 保存开发者列表、训练集ids、测试集ids到文件
	# 处理词汇表,暂不处理，使用全部的词汇表
	#

	# 处理活跃度列表
	return developers, trainset_ids, valset_ids

# def split_dataset_by_eight_to_two(bug_msg_all):
# 	'''
# 	按照8:2的比例划分训练集和测试集。
# 	2018-12-19 18:59:22 修改，按照6:2:2划分训练集、验证集和测试集
# 	:param bug_msg_all:
# 	:return:
# 	'''
# 	bug_ids = sorted(bug_msg_all.keys())  # 升序排列
# 	# delta = int(len(bug_ids) * 0.8)     # 取整个数据集的前80%作为训练集
# 	delta = int(len(bug_ids)/10)
# 	train_set = bug_ids[:int(delta*6)]           # 按照7:1:1划分
# 	val_set = bug_ids[int(delta*6) : delta*8]
# 	eval_set = bug_ids[delta*8:]
#
# 	train_set = train_set[(len(train_set) % 32):]      # 删掉多余的，即不满足batch_size的
# 	val_set = val_set[(len(val_set) % 32):]
# 	eval_set = eval_set[(len(eval_set) % 32):]
# 	return [train_set, eval_set, val_set]       # 注意顺序，是训练集、测试集、验证集排列

def split_dataset_by_eight_to_two(bug_msg_all):
	'''
	按照8:2的比例划分训练集和测试集。
	:param bug_msg_all:
	:return:
	'''
	bug_ids = sorted(bug_msg_all.keys())  # 升序排列
	delta = int(len(bug_ids) * 0.8)     # 取整个数据集的前80%作为训练集
	train_set = bug_ids[:delta]           # 划分
	val_set = bug_ids[delta:]
	eval_set = bug_ids[delta:]

	train_set = train_set[(len(train_set) % 32):]      # 删掉多余的，即不满足batch_size的
	eval_set = eval_set[(len(eval_set) % 32):]
	val_set = val_set[(len(val_set) % 32):]
	return [train_set, eval_set, val_set]       # 注意顺序，是训练集、测试集

def __split_dataset_by_category(bug_msg_all):
	'''
	按照类别划分，每个类别取80%做训练集，20%做测试集。
	:param bug_msg_all: 
	:return: 
	'''
	fixed_by_developers = {}
	# for i in range(len(bug_msg_all)):  # 统计每个开发者修复的所有bug的id
	for bugid, value in bug_msg_all.items():
		devr = value[0]
		if devr not in fixed_by_developers.keys():
			fixed_by_developers[devr] = []
		fixed_by_developers[devr].append(bugid)
	train_ids = []
	eval_ids = []
	for devr, value in fixed_by_developers.items():
		# value = sorted(value)       # 将每类按照bugid排序，就算没有这句，value其实也是具有顺序的。dict虽然是无序的，但是它每次无序都一样
		delta = int(len(value)*0.8)
		train_ids += value[:delta]
		eval_ids += value[delta:]
	# 按bugid分别排序训练集和测试集，这样做有个好处，就是按照固定形式将类别数据打散
	return [sorted(train_ids), sorted(eval_ids)]



if __name__ == '__main__':
	# pass
	# dataset_generator()
	# print(calculate_doc_max_len())
	# lens = calculate_doc_max_len()
	# with open('../data/lens_Eclipse.txt', 'w') as writer:
	# 	for key in lens.keys():
	# 		writer.write(key + '\t' + str(lens[key]) +'\n')
	bug_msg_all, sorted_bugs = get_msg_all()
	train, eval, val = split_dataset_by_eight_to_two(bug_msg_all)
	print(len(train))
	print(len(eval))
	print(len(val))