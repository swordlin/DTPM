import numpy as np
import nltk

import util.data_helper as data_helper
import util.PATH as PATH
'''
预处理全部数据集，提取开发者列表和词汇表，
删掉修复数量少于50的开发者及由他们修复的bug，
删除超过50%文档中出现过的单词，少于10次的单词。
'''

def extract_vocabulary(bug_msg_all, file):
	'''
	对经过一次处理之后的语料库，进行词干化处理。
	提取词干化之后的新vocabualry，并且将所有文档中的单词词干化之后重新保存。
	之所以不在第一次处理的Java代码中加入本模块，是因为Java词干化写起来很不舒服，懒得搞。宁愿分二次处理。
	:param bug_msg_all: 
	:param file: 
	:return: 
	'''
	vocabualry = set()      # 记录词汇表

	# porter = nltk.stem.SnowballStemmer('english')       # 词干化
	porter = nltk.stem.PorterStemmer('NLTK_EXTENSIONS')       # 词干化
	# 统计所有单词
	for name in bug_msg_all.keys():
		print(name)         # 辅助输出
		words = []
		with open(PATH.path_origin_corpus + str(name), 'r') as reader:
			for line in reader.readlines():
				# word = line.strip()
				word = porter.stem(line.strip())    # 词干化每个单词
				words.append(word)                  # 记录每个被词干化之后的单词
				vocabualry.add(word)
		# 将词干化之后的文档保存
		with open(PATH.path_corpus + str(name), 'w') as writer:
			for word in words:
				writer.write('{}\n'.format(word))
	print('len=', len(vocabualry))

	with open(file, 'w') as writer:     # 保存词汇表至指定文件
		for word in vocabualry:
			writer.write('{}\n'.format(word))

def del_invalid_developers(bug_msg_all):
	'''
	主要是用来检查数据集中是否还存在无效开发者
	:param bug_msg_all: 
	:return: 
	'''
	invalids = ["nobody", "inbox", "webmaster","platform", "unassigned", "issues", "needsconfirm","swneedsconfirm"]
	invalid_bugs = []
	invalid_devs = set()
	for key, value in bug_msg_all.items():
		if True in list(map(lambda x :x in value[0].lower(), invalids)):        # 该条bug是由无效开发者修复的
			invalid_bugs.append(key)
			invalid_devs.add(value[0])
	print('无效的bug条数={}'.format(len(invalid_bugs)))
	print('无效的开发者数目={}'.format(len(invalid_devs)))
	return invalid_bugs

if __name__ == '__main__':
	bug_msg_all, _ = data_helper.get_msg_all()      # # key=bug_id   value= {assign_to   creation_ts delta_ts    product component}
	# extract_vocabulary(bug_msg_all, PATH.path_vocabulary)
	del_invalid_developers(bug_msg_all)

