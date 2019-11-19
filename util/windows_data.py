import numpy as np
import util.PATH as PATH
from util.data_helper import get_msg_all
from util.data_helper import split_dataset_by_time_windows

'''
提取窗口0的词汇表信息
'''
# 字典，key=bugid，value=
bug_msg_all, _ = get_msg_all()
windows = split_dataset_by_time_windows(bug_msg_all)

vocabulary = []
for i in range(len(windows[0])):
	print(i)
	with open(PATH.path_corpus + str(windows[0][i]), 'r') as reader:
		for line in reader.readlines():
			if line.strip() not in vocabulary:
				vocabulary.append(line.strip())

with open('../data/windows/window_0_vocabulary.txt', 'w') as writer:
	for word in vocabulary:
		writer.write('{}\n'.format(word))
