import os
import time

import util.PATH as PATH
import util.data_helper as data_helper

'''
为每轮实验的训练集和测试集生成活跃度列表，并且存储起来。
训练集的活跃度获取规则比较简单，假如当前bug的id=1000，那就把上面那1000条bug全部遍历，找到修复时间在我们需要的这个区间里的bug，提取其开发者和修复时间，用来后续排序。
测试集的活跃度提取稍有不同，针对的是上一个训练集中的bug，找到在当前bug提交时间之前修复的所有bug，一直提够25条，暂时设置不管所谓的3个月内的问题。
'''

delta_time = 3 * 30 * 24 * 60 * 60  # 3个月的秒数, 作为时间戳的差值


def write_active_sequence_to_file(sequence, sign, bug_id):
	path = '{}{}/'.format(PATH.path_active_list, str(sign))
	if not os.path.exists(path):
		os.makedirs(path)
	with open(path + str(bug_id), 'w') as writer:
		for name in sequence:
			writer.write(name + '\n')


def get_developer_active_sequence(bug_msg_all, bugids, current_id, position, train=True, weaks=None):
	'''

	:param bug_msg_all: key=bugid, value=[developer, 提交时间，修复时间，产品，组件]
	:param current_id: 
	:return: 
	'''
	# 提取当前bug的creation_ts, product, component;
	# 从得到的creation_ts往前推指定时间段, 比如说前移3个月, 找到这期间所有同product&component的bug报告
	# 取出后25条bug的所有修复者, 组成一个开发者活动序列
	# 写入文件
	# print('2_start_time', time.time())
	current_bug = bug_msg_all[current_id]
	c_product = current_bug[3]
	c_component = current_bug[4]
	c_creation_ts = current_bug[1]
	# print(c_creation_ts)
	# 时间顺序上由远到近,也就是delta的时间戳由小到大
	results = []

	# for i in range(current_id, -1, -1):     # 从当前id往上，逆序一直检索到key=0
	for i in range(position, -1, -1):  # 从当前id往上，逆序一直检索到key=0
		# if i in bug_msg_all.keys():         # 检查0～current_id内存在的所有id
		value = bug_msg_all[bugids[i]]
		bugid = value[2]
		if value[3] == c_product and value[4] == c_component:
			results.append((bugid, value[0]))  # 提取开发者，需要注意的是，因为是逆序提取，所以离得最近的排在前面，因此最后需要逆转。
		if len(results) == 25:              # 最多只提取25条，事实上，我想知道为什么
			break
	results = sorted(results, key=lambda x: x[0])[-25:]  #

	# temp = filter(lambda item: item.value[5] == c_product and item.value[6] == c_component and item.value[4] >= (c_creation_ts-delta_time) and item.value[4] <= c_creation_ts, bug_msg_all.values)
	# 相当多的数据
	# temp = filter(lambda x: x[1][5] == c_product and x[1][6] == c_component  and x[1][4] >= (c_creation_ts-delta_time), sorted_bugs)
	# temp = list(temp)
	# print('2_send_time', time.time())
	# print(temp)
	# return list(reversed(results))[-25:]      # 出于上面的原因，需要逆转列表
	return [results[i][1] for i in range(len(results))]  # 需要单抽出开发者


def traverse_all_bugs(all_bugids, new_bugids, sign, train=True, weaks=None):
	'''

	:param all_bugids: 
	:param new_bugids: 
	:param sign: 用来标识是哪一轮实验，提取好的训练集和测试集活跃度数据会放在以sign明明的文件夹里。
	:param train: 
	:return: 
	'''
	start_time = time.time()
	print('start_time:', start_time)
	bugs_counts = {}
	bug_msg_all, _ = data_helper.get_msg_all()
	count = 0
	# all_bugids = sorted(bug_msg_all.keys())  # 升序排好
	# for id in bug_msg_all.keys():

	for id in new_bugids:
		print(count)
		count += 1
		if train:
			actives = get_developer_active_sequence(bug_msg_all, all_bugids, id, all_bugids.index(id) - 1, train, weaks)
		else:
			actives = get_developer_active_sequence(bug_msg_all, all_bugids, id, len(all_bugids) - 1, train, weaks)
		bugs_counts[id] = len(actives)
		write_active_sequence_to_file(actives, sign, id)

	print('计算消耗时间:', time.time() - start_time)
	with open('../data/active_counts_{}.txt'.format(time.time()), 'w') as writer:
		for key in bugs_counts.keys():
			writer.write(str(key) + '\t' + str(bugs_counts[key]) + '\n')
	print('最终结束时间:', time.time())


if __name__ == '__main__':

	# traverse_all_bugs()
	bug_msg_all, _ = data_helper.get_msg_all()
	# time_windows = data_helper.split_dataset_by_time_windows(bug_msg_all)
	time_windows = data_helper.split_dataset_by_eight_to_two(bug_msg_all)
	print(len(time_windows[0]))
	print(len(time_windows[1]))
	print(len(time_windows[2]))
	sign = 1.2
	for i in [0]:
		traverse_all_bugs(time_windows[i], time_windows[i], sign=sign, train=True)  # 针对训练集
		traverse_all_bugs(time_windows[i]+time_windows[i+2], time_windows[i + 1], sign=sign, train=False)  # 针对测试集
		traverse_all_bugs(time_windows[i], time_windows[i + 2], sign=sign, train=False)  # 针对验证集
