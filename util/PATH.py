import os
# 项目用到的各文件路径
# root = '/document/Bug_msg/Mozilla/'
root = '/document/Bug_msg/PartFive/GCC/'#'/document/Bug_msg/Eclipse_modified_50/'
path_bug_msg_all = root + 'bug_msg_all'
# path_bug_msg_all = root + 'commenter_bug_msg_all'

path_origin_corpus = root + '{}_total.csv'.format(root.split('/')[-2])#'keywords_stemed/'      # 没有经过词干化处理的原始语料库
path_corpus = root + 'corpus_modified/'
if not os.path.exists(path_corpus):
	os.makedirs(path_corpus)
# path_orign_corpus = root + 'keywords_stemed/'
# path_vocabulary = root + 'vocabulary.txt'
path_vocabulary = root + 'modified_vocabulary.txt'
path_developer = root + 'developer'
# path_developer = root + 'commenter_developer'
# path_active_list = root + 'active_list_3m/'
path_active_list = root + 'round_active_list_3m/'
if not os.path.exists(path_active_list):
	os.makedirs(path_active_list)
# path_sequence_len_path = root + 'active_sequence_java.txt'

path_active_list_small = root + 'small/'

path_little_train_bug_msg_all = '../data/little_train_bugs_msg_id.txt'
path_little_eval_bug_msg_all = '../data/little_eval_bugs_msg_id.txt'

a = '119'