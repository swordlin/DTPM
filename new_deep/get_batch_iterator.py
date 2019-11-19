import tensorflow as tf
import util.data_helper as data_helper

def get_dataset_iterator(single_window, epoch, round_id, vocabulary, developers_list, bug_msg_all,sequence_length , active_size, batch_size, shuffle=True):
	dataset = tf.data.Dataset.from_generator(
		generator=lambda: data_helper.dataset_generator(vocabulary, developers_list, bug_msg_all, single_window, sequence_length, active_size, round_id),
		output_types=(tf.int32, tf.int32, tf.int32, tf.int32, tf.int32),
		# 一维以上（不包括一维），如果不声明大小，需要置为None，如果是一维，不声明大小的话，置为[]即可
		output_shapes=(
		tf.TensorShape([sequence_length]), tf.TensorShape([]), tf.TensorShape([active_size]), tf.TensorShape([]), tf.TensorShape([])))

	if shuffle: # 如果是训练进程，传入shuffle=True，即确认洗牌；
		dataset = dataset.shuffle(buffer_size=1000, seed=1).batch(batch_size=batch_size).repeat(count=epoch)  # delete shuffle
	else:
		dataset = dataset.batch(batch_size=batch_size).repeat(count=epoch)  # delete shuffle
	iterator = dataset.make_one_shot_iterator()  # 创建数据集迭代器, 稍后需要初始化
	return iterator