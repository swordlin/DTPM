import tensorflow as tf

import util.data_helper as data_helper

import numpy as np
from new_deep.DeepModel import DeepModel
from new_deep.get_batch_iterator import get_dataset_iterator

# import DE.manually_specified as manually_specified

import os
import time
import datetime

# Flags = tf.flags.FLAGS        # 这里其实可以使用Flags接收命令行传递参数

def main(round_id, cost_matrix=None):
	hidden_size = 16
	sequence_length = 400     # 句子最大长度, 这个稍后要从date_helper传进来, 或者自定义
	embedding_size = 16      # word向量的长度
	epoch = 1
	layer_num = 1
	dropout_keep_prob = 0.5
	log_device_placement = False
	batch_size = 8
	learning_rate = 0.001
	active_size = 25

	bug_msg_all, _ = data_helper.get_msg_all()
	vocabulary = data_helper.create_vocabulary()
	developers_list = data_helper.create_developers_list()
	# time_windows = data_helper.split_dataset_by_time_windows(bug_msg_all) # 这行是正常的按照全数据集划分的11个时间窗口
	time_windows = data_helper.split_dataset_by_eight_to_two(bug_msg_all)

	num_classes = len(developers_list)
	vocabulary_size = len(vocabulary)
	developers_size = len(developers_list)
	print(developers_size)

	# 把配置参数写入文件作为记录
	def write_configuration_info_to_file(root_dir):
		with open(os.path.join(root_dir, 'configuration.txt'), 'w') as writer:
			writer.write('hidden_size = {}\n'.format(hidden_size))
			writer.write('embedding_size = {}\n'.format(embedding_size))
			writer.write('epoch = {}\n'.format(epoch))
			writer.write('layer_num = {}\n'.format(layer_num))
			writer.write('batch_size = {}\n'.format(batch_size))
			writer.write('learning_rate = {}\n'.format(learning_rate))

	# 显式的创建会话和图保证资源在不需要的时候合理释放
	with tf.Graph().as_default():
	# with tf.device('/cpu:0'):
		session_conf = tf.ConfigProto(
			log_device_placement = log_device_placement,    # 在指定设备上存储日志文件, 辅助debug
		)
		session_conf.gpu_options.allow_growth=True
		# session_conf.gpu_options.per_process_gpu_memory_fraction=0.4
		sess = tf.Session(config=session_conf)

		with sess.as_default():
			deepModel = DeepModel(sequence_length = sequence_length,
			              num_classes = num_classes,
			              vocabulary_size = vocabulary_size,
			              developers_size = developers_size,
			              embedding_size=embedding_size,
			              hidden_size=hidden_size,
			              active_size=active_size,
			              layer_num=layer_num,
			              cost_matrix=cost_matrix,
			              batch_size=batch_size)
		global_step = tf.Variable(0, name="global_step", trainable=False)
		# 优化损失函数放在外面
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

		from tensorflow.python.ops import clip_ops
		variables = tf.trainable_variables()
		# variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cost')
		# variables = tf.get_variable('cost/cost_matrix:0')
		# Compute gradients.
		gradients = optimizer.compute_gradients(loss=deepModel.loss, var_list=[deepModel.weights])
		capped_gvs = [(tf.clip_by_value(grad, clip_value_min=-5, clip_value_max=5), var) for grad, var in gradients]      # gradient clip, clip_value_min not >= 0
		# capped_gvs = tf.clip_by_global_norm(tf.gradients(deepModel.loss, variables), clip_norm=5)      # gradient clip, clip_value_min not >= 0
		grad_sum=tf.summary.scalar("global_norm/gradient_norm", clip_ops.global_norm(list(zip(*capped_gvs))[0]))
		# train_op = optimizer.minimize(deepModel.loss, global_step=global_step)
		train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)       #
		# print(train_op)

		# 计算最终的top_k正确率
		top_1_acc= deepModel.metrics['top_1']
		top_5_acc= deepModel.metrics['top_5']

		# abspath: 返回绝对路径
		timestamp = str(int(time.time()))       # 将当前时间戳作为目录name的一部分, 防止重写
		model_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
		print('writing to {}'.format(model_dir))

		# summaries for loss and accuracy
		loss_summary = tf.summary.scalar('loss', deepModel.loss)
		top_1_summary = tf.summary.scalar('top_1', top_1_acc)
		top_5_summary = tf.summary.scalar('top_5', top_5_acc)
		l2_lost_summary = tf.summary.scalar('l2_loss', deepModel.l2_loss)
		# train summaries
		# train_summary_op = tf.summary.merge([loss_summary, top_1_summary, top_5_summary,grad_sum, l2_lost_summary])
		train_summary_op = tf.summary.merge([loss_summary, top_1_summary, top_5_summary, grad_sum, l2_lost_summary])
		train_summary_dir = os.path.join(model_dir, 'summaries', 'train')
		train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
		# eval summaries
		eval_summary_op = tf.summary.merge([loss_summary, top_1_summary, top_5_summary, grad_sum, l2_lost_summary])
		eval_summary_dir = os.path.join(model_dir, 'summaries', 'eval')
		eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, sess.graph)
		# # 通过checkpoints(检查点)来存储模型参数
		checkpoint_dir = os.path.abspath(os.path.join(model_dir, 'checkpoints'))
		checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
		if not os.path.exists(checkpoint_dir):
			os.makedirs(checkpoint_dir)
		# saver = tf.train.Saver(max_to_keep=10)
		saver = tf.train.Saver(max_to_keep=10, var_list=[var for var in tf.trainable_variables() if var not in [deepModel.weights]])
		write_configuration_info_to_file(model_dir)  # 将配置文件写入文件
		# 初始化全部变量
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

		# root = '/home/wanglinhui/PycharmProjects/2LSTM/new_deep/runs/1545044550/checkpoints'
		# saver = tf.train.import_meta_graph(root + '/model-0.meta')
		# saver.restore(sess, root + '/model-0')

		'''单次训练步'''
		def train_step(model, batch_feature, batch_label, batch_active_feature, batch_sequence_lengths, batch_active_actual_lengths):
			feed_dict = {
				model.b_features: batch_feature,
				model.b_labels: batch_label,
				model.b_active_features: batch_active_feature,
				model.b_sequence_lengths: batch_sequence_lengths,
				model.b_active_actual_lengths: batch_active_actual_lengths,
				model.place_dropout_keep_prob: dropout_keep_prob,
			}
			_, step , summaries, metrics, loss = sess.run([train_op, global_step, train_summary_op, model.metrics, model.loss], feed_dict=feed_dict)
			time_str = datetime.datetime.now().isoformat()
			if step % 20 == 0:
				print('train_time: {0}, top1: {1:.3f}, top5: {2:.3f}, loss: {3}'.format(time_str, metrics['top_1'], metrics['top_5'], loss))
			train_summary_writer.add_summary(summaries, step)     # 写入tensorboard
		'''单次检验步'''
		def eval_step(model, batch_feature, batch_label, batch_active_feature, batch_sequence_lengths, batch_active_actual_lengths):
			feed_dict = {
				model.b_features: batch_feature,
				model.b_labels: batch_label,
				model.b_active_features: batch_active_feature,
				model.b_sequence_lengths: batch_sequence_lengths,
				model.b_active_actual_lengths: batch_active_actual_lengths,
				model.place_dropout_keep_prob: 1.0,
			}
			step, summaries, metrics, loss, top_5_indices = sess.run([global_step, eval_summary_op, model.metrics, model.loss, model.top_5_indices], feed_dict=feed_dict)
			time_str = datetime.datetime.now().isoformat()
			# if step % 5 == 0:
			# print('eval_time: {0}, top1: {1:.3f}, top5: {2:.3f}, loss: {3}'.format(time_str, metrics['top_1'], metrics['top_5'], loss))
			eval_summary_writer.add_summary(summaries, step)  # 写入tensorboard
			return top_5_indices,metrics

		def get_val_or_eval_top_acc(bug_ids):
			acc_list = {'top1': [], 'top5': []}
			iterator = get_dataset_iterator(bug_ids, 1, round_id, vocabulary, developers_list, bug_msg_all,
			                                    sequence_length, active_size, batch_size,
			                                    shuffle=False)  # 测试的话, 走一个epoch就可以了其实....
			batch = iterator.get_next()  # 新批次数据
			for _ in range(int(len(bug_ids) / batch_size)):
				batch_features, batch_labels, batch_active_features, batch_sequence_lengths, batch_active_actual_lengths = sess.run(batch)
				if batch_labels.shape[0] != batch_size:
					print('skip')
					continue
				_, metrics_1 = eval_step(deepModel, batch_features, batch_labels, batch_active_features, batch_sequence_lengths, batch_active_actual_lengths)
				acc_list['top1'].append(metrics_1['top_1'])
				acc_list['top5'].append(metrics_1['top_5'])
			top1 = sum(acc_list['top1']) / len(acc_list['top1'])
			top5 = sum(acc_list['top5']) / len(acc_list['top5'])
			return top1, top5

		for i in [0]:
			train_iterator = get_dataset_iterator(time_windows[i], epoch, round_id, vocabulary, developers_list, bug_msg_all, sequence_length, active_size, batch_size)
			batch = train_iterator.get_next()
			epoch_length = len(time_windows[0]) / batch_size
			print('epoch_length=', epoch_length)
			for step in range(10000000):
				try:
					current_step = tf.train.global_step(sess, global_step)
					batch_features, batch_labels, batch_active_features, batch_sequence_lengths, batch_active_actual_lengths= sess.run(batch)
					if batch_labels.shape[0] != batch_size:
						print('skip')
						continue
					train_step(deepModel, batch_features, batch_labels, batch_active_features, batch_sequence_lengths, batch_active_actual_lengths)
					if (step+1) % epoch_length == 0:        # 每个epoch进行一次测试步

						eval_top1, eval_top5 = get_val_or_eval_top_acc(time_windows[i+1])
						val_top1, val_top5 = get_val_or_eval_top_acc(time_windows[i+2])

						print('验证步：{:.4f}\t{:.4f}, 测试步：{:.4f}\t{:.4f}'.format(val_top1, val_top5, eval_top1, eval_top5))

						with open(os.path.join(model_dir, 'configuration.txt'), 'a') as writer:
							writer.write('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(val_top1, val_top5, eval_top1, eval_top5))
						saver.save(sess, checkpoint_prefix, global_step=step)
						print('model save success!')
				except tf.errors.OutOfRangeError:
					print('已完成所有epoch迭代')
					break
			#
			saver.save(sess, checkpoint_prefix, global_step=i)
			print("下面是第{}个窗口的验证:".format(i+1))
			# # 第i+1叠数据用来测试
			eval_iterator = get_dataset_iterator(time_windows[i + 1], 1, round_id, vocabulary, developers_list, bug_msg_all, sequence_length, active_size, batch_size, shuffle=False)  # 测试的话, 走一个epoch就可以了其实....
			batch_eval = eval_iterator.get_next()  # 新批次数据

			acc_list = {'top1': [], 'top5': []}
			num_list = {'top1': [], 'top5': []}

			for _ in range(int(len(time_windows[i+1])/batch_size)):  # 走20个batch就够了
				try:
					current_step = tf.train.global_step(sess, global_step) # 将测试的step也加入全局step
					batch_features, batch_labels, batch_active_features, batch_sequence_lengths, batch_active_actual_lengths = sess.run(batch_eval)
					if batch_labels.shape[0] != batch_size:
						print('skip')
						continue
					_, metrics_1 = eval_step(deepModel, batch_features, batch_labels, batch_active_features, batch_sequence_lengths, batch_active_actual_lengths)
					acc_list['top1'].append(metrics_1['top_1'])
					acc_list['top5'].append(metrics_1['top_5'])

					num_list['top1'].append(metrics_1['top_1'] * batch_size)
					num_list['top5'].append(metrics_1['top_5'] * batch_size)

				except tf.errors.OutOfRangeError:
					print('eval finish!')
					print('{}\t{}'.format(sum(num_list['top1']) / (len(num_list['top1'])*batch_size),
					                      sum(num_list['top5']) / (len(num_list['top1'])*batch_size)))
					break
			print('{}\t{}'.format(sum(acc_list['top1']) / (len(acc_list['top1'])),
			                      sum(acc_list['top5']) / (len(acc_list['top1']))))
if __name__ == '__main__':
	# time.sleep(60*60*3)
	# time.sleep(20)
	# developers = data_helper.create_developers_list()
	# cost_matrix = manually_specified.implement_random_cost_matrix(developer_size=len(developers))
	# cost_matrix = manually_specified.get_cost_matrix_from_data_distribution()
	# main(cost_matrix, round_id=0)
	main(round_id=0)