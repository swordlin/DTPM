import tensorflow as tf


class DeepModel(object):
	def __init__(self, sequence_length, num_classes, vocabulary_size, developers_size, embedding_size, hidden_size, active_size, layer_num, cost_matrix, batch_size):
		self.place_dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
		self.b_features = tf.placeholder(dtype=tf.int32, shape=[None, sequence_length], name='place_features')
		self.b_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='place_labels')
		# 活跃度提取部分的features
		self.b_active_features = tf.placeholder(dtype=tf.int32, shape=[None, active_size], name='batch_active_features')
		self.b_sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='place_sequence_lengths')
		self.b_active_actual_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='place_active_actual_lengths')
		# self.cost_matrix = tf.constant(cost_matrix, dtype=tf.float32, name='cost_matrix')
		self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
		# print(self.cost_matrix)

		with tf.name_scope('embedding'):
			embedding = tf.get_variable('embedding', [vocabulary_size, embedding_size], dtype=tf.float32)
			self.inputs = tf.nn.embedding_lookup(embedding, self.b_features)
			print('这是input的shape: {}'.format(self.inputs.shape))  # (?, 400, 19704)

		def get_a_lstm_cell():
			cell = tf.nn.rnn_cell.GRUCell(hidden_size)
			cellD = tf.nn.rnn_cell.DropoutWrapper(cell=cell, input_keep_prob=self.place_dropout_keep_prob)
			return cellD
			# return cell

		with tf.name_scope('rnn_active'):
			inputs_active = tf.one_hot(self.b_active_features, depth=developers_size, dtype=tf.float32)
			print(inputs_active)
			cells = tf.nn.rnn_cell.MultiRNNCell([get_a_lstm_cell() for _ in range(layer_num)],
			                                            state_is_tuple=True)
			active_outputs, active_states = tf.nn.dynamic_rnn(cells, inputs=inputs_active, dtype=tf.float32, sequence_length=self.b_active_actual_lengths)

			print('active_outputs:{}'.format(active_outputs))  # (?, 25, hidden_size)

		with tf.name_scope('bi_rnn'):
			# todo: 替换成# lstm_rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
			lstm_fw_cells = tf.nn.rnn_cell.MultiRNNCell([get_a_lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
			lstm_bw_cells = tf.nn.rnn_cell.MultiRNNCell([get_a_lstm_cell() for _ in range(layer_num)], state_is_tuple=True)
			outputs, _, = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cells, lstm_bw_cells, self.inputs, sequence_length=self.b_sequence_lengths, dtype=tf.float32)
			print('outputs: {}'.format(outputs))
			fw_output = tf.expand_dims(outputs[0], axis=-1)
			bw_output = tf.expand_dims(outputs[1], axis=-1)
			# todo: 考虑ksize=[1,1,hidden_size,1]
			fw_output = tf.nn.max_pool(fw_output, ksize=[1, sequence_length, 1, 1], strides=[1,1,1,1], padding='VALID')
			bw_output = tf.nn.max_pool(bw_output, ksize=[1, sequence_length, 1, 1], strides=[1,1,1,1], padding='VALID')

			print('fw_output: {}'.format(fw_output))
			print('bw_output: {}'.format(bw_output))

			bi_outputs = tf.reshape(tf.concat([tf.reshape(fw_output, shape=(-1, hidden_size)), tf.reshape(bw_output, shape=(-1, hidden_size))], 1), [-1, 2*hidden_size]) # 拼接前向输出和后向输出
			# 将前向输出与后向输出相加
			# bi_outputs = tf.reshape(tf.add(fw_output, bw_output), [-1, hidden_size], name='bi_fusion_outputs')
			print('bi_outputs: {}'.format(bi_outputs))

		with tf.name_scope('transform'):# transform bi_outputs to the size: hidden_size, in order to multiply active outputs
			transform_bi_outputs = tf.layers.dense(inputs=bi_outputs, units=hidden_size, activation=tf.nn.relu)

		with tf.name_scope('fusion'):		# 特征融合, 其实就是两个RNN的输出进行融合, 最终得到的融合特征送到输入层
			active_outputs = active_outputs[:, -1, :]
			active_outputs = tf.layers.dense(active_outputs, units=hidden_size, activation=tf.nn.sigmoid)
			# 将两个神经网络的高层特征通过元素间相乘予以融合
			fusion_outputs = tf.multiply(transform_bi_outputs, active_outputs, name='fusion_outputs')  # [batch_size, hidden_size]
			# fusion_outputs = tf.add(bi_outputs, active_outputs, name='fusion_outputs')  # [batch_size, hidden_size]
			# fusion_outputs = tf.concat([bi_outputs, active_outputs], axis=1, name='fusion_outputs')  # [batch_size, hidden_size]
			# fusion_outputs = transform_bi_outputs
			print('fusion_outputs={}'.format(fusion_outputs))

		with tf.name_scope('dropout'):
			l_dropout = tf.layers.dropout(fusion_outputs, self.place_dropout_keep_prob)

		with tf.name_scope('output'):

			w_softmax = tf.get_variable('w_softmax', shape=[hidden_size, num_classes])
			b_softmax = tf.get_variable('b_softmax', shape=[num_classes])
			logits = tf.matmul(l_dropout, w_softmax) + b_softmax
			print('logits.shape()={}'.format(logits.shape))
			loss_origin = tf.losses.sparse_softmax_cross_entropy(labels=self.b_labels, logits=logits)

		with tf.name_scope('cost'):
			self.weights = tf.Variable(tf.random_normal([developers_size, developers_size]), name='cost_matrix')
			print(self.weights)
			# biases = tf.Variable(tf.zeros([1, units]) + 0.1)
			y = tf.matmul(logits, self.weights)  # 每行对应点乘，之后每列求和，化为[1, K]

		with tf.name_scope('loss'):
			self.l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])          # 加入全连接
			# y = tf.matmul(logits, 1-self.cost_matrix)
			print(y)
			# y = logits
			# y = tf.nn.softmax(logits)
			# print('y={}'.format(y))
			# y = tf.matmul(y, 1-self.cost_matrix)
			# y = y/tf.reduce_sum(y, axis=1, keep_dims=True)          # 对惩罚加和之后的概率归一化,方便后面计算loss
			# one_labels = tf.one_hot(self.b_labels, depth=developers_size)       # 在计算交叉熵之前应该one-hot化
			# losses = (-tf.reduce_sum(tf.multiply(tf.to_float(one_labels), tf.log(y)), axis=1))
			# print(losses)
			losses = tf.losses.sparse_softmax_cross_entropy(labels=self.b_labels, logits=y)
			self.l2_loss = 0.01 * self.l2_loss
			# self.loss = tf.reduce_mean(losses) +  self.l2_loss + tf.reduce_mean(loss_origin)
			self.loss = tf.reduce_mean(losses) +  self.l2_loss
			print(self.loss)

		def caculate_topK(indices, k):
			print(self.b_labels)
			a = indices - tf.reshape(self.b_labels, (batch_size, 1))
			b = tf.equal(a, tf.zeros(shape=(batch_size, k), dtype=tf.int32))
			return tf.reduce_mean(tf.reduce_sum(tf.cast(b, tf.float32), axis=1), name='top_{}'.format(k))


		with tf.name_scope('accuracy'):
			_, self.top_1_indices = tf.nn.top_k(y, k=1, name='top_1_indices')
			_, self.top_5_indices = tf.nn.top_k(y, k=5, name='top_5_indices')

			self.acc_top_1 = caculate_topK(self.top_1_indices, 1)
			self.acc_top_5 = caculate_topK(self.top_5_indices, 5)
			print(self.acc_top_1)
			print(self.acc_top_5)
			# bool_top_1 = tf.nn.in_top_k(predictions=y, targets=self.b_labels, k=1, name='bool_top_1')
			# bool_top_5 = tf.nn.in_top_k(predictions=y, targets=self.b_labels, k=5, name='bool_top_5')
			#
			# #这里是为了检查都是哪些样本预测出错，所以获取了top5的索引
			# _, self.top_5_indices = tf.nn.top_k(y, k=5)
			# self.acc_top_1 = tf.reduce_mean(tf.cast(bool_top_1, dtype=tf.float32), name="acc_top_1")
			# self.acc_top_5 = tf.reduce_mean(tf.cast(bool_top_5, dtype=tf.float32), name="acc_top_5")
			# print(self.acc_top_1)
			# _, top_1_indices = tf.nn.top_k(y, k=1)
			# _, top_5_indices = tf.nn.top_k(y, k=5)
			#
			# _, top_1 = tf.metrics.precision_at_top_k(self.b_labels, top_1_indices, k=1)
			# _, top_5 = tf.metrics.precision_at_top_k(self.b_labels, top_5_indices, k=5)
			# self.acc_top_1 = top_1 * 1
			# self.acc_top_5 = top_5 * 5

			self.metrics = {
				'top_1': self.acc_top_1,
				'top_5': self.acc_top_5
			}
