import numpy as np


def sigmoid(a, b, x):
	'''
	定义Sigmoid函数: g(z) = 1/(1+e^-(ax+b))
	'''
	return 1.0/ (1 + np.exp(-1.0 * (x.dot(a) + b)))

def ELM(X, T, D, L):
	np.random.seed(1)
	a = np.random.normal(0, 1, (D, L))
	b = np.random.normal(0, 1)
	# 使用特征映射求解输出矩阵
	H = sigmoid(a, b, X)
	# 计算输出权重和输出函数
	HH = H.T.dot(H)
	HT = H.T.dot(T)
	beta = np.linalg.pinv(HH).dot(HT)
	Fl = H.dot(beta)
	return beta, Fl

def input_2_hidden(D, L):
	np.random.seed(1)
	W = np.random.normal(0, 1, (D, L))
	b = np.random.normal(0, 1)
	return W, b

def calculate_beta(H, T):
	# 计算输出权重和输出函数
	HH = H.T.dot(H)
	HT = H.T.dot(T)
	beta = np.linalg.pinv(HH).dot(HT)
	return beta

def caculate_top(labels, predicts):
	top1_num = 0
	top5_num = 0
	predicts = np.array(predicts)
	for i in range(len(labels)):
		# print(i)
		top1 = np.argmax(predicts[i])
		top5 = np.argsort(predicts[i])[-5:]
		if labels[i] == top1:
			top1_num += 1
		# print('label={}, top5={}'.format(labels[i], ' '.join(list(map(str, top5)))))
		if labels[i] in top5:
			top5_num += 1
	# print('top1_num={}, top5_num={}'.format(top1_num, top5_num))
	return top1_num/len(labels), top5_num / len(labels)

def get_probs_and_labels(root_dir, timestamp, round_id):
	datas = []
	labels = []
	with open('{}/{}/checkpoints/origin_prob_{}.txt'.format(root_dir, timestamp, round_id,), 'r') as reader:
		for line in reader.readlines():
			temp = line.strip().split(',')
			datas.append(list(map(float, temp[0].split(' '))))
			labels.append(int(temp[1]))
	# softmax原始概率
	for i in range(len(datas)):
		datas[i] = np.exp(datas[i]) / np.sum(np.exp(datas[i]))
		# for j in range(len(datas[i])):
		# 	datas[i][j] = 1.0/(1+np.exp(-datas[i][j]))
	return datas, labels

root_dir = '/home/wanglinhui/PycharmProjects/2LSTM/new_deep/runs'
# timestamp, name = '1545588050', 'Netbeans'
# timestamp, name = '1545376541', 'OpenOffice'
# timestamp, name = '1545587980', 'Mozilla'
# timestamp, name = '1545588335', 'Eclipse'
timestamp, name = '1545010144', 'GCC'
# timestamp, name = '1545044550', ''

print(timestamp, name)
train_probs, train_labels = get_probs_and_labels(root_dir, timestamp, 0)
thresold = int((len(train_probs)/8)*7)
temp_probs, temp_labels = train_probs[-thresold:], train_labels[-thresold:]
test_probs, test_labels = get_probs_and_labels(root_dir, timestamp, 1)
# val_probs, val_labels = gent_probs_and_labels(timestamp, 2)
val_probs, val_labels = test_probs, test_labels

top1, top5 = caculate_top(val_labels, val_probs)
print('验证集：top1={:.4f}, top5={:.4f}'.format(top1, top5))
top1, top5 = caculate_top(test_labels, test_probs)
print('测试集：top1={:.4f}, top5={:.4f}'.format(top1, top5))

developer_size = len(val_probs[1])
val_B = np.zeros(shape=(len(val_probs), developer_size))        # shape=(n_samples, n_developers)
for i in range(len(val_labels)):
	val_B[i][val_labels[i]] = 1.0

W, b = input_2_hidden(developer_size, L=1000)
H = sigmoid(W, b, np.mat(val_probs))
beta = calculate_beta(H, val_B)

H_test = sigmoid(W, b, np.mat(test_probs))

top1, top5 = caculate_top(test_labels, np.matmul(H_test, beta))
print('修改之后的测试集：top1={:.4f}, top5={:.4f}'.format(top1, top5))


