import numpy as np
import copy
# np.random.seed(1)
# a = np.random.randint(0, 5, size=(4,3))
# print(a)
# a = a / a.sum(1, keepdims=True)
# print(a)
# a = np.mat(a)       # 将数组转化为矩阵
# print(a.I)          # 求矩阵的逆
# b = [[1,0,0], [1,0,0], [0,1,0], [0,0,1]]
# b = np.array(b)
# print(b)
# x = np.matmul(a.I, b)
# print(x)
# x_n = copy.deepcopy(x)
# for i in range(len(x)):
# 	x_n[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
# print(x_n)
# print(np.matmul(a, x_n))
# print(np.matmul(a, x))

def caculate_top(labels, predicts):
	top1_num = 0
	top5_num = 0
	predicts = np.array(predicts)
	for i in range(len(labels)):
		# print(i)
		top1 = np.argmax(predicts[i])
		# top1 = np.argmin(predicts[i])
		top5 = np.argsort(predicts[i])[-5:]
		# top5 = np.argsort(predicts[i])[:5]
		if labels[i] == top1:
			top1_num += 1
		# print(np.shape(top5))
		# print('label={}, top5={}'.format(labels[i], ' '.join(list(map(str, top5)))))
		if labels[i] in top5:
			top5_num += 1
	# print('top1_num={}, top5_num={}'.format(top1_num, top5_num))
	return top1_num/len(labels), top5_num / len(labels)

def get_probs_and_labels(timestamp, round_id):
	datas = []
	labels = []
	with open('../data/{}/checkpoints/origin_prob_{}.txt'.format(timestamp, round_id, timestamp), 'r') as reader:
		for line in reader.readlines():
			temp = line.strip().split(',')
			datas.append(list(map(float, temp[0].split(' '))))
			labels.append(int(temp[1]))
	# softmax原始概率
	for i in range(len(datas)):
		datas[i] = np.exp(datas[i]) / np.sum(np.exp(datas[i]))
	return datas, labels
timestamp, name = '1545376541', 'OpenOffice'      # OpenOffice
# timestamp, name = '1545035704', 'Mozilla'      # Mozilla
# timestamp, name = '1545051846', 'Eclipse'        # Eclipse
# timestamp, name = '1545036268', 'Netbeans'        # Netbeans
# timestamp , name= '1545008884', 'GCC'        # GCC
# timestamp = '8978'
# timestamp = '0300'
print(timestamp, name)
train_probs, train_labels = get_probs_and_labels(timestamp, 0)
thresold = int(len(train_labels)/2)
val_probs, val_labels = train_probs[-thresold:], train_labels[-thresold:]
test_probs, test_labels = get_probs_and_labels(timestamp, 1)
# thresold = int(len(origin_test_probs)/2)
# val_probs, val_labels = origin_test_probs[:thresold], origin_test_labels[:thresold]
# test_probs, test_labels = origin_test_probs[-thresold:], origin_test_labels[-thresold:]

print(len(val_probs))
print(len(test_probs))

top1, top5 = caculate_top(val_labels, val_probs)
print('验证集：top1={:.4f}, top5={:.4f}'.format(top1, top5))
top1, top5 = caculate_top(test_labels, test_probs)
print('测试集：top1={:.4f}, top5={:.4f}'.format(top1, top5))

developer_size = len(val_probs[1])
B = np.zeros(shape=(len(val_probs), developer_size))        # shape=(n_samples, n_developers)
for i in range(len(val_labels)):
	B[i][val_labels[i]] = 1

A = np.mat(val_probs)
x = np.matmul(A.I, B)
new_B = np.matmul(A, x)

# test_B = np.zeros(shape=(len(test_probs), developer_size))        # shape=(n_samples, n_developers)
# for i in range(len(test_labels)):
# 	test_B[i][test_labels[i]] = 1
# test_x = np.matmul(np.mat(test_probs).I, test_B)

top1, top5 = caculate_top(val_labels, np.matmul(A, x))
print('修改之后的验证集：top1={:.4f}, top5={:.4f}'.format(top1, top5))
top1, top5 = caculate_top(val_labels, B)
print('B验证集：top1={:.4f}, top5={:.4f}'.format(top1, top5))

new_test_B = np.matmul(np.mat(test_probs), x)
print(np.shape(new_test_B))
top1, top5 = caculate_top(test_labels, new_test_B)
print('修改之后的测试集：top1={:.4f}, top5={:.4f}'.format(top1, top5))



