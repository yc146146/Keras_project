import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import LabelBinarizer

import numpy as np


# 训练数据
x_train1 = np.random.randint(0, 99, (100, 1))
x_train2 = np.random.randint(100, 999, (100, 1))
x_train3 = np.random.randint(1000, 9999, (100, 1))

xs = np.concatenate((x_train1, x_train2, x_train3))
# y_train = np.array([[1,0,0]*100 + [0,1,0]*100 + [0,0,1]*100])
y_train = np.array([[0]*100 + [1]*100 + [2]*100])


labels = y_train.reshape((300, 1))
labels = y_train[0]

x_train = xs.astype(float)
y_train = labels.astype(float)

# 打乱数据
# 学习效率高
# arr = np.arange(x_train.shape[0])
# np.random.shuffle(arr)
# x_train = x_train[arr, :]
# y_train = labels[arr, :]

train_size, num_features = x_train.shape

# print(x_train.shape)
# print(y_train)

# x = torch.from_numpy(x_train)
# y = torch.from_numpy(y_train)

x = torch.tensor(x_train, dtype=torch.float32).clone().detach()
y = torch.tensor(y_train, dtype=torch.long).clone().detach()

#读取模型
model = torch.load('net.pkl')

test = torch.from_numpy(np.array([22]))
test = torch.tensor(test, dtype=torch.float32)
print(model(test))

learning_rate = 0.005

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer.load_state_dict(model['optimizer'])

loss_fn = torch.nn.CrossEntropyLoss()

#继续训练
for t in range(500):
	# x = torch.tensor(x, dtype=torch.float32)
	y_pred = model(x)
	# y_pred = net.forward(x)
	
	# y = Variable(y).type(torch.LongTensor)
	loss = loss_fn(y_pred, y)
	print(t, loss.item())

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


test = torch.tensor([[22],[55],[88]], dtype=torch.float32)
res = model(test)
print(torch.max(res, 0)[1].item())