import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data

from sklearn.preprocessing import LabelBinarizer

import numpy as np

#评估
def model_assessment(all_datas, model, total_len):

	#评估模型丢失率与准确率
	for epoch in range(1):
		train_acc = 0
		train_loss = 0
		for step, (batch_x, batch_y) in enumerate(all_datas):
			# print("epoch:",epoch,"step:",step,"batch_x:",batch_x.numpy(),"batch_y",batch_y.numpy())
			# print("epoch:",epoch,"step:",step)

		 	y_pred = model(batch_x)
		 	loss_temp = loss_fn(y_pred, batch_y)

		 	train_loss += loss_temp.item()

		 	# print(step, loss.item())
		 	train_correct = (torch.max(y_pred, 1)[1] == batch_y.squeeze(0)).sum()
		 	# print(batch_y.squeeze(0))
		 	# print(torch.max(y_pred, 1)[1])
		 	# print(train_correct.item())
		 	# print(train_correct.data[0])
		 	train_acc += train_correct.item()

		 	optimizer.zero_grad()
		 	loss_temp.backward()
		 	optimizer.step()
		#计算每次的准确率
		acc = train_acc / total_len
		loss = train_loss / len(all_datas)
		print("评估：loss:",loss, "准确率：acc:",acc)


#训练
def mdoel_fit(epoch_num, all_datas, model, total_len):
	#学习 1000轮
	for epoch in range(epoch_num):
		train_acc = 0
		train_loss = 0
		for step, (batch_x, batch_y) in enumerate(all_datas):
			# print("epoch:",epoch,"step:",step,"batch_x:",batch_x.numpy(),"batch_y",batch_y.numpy())
			# print("epoch:",epoch,"step:",step)

		 	y_pred = model(batch_x)
		 	loss_temp = loss_fn(y_pred, batch_y)

		 	train_loss += loss_temp.item()

		 	# print(step, loss.item())
		 	train_correct = (torch.max(y_pred, 1)[1] == batch_y.squeeze(0)).sum()
		 	# print(batch_y.squeeze(0))
		 	# print(torch.max(y_pred, 1)[1])
		 	# print(train_correct.item())
		 	# print(train_correct.data[0])
		 	
		 	train_acc += train_correct.item()

		 	optimizer.zero_grad()
		 	loss_temp.backward()
		 	optimizer.step()
		#计算每次的准确率
		acc = train_acc / total_len
		loss = train_loss / len(all_datas)
		print("epoch:",epoch,"loss:",loss, "acc:",acc)



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

x = torch.tensor(x_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.long)

total_len = len(y)
batch_size = 128
learning_rate = 0.001

#自定义数据集
dataset = Data.TensorDataset(x, y)


# 定义数据加载器
all_datas = Data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)


model = torch.nn.Sequential(
	torch.nn.Linear(num_features, 9),
	torch.nn.ReLU(),
    torch.nn.Linear(9, 3)
)

# class Net(torch.nn.Module):
#     def __init__(self, n_feature, n_hidden, n_output):
#         super(Net, self).__init__()
#         self.hidden = torch.nn.Linear(num_features, 9)
#         self.predict = torch.nn.Linear(9, 3)

#     def forward(self, x):
#         x = F.relu(self.hidden(x))
#         x = self.predict(x)
#         return x

# model = Net(1, 9, 3)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


loss_fn = torch.nn.CrossEntropyLoss()



mdoel_fit(1000, all_datas, model, total_len)

model_assessment(all_datas, model, total_len)






test = torch.tensor([[22],[222],[2222]], dtype=torch.float32)
res = model(test)

# print(res)

print(torch.max(res, 1)[1].numpy())

#保存
torch.save(model, 'net.pkl')  # save entire net
# torch.save(model.state_dict(), 'net_params.pkl')

#读取模型
model2 = torch.load('net.pkl')
model_assessment(all_datas, model2, total_len)


