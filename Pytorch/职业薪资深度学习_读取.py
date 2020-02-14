import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as Data


model_file = "/ck/test.h5"

data_list = pd.read_csv("../data/python_test.csv",encoding="gbk")
# data_list = pd.read_csv("../data/python_new.csv",encoding="gbk")
# print(data_list.head())
data_ck_list = pd.read_csv("../data/python_ck_2.csv",encoding="gbk")

x_train = data_list.drop(["salary"], axis=1)
# print(data_list.shape)
# print(x_train.shape)

# print(x_train["education"])

# education_data = x_train["education"].drop_duplicates(keep='first',inplace=False)
education_data = set(x_train["education"].values)
address_data = set(x_train["address"].values)
job_type_data = set(x_train["job_type"].values)
# print(education_data)

file = open("./数据化文件.txt", "w")

file.write("\n学历\n")
#数字化学历
for i, element in enumerate(education_data):
    # print(i,element)
    x_train["education"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

file.write("\n城市名\n")
#数字化城市名
for i, element in enumerate(address_data):
    # print(i,element)
    x_train["address"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

file.write("\n工作标签\n")
#数字化工作标签
for i, element in enumerate(job_type_data):
    # print(i,element)
    x_train["job_type"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

# 归一化
# x_train["education"] = (x_train["education"]-x_train["education"].min())/(x_train["education"].max()-x_train["education"].min())
# x_train["address"] = (x_train["address"]-x_train["address"].min())/(x_train["address"].max()-x_train["address"].min())
# x_train["job_type"] = (x_train["job_type"]-x_train["job_type"].min())/(x_train["job_type"].max()-x_train["job_type"].min())

# x_train["education"] = (x_train["education"]-x_train["education"].mean())/x_train["education"].std()
# x_train["address"] = (x_train["address"]-x_train["address"].mean())/x_train["address"].std()
# x_train["job_type"] = (x_train["job_type"]-x_train["job_type"].min())/(x_train["job_type"].max()-x_train["job_type"].min())


# print(x_train["education"].head())
# print(x_train["address"].head())
# print(x_train["job_type"].head())

#设置 x的训练集
# print(x_train)
X = x_train.values
# print(x_train)

salary_data = ['1000-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000', '25000-30000','30000以上' ]

y_train = data_list[["salary"]].copy()

file.write("\n金额\n")

#数字化目标标签
for i, element in enumerate(salary_data):
    # print(temp)
    # print(i,element)
    y_train["salary"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

# print(y_train["salary"])

y_train = y_train["salary"]
y_train = y_train.astype("float32")
# print(y_train.values)

# y_train = data_list[["salary"]]
# y_train = pd.get_dummies(y_train)

#one-hot查看列名
# print(list(y_train))
# file.write(str(list(y_train)))
file.close()

# 提取one-hot 值
# print(y_train.values)

Y = y_train.values




# print(x_train)
# print(y_train)

# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=100)


#打乱
# np.random.seed(200)
# np.random.shuffle(x_train)
# np.random.seed(200)
# np.random.shuffle(y_train)
# np.random.seed(200)
# np.random.shuffle(x_test)
# np.random.seed(200)
# np.random.shuffle(y_test)

X = torch.tensor(X, dtype=torch.float32).clone().detach()
Y = torch.tensor(Y, dtype=torch.long).clone().detach()



#验证数据
x_ck = data_ck_list.drop(["salary"], axis=1)

#数字化学历
for i, element in enumerate(education_data):
    # print(i,element)
    x_ck["education"].replace(element, i, inplace=True)

#数字化城市名
for i, element in enumerate(address_data):
    # print(i,element)
    x_ck["address"].replace(element, i, inplace=True)

#数字化工作标签
for i, element in enumerate(job_type_data):
    # print(i,element)
    x_ck["job_type"].replace(element, i, inplace=True)

x_ck = x_ck.values


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
        print("loss:",loss, "acc:",acc)


train_size, num_features = x_train.shape
batch_size = 512
learning_rate = 0.005
total_len = len(Y)

#自定义数据集
dataset = Data.TensorDataset(X, Y)


# 定义数据加载器
all_datas = Data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)



#读取模型
model = torch.load('./cp/net.pkl')

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer.load_state_dict(model['optimizer'])

loss_fn = torch.nn.CrossEntropyLoss()

model_assessment(all_datas, model, total_len)

# ck_x = torch.tensor(x_ck, dtype=torch.float32)
# res = model(ck_x)

# print(res)
# print(torch.max(res, 1))
# print(torch.max(res, 1)[1].numpy())