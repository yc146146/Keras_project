import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM, Embedding, Conv1D, Flatten, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split

model_file = "/ck/test.h5"

data_list = pd.read_csv("../data/python_new.csv",encoding="gbk")
# data_list = pd.read_csv("../data/python_new.csv",encoding="gbk")
# print(data_list.head())
data_ck_list = pd.read_csv("../data/python_ck.csv",encoding="gbk")

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


# print(x_train["education"].head())
# print(x_train["address"].head())
# print(x_train["job_type"].head())

#设置 x的训练集
# print(x_train)
x_train = x_train.values
# print(x_train)

y_train = data_list[["salary"]]
y_train = pd.get_dummies(y_train)

#one-hot查看列名
# print(list(y_train))
file.write(str(list(y_train)))
file.close()

# 提取one-hot 值
# print(y_train.values)

y_train = y_train.values
# print(y_train.shape[1])

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

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

# tanh relu
#创建学习模型
model = Sequential()

# input_length:输入序列的长度
# input_dim: 就是数据的维度
# output_dim: 词向量的维度
#loss: 0.42399230333719756               accuracy: 0.8224951708678723 100 
model.add(Embedding(input_dim = 1000, output_dim = 8, input_length=903))
# model.add(GlobalMaxPool1D())
# model.add(Conv1D(128, 5, activation='relu'))
model.add(Flatten())

#CNN 卷积神经网络 
# x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
# # print(x_train.shape)
# model.add(Conv1D(100, 5, padding='valid', activation="relu", input_shape=(x_train.shape[1],1)))
# model.add(Flatten())
# model.add(Dropout(0.4))

# RNN 循环神经网络
# model.add(Embedding(903,32))
# model.add(LSTM(32))

# model.add(Dense(512, input_dim=903, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(256, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(128, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(64, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(32, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(y_train.shape[1]*2, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))

model.add(Dense(y_train.shape[1], kernel_initializer='random_uniform', bias_initializer='random_uniform'))

# model.add(Dense(y_train.shape[1],input_dim=903, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Activation('softmax'))

# ada = optimizers.Adam(lr=0.005, epsilon=1e-8)
# ada = optimizers.Adagrad(lr=0.005, epsilon=1e-8)
# ada = optimizers.Adamax(lr=0.005, epsilon=1e-8, beta_1=0.9, beta_2=0.999)


#loss: 0.048335879430386856 		accuracy: 0.9806170850154281 python 2048 10
#loss: 0.04031918211255142 		accuracy: 0.9823382718773741 python 512 10
#loss: 0.047787329152792594 		accuracy: 0.9808909100784625 python 512 10 无 Embedding
#loss: 0.04719234061323431 		accuracy: 0.9851261361139098 python 512 100

#loss: 0.07490006572916721          accuracy: 0.9759101818638884 all 2048 10 
#loss: 0.028386146681590733 		accuracy: 0.9877186980202701 all 2048 100
ada = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)

#loss: 0.08284054080844909               accuracy: 0.9748895210990844 all 2048 10 
# ada = optimizers.Adamax(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

# loss: 1.3507972480963186                accuracy: 0.5086978649411515 all 2048 10 
# ada = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

# loss: 0.16920969194946764               accuracy: 0.9512669263341215  all 2048 10 
# ada = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
# ada = optimizers.Adagrad(lr=0.1, epsilon=None, decay=0.0)


# loss: 0.1087048964805871                accuracy: 0.966727016299333 all 2048 10 
# ada = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)


# loss: 0.0546744127769899                accuracy: 0.9801958445461728 all 2048 10 
# loss: 0.04195473445235419               accuracy: 0.9857969636658466 all 2048 100
# ada = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

# loss: 0.08993382060727524               accuracy: 0.9710738629871115 all 2048 10 
# ada = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

#模型
model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=512, epochs=10, shuffle=True)

score = model.evaluate(x_train, y_train, batch_size=512, verbose=0)

print('loss:', score[0], '\t\taccuracy:', score[1])

score2 = model.evaluate(x_test, y_test, batch_size=512, verbose=0)

print('loss:', score2[0], '\t\taccuracy:', score2[1])

# model.save(model_file)

# #测试结果
res = model.predict(np.array(x_ck))

## 四舍五入显示结果
print("res:", np.round(res))

print("res:", res)