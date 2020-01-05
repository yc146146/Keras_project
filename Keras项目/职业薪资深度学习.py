import pandas as pd

# from keras.layers import Dense, Activation, Dropout
# from keras.models import Sequential
# from keras import optimizers
# import keras as K
# import numpy as np


data_list = pd.read_csv("./python.csv",encoding="gbk")
# print(data_list.head())

x_train = data_list.drop(["salary"], axis=1)
# print(x_train.head())

# print(x_train["education"])

# education_data = x_train["education"].drop_duplicates(keep='first',inplace=False)
education_data = set(x_train["education"].values)
address_data = set(x_train["address"].values)
job_type_data = set(x_train["job_type"].values)
# print(education_data)

#数字化学历
for i, element in enumerate(education_data):
    print(i,element)
    x_train["education"].replace(element, i, inplace=True)

#数字化城市名
for i, element in enumerate(address_data):
    print(i,element)
    x_train["address"].replace(element, i, inplace=True)

#数字化工作标签
for i, element in enumerate(job_type_data):
    print(i,element)
    x_train["job_type"].replace(element, i, inplace=True)

print(x_train["education"].head())
print(x_train["address"].head())
print(x_train["job_type"].head())

#设置 x的训练集
# print(x_train.values)
x_train = x_train.values




y_train = data_list[["salary"]]
y_train = pd.get_dummies(y_train)

#one-hot查看列名
# print(list(y_train))

# 提取one-hot 值
# print(y_train.values)

y_train = y_train.values



#创建学习模型
# model = Sequential()

# model.add(Dense(3, kernel_initializer='random_uniform', bias_initializer='random_uniform', input_dim=903))
# model.add(Activation('softmax'))

# ada = optimizers.Adam(lr=0.01, epsilon=1e-8)
# model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=128, epochs=100, shuffle=True)

# score = model.evaluate(x_train, y_train, batch_size=128)

# print('loss:', score[0], '\t\taccuracy:', score[1])