import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split

model_file = "./cp/model.h5"

data_list = pd.read_csv("../data/python_all_no0.csv",encoding="gbk")

data_ck_list = pd.read_csv("../data/python_ck.csv",encoding="gbk")

x_train = data_list.drop(["salary"], axis=1)


education_data = set(x_train["education"].values)
address_data = set(x_train["address"].values)
job_type_data = set(x_train["job_type"].values)


for i, element in enumerate(education_data):
    # print(i,element)
    x_train["education"].replace(element, i, inplace=True)

for i, element in enumerate(address_data):
    # print(i,element)
    x_train["address"].replace(element, i, inplace=True)

for i, element in enumerate(job_type_data):
    # print(i,element)
    x_train["job_type"].replace(element, i, inplace=True)


x_train = x_train.values
# print(x_train)

y_train = data_list[["salary"]]
y_train = pd.get_dummies(y_train)

y_train = y_train.values


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

#验证数据
x_ck = data_ck_list.drop(["salary"], axis=1)


for i, element in enumerate(education_data):
    # print(i,element)
    x_ck["education"].replace(element, i, inplace=True)


for i, element in enumerate(address_data):
    # print(i,element)
    x_ck["address"].replace(element, i, inplace=True)


for i, element in enumerate(job_type_data):
    # print(i,element)
    x_ck["job_type"].replace(element, i, inplace=True)

x_ck = x_ck.values

# tanh relu
#创建学习模型
model = Sequential()


model.add(Embedding(input_dim = 1000, output_dim = 8, input_length=903))
model.add(Flatten())
model.add(Dense(256, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(64, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(y_train.shape[1], kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Activation('softmax'))


optimizer = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)


#模型
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=512, epochs=10, shuffle=True)

score = model.evaluate(x_train, y_train, batch_size=512, verbose=0)

print('loss:', score[0], '\t\taccuracy:', score[1])

score2 = model.evaluate(x_test, y_test, batch_size=512, verbose=0)

print('loss:', score2[0], '\t\taccuracy:', score2[1])

#保存模型
model.save(model_file)

# #测试结果
res = model.predict(np.array(x_ck))

## 四舍五入显示结果
print("res:", np.round(res))

print("res:", res)