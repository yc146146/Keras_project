import pandas as pd

from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import optimizers
import keras as K
import numpy as np


data_list = pd.read_csv("../data/test.csv",encoding="gbk")
# print(data_list.head())

x_train = data_list.drop(["salary"], axis=1)
# print(data_list.shape)
# print(x_train.shape)

# print(x_train["education"])

# education_data = x_train["education"].drop_duplicates(keep='first',inplace=False)
# education_data = set(x_train["education"].values)
# address_data = set(x_train["address"].values)
job_type_data = set(x_train["job_type"].values)
# print(education_data)

education_data = ["大专","本科","研究生"]
address_data = ["北京","上海","深圳"]

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

# 打乱数据
# x_train = x_train.astype(float)
# y_train = y_train.astype(float)


# # 学习效率高
# arr = np.arange(x_train.shape[0])
# np.random.shuffle(arr)
# x_train = x_train[arr, :]
# y_train = y_train[arr, :]

# print(x_train)
# print(y_train)

#创建学习模型
model = Sequential()
# model.add(Dense(y_train.shape[1]*128, input_dim=903, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(y_train.shape[1]*64, input_dim=903, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(y_train.shape[1]*64, input_dim=5, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))

model.add(Dense(y_train.shape[1]*16,  activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(y_train.shape[1]*4, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))

model.add(Dense(y_train.shape[1], kernel_initializer='random_uniform', bias_initializer='random_uniform'))

# model.add(Dense(y_train.shape[1],input_dim=903, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Activation('softmax'))

# ada = optimizers.Adam(lr=0.005, epsilon=1e-8)
# ada = optimizers.Adagrad(lr=0.005, epsilon=1e-8)
# ada = optimizers.Adamax(lr=0.005, epsilon=1e-8, beta_1=0.9, beta_2=0.999)

#loss: 0.7141936179002323 		accuracy: 0.7062800257894734 10
#loss: 0.46716969466706537 		accuracy: 0.7896587588559063 500
#loss: 0.7795951458292053 		accuracy: 0.6785944234763381
#loss: 0.6422989048142353 		accuracy: 0.7227229956744312 relu
#loss: 0.5951487899216269 		accuracy: 0.7445891733291415
#loss: 0.4680684206655904 		accuracy: 0.8117478082818963 500
ada = optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#loss: 0.81395600688244365 		accuracy: 0.675672329379594
# ada = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)

# loss: 1.3253612228006908 		accuracy: 0.46983804696736625 10
# loss: 0.8919344719082928 		accuracy: 0.6150017332064793 500
# loss: 1.3256939558895233 		accuracy: 0.46983804696736625
# ada = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

#loss: 0.7281321605148927 		accuracy: 0.710588876043054
# ada = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)

# loss: 1.0462341844448098 		accuracy: 0.5548511715205946 10
# loss: 0.4888109677135841 		accuracy: 0.7862661583487024 500
# loss: 1.217691098838981 		accuracy: 0.47647466719354237
# # loss: 0.45390464728952284 		accuracy: 0.7896339951968584 500
# ada = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)

#loss: 0.6615062530195074 		accuracy: 0.7109355655929662
# ada = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

# loss: 0.7522785709892728 		accuracy: 0.6857758405380868
# ada = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)


model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=512, epochs=1000, shuffle=True)

score = model.evaluate(x_train, y_train, batch_size=512)

print('loss:', score[0], '\t\taccuracy:', score[1])

#测试结果
res = model.predict(np.array([[0,0,0,0,0],[1,0,1,1,0],[2,0,2,1,1]]))

#四舍五入显示结果
print("res:", np.round(res))

print("res:", res)