import pandas as pd

from keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM, Embedding, Conv1D, Flatten, GlobalMaxPool1D, Input
from keras.models import Sequential
from keras import optimizers
import keras as K
import numpy as np
from sklearn.model_selection import train_test_split

model_file = "/ck/test.h5"

data_list = pd.read_csv("../data/python_all_no0.csv",encoding="gbk")
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
x_train["education"] = (x_train["education"]-x_train["education"].min())/(x_train["education"].max()-x_train["education"].min())
x_train["address"] = (x_train["address"]-x_train["address"].min())/(x_train["address"].max()-x_train["address"].min())
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
    temp = [0,0,0,0,0,0,0]
    temp[i] = 1
    # print(type(temp))
    temp = ','.join('%s' %i for i in temp)
    # print(temp)
    # print(i,element)
    y_train["salary"].replace(element, temp, inplace=True)
    file.write("数字:"+temp+"名称:"+element+"\n")

# print(y_train["salary"])

y_train = y_train["salary"].str.split(',',expand = True)
y_train = y_train.astype("int")
# print(y_train.values)

# y_train = data_list[["salary"]]
# y_train = pd.get_dummies(y_train)

#one-hot查看列名
# print(list(y_train))
file.write(str(list(y_train)))
file.close()

# 提取one-hot 值
# print(y_train.values)

Y = y_train.values




# print(x_train)
# print(y_train)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=100)


#打乱
np.random.seed(200)
np.random.shuffle(x_train)
np.random.seed(200)
np.random.shuffle(y_train)
np.random.seed(200)
np.random.shuffle(x_test)
np.random.seed(200)
np.random.shuffle(y_test)

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

print(x_train.shape)
# tanh relu
#创建学习模型
model = Sequential()

# input_length:输入序列的长度
# input_dim: 就是数据的维度
# output_dim: 词向量的维度
#loss: 0.42399230333719756               accuracy: 0.8224951708678723 100 
# model.add(Embedding(input_dim = 1000, output_dim = 8, input_length=903))
# model.add(Flatten())

#CNN 卷积神经网络 
# x_train = x_train.reshape((x_train.shape[0],x_train.shape[1],1))
# # print(x_train.shape)
# model.add(Conv1D(100, 5, padding='valid', activation="relu", input_shape=(x_train.shape[1],1)))
# model.add(Flatten())
# model.add(Dropout(0.4))

# RNN 循环神经网络
# model.add(Embedding(903,32))
# model.add(LSTM(32))

#relu sigmoid tanh

#sigmoid
#train: loss: 0.253910186316774         accuracy: 0.8900046348571777
#test:   loss: 0.29255559049540525       accuracy: 0.8825555443763733

#relu
#train: loss: 0.2523821475775086        accuracy: 0.8929339647293091
#test:   loss: 0.30454058825382213       accuracy: 0.8856430649757385

# model.add(Dense(512, input_dim=903, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(2048, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(2048, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(1024, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(256, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dropout(0.5))
model.add(Dense(128, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(64, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dropout(0.5))
# model.add(Dense(32, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(y_train.shape[1]*2, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))

model.add(Dense(y_train.shape[1], kernel_initializer='random_uniform', bias_initializer='random_uniform'))

# model.add(Dense(y_train.shape[1],input_dim=903, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Activation('softmax'))

# ada = optimizers.Adam(lr=0.005, epsilon=1e-8)
# ada = optimizers.Adagrad(lr=0.005, epsilon=1e-8)
# ada = optimizers.Adamax(lr=0.005, epsilon=1e-8, beta_1=0.9, beta_2=0.999)


#python_all 10 Dropout 64 256
#train:  loss: 0.36931401099332556       accuracy: 0.8561852609363279 
#test:  loss: 0.41405315484319416       accuracy: 0.8470490442617902
#python_all 10 64 256
#train: loss: 0.31343763831690985       accuracy: 0.8793428778782081
#test:  loss: 0.3617464618839429        accuracy: 0.8686616783427479
#python_all 1000 64 256
#train loss: 0.2238634170517409          train accuracy: 0.899742693145868
#test loss: 0.29528876053134645       test accuracy: 0.8935993349038285
#python_all 10 128
# train:  loss: 0.2521736741714839        accuracy: 0.8911921977996826
# test:   loss: 0.3125471507882919        accuracy: 0.8817242383956909

#Embedding
# 1024 256 64 10  1000 2 903 8 
# train:  loss: 0.23285052501692058       accuracy: 0.8986079096794128
# test:   loss: 0.28460669226961255       accuracy: 0.8902742862701416
# 1024 256 64 10 1000 2 903
# train:    loss: 0.2500555482228922        accuracy: 0.8924984931945801
# test:   loss: 0.30326118313590955       accuracy: 0.885286808013916
# 1024 256 64 10 1000 16 903
# train:    loss: 0.22915642012647772       accuracy: 0.8998218774795532
# test:   loss: 0.2798281347697824        accuracy: 0.8920555710792542
# 1024 256 64 10 1000 4 903
# train:    loss: 0.23833893809078976       accuracy: 0.897420346736908
# test:   loss: 0.2901196311763369        accuracy: 0.888730525970459

#16
# train:  loss: 0.2653435735227235        accuracy: 0.890110194683075
# test:   loss: 0.3219580901814334        accuracy: 0.8801804780960083

#8
# train:  loss: 0.3006102982031298        accuracy: 0.8761364221572876
# test:   loss: 0.3643053164781945        accuracy: 0.8685429096221924

# 1024 256 64 10
# train:  loss: 0.43118038875427317       accuracy: 0.8356139063835144
# test:   loss: 0.47809720788584115       accuracy: 0.8236551284790039

# 1024 256 64 50
# train:  loss: 0.2694371644854585        accuracy: 0.8849112391471863
# test:   loss: 0.3905982248411653        accuracy: 0.8741242289543152

# 1024 256 64 100
# train:  loss: 0.27230331986501566       accuracy: 0.8877878189086914
# test:   loss: 0.38153982259445496       accuracy: 0.8785179853439331

# train:  loss: 0.2415734855168734        accuracy: 0.8941875100135803
# test:   loss: 0.28906015335642915       accuracy: 0.8855242729187012


optimizer = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)

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
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, Y, batch_size=512, epochs=10, shuffle=True)

score = model.evaluate(X, Y, batch_size=512, verbose=0)

print('train:\tloss:', score[0], '\t\taccuracy:', score[1])

score2 = model.evaluate(x_test, y_test, batch_size=512, verbose=0)

print('test:\tloss:', score2[0], '\t\taccuracy:', score2[1])

# #测试结果
res = model.predict(np.array(x_ck))

## 四舍五入显示结果
print("res:\n", np.round(res))

print("res:\n", res)