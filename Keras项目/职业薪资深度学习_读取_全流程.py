import pandas as pd

from keras.layers import Dense, Activation, Dropout, SimpleRNN, LSTM, Embedding, Conv1D, Flatten, GlobalMaxPool1D
from keras.models import Sequential, load_model, model_from_json
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import keras as K
import numpy as np
import os
from sklearn.model_selection import train_test_split
from numpy import float64
import gensim
from gensim.models.doc2vec import Doc2Vec
import jieba
import codecs
import jieba.posseg as pseg
import chardet
import re

# from keras import backend as K
# import tensorflow as tf
# from keras.backend import manual_variable_initialization
# from tensorflow.keras.backend import set_session

def content_to_vector(content):
  # content = re.sub("\\W+", ' ', content)
  content = re.sub("[^\u4e00-\u9fa5^a-z^A-Z^ ]", ' ', content)

  sentence_cut = jieba.cut(content)
  sentence_split = " ".join(sentence_cut).split()
  content = [word for word in sentence_split]
  # print(content)

  model = Doc2Vec.load("./doc2vec/salary.model")
  inferred_vector = model.infer_vector(doc_words=content, alpha=0.025, steps=1000)
  sims = model.docvecs.most_similar([inferred_vector], topn=1)
  # print(sims[0][0])
  job_info_num = sims[0][0]
  return job_info_num


model_file = "./cp/model.h5"

data_list = pd.read_csv("../data/all_12_info.csv",encoding="gbk")
# data_list = pd.read_csv("../data/python_new.csv",encoding="gbk")
# print(data_list.head())
data_ck_list = pd.read_csv("../data/python_ck_true.csv",encoding="gbk")

x_train = data_list.drop(["salary"], axis=1)
# print(data_list.shape)
# print(x_train.shape)

# print(x_train["education"])



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


# print(y_train)

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

#测试集
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# np.random.seed(200)
# np.random.shuffle(x_train)
# np.random.seed(200)
# np.random.shuffle(y_train)


# print(x_train)
# print(y_train)


# x_train = X[:-1000,:]
# y_train = Y[:-1000,:]
# x_test = X[-1000:,:]
# y_test = Y[-1000:,:]

#验证数据
data_ck_list["job_info_num"] = data_ck_list["job_info"].apply(lambda x: content_to_vector(x))

# print(data_ck_list)

x_ck = data_ck_list.drop(["salary", "job_info"], axis=1)

# print(x_ck.columns)

#数字化学历
for i, element in enumerate(education_data):
    # print(i,element)
    x_ck["education"].replace(element, i, inplace=True)

#数字化城市名
for i, element in enumerate(address_data):
    # print(i,element)
    x_ck["address"].replace(element, i, inplace=True)

# #数字化工作标签
for i, element in enumerate(job_type_data):
    # print(i,element)
    x_ck["job_type"].replace(element, i, inplace=True)

x_ck = x_ck.values


if os.path.exists(model_file):
    # sess = tf.Session()

    # set_session(sess)
    # sess.run(tf.global_variables_initializer())

    model = load_model(model_file)

    # model = model_from_json(open('./cp/test.json').read())
    # model.load_weights('./cp/test.h5')

    # ada = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

    # model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

    score = model.evaluate(X, Y, verbose=0, batch_size=512)

    print('loss:', score[0], '\t\taccuracy:', score[1])

    # weight_Dense_1,bias_Dense_1 = model.get_layer('Dense_1').get_weights()
    # weight_Dense_2,bias_Dense_2 = model.get_layer('Dense_2').get_weights()
    # weight_Dense_3,bias_Dense_3 = model.get_layer('Dense_3').get_weights()


    # print(weight_Dense_1)
    # print(bias_Dense_1)
    # print(weight_Dense_2)
    # print(bias_Dense_2)
    # print(weight_Dense_3)
    # print(bias_Dense_3)

    print(np.array(x_ck))
    #测试结果
    res = model.predict(np.array(x_ck))

    ## 四舍五入显示结果
    print("res:", np.round(res))

    print("res:", res)