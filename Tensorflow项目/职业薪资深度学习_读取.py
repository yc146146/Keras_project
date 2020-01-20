import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import os

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


X = x_train.values


y_train = data_list[["salary"]]
y_train = pd.get_dummies(y_train)

Y = y_train.values




#测试集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# np.random.seed(200)
# np.random.shuffle(x_train)
# np.random.seed(200)
# np.random.shuffle(y_train)
# np.random.seed(200)
# np.random.shuffle(x_test)
# np.random.seed(200)
# np.random.shuffle(y_test)


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



if os.path.exists(model_file):
    model = load_model(model_file)

    score = model.evaluate(x_train, y_train, verbose=0, batch_size=512)

    print('loss:', score[0], '\t\taccuracy:', score[1])


    # #测试结果
    res = model.predict(np.array(x_ck))

    ## 四舍五入显示结果
    print("res:\n", np.round(res))

    print("res:\n", res)