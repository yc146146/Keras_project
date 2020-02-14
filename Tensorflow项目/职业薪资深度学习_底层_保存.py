
# coding:utf-8
'''
使用Softmax进行多类别分类
'''
import numpy as np
# import tensorflow as tf
# import matplotlib.pyplot as plt
import pandas as pd
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

data_list = pd.read_csv("../data/python_all_no0.csv",encoding="gbk")
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
#格式化
# def make_data(education_data, address_data, job_type_data, file):

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

# print(x_ck)

train_size, num_features = x_train.shape
learning_rate = 0.005
training_epochs = 500
num_labels = y_train.shape[1]
batch_size = 512

# 全连接层函数
def fcn_layer(
        inputs,  # 输入数据
        input_dim,  # 输入层神经元数量
        output_dim,  # 输出层神经元数量
        activation=None):  # 激活函数

    W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    # 以截断正态分布的随机初始化W
    b = tf.Variable(tf.zeros([output_dim]))
    # 以0初始化b
    XWb = tf.matmul(inputs, W)+b  # Y=WX+B

    if(activation == None):  # 默认不使用激活函数
        outputs = XWb
    else:
        outputs = activation(XWb)  # 代入参数选择的激活函数
    return outputs  # 返回

X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

#产生截断正态分布随机数
# W = tf.Variable(tf.truncated_normal([num_features, 9], stddev=0.1))
# #初始化为零向量
# b = tf.Variable(tf.zeros([9]))

# layer_1 = tf.nn.tanh(tf.matmul(X, W) + b)

# #产生截断正态分布随机数
# W_2 = tf.Variable(tf.truncated_normal([9, num_labels], stddev=0.1))
# #初始化为零向量
# b_2 = tf.Variable(tf.zeros([num_labels]))

# y_model = tf.nn.softmax(tf.matmul(layer_1, W_2) + b_2)


X = tf.cast(tf.convert_to_tensor(X), tf.float32)


#2层
#隐层 relu tanh
h1 = fcn_layer(X, num_features, 256, tf.nn.relu)
# h2 = fcn_layer(h1, 256, 128, tf.nn.relu)
h3 = fcn_layer(h1, 256, 64, tf.nn.relu)
# #全连接输出层 Sigmoid Softmax 
y_model = fcn_layer(h3, 64, num_labels, tf.nn.softmax)

#1层
# y_model = fcn_layer(X, num_features, num_labels, tf.nn.softmax)

# cost = -tf.reduce_sum(Y * tf.log(y_model))
cost = -tf.reduce_mean(Y * tf.log(y_model))

# cost = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_model, labels=Y))

# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


predict = tf.argmax(y_model, 1, name='predict')

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


saver = tf.train.Saver()


if os.path.exists("./checkpoint/checkpoint"):
    print("old")

    with tf.Session()as sess3:
        sess3.run(tf.global_variables_initializer())
        saver.restore(sess3, "./checkpoint/test.ckpt")
        print("accuracy", accuracy.eval(feed_dict={X: x_train, Y: y_train}))
        print("loss:", cost.eval(feed_dict={X: x_train, Y: y_train}))

else:

	print("new")
	


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # training_epochs * train_size // batch_size

    for step in range(training_epochs):

        offset = (step * batch_size) % train_size

        batch_xs = x_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size)]

        # print(batch_xs)
        # print(batch_labels)

        # print(batch_xs)
        # print(batch_labels)
        err = sess.run([cost, train_op], feed_dict={
                       X: batch_xs, Y: batch_labels})
        print(step, err)
    # W_val = sess.run(W)
    # print('w', W_val)
    # b_val = sess.run(b)
    # print('b', b_val)
    print("accuracy:", accuracy.eval(feed_dict={X: x_train, Y: y_train}))
    print("loss:", cost.eval(feed_dict={X: x_train, Y: y_train}))
    # print("predict", predict.eval(feed_dict={X: x_ck}))
    
    saver.save(sess, "./checkpoint/test.ckpt")

with tf.Session()as sess2:
	sess2.run(tf.global_variables_initializer())
	saver.restore(sess2, "./checkpoint/test.ckpt")
	print("accuracy", accuracy.eval(feed_dict={X: x_train, Y: y_train}))