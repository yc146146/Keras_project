
# coding:utf-8
'''
使用Softmax进行多类别分类
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

LAYER_0_NAME = 'layer0'  # 第一层的名字
LAYER_1_NAME = 'layer1'  # 第一层的名字
LAYER_2_NAME = 'layer2'  # 第二层的名字

# 训练数据
data_list = pd.read_csv("../data/Python.csv",encoding="gbk")
# print(data_list.head())

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


train_size, num_features = x_train.shape
learning_rate = 0.001
training_epochs = 10
num_labels = y_train.shape[1]
batch_size = 256

X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

# W = tf.Variable(tf.zeros([num_features, num_labels]))
# b = tf.Variable(tf.zeros([num_labels]))

# y_model = tf.nn.softmax(tf.matmul(X, W) + b)


# 全连接层函数
# 2层
layer0 = tf.layers.dense(inputs=X, units=256, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name=LAYER_0_NAME)
layer1 = tf.layers.dense(inputs=layer0, units=64, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name=LAYER_1_NAME)
layer2 = tf.layers.dense(inputs=layer1, units=num_labels, activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name=LAYER_2_NAME)

#1层
# layer2 = tf.layers.dense(inputs=X, units=num_labels, activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(
#     mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name=LAYER_2_NAME)

# loss1 = tf.reduce_mean((layer1 - X) ** 2)

# loss1 = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
# loss1 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer2, labels=Y))
loss1 = -tf.reduce_mean(Y * tf.log(layer2))

# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss1)

predict = tf.argmax(layer2, 1, name='predict')

correct_prediction = tf.equal(tf.argmax(layer2, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # training_epochs * train_size // batch_size

    # tf.random.shuffle(xs)
    # print(xs[0:5, :])
    for step in range(1000):

        # 每次训练的数量
        offset = (step * batch_size) % train_size
        batch_xs = x_train[offset:(offset + batch_size), :]
        batch_labels = y_train[offset:(offset + batch_size)]

        # print(batch_xs)
        # print(batch_labels)

        # print(batch_xs)
        # print(batch_labels)
        err = sess.run([train_op, loss1], feed_dict={
                       X: batch_xs, Y: batch_labels})
        # print("次数:%d, 丢失:%10.10f"%(step, err))
        print(step, err)

    # W_val = sess.run(W)
    # print('w', W_val)
    # b_val = sess.run(b)
    # print('b', b_val)
    print("accuracy", accuracy.eval(feed_dict={X: x_train, Y: y_train}))
    # print("predict", predict.eval(feed_dict={X: [[15.], [40.], [90.], [60]]}))

    # 把模型保存成checkpoint
    # saver = tf.train.Saver()
    # save_path = saver.save(sess, './checkpoint/model.ckpt')
    # print("model saved in path: %s" % save_path, flush=True)
    # # 读取刚保存的checkpoint
    # reader = tf.train.NewCheckpointReader(save_path)
    # # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
    # weights = reader.get_tensor(LAYER_1_NAME + '/kernel')
    # bias = reader.get_tensor(LAYER_1_NAME + '/bias')  # bias的名字
    # print("weights",weights)
    # print("bias",bias)
    # 如果想打印模型中的所有参数名和参数值的话，把下面几行取消注释
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor name: ", key)
    #     print(reader.get_tensor(key))
