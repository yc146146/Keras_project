import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

from numpy.random import seed
seed(42)  # seed fixing for keras
# import tensorflow as tf
tf.set_random_seed(42)  # tensorflow seed fixing


fname = "model"
# model_file = "../model.h5"
model_file = os.path.join("test", fname + ".h5")

data_list = pd.read_csv("../data/python_all_no0.csv", encoding="gbk")

data_ck_list = pd.read_csv("../data/python_ck.csv", encoding="gbk")

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
print(x_train.shape)
# sf
# print(x_train)
# for i in range(x_train.shape[0]):
# 	for j in range(x_train.shape[1]):
# 		x_train[i,j]=int(x_train[i,j])
y_train = data_list[["salary"]]
y_train = pd.get_dummies(y_train)

y_train = y_train.values
print(y_train.shape)

x_test = x_train
y_test = y_train
# x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.000001)
# x_train=x_train

# 验证数据
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
# 创建学习模型

train_size, num_features = x_train.shape
learning_rate = 0.001
training_epochs = 10
num_labels = y_train.shape[1]
batch_size = 256


X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

train_inputs = tf.cast(tf.convert_to_tensor(X), tf.float32)

# Embedding lookup table currently only implemented in CPU with
# with tf.name_scope("embeddings"):
#    embeddings = tf.Variable(tf.random_uniform([x_train.shape[1], 8], -1.0, 1.0), name = 'embedding')
#    # This is essentialy a lookup table
#    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# embed = tf.layers.dense(inputs=train_inputs, units=1000,
#                         activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(
#                             mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name='l00')
# embed = tf.layers.dense(inputs=embed, units=8,
#                         activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(
#                             mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name='l0')
# embed = tf.layers.dense(inputs=train_inputs, units=256,
#                         activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(
#                             mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name='l1')
# embed = tf.layers.dense(inputs=embed, units=64,
#                         activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(
#                             mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name='l2')
# embed = tf.layers.dense(inputs=embed, units=6,
#                         activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(
#                             mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name='l3')

layer0 = tf.layers.dense(inputs=X, units=256, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name="l1")
layer1 = tf.layers.dense(inputs=layer0, units=64, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name="l2")
embed = tf.layers.dense(inputs=layer1, units=num_labels, activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name="l3")


loss = -tf.reduce_mean(Y * tf.log(embed))


predict = tf.argmax(embed, 1, name='predict')

correct_prediction = tf.equal(tf.argmax(embed, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
#     labels=y_train, logits=embed))

opt = tf.train.AdamOptimizer(0.005).minimize(loss)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())

# # tf.train.Saver().restore(sess, "./checkpoint/model.ckpt")

# for i in range(50):
#     print(sess.run([opt, loss],feed_dict={
#                        X: x_train, Y: y_train}))

# print(sess.run([accuracy],feed_dict={
#                        X: x_train, Y: y_train}))

# print("predict", sess.run([predict],feed_dict={
#                        X: x_ck}))


# result = sess.run(embed)
# hotresult = np.argmax(result, axis=1)
# print(hotresult.shape)
# hotground = np.argmax(y_train, axis=1)
# n = 0
# for i in range(hotresult.shape[0]):
#     if hotresult[i] == hotground[i]:
#         n += 1

# print(n/hotresult.shape[0])

#================================================================
train_size, num_features = x_train.shape
learning_rate = 0.005
training_epochs = 1000
num_labels = y_train.shape[1]
batch_size = 512

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # tf.train.Saver().restore(sess, "./checkpoint/model.ckpt")
    # training_epochs * train_size // batch_size

    for step in range(150):

        # offset = (step * batch_size) % train_size

        # batch_xs = x_train[offset:(offset + batch_size), :]
        # batch_labels = y_train[offset:(offset + batch_size)]

        err = sess.run([loss, opt], feed_dict={
                       X: x_train, Y: y_train})
        print(step, err)

    print("accuracy", accuracy.eval(feed_dict={X: x_train, Y: y_train}))
    print("loss", loss.eval(feed_dict={X: x_train, Y: y_train}))
    print("predict", predict.eval(feed_dict={X: x_ck}))
    
    


    tf.train.Saver().save(sess, "./checkpoint/model.ckpt")
