
# coding:utf-8
'''
使用Softmax进行多类别分类
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# x_train1 = np.random.randint(0, 39, (100, 1))
# x_train2 = np.random.randint(40, 59, (100, 1))
# x_train3 = np.random.randint(60, 79, (100, 1))
# x_train4 = np.random.randint(80, 100, (100, 1))

x_train1 = np.random.randint(0, 99, (100, 1))
x_train2 = np.random.randint(100, 999, (100, 1))
x_train3 = np.random.randint(1000, 9999, (100, 1))
x_train4 = np.random.randint(10000, 99999, (100, 1))

xs = np.concatenate((x_train1, x_train2, x_train3, x_train4))
y_train = np.array([[1, 0, 0, 0]*100 + [0, 1, 0, 0]*100 + [0, 0, 1, 0]*100 + [0, 0, 0, 1]*100])
labels = y_train.reshape((400, 4))

xs = xs.astype(float)
labels = labels.astype(float)


# 打乱数据
arr = np.arange(xs.shape[0])
np.random.shuffle(arr)
xs = xs[arr, :]
labels = labels[arr, :]

# print(xs)
# print(labels)

# x_test1 = np.random.randint(0, 39, (100, 1))
# x_test2 = np.random.randint(40, 59, (100, 1))
# x_test3 = np.random.randint(60, 79, (100, 1))
# x_test4 = np.random.randint(80, 100, (100, 1))

# xs_test = np.concatenate((x_test1, x_test2, x_test3, x_test4))
# y_test = np.array([[1, 0, 0, 0]*100 + [0, 1, 0, 0]*100 + [0, 0, 1, 0]*100 + [0, 0, 0, 1]*100])
# labels_test = y_test.reshape((400, 4))

# xs_test = xs_test.astype(float)
# labels_test = labels_test.astype(float)


train_size, num_features = xs.shape
#调整学习率 提高准确率
learning_rate = 0.0001
training_epochs = 1000
num_labels = 4
batch_size = 100

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

#隐层 relu tanh
# h1 = fcn_layer(X, num_features, 16, tf.nn.tanh)
# #全连接输出层 Sigmoid Softmax 
# y_model = fcn_layer(h1, 16, 4, tf.nn.softmax)


# cost = -tf.reduce_sum(Y * tf.log(y_model))
cost = -tf.reduce_mean(Y * tf.log(y_model))

# cost =-tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=Y))

# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


predict = tf.argmax(y_model, 1, name='predict')

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # training_epochs * train_size // batch_size

    for step in range(20000):

        offset = (step * batch_size) % train_size

        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]

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
    print("accuracy", accuracy.eval(feed_dict={X: xs, Y: labels}))
    print("predict", predict.eval(feed_dict={X: [[15.], [555.], [6542.], [88888.], [1234.]]}))
