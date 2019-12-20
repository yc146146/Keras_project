
# coding:utf-8
'''
使用Softmax进行多类别分类
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#训练数据
x_train1 = np.random.randint(0, 30, (100, 1))
x_train2 = np.random.randint(31, 60, (100, 1))
x_train3 = np.random.randint(60, 100, (100, 1))

xs = np.concatenate((x_train1, x_train2, x_train3))
y_train = np.array([[1, 0, 0]*100 + [0, 1, 0]*100 + [0, 0, 1]*100])
labels = y_train.reshape((300, 3))

xs = xs.astype(float)
labels = labels.astype(float)

# 打乱数据
arr = np.arange(xs.shape[0]) 
np.random.shuffle(arr) 
xs = xs[arr, :] 
labels = labels[arr, :] 

# print(xs)
# print(labels)

#测试数据
x_test1 = np.random.randint(0, 30, (100, 1))
x_test2 = np.random.randint(31, 60, (100, 1))
x_test3 = np.random.randint(60, 100, (100, 1))


xs_test = np.concatenate((x_test1, x_test2, x_test3))
y_test = np.array([[1, 0, 0]*100 + [0, 1, 0]*100 + [0, 0, 1]*100])
labels_test = y_test.reshape((300, 3))

xs_test = xs_test.astype(float)
labels_test = labels_test.astype(float)


train_size, num_features = xs.shape
learning_rate = 0.01
training_epochs = 1000
num_labels = 3
batch_size = 128

X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

y_model = tf.nn.softmax(tf.matmul(X, W) + b)

X_tensor = tf.convert_to_tensor(value=X, dtype=tf.float32)
#全连接层函数
# layer1_test = tf.matmul(X, W) + b
layer1 = tf.layers.dense(inputs = X, units=9, activation = tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0))
layer2 = tf.layers.dense(inputs = layer1, units=3, activation = tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),bias_initializer=tf.constant_initializer(0))

# logits = layer1(X)
# logits = layer2(fx)

# cost = -tf.reduce_sum(Y * tf.log(y_model))
cost = -tf.reduce_mean(Y * tf.log(y_model))

# loss1 = tf.reduce_mean((layer1 - X) ** 2)

# loss1 = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
loss1 = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer2, labels=Y))
# loss1 = -tf.reduce_mean(Y * tf.log(layer1))

# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predict = tf.argmax(y_model, 1, name='predict')

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # training_epochs * train_size // batch_size

    # tf.random.shuffle(xs)
    # print(xs[0:5, :])
    for step in range(1000):

        #每次训练的数量
        offset = (step * batch_size) % train_size
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]

        # print(batch_xs)
        # print(batch_labels)

        # print(batch_xs)
        # print(batch_labels)
        err = sess.run([cost, train_op, loss1], feed_dict={X: batch_xs, Y: batch_labels})
        # print("次数:%d, 丢失:%10.10f"%(step, err))
        print(step, err)

    W_val = sess.run(W)
    print('w', W_val)
    b_val = sess.run(b)
    print('b', b_val)
    print("accuracy", accuracy.eval(feed_dict={X:xs_test, Y:labels_test}))
    print("predict", predict.eval(feed_dict={X:[[ 15.],[40.],[90.]]}))
