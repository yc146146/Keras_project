
# coding:utf-8
'''
使用Softmax进行多类别分类
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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

print(xs)
print(labels)



train_size, num_features = xs.shape
learning_rate = 0.01
training_epochs = 1000
num_labels = 3
batch_size = 100

X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

W = tf.Variable(tf.zeros([num_features, num_labels]))
b = tf.Variable(tf.zeros([num_labels]))

y_model = tf.nn.softmax(tf.matmul(X, W) + b)

cost = -tf.reduce_sum(Y * tf.log(y_model))

# cost =-tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits_v2(logits=X, labels=Y))

# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


predict = tf.argmax(y_model, 1, name='predict')

correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     for step in range(training_epochs * train_size // batch_size):
#         offset = (step * batch_size) % train_size
#         batch_xs = xs[offset:(offset + batch_size), :]
#         batch_labels = labels[offset:(offset + batch_size)]
#         err, _ = sess.run([cost, train_op], feed_dict={X:batch_xs, Y:batch_labels})
#         print (step, err)
#     W_val = sess.run(W)
#     print('w', W_val)
#     b_val = sess.run(b)
#     print('b', b_val)
#     print("accuracy", accuracy.eval(feed_dict={X:batch_xs, Y:batch_labels}))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    # training_epochs * train_size // batch_size

    # tf.random.shuffle(xs)
    # print(xs[0:5, :])
    for step in range(1000):

        offset = (step * batch_size) % train_size
        # batch_xs = xs[0:5, :]
        # batch_labels = labels[0:5]
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]

        # print(batch_xs)
        # print(batch_labels)

        # print(batch_xs)
        # print(batch_labels)
        err = sess.run([cost, train_op], feed_dict={X: batch_xs, Y: batch_labels})
        print(step, err)
    W_val = sess.run(W)
    print('w', W_val)
    b_val = sess.run(b)
    print('b', b_val)
    print("accuracy", accuracy.eval(feed_dict={X:batch_xs, Y:batch_labels}))
    print("predict", predict.eval(feed_dict={X:[[ 15.],[40.],[90.]]}))
