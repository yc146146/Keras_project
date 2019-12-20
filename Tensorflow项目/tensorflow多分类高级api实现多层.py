
# coding:utf-8
'''
使用Softmax进行多类别分类
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

LAYER_1_NAME = 'layer1'  # 第一层的名字
LAYER_2_NAME = 'layer2'  # 第二层的名字

# 训练数据
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

# 测试数据
x_test1 = np.random.randint(0, 30, (100, 1))
x_test2 = np.random.randint(31, 59, (100, 1))
x_test3 = np.random.randint(60, 100, (100, 1))


xs_test = np.concatenate((x_test1, x_test2, x_test3))
y_test = np.array([[1, 0, 0]*100 + [0, 1, 0]*100 + [0, 0, 1]*100])
labels_test = y_test.reshape((300, 3))

xs_test = xs_test.astype(float)
labels_test = labels_test.astype(float)


train_size, num_features = xs.shape
learning_rate = 0.005
training_epochs = 1000
num_labels = 3
batch_size = 128

X = tf.placeholder("float", shape=[None, num_features])
Y = tf.placeholder("float", shape=[None, num_labels])

# W = tf.Variable(tf.zeros([num_features, num_labels]))
# b = tf.Variable(tf.zeros([num_labels]))

# y_model = tf.nn.softmax(tf.matmul(X, W) + b)


# 全连接层函数
# layer1_test = tf.matmul(X, W) + b
layer1 = tf.layers.dense(inputs=X, units=9, activation=tf.nn.tanh, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name=LAYER_1_NAME)
layer2 = tf.layers.dense(inputs=layer1, units=3, activation=tf.nn.softmax, kernel_initializer=tf.random_normal_initializer(
    mean=0, stddev=0.1), bias_initializer=tf.constant_initializer(0), name=LAYER_2_NAME)


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
        batch_xs = xs[offset:(offset + batch_size), :]
        batch_labels = labels[offset:(offset + batch_size)]

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
    print("accuracy", accuracy.eval(feed_dict={X: xs_test, Y: labels_test}))
    print("predict", predict.eval(feed_dict={X: [[15.], [40.], [90.], [60]]}))

    # 把模型保存成checkpoint
    saver = tf.train.Saver()
    save_path = saver.save(sess, './checkpoint/model.ckpt')
    print("model saved in path: %s" % save_path, flush=True)
    # 读取刚保存的checkpoint
    reader = tf.train.NewCheckpointReader(save_path)
    # weight的名字，是由对应层的名字，加上默认的"kernel"组成的
    weights = reader.get_tensor(LAYER_1_NAME + '/kernel')
    bias = reader.get_tensor(LAYER_1_NAME + '/bias')  # bias的名字
    print("weights",weights)
    print("bias",bias)
    # 如果想打印模型中的所有参数名和参数值的话，把下面几行取消注释
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # for key in var_to_shape_map:
    #     print("tensor name: ", key)
    #     print(reader.get_tensor(key))
