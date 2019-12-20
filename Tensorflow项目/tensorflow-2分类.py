import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


# X = np.loadtxt("3data.csv",delimiter="," , usecols=(0) , dtype=float)
# # x_train = x_train.reshape((100, 4))


# Y = np.loadtxt("3data.csv",delimiter="," , usecols=(1) , dtype=float)

x_train1 = 1*np.random.randint(0,100,(100, 2))
x_train2 = [-1, 1]*np.random.randint(0,100,(100, 2))

X = np.concatenate((x_train1, x_train2))

y_train = np.array([[0]*100 + [1]*100])
Y = y_train.reshape((200, 1))

# print(X)
# print(Y)


# 定义维度
lab_dim = 1
input_dim = 2


# 定义占位符数据
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_labels = tf.placeholder(tf.int32, [None, lab_dim])


# 定义变量
W = tf.Variable(tf.random_normal([input_dim, lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim], name="bias"))

# 输出数据
output = tf.nn.sigmoid(tf.matmul(input_features, W)+b)

predict = tf.argmax(output,1,name='predict')


coross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=output, labels=input_labels)

# # 交叉熵
# coross_entropy = -(input_labels*tf.log(output) +
#                    (1-input_labels)*tf.log(1-output))

# 误差
# ser = tf.square(input_labels - output)

_, acc_op = tf.metrics.accuracy(labels=input_labels,predictions=predict)

# 损失函数
loss = tf.reduce_mean(coross_entropy)

# 误差均值
err = tf.reduce_mean(coross_entropy)

# 优化器
optimizer = tf.train.AdamOptimizer(0.04)

train = optimizer.minimize(loss)

maxEpochs = 50
minibatchSize = 25
batch_size = 50
dataset_size = 200

with tf.Session() as sess:
    # 初始化所有变量与占位符
    sess.run(tf.global_variables_initializer())

    for epoch in range(10):
        sumerr = 0
        # 对于每一个batch
        for i in range(np.int32(len(Y)/minibatchSize)):
            # 取出X值
            # x1 = X[i*minibatchSize:(i+1)*minibatchSize]

            # # 取出Y值
            # y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize], [-1, 1])
            # 改变y的数据结构，变成tensor数据
            # tf.reshape(y1, [-1, 1])

            start = (i * batch_size) % dataset_size
            end = min(start + batch_size,dataset_size)

            # print(start)
            # print(end)

            x1 = X[start:end]
            y1 = Y[start:end]

            # print(x1)
            # print(y1)


            # 对相关结果进行计算
            _, lossval, outputval, errval = sess.run([train, loss, output, err], feed_dict={
                                                     input_features: x1, input_labels: y1})

            print(output)
            
            # 计算误差和
            sumerr = sumerr+errval

        print("epoch:", epoch)
        print("cost=", lossval, "err=", sumerr)
        print("acc_op:", acc_op)

    # 结果可视化
    # train_X, train_Y = generate(100, mean, cov, [3.0],True)
    # colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    # plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    # plt.colorbar()


#    x1w1+x2*w2+b=0
#    x2=-x1* w1/w2-b/w2
    # x = np.linspace(-1,8,200)
    # y=-x*(sess.run(W)[0]/sess.run(W)[1])-sess.run(b)/sess.run(W)[1]
    # plt.plot(x,y, label='Fitted line')
    # plt.legend()
    # plt.show()
