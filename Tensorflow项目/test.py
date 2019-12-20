import tensorflow as tf

# _logits = [[2., 3., 4.], [-20., 30., 40.], [-200., -300., 400.]]

#_logits = [[7.78738701e+00,2.47106420e-01], [4.87610424e+00, 3.66376356e+00], [-7.38594458e-01, 7.18952299e-01]]
_logits = [[0, 3.], [0., 30.], [-200., -300.]]

_labels = [0, 1, 2]

W = tf.Variable(tf.zeros([2, 3]))  
b = tf.Variable(tf.zeros([3])) 




with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # Softmax本身的算法很简单，就是把所有值用e的n次方计算出来，求和后算每个值占的比率，保证总和为1，一般我们可以认为Softmax出来的就是confidence也就是概率
    # [[0.35591307  0.32204348  0.32204348]
    #  [0.32893291  0.40175956  0.26930749]
    #  [0.30060959  0.33222499  0.36716539]]
    # print(sess.run(tf.nn.softmax(_logits)))
    
    print(W)
    y_model = tf.matmul(_logits, W)+b
    print(sess.run(y_model))


    # Y = tf.one_hot(_labels,depth=len(_labels))

    # #print(sess.run(Y))

    # cost = sess.run(-tf.reduce_sum(Y * tf.log(tf.nn.softmax(y_model))))

    # print(cost)

    # 分布计算交叉熵
    # print(sess.run(-tf.reduce_sum(Y * tf.log(tf.nn.softmax(_logits)))))

    # 对 _logits 进行降维处理，返回每一维的合计
    # [1.  1.  0.99999994]
    # print(sess.run(tf.reduce_sum(tf.nn.softmax(_logits), 1)))

    # 传入的 lables 需要先进行 独热编码 处理。
    # 整体计算交叉熵
    #loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_model, labels=tf.one_hot(_labels,depth=len(_labels))))
    # [ 1.03306878  0.91190147  1.00194287]
    #print(sess.run(loss))

# import tensorflow as tf

# x=tf.Variable(4.0,dtype=tf.float32)

# y=tf.pow(x-1,2.0)

# # 梯度下降，学习率设置为0.25
# opti=tf.train.GradientDescentOptimizer(0.25).minimize(y)

# session=tf.Session()
# session.run(tf.global_variables_initializer())

# # 三次迭代
# for i in range(3):
#     session.run(opti)
    
#     print(session.run(x))
#     print(session.run(y))


