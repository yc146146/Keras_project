import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers

# 此方法只适合做2分类

# collect the training data
x_train = np.array([[1, 5], [2, 7], [9, 14], [6, 10], [8, 21], [16, 19],
                    [5, -1], [7, -2], [14, -9], [10, -6], [21, -8],[19, -16]])
y_train = np.array([[1], [1], [1], [1], [1], [1],
                    [0], [0], [0], [0], [0], [0]])


# x_train = np.array([[1, 5], [2, 7], [9, 14], [-6, 10], [-8, 21], [-16, 19],
#                     [5, -1], [7, -2], [14, -9], [10, -6], [21, -8],[19, -16]])
# y_train = np.array([[1], [1], [1], [2], [2], [2],
#                     [0], [0], [0], [0], [0], [0]])


print(x_train)
print(y_train)

# 创建训练层 维度为2
model = Sequential()
model.add(Dense(1, input_dim=2, activation=None, use_bias=False))

#sigmoid的输出在0和1之间，我们在二分类任务中，采用sigmoid的输出的是事件概率
model.add(Activation('sigmoid'))

# 优化器
ada = optimizers.Adagrad(lr=0.1, epsilon=1e-8)
model.compile(optimizer=ada, loss='binary_crossentropy', metrics=['accuracy'])

# 训练
print('training')
model.fit(x_train, y_train, batch_size=4, epochs=100, shuffle=True)
model.fit(x_train, y_train, batch_size=12, epochs=100, shuffle=True)

# 测试
test_ans = model.predict(np.array([[2, 20], [20, -2]]), batch_size=2)
print('model_weight')
print(model.layers[0].get_weights())
print('ans')
print(test_ans)