from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
import numpy as np


#构建训练分数数据

# x_train1 = np.random.randint(0,29,(100,1))

# x_train2 = np.random.randint(30,59,(100,1))

# x_train3 = np.random.randint(60,79,(100,1))

# x_train4 = np.random.randint(80,100,(100,1))

# x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4))

x_train = np.array([[2,10], [3,20], [4,30], [5,40],
                    [-6,70], [-7,80],[-8,90], [-9,100],
                    [2,-10], [3,-20], [4,-30],[5,-40],
                    [-6,-50], [-7,-80], [-8,-90],[-9,-100]])

y_train = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],
                    [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
                    [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
                    [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]
            ])

x_test = np.array([[1, 5], [2, 7], [9, 14], [6, 10],
                    [-8, 21], [-16, 19],[-5, 1], [-7, 2],
                    [8, -21], [16, -19],[5, -1], [7, -2],
                    [-14, -9], [-10, -6], [-21, -8],[-19, -16]])

y_test = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],
                    [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
                    [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
                    [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]
            ])

# x_train = np.array([[100,123], [156,177], [199,189], [165,132],
#                     [1000,1234], [1235,1897],[1357,1879], [1358,1999],
#                     [10000,12345], [13256,12587], [14568,18798],[19999,15678],
#                     [100000,125468], [125578,125789], [154687,123578],[199999,154876]])

# y_train = np.array([[1,0,0,0],[1,0,0,0],[1,0,0,0],[1,0,0,0],
#                     [0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0],
#                     [0,0,1,0], [0,0,1,0], [0,0,1,0], [0,0,1,0],
#                     [0,0,0,1], [0,0,0,1], [0,0,0,1], [0,0,0,1]
#             ])

#构建测试目标数据
# 分别为 不及格 良 优
# y_train = np.array([[1, 0, 0, 0]*100 + [0, 1, 0, 0]*100 + [0, 0, 1, 0]*100 + [0, 0, 1, 1]*100])
# y_train = y_train.reshape((400, 4))


# x_test1 = np.random.randint(0,29,(100,1))

# x_test2 = np.random.randint(30,59,(100,1))

# x_test3 = np.random.randint(60,79,(100,1))

# x_test4 = np.random.randint(80,100,(100,1))

# x_test = np.concatenate((x_test1, x_test2, x_test3, x_test4))

# y_test = np.array([[1, 0, 0, 0]*100 + [0, 1, 0, 0]*100 + [0, 0, 1, 0]*100 + [0, 0, 1, 1]*100])
# y_test = y_test.reshape((400, 4))


# print(x_test)
# print(y_test)

# print(x_test.shape)
# print(y_test.shape)

# 创建训练模型
model = Sequential()
model.add(Dense(units=4, input_dim=2, activation=None, use_bias=False))
# model.add(Activation("softmax"))
model.add(Activation('softmax'))


ada = optimizers.Adam(lr=0.01, epsilon=1e-8)
model.compile(optimizer=ada, loss="categorical_crossentropy", metrics=["accuracy"])


# 训练
model.fit(x_train, y_train, batch_size=512, epochs=10000, shuffle=True)

score = model.evaluate(x_train, y_train, batch_size=512)

print('loss:', score[0], '\t\taccuracy:', score[1])


#测试结果
res = model.predict(np.array([[100,123],[1000,1234],[10000,12345],[100000,125468]]))

#四舍五入显示结果
print("res:", np.round(res))