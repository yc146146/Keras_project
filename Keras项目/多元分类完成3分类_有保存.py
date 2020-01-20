from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import optimizers
import keras as K
import numpy as np

from sklearn.preprocessing import LabelBinarizer

model_file = "./cp/多分类3.h5"

# 训练数据
x_train1 = np.random.randint(0, 29, (1000, 1))
x_train2 = np.random.randint(30, 59, (1000, 1))
x_train3 = np.random.randint(60, 100, (1000, 1))

xs = np.concatenate((x_train1, x_train2, x_train3))
y_train = np.array([[1, 0, 0]*1000 + [0, 1, 0]*1000 + [0, 0, 1]*1000])
labels = y_train.reshape((3000, 3))

x_train = xs.astype(float)
y_train = labels.astype(float)

# 打乱数据
# 学习效率高
arr = np.arange(x_train.shape[0])
np.random.shuffle(arr)
x_train = x_train[arr, :]
y_train = labels[arr, :]

# print(x_train)
# print(y_train)



model = Sequential()
# model.add(Dense(16, activation="relu", input_dim=1, kernel_initializer=init))
# model.add(Dense(8, activation="relu", kernel_initializer=init))
model.add(Dense(3, kernel_initializer='random_uniform', bias_initializer='random_uniform', input_dim=1))
model.add(Activation('softmax'))

ada = optimizers.Adam(lr=0.01, epsilon=1e-8)
model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=1000, shuffle=True)

score = model.evaluate(x_train, y_train, batch_size=64)
print('loss:', score[0], '\t\taccuracy:', score[1])

model.save(model_file)

#测试结果
res = model.predict(np.array([[22],[44],[66]]))

#四舍五入显示结果
print("res:\n", np.round(res))

print("res:\n", res)
