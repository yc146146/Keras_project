from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
import numpy as np



x_train1 = 1*np.random.randint(0,100,(100, 2))
x_train2 = [-1, 1]*np.random.randint(0,100,(100, 2))
x_train3 = -1*np.random.randint(0,100,(100, 2))
x_train4 = [1, -1]*np.random.randint(0,100,(100, 2))

x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4))
y_train = np.array([[1, 0, 0, 0]*100 + [0, 1, 0, 0]*100 + [0, 0, 1, 0]*100 + [0, 0, 0, 1]*100])
y_train = y_train.reshape((400, 4))


x_test1 = 1*np.random.randint(0,100,(100, 2))
x_test2 = [-1, 1]*np.random.randint(0,100,(100, 2))
x_test3 = -1*np.random.randint(0,100,(100, 2))
x_test4 = [1, -1]*np.random.randint(0,100,(100, 2))

x_test = np.concatenate((x_test1, x_test2, x_test3, x_test4))
y_test = y_train

print(x_test.shape)
print(y_test.shape)

# model = Sequential()
# model.add(Dense(4, input_dim=2, activation=None, use_bias=False))
# model.add(Activation('softmax'))

# ada = optimizers.Adagrad(lr=0.1, epsilon=1e-8)
# model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=400, epochs=100, shuffle=False)

# score = model.evaluate(x_test, y_test, batch_size=400)
# print('loss:', score[0], '\t\taccuracy:', score[1])


# #测试结果
# res = model.predict(np.array([[115,220],[135,-240],[-155,-260],[-175,280]]))

# #四舍五入显示结果
# print("res:", np.round(res))

# print("res:", res)
