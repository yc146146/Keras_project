from keras.layers import Dense, Activation
from keras.models import Sequential
from keras import optimizers
import numpy as np

# collect the data
x_train1 = 100*np.random.random((100, 2))
x_train2 = [-100, 100]*np.random.random((100, 2))
x_train3 = -100*np.random.random((100, 2))
x_train4 = [100, -100]*np.random.random((100, 2))
x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4))
y_train = np.array([[1, 0, 0, 0]*100 + [0, 1, 0, 0]*100 + [0, 0, 1, 0]*100 + [0, 0, 0, 1]*100])
y_train = y_train.reshape((400, 4))

x_test1 = 100*np.random.random((100, 2))
x_test2 = [-100, 100]*np.random.random((100, 2))
x_test3 = -100*np.random.random((100, 2))
x_test4 = [100, -100]*np.random.random((100, 2))
x_test = np.concatenate((x_test1, x_test2, x_test3, x_test4))
y_test = y_train

# set the model
model = Sequential()
model.add(Dense(4, input_dim=2, activation=None, use_bias=False))
model.add(Activation('softmax'))

# compile the model and pick the loss function and optimizer
ada = optimizers.Adagrad(lr=0.1, epsilon=1e-8)
model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

# training the model
ly = []
for i in range(10):
    model.fit(x_train, y_train, batch_size=400, epochs=100, shuffle=False)
    ly.append(model.layers[0].get_weights())

# test the model

score = model.evaluate(x_test, y_test, batch_size=400)
print('loss:', score[0], '\t\taccuracy:', score[1])
for i in range(10):
    print('first weight:\t', ly[i][0][0], '\t\tsecond weight:\t', ly[i][0][1])

#测试结果
res = model.predict(np.array([[15,20],[15,-20],[-15,-20],[-15,20]]))

#四舍五入显示结果
print("res:", np.round(res))

# rounded = [round(x) for x in res]
# print(rounded)
