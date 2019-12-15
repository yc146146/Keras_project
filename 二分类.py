from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import optimizers
import keras as K
import numpy as np



x_train1 = 1*np.random.randint(1,100,(100, 1))
x_train2 = 100*np.random.randint(1,100,(100, 1))


x_train = np.concatenate((x_train1, x_train2))
y_train = np.array([[1]*100 + [0]*100])
y_train = y_train.reshape((200, 1))


x_test1 = 1*np.random.randint(1,100,(100, 1))
x_test2 = 100*np.random.randint(1,100,(100, 1))


x_test = np.concatenate((x_test1, x_test2))
y_test = y_train

# print(x_test)

init = K.initializers.glorot_uniform(seed=1)

model = Sequential()
# model.add(Dense(16, activation="relu", input_dim=1, kernel_initializer=init))
# model.add(Dense(8, activation="relu", kernel_initializer=init))
model.add(Dense(1, kernel_initializer=init,  input_dim=1))
model.add(Activation('sigmoid'))

ada = optimizers.Adam(lr=0.01, epsilon=1e-8)
model.compile(optimizer=ada, loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=128, epochs=1000, shuffle=False)

score = model.evaluate(x_test, y_test, batch_size=128)
print('loss:', score[0], '\t\taccuracy:', score[1])


#测试结果
res = model.predict(np.array([[50], [3500], [99],[1256],[2222]]))

#四舍五入显示结果
print("res:", np.round(res))

print("res:", res)
