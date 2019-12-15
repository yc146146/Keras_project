from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras import optimizers
import keras as K
import numpy as np

from sklearn.preprocessing import LabelBinarizer

x_train = np.loadtxt("3data.csv",delimiter="," , usecols=(0) , dtype=float)
# x_train = x_train.reshape((100, 4))
print(x_train)

y_train = np.loadtxt("3data.csv",delimiter="," , usecols=(1) , dtype=float)




# 特征矩阵
featureList=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]

# labelList=[1, 2, 3, 4]

lb = LabelBinarizer()
y_train=lb.fit_transform(y_train)


# print(y_train)


# x_train1 = 1*np.random.randint(1,100,(100, 4))
# x_train2 = 100*np.random.randint(1,100,(100, 4))
# x_train3 = 100*100*np.random.randint(1,100,(100, 4))
# x_train4 = 100*100*100*np.random.randint(1,100,(100, 4))

# x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4))
# y_train = np.array([[1, 0, 0, 0]*100 + [0, 1, 0, 0]*100 + [0, 0, 1, 0]*100 + [0, 0, 0, 1]*100])
# y_train = y_train.reshape((400, 4))


# x_test1 = 1*np.random.randint(1,100,(100, 4))
# x_test2 = 100*np.random.randint(1,100,(100, 4))
# x_test3 = 100*100*np.random.randint(1,100,(100, 4))
# x_test4 = 100*100*100*np.random.randint(1,100,(100, 4))

# x_test = np.concatenate((x_test1, x_test2, x_test3, x_test4))
# y_test = y_train

# print(x_train)

init = K.initializers.glorot_uniform(seed=1)

model = Sequential()
# model.add(Dense(16, activation="relu", input_dim=1, kernel_initializer=init))
# model.add(Dense(8, activation="relu", kernel_initializer=init))
model.add(Dense(4, kernel_initializer=init, input_dim=1))
model.add(Activation('softmax'))

ada = optimizers.Adam(lr=0.001, epsilon=1e-8)
model.compile(optimizer=ada, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=4096, epochs=10000, shuffle=True)

score = model.evaluate(x_train, y_train, batch_size=4096)
print('loss:', score[0], '\t\taccuracy:', score[1])


#测试结果
res = model.predict(np.array([[50],[1050],[400050]]))

#四舍五入显示结果
print("res:", np.round(res))

print("res:", res)
