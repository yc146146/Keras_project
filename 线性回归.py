from keras.models import Sequential
from keras.layers import Dense
import numpy as np


x_train = [1, 2, 3, 4, 5]
y_train = [2, 3, 4, 5, 6]
x_train = np.array(x_train)
y_train = np.array(y_train)
# define a model
model = Sequential()
model.add(Dense(
    input_dim=1,
    units=1,
    use_bias=True,
))
model.add(Dense(
    input_dim=1,
    units=1,
    use_bias=True,
))
model.compile(loss='mse', optimizer='sgd')

# training the model
model.fit(x_train, y_train, batch_size=6, epochs=40, initial_epoch=0)

# test the model
score = model.evaluate(x_train, y_train, batch_size=5)
test_data = model.predict(np.array([[5]]), batch_size=1)

#但数据测试
print(test_data)
print(model.layers[0].get_weights(), '\n', model.layers[1].get_weights())
print(score)