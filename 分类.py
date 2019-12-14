import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers


# collect the training data
x_train = np.array([[1, 5], [2, 7], [9, 14], [6, 10], [8, 21], [16, 19],
                    [5, 1], [7, 2], [14, 9], [10, 6], [21, 8],[19, 16]])
y_train = np.array([[1], [1], [1], [1], [1], [1],
                    [0], [0], [0], [0], [0], [0]])
print(x_train)
print(y_train)

# design the model
model = Sequential()
model.add(Dense(1, input_dim=2, activation=None, use_bias=False))
model.add(Activation('sigmoid'))

# compile the model and pick the optimizer and loss function
ada = optimizers.Adagrad(lr=0.1, epsilon=1e-8)
model.compile(optimizer=ada, loss='binary_crossentropy', metrics=['accuracy'])

# training the model
print('training')
model.fit(x_train, y_train, batch_size=4, epochs=100, shuffle=True)
model.fit(x_train, y_train, batch_size=12, epochs=100, shuffle=True)

# test the model
test_ans = model.predict(np.array([[2, 20], [20, 2]]), batch_size=2)
print('model_weight')
print(model.layers[0].get_weights())
print('ans')
print(test_ans)