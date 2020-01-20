from sklearn.preprocessing import LabelBinarizer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


import numpy as np


# 训练数据
x_train1 = np.random.randint(0, 99, (100, 1))
x_train2 = np.random.randint(100, 999, (100, 1))
x_train3 = np.random.randint(1000, 9999, (100, 1))

xs = np.concatenate((x_train1, x_train2, x_train3))
# y_train = np.array([[1,0,0]*100 + [0,1,0]*100 + [0,0,1]*100])
y_train = np.array([[0]*100 + [1]*100 + [2]*100])


labels = y_train.reshape((300, 1))
labels = y_train[0]

x = xs.astype(float)
y = labels.astype(float)

# 打乱数据
# 学习效率高
# arr = np.arange(x_train.shape[0])
# np.random.shuffle(arr)
# x_train = x_train[arr, :]
# y_train = labels[arr, :]



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)


kc_model = KNeighborsClassifier()
kc_model.fit(x_train, y_train)


predict_data = kc_model.predict(x_test)
accuracy = np.mean(predict_data==y_test)
print(accuracy)

res = kc_model.predict([[11],[222],[3333]])
print(res)
