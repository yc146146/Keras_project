import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, Flatten, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow import keras

model_file = "/ck/test.h5"

data_list = pd.read_csv("../data/python_new.csv",encoding="gbk")
# data_list = pd.read_csv("../data/python_new.csv",encoding="gbk")
# print(data_list.head())
data_ck_list = pd.read_csv("../data/python_ck.csv",encoding="gbk")

x_train = data_list.drop(["salary"], axis=1)
# print(data_list.shape)
# print(x_train.shape)

# print(x_train["education"])

# education_data = x_train["education"].drop_duplicates(keep='first',inplace=False)
education_data = set(x_train["education"].values)
address_data = set(x_train["address"].values)
job_type_data = set(x_train["job_type"].values)
# print(education_data)

file = open("./数据化文件.txt", "w")

file.write("\n学历\n")
#数字化学历
for i, element in enumerate(education_data):
    # print(i,element)
    x_train["education"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

file.write("\n城市名\n")
#数字化城市名
for i, element in enumerate(address_data):
    # print(i,element)
    x_train["address"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

file.write("\n工作标签\n")
#数字化工作标签
for i, element in enumerate(job_type_data):
    # print(i,element)
    x_train["job_type"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

# 归一化
# x_train["education"] = (x_train["education"]-x_train["education"].min())/(x_train["education"].max()-x_train["education"].min())
# x_train["address"] = (x_train["address"]-x_train["address"].min())/(x_train["address"].max()-x_train["address"].min())
# x_train["job_type"] = (x_train["job_type"]-x_train["job_type"].min())/(x_train["job_type"].max()-x_train["job_type"].min())


# print(x_train["education"].head())
# print(x_train["address"].head())
# print(x_train["job_type"].head())

#设置 x的训练集
# print(x_train)
x_train = x_train.values
# print(x_train)

y_train = data_list[["salary"]]
y_train = pd.get_dummies(y_train)

#one-hot查看列名
# print(list(y_train))
file.write(str(list(y_train)))
file.close()

# 提取one-hot 值
# print(y_train.values)

y_train = y_train.values
# print(y_train.shape[1])

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2)

#打乱
np.random.seed(200)
np.random.shuffle(x_train)
np.random.seed(200)
np.random.shuffle(y_train)
np.random.seed(200)
np.random.shuffle(x_test)
np.random.seed(200)
np.random.shuffle(y_test)

#验证数据
x_ck = data_ck_list.drop(["salary"], axis=1)

#数字化学历
for i, element in enumerate(education_data):
    # print(i,element)
    x_ck["education"].replace(element, i, inplace=True)

#数字化城市名
for i, element in enumerate(address_data):
    # print(i,element)
    x_ck["address"].replace(element, i, inplace=True)

#数字化工作标签
for i, element in enumerate(job_type_data):
    # print(i,element)
    x_ck["job_type"].replace(element, i, inplace=True)

x_ck = x_ck.values

class DenseLayer(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(DenseLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="kernel", shape = (input_shape[1], self.units),
            initializer = "uniform",
            trainable = True)

        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer = "zeros", trainable=True)

        super(DenseLayer, self).build(input_shape)

    def call(self, x):

        return self.activation(x @ self.kernel + self.bias)

# tanh relu
#创建学习模型
model = Sequential()


# model.add(Dense(512, input_dim=903, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(256, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dropout(0.5))
# model.add(Dense(128, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dense(64, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
model.add(Dropout(0.5))
# model.add(Dense(32, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))
# model.add(Dense(y_train.shape[1]*2, activation="relu", kernel_initializer='random_uniform', bias_initializer='random_uniform' ))

model.add(Dense(y_train.shape[1], kernel_initializer='random_uniform', bias_initializer='random_uniform'))

# model.add(DenseLayer(y_train.shape[1]))

# model.add(Dense(y_train.shape[1],input_dim=903, kernel_initializer='random_uniform', bias_initializer='random_uniform'))
model.add(Activation('softmax'))


optimizer = optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False)



#自定义训练
#1.batch 遍历训练集
# 自动求导
#2.epoch 结束验证

#循环次数
epochs=10

#每次提取大小
batch_size=512

#训练多少步
steps_per_epoch = len(x_train)//batch_size

#损失函数
# metric = keras.metrics.MeanSquaredError()
metric = keras.metrics.CategoricalCrossentropy()

# 随机提取数据
def random_batch(x, y, batch_size=512):
    idx = np.random.randint(0, len(x), size=batch_size)
    return x[idx], y[idx]

#开始训练
for epoch in range(epochs):
    #重置
    metric.reset_states()
    for step in range(steps_per_epoch):
        x_batch, y_batch = random_batch(x_train, y_train, batch_size)

        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_batch, y_pred))

            metric(y_batch, y_pred)
        #求导
        grads = tape.gradient(loss, model.variables)
        #绑定
        grads_and_vars = zip(grads, model.variables)
        optimizer.apply_gradients(grads_and_vars)
        print("\rEpoch",epoch,"train loss:", metric.result().numpy(), end="")

    #测试集
    y_valid_pred = model(x_test)
    valid_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(y_test, y_valid_pred))
    acc = tf.reduce_mean(keras.metrics.categorical_accuracy( y_test,y_valid_pred))
    print("\t","valid loss:", valid_loss.numpy(),"\t acc:",acc.numpy())




# model.save(model_file)

#测试结果
res = model.predict(np.array(x_ck))

## 四舍五入显示结果
print("res:", np.round(res))

print("res:", res)