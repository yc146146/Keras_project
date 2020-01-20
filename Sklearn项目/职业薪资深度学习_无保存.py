import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

model_file = "/ck/test.h5"

data_list = pd.read_csv("../data/python_12_no0.csv",encoding="gbk")
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
x_train["education"] = (x_train["education"]-x_train["education"].min())/(x_train["education"].max()-x_train["education"].min())
x_train["address"] = (x_train["address"]-x_train["address"].min())/(x_train["address"].max()-x_train["address"].min())
# x_train["job_type"] = (x_train["job_type"]-x_train["job_type"].min())/(x_train["job_type"].max()-x_train["job_type"].min())

# x_train["education"] = (x_train["education"]-x_train["education"].mean())/x_train["education"].std()
# x_train["address"] = (x_train["address"]-x_train["address"].mean())/x_train["address"].std()
# x_train["job_type"] = (x_train["job_type"]-x_train["job_type"].min())/(x_train["job_type"].max()-x_train["job_type"].min())


# print(x_train["education"].head())
# print(x_train["address"].head())
# print(x_train["job_type"].head())

#设置 x的训练集
# print(x_train)
X = x_train.values
# print(x_train)

salary_data = ['1000-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000', '25000-30000','30000以上' ]

y_train = data_list[["salary"]].copy()

file.write("\n金额\n")

#数字化目标标签
for i, element in enumerate(salary_data):
    # print(temp)
    # print(i,element)
    y_train["salary"].replace(element, i, inplace=True)
    file.write("数字:"+str(i)+"名称:"+element+"\n")

# print(y_train["salary"])

y_train = y_train["salary"]
y_train = y_train.astype("float32")
# print(y_train.values)

# y_train = data_list[["salary"]]
# y_train = pd.get_dummies(y_train)

#one-hot查看列名
# print(list(y_train))
# file.write(str(list(y_train)))
file.close()

# 提取one-hot 值
# print(y_train.values)

Y = y_train.values




# print(x_train)
# print(y_train)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)


#打乱
# np.random.seed(200)
# np.random.shuffle(x_train)
# np.random.seed(200)
# np.random.shuffle(y_train)
# np.random.seed(200)
# np.random.shuffle(x_test)
# np.random.seed(200)
# np.random.shuffle(y_test)

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



from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


#将数据标准化
# ss = StandardScaler()
# x_train = ss.fit_transform(x_train)
# x_test = ss.fit_transform(x_test)

# model = KNeighborsClassifier()
# model = RandomForestClassifier(n_estimators=16)
# model = tree.DecisionTreeClassifier()

# model = QuadraticDiscriminantAnalysis()

# model = GaussianNB()
model = GradientBoostingClassifier(n_estimators=200)
# model = AdaBoostClassifier()
# model = LinearDiscriminantAnalysis()
# model = SVC(kernel='rbf', probability=True)
# model = MultinomialNB(alpha=0.01)



model.fit(x_train, y_train)


y_pred = model.predict(x_test)
# accuracy = np.mean(predict_data==y_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)

score = model.score(x_test, y_test)
print(score)

# print(precision_score(y_test, y_pred, average=None))
# print(recall_score(y_test, y_pred, average=None))

# print(f1_score(y_test, y_pred, average=None))

res = model.predict(x_ck)
print(res)
