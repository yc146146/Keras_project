from keras.models import Sequential, load_model, model_from_json
import keras as K
import numpy as np
import os
import sys
from gensim.models.doc2vec import Doc2Vec
import jieba
import re
import pandas as pd

def content_to_vector(content):
	# content = re.sub("\\W+", ' ', content)
	content = re.sub("[^\u4e00-\u9fa5^a-z^A-Z^ ]", ' ', content)

	sentence_cut = jieba.cut(content)
	sentence_split = " ".join(sentence_cut).split()
	content = [word for word in sentence_split]
	# print(content)

	model = Doc2Vec.load("./doc2vec/salary.model")
	inferred_vector = model.infer_vector(doc_words=content, alpha=0.025, steps=1000)
	sims = model.docvecs.most_similar([inferred_vector], topn=1)
	# print(sims[0][0])
	job_info_num = sims[0][0]
	return job_info_num

# print(sys.argv)


job_type = sys.argv[1]
address = sys.argv[2]
education = sys.argv[3]
job_info = sys.argv[4]
# print(job_type, address, education, job_info)



address_data = pd.read_csv("./city.csv", "gbk",engine='python')["city"].values
education_data = pd.read_csv("./education.csv", "gbk",engine='python')["education"].values
job_type_data = pd.read_csv("./job_type.csv", "gbk",engine='python')["job_type"].values
# print(education_data)

#数字化学历
for i, element in enumerate(education_data):
	if education == element:
		education = i

#数字化城市名
for i, element in enumerate(address_data):
	if address == element:
		address = i
# #数字化工作标签
for i, element in enumerate(job_type_data):
	if job_type == element:
		job_type = i

job_info = content_to_vector(job_info)

# print(job_type, address, education, job_info)

apply_list = [job_type, address, education, job_info]
# print(apply_list)

model_file = "./cp/model.h5"
model = load_model(model_file)
#验证数据

# print(np.array([apply_list]))
#测试结果
res = model.predict(np.array([apply_list]))

## 四舍五入显示结果
# print("res:", np.round(res))
# print(np.round(res))
print(np.argmax(np.round(res)[0], axis=0))
