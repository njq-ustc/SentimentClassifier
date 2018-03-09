# coding: utf-8
from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
import csv
from sklearn.externals import joblib


# 对每个句子的所有词向量取均值，来生成一个句子的vector
def build_sentence_vector(text, size, imdb_w2v):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += imdb_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

# 构建待预测句子的向量
def get_predict_vecs(words):
    n_dim = 300
    imdb_w2v = Word2Vec.load(r'..\test\sentiment-analysis\svm_data\w2v_model\w2v_model.pkl')
    train_vecs = build_sentence_vector(words, n_dim, imdb_w2v)
    return train_vecs

# 对单个句子进行情感判断
def svm_predict(string):
    words = jieba.lcut(string)
    words_vecs = get_predict_vecs(words)  # 构建测试集的词向量
    # 加载训练好的模型
    clf = joblib.load(r'..\test\sentiment-analysis\svm_data\svm_model\model.pkl')

    result = clf.predict(words_vecs)
    if int(result[0]) == 1:
            #print("positive")
            return "1"
    else:
        #print("negetive")
        return "-1"

count = 0
prodict = 0
# 计算准确度
with open(r'..\test\sentiment-analysis\test.csv',encoding='utf-8') as csvfile:
    online = csv.reader(csvfile)
    for lonly in enumerate(online):
        count = count + 1
        identify = svm_predict(lonly[1][0])
        print(lonly[1][1])
        if identify == lonly[1][1]:
            prodict = prodict + 1

accuracy = prodict/count*100.0
print(accuracy)
