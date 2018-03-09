# coding: utf-8
# 用gensim去做word2vec的处理
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba

#  载入数据，做预处理(分词)，切分训练集与测试集
def load_file_and_preprocessing():
    neg = pd.read_csv(r'..\test\sentiment-analysis\positive.csv', header=None,encoding='utf-8')
    pos = pd.read_csv(r'..\test\sentiment-analysis\negative.csv', header=None,encoding='utf-8')

    # 定义分词函数
    cw = lambda x: list(jieba.cut(x))

    # 新增一列 word ,存放分好词的评论，pos[0]代表表格第一列
    pos['words'] = pos[0].apply(cw)
    neg['words'] = neg[0].apply(cw)

    # np.ones(len(pos)) 新建一个长度为len(pos)的数组并初始化元素全为1来标注好评
    # np.concatenate（）连接数组
    # axis=0 向下执行方法 axis=1向右执行方法
    y = np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0)

    # 将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos['words'], neg['words'])), y, test_size=0.2)

    np.save(r'..\test\sentiment-analysis\svm_data\y_train.npy', y_train)
    np.save(r'..\test\sentiment-analysis\svm_data\y_test.npy', y_test)
    return x_train, x_test


# 计算词向量
def get_train_vecs(x_train, x_test):
    n_dim = 300
    # 初始化模型和词表
    imdb_w2v = Word2Vec(x_train, size=n_dim, min_count=10)
    train_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_train])

    np.save(r'..\test\sentiment-analysis\svm_data\train_vecs.npy', train_vecs)
    print(train_vecs.shape)
    # 在测试集上训练
    imdb_w2v.train(x_test,
                   total_examples=imdb_w2v.corpus_count,
                   epochs=imdb_w2v.iter)

    imdb_w2v.save(r'..\test\sentiment-analysis\svm_data\w2v_model\w2v_model.pkl')
    test_vecs = np.concatenate([build_sentence_vector(z, n_dim, imdb_w2v) for z in x_test])
    np.save(r'..\test\sentiment-analysis\svm_data\test_vecs.npy', test_vecs)
    print(test_vecs.shape)


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

x_train,x_test = load_file_and_preprocessing()
get_train_vecs(x_train,x_test)