# coding: utf-8
# 用sklearn当中的SVM进行建模
import numpy as np
from sklearn.externals import joblib
from sklearn.svm import SVC

# 加载训练的两个word2vec模型
def get_data():
    train_vecs = np.load(r'..\test\sentiment-analysis\svm_data\train_vecs.npy')
    y_train = np.load(r'..\test\sentiment-analysis\svm_data\y_train.npy')
    test_vecs = np.load(r'..\test\sentiment-analysis\svm_data\test_vecs.npy')
    y_test = np.load(r'..\test\sentiment-analysis\svm_data\y_test.npy')
    return train_vecs, y_train, test_vecs, y_test

# 训练svm模型
def svm_train(train_vecs, y_train, test_vecs, y_test):
    clf = SVC(kernel='rbf', verbose=True)
    clf.fit(train_vecs, y_train)
    joblib.dump(clf,r'..\test\sentiment-analysis\svm_data\svm_model\model.pkl')
    print("准确率:",clf.score(test_vecs, y_test))

    # y_hat = clf.predict(test_vecs)
    # accuracy = getAccuracy(y_test,y_hat)
    # print("准确率:",accuracy)

# #实际的值与计算的值的比值，计算准确率
# def getAccuracy(testSet, predictions):
#     correct = 0
#     for x in range(len(testSet)):
#         if testSet[x] == predictions[x]:
#             correct += 1
#     return (correct/float(len(testSet)))*100.0

train_vecs,y_train,test_vecs,y_test = get_data()
svm_train(train_vecs,y_train,test_vecs,y_test)