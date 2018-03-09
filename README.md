# SentimentClassifier

This is an emotional polarity classifier based on SVM and word2vec,is a two-class classifier

filter.py: preprocess the dataset, match only the Chinese text, remove special symbols such as characters and expressions, and convert Traditional to Simplified

txt_write_csv.py: read txt text line by line into cdv file

addLine.py: label the test data set

word2vec.py: use word2vec to train the training set and test set word vectors to facilitate the training of classification models

tran_model.py: using svm to train the classification model

predict.py: using the classification model, reading the document to be tested for classification, comparing the result with the label that was originally marked, and calculating the accuracy

langconv.py: convert Traditional to Simplified Handlers

zh_wiki.py:  Traditional and Simplified Word Banks
