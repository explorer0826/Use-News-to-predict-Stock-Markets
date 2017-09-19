#!/usr/bin/pytthon
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from datetime import date

data = pd.read_csv('Combined_News_DJIA.csv')
data.head()

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

#把每条新闻做成一个单独的句子，集合在一起
X_train = train[train.columns[2:]]
corpus = X_train.values.flatten().astype(str)
X_train = X_train.values.astype(str)
X_train = np.array([' '.join(x) for x in X_train])
X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])
y_train = train['Label'].values
y_test = test['Label'].values
# print (corpus[:3])
# print (X_train[:1])
# print (y_train[:5])

from nltk.tokenize import word_tokenize

corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]

print (X_train[:2])
print (corpus[:2])

#文本预处理
from nltk.corpus import stopwords
stop = stopwords.words('english')
import re

def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    word= word.lower()
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True

# 把上面的方法综合起来
def preprocessing(sen):
    res = []
    for word in sen:
        if check(word):
            # 去除python里面byte存str时候留下的标识
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res

corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]
# print(corpus[553])
# print(X_train[523])

from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
# print (model['ok'])
vocab = model.vocab

def get_vector(word_list):
    # 建立一个全是0的array
    res =np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res/count

# print (get_vector(['hello', 'from', 'the', 'other', 'side']))

#为了之后内容的方便，我先把之前我们处理好的wordlist给存下来。
wordlist_train = X_train
wordlist_test = X_test
X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]
print(X_train[10])

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

params = [0.1,0.5,1,3,5,7,10,12,16,20,25,30,35,40]
test_scores = []
for param in params:
    clf = SVR(gamma=param)
    test_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
    test_scores.append(np.mean(test_score))

import matplotlib.pyplot as plt
plt.plot(params, test_scores)
plt.title("Param vs CV AUC Score")
plt.show()

def transform_to_matrix(x, padding_size=256, vec_size=128):
    res = []
    for sen in x:
        matrix = []
        for i in range(padding_size):
            try:
                matrix.append(model[sen[i]].tolist())
            except:
                # 两种except情况
                # 1. 这个单词找不到
                # 2. sen没那么长
                # 不管哪种情况，直接贴上全是0的vec
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res

X_train = transform_to_matrix(wordlist_train)
X_test = transform_to_matrix(wordlist_test)

print(X_train[123])

# 转换成np数组格式，便于处理
X_train = np.array(X_train)
X_test = np.array(X_test)

# print(X_train.shape)
# print(X_test.shape)

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])

# print(X_train.shape)
# print(X_test.shape)

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten

# 设置参数
batch_size = 32
n_filter = 16
filter_length = 4
nb_epoch = 5
n_pool = 2

# 新建一个sequential的模型
model = Sequential()
model.add(Convolution2D(n_filter,filter_length,filter_length,
                        input_shape=(1, 256, 128)))
model.add(Activation('relu'))
model.add(Convolution2D(n_filter,filter_length,filter_length))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
# 后面接上一个ANN
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))
# compile模型
model.compile(loss='mse',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])