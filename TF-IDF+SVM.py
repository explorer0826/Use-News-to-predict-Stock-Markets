#!/usr/bin/python
#-*- coding;utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from datetime import date

pd.set_option('display.width',200)
data = pd.read_csv('Combined_News_DJIA.csv')
# print (data.head())
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
# print (data.head())

train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']
feature_extraction = TfidfVectorizer()
X_train = feature_extraction.fit_transform(train["combined_news"].values)
X_test = feature_extraction.transform(test["combined_news"].values)
y_train = train['Label'].values
y_test = test['Label'].values

print (u'SVM预测')

clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))


X_train = train["combined_news"].str.lower().str.replace('"', '').str.replace("'", '').str.split()
X_test = test["combined_news"].str.lower().str.replace('"', '').str.replace("'", '').str.split()
print(X_test[1611])

from nltk.corpus import stopwords
stop = stopwords.words('english')

import re
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

#前几步做一个综合处理
def check(word):
    """
    如果需要这个单词，则True
    如果应该去除，则False
    """
    if word in stop:
        return False
    elif hasNumbers(word):
        return False
    else:
        return True

#预处理完成后添加到训练测试集上
X_train = X_train.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
X_test = X_test.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
# print(X_test[1])

X_train = X_train.apply(lambda x: ' '.join(x))
X_test = X_test.apply(lambda x: ' '.join(x))
# print(X_test[1])

feature_extraction = TfidfVectorizer(lowercase=False)
X_train = feature_extraction.fit_transform(X_train.values)
X_test = feature_extraction.transform(X_test.values)

clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))


