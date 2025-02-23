# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:13:57 2020

@author: ThinkPad
"""

from sklearn.feature_extraction.text import CountVectorizer as CV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as NB


if __name__ == '__main__':
    with open('./SMSSpamCollection.txt', 'r', encoding='utf8') as f:
        sms = [line.split('\t') for line in f]
    y, x = zip(*sms)
    
    y = [label == 'spam' for label in y]
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    counter = CV(token_pattern='[a-zA-Z]{2,}')
    x_train = counter.fit_transform(x_train)
    x_test = counter.transform(x_test)
    
    model = NB()
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print("train score:", train_score)
    print("test score:", test_score)