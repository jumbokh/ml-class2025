# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 15:20:44 2020

@author: ThinkPad
"""

from collections import Counter

import numpy as np


class Dataset:
    
    def __init__(self):
        with open('./SMSSpamCollection.txt', 'r', encoding='utf8') as f:
            data = [self._split(line) for line in f]
        self.y, self.x = zip(*data)
        words = [word for line in self.x for word in line]
        dict_ = {word: i + 1 for i, word in enumerate(set(words))}
        self.word2ind = lambda word: dict_.get(word, -1)
        self.x = [list(map(self.word2ind, line)) for line in self.x]
        self.y = list(map(lambda label: label == 'spam', self.y))
    
    def __len__(self) -> int:
        assert len(self.x) == len(self.y)
        return len(self.x)
    
    def _split(self, line: str):
        label, text = line.split("\t")
        text = ''.join([
            ch if ch.isalnum() else ' ' 
            for ch in text
            ])
        text = [word for word in text.split() if word]
        return label, text


class NBClassifier:
    
    def __init__(self):
        self.N = np.array([0, 0])
        self.freq = [Counter(), Counter()]
        self.mapper = [
            lambda word: self.freq[0][word],
            lambda word: self.freq[1][word],
            ]
    
    def train(self, data: Dataset):
        npos = sum(data.y)
        self.N += [len(data) - npos, npos]
        for line, label in zip(data.x, data.y):
            self.freq[label].update(set(line))
    
    def test(self, data: Dataset) -> list:
        P1 = (self.N[1] + 1) / (self.N.sum() + 2)
        logP = np.log2([1 - P1, P1])
        
        pred = []
        for words in data.x:
            freq = [np.array(list(map(m, set(words)))) for m in self.mapper]
            freq = [
                np.sum(np.log2((f + 1)/(n + 2))) 
                for f, n in zip(freq, self.N)
                ] + logP
            pred.append(freq[0] / freq.sum())
        return pred


def evaluate(gt: list, pred: list) -> int:
    assert len(gt) == len(pred)
    pred = np.array(pred) > 0.54
    return np.sum(pred == gt) / len(pred)


if __name__ == '__main__':
    data = Dataset()
    model = NBClassifier()
    model.train(data)
    pred = model.test(data)
    acc = evaluate(data.y, pred)
    print(acc)

