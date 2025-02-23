# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#读入数据
df = pd.read_csv('churn.csv',
                 sep=',',
                 encoding='utf-8')

#观察数据
df.shape  #(3333, 21)
df.info
df.head()
df.dtypes
df['Churn?'].value_counts()
#False.    2850
#True.      483

#整理变量
col_names = df.columns.tolist()
#['State',
# 'Account Length',
# 'Area Code',
# 'Phone',
# "Int'l Plan",
# 'VMail Plan',
# 'VMail Message',
# 'Day Mins',
# 'Day Calls',
# 'Day Charge',
# 'Eve Mins',
# 'Eve Calls',
# 'Eve Charge',
# 'Night Mins',
# 'Night Calls',
# 'Night Charge',
# 'Intl Mins',
# 'Intl Calls',
# 'Intl Charge',
# 'CustServ Calls',
# 'Churn?']

to_show = col_names[:6] + col_names[-6:]

#['State',
# 'Account Length',
# 'Area Code',
# 'Phone',
# "Int'l Plan",
# 'VMail Plan',
# 'Night Charge',
# 'Intl Mins',
# 'Intl Calls',
# 'Intl Charge',
# 'CustServ Calls',
# 'Churn?']

#整理因变量y
churn_result = df['Churn?']
y = np.where(churn_result == 'True.',1,0)

#删除无用的变量
to_drop = ['State','Area Code','Phone','Churn?']
churn_feat_space = df.drop(to_drop,axis=1)

#字符型变量转数值型变量
yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'


features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

# 自变量标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print ("Feature space holds %d observations and %d features" % X.shape)
print ("Unique target labels:", np.unique(y))
print (X[0])
print (len(y[y == 0]))


#5折交叉验证
from sklearn.model_selection import KFold

def run_cv(X,y,clf_class,**kwargs):
    
    kf = KFold(n_splits=5,shuffle=True)
    y_pred = y.copy()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

#代入三种模型
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

#输出精度
def accuracy(y_true,y_pred):
    return np.mean(y_true == y_pred)

print ("Support vector machines:")
print ("%.3f" % accuracy(y, run_cv(X,y,SVC)))
print ("Random forest:")
print ("%.3f" % accuracy(y, run_cv(X,y,RF)))
print ("K-nearest-neighbors:")
print ("%.3f" % accuracy(y, run_cv(X,y,KNN)))


#输出预测结果是否流失可能性
def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(n_splits=5,shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob


#使用10 estimators
pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
#得出流失可能性概率
pred_churn = pred_prob[:,1]
is_churn = y == 1

#统计预测结果不同流失概率对应的用户数
counts = pd.value_counts(pred_churn)

#针对预测结果不同流失概率对应的真正流失用户占比
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

#合并数据
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts






