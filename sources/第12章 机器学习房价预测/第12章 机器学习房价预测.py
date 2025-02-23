import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor as xgb
import math


dic={}
ref=0

#读取训练集数据并处理
def getdata(f):
    global ref,dic
    d=pd.read_csv(f)
    #print(d['Id'])
    d.drop(['Id'],axis=1,inplace=True)
    tmphead=list(d.head())
    tmplist=list(d.values)
    rlist=[]
    #对训练集中的字符串数据进行处理
    for i in tmplist:
        tmp=[]
        for t in i:
            try:
                if(math.isnan(float(t))):
                    #用字典dic把字符串类型的数据映射到float上
                    try:
                        tmp.append(dic['NA'])
                    except:
                        dic['NA']=ref
                        ref+=1
                        tmp.append(dic['NA'])
                else:
                    tmp.append(float(t))
            except:
                try:
                    tmp.append(dic[t])
                except:
                    dic[t]=ref
                    ref+=1
                    tmp.append(dic[t])
        rlist.append(tmp)
    return pd.DataFrame(rlist, columns=tmphead)

#读取测试集数据并处理
def gettarget(f):
    global dic
    d=pd.read_csv(f)
    tmphead=list(d.head())
    tmplist=list(d.values)
    rlist=[] 
    for i in tmplist:
        tmp=[]
        for t in i:
            try:
                if(math.isnan(float(t))):
                    tmp.append(dic['NA'])
                else:
                    tmp.append(float(t))
            except:
                tmp.append(dic[t])
        rlist.append(tmp)
        
    return pd.DataFrame(rlist, columns=tmphead)

#文件写入操作
def writedata(idi,data,file):
    with open(file,'w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Id','SalePrice'])
        for i in range(len(data)):
            writer.writerow([int(idi[i]),data[i]])
            
def getresult(mymodel,target):
    return mymodel.predict(target)

#选择模型最优参数并训练、预测    
def selectmodel(train,label):
    #Xgboost的基分类器：选用树或线性分类
    params={'booster':['gbtree', 'gblinear' , 'dart']}
    mymodel= GridSearchCV(xgb(), params, error_score=1,refit=True)
    mymodel.fit(train,label)
    return mymodel,mymodel.best_score_



d=getdata('train.csv')      #getdata返回经处理过的数据
corrmat = d.corr()          #计算相关系数
rela=list(corrmat['SalePrice'].abs().sort_values().index)[:-1]      #将特征列按相关性大小排序
features=68
select_feat=rela[-features:]            #取相关性好的前68列

train,label=d.drop(['SalePrice'],axis=1,inplace=False),d['SalePrice']
train=train[select_feat]
d=gettarget('test.csv')
idi,target=d['Id'],d.drop(['Id'],axis=1,inplace=False)
target=target[select_feat]


mymodel,score=selectmodel(train,label)
print(score)

#将预测值整理到submission1.csv，提交submission1.csv在Kaggle平台评测
result=getresult(mymodel,target)
writedata(idi,result,' submission1.csv')
