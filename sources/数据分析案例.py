# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#引入工具包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
#读入数据
path='/Users/gray/Desktop'
    #注意文件夹路径，双斜杠
path_name=os.path.join(path,'data.csv')
    #注意，os.path.join函数，执行路径拼接
    #'C:\\Users\\wdxiaochun\\Desktop\\car_loan\\accepts.csv'

data = pd.read_csv(path_name)
#data = pd.read_csv('data.csv')
#查看样本形状，样本条数，样本属性数量
data.shape 
#查看数据前5条数据
data.head()
#查看数据大概情况
data.describe().T
#观察并通过duplicated方检查发现现application_id account_number都是样本的唯一编号且二者同值，取其一即可
data.loc[:,['application_id','account_number']].duplicated().sum()
data.dtypes
#分别划分X变量与Y变量
x_list=['vehicle_year', 'vehicle_make', 'bankruptcy_ind', 'tot_derog', 'tot_tr',
            'age_oldest_tr', 'tot_open_tr', 'tot_rev_tr', 'tot_rev_debt', 'tot_rev_line', 'rev_util', 'fico_score', 'purch_price',
            'msrp', 'down_pyt', 'loan_term', 'loan_amt', 'ltv', 'tot_income', 'veh_mileage', 'used_ind', 'weight']
data_x=data.loc[:,x_list]
data_y=data.loc[:,'bad_ind']


#查看Y变量值的分布 正负样本数量
data_y.value_counts()
#真对X变量进行数据探索
pd.set_option('display.max_columns', None)
data_x_des=data_x.describe(include='all').T
data_x_des
#将变量中文名与变量一一对应，方便后续进行查看
all_list=list(data.columns)
cn_label=['申请者ID','帐户号','是否违约','汽车购买时间','汽车制造商','曾经破产标识','五年内信用不良事件数量','全部帐户数量',
           '账号存续月份数','开户帐户数量','信用卡数量','信用卡欠款余额','信用卡授信额度','信用卡额度使用比例',
           'FICO打分','汽车购买金额','建议售价','分期付款的首次交款','贷款期限','贷款金额','贷款金额/售价','月均收入','行驶里程','是否二手车','样本权重'
           ]
label_dict={}
for i in range(len(all_list)):
    label_dict[all_list[i]]=cn_label[i]
#将label拼接到X变量
label_series=pd.Series(label_dict)
data_x_des['label']=label_series
data_x.isnull().sum()

###########################
#查看月均收入的总体分布情况和缺失值情况
data_x['tot_income'].value_counts(dropna=False)
data_x['tot_income'].describe().T
data_x['tot_income'].isnull().sum()
#查看每项的违约情况
data_y.groupby(data_x['tot_income']).agg(['count','mean'])
#缺失值采用中位数填充
data_x['tot_income']=data_x['tot_income'].fillna(data_x['tot_income'].median())
#盖帽处理
q25=data_x['tot_income'].quantile(0.25)
q75=data_x['tot_income'].quantile(0.75)
max_qz=q75+1.5*(q75-q25)
sum(data_x['tot_income']>max_qz)
#359 #存在259个样本的取值超过理论极大值，进行盖帽

temp_series=data_x['tot_income']>max_qz
data_x.loc[temp_series,'tot_income']=max_qz
data_x['tot_income'].describe()

##对信用卡授信额度进行预处理
data_x['tot_rev_line'].value_counts(dropna=False)
data_x['tot_rev_line'].describe().T
data_x['tot_rev_line'].isnull().sum()
#查看每项的违约情况
data_y.groupby(data_x['tot_rev_line']).agg(['count','mean'])
data_x['tot_rev_line1']=data_x['tot_rev_line'].fillna('unknown')
data_y.groupby(data_x['tot_rev_line1']).agg(['count','mean'])
#盖帽处理
q25=data_x['tot_rev_line'].quantile(0.25)
q75=data_x['tot_rev_line'].quantile(0.75)
max_qz=q75+1.5*(q75-q25)
sum(data_x['tot_rev_line']>max_qz)
#259 #存在259个样本的取值超过理论极大值，进行盖帽

temp_series=data_x['tot_rev_line']>max_qz
data_x.loc[temp_series,'tot_rev_line']=max_qz
data_x['tot_rev_line'].describe()
#将数据分箱 用999999填充缺失值
data_x['tot_rev_line_fx']=pd.qcut(data_x['tot_rev_line'],10,labels=False,duplicates='drop')
data_x['tot_rev_line_fx']=data_x['tot_rev_line_fx'].fillna(999999)

#查看不同码值对应的违约率情况
data_y.groupby(data_x['tot_rev_line_fx']).agg(['count','mean'])

#汽车制造年份预处理
#产看其缺失值情况
data_x.loc[:,'vehicle_year'].value_counts().sort_index()
data_x['vehicle_year'].isnull().sum()
data_y.groupby(data_x['vehicle_year']).agg(['count','mean'])

#填充缺失值
data_x.loc[:,'vehicle_year'][data_x.loc[:,'vehicle_year'].isin([0,9999])]=np.nan
data_x['vehicle_year']=data_x['vehicle_year'].fillna(data_x['vehicle_year'].median())

#对破产标示进行预处理
data_x['bankruptcy_ind'].value_counts(dropna=False)
data_x['bankruptcy_ind1']=data_x['bankruptcy_ind'].fillna('unknown')
data_y.groupby(data_x['bankruptcy_ind1']).agg(['count','mean'])

#重新划分X与Y变量
x_var_list=['tot_derog', 'tot_tr', 'age_oldest_tr', 'tot_open_tr', 'tot_rev_tr', 'tot_rev_debt', 'tot_rev_line', 'rev_util', 'fico_score', 'purch_price',
            'msrp', 'down_pyt', 'loan_term', 'loan_amt', 'ltv', 'tot_income', 'veh_mileage', 'used_ind']
data_x=data.loc[:,x_var_list]
data_y=data.loc[:,'bad_ind']
#用中位数填充缺失值
temp=data_x.median()
temp_dict={} 
for i in range(len(list(temp.index))):
    temp_dict[list(temp.index)[i]]=list(temp.values)[i]    

data_x_fill=data_x.fillna(temp_dict)
#使用train_test_split划分训练集与测试集
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y=train_test_split(data_x_fill, data_y, test_size=0.25, random_state=12345)

#引入线性回归工具包
from sklearn.linear_model import LinearRegression
linear = LinearRegression()
#模型训练
model = linear.fit(train_x,train_y)
#查看相关系数
linear.intercept_
linear.coef_
#排序得出权重最大的几个变量
var_coef=pd.DataFrame()
var_coef['var']=x_var_list
var_coef['coef']=linear.coef_
var_coef.sort_values(by='coef', ascending=False)

#clf.predict_proba，计算每个样本的预测概率，预测y=0，y=1，提取y=1的一列数据
#注意，roc_curve函数，返回假正率、真正率、门槛值
#基于测试集,进行auc模型评估
import sklearn.metrics as metrics
fpr, tpr, th = metrics.roc_curve(test_y, linear.predict(test_x))
metrics.auc(fpr, tpr)
#绘制ROC曲线
import matplotlib.pyplot as plt
plt.figure(figsize=[8, 8])
plt.plot(fpr, tpr, color='b')
plt.plot([0, 1], [0, 1], color='r', alpha=.5, linestyle='--') 
plt.show()


##使用决策树模型进行数据分析
from sklearn.tree import DecisionTreeClassifier, plot_tree
tree = DecisionTreeClassifier()
#模型训练
tree.fit(train_x,train_y)
#查看树的深度
len(np.unique(tree.apply(train_x)))
#查看模型训练效果
fpr, tpr, th = metrics.roc_curve(test_y, tree.predict_proba(test_x.values)[:,1])
metrics.auc(fpr, tpr)

#调整决策树参数重新构建决策树 重新设置树的最大深度和叶节点大小
tree2 = DecisionTreeClassifier(max_depth=20,min_samples_leaf=100)
tree2.fit(train_x,train_y)
#查看树的深度
len(np.unique(tree2.apply(train_x)))

#查看auc评估指标
fpr, tpr, th = metrics.roc_curve(test_y, tree2.predict_proba(test_x.values)[:,1])
metrics.auc(fpr, tpr)
#查看决策树结构
plt.figure(figsize=[16,10])
plot_tree(tree2, filled=True)
    #plot_tree函数，绘制决策树的整体结构
plt.show()
#绘制ROC曲线
plt.figure(figsize=[8, 8])
plt.plot(fpr, tpr, color='b')
plt.plot([0, 1], [0, 1], color='r', alpha=.5, linestyle='--') 
plt.show()

#采用随机森林模型进行分类预测
from sklearn.ensemble import RandomForestClassifier
#forest = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=100,random_state=11223)
forest =  RandomForestClassifier()
#模型训练
forest.fit(train_x,train_y)
#查看auc评估指标
fpr, tpr, th = metrics.roc_curve(test_y, forest.predict_proba(test_x.values)[:,1])
metrics.auc(fpr, tpr)
#调参构建新的随机森林
forest1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_leaf=100,random_state=11223)
#构建新的随机森林模型
forest1.fit(train_x,train_y)
#查看auc评估指标
fpr, tpr, th = metrics.roc_curve(test_y, forest1.predict_proba(test_x.values)[:,1])
metrics.auc(fpr, tpr)
#绘制ROC曲线
plt.figure(figsize=[8, 8])
plt.plot(fpr, tpr, color='b')
plt.plot([0, 1], [0, 1], color='r', alpha=.5, linestyle='--') 
plt.show()


