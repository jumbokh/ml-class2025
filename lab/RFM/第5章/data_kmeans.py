#-*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pandas import to_datetime
#引入sklearn框架，导入K均值聚类算法
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
#from sklearn.manifold import

inputfile = r'./data/transformdata.xls' #待聚类的数据文件
outputfile=r'./data/data_type.xls'
#读取数据并进行聚类分析
data = pd.read_excel(inputfile) #读取数据
#利用K-Means聚类算法对客户数据进行客户分群，聚成4类
k = 4                       #需要进行的聚类类别数
iteration=500
kmodel = KMeans(n_clusters = k,max_iter=iteration) 
kmodel.fit(data) #训练模型
r1=pd.Series(kmodel.labels_).value_counts()
r2=pd.DataFrame(kmodel.cluster_centers_)
r=pd.concat([r2,r1],axis=1)
r.columns=list(data.columns)+[u'聚类数量']
r3 = pd.Series(kmodel.labels_,index=data.index)
r = pd.concat([data,r3], axis=1)
r.columns = list(data.columns)+[u'聚类类别']
r.to_excel(outputfile)
kmodel.cluster_centers_
kmodel.labels_
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
for i in range(k):
  cls=data[r[u'聚类类别']==i]
  cls.plot(kind='kde',linewidth=2,subplots=True,sharex=False)
  plt.suptitle('客户群=%d;聚类数量=%d' %(i,r1[i]))
plt.legend()
plt.show()




