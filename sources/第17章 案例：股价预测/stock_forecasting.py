
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
plt.style.use('seaborn')
import seaborn as sns

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.ensemble import AdaBoostRegressor
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import datetime

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

#读入数据
x = pd.read_csv('dataset.csv', sep=',', encoding = 'utf-8')

#查看数据类型
x.dtypes

#查看数据前几行
x.head()

#查看数据信息
x.info()

#画出时序特征趋势线
x.drop(['index_code', 'date','time',"volume","money"], axis=1).plot(figsize=(15, 6))
plt.show()

#转数据类型，设置时间索引
x.index = pd.to_datetime(x['date'])

#按照时间序列进行移窗
df_shift, y = make_forecasting_frame(x["high"], kind="price", max_timeshift=20, rolling_direction=1)

#查看移窗数据
df_shift.head()

#查看移窗数据格式
df_shift.shape

#使用tsfresh包进行特征提取，默认为按最大特征提取，默认值ComprehensiveFCParameters
X = extract_features(df_shift, 
                     column_id="id", 
                     column_sort="time",
                     column_value="value", 
                     impute_function=impute,
                     show_warnings=False)

#查看X格式，过滤掉方差为0的自变量
print(X.shape)  #(6108, 794)
X = X.loc[:, X.apply(pd.Series.nunique) != 1] 
print(X.shape) #(6108, 794)

#将因变量位移一位拼到自变量上
X["feature_last_value"] = y.shift(1)

#因为位移后，会出现第一个变量为空，所以数据从第二行开始截取
X = X.iloc[1:, ]
y = y.iloc[1: ]

#查看自变量X
X.head()

#使用adaboost模型进行回归预测
ada = AdaBoostRegressor(n_estimators=10)
y_pred = [np.NaN] * len(y)

isp = 100   
assert isp > 0

for i in tqdm(range(isp, len(y))):
    
    ada.fit(X.iloc[:i], y[:i])
    y_pred[i] = ada.predict(X.iloc[i, :].values.reshape((1, -1)))[0]
    
y_pred = pd.Series(data=y_pred, index=y.index)


#拼接预测结果和真实结果
ys = pd.concat([y_pred, y], axis = 1).rename(columns = {0: 'pred', 'value': 'true'})

#转数据类型，设置时间序列索引
ys.index = pd.to_datetime(ys.index)
ys.head()

#画出预测结果
ys.plot(figsize=(15, 8))
plt.title('Predicted and True Price')
plt.show()

#画出真实值与前一个值的趋势图
ys['y-1'] = ys['true'].shift(1)
ys[['y-1', 'true']].plot(figsize = (15, 8))
plt.title('Benchmark Prediction and True Price')
plt.show()

#输出mae评价指标
print("MAE y-1: \t{}".format(np.mean(np.abs(np.diff(y))[isp-1:] )))
print("MAE ada: \t{}".format(np.mean(np.abs(y_pred - y)[isp:])))
#MAE y-1:        21.03476444148493
#MAE ada:        65.75323360939149

#由上述mae评价指标看出ada没有达到y-1的基准，
#所以需要迭代开发投入更多特征进入升维操作或者使用更高级别的模型


#输出ada模型重要特征
importances = pd.Series(index=X.columns, data=ada.feature_importances_)
importances.sort_values(ascending=False).head(10)

