import pandas as pd
#标准化处理
datafile = r'./data/data.xls' #需要进行标准化的数据文件；
transformfile = r'./data/transformdata.xls' #标准化后的数据存储路径文件；
data = pd.read_excel(datafile)
data=data[['R','F','买家实际支付金额']]
data = (data - data.mean(axis = 0))/(data.std(axis = 0)) #简洁的语句实现了标准化变换，类似地可以实现任何想要的变换。
data.columns=['R','F','M'] #表头重命名。
data.to_excel(transformfile, index = False) #数据写入
