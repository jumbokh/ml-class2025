from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

"""
函数说明：数据读取
Parameters：
	filename - 文件路径
Returns：
	Matrix - 数据矩阵
	labels - 数据标签列表
"""
def dataGenerate(filename):
	#打开文件
    file = open(filename)
	#读取数据
    lines = file.readlines()
	#获取数据个数
    linesNum = len(lines)
	#根据数据初始化矩阵
    Matrix = np.zeros((linesNum,14))
	#初始化标签列表
    labels = []
	#将数据读入矩阵
    for (i,line) in enumerate(lines):
        line = line.strip()
        line_data = line.split(',')
        Matrix[i,:] = line_data[0:14]
	#打乱数据的排列，为随机抽取测试集做准备
    np.random.shuffle(Matrix)

    labels = Matrix[:,0]
    Matrix = Matrix[:,1:14]

    return Matrix, labels

"""
函数说明：数据归一化
Parameters：
	dataMatrix - 数据矩阵
Returns：
	matNormalized - 归一化后的数据矩阵
	ranges - 每一维数据max-min的值
	mins - 1*14矩阵，存储数据对应维的最小值
"""
def dataNormalization(dataMatrix):
	#获取列最小值（参数为1是获取行最小值）
	mins = dataMatrix.min(0)
	#获取列最大值（参数为1是获取行最大值）
	maxs = dataMatrix.max(0)
	#计算每一维数据max-min的值
	ranges = maxs - mins
	#根据数据矩阵行列数初始化归一化矩阵
	matNormalized = np.zeros(np.shape(dataMatrix))
	#获取数据矩阵行数
	m = dataMatrix.shape[0]
	#归一化计算
	matNormalized = dataMatrix - np.tile(mins, (m, 1))
	matNormalized = matNormalized / np.tile(ranges, (m, 1))
	return matNormalized, ranges, mins

"""
函数说明：对单个数据单元进行分类
Parameters：
	testUnit - 1*14矩阵，单个测试数据
	traingSet - 训练集
	labels - 标签列表
	k - KNN参数k
Returns：
	sortedClassCountDict[0][0] - 预测分类结果
"""
def unitclassify(testUnit, traingSet, labels, k):
	#获得训练集数据个数
	traingSetSize = traingSet.shape[0]
	#初始化距离矩阵并计算与训练集的差值
	diffMatrix = np.tile(testUnit, (traingSetSize, 1)) - traingSet
	#矩阵每个数据平方
	sqMatrix = diffMatrix**2
	#矩阵每一行加总（axis为0代表列加总）
	distanceMatrix = sqMatrix.sum(axis=1)
	#矩阵每一行开方，此时每一行数据代表测试数据与训练集每个数据的距离
	distances = distanceMatrix**0.5
	#距离按从小到大排序返回排序后数据对应索引值的列表
	sortedLabels = distances.argsort()
	#根据索引取前k个最近数据的标签存入字典
	classCountDict = {}
	for i in range(k):
		label = labels[sortedLabels[i]]
		classCountDict[label] = classCountDict.get(label,0) + 1
	#根据字典元素的value值排序
	sortedClassCountDict = sorted(classCountDict.items(),key=lambda unit:unit[1],reverse=True)
	#返回出现次数最多的类
	return sortedClassCountDict[0][0]

"""
函数说明：数据可视化
Parameters：
	dataMatrix - 数据矩阵
	labels - 数据标签
Returns：

"""

def showdatas(dataMatrix, labels):
	#设置画布为13*8并设置其为2*2布局
	fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))
	#初始化颜色标签列表
	colorLabels = []
	#根据数据标签设置散点图颜色标签
	#种类1为蓝色，种类2为橙色，种类3为红色
	for i in labels:
		if i == 1:
			colorLabels.append('blue')
		if i == 2:
			colorLabels.append('orange')
		if i == 3:
			colorLabels.append('red')
	#画布第一部分X轴为所有数据的第一维，Y轴为所有数据的第二维
	#根据颜色标签列表染色并设置散点大小和透明度
	axs[0][0].scatter(x=dataMatrix[:,0], y=dataMatrix[:,1], color=colorLabels,s=15, alpha=.5)
	#设置标题和X、Y轴标签
	axs0_title = axs[0][0].set_title('Alcohol and Malic acid')
	axs0_x = axs[0][0].set_xlabel('Alcohol')
	axs0_y = axs[0][0].set_ylabel('Malic acid')
	#设置标题和标签的字体大小和颜色
	plt.setp(axs0_title, size=9, weight='bold', color='black')
	plt.setp(axs0_x, size=7, color='black')
	plt.setp(axs0_y, size=7, color='black')

	axs[0][1].scatter(x=dataMatrix[:,2], y=dataMatrix[:,3], color=colorLabels,s=15, alpha=.5)
	axs1_title = axs[0][1].set_title('Ash and Alcalinity of ash')
	axs1_x = axs[0][1].set_xlabel('Ash')
	axs1_y = axs[0][1].set_ylabel('Alcalinity of ash')
	plt.setp(axs1_title, size=9, weight='bold', color='black')
	plt.setp(axs1_x, size=7, color='black')
	plt.setp(axs1_y, size=7, color='black')

	axs[1][0].scatter(x=dataMatrix[:,4], y=dataMatrix[:,5], color=colorLabels,s=15, alpha=.5)
	axs2_title = axs[1][0].set_title('Magnesium and Total phenols')
	axs2_x = axs[1][0].set_xlabel('Magnesium')
	axs2_y = axs[1][0].set_ylabel('Total phenols')
	plt.setp(axs2_title, size=9, weight='bold', color='black')
	plt.setp(axs2_x, size=7, color='black')
	plt.setp(axs2_y, size=7, color='black')

	axs[1][1].scatter(x=dataMatrix[:,6], y=dataMatrix[:,7], color=colorLabels,s=15, alpha=.5)
	axs3_title = axs[1][1].set_title('Flavanoids and Nonflavanoid phenols')
	axs3_x = axs[1][1].set_xlabel('Flavanoids')
	axs3_y = axs[1][1].set_ylabel('Nonflavanoid phenols')
	plt.setp(axs3_title, size=9, weight='bold', color='black')
	plt.setp(axs3_x, size=7, color='black')
	plt.setp(axs3_y, size=7, color='black')
	#生成图例
	class1 = mlines.Line2D([], [], color='blue', marker='.', markersize=6, label='1')
	class2 = mlines.Line2D([], [], color='orange', marker='.', markersize=6, label='2')
	class3 = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='3')
	#将图例设置到四个图中
	axs[0][0].legend(handles=[class1, class2, class3])
	axs[0][1].legend(handles=[class1, class2, class3])
	axs[1][0].legend(handles=[class1, class2, class3])
	axs[1][1].legend(handles=[class1, class2, class3])
	#绘图
	plt.show()


"""
函数说明：对所有测试集数据进行分类检验分类器正确率
Parameters：

Returns：

"""
def modelTest():
	filename = "wine.data"
	#生成数据
	Matrix, labels = dataGenerate(filename)

	showdatas(Matrix,labels)
	#测试集占比，这里取10%数据作为测试集
	randomRatio = 0.10
	#数据归一化
	matNormalized, ranges, mins = dataNormalization(Matrix)
	#根据比例与数据总数计算测试集数据量
	m = matNormalized.shape[0]
	numTest = int(m * randomRatio)
	correctCount = 0.0
	#取前numTest个数据依次作为测试数据单元
	for i in range(numTest):
		#将下标numTest到m-1的数据作为训练集，标签取对应部分
		result = unitclassify(matNormalized[i,:],matNormalized[numTest:m,:], labels[numTest:m], 5)
		print("分类结果:%s\t真实类别:%s" % (result, labels[i]))
		if result == labels[i]:
			correctCount += 1.0
	#打印正确率
	print("正确率:%.2f%%" %(correctCount/float(numTest)*100))

"""
函数说明：对所有测试集数据进行分类检验分类器正确率
Parameters：

Returns：

"""
def sampleTest():
	#接受用户输入
	property1 = float(input("property1:"))
	property2 = float(input("property2:"))
	property3 = float(input("property3:"))
	property4 = float(input("property4:"))
	property5 = float(input("property5:"))
	property6 = float(input("property6:"))
	property7 = float(input("property7:"))
	property8 = float(input("property8:"))
	property9 = float(input("property9:"))
	property10 = float(input("property10:"))
	property11 = float(input("property11:"))
	property12 = float(input("property12:"))
	property13 = float(input("property13:"))
	filename = "wine.data"
	#从文本中获取数据
	Matrix, labels = dataGenerate(filename)
	#数据归一化
	matNormalized, ranges, mins = dataNormalization(Matrix)
	#构建测试数据单元
	testUnit = np.array([property1, property2, property3,
	                     property4, property5, property6,
	                     property7, property8,property9,
	                     property10, property11, property12,
	                     property13])
	#对测试数据归一化
	testUnit_normalized = (testUnit - mins) / ranges
	#获取数据分类结果
	result = unitclassify(testUnit_normalized, matNormalized, labels, 5)
	print("这可能是第%d类酒" % result)

if __name__ == '__main__':
	modelTest()
	sampleTest()
