import random
"""
函数说明：加载文本中的数据加
Parameters：
	filename - 文件路径
Returns：
	lines - 二维列表，存放数据
	linesNum - 数据个数
"""
def data_generate(filename):
	#打开文件
	fr = open(filename)
	#将文件内容读入lines
	lines = fr.readlines()
	#获取文件行数
	linesNum = len(lines)
	#对每一行的数据进行分割，最后返回一个二维列表和数据个数
	for (i,line) in enumerate(lines):
		line = line.split(',')
		tempLine = []
		for number in line:
			tempLine.append(float(number))
		lines[i] = tempLine

	return lines,linesNum
"""
函数说明：数据归一化
Parameters：
	lines- 从文本文件加载的原始数据
Returns：
	lines - 归一化后的数据
"""
def data_normalization(lines):
	#max和min列表存储着当前每一维的最大值与最小值
	max=lines[0][:]
	min=lines[0][:]
	#遍历所有line动态更新最大值与最小值
	for line in lines:
		for (i,item) in enumerate(line):
			if i > 0:
				if item > max[i]:
					max[i] = item
				if item < min[i]:
					min[i] = item
	#根据max和min向量，对数据进行归一化返回归一化后的数据
	for line in lines:
		for (i,item) in enumerate(line):
			if i > 0:
				line[i] = (line[i] - min[i])/(max[i] - min[i])
	return lines

"""
函数说明：将数据分为训练集与测试集
Parameters：
	lines - 归一化后的数据
Returns：
	TestingSet - 测试集
	lines - 训练集
"""
def data_classification(lines):
	#测试集
	TestingSet = []
	#按照比例抽取数据
	#class 1 59
	#class 2 71
	#class 3 48
	for i in range(0,6):
		TestingSet.append(lines.pop(random.randint(0,58-i)))
	for i in range(0,7):
		TestingSet.append(lines.pop(random.randint(53,123-i)))
	for i in range(0,5):
		TestingSet.append(lines.pop(random.randint(117,164-i)))
	return TestingSet,lines

def get_Euclidean_distance(sample1,sample2):
	distance = 0
	for i in range(1,14):
		distance += (sample1[i]-sample2[i])**2
	return distance**0.5

"""
函数说明：测试分类器的正确率
Parameters：
	TestingSet - 测试集
	TrainingSet - 训练集
	k - 参数，决定所取最近数据的个数
Returns：
	correctRatio - 正确率
"""
def KNN_Test(TestingSet,TrainingSet,k):
	#分类正确的数量
	correctNum = 0
	#对测试集中的每个数据计算距离最近的k个数据
	for TestingUnit in TestingSet:
		#存放距离TestingUnit最近的k个数据的类别与距离
		shortest=[]
		#计算出k个最近的数据
		for (i,TrainingUnit) in enumerate(TrainingSet):
			if len(shortest) < k:
				shortest.append([TrainingUnit[0],get_Euclidean_distance(TestingUnit,TrainingUnit)])
			else :
				distance = get_Euclidean_distance(TestingUnit,TrainingUnit)
				for j in range(0,k):
					if shortest[j][1] > distance:
						shortest.pop(j)
						shortest.insert(j,[TrainingUnit[0],distance])
						break
		#统计种类
		r0 = []
		for i in shortest:
			r0.append(i[0])
		r = []
		for i in range(1,4):
			r.append([i,r0.count(i)])
		#选取数量最多的类别
		result = []
		max = 0
		for item in r:
			if len(result) == 0:
				result.append(item)
				max = item[1]
			else:
				if item[1] > max:
					result.clear()
					result.append(item)
				elif item[1] == max:
					result.append(item)
		#对于数量最多种类大于一个的情况，随机选取一个
		item = result[random.randrange(0,len(result))]
		#如果预测结果正确，correctNum加一
		if item[0] == TestingUnit[0]:
			correctNum+=1
	#计算正确率
	correctRatio = 100 * correctNum/len(TestingSet)
	return correctRatio

if __name__ == '__main__':
	filename = 'C:\\Users\\lihongyu\\Desktop\\wine.data'
	Lines, LinesNum = data_generate(filename)
	Lines = data_normalization(Lines)
	TestingSet,TrainingSet = data_classification(Lines)
	print('分类正确率为：%.2f%%' % KNN_Test(TestingSet,TrainingSet,5))