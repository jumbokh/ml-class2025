import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据
iris = load_iris()
X = iris.data
y = iris.target

# 1. 数据分布及可视化
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

# 散点矩阵图
sns.pairplot(df, hue='target', palette='viridis')
plt.savefig('iris_pairplot.png')
plt.close()

# 相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.savefig('iris_heatmap.png')
plt.close()

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建管道
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', LogisticRegression())
    ]),
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', DecisionTreeClassifier())
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', KNeighborsClassifier())
    ]),
    'Naive Bayes': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', GaussianNB())
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', SVC())
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2)),
        ('clf', RandomForestClassifier())
    ])
}

# 训练和评估模型
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print(f"\n{name}:")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    plt.close()
