import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import re
from sklearn.preprocessing import LabelEncoder

# 載入數據
try:
    # 讀取原始數據
    with zipfile.ZipFile('Dimensionality reduction (PCA, tSNE)/train.csv.zip', 'r') as z:
        with z.open(z.namelist()[0]) as f:
            # 讀取表頭行
            header = f.readline().decode('utf-8').strip()
            # 使用正則表達式分割列名
            import re
            columns = re.findall(r'([a-z_]+(?:\d+)?)', header)
            
            # 讀取數據行
            data = []
            for line in f:
                line = line.decode('utf-8').strip()
                # 使用正則表達式分割數值
                values = re.findall(r'(-?\d+\.?\d*)', line)
                data.append(values)
                
    # 創建DataFrame
    train = pd.DataFrame(data, columns=columns)
    y = train['target'].astype(float).values
except Exception as e:
    print(f"Error loading data: {e}")
    exit(1)

# 對y進行編碼(如果是分類變量)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 繪製y的分布
plt.figure(figsize=(10, 6))

# 如果是分類變量
if len(np.unique(y)) < 10:
    counts = np.bincount(y_encoded)
    plt.bar(range(len(counts)), counts)
    plt.xticks(range(len(counts)), le.classes_)
    plt.title('Distribution of Target Variable (y)')
    plt.xlabel('Class')
    plt.ylabel('Count')
# 如果是連續變量
else:
    plt.hist(y, bins=50)
    plt.title('Distribution of Target Variable (y)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('y_distribution.png')
plt.show()
