# 資料降維與視覺化：PCA 與 t-SNE 教學教材

## 課程主題
本單元介紹資料科學中常見的兩種資料降維方法：
- 主成分分析（Principal Component Analysis, PCA）
- t-分佈隨機鄰域嵌入（t-distributed Stochastic Neighbor Embedding, t-SNE）

並透過程式實作與視覺化來幫助學生理解其應用場景與差異。

---

## 一、資料降維的概念
在資料分析與機器學習中，常常會遇到「高維資料」，也就是每一筆資料都擁有許多特徵（features）。例如圖片資料，一張 8x8 像素的圖片就有 64 維特徵。

資料降維（Dimensionality Reduction）是一種將高維資料映射到低維空間的技術，其目的包括：
- 降低計算成本
- 去除噪音
- 幫助資料視覺化與解釋

---

## 二、主成分分析（PCA）

### 1. 定義與原理
PCA 是一種**線性降維方法**，它透過尋找數據中「變異性最大」的方向，將資料投影到新的座標軸（主成分）上。

### 2. 特點
- 保留資料的主要變化方向
- 主成分彼此正交（無相關性）
- 計算速度快，適合初步資料探索與視覺化

### 3. 程式實作（以手寫數字資料為例）
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 載入資料
digits = load_digits()
X = digits.data  # 每筆資料為 64 維向量
y = digits.target  # 標籤 0~9

# PCA 降維至 2 維
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 繪圖
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=15)
plt.colorbar()
plt.title('PCA Visualization of Digits Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
```

---

## 三、t-SNE（t-distributed Stochastic Neighbor Embedding）

### 1. 定義與原理
 t-SNE 是一種**非線性降維技術**，特別適用於將高維資料壓縮到二維或三維以進行視覺化。

 t-SNE 會試圖讓低維空間中的資料分布儘可能保留原始高維空間中的鄰近關係，也就是「鄰近的資料點仍然靠近」。

### 2. 特點
- 強調局部結構，適合用來找出群集（clusters）
- 非常適合做資料視覺化
- 計算量大，處理大型資料集速度較慢

### 3. 程式實作
```python
from sklearn.manifold import TSNE

# t-SNE 降維
tsne = TSNE(n_components=2, perplexity=30.0, random_state=0)
X_tsne = tsne.fit_transform(X)

# 繪圖
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=15)
plt.colorbar()
plt.title('t-SNE Visualization of Digits Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
```

### 4. t-SNE 參數簡介
- `perplexity`：平衡局部與全域資訊，通常介於 5~50
- `n_components`：目標維度（通常為 2 或 3）
- `random_state`：設定亂數種子以方便重現

---

## 四、PCA vs t-SNE 比較
| 項目 | PCA | t-SNE |
|------|-----|-------|
| 類型 | 線性 | 非線性 |
| 強項 | 保留全域結構、快速 | 顯示群集、強調局部關係 |
| 缺點 | 無法展現非線性結構 | 計算慢，不能用來預測新資料 |

---

## 五、課堂練習建議
1. 嘗試使用不同的資料集（如 Iris、MNIST）做降維視覺化
2. 比較不同 `perplexity` 值對 t-SNE 結果的影響
3. 使用其他降維方法如 UMAP 做進階學習

---

## 六、小結
- PCA 與 t-SNE 是資料探索與視覺化常用工具
- PCA 側重變異結構；t-SNE 側重群集與鄰近關係
- 理解其原理與使用場景，有助於日後進行資料分析與機器學習任務

---




