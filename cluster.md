這是一份為**兩小時機器學習課程「聚類分析」**所設計的課綱與實作內容安排，適合大學部學生在 Jupyter Notebook 使用 Python + scikit-learn 進行操作。包含 **理論簡介 + 三個實作範例 + 問題討論與延伸任務**。

---

## 🧠 課程主題：聚類分析（Clustering）  
**課程長度：2 小時**  
**工具：Python + scikit-learn + Jupyter Notebook**  
**學生程度：具備基本 Python 與 NumPy、pandas、matplotlib 操作能力**

---

## ⏱️ 課程時間安排

| 時間       | 主題內容                       | 教學方式       |
|------------|--------------------------------|----------------|
| 0:00–0:20  | 聚類簡介與常見演算法概述        | 投影片 + 互動討論 |
| 0:20–0:50  | 範例一：KMeans 分群 – Iris 資料集 | Notebook 實作 |
| 0:50–1:20  | 範例二：DBSCAN 分群 – 有噪音的資料 | Notebook 實作 |
| 1:20–1:40  | 範例三：影像壓縮 – 使用 KMeans 壓縮圖像 | Notebook 實作 |
| 1:40–2:00  | 問題討論 + 延伸任務 + Q&A      | 問題討論與反思 |

---

## 📝 課程內容詳述

### 🔹 0:00–0:20 聚類理論概論

簡報內容包含：
- 聚類是什麼？(Unsupervised Learning)
- 常見聚類方法簡介：
  - KMeans（適合球狀分佈）
  - DBSCAN（密度導向，可處理異常值）
  - 層次聚類（Hierarchical Clustering）
- 評估聚類效果的方法：
  - Silhouette Score
  - 可視化評估
- 聚類應用場景簡介：市場分群、社群分析、影像壓縮、異常偵測...

---

### 🔹 0:20–0:50 範例一：KMeans 分群（Iris 資料集）

**教學重點：**
- 了解 `KMeans` 的流程與參數（如 `n_clusters`、`init`）
- 比較預測群與真實分類的對應關係（雖然是非監督學習）
- 使用 Silhouette Score 評估群集品質

📘 Dataset: `sklearn.datasets.load_iris()`

---

### 🔹 0:50–1:20 範例二：DBSCAN 密度導向聚類

**教學重點：**
- 使用 `make_moons()` 或 `make_blobs()` 產生非線性分布資料
- 比較 KMeans vs DBSCAN 的分群結果
- 討論 `eps` 和 `min_samples` 對結果的影響
- 處理噪音點（label = -1）

📘 Dataset: `sklearn.datasets.make_moons()` or `make_blobs()`

---

### 🔹 1:20–1:40 範例三：影像壓縮（KMeans）

**教學重點：**
- 將圖像像素視為三維資料點（RGB）
- 使用 `KMeans` 對圖像進行色彩聚類
- 減少顏色總數（壓縮色彩）
- 原圖與壓縮後圖像的比較視覺化

📘 圖片來源：Lenna.png 或隨機選取開源圖像（可使用 `skimage.data.astronaut()`）

---

### 🔹 1:40–2:00 問題討論與延伸任務

**討論題目：**
- 如何決定 `k` 值（Elbow Method vs Silhouette）
- 當資料含有異常值或非球狀結構時，應使用哪種聚類方法？

**延伸任務（可當作作業）：**
1. 嘗試在 `Wine` 資料集或你自己的資料上進行 KMeans 與 DBSCAN 分群
2. 嘗試使用 `AgglomerativeClustering` 進行階層式聚類
3. 製作聚類效果的視覺化動圖（可選）

---

## 📂 附檔內容

