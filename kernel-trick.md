好的！下面是用**简体中文**重写的《机器学习中的核技巧（kernel hacks）》速查表，保留可直接上手的做法与代码。

# “核”是什么（一句话）

用相似度函数 (k(x,x')=\langle \phi(x),\phi(x')\rangle) 代替显式特征，只与**核矩阵** (K_{ij}=k(x_i,x_j)) 打交道。很多算法在 (\phi(\cdot)) 空间线性、在原空间非线性。

# 常用核函数

* **线性核** (k(x,x')=x^\top x')（基线，快）。
* **RBF/高斯核** (k(x,x')=\exp(-\gamma|x-x'|^2))（默认好用）。
* **多项式核** ((\alpha x^\top x' + c)^d)（可控非线性）。
* **Laplacian核** (\exp(-\gamma|x-x'|_1))（更抗噪）。
* **卡方核 / 直方图交集核**（图像直方图常用）。
* **余弦核** (\frac{x^\top x'}{|x||x'|})（文本/TF-IDF）。

# 立竿见影的小技巧

1. **RBF 的“中位数启发”选 (\gamma)**
   (\gamma=\tfrac{1}{2\operatorname{median}(|x-x'|)^2})。很强的初值。

2. **核矩阵归一化**
   对角统一为 1：(K \leftarrow D^{-1/2}KD^{-1/2})，其中 (D_{ii}=K_{ii})。稳定 SVM/KRR。

3. **核中心化（做 KPCA/HSIC/MMD 必备）**
   (K_c = HKH)，(H=I-\tfrac{1}{n}\mathbf{1}\mathbf{1}^\top)。避免均值偏移。

4. **SVM 用预计算核**
   有自定义相似度？自己构建 (K)，用 `kernel="precomputed"`。

5. **非 PSD 核的修复**
   (K=Q\Lambda Q^\top) 做特征分解；将负特征值截为 0，再重建 (K)。避免求解器报错。

6. **大样本扩展**

   * **随机傅里叶特征（RFF）**：近似 RBF，接线性模型。
   * **Nyström**：采样地标近似 (K)。
     把 (O(n^2)) 变得可训练。

7. **简易多核学习（MKL-lite）**
   融合多视角：(K=\sum_m w_m K^{(m)})，不同核捕捉不同模态；用验证集调 (w_m)。

8. **类别不平衡的 SVM**
   `class_weight="balanced"`，并可为各类设置不同惩罚。

9. **按数据类型选核**

   * 文本：余弦核 / 字符串核。
   * 直方图（视觉）：卡方/交集核。
   * 时序：DTW 类核（若非 PSD，先做 PSD 修复）。

# 极简但好用的代码

**1）RBF-SVM：中位数启发 + 归一化 + 预计算核**

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances

X, y = ...  # 你的训练数据

# 1) 用中位数启发选 gamma
dists = pairwise_distances(X, metric="euclidean")
med = np.median(dists[np.triu_indices_from(dists, 1)])
gamma = 1.0 / (2 * (med**2 + 1e-12))

# 2) 构建并归一化核矩阵
K = np.exp(-gamma * dists**2)
D = np.diag(1.0 / np.sqrt(np.diag(K) + 1e-12))
K_norm = D @ K @ D

# 3) 用预计算核训练 SVM
clf = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
clf.fit(K_norm, y)

# 4) 对新样本 Z 预测
from sklearn.metrics import pairwise_distances
dZ = pairwise_distances(Z := ..., X, metric="euclidean")
KZ = np.exp(-gamma * dZ**2)
# 用训练时的 D 做左归一化即可
KZ_norm = (KZ @ D)
ypred = clf.predict(KZ_norm)
```

**2）随机傅里叶特征 → 逻辑回归（快 & 可扩展）**

```python
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

gamma = 1.0 / (2 * (med**2 + 1e-12))
model = make_pipeline(
    RBFSampler(gamma=gamma, n_components=2000, random_state=0),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)
model.fit(X, y)
```

**3）Nyström 特征 → 线性 SVM**

```python
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

feat = Nystroem(kernel="rbf", gamma=gamma, n_components=1000, random_state=0)
svm_lin = LinearSVC()
pipe = make_pipeline(feat, svm_lin)
pipe.fit(X, y)
```

**4）核 PCA：正确中心化 + Nyström 式外推**

```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=50, kernel="rbf", gamma=gamma, fit_inverse_transform=False)
X_k = kpca.fit_transform(X)
Z_k = kpca.transform(Z)  # 内部使用 Nyström 外推
```

# 核方法的高光场景

* **SVM / 核岭回归**：小到中等数据集的强基线。
* **KPCA / 谱聚类 / 扩散映射**：学习非线性流形结构。
* **高斯过程**：RBF/Matérn 核带来校准的不确定性。
* **两样本检验与独立性检验**：**MMD**、**HSIC**。

# 常见坑

* (O(n^2)) 时间/内存：样本 >~2 万时用 RFF/Nyström。
* 超参敏感：对 (C,\gamma) 做对数网格 + 分层交叉验证。
* 非 PSD 破坏凸性：自定义核要做 PSD 修复。
* 避免信息泄漏：(\gamma) 只在训练折上选，不看全量数据。

# 口袋清单（打印随身）

1. 输入先标准化；
2. RBF + 中位数启发 → CV 微调 (C,\gamma)；
3. 大样本 → RFF/Nyström + 线性模型；
4. 核做归一化、必要时中心化；
5. 永远保留线性基线并汇报对比。

---

如果你说的“kernel hacks”指的是**操作系统/GPU 的内核层优化**（如 Linux/eBPF 数据采集、CUDA kernel 调优、调度器设置等），我也可以给一份精炼的实战清单。

