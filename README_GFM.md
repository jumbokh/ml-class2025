# 机器学习中的核技巧（Kernel Hacks）— GitHub 友好版

> 说明：这是为 GitHub Flavored Markdown（GFM）重排的版本，尽量避免会在 GitHub 上误渲染的 LaTeX/HTML 写法。需要数学公式版（使用 `$...$`/`$$...$$`）也可以另行提供。

---

## 一、快速排查清单（GitHub 渲染常见问题）
1. **公式不渲染**：尽量避免 `\(...\)`/`\[...\]`；GitHub 目前支持 `$...$` 与 `$$...$$`（KaTeX），但**不要**放在代码块或反引号中，也尽量不要放在表格里。  
2. **尖括号被吞**：`<phi(x)>` 会被当成 HTML；用**反引号**包裹，或改用 Unicode 角括号 **⟨ ⟩**。  
3. **下划线触发斜体**：如 `K_{ij}` 在纯文本会被解析；用**反引号**包裹：``K_{ij}``。  
4. **标题井号后要空格**：`#标题` 可能不识别，应写 `# 标题`。  
5. **列表里的代码块**：在列表项中使用三反引号代码块时，**上一行空一行**；可不缩进，但要保持空行。  
6. **表格语法**：要有对齐行：`| --- | ---: | :---: |`。  
7. **HTML 被净化**：复杂内联 HTML/CSS 可能被移除；尽量用纯 Markdown。

---

## 二、核方法的直觉
使用相似度函数 `k(x, x') = <phi(x), phi(x')>` 代替显式特征，只与**核矩阵** ``K[i,j] = k(x_i, x_j)`` 打交道。很多算法在 `phi(.)` 空间线性、在原空间非线性。  
（为避免 HTML 误解，也可写作 `k(x, x') = ⟨φ(x), φ(x')⟩`）

---

## 三、常用核函数
- **线性核**：`k(x, x') = x^T x'`  
- **RBF/高斯核**：`k(x, x') = exp(-gamma * ||x - x'||^2)`  
- **多项式核**：`(alpha * x^T x' + c)^d`  
- **Laplacian 核**：`exp(-gamma * ||x - x'||_1)`  
- **直方图类**：卡方核 / 直方图交集核  
- **文本类**：余弦相似度核（`cosine`）

---

## 四、上手就灵的“小技巧”
- **中位数启发（RBF 的 `gamma` 初值）**：  
  `gamma = 1 / (2 * median(pairwise ||x - x'||)^2)`
- **核矩阵归一化**：  
  令 `D = diag(K)`，做 `K <- D^(-1/2) * K * D^(-1/2)`，使对角为 1，利于 SVM/KRR 稳定。
- **中心化核**（用于 KPCA/HSIC/MMD）：  
  `Kc = H * K * H`，其中 `H = I - (1/n) * 11^T`。
- **SVM 预计算核**：自定义相似度时，`kernel="precomputed"`。
- **非 PSD 修复**：若 `K = QΛQ^T`，将负特征值截为 0 再重建。
- **大样本可扩展**：  
  - **RFF（随机傅里叶特征）**：近似 RBF，后接线性模型；  
  - **Nyström**：采样 landmark 近似 `K`。
- **多核融合（简版 MKL）**：`K = sum_m w_m * K^(m)`，多模态/多视角时好用。
- **类别不平衡**：`class_weight="balanced"`，必要时对不同类设不同惩罚。
- **按数据类型选核**：文本→余弦/字符串；直方图→卡方/交集；时序→DTW 类核（必要时先做 PSD 修复）。

---

## 五、代码片段（GFM 兼容）

> 代码块前后各空一行；已用语言标注启用语法高亮。

**(1) RBF-SVM：中位数启发 + 归一化 + 预计算核**
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
Z = ...
dZ = pairwise_distances(Z, X, metric="euclidean")
KZ = np.exp(-gamma * dZ**2)
KZ_norm = KZ @ D  # 用训练时的 D 左归一化即可
ypred = clf.predict(KZ_norm)
```

**(2) 随机傅里叶特征（RFF）→ 逻辑回归（快 & 可扩展）**
```python
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 仍然可用上面的 med/gamma 估计方式
gamma = 1.0 / (2 * (med**2 + 1e-12))
model = make_pipeline(
    RBFSampler(gamma=gamma, n_components=2000, random_state=0),
    LogisticRegression(max_iter=2000, class_weight="balanced")
)
model.fit(X, y)
```

**(3) Nyström 特征 → 线性 SVM**
```python
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

feat = Nystroem(kernel="rbf", gamma=gamma, n_components=1000, random_state=0)
svm_lin = LinearSVC()
pipe = make_pipeline(feat, svm_lin)
pipe.fit(X, y)
```

**(4) 核 PCA：中心化 + Nyström 式外推**
```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=50, kernel="rbf", gamma=gamma, fit_inverse_transform=False)
X_k = kpca.fit_transform(X)
Z_k = kpca.transform(Z)  # 内部使用 Nyström 外推
```

---

## 六、何时用核方法
- **SVM / 核岭回归**：小到中等规模数据的强基线；  
- **KPCA / 谱聚类 / 扩散映射**：学习非线性流形结构；  
- **高斯过程**：RBF/Matérn 核带来校准的不确定性；  
- **统计检验**：MMD、HSIC 等核化方法。

---

## 七、常见坑
- `O(n^2)` 内存/时间：样本数 ≳ 2 万时应改用 RFF/Nyström；  
- 超参敏感：`C, gamma` 做对数网格 + 分层交叉验证；  
- 非 PSD 破坏凸性：自定义核先做 PSD 修复；  
- 避免信息泄漏：`gamma` 只在训练折上估计。

---

## 八、口袋清单
1) 标准化输入；  
2) RBF + 中位数启发 → 交叉验证微调 `C, gamma`；  
3) 大样本 → RFF/Nyström + 线性模型；  
4) 核归一化，必要时中心化；  
5) 始终保留线性基线并报告对比。

---

**Tip**：如需包含数学公式的 README（KaTeX 版），或要把本文拆成“讲义 + 代码示例”的多文件结构，我可以再生成一份仓库骨架（含 `README.md`、`/code`、`/data`、`.gitignore` 等）。
