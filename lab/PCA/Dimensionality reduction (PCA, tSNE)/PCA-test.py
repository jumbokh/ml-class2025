# 单元测试代码，请用中文写代码。
import unittest
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

class TestTSNEVisualization(unittest.TestCase):
    def setUp(self):
        # 设置测试数据
        self.X = np.random.rand(100, 50)  # 随机生成100行50列的特征数据
        self.y = np.random.randint(0, 2, 100)  # 随机生成0或1的标签
        self.colors = ['red', 'blue']  # 定义颜色
        self.target_names = ['Class 0', 'Class 1']  # 定义目标名称

    @patch('matplotlib.pyplot.show')  # 模拟plt.show()，避免实际显示图像
    @patch('matplotlib.pyplot.savefig')  # 模拟plt.savefig()，避免实际保存文件
    def test_tsne_visualization(self, mock_savefig, mock_show):
        # 定义测试用例表
        test_cases = [
            {
                "description": "默认参数测试",
                "n_components": 2,
                "perplexity": 30,
                "n_iter": 1000,
                "expected_shape": (100, 2),  # t-SNE降维后应为100行2列
            },
            {
                "description": "较小的perplexity测试",
                "n_components": 2,
                "perplexity": 5,
                "n_iter": 500,
                "expected_shape": (100, 2),
            },
            {
                "description": "较大的迭代次数测试",
                "n_components": 2,
                "perplexity": 30,
                "n_iter": 2000,
                "expected_shape": (100, 2),
            },
        ]

        for case in test_cases:
            with self.subTest(case["description"]):
                # 初始化t-SNE
                tsne = TSNE(
                    n_components=case["n_components"],
                    init='pca',
                    random_state=1001,
                    perplexity=case["perplexity"],
                    method='barnes_hut',
                    n_iter=case["n_iter"],
                    verbose=0
                )
                # 执行t-SNE降维
                X_tsne = tsne.fit_transform(self.X)

                # 验证降维结果的形状
                self.assertEqual(X_tsne.shape, case["expected_shape"], f"降维结果形状不符合预期: {case['description']}")

                # 绘制散点图
                plt.figure(2, figsize=(10, 10))
                for color, i, target_name in zip(self.colors, [0, 1], self.target_names):
                    plt.scatter(
                        X_tsne[self.y == i, 0],
                        X_tsne[self.y == i, 1],
                        color=color,
                        s=1,
                        alpha=0.8,
                        label=target_name,
                        marker='.'
                    )
                plt.legend(loc='best', shadow=False, scatterpoints=3)
                plt.title('Scatter plot of t-SNE embedding')
                plt.xlabel('X')
                plt.ylabel('Y')

                # 验证保存和显示是否被调用
                plt.savefig('t-SNE-porto-01.png', dpi=150)
                mock_savefig.assert_called_once_with('t-SNE-porto-01.png', dpi=150)
                plt.show()
                mock_show.assert_called_once()

                # 清理绘图
                plt.clf()
                mock_savefig.reset_mock()
                mock_show.reset_mock()

if __name__ == '__main__':
    unittest.main()