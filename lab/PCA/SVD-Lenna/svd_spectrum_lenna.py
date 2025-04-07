import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_grayscale_image(path):
    """
    載入並轉為灰階圖像
    """
    img = Image.open(path).convert('L')
    return np.array(img)

def plot_singular_value_spectrum(matrix, title="Singular Value Spectrum"):
    """
    計算並繪製奇異值分佈圖
    """
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(S, marker='o', color='orange')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return S  # 回傳奇異值序列供分析用

def svd_compress_grayscale(matrix, k):
    """
    使用前 k 個奇異值壓縮灰階圖像
    """
    U, S, VT = np.linalg.svd(matrix, full_matrices=False)
    S_k = np.diag(S[:k])
    U_k = U[:, :k]
    VT_k = VT[:k, :]
    compressed = np.dot(U_k, np.dot(S_k, VT_k))
    return np.clip(compressed, 0, 255).astype(np.uint8)

def show_comparison(original, compressed, k):
    """
    顯示原圖與壓縮後的比較圖
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Grayscale Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(compressed, cmap='gray')
    plt.title(f"SVD Compressed (k={k})")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def save_image(img_array, path):
    """
    儲存圖像為 PNG 格式
    """
    Image.fromarray(img_array).save(path)
    print(f"✅ Saved compressed image to: {path}")

# --- 主程式區域 ---
if __name__ == "__main__":
    # 圖像路徑
    image_path = "Lenna.png"

    # 讀取圖像並轉為灰階
    gray_img = load_grayscale_image(image_path)

    # 畫出奇異值分佈圖
    singular_values = plot_singular_value_spectrum(gray_img, title="Singular Value Spectrum of Lenna Image (Grayscale)")

    # 選定 k 值進行壓縮
    k = 50
    compressed_img = svd_compress_grayscale(gray_img, k)

    # 顯示原圖與壓縮後圖像比較
    show_comparison(gray_img, compressed_img, k)

    # 儲存結果（可選）
    save_image(compressed_img, "compressed_lenna_grayscale.png")
