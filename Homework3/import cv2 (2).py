import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def find_opencv_sample_image(image_name="lena.jpg"):
    # 方法1：使用 cv2.samples.findFile（OpenCV 3+）
    try:
        path = cv2.samples.findFile(image_name)
        if os.path.exists(path):
            return path
    except:
        pass
    # 方法2：常见安装路径
    possible_paths = [
        os.path.join(cv2.__path__[0], "samples/data", image_name),
        os.path.join(cv2.__path__[0], "../samples/data", image_name),
        os.path.join("/usr/share/opencv4/samples/data", image_name),
        os.path.join("/usr/local/share/opencv4/samples/data", image_name),
        image_name  # 当前目录
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def mse(img1, img2):
    """均方误差"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

def psnr(img1, img2):
    """峰值信噪比"""
    m = mse(img1, img2)
    if m == 0:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / m)

def show_spectrum(img, title, ax):
    """显示傅里叶频谱（中心化+对数）"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log(1 + magnitude)
    ax.imshow(log_magnitude, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

def block_dct(img, block_size=8):
    """对图像进行分块 DCT（块大小 block_size x block_size）"""
    h, w = img.shape
    dct_coeffs = np.zeros_like(img, dtype=float)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            dct_coeffs[i:i+block_size, j:j+block_size] = cv2.dct(block.astype(float))
    return dct_coeffs

def dct_low_energy_ratio(dct_coeffs, block_size=8, low_ratio=0.5):
    """
    计算 DCT 系数中低频区域能量占比
    低频区域定义为每个块左上角 (low_ratio*block_size) x (low_ratio*block_size) 的区域
    """
    h, w = dct_coeffs.shape
    total_energy = np.sum(dct_coeffs ** 2)
    if total_energy == 0:
        return 0
    low_energy = 0
    low_h = int(block_size * low_ratio)
    low_w = int(block_size * low_ratio)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = dct_coeffs[i:i+block_size, j:j+block_size]
            low_energy += np.sum(block[:low_h, :low_w] ** 2)
    return low_energy / total_energy

def main():
    # 1. 加载图像（优先使用 OpenCV 自带 Lena）
    img_path = find_opencv_sample_image("lena.jpg")
    if img_path is None:
        print("未找到 OpenCV 示例图片 lena.jpg，将创建一个 512x512 棋盘格图像用于演示。")
        img = np.zeros((512, 512), dtype=np.uint8)
        # 生成棋盘格（8x8 方格）
        for i in range(64):
            for j in range(64):
                if (i + j) % 2 == 0:
                    img[i*8:(i+1)*8, j*8:(j+1)*8] = 255
    else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(f"已加载图像: {img_path}, 尺寸: {img.shape}")

    # 确保图像尺寸为偶数（方便下采样）
    h, w = img.shape
    img = img[:h - h%2, :w - w%2]
    h, w = img.shape
    print(f"预处理后尺寸: {h} x {w}")

    # 2. 下采样（缩小 1/2）
    # 方法 A：直接下采样
    down_direct = img[::2, ::2]
    # 方法 B：先高斯平滑再下采样
    gaussian_blurred = cv2.GaussianBlur(img, (5,5), 1.0)
    down_gaussian = gaussian_blurred[::2, ::2]

    # 3. 图像恢复（将缩小图放大回原尺寸）
    h_orig, w_orig = h, w
    restore_nearest = cv2.resize(down_direct, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
    restore_bilinear = cv2.resize(down_direct, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    restore_bicubic = cv2.resize(down_direct, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)

    # 4. 空间域质量评价
    print("\n=== 空间域质量评价 ===")
    print(f"MSE (最近邻): {mse(img, restore_nearest):.4f}")
    print(f"PSNR (最近邻): {psnr(img, restore_nearest):.2f} dB")
    print(f"MSE (双线性): {mse(img, restore_bilinear):.4f}")
    print(f"PSNR (双线性): {psnr(img, restore_bilinear):.2f} dB")
    print(f"MSE (双三次): {mse(img, restore_bicubic):.4f}")
    print(f"PSNR (双三次): {psnr(img, restore_bicubic):.2f} dB")

    # 5. 显示图像（空间域）
    plt.figure(figsize=(14, 10))
    plt.subplot(2,3,1); plt.imshow(img, cmap='gray'); plt.title("原图"); plt.axis('off')
    plt.subplot(2,3,2); plt.imshow(down_direct, cmap='gray'); plt.title("直接缩小 (1/2)"); plt.axis('off')
    plt.subplot(2,3,3); plt.imshow(restore_nearest, cmap='gray'); plt.title("最近邻恢复"); plt.axis('off')
    plt.subplot(2,3,4); plt.imshow(restore_bilinear, cmap='gray'); plt.title("双线性恢复"); plt.axis('off')
    plt.subplot(2,3,5); plt.imshow(restore_bicubic, cmap='gray'); plt.title("双三次恢复"); plt.axis('off')
    plt.subplot(2,3,6); plt.imshow(down_gaussian, cmap='gray'); plt.title("高斯滤波后缩小"); plt.axis('off')
    plt.tight_layout()
    plt.savefig("spatial_comparison.png")
    plt.show()

    # 6. 傅里叶变换分析
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    show_spectrum(img, "原图频谱", axes[0,0])
    show_spectrum(down_direct, "直接缩小图频谱", axes[0,1])
    show_spectrum(restore_bilinear, "双线性恢复图频谱", axes[0,2])
    show_spectrum(down_gaussian, "高斯平滑后缩小频谱", axes[1,0])
    show_spectrum(restore_nearest, "最近邻恢复频谱", axes[1,1])
    show_spectrum(restore_bicubic, "双三次恢复频谱", axes[1,2])
    plt.tight_layout()
    plt.savefig("fft_spectra.png")
    plt.show()

    # 7. DCT 分块分析（块大小 8x8，低频区域占块面积 1/4）
    block_size = 8
    low_ratio = 0.5   # 低频区域为 4x4
    dct_original = block_dct(img, block_size)
    dct_nearest = block_dct(restore_nearest, block_size)
    dct_bilinear = block_dct(restore_bilinear, block_size)
    dct_bicubic = block_dct(restore_bicubic, block_size)

    ratio_original = dct_low_energy_ratio(dct_original, block_size, low_ratio)
    ratio_nearest = dct_low_energy_ratio(dct_nearest, block_size, low_ratio)
    ratio_bilinear = dct_low_energy_ratio(dct_bilinear, block_size, low_ratio)
    ratio_bicubic = dct_low_energy_ratio(dct_bicubic, block_size, low_ratio)

    print("\n=== DCT 能量分布（低频区域能量占比）===")
    print(f"原图: {ratio_original:.4f}")
    print(f"最近邻恢复: {ratio_nearest:.4f}")
    print(f"双线性恢复: {ratio_bilinear:.4f}")
    print(f"双三次恢复: {ratio_bicubic:.4f}")

    # 显示 DCT 系数图（取对数）
    def show_dct(dct_coeffs, title, ax):
        log_coeffs = np.log(1 + np.abs(dct_coeffs))
        ax.imshow(log_coeffs, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    show_dct(dct_original, "原图 DCT 系数", axes[0,0])
    show_dct(dct_nearest, "最近邻恢复 DCT", axes[0,1])
    show_dct(dct_bilinear, "双线性恢复 DCT", axes[1,0])
    show_dct(dct_bicubic, "双三次恢复 DCT", axes[1,1])
    plt.tight_layout()
    plt.savefig("dct_coeffs.png")
    plt.show()

    print("\n实验完成。生成的文件：spatial_comparison.png, fft_spectra.png, dct_coeffs.png")

if __name__ == "__main__":
    main()