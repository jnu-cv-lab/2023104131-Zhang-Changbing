import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']   # 使用文泉驿
# 或者 ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

def generate_checkerboard(size=512, squares=16):
    """生成棋盘格测试图"""
    img = np.zeros((size, size), dtype=np.uint8)
    block = size // squares
    for i in range(squares):
        for j in range(squares):
            if (i + j) % 2 == 0:
                img[i*block:(i+1)*block, j*block:(j+1)*block] = 255
    return img

def generate_chirp(size=512, f0=0, f1=0.5):
    """
    生成二维线性调频信号（chirp），频率从中心向四周线性增加
    f0: 中心最小归一化频率（0~0.5），f1: 边缘最大归一化频率
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X**2 + Y**2)
    # 频率随半径线性增加
    freq = f0 + (f1 - f0) * r
    phase = 2 * np.pi * freq * r
    chirp = 127 + 127 * np.sin(phase)
    return chirp.astype(np.uint8)

def downsample(img, factor=2):
    """直接下采样（隔点抽取）"""
    return img[::factor, ::factor]

def downsample_with_gaussian(img, sigma=1.0, factor=2):
    """先高斯平滑再下采样"""
    # 高斯核大小自动根据 sigma 确定（6*sigma+1 确保足够大）
    ksize = int(6 * sigma) + 1
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    return blurred[::factor, ::factor]

def compute_spectrum(img):
    """计算傅里叶频谱（中心化+对数）"""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_magnitude = np.log(1 + magnitude)
    return log_magnitude

def local_gradient_magnitude(img):
    """计算局部梯度幅值（Sobel）"""
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x**2 + grad_y**2)
    return mag

def estimate_local_M(grad_mag, block_size=32, percentile=80):
    """
    根据局部梯度幅值估计每个块的下采样因子 M
    梯度大的区域 M 应较小（保留更多细节），梯度小的区域 M 可以较大
    返回与图像等大的 M 矩阵（每个像素对应的 M 值，实际用块平均值）
    """
    h, w = grad_mag.shape
    M_map = np.ones((h, w), dtype=np.float32) * 4  # 默认 M=4
    # 分块计算平均梯度
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = grad_mag[i:i+block_size, j:j+block_size]
            mean_grad = np.mean(block)
            # 根据梯度大小映射 M 值（梯度越大 M 越小）
            # 设定梯度阈值：低梯度区域 M=4，高梯度区域 M=2
            if mean_grad < np.percentile(grad_mag, 30):
                M_val = 4
            elif mean_grad < np.percentile(grad_mag, 70):
                M_val = 3
            else:
                M_val = 2
            M_map[i:i+block_size, j:j+block_size] = M_val
    return M_map

def adaptive_downsample(img, M_map):
    """
    根据局部 M 值自适应下采样
    对每个像素，以其为中心的 M×M 区域做高斯滤波（σ=0.45M），然后抽取中心像素
    为简化，分块处理
    """
    h, w = img.shape
    result = np.zeros((h, w), dtype=np.uint8)
    # 对每个可能的 M 值分别处理
    unique_M = np.unique(M_map).astype(int)
    for M in unique_M:
        mask = (M_map == M)
        for i in range(0, h, M):
            for j in range(0, w, M):
                # 如果该块内大部分像素属于当前 M
                block_mask = mask[i:i+M, j:j+M]
                if np.mean(block_mask) > 0.5:
                    # 对该块进行高斯滤波（σ=0.45*M）
                    sigma = 0.45 * M
                    ksize = int(6*sigma) + 1
                    if ksize % 2 == 0:
                        ksize += 1
                    block = img[i:i+M, j:j+M]
                    if block.shape[0] == M and block.shape[1] == M:
                        blurred = cv2.GaussianBlur(block, (ksize, ksize), sigma)
                        # 取中心像素作为下采样结果
                        center = blurred[M//2, M//2]
                        # 将结果赋值到输出图像对应位置（用该值填充整个块，模拟下采样后放大）
                        result[i:i+M, j:j+M] = center
                    else:
                        # 边界块直接保留原值
                        result[i:i+M, j:j+M] = img[i:i+M, j:j+M]
    return result

def main():
    # 生成测试图像
    size = 512
    checker = generate_checkerboard(size, squares=16)
    chirp = generate_chirp(size, f0=0.01, f1=0.495)   

    # 第一部分：观察混叠
    factor = 2
    # 直接下采样
    down_direct = downsample(checker, factor)
    down_direct_chirp = downsample(chirp, factor)
    # 高斯平滑后下采样 (σ=0.45*factor=0.9)
    sigma = 0.45 * factor
    down_gaussian = downsample_with_gaussian(checker, sigma, factor)
    down_gaussian_chirp = downsample_with_gaussian(chirp, sigma, factor)
    
    # 计算频谱
    spec_checker = compute_spectrum(checker)
    spec_chirp = compute_spectrum(chirp)
    spec_direct = compute_spectrum(down_direct)
    spec_direct_chirp = compute_spectrum(down_direct_chirp)
    spec_gaussian = compute_spectrum(down_gaussian)
    spec_gaussian_chirp = compute_spectrum(down_gaussian_chirp)
    
    # 显示第一部分结果
    fig1, axes = plt.subplots(3, 4, figsize=(16, 12))
    titles = ['棋盘格原图', 'Chirp原图', '直接下采样棋盘格', '直接下采样Chirp',
              '高斯滤波后下采样棋盘格', '高斯滤波后下采样Chirp', '棋盘格频谱', 'Chirp频谱',
              '直接下采样频谱', '直接下采样Chirp频谱', '高斯下采样频谱', '高斯下采样Chirp频谱']
    images = [checker, chirp, down_direct, down_direct_chirp,
              down_gaussian, down_gaussian_chirp, spec_checker, spec_chirp,
              spec_direct, spec_direct_chirp, spec_gaussian, spec_gaussian_chirp]
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('part1_aliasing.png')
    plt.show()
    
    # 第二部分：固定 M=4，不同 σ 的影响
    M = 4
    sigmas = [0.5, 1.0, 2.0, 4.0]
    # 使用 chirp 图测试
    img_test = chirp
    results = []
    specs = []
    for s in sigmas:
        down = downsample_with_gaussian(img_test, s, M)
        results.append(down)
        specs.append(compute_spectrum(down))
    
    # 显示结果
    fig2, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, (s, down, spec) in enumerate(zip(sigmas, results, specs)):
        axes[0, i].imshow(down, cmap='gray')
        axes[0, i].set_title(f'σ={s}, M={M}')
        axes[0, i].axis('off')
        axes[1, i].imshow(spec, cmap='gray')
        axes[1, i].set_title(f'频谱 σ={s}')
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig('part2_sigma_comparison.png')
    plt.show()
    
    # 计算 PSNR 或混叠能量，找到最佳 σ
    # 这里简单比较频谱高频能量（高频部分方差）
    # 理论最佳 σ = 0.45 * M = 1.8
    high_freq_energy = []
    for spec in specs:
        # 只考虑高频区（距离中心大于 0.3*size）
        h, w = spec.shape
        center = (h//2, w//2)
        high_mask = np.zeros_like(spec, dtype=bool)
        for i in range(h):
            for j in range(w):
                if (i-center[0])**2 + (j-center[1])**2 > (0.3*min(h,w))**2:
                    high_mask[i,j] = True
        energy = np.sum(spec[high_mask]**2)
        high_freq_energy.append(energy)
    best_idx = np.argmin(high_freq_energy)
    print(f"第二部分：不同σ下高频能量：{high_freq_energy}")
    print(f"最佳σ = {sigmas[best_idx]}, 理论最佳 = 1.8")
    
    # 第三部分：自适应下采样
    # 使用自然图像（这里用 chirp 或棋盘格，也可以读入真实图像）
    img_adapt = chirp
    # 计算梯度幅值
    grad = local_gradient_magnitude(img_adapt)
    # 估计局部 M 值
    M_map = estimate_local_M(grad, block_size=32)
    # 自适应下采样
    adapt_down = adaptive_downsample(img_adapt, M_map)
    # 统一下采样（用 M=4 直接下采样）
    uniform_down = downsample(img_adapt, 4)
    # 将统一下采样放大到原尺寸以便比较误差（最近邻放大）
    uniform_up = cv2.resize(uniform_down, (size, size), interpolation=cv2.INTER_NEAREST)
    # 计算误差图
    error_adapt = np.abs(img_adapt.astype(float) - adapt_down.astype(float))
    error_uniform = np.abs(img_adapt.astype(float) - uniform_up.astype(float))
    
    # 显示自适应下采样结果
    fig3, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0,0].imshow(img_adapt, cmap='gray')
    axes[0,0].set_title('原图')
    axes[0,0].axis('off')
    axes[0,1].imshow(grad, cmap='jet')
    axes[0,1].set_title('梯度幅值')
    axes[0,1].axis('off')
    axes[0,2].imshow(M_map, cmap='jet')
    axes[0,2].set_title('局部M值 (2~4)')
    axes[0,2].axis('off')
    axes[1,0].imshow(adapt_down, cmap='gray')
    axes[1,0].set_title('自适应下采样结果')
    axes[1,0].axis('off')
    axes[1,1].imshow(uniform_up, cmap='gray')
    axes[1,1].set_title('统一下采样(M=4)放大')
    axes[1,1].axis('off')
    axes[1,2].imshow(error_uniform, cmap='hot')
    axes[1,2].set_title('统一下采样误差')
    axes[1,2].axis('off')
    # 额外显示自适应误差
    fig4, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].imshow(error_adapt, cmap='hot')
    ax[0].set_title('自适应下采样误差')
    ax[0].axis('off')
    ax[1].imshow(error_uniform - error_adapt, cmap='seismic')
    ax[1].set_title('误差差异 (统一 - 自适应)')
    ax[1].axis('off')
    plt.tight_layout()
    plt.savefig('part3_adaptive.png')
    plt.show()
    
    print("实验完成！所有结果已保存为图片文件。")

if __name__ == "__main__":
    main()