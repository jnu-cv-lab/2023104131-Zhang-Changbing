import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

def main():
    # ===== 自动获取脚本所在目录，并创建 output 文件夹 =====
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    # ===== 任务1：使用文件对话框选择图片 =====
    print("任务1：请选择一张测试图片")
    # 隐藏 tkinter 主窗口
    root = tk.Tk()
    root.withdraw()
    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择一张图片",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
    )

    if not file_path:
        print("未选择任何文件，程序退出。")
        return

    # 读取图片
    img = cv2.imread(file_path)
    if img is None:
        print(f"错误：无法读取图片 {file_path}，请检查文件格式。")
        return

    print(f"成功加载图片：{file_path}")

    # ===== 任务2：输出图像基本信息 =====
    print("\n任务2：图像基本信息")
    height, width = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 1
    dtype = img.dtype
    print(f"图像尺寸：{width} x {height} (宽度 x 高度)")
    print(f"图像通道数：{channels}")
    print(f"像素数据类型：{dtype}")
    print(f"图像总像素数：{img.size}")

    # ===== 任务3：显示原图（转换为RGB供matplotlib显示）=====
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title('原图 (RGB)')
    plt.axis('off')

    # ===== 任务4：转换为灰度图 =====
    print("\n任务4：转换为灰度图")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.subplot(1, 4, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('灰度图')
    plt.axis('off')

    # ===== 任务5：保存灰度图 =====
    print("任务5：保存灰度图")
    gray_output_path = os.path.join(output_dir, 'gray_image.jpg')
    cv2.imwrite(gray_output_path, gray_img)
    print(f"灰度图已保存到：{gray_output_path}")

    # ===== 任务6：NumPy简单操作 =====
    print("\n任务6：NumPy简单操作")

    # 操作1：输出某个像素值（取图像中心点或(100,100)安全位置）
    x = min(100, width - 1)
    y = min(100, height - 1)
    pixel_bgr = img[y, x]
    pixel_rgb = img_rgb[y, x]
    print(f"像素点({x}, {y})的BGR值：{pixel_bgr}")
    print(f"像素点({x}, {y})的RGB值：{pixel_rgb}")

    # 操作2：裁剪左上角100x100区域（如果图像足够大）
    crop_size = 100
    if height >= crop_size and width >= crop_size:
        top_left_crop = img_rgb[:crop_size, :crop_size]
        plt.subplot(1, 4, 3)
        plt.imshow(top_left_crop)
        plt.title(f'左上角{crop_size}x{crop_size}裁剪')
        plt.axis('off')

        crop_output_path = os.path.join(output_dir, 'cropped_region.jpg')
        # 保存时转回BGR
        cv2.imwrite(crop_output_path, cv2.cvtColor(top_left_crop, cv2.COLOR_RGB2BGR))
        print(f"裁剪区域已保存到：{crop_output_path}")
    else:
        print(f"图像尺寸({width}x{height})小于{crop_size}x{crop_size}，跳过裁剪操作。")
        # 占位空白子图
        plt.subplot(1, 4, 3)
        plt.text(0.5, 0.5, '图像太小\n无法裁剪', ha='center', va='center')
        plt.axis('off')

    # 操作3：提取红色通道并显示
    if channels == 3:
        red_channel = img.copy()
        red_channel[:, :, 0] = 0  # 蓝色通道归零
        red_channel[:, :, 1] = 0  # 绿色通道归零
        red_channel_rgb = cv2.cvtColor(red_channel, cv2.COLOR_BGR2RGB)

        plt.subplot(1, 4, 4)
        plt.imshow(red_channel_rgb)
        plt.title('红色通道')
        plt.axis('off')
    else:
        plt.subplot(1, 4, 4)
        plt.text(0.5, 0.5, '非彩色图\n无红色通道', ha='center', va='center')
        plt.axis('off')

    # 调整布局并显示
    plt.tight_layout()
    plt.show()

    # 保存组合结果图
    result_path = os.path.join(output_dir, 'processed_results.png')
    plt.savefig(result_path, dpi=300, bbox_inches='tight')
    print(f"\n结果图已保存到：{result_path}")

    # 额外的NumPy操作：图像统计信息
    print("\n--- 图像统计信息 ---")
    print(f"原图 - 最小值：{np.min(img)}，最大值：{np.max(img)}，平均值：{np.mean(img):.2f}")
    print(f"灰度图 - 最小值：{np.min(gray_img)}，最大值：{np.max(gray_img)}，平均值：{np.mean(gray_img):.2f}")

    print("\n所有任务完成！")

if __name__ == "__main__":
    main()