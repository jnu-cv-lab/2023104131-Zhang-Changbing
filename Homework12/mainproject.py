import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# ==================== 配置参数 ====================

# 棋盘格内角点数量（列数，行数）—— 即黑白方格交界点的数量
CHECKERBOARD = (9, 6)         
SQUARE_SIZE = 25               

# 标定图片所在文件夹
IMAGE_FOLDER = "./calibration_images/*.jpg" 

# 是否显示角点检测结果（每张图显示200ms）
SHOW_CHESSBOARD = True

# ==================== 1. 生成棋盘格三维世界坐标 ====================
# 棋盘格角点在标定板坐标系中的坐标 (X, Y, Z=0)
pattern_points = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
pattern_points[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
pattern_points *= SQUARE_SIZE   # 单位：mm

# 存储所有图片检测到的角点
world_points = []   # 世界坐标（3D）
image_points = []   # 图像坐标（2D）

# ==================== 2. 读取图片并检测角点 ====================
images = glob.glob(IMAGE_FOLDER)
if len(images) == 0:
    print("错误：未找到标定图片，请检查路径！")
    exit()

print(f"找到 {len(images)} 张标定图片")

img_with_corners = []   # 用于保存绘制了角点的图片
success_count = 0

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"警告：无法读取图片 {fname}，跳过")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        success_count += 1
        world_points.append(pattern_points)

        # 亚像素精度优化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        image_points.append(corners_sub)

        # 绘制角点并保存用于可视化
        img_copy = img.copy()
        cv2.drawChessboardCorners(img_copy, CHECKERBOARD, corners_sub, ret)
        img_with_corners.append(img_copy)

        if SHOW_CHESSBOARD:
            cv2.imshow('Chessboard Corners', img_copy)
            cv2.waitKey(200)   # 每张图显示200ms，可改为0等待按键
    else:
        print(f"警告：未能检测到棋盘格角点：{fname}")

cv2.destroyAllWindows()
print(f"成功检测角点的图片数：{success_count}/{len(images)}")

if success_count < 5:
    print("标定图片数量不足，请至少使用15张图片，且姿态多样。")
    exit()

# ==================== 3. 相机标定 ====================
print("\n开始标定...")
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    world_points, image_points, gray.shape[::-1], None, None
)

print("\n=== 标定结果 ===")
print(f"重投影误差：{ret:.4f} 像素")
print("\n相机内参矩阵 K：")
print(camera_matrix)
print("\n畸变参数 D = [k1, k2, p1, p2, k3]：")
print(dist_coeffs.ravel())

# ==================== 4. 保存结果到文本文件 ====================
with open("calibration_results.txt", "w") as f:
    f.write("=== 相机标定结果 ===\n")
    f.write(f"棋盘格内角点：{CHECKERBOARD[0]} × {CHECKERBOARD[1]}\n")
    f.write(f"方格边长：{SQUARE_SIZE} mm\n")
    f.write(f"成功标定图片数：{success_count}\n")
    f.write(f"重投影误差：{ret:.6f} 像素\n\n")
    f.write("内参矩阵 K：\n")
    f.write(str(camera_matrix) + "\n\n")
    f.write("畸变参数 (k1, k2, p1, p2, k3)：\n")
    f.write(str(dist_coeffs.ravel()) + "\n")
print("\n标定结果已保存到 calibration_results.txt")

# ==================== 5. 去畸变处理并对比 ====================
if len(images) > 0:
    # 选择第一张图片进行去畸变
    img_orig = cv2.imread(images[0])
    if img_orig is not None:
        h, w = img_orig.shape[:2]

        # 直接去畸变
        img_undistorted = cv2.undistort(img_orig, camera_matrix, dist_coeffs)

        # 保存原图和去畸变图
        cv2.imwrite("original_image.jpg", img_orig)
        cv2.imwrite("undistorted_image.jpg", img_undistorted)

        # 使用 matplotlib 显示对比
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB))
        plt.title('去畸变后图像')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig('undistortion_comparison.png')
        plt.show()
        print("去畸变对比图已保存为 undistortion_comparison.png")
    else:
        print("无法读取图片进行去畸变演示")

# ==================== 6. 保存角点检测示例图（至少2张） ====================
for i, img_corners in enumerate(img_with_corners[:4]):
    cv2.imwrite(f"corners_detected_{i}.jpg", img_corners)
print("角点检测示例图已保存（corners_detected_0.jpg 等）")

# ==================== 7. 外参示例 ====================
print("\n每张图片的外参（旋转向量 rvecs 和平移向量 tvecs）已保存在 rvecs, tvecs 中。")
print("例如第 1 张图片的旋转向量：", rvecs[0].ravel())
print("第 1 张图片的平移向量：", tvecs[0].ravel())

print("\n标定完成！")