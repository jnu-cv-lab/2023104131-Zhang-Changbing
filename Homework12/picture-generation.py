import cv2
import numpy as np
import os
import random

def generate_checkerboard(rows=9, cols=6, square_size=80):
    squares_rows = rows + 1
    squares_cols = cols + 1
    h = squares_rows * square_size
    w = squares_cols * square_size
    img = np.ones((h, w), dtype=np.uint8) * 255
    for i in range(squares_rows):
        for j in range(squares_cols):
            if (i + j) % 2 == 0:
                img[i*square_size:(i+1)*square_size, j*square_size:(j+1)*square_size] = 0
    return img

def apply_affine_transform(img, angle=0, scale=1.0, tx=0, ty=0, output_size=(640, 480)):
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    transformed = cv2.warpAffine(img, M, (w, h), borderValue=255)
    if output_size:
        transformed = cv2.resize(transformed, output_size)
    return transformed

def is_detectable(img, checkerboard=(9,6)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    ret, _ = cv2.findChessboardCorners(gray, checkerboard, None)
    return ret

base = generate_checkerboard(9, 6, 80)

out_dir = "calibration_images"
os.makedirs(out_dir, exist_ok=True)

target_count = 20   # 目标成功图片数量
success_count = 0
max_attempts = 200  # 防止死循环

# 温和的随机参数范围
angle_range = (-8, 8)
scale_range = (0.85, 1.15)
tx_range = (-40, 40)
ty_range = (-40, 40)

attempt = 0
while success_count < target_count and attempt < max_attempts:
    attempt += 1
    angle = random.uniform(*angle_range)
    scale = random.uniform(*scale_range)
    tx = random.uniform(*tx_range)
    ty = random.uniform(*ty_range)
    
    img = apply_affine_transform(base, angle, scale, tx, ty, output_size=(640, 480))
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if is_detectable(img_color, (9,6)):
        filename = f"{out_dir}/calib_{success_count+1:02d}.jpg"
        cv2.imwrite(filename, img_color)
        success_count += 1
        print(f"成功生成第 {success_count} 张图片: {filename}")
    else:
        print(f"尝试 {attempt}: 检测失败，跳过")

print(f"完成！共生成 {success_count} 张可检测的标定图片。")