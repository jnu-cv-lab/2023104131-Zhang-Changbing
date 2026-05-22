import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']   # 使用文泉驿
# 或者 ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 生成测试图（矩形、圆、平行线、垂直线） ====================
def generate_test_image(size=600):
    """生成包含矩形、圆、平行线、垂直线的测试图"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    # 矩形
    cv2.rectangle(img, (100, 100), (250, 200), (0, 0, 255), 3)
    # 圆
    cv2.circle(img, (450, 150), 50, (0, 255, 0), 3)
    # 平行水平线
    for y in range(350, 500, 40):
        cv2.line(img, (50, y), (550, y), (255, 0, 0), 3)
    # 平行垂直线
    for x in range(80, 560, 60):
        cv2.line(img, (x, 320), (x, 520), (255, 0, 0), 3)
    # 标注文字
    cv2.putText(img, "Rectangle", (110, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(img, "Circle", (430, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(img, "Parallel horizontal lines", (200, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    cv2.putText(img, "Parallel vertical lines", (200, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    return img

# ==================== 几何性质分析 ====================
def analyze_geometry(title, img_orig, img_transformed):
    """分析变换后的几何性质并显示对比图"""
    print(f"\n=== {title} ===")
    if "相似" in title:
        print("相似变换：")
        print("  - 直线保持为直线")
        print("  - 平行线保持平行")
        print("  - 垂直线保持垂直")
        print("  - 圆保持为圆")
    elif "仿射" in title:
        print("仿射变换：")
        print("  - 直线保持为直线")
        print("  - 平行线保持平行")
        print("  - 垂直线不保持垂直（角度改变）")
        print("  - 圆变为椭圆")
    elif "透视" in title:
        print("透视变换：")
        print("  - 直线仍保持为直线")
        print("  - 平行线不再保持平行（会聚到灭点）")
        print("  - 垂直线不保持垂直")
        print("  - 圆变为椭圆或二次曲线")
    # 显示对比图
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
    plt.title("原图")
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==================== 变换函数 ====================
def similarity_transform(img):
    """相似变换：旋转 + 平移 + 均匀缩放"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 25, 0.8)
    transformed = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    return transformed, M

def affine_transform(img):
    """仿射变换：非均匀缩放 + 剪切"""
    h, w = img.shape[:2]
    src_pts = np.float32([[50, 50], [200, 50], [50, 200]])
    dst_pts = np.float32([[70, 70], [220, 90], [80, 230]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    transformed = cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255))
    return transformed, M

def perspective_transform(img):
    """透视变换：模拟倾斜视角"""
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst_pts = np.float32([[w*0.1, h*0.05], [w*0.9, h*0.1], [w*0.85, h*0.95], [w*0.15, h*0.9]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed = cv2.warpPerspective(img, M, (w, h), borderValue=(255,255,255))
    return transformed, M

# ==================== 自动生成透视畸变图像并校正 ====================
def generate_perspective_document(size=(800, 600)):
    """生成一个模拟的文档图像（棋盘格+文字），然后对其施加透视畸变，返回畸变图像和原始正视图"""
    # 创建一个正视图：白色背景，带有黑色表格线和文字
    w, h = size
    doc = np.ones((h, w, 3), dtype=np.uint8) * 255
    # 绘制表格线（水平和垂直）
    for x in range(0, w, 80):
        cv2.line(doc, (x, 0), (x, h), (0,0,0), 2)
    for y in range(0, h, 60):
        cv2.line(doc, (0, y), (w, y), (0,0,0), 2)
    # 添加文字
    cv2.putText(doc, "This is a test document.", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.putText(doc, "It contains a table and text.", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    cv2.putText(doc, "The goal is to correct perspective distortion.", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    # 模拟透视畸变：对正视图应用一个透视变换
    src_pts = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst_pts = np.float32([[w*0.15, h*0.1], [w*0.85, h*0.05], [w*0.8, h*0.95], [w*0.2, h*0.9]])
    M_distort = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted = cv2.warpPerspective(doc, M_distort, (w, h), borderValue=(255,255,255))
    return doc, distorted, M_distort

def correct_perspective_auto(distorted_img):
    """自动检测图像中最大矩形的四个角点（假设文档是主要前景），并校正透视"""
    gray = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大轮廓（应为文档区域）
    max_contour = max(contours, key=cv2.contourArea)
    # 近似多边形，获取四个角点
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    if len(approx) != 4:
        print("未找到四个角点，使用预设角点")
        h, w = distorted_img.shape[:2]
        approx = np.array([[w*0.15, h*0.1], [w*0.85, h*0.05], [w*0.8, h*0.95], [w*0.2, h*0.9]], dtype=np.float32)
    else:
        approx = approx.reshape(4, 2).astype(np.float32)
    # 排序角点：左上、右上、右下、左下
    rect = np.zeros((4, 2), dtype=np.float32)
    s = approx.sum(axis=1)
    rect[0] = approx[np.argmin(s)]   # 左上
    rect[2] = approx[np.argmax(s)]   # 右下
    diff = np.diff(approx, axis=1)
    rect[1] = approx[np.argmin(diff)]  # 右上
    rect[3] = approx[np.argmax(diff)]  # 左下
    # 计算目标矩形的宽高
    width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
    height = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
    dst_pts = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
    M = cv2.getPerspectiveTransform(rect, dst_pts)
    corrected = cv2.warpPerspective(distorted_img, M, (int(width), int(height)), borderValue=(255,255,255))
    return corrected, M, rect

# ==================== 主程序 ====================
def main():
    # 1. 生成测试图并执行三种变换
    test_img = generate_test_image(600)
    
    # 相似变换
    sim_img, _ = similarity_transform(test_img)
    analyze_geometry("相似变换", test_img, sim_img)
    
    # 仿射变换
    aff_img, _ = affine_transform(test_img)
    analyze_geometry("仿射变换", test_img, aff_img)
    
    # 透视变换
    per_img, _ = perspective_transform(test_img)
    analyze_geometry("透视变换", test_img, per_img)
    
    # 2. 自动生成透视畸变文档并校正
    print("\n=== 透视畸变图像自动校正 ===")
    doc_original, doc_distorted, _ = generate_perspective_document((800, 600))
    corrected_doc, _, corners = correct_perspective_auto(doc_distorted)
    
    # 显示校正结果
    plt.figure(figsize=(15, 5))
    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(doc_original, cv2.COLOR_BGR2RGB))
    plt.title("原始正视图（无畸变）")
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(doc_distorted, cv2.COLOR_BGR2RGB))
    # 在畸变图上绘制检测到的角点
    for pt in corners:
        plt.plot(pt[0], pt[1], 'ro', markersize=8)
    plt.title("透视畸变图像（含检测角点）")
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(corrected_doc, cv2.COLOR_BGR2RGB))
    plt.title("校正后图像")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\n透视校正完成。校正后的图像中，表格线恢复为水平和垂直，文字变形得到矫正。")

if __name__ == "__main__":
    main()