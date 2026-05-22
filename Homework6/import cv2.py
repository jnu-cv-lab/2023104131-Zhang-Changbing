import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']   # 使用文泉驿
# 或者 ['Noto Sans CJK SC']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 辅助函数 ====================
def draw_matches(img1, kp1, img2, kp2, matches, title, max_matches=50):
    """绘制匹配线图"""
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_matches], None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
    return img_matches

def draw_keypoints(img, kp, title, save_path=None):
    """绘制关键点"""
    img_kp = cv2.drawKeypoints(img, kp, None, color=(0,255,0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def compute_homography_and_inliers(kp1, kp2, matches, reproj_thresh=5.0):
    """计算单应矩阵并筛选内点"""
    if len(matches) < 4:
        return None, None, 0
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    if mask is not None:
        inliers = mask.ravel().tolist().count(1)
    else:
        inliers = 0
    return H, mask, inliers

def draw_inlier_matches(img1, kp1, img2, kp2, matches, mask, title):
    """绘制RANSAC内点匹配"""
    if mask is None:
        print("无法绘制内点匹配：mask为空")
        return None
    inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
    img_inliers = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15, 8))
    plt.imshow(cv2.cvtColor(img_inliers, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
    return img_inliers

def draw_target_box(img, corners, title):
    """在场景图中画出目标边框"""
    img_out = img.copy()
    # 修复：确保 corners 是 (N,2) 的 int32 类型
    if len(corners.shape) == 3:
        corners = corners.reshape(-1, 2)
    corners = np.int32(corners)
    cv2.polylines(img_out, [corners], True, (0, 255, 0), 3)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# ==================== 任务 1：ORB 特征检测 ====================
def task1_orb_detection(box_path, scene_path, nfeatures=1000):
    img_box = cv2.imread(box_path)
    img_scene = cv2.imread(scene_path)
    if img_box is None or img_scene is None:
        print(f"错误：无法读取图像，请检查 {box_path} 和 {scene_path} 是否存在")
        return None, None, None, None, None, None
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp_box, des_box = orb.detectAndCompute(img_box, None)
    kp_scene, des_scene = orb.detectAndCompute(img_scene, None)
    print(f"任务1：nfeatures={nfeatures}")
    print(f"  box.png 关键点数量：{len(kp_box)}")
    print(f"  box_in_scene.png 关键点数量：{len(kp_scene)}")
    if des_box is not None:
        print(f"  描述子维度：{des_box.shape[1]}")
    else:
        print("  描述子维度：无")
    draw_keypoints(img_box, kp_box, f"box.png 关键点 (nfeatures={nfeatures})", f"box_keypoints_{nfeatures}.png")
    draw_keypoints(img_scene, kp_scene, f"scene.png 关键点 (nfeatures={nfeatures})", f"scene_keypoints_{nfeatures}.png")
    return img_box, img_scene, kp_box, kp_scene, des_box, des_scene

# ==================== 任务 2：ORB 特征匹配 ====================
def task2_orb_matching(des1, des2, kp1, kp2, img1, img2, max_matches=50):
    if des1 is None or des2 is None:
        print("描述子为空，无法匹配")
        return []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print(f"\n任务2：总匹配数量 = {len(matches)}")
    if len(matches) > 0:
        draw_matches(img1, kp1, img2, kp2, matches, f"ORB 初始匹配 (前{max_matches}对)", max_matches)
    else:
        print("未找到任何匹配")
    return matches

# ==================== 任务 3：RANSAC 剔除错误匹配 ====================
def task3_ransac(matches, kp1, kp2, img1, img2, reproj_thresh=5.0):
    H, mask, inliers = compute_homography_and_inliers(kp1, kp2, matches, reproj_thresh)
    total_matches = len(matches)
    inlier_ratio = inliers / total_matches if total_matches > 0 else 0
    print(f"\n任务3：")
    if H is not None:
        print(f"  Homography 矩阵:\n{H}")
    else:
        print("  Homography 矩阵：无法计算（匹配点不足或共线）")
    print(f"  总匹配数量：{total_matches}")
    print(f"  RANSAC 内点数量：{inliers}")
    print(f"  内点比例：{inlier_ratio:.4f}")
    if mask is not None and len(matches) > 0:
        draw_inlier_matches(img1, kp1, img2, kp2, matches, mask, f"RANSAC 内点匹配 (内点比例={inlier_ratio:.2f})")
    return H, mask, inliers

# ==================== 任务 4：目标定位 ====================
def task4_object_localization(box_img, scene_img, H):
    if H is None:
        print("\n任务4：无法进行目标定位（Homography 矩阵为空）")
        return
    h, w = box_img.shape[:2]
    box_corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]]).reshape(-1,1,2)
    transformed_corners = cv2.perspectiveTransform(box_corners, H)
    draw_target_box(scene_img, transformed_corners, "目标定位结果 (绿色边框)")
    print("\n任务4：目标定位完成。")

# ==================== 任务 6：参数对比实验 ====================
def task6_parameter_comparison(box_path, scene_path, nfeatures_list):
    print("\n" + "="*50)
    print("任务6：参数对比实验 (nfeatures 影响)")
    print("="*50)
    results = []
    for n in nfeatures_list:
        print(f"\n--- nfeatures = {n} ---")
        img_box, img_scene, kp_box, kp_scene, des_box, des_scene = task1_orb_detection(box_path, scene_path, n)
        if des_box is None or des_scene is None:
            print("描述子为空，跳过此参数")
            continue
        matches = task2_orb_matching(des_box, des_scene, kp_box, kp_scene, img_box, img_scene, max_matches=50)
        if len(matches) < 4:
            print("匹配点不足4个，跳过RANSAC和定位")
            results.append({
                'nfeatures': n,
                'box_kp': len(kp_box),
                'scene_kp': len(kp_scene),
                'matches': len(matches),
                'inliers': 0,
                'inlier_ratio': 0,
                'success': False
            })
            continue
        H, mask, inliers = task3_ransac(matches, kp_box, kp_scene, img_box, img_scene, reproj_thresh=5.0)
        inlier_ratio = inliers / len(matches) if len(matches)>0 else 0
        # 判断定位是否成功
        success = False
        if H is not None:
            try:
                h, w = img_box.shape[:2]
                corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]]).reshape(-1,1,2)
                transformed = cv2.perspectiveTransform(corners, H)
                success = not np.any(np.isnan(transformed)) and not np.any(np.isinf(transformed))
            except:
                success = False
        results.append({
            'nfeatures': n,
            'box_kp': len(kp_box),
            'scene_kp': len(kp_scene),
            'matches': len(matches),
            'inliers': inliers,
            'inlier_ratio': inlier_ratio,
            'success': success
        })
    print("\n参数对比结果汇总表：")
    print(f"{'nfeatures':<10} {'模板图关键点':<12} {'场景图关键点':<12} {'匹配数量':<10} {'RANSAC内点':<10} {'内点比例':<10} {'定位成功':<10}")
    for r in results:
        print(f"{r['nfeatures']:<10} {r['box_kp']:<12} {r['scene_kp']:<12} {r['matches']:<10} {r['inliers']:<10} {r['inlier_ratio']:<10.4f} {r['success']:<10}")
    return results

# ==================== 选做任务：SIFT 特征匹配 ====================
def optional_sift(box_path, scene_path):
    print("\n" + "="*50)
    print("选做任务：SIFT 特征匹配对比")
    print("="*50)
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        print("当前 OpenCV 版本不支持 SIFT，请安装 opencv-contrib-python")
        return None
    img_box = cv2.imread(box_path)
    img_scene = cv2.imread(scene_path)
    if img_box is None or img_scene is None:
        print("无法读取图像")
        return None
    kp1, des1 = sift.detectAndCompute(img_box, None)
    kp2, des2 = sift.detectAndCompute(img_scene, None)
    if des1 is None or des2 is None:
        print("SIFT 未检测到足够特征点")
        return None
    print(f"SIFT 关键点数量：box={len(kp1)}, scene={len(kp2)}")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(f"SIFT 匹配数量（Lowe ratio test后）：{len(good)}")
    if len(good) >= 4:
        H, mask, inliers = compute_homography_and_inliers(kp1, kp2, good, reproj_thresh=5.0)
        inlier_ratio = inliers / len(good) if len(good)>0 else 0
        print(f"SIFT RANSAC 内点数量：{inliers}, 内点比例：{inlier_ratio:.4f}")
        if H is not None:
            h, w = img_box.shape[:2]
            box_corners = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]]).reshape(-1,1,2)
            transf_corners = cv2.perspectiveTransform(box_corners, H)
            draw_target_box(img_scene, transf_corners, "SIFT 目标定位结果")
        return {'matches': len(good), 'inliers': inliers, 'ratio': inlier_ratio, 'success': H is not None}
    else:
        print("SIFT 匹配点不足4个，无法计算Homography")
        return None

# ==================== 主程序 ====================
def main():
    box_path = "/home/chbing/Homework6/box.png"
    scene_path = "/home/chbing/Homework6/box_in_scene.png"
    
    print("=== 默认参数 nfeatures=1000 运行 ===")
    img_box, img_scene, kp_box, kp_scene, des_box, des_scene = task1_orb_detection(box_path, scene_path, nfeatures=1000)
    if des_box is None or des_scene is None:
        print("无法检测特征点，请检查图像文件")
        return
    matches = task2_orb_matching(des_box, des_scene, kp_box, kp_scene, img_box, img_scene, max_matches=50)
    if len(matches) >= 4:
        H, mask, inliers = task3_ransac(matches, kp_box, kp_scene, img_box, img_scene, reproj_thresh=5.0)
        task4_object_localization(img_box, img_scene, H)
    else:
        print("匹配点不足4个，跳过RANSAC和定位")
    
    # 任务6：参数对比
    nfeatures_list = [500, 1000, 2000]
    task6_parameter_comparison(box_path, scene_path, nfeatures_list)
    
    # 选做：SIFT
    optional_sift(box_path, scene_path)
    
    print("\n实验完成。请根据输出填写实验报告中的表格和回答问题。")

if __name__ == "__main__":
    main()