import os
import cv2
import numpy as np

def crop_depth_image_by_lidar_bounds(
    projected_depth_dir,
    depth_image_dir,
    output_dir
):
    """
    遍历 projected_depth_dir 中的每个图像，根据其非零像素的行索引范围，
    截取 depth_image_dir 中对应文件并保存到 output_dir。
    """
    # 若输出目录不存在，自动创建
    os.makedirs(output_dir, exist_ok=True)

    # 遍历投影雷达深度图文件
    for filename in os.listdir(projected_depth_dir):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue  # 跳过非图像文件
        
        # 构造完整路径
        lidar_path = os.path.join(projected_depth_dir, filename)
        # 相应的第三张 depth_image（这里假设同名，只在另一个目录）
        depth_path = os.path.join(depth_image_dir, filename)
        
        if not os.path.exists(depth_path):
            print(f"[WARN] 对应的 depth_image 不存在: {depth_path}")
            continue
        
        # 读取第一张 (projected_depth)
        lidar_img = cv2.imread(lidar_path, cv2.IMREAD_UNCHANGED)
        if lidar_img is None:
            print(f"[WARN] 读取投影雷达图失败: {lidar_path}")
            continue
        
        # 读取第三张 (depth_image)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"[WARN] 读取第三张深度图失败: {depth_path}")
            continue
        
        # 1) 根据第一张图的有效像素，计算上下边界
        #    假定雷达投影图是单通道（若多通道需改为灰度或取某个通道）
        #    这里用非零判断来定位最顶和最底
        non_zero = np.where(lidar_img > 0)
        if len(non_zero[0]) == 0:
            print(f"[INFO] {filename} 中没有非零像素，跳过裁剪。")
            continue
        
        top = np.min(non_zero[0])
        bottom = np.max(non_zero[0])
        
        # 2) 截取第三张图的行 [top, bottom]，列不变
        #    若两张图大小不一致，需要先判定是否可以直接对应行索引
        #    这里假定它们高度相同
        H_lidar, W_lidar = lidar_img.shape[:2]
        H_depth, W_depth = depth_img.shape[:2]
        
        if H_lidar != H_depth or W_lidar != W_depth:
            print(f"[WARN] 图像尺寸不一致: {lidar_path} vs {depth_path}")
            # 若高度不同，可以选择适当的映射策略，这里仅跳过
            continue
        
        # 执行裁剪（行方向上）
        # 注意：OpenCV 的图像数组是 [row, col] => depth_img[row_start: row_end+1, :]
        cropped_depth = depth_img[top:bottom+1, :]
        
        # 3) 保存结果
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, cropped_depth)
        
        print(f"[OK] 裁剪并保存: {out_path}, top={top}, bottom={bottom}")

def main():
    # 示例：请根据自己实际路径来修改
    dataset_root = "D:\Desktop\EECS568\Project\DenseFusionMap\data\slam"
    projected_depth_dir = os.path.join(dataset_root, "projected_depth")
    depth_image_dir = os.path.join(dataset_root, "rgb")
    output_dir = os.path.join(dataset_root, "clipped_rgb")

    crop_depth_image_by_lidar_bounds(
        projected_depth_dir,
        depth_image_dir,
        output_dir
    )

if __name__ == "__main__":
    main()
