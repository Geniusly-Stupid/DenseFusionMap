import numpy as np
import open3d as o3d
import cv2
import os

def load_calib(calib_file):
    """
    从 calib.txt 中读取相机内参 fx, fy, cx, cy。
    假设 calib.txt 中包含类似下面这一行 (KITTI 格式)：
      P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 
                 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 
                 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03
    则 fx = 7.215377e+02, cx = 6.095593e+02, fy = 7.215377e+02, cy = 1.728540e+02.
    如果 calib.txt 中相机内参行的前缀不是 P_rect_02，请根据实际情况修改下面的判断条件。
    """
    fx = fy = cx = cy = None
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 选择合适的标识符：例如 P_rect_02 或者 P2:
            if line.startswith('P_rect_02:') or line.startswith('P2:'):
                vals = line.split()
                if len(vals) < 13:
                    continue  # 数量不足则跳过
                # 从 vals[1:] 中解析出12个数字
                raw = vals[1:]
                # 假设 KITTI 的格式为:
                # [fx, 0, cx, Tx, 0, fy, cy, Ty, 0, 0, 1, 0]
                fx = float(raw[0])
                cx = float(raw[2])
                fy = float(raw[5])
                cy = float(raw[6])
                break
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(f"无法从 {calib_file} 中正确解析相机内参，请检查 calib 文件格式.")
    print(f"[load_calib] fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    return fx, fy, cx, cy

def load_poses(poses_file):
    """
    从 poses.txt 文件中读取位姿信息。
    每行应包含 3x4=12 个数字，表示 [R|t]（假定为 camera->world）。
    返回一个 list，每个元素为 (4,4) 的变换矩阵。
    """
    poses = []
    with open(poses_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split()))
            if len(vals) != 12:
                raise ValueError(f"第 {idx} 行不符合 3x4=12 个数字的格式: {line}")
            T = np.eye(4, dtype=np.float32)
            T[0:3, 0:4] = np.array(vals).reshape(3, 4)
            poses.append(T)
    print(f"[load_poses] 加载了 {len(poses)} 个位姿.")
    return poses

def create_point_cloud_from_depth(rgb_img, depth_img, fx, fy, cx, cy, depth_min=0.1, depth_max=1000.0):
    """
    根据彩色图和深度图以及相机内参（单位：米），构建当前帧相机坐标系下的点云与颜色信息。
    
    参数：
      rgb_img: 彩色图 (H x W x 3)，uint8 类型
      depth_img: 深度图 (H x W)，单位假定为米
      fx, fy, cx, cy: 相机内参
      depth_min, depth_max: 用于过滤的深度最小/最大值
    
    返回：
      points_cam: (N, 3) 相机坐标系下的 3D 点
      colors_valid: (N, 3) 对应颜色（已归一化至 [0, 1]）
    """
    # 确保彩色图和深度图尺寸一致
    assert rgb_img.shape[:2] == depth_img.shape[:2], "[create_point_cloud_from_depth] 彩色图与深度图尺寸不一致！"
    H, W = depth_img.shape[:2]
    
    # 生成像素坐标网格
    u_coord = np.arange(W)
    v_coord = np.arange(H)
    u_grid, v_grid = np.meshgrid(u_coord, v_coord)
    
    # 拉平数组
    u_grid_flat = u_grid.flatten().astype(np.float32)
    v_grid_flat = v_grid.flatten().astype(np.float32)
    depth_flat = depth_img.flatten().astype(np.float32)
    
    # 过滤无效深度
    valid_mask = (depth_flat > depth_min) & (depth_flat < depth_max)
    u_valid = u_grid_flat[valid_mask]
    v_valid = v_grid_flat[valid_mask]
    z_valid = depth_flat[valid_mask]
    
    # 像素投影到相机坐标系
    X = (u_valid - cx) * z_valid / fx
    Y = (v_valid - cy) * z_valid / fy
    Z = z_valid
    
    points_cam = np.stack((X, Y, Z), axis=-1)
    
    # 获取对应颜色
    rgb_flat = rgb_img.reshape(-1, 3)
    colors_valid = rgb_flat[valid_mask].astype(np.float32) / 255.0
    
    return points_cam, colors_valid

def transform_points(points, T):
    """
    使用 4x4 变换矩阵 T 将点云 (N, 3) 从相机坐标系转换到世界坐标系。
    T 应为相机->世界变换矩阵。
    """
    if points.shape[0] == 0:
        return points
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    homo_points = np.hstack([points, ones])
    transformed = (T @ homo_points.T).T
    return transformed[:, :3]

def main():
    # 请根据实际路径修改
    dataset_root = r"D:\Desktop\EECS568\Project\DenseFusionMap\data\slam"
    calib_file   = os.path.join(dataset_root, "calib.txt")
    poses_file   = os.path.join(dataset_root, "poses.txt")
    rgb_dir      = os.path.join(dataset_root, "clipped_rgb")
    depth_dir    = os.path.join(dataset_root, "clipped_depth")
    
    # 1. 读取相机内参
    fx, fy, cx, cy = load_calib(calib_file)
    if abs(fy) < 1e-6:
        raise ValueError("fy 解析结果为 0，请检查 calib 文件格式！")
    
    # 2. 读取位姿信息（假设为 camera->world）
    poses = load_poses(poses_file)
    
    max_frames = min(200, len(poses))
    
    all_points = []
    all_colors = []
    
    for i in range(max_frames):
        rgb_path = os.path.join(rgb_dir, f"{i:06d}.png")
        depth_path = os.path.join(depth_dir, f"{i:06d}.png")
        
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"[WARN] 第 {i} 帧图像或深度图不存在，跳过。")
            continue
        
        # 读取彩色图，并转换为 RGB
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            print(f"[WARN] 第 {i} 帧的彩色图读取失败。")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        # 读取深度图
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"[WARN] 第 {i} 帧的深度图读取失败。")
            continue
        
        # 如果深度图为 uint16，转换为米（假设原单位通过除以1000得到以米为单位）
        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) / 1000.0
        
        # 输出深度图数值范围，便于调试
        min_depth, max_depth = depth_img.min(), depth_img.max()
        print(f"[Frame {i}] depth range: ({min_depth:.4f}, {max_depth:.4f})")
        
        # 生成当前帧点云（可根据实际情况调整 depth_max）
        points_cam, colors = create_point_cloud_from_depth(
            rgb_img, depth_img, fx, fy, cx, cy,
            depth_min=0.1, depth_max=80.0
        )
        if points_cam.shape[0] == 0:
            print(f"[Frame {i}] 有效点数为 0，跳过。")
            continue
        
        # 获取当前帧的位姿 (camera->world)
        Tcw = poses[i]
        print(f"[Frame {i}] 有效点数: {points_cam.shape[0]}")
        
        # 点云转换到世界坐标系
        points_world = transform_points(points_cam, Tcw)
        
        all_points.append(points_world)
        all_colors.append(colors)
        
        if (i+1) % 20 == 0:
            print(f"  -> 已处理 {i+1} 帧.")
    
    # 合并所有帧的点云
    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
    else:
        print("[Error] 没有获取到任何有效点云，请检查数据！")
        return

    print(f"[Done] 累计有效点数 = {all_points.shape[0]}")
    
    # 创建 Open3D 点云对象并赋予颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    
    # 点云下采样（可选）
    voxel_size = 0.1
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 可视化
    print("[Visualization] 正在可视化点云...")
    o3d.visualization.draw_geometries([pcd_down], window_name="Reconstructed Scene from first 100 frames")
    
    # 如需要保存点云，可取消下面注释：
    # o3d.io.write_point_cloud("reconstructed_scene.ply", pcd_down)
    print("[Finish] 可视化结束.")

if __name__ == "__main__":
    main()
