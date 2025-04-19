import numpy as np
import open3d as o3d
import cv2
import os
import argparse

def load_calib(calib_file):
    fx = fy = cx = cy = None
    with open(calib_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('P_rect_02:') or line.startswith('P2:'):
                vals = line.split()
                if len(vals) < 13:
                    continue
                raw = vals[1:]
                fx = float(raw[0])
                cx = float(raw[2])
                fy = float(raw[5])
                cy = float(raw[6])
                break
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(f"无法从 {calib_file} 中正确解析相机内参，请检查 calib 文件格式.")
    print(f"[load_calib] fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
    return fx, fy, cx, cy

def create_point_cloud_from_depth(rgb_img, depth_img, fx, fy, cx, cy, depth_min=0.1, depth_max=1000.0):
    assert rgb_img.shape[:2] == depth_img.shape[:2], "[create_point_cloud_from_depth] 彩色图与深度图尺寸不一致！"
    H, W = depth_img.shape[:2]

    u_coord = np.arange(W)
    v_coord = np.arange(H)
    u_grid, v_grid = np.meshgrid(u_coord, v_coord)

    u_grid_flat = u_grid.flatten().astype(np.float32)
    v_grid_flat = v_grid.flatten().astype(np.float32)
    depth_flat = depth_img.flatten().astype(np.float32)

    valid_mask = (depth_flat > depth_min) & (depth_flat < depth_max)
    u_valid = u_grid_flat[valid_mask]
    v_valid = v_grid_flat[valid_mask]
    z_valid = depth_flat[valid_mask]

    X = (u_valid - cx) * z_valid / fx
    Y = (v_valid - cy) * z_valid / fy
    Z = z_valid
    points_cam = np.stack((X, Y, Z), axis=-1)

    rgb_flat = rgb_img.reshape(-1, 3)
    colors_valid = rgb_flat[valid_mask].astype(np.float32) / 255.0

    return points_cam, colors_valid

def transform_points(points, T):
    if points.shape[0] == 0:
        return points
    N = points.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    homo_points = np.hstack([points, ones])
    transformed = (T @ homo_points.T).T
    return transformed[:, :3]

def load_poses(poses_file):
    poses = []
    with open(poses_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            vals = list(map(float, line.split(',')))
            if len(vals) != 16:
                raise ValueError(f"第 {idx} 行不符合 4x4=16 个数字的格式: {line}")
            T = np.array(vals, dtype=np.float32).reshape(4, 4)
            poses.append(T)
    print(f"[load_poses] 加载了 {len(poses)} 个位姿.")
    return poses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point clouds from multiple frames")
    parser.add_argument('--root', type=str, default=r"D:\Desktop\EECS568\Project\DenseFusionMap\data\slam", help='Dataset root path')
    parser.add_argument('--num_frames', type=int, default=100, help='Number of frames to visualize')
    args = parser.parse_args()

    dataset_root = args.root
    num_frames = args.num_frames

    calib_file   = os.path.join(dataset_root, "calib.txt")
    poses_file   = os.path.join(dataset_root, "poses_estimated.csv")
    rgb_dir      = os.path.join(dataset_root, "rgb")
    depth_dir    = os.path.join(dataset_root, "depth")

    fx, fy, cx, cy = load_calib(calib_file)
    poses = load_poses(poses_file)

    all_points = []
    all_colors = []

    for i in range(min(num_frames, len(poses))):
        rgb_path = os.path.join(rgb_dir, f"{i:06d}.png")
        depth_path = os.path.join(depth_dir, f"{i:06d}.png")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            print(f"[WARN] 第 {i} 帧图像或深度图不存在，跳过。")
            continue

        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb_img is None:
            print(f"[WARN] 第 {i} 帧的彩色图读取失败。")
            continue
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            print(f"[WARN] 第 {i} 帧的深度图读取失败。")
            continue

        if depth_img.dtype == np.uint16:
            depth_img = depth_img.astype(np.float32) / 256.0

        points_cam, colors = create_point_cloud_from_depth(
            rgb_img, depth_img, fx, fy, cx, cy,
            depth_min=0.1, depth_max=100.0
        )

        if points_cam.shape[0] == 0:
            print(f"[Frame {i}] 有效点数为 0，跳过。")
            continue

        Tcw = poses[i]
        points_world = transform_points(points_cam, Tcw)

        all_points.append(points_world)
        all_colors.append(colors)

        if (i+1) % 20 == 0:
            print(f"  -> 已处理 {i+1} 帧.")

    if all_points:
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
    else:
        print("[Error] 没有获取到任何有效点云，请检查数据！")
        exit()

    print(f"[Done] 累计有效点数 = {all_points.shape[0]}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    print("[Visualization] 正在可视化点云...")
    o3d.visualization.draw_geometries([pcd], window_name="Multi-frame Point Cloud")
    print("[Finish] 可视化结束.")
