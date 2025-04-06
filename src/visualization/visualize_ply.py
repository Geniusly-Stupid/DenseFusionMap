import open3d as o3d
import argparse
import sys

def load_and_visualize(ply_path):
    # 尝试以三角网格的方式加载 PLY 文件
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if not mesh.is_empty():
        print("成功加载为网格。")
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh])
    else:
        # 如果网格加载失败，尝试以点云方式加载 PLY 文件
        pcd = o3d.io.read_point_cloud(ply_path)
        if not pcd.is_empty():
            print("成功加载为点云。")
            o3d.visualization.draw_geometries([pcd])
        else:
            print("错误：无法加载该 PLY 文件。")
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 Open3D 可视化 PLY 文件")
    parser.add_argument("file", help="PLY 文件的路径")
    args = parser.parse_args()
    load_and_visualize(args.file)
