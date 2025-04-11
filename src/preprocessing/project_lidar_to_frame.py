import numpy as np
import os
import matplotlib.pyplot as plt
import imageio

def load_calib_file(calib_file):
    """
    Reads the KITTI calibration file and returns a dictionary with matrices:
      - 'P2': Projection matrix for the left camera.
      - 'Tr_velo_to_cam': Transformation from LiDAR to camera coordinates.
      - 'R0_rect': Rectification matrix (set to identity if missing).
      
    This function looks for 'P2:' and 'Tr:' lines. Since 'R0_rect:' is missing in your file,
    it assigns an identity matrix for rectification.
    """
    calib = {}
    with open(calib_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.startswith('P2:'):
                calib['P2'] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            # Check for R0_rect only if available (in your case, it isn't)
            elif line.startswith('R0_rect:'):
                calib['R0_rect'] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 3)
            elif line.startswith('Tr:'):
                # Use 'Tr' as the LiDAR to camera transformation
                calib['Tr_velo_to_cam'] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
    
    if 'R0_rect' not in calib:
        # When the calibration file does not include R0_rect, assume identity
        print("Warning: 'R0_rect' not found in calibration file. Using identity matrix.")
        calib['R0_rect'] = np.eye(3)
    return calib

def read_velo_points(velo_file):
    """
    Reads the Velodyne LiDAR points from a binary file.
    Each point is represented by 4 floats: x, y, z, and reflectance.
    """
    if not os.path.isfile(velo_file):
        raise FileNotFoundError(f"{velo_file} does not exist or is not a file.")
    points = np.fromfile(velo_file, dtype=np.float32).reshape(-1, 4)
    return points

def project_lidar_to_camera(velo_points, calib):
    """
    Projects LiDAR points into the camera coordinate system and image plane.
    
    Steps:
      1. Converts the 3D LiDAR points (x, y, z) to homogeneous coordinates.
      2. Transforms from LiDAR to camera coordinates using Tr_velo_to_cam.
      3. Filters out points behind the camera (z <= 0).
      4. Applies camera rectification using R0_rect.
      5. Projects the 3D points into the image plane using P2.
    
    Returns:
      - pts_img: 2D pixel coordinates (u,v,1) for each valid point.
      - depth: The depth (z in the camera coordinate system) for each valid point.
    """
    # Use only the x, y, z coordinates
    pts = velo_points[:, :3]
    num_points = pts.shape[0]

    # Convert to homogeneous coordinates (N x 4)
    pts_hom = np.hstack((pts, np.ones((num_points, 1))))
    
    # Transform to camera coordinates
    pts_cam = np.dot(calib['Tr_velo_to_cam'], pts_hom.T).T

    # Keep points in front of the camera (z > 0)
    valid = pts_cam[:, 2] > 0
    pts_cam = pts_cam[valid]
    
    # Apply rectification (or identity if R0_rect was missing)
    pts_rect = np.dot(calib['R0_rect'], pts_cam.T).T
    
    # Convert to homogeneous for projection (add 1)
    pts_rect_hom = np.hstack((pts_rect, np.ones((pts_rect.shape[0], 1))))
    
    # Project into the image plane with P2
    pts_img = np.dot(calib['P2'], pts_rect_hom.T).T
    pts_img = pts_img / pts_img[:, 2:3]  # Normalize to get pixel coordinates

    depth = pts_rect[:, 2]  # Use Z coordinate as depth (in meters)
    return pts_img, depth

def create_depth_map(pts_img, depth, image_shape):
    """
    Creates a depth map from the 2D projected image points and their corresponding depth values.
    
    For each valid pixel, the depth value (in meters) is recorded.
    If multiple points project to the same pixel, the closest (minimum depth) is kept.
    
    Args:
      pts_img: Nx3 array of projected points (u, v, 1) in image coordinates.
      depth: 1D array of depth values for each point.
      image_shape: Tuple (height, width) that defines the output depth map dimensions.
    
    Returns:
      depth_map: A 2D numpy array (of floats in meters) containing the depth values.
    """
    height, width = image_shape
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # Convert projected points to integer pixel indices
    u = np.round(pts_img[:, 0]).astype(np.int32)
    v = np.round(pts_img[:, 1]).astype(np.int32)
    
    # Filter points outside image boundaries
    valid = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid]
    v = v[valid]
    depth = depth[valid]
    
    # Fill the depth map: choose the smallest (closest) depth if multiple points land in one pixel
    for i in range(len(u)):
        if depth_map[v[i], u[i]] == 0 or depth[i] < depth_map[v[i], u[i]]:
            depth_map[v[i], u[i]] = depth[i]
    
    return depth_map

def save_depth_as_kitti_png(depth_map, output_path):
    """
    Saves a depth map to a PNG file using KITTI conventions:
      - 16-bit PNG with depth values in millimeters.
      - 0 indicates no measurement.
    
    Args:
      depth_map: (H, W) numpy array of floats (depth values in meters).
      output_path: File path to save the PNG (e.g., '000000_depth.png').
    """
    # Convert depth from meters to millimeters, then round and cast to uint16.
    depth_mm = (depth_map * 256).round().astype(np.uint16)
    imageio.imwrite(output_path, depth_mm)

def main():
    # Set paths (adjust these to match your setup)
    calib_file = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\calib.txt'
    velo_folder = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\velodyne'
    output_folder = r'D:\Desktop\EECS568\Project\DenseFusionMap\data\slam\projected_depth'
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load calibration data
    calib = load_calib_file(calib_file)
    
    # Define image dimensions for the left camera (KITTI default: 375 x 1242)
    image_shape = (375, 1242)
    
    # Process all .bin files in the Velodyne folder
    velo_files = sorted([f for f in os.listdir(velo_folder) if f.endswith('.bin')])
    for file_name in velo_files:
        velo_file = os.path.join(velo_folder, file_name)
        lidar_points = read_velo_points(velo_file)
        
        pts_img, depth = project_lidar_to_camera(lidar_points, calib)
        depth_map = create_depth_map(pts_img, depth, image_shape)
        
        # Construct output filename: e.g., "000000_depth.png"
        output_path = os.path.join(output_folder, file_name.replace('.bin', '.png'))
        save_depth_as_kitti_png(depth_map, output_path)
        
        print(f"Saved depth map for {file_name} to {output_path}")
        
        # Uncomment the lines below if you want to quickly visualize one result:
        # plt.figure(figsize=(10, 4))
        # plt.imshow(depth_map, cmap='plasma')
        # plt.title(f"Depth Map: {file_name}")
        # plt.xlabel('Pixel x-coordinate')
        # plt.ylabel('Pixel y-coordinate')
        # plt.colorbar(label='Depth (m)')
        # plt.show()

if __name__ == '__main__':
    main()
