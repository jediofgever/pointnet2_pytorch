import open3d as o3d
import numpy as np
import os

"""
Given original Cloud and segmneted cloud
Denormialized cloud which is in [-1.0,1.0]
"""


def pc_denormalize(original_pc, normalized_pc):
    original_points = np.asarray(original_pc.points).astype(np.float32)
    normalized_points = np.asarray(normalized_pc.points).astype(np.float32)
    normalized_points_color = np.asarray(normalized_pc.colors).astype(np.float32)
    l = original_points.shape[0]
    centroid = np.mean(original_points, axis=0)
    m = np.max(np.sqrt(np.sum(original_points ** 2, axis=1)))
    normalized_points = normalized_points * m
    normalized_points = normalized_points + centroid
    normalized_pc.points = o3d.utility.Vector3dVector(normalized_points)
    normalized_pc.colors = o3d.utility.Vector3dVector(normalized_points_color)
    return normalized_pc


original_pc = o3d.io.read_point_cloud(os.path.join("../data/raw/train_2.pcd"))
normalized_pc = o3d.io.read_point_cloud(os.path.join("../data/segmented_cloud.pcd"))

de_normalized = pc_denormalize(original_pc, normalized_pc)
geometries = []
geometries.append(de_normalized)
o3d.visualization.draw_geometries(geometries, point_show_normal=False)

o3d.io.write_point_cloud(
    "../data/denormal_segmented_cloud.pcd",
    de_normalized,
    write_ascii=False,
    compressed=False,
    print_progress=False,
)
