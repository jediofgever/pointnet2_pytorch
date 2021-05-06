import open3d as o3d
import numpy as np
import os

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    print(centroid)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

pcd = o3d.io.read_point_cloud(os.path.join("../data/train/train1.pcd"))

points = np.asarray(pcd.points).astype(np.float32)
points = pc_normalize(points)
pcd.points = o3d.utility.Vector3dVector(points)

geometries = []
geometries.append(pcd)
o3d.visualization.draw_geometries(geometries, point_show_normal=False)

o3d.io.write_point_cloud("../data/train/norm_train1.pcd", pcd, write_ascii=False, compressed=False, print_progress=False)