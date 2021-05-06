import open3d as o3d
import numpy as np
import os

pcd = o3d.io.read_point_cloud(os.path.join("../data/norm_train_2.pcd"))
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=20))

geometries = []
geometries.append(pcd)
o3d.visualization.draw_geometries(geometries, point_show_normal=True)
