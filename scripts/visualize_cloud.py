import open3d as o3d
import numpy as np
import os

pcd = o3d.io.read_point_cloud(os.path.join("../data/decomposed_traversability_cloud.pcd"))

geometries = []
geometries.append(pcd)
o3d.visualization.draw_geometries(geometries, point_show_normal=False)
