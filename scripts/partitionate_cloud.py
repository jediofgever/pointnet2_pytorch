import open3d as o3d
import numpy as np
import os

root = "../data"
pcd = o3d.io.read_point_cloud(os.path.join(root, "train","train0.pcd"))

aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)

box_corners = np.asarray(aabb.get_box_points())

min_corner = box_corners[0]
max_corner = box_corners[4]

print(box_corners)

x_dist = abs(max_corner[0] - min_corner[0])
y_dist = abs(max_corner[1] - min_corner[1])

step_size = 20

geometries = []

# geometries.append(aabb)
train_regex = "train"
test_regex = "test"

train_index = 0
test_index = 0

overall = 0

for x in range(0, int(x_dist / step_size + 1)):

    for y in range(0, int(y_dist / step_size + 1)):

        current_min_corner = [
            min_corner[0] + step_size * x,
            min_corner[1] + step_size * y,
            min_corner[2],
        ]

        current_max_corner = [
            current_min_corner[0] + step_size,
            current_min_corner[1] + step_size,
            max_corner[2],
        ]

        this_box = o3d.geometry.AxisAlignedBoundingBox(
            current_min_corner, current_max_corner
        )

        cropped_pcd = pcd.crop(this_box)
        cropped_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.2, max_nn=20)
        )

        print(cropped_pcd)
        geometries.append(cropped_pcd)

        # geometries.append(this_box)
        if len(np.asarray(cropped_pcd.points)) < 2048:
            print(
                "PCL with to few points PCL has POINTS !",
                len(np.asarray(cropped_pcd.points)),
            )
        else:
            if overall % 5 == 0:
                print(
                    "Saving this partitioned cloud for testing",
                    (test_regex + str(test_index) + ".pcd"),
                )
                o3d.io.write_point_cloud(
                    os.path.join(
                        root,
                        "test",
                        (test_regex + str(test_index) + ".pcd"),
                    ),
                    cropped_pcd,
                    write_ascii=False,
                    compressed=False,
                    print_progress=False,
                )
                test_index += 1
            else:
                print(
                    "Saving this partitioned cloud for training",
                    (train_regex + str(train_index) + ".pcd"),
                )
                o3d.io.write_point_cloud(
                    os.path.join(
                        root,
                        "train",
                        (train_regex + str(train_index) + ".pcd"),
                    ),
                    cropped_pcd,
                    write_ascii=False,
                    compressed=False,
                    print_progress=False,
                )
                train_index += 1
        overall += 1

o3d.visualization.draw_geometries(geometries, point_show_normal=False)
