// Copyright (c) 2020 Fetullah Atas, Norwegian University of Life Sciences
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pointnet2_pytorch/uneven_ground_dataset.hpp>

namespace uneven_ground_dataset
{

UnevenGroudDataset::UnevenGroudDataset(
  std::string root_dir, at::Device device,
  int num_point_per_batch, double downsample_leaf_size, bool use_normals_as_feature)
{
  downsample_leaf_size_ = downsample_leaf_size;
  num_point_per_batch_ = num_point_per_batch;
  root_dir_ = root_dir;
  use_normals_as_feature_ = use_normals_as_feature;

  std::cout << "given root directory:" << root_dir_ << std::endl;
  for (auto & entry : std::experimental::filesystem::directory_iterator(root_dir_)) {
    std::cout << "processing: " << entry.path() << std::endl;
    filename_vector_.push_back(entry.path());
  }

  int index = 0;

  for (auto && crr_file : filename_vector_) {
    std::pair<at::Tensor, at::Tensor> xyz_labels_pair = load_pcl_as_torch_tensor(
      crr_file, num_point_per_batch_, device);
    if (index == 0) {
      xyz_ = xyz_labels_pair.first;
      labels_ = xyz_labels_pair.second;
    } else {
      torch::cat({xyz_, xyz_labels_pair.first}, 0);
      torch::cat({labels_, xyz_labels_pair.second}, 0);
    }
    index++;
  }
  std::cout << "shape of input data xyz_ " << xyz_.sizes() << std::endl;
  std::cout << "shape of input labels labels_" << labels_.sizes() << std::endl;
}

UnevenGroudDataset::~UnevenGroudDataset()
{
}

std::pair<at::Tensor, at::Tensor> UnevenGroudDataset::load_pcl_as_torch_tensor(
  const std::string cloud_filename, int N, torch::Device device)
{

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  if (!pcl::io::loadPCDFile(cloud_filename, *cloud)) {
    std::cout << "Gonna load a cloud with " << cloud->points.size() << " points" << std::endl;
  } else {
    std::cerr << "Could not read PCD file: " << cloud_filename << std::endl;
    return std::make_pair(at::empty({1}, device), at::empty({1}, device));
  }

  pcl::PointCloud<pcl::Normal>::Ptr normals_out(new pcl::PointCloud<pcl::Normal>);
  if (use_normals_as_feature_) {
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> norm_est;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    norm_est.setSearchMethod(tree);
    norm_est.setRadiusSearch(0.1);
    norm_est.setInputCloud(cloud);
    norm_est.setSearchSurface(cloud);
    norm_est.compute(*normals_out);
  }

  if (downsample_leaf_size_ > 0.0) {
    std::cout << "Gonna downsample cloud with leaf size of  " << downsample_leaf_size_ << std::endl;
    *cloud = downsampleInputCloud(cloud, downsample_leaf_size_);
  }
  std::cout << "Cloud has " << cloud->points.size() << " points after downsample" << std::endl;

  // Convert cloud to a tensor with shape of [B,N,C]
  // Determine batches
  int B = std::floor(cloud->points.size() / N);
  int C = 3;

  if (normals_out->points.size()) {
    C += 3;
  }

  at::Tensor xyz = torch::zeros({B, N, C}, device);

  // Two classes, traversable and NONtraversable
  at::Tensor labels = torch::zeros({B, N, 1}, device);

  for (int i = 0; i < B; i++) {
    for (int j = 0; j < N; j++) {

      if (i * N + j < cloud->points.size()) {
        pcl::PointXYZRGB crr_point = cloud->points[i * N + j];
        at::Tensor crr_xyz = at::zeros({1, 3}, device);
        crr_xyz.index_put_({0, 0}, crr_point.x);
        crr_xyz.index_put_({0, 1}, crr_point.y);
        crr_xyz.index_put_({0, 2}, crr_point.z);
        xyz.index_put_({i, j, torch::indexing::Slice(torch::indexing::None, 3)}, crr_xyz);

        if (normals_out->points.size()) {
          pcl::Normal crr_point_normal = normals_out->points[i * N + j];
          at::Tensor crr_xyz_normal = at::zeros({1, 3}, device);
          crr_xyz_normal.index_put_({0, 0}, crr_point_normal.normal_x);
          crr_xyz_normal.index_put_({0, 1}, crr_point_normal.normal_y);
          crr_xyz_normal.index_put_({0, 2}, crr_point_normal.normal_z);
          xyz.index_put_({i, j, torch::indexing::Slice(3, torch::indexing::None)}, crr_xyz_normal);
        }

        at::Tensor crr_label = at::zeros({1, 1}, device);
        if (crr_point.r /* red points ar NON traversable*/) {
          crr_label.index_put_({0, 0}, 0);
        } else {
          crr_label.index_put_({0, 0}, 1);
        }
        labels.index_put_({i, j}, crr_label);
      }
    }

  }
  return std::make_pair(xyz, labels);
}

torch::data::Example<at::Tensor, at::Tensor> UnevenGroudDataset::get(size_t index)
{
  return {xyz_[index], labels_[index]};
}

torch::optional<size_t> UnevenGroudDataset::size() const
{
  return xyz_.sizes().front();
}

pcl::PointCloud<pcl::PointXYZRGB> UnevenGroudDataset::downsampleInputCloud(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud, double downsmaple_leaf_size)
{
  pcl::VoxelGrid<pcl::PointXYZRGB> voxelGrid;
  voxelGrid.setInputCloud(inputCloud);
  voxelGrid.setLeafSize(downsmaple_leaf_size, downsmaple_leaf_size, downsmaple_leaf_size);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  voxelGrid.filter(*downsampledCloud);
  return *downsampledCloud;
}
}  // namespace uneven_ground_dataset
