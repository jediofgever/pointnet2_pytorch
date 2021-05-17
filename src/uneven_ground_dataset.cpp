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
: downsample_leaf_size_(downsample_leaf_size),
  num_point_per_batch_(num_point_per_batch),
  root_dir_(root_dir),
  use_normals_as_feature_(use_normals_as_feature)
{
  std::cout << "UnevenGroudDataset: given root directory is" << root_dir_ << std::endl;
  for (auto & entry : std::experimental::filesystem::directory_iterator(root_dir_)) {
    std::cout << "UnevenGroudDataset: processing given file " << entry.path() << std::endl;
    filenames_.push_back(entry.path());
  }

  for (int i = 0; i < filenames_.size(); i++) {

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    pcl::PointCloud<pcl::Normal> normals;

    if (!pcl::io::loadPCDFile(filenames_[i], *cloud)) {
      std::cout << "Gonna load a cloud with " << cloud->points.size() << " points" << std::endl;
    } else {
      std::cerr << "Could not read PCD file: " << filenames_[i] << std::endl;
    }

    *cloud = downsampleInputCloud(cloud, downsample_leaf_size_);

    if (use_normals_as_feature) {
      normals = estimateCloudNormals(cloud, 0.5);
    }

    at::Tensor xyz_feature_tensor = pclXYZFeature2Tensor(
      cloud,
      num_point_per_batch,
      device);
    at::Tensor normal_feature_tensor = pclNormalFeature2Tensor(
      normals,
      num_point_per_batch,
      device);
    at::Tensor label_tensor = extractPCLLabelsfromRGB(cloud, num_point_per_batch, device);

    auto xyz = torch::cat({xyz_feature_tensor, normal_feature_tensor}, 2);

    if (i == 0) {
      xyz_ = xyz;
      labels_ = label_tensor;
    } else {
      xyz_ = torch::cat({xyz_, xyz}, 0);
      labels_ = torch::cat({labels_, label_tensor}, 0);
    }
  }

  std::cout << "UnevenGroudDataset: shape of input data xyz_ " << xyz_.sizes() << std::endl;
  std::cout << "UnevenGroudDataset: shape of input labels labels_" << labels_.sizes() << std::endl;
}

UnevenGroudDataset::~UnevenGroudDataset()
{
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

pcl::PointCloud<pcl::Normal> UnevenGroudDataset::estimateCloudNormals(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud, double radius)
{
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
  normal_estimator.setSearchMethod(kd_tree);
  normal_estimator.setRadiusSearch(radius);
  normal_estimator.setInputCloud(inputCloud);
  normal_estimator.setSearchSurface(inputCloud);
  normal_estimator.compute(*normals);
  normal_estimator.setKSearch(20);
  return *normals;
}

at::Tensor UnevenGroudDataset::pclXYZFeature2Tensor(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int N, torch::Device device)
{
  int B = std::floor(cloud->points.size() / N);
  int C = 3;
  at::Tensor feature_tensor = torch::zeros({B, N, C}, device);
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < N; j++) {
      if (i * N + j < cloud->points.size()) {
        auto curr_point_feature = cloud->points[i * N + j];
        at::Tensor curr_point_feature_tensor = at::zeros({1, 3}, device);
        curr_point_feature_tensor.index_put_({0, 0}, curr_point_feature.x);
        curr_point_feature_tensor.index_put_({0, 1}, curr_point_feature.y);
        curr_point_feature_tensor.index_put_({0, 2}, curr_point_feature.z);
        feature_tensor.index_put_(
          {i, j, torch::indexing::Slice(
              torch::indexing::None,
              3)}, curr_point_feature_tensor);
      }
    }
  }
  return feature_tensor;
}

at::Tensor UnevenGroudDataset::pclNormalFeature2Tensor(
  pcl::PointCloud<pcl::Normal> normals, int N, torch::Device device)
{
  int B = std::floor(normals.points.size() / N);
  int C = 3;
  at::Tensor feature_tensor = torch::zeros({B, N, C}, device);
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < N; j++) {
      if (i * N + j < normals.points.size()) {
        auto curr_point_feature = normals.points[i * N + j];
        at::Tensor curr_point_feature_tensor = at::zeros({1, 3}, device);
        curr_point_feature_tensor.index_put_({0, 0}, curr_point_feature.normal_x);
        curr_point_feature_tensor.index_put_({0, 1}, curr_point_feature.normal_y);
        curr_point_feature_tensor.index_put_({0, 2}, curr_point_feature.normal_z);
        feature_tensor.index_put_(
          {i, j, torch::indexing::Slice(
              torch::indexing::None,
              3)}, curr_point_feature_tensor);
      }
    }
  }
  return feature_tensor;
}

at::Tensor UnevenGroudDataset::extractPCLLabelsfromRGB(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud, int N, torch::Device device)
{
  int B = std::floor(inputCloud->points.size() / N);
  // Two classes, traversable and NONtraversable
  at::Tensor labels = torch::zeros({B, N, 1}, device);
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < N; j++) {
      pcl::PointXYZRGB point = inputCloud->points[i * N + j];
      at::Tensor point_label_tensor = at::zeros({1, 1}, device);
      if (point.r /* red points ar NON traversable*/) {
        point_label_tensor.index_put_({0, 0}, 0);
      } else {
        point_label_tensor.index_put_({0, 0}, 1);
      }
      labels.index_put_({i, j}, point_label_tensor);
    }
  }
  return labels;
}


}  // namespace uneven_ground_dataset
