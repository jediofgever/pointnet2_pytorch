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
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>

namespace uneven_ground_dataset
{

UnevenGroudDataset::UnevenGroudDataset(Parameters params)
{
  std::cout << "UnevenGroudDataset: given root directory is " << params.root_dir << std::endl;
  std::cout << "UnevenGroudDataset: given split directory is " << params.split << std::endl;

  namespace fs = std::experimental::filesystem;
  fs::path fs_dir(params.root_dir);
  fs::path fs_split(params.split);
  fs::path full_path = fs_dir / fs_split;

  std::vector<std::string> filenames;
  for (auto & entry : fs::directory_iterator(full_path)) {
    filenames.push_back(entry.path());
  }

  std::cout << "UnevenGroudDataset: Processing files in total: " << filenames.size() << std::endl;

  for (int i = 0; i < filenames.size(); i++) {

    std::cout << "UnevenGroudDataset: Processing: " << filenames << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    // LOAD FILE
    if (!pcl::io::loadPCDFile(filenames[i], *cloud)) {
      std::cout << "UnevenGroudDataset: Loading a cloud with " << cloud->points.size() <<
        " points" << std::endl;
    } else {
      std::cerr << "UnevenGroudDataset: Could not read PCD file: " << filenames[i] << std::endl;
    }

    // DOWNSAMPLE IF REQUESTED
    if (params.downsample_leaf_size > 0.0) {
      cloud = downsampleInputCloud(cloud, params.downsample_leaf_size);
      std::cout << "UnevenGroudDataset: Downsampled cloud has points: " << cloud->points.size() <<
        std::endl;
    }

    // PART CLOUDS INTO SMALLER GRIDS
    auto parted_clouds = partitionateCloud(cloud, params.partition_step_size);


    std::cout << "UnevenGroudDataset: Divided cloud into parts:" << parted_clouds.size() <<
      std::endl;

    // NOW PROCESS ALL OF THESE CLOUD GRIDS
    for (auto && parted_cloud : parted_clouds) {

      // NORMALS MIGHT BE USED AS FEATURES
      pcl::PointCloud<pcl::Normal> normals;
      if (params.use_normals_as_feature) {
        normals = estimateCloudNormals(parted_cloud, params.normal_estimation_radius);
      }

      // WHEN TRAINING AND TESTING WE NORMALIZE CLOUD GRIDS TO [-1.0 , 1.0] RANGE
      auto normalized_parted_cloud = normalizeCloud(parted_cloud);

      // COMBINE ALL FEATURES TO XYZ_ TENSOR
      at::Tensor normalized_xyz_tensor = pclXYZFeature2Tensor(
        normalized_parted_cloud,
        params.num_point_per_batch,
        params.device);

      at::Tensor normals_tensor = pclNormalFeature2Tensor(
        normals,
        params.num_point_per_batch,
        params.device);

      // PUSH GROUND TRUTH LABELS TO LABELS_
      at::Tensor labels = extractPCLLabelsfromRGB(
        parted_cloud,
        params.num_point_per_batch,
        params.device);

      // NON NORMALIZED XYZ(positions) ARE STORED AS MEMBER VAR, IN CASE USER WANT TO ACCESS ORIGINAL DATA
      at::Tensor original_xyz_tensor = pclXYZFeature2Tensor(
        parted_cloud,
        params.num_point_per_batch,
        params.device);

      // DEFAULT FETAURES ARE JUST XYZ(positions)
      auto xyz = normalized_xyz_tensor;

      // ACCUMULATE ORIGINAL XYZ(POSTIIONS)
      auto original_xyz = original_xyz_tensor;

      //COMBINE NORMALS TO XYZ_ ONLY IF REQUESTED
      if (params.use_normals_as_feature) {
        xyz = torch::cat({xyz, normals_tensor}, 2);
      }

      // INIIALLY ASSIGN XYZ_, LABELS_ AND
      if (!xyz_.sizes().front()) {
        xyz_ = xyz;
        labels_ = labels;
        original_xyz_ = original_xyz;
      } else {
        xyz_ = torch::cat({xyz_, xyz}, 0);
        labels_ = torch::cat({labels_, labels}, 0);
        original_xyz_ = torch::cat({original_xyz_, original_xyz}, 0);
      }
    }
  }


  std::cout << "UnevenGroudDataset: shape of input data xyz_ " << xyz_.sizes() << std::endl;
  std::cout << "UnevenGroudDataset: shape of input labels labels_ " << labels_.sizes() << std::endl;
}

UnevenGroudDataset::~UnevenGroudDataset()
{
  std::cout << "UnevenGroudDataset : Destroying. " << xyz_.sizes() << std::endl;
}

torch::data::Example<at::Tensor, at::Tensor> UnevenGroudDataset::get(size_t index)
{
  return {xyz_[index], labels_[index]};
}

at::Tensor UnevenGroudDataset::get_non_normalized_data()
{
  return original_xyz_;
}

torch::optional<size_t> UnevenGroudDataset::size() const
{
  return xyz_.sizes().front();
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr UnevenGroudDataset::downsampleInputCloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, double downsmaple_leaf_size)
{
  pcl::VoxelGrid<pcl::PointXYZRGB> voxelGrid;
  voxelGrid.setInputCloud(inputCloud);
  voxelGrid.setLeafSize(downsmaple_leaf_size, downsmaple_leaf_size, downsmaple_leaf_size);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  voxelGrid.filter(*downsampledCloud);
  return downsampledCloud;
}

pcl::PointCloud<pcl::Normal> UnevenGroudDataset::estimateCloudNormals(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, double radius)
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
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud, int N, torch::Device device)
{
  int B = std::floor(cloud->points.size() / N);
  int C = 3;
  at::Tensor feature_tensor = torch::zeros({B, N, C}, device);

  #pragma omp parallel for
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
  const pcl::PointCloud<pcl::Normal> & normals, int N, torch::Device device)
{
  int B = std::floor(normals.points.size() / N);
  int C = 3;
  at::Tensor feature_tensor = torch::zeros({B, N, C}, device);

  #pragma omp parallel for
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
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, int N, torch::Device device)
{
  int B = std::floor(inputCloud->points.size() / N);
  // Two classes, traversable and NONtraversable
  at::Tensor labels = torch::zeros({B, N, 1}, device);

  #pragma omp parallel for
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

std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> UnevenGroudDataset::partitionateCloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, double step)
{
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> parted_clouds;
  pcl::PointXYZRGB min_pt, max_pt;
  pcl::getMinMax3D<pcl::PointXYZRGB>(*inputCloud, min_pt, max_pt);
  auto x_dist = std::abs(max_pt.x - min_pt.x);
  auto y_dist = std::abs(max_pt.y - min_pt.y);
  for (int x = 0; x < static_cast<int>(y_dist / step ); x++) {
    for (int y = 0; y < static_cast<int>(y_dist / step ); y++) {
      pcl::CropBox<pcl::PointXYZRGB> crop_box;
      crop_box.setInputCloud(inputCloud);
      Eigen::Vector4f current_min_corner(
        min_pt.x + step * x,
        min_pt.y + step * y,
        min_pt.z, 1);
      Eigen::Vector4f current_max_corner(
        current_min_corner.x() + step,
        current_min_corner.y() + step,
        max_pt.z, 1);
      crop_box.setMin(current_min_corner);
      crop_box.setMax(current_max_corner);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cropped_pcd(
        new pcl::PointCloud<pcl::PointXYZRGB>());
      crop_box.filter(*cropped_pcd);
      parted_clouds.push_back(cropped_pcd);
    }
  }
  return parted_clouds;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr UnevenGroudDataset::normalizeCloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud)
{
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid<pcl::PointXYZRGB>(*inputCloud, centroid);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr normalized_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>());

  for (auto && i : inputCloud->points) {
    pcl::PointXYZRGB p;
    p = i;
    p.x -= centroid.x();
    p.y -= centroid.y();
    p.z -= centroid.z();
    normalized_cloud->push_back(p);
  }

  Eigen::Vector4f max_point, pivot_point;
  pcl::getMaxDistance<pcl::PointXYZRGB>(*normalized_cloud, pivot_point, max_point);

  auto max_distaxis = std::max(std::abs(max_point.x()), std::abs(max_point.y()));
  float dist = std::sqrt(std::pow(max_distaxis, 2));

  for (auto && i : normalized_cloud->points) {
    i.x /= dist;
    i.y /= dist;
    i.z /= dist;
  }

  return normalized_cloud;
}

}  // namespace uneven_ground_dataset
