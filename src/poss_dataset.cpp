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

#include <pointnet2_pytorch/poss_dataset.hpp>
#include <pcl/common/common.h>
#include <pcl/filters/crop_box.h>

namespace poss_dataset
{

POSSDataset::POSSDataset(Parameters params)
{
  std::cout << "POSSDataset: given root directory is " << params.root_dir << std::endl;
  std::cout << "POSSDataset: given split directory is " << params.split << std::endl;

  namespace fs = std::experimental::filesystem;
  fs::path root_dir_fs(params.root_dir);
  fs::path split_fs(params.split);
  fs::path root_dir_split_fs = root_dir_fs / split_fs;

  std::vector<std::string> sequences;
  for (auto & p : fs::recursive_directory_iterator(root_dir_split_fs)) {
    if (fs::is_directory(p.path())) {
      sequences.push_back(p.path().string());
    }
  }

  std::cout << "POSSDataset: Found, sequences " << sequences.size() << std::endl;

  for (auto && curr_sequence : sequences) {

    // Now there is data and label sub directories for each sequence
    const std::string labels_subdirectory("labels");
    const std::string train_subdirectory("velodyne");

    fs::path curr_seq_label_fs = fs::path(curr_sequence) / fs::path(
      labels_subdirectory);

    fs::path curr_seq_data_fs = fs::path(curr_sequence) / fs::path(
      train_subdirectory);

    // Now lets get all data and label files of thi sequence
    std::vector<std::string> curr_sequence_labels_filenames;
    for (auto & entry : fs::directory_iterator(curr_seq_label_fs)) {
      curr_sequence_labels_filenames.push_back(entry.path());
    }
    std::vector<std::string> curr_sequence_data_filenames;
    for (auto & entry : fs::directory_iterator(curr_seq_data_fs)) {
      curr_sequence_data_filenames.push_back(entry.path());
    }

    // At this point we have all filename for data and labels

    if (curr_sequence_data_filenames.size() != curr_sequence_labels_filenames.size()) {
      std::cerr <<
        "POSSDataset: The number of data files and labels does not match! something aint right, halting."
                <<
        std::endl;

      std::cerr <<
        "POSSDataset: curr_sequence_data_filenames.size() " <<
        curr_sequence_data_filenames.size() <<
        std::endl;

      std::cerr <<
        "POSSDataset: curr_sequence_labels_filenames.size() " <<
        curr_sequence_labels_filenames.size() <<
        std::endl;
    }

    std::cout << "POSSDataset: Processing files in total: " <<
      curr_sequence_data_filenames.size() << std::endl;

    for (int i = 1; i < curr_sequence_data_filenames.size() + 1; i++) {
      std::stringstream buffer;
      buffer << std::setfill('0') << std::setw(6) << i;

      fs::path cloud_filepath = curr_seq_data_fs / (buffer.str() + std::string(".bin"));
      fs::path label_filepath = curr_seq_label_fs / (buffer.str() + std::string(".label"));

      std::cout << "POSSDataset: Processing: " << cloud_filepath.string() << std::endl;
      std::cout << "POSSDataset: Processing: " << label_filepath.string() << std::endl;

      auto cloud = readBinFile(cloud_filepath.string());
      auto label_vector = readLabels(label_filepath.string());

      std::cout << "POSSDataset: Cloud has points: " << cloud->points.size() << std::endl;
      std::cout << "POSSDataset: Cloud has labels: " << label_vector.size() << std::endl;

      testLabels(cloud, label_vector);

      // DOWNSAMPLE IF REQUESTED
      if (params.downsample_leaf_size > 0.0) {
        cloud = downsampleInputCloud(cloud, params.downsample_leaf_size);
        std::cout << "POSSDataset: Downsampled cloud has points: " << cloud->points.size() <<
          std::endl;
      }
      // CROP CLOUD
      cloud = cropCloud<pcl::PointXYZRGB>(
        cloud,
        Eigen::Vector4f(-20.0f, -20.0f, -2.0f, 1.0f),
        Eigen::Vector4f(20.0f, 20.0f, 2.0f, 1.0f),
        false);

      // NORMALS MIGHT BE USED AS FEATURES
      pcl::PointCloud<pcl::Normal> normals;
      if (params.use_normals_as_feature) {
        normals = estimateCloudNormals(cloud, params.normal_estimation_radius);
      }

      // WHEN TRAINING AND TESTING WE NORMALIZE CLOUD GRIDS TO [-1.0 , 1.0] RANGE
      auto normalized_cloud = normalizeCloud(cloud);

      // COMBINE ALL FEATURES TO XYZ_ TENSOR
      at::Tensor normalized_xyz_tensor = pclXYZFeature2Tensor(
        normalized_cloud,
        params.num_point_per_batch,
        params.device);

      at::Tensor normals_tensor = pclNormalFeature2Tensor(
        normals,
        params.num_point_per_batch,
        params.device);

      // PUSH GROUND TRUTH LABELS TO LABELS_
      at::Tensor labels = extractLabelsfromVector(
        label_vector,
        params.num_point_per_batch,
        params.device);

      // NON NORMALIZED XYZ(positions) ARE STORED AS MEMBER VAR, IN CASE USER WANT TO ACCESS ORIGINAL DATA
      at::Tensor original_xyz_tensor = pclXYZFeature2Tensor(
        cloud,
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
  std::cout << "POSSDataset: shape of input data xyz_ " << xyz_.sizes() << std::endl;
  std::cout << "POSSDataset: shape of input labels labels_ " << labels_.sizes() << std::endl;
}

POSSDataset::~POSSDataset()
{
  std::cout << "POSSDataset : Destroying. " << xyz_.sizes() << std::endl;
}

torch::data::Example<at::Tensor, at::Tensor> POSSDataset::get(size_t index)
{
  return {xyz_[index], labels_[index]};
}

at::Tensor POSSDataset::get_non_normalized_data()
{
  return original_xyz_;
}

torch::optional<size_t> POSSDataset::size() const
{
  return xyz_.sizes().front();
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr POSSDataset::downsampleInputCloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, double downsmaple_leaf_size)
{
  pcl::VoxelGrid<pcl::PointXYZRGB> voxelGrid;
  voxelGrid.setInputCloud(inputCloud);
  voxelGrid.setLeafSize(downsmaple_leaf_size, downsmaple_leaf_size, downsmaple_leaf_size);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampledCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
  voxelGrid.filter(*downsampledCloud);
  return downsampledCloud;
}

pcl::PointCloud<pcl::Normal> POSSDataset::estimateCloudNormals(
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

at::Tensor POSSDataset::pclXYZFeature2Tensor(
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

at::Tensor POSSDataset::pclNormalFeature2Tensor(
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr POSSDataset::normalizeCloud(
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr POSSDataset::readBinFile(std::string filepath)
{
  std::fstream input(filepath.c_str(), std::ios::in | std::ios::binary);
  if (!input.good()) {
    std::cerr << "Could not read  POINT CLOUD file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }
  input.seekg(0, std::ios::beg);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);

  for (int i = 0; input.good() && !input.eof(); i++) {
    float intensity;
    pcl::PointXYZRGB point;
    input.read((char *)&point.x, 3 * sizeof(float));
    input.read((char *)&intensity, sizeof(float));
    cloud->push_back(point);
  }
  input.close();
  return cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr POSSDataset::readBinFileI(std::string filepath)
{
  std::fstream input(filepath.c_str(), std::ios::in | std::ios::binary);
  if (!input.good()) {
    std::cerr << "Could not read  POINT CLOUD file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }
  input.seekg(0, std::ios::beg);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
    new pcl::PointCloud<pcl::PointXYZI>);

  for (int i = 0; input.good() && !input.eof(); i++) {
    pcl::PointXYZI point;
    input.read((char *)&point.x, 3 * sizeof(float));
    input.read((char *)&point.intensity, sizeof(float));
    cloud->push_back(point);
  }
  input.close();

  return cloud;
}

std::vector<int> POSSDataset::readLabels(std::string filepath)
{
  std::fstream input(filepath.c_str(), std::ios::in | std::ios::binary);
  if (!input.good()) {
    std::cerr << "Could not read label file: " << filepath << std::endl;
    exit(EXIT_FAILURE);
  }
  input.seekg(0, std::ios::beg);
  std::vector<int> labels;
  for (int i = 0; input.good() && !input.eof(); i++) {
    u_int16_t label;
    input.read((char *)&label, 1 * sizeof(u_int32_t));
    label = (label & 0xffff); // Lower half of 32 bit int is class label
    //label = (label >> 16);     // higher half of 32 bit int is instance label
    labels.push_back(label);
  }
  input.close();

  return labels;
}

at::Tensor POSSDataset::extractLabelsfromVector(
  const std::vector<int> label_vector, int N, torch::Device device)
{
  int B = std::floor(label_vector.size() / N);
  at::Tensor labels = torch::zeros({B, N, 1}, device);
  #pragma omp parallel for
  for (int i = 0; i < B; i++) {
    for (int j = 0; j < N; j++) {
      int label = label_vector[i * N + j];
      at::Tensor point_label_tensor = at::zeros({1, 1}, device);
      point_label_tensor.index_put_({0, 0}, label);

      labels.index_put_({i, j}, point_label_tensor);
    }
  }
  return labels;
}

void POSSDataset::testLabels(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud,
  const std::vector<int> cloud_labels)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr test_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>());

  for (size_t i = 0; i < cloud_labels.size(); i++) {
    pcl::PointXYZRGB p;
    p.x = cloud->points[i].x;
    p.y = cloud->points[i].y;
    p.z = cloud->points[i].z;
    p.r = cloud_labels[i] * 2;
    p.g = cloud_labels[i] * 3;
    p.b = cloud_labels[i] * 4;
    test_cloud->points.push_back(p);
  }
  test_cloud->height = 1;
  test_cloud->width = cloud->points.size();
  pcl::PCDWriter wr;
  wr.writeASCII("/home/atas/tets.pcd", *test_cloud);
}

}  // namespace poss_dataset
