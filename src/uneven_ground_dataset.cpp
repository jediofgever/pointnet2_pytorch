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

UnevenGroudDataset::UnevenGroudDataset(std::string root_dir, at::Device device)
{
  root_dir_ = root_dir;
  for (const auto & entry : std::filesystem::directory_iterator(root_dir_)) {
    std::cout << entry.path() << std::endl;
    filename_vector_.push_back(entry.path());
  }

  int index = 0;

  for (auto && crr_file : filename_vector_) {
    std::pair<at::Tensor, at::Tensor> xyz_labels_pair = load_pcl_as_torch_tensor(
      crr_file, 2200, device);
    if (index == 0) {
      xyz_ = xyz_labels_pair.first;
      labels_ = xyz_labels_pair.first;
    } else {
      torch::cat({xyz_, xyz_labels_pair.first}, 0);
      torch::cat({labels_, xyz_labels_pair.second}, 0);
    }
  }
}

UnevenGroudDataset::~UnevenGroudDataset()
{
}

std::pair<at::Tensor, at::Tensor> UnevenGroudDataset::load_pcl_as_torch_tensor(
  const std::string cloud_filename, int N, torch::Device device)
{
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  if (!pcl::io::loadPCDFile(cloud_filename, cloud)) {
    std::cout << "Gonna load a cloud with " << cloud.points.size() << " points" << std::endl;
  } else {
    std::cerr << "Could not read PCD file: " << cloud_filename << std::endl;
  }
  // Convert cloud to a tensor with shape of [B,N,C]
  // Determine batches
  int B = cloud.points.size() % N;
  at::Tensor xyz = torch::zeros({B, N, 3}, device);
  at::Tensor labels = torch::zeros({B, N, 1}, device);

  for (int i = 0; i < B; i++) {
    for (int j = 0; j < N; j++) {
      if (i * N + j < cloud.points.size()) {
        pcl::PointXYZRGB crr_point = cloud.points[i * N + j];
        at::Tensor crr_xyz = at::zeros({1, 3}, device);
        crr_xyz.index_put_({0, 0}, crr_point.x);
        crr_xyz.index_put_({0, 1}, crr_point.y);
        crr_xyz.index_put_({0, 2}, crr_point.z);
        xyz.index_put_({i, j}, crr_xyz);
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
  return filename_vector_.size();
}
}  // namespace uneven_ground_dataset
