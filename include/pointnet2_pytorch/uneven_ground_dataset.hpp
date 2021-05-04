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

#include <torch/torch.h>
#include <experimental/filesystem>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/impl/search.hpp>

namespace uneven_ground_dataset
{
class UnevenGroudDataset : public torch::data::Dataset<UnevenGroudDataset>
{
public:
  UnevenGroudDataset(
    std::string root_dir /*root directory to dataset*/, at::Device device,
    int num_point_per_batch, double downsample_leaf_size, bool use_normals_as_feature);
  ~UnevenGroudDataset();

  torch::data::Example<at::Tensor, at::Tensor> get(size_t index) override;
  torch::optional<size_t> size() const override;
  std::pair<at::Tensor, at::Tensor> load_pcl_as_torch_tensor(
    const std::string cloud_filename, int N, torch::Device device);
  pcl::PointCloud<pcl::PointXYZRGB> downsampleInputCloud(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inputCloud, double downsample_leaf_size);

private:
  std::string root_dir_;
  std::vector<std::string> filename_vector_;
  at::Tensor xyz_;
  at::Tensor labels_;
  int num_point_per_batch_;
  double downsample_leaf_size_;
  bool use_normals_as_feature_;
};

}  // namespace uneven_ground_dataset
