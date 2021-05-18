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
#include <pcl/octree/octree_search.h>
#include <pcl/filters/uniform_sampling.h>


namespace uneven_ground_dataset
{
class UnevenGroudDataset : public torch::data::Dataset<UnevenGroudDataset>
{
public:
  struct Parameters
  {
    std::string root_dir;
    at::Device device;
    int num_point_per_batch;
    double downsample_leaf_size;
    bool use_normals_as_feature;
    bool normal_estimation_radius;
    double partition_step_size;
    std::string split;
    bool is_training;

    // Assign meaningful default values to this parameters
    Parameters()
    : root_dir("/home"),
      device(torch::kCPU),
      num_point_per_batch(2048),
      downsample_leaf_size(0.0),
      use_normals_as_feature(true),
      normal_estimation_radius(0.5),
      partition_step_size(10.0),
      split("train"),
      is_training(true)
    {
    }
  };
  /**
   * @brief Construct a new Uneven Groud Dataset object
   *
   * @param params
   */
  UnevenGroudDataset(Parameters params);

  /**
   * @brief Destroy the Uneven Groud Dataset object
   *
   */
  ~UnevenGroudDataset();

  /**
   * @brief Given the batch index , return the data at the batch index
   *
   * @param index
   * @return torch::data::Example<at::Tensor, at::Tensor>
   */
  torch::data::Example<at::Tensor, at::Tensor> get(size_t index) override;

  /**
   * @brief return total number of batch samples
   *
   * @return torch::optional<size_t>
   */
  torch::optional<size_t> size() const override;

  /**
   * @brief
   *
   * @return at::Tensor
   */
  at::Tensor get_non_normalized_data();

  /**
   * @brief given a file path to a pcd file,
   * load it as tensor, with N number of points in each batch
   *
   * @param cloud_filename
   * @param N
   * @param device
   * @return std::pair<at::Tensor, at::Tensor>
   */

  /**
   * @brief Downsample a cloud with leaf size of downsample_leaf_size
   *
   * @param inputCloud
   * @param downsample_leaf_size
   * @return pcl::PointCloud<pcl::PointXYZRGB>
   */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampleInputCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, double downsample_leaf_size);

  /**
   * @brief estimate and retun normals of inputCloud
   *
   * @param inputCloud
   * @param radius
   * @return pcl::PointCloud<pcl::Normal>
   */
  pcl::PointCloud<pcl::Normal> estimateCloudNormals(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, double radius);

  /**
   * @brief Construct a new pcl X Y Z Feature2 Tensor object
   *
   * @param cloud
   * @param N
   * @param device
   */
  at::Tensor pclXYZFeature2Tensor(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloud, int N, torch::Device device);

  /**
  * @brief Construct a new pcl X Y Z Feature2 Tensor object
  *
  * @param cloud
  * @param N
  * @param device
  */
  at::Tensor pclNormalFeature2Tensor(
    const pcl::PointCloud<pcl::Normal> & normals, int N, torch::Device device);

  /**
   * @brief
   *
   * @param inputCloud
   * @param N
   * @param device
   * @return at::Tensor
   */
  at::Tensor extractPCLLabelsfromRGB(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, int N, torch::Device device);

  /**
   * @brief part the cloud with given step
   *
   * @param inputCloud
   * @param step
   */
  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> partitionateCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud, double step);

  /**
   * @brief
   *
   * @param inputCloud
   * @return pcl::PointCloud<pcl::PointXYZRGB>::Ptr
   */
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr  normalizeCloud(
    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr & inputCloud);

private:
  at::Tensor xyz_;
  at::Tensor labels_;
  // only positions
  at::Tensor original_xyz_;
};

}  // namespace uneven_ground_dataset
