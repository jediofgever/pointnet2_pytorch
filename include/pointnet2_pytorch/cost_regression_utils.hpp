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

#include <pointnet2_pytorch/pointnet2_utils.hpp>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace cost_regression_utils
{

pcl::PointCloud<pcl::PointXYZRGB>::Ptr denoise_segmented_cloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double radius,
  double tolerated_divergence_rate, int min_num_neighbours);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_traversable_points(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr get_non_traversable_points(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr uniformly_sample_cloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double radius);

std::vector<std::pair<pcl::PointXYZRGB,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> decompose_traversability_cloud(
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr pure_traversable_pcl,
  const pcl::PointCloud<pcl::PointXYZRGB>::Ptr uniformly_sampled_nodes,
  double radius);

pcl::ModelCoefficients fit_plane_to_cloud(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
  double dist_thes);

pcl::PointCloud<pcl::PointXYZRGB>::Ptr set_cloud_color(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
  std::vector<double> colors);

std::vector<double> absolute_rpy_from_plane(pcl::ModelCoefficients plane_model);

double average_point_deviation_from_plane(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
  pcl::ModelCoefficients plane_model);

double max_energy_gap_in_cloud(
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, double m, double v);


}  // namespace pointnet2_utils
