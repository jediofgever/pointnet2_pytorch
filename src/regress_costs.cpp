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

#include <pointnet2_pytorch/cost_regression_utils.hpp>
#include <pcl/common/centroid.h>

int main()
{
  // PARAMETERS
  double CELL_RADIUS = 0.015;
  double MAX_ALLOWED_TILT = 25.0; // degrees
  double MAX_ALLOWED_POINT_DEVIATION = 0.004;
  double MAX_ALLOWED_ENERGY_GAP = 0.02;
  double NODE_ELEVATION_DISTANCE = 0.005;
  const double kMAX_COLOR_RANGE = 255.0;

  // LOAD SEGMENTED CLOUD FIRST
  std::string segmneted_pcl_filename = "../data/segmented_cloud.pcd";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_pcl(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  if (!pcl::io::loadPCDFile(segmneted_pcl_filename, *segmented_pcl)) {
    std::cout << "Gonna load a cloud with " << segmented_pcl->points.size() <<
      " points" <<
      std::endl;
  } else {
    std::cerr << "Could not read PCD file: " << segmneted_pcl_filename << std::endl;
  }

  // DENOISE THE LOUD IF HAVENT ALREADY
  //auto denoised_cloud = denoise_segmented_cloud(segmented_pcl, 0.025, 0.3, 10);
  //pcl::io::savePCDFile("../data/denoised_cloud.pcd", denoised_cloud, false);

  // WHEN CLOUS IS DENOISED JUST LOAD IT
  std::string denoised_pcl_filename = "../data/denoised_cloud.pcd";
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr denoised_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  if (!pcl::io::loadPCDFile(denoised_pcl_filename, *denoised_cloud)) {
    std::cout << "Gonna load a cloud with " << denoised_cloud->points.size() <<
      " points" << std::endl;
  } else {
    std::cerr << "Could not read PCD file: " << denoised_pcl_filename << std::endl;
  }

  // REMOVE NON TRAVERSABLE POINTS(RED POINTS)
  auto pure_traversable_pcl = cost_regression_utils::remove_non_traversable_points(denoised_cloud);

  // UNIFORMLY SAMPLE NODES ON TOP OF TRAVERSABLE CLOUD
  auto uniformly_sampled_nodes = cost_regression_utils::uniformly_sample_cloud(
    pure_traversable_pcl,
    CELL_RADIUS);

  // THIS IS BASICALLY VECTOR OF CLOUD SEGMENTS, EACH SEGMENT INCLUDES POINTS REPRESENTING CELL,
  // THE FIRST ELEMNET OF PAIR IS CENTROID WHILE SECOND IS THE POINTCLOUD ITSELF
  std::vector<std::pair<pcl::PointXYZRGB,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr>> decomposed_cells =
    cost_regression_utils::decompose_traversability_cloud(
    pure_traversable_pcl,
    uniformly_sampled_nodes, CELL_RADIUS);

  pcl::PointCloud<pcl::PointXYZRGB> cld;
  pcl::PointCloud<pcl::PointXYZRGB> elevated_nodes_cloud;
  for (auto && i : decomposed_cells) {

    auto plane_model = cost_regression_utils::fit_plane_to_cloud(i.second);
    auto rpy_from_plane_model = cost_regression_utils::absolute_rpy_from_plane(plane_model);
    auto pitch = rpy_from_plane_model[0] / MAX_ALLOWED_TILT * kMAX_COLOR_RANGE;
    auto roll = rpy_from_plane_model[1] / MAX_ALLOWED_TILT * kMAX_COLOR_RANGE;
    auto yaw = rpy_from_plane_model[2] / MAX_ALLOWED_TILT * kMAX_COLOR_RANGE;

    double average_point_deviation_from_plane =
      cost_regression_utils::average_point_deviation_from_plane(i.second, plane_model);

    double max_energy_gap_in_cloud =
      cost_regression_utils::max_energy_gap_in_cloud(i.second, 0.1, 1.0);

    double deviation_of_points_cost = average_point_deviation_from_plane /
      MAX_ALLOWED_POINT_DEVIATION *
      kMAX_COLOR_RANGE;
    double energy_gap_cost = max_energy_gap_in_cloud / MAX_ALLOWED_ENERGY_GAP *
      kMAX_COLOR_RANGE;

    double slope_cost = std::max(pitch, roll);

    double total_cost = 0.0 * slope_cost + 1.0 * deviation_of_points_cost + 0.0 * energy_gap_cost;

    auto plane_fitted_cell =
      cost_regression_utils::set_cloud_color(
      i.second, std::vector<double>(
    {
      total_cost,
      kMAX_COLOR_RANGE - total_cost,
      0}));

    pcl::PointXYZRGB elevated_node;
    elevated_node.x = i.first.x + NODE_ELEVATION_DISTANCE * plane_model.values[0];
    elevated_node.y = i.first.y + NODE_ELEVATION_DISTANCE * plane_model.values[1];
    elevated_node.z = i.first.z + NODE_ELEVATION_DISTANCE * plane_model.values[2];
    elevated_node.b = kMAX_COLOR_RANGE;
    elevated_nodes_cloud.points.push_back(elevated_node);

    cld += *plane_fitted_cell;
  }
  elevated_nodes_cloud.height = 1;
  elevated_nodes_cloud.width = elevated_nodes_cloud.points.size();
  cld += elevated_nodes_cloud;

  pcl::io::savePCDFile("../data/decomposed_traversability_cloud.pcd", cld, false);
  std::cout << "===================================" << std::endl;
  std::cout << "Testing finished!" << std::endl;
  return EXIT_SUCCESS;
}
