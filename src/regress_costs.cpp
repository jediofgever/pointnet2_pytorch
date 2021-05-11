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

int main()
{
  // PARAMETERS
  double CELL_RADIUS = 0.015;
  double MAX_ALLOWED_TILT = 25.0;
  const double kMAX_COLOR_RANGE = 255.0;

  // LOAD SEGMENTED CLOUD FIRST AND FOREMOST
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

  std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> decomposed_cells =
    cost_regression_utils::decompose_traversability_cloud(
    pure_traversable_pcl, uniformly_sampled_nodes, CELL_RADIUS);

  pcl::PointCloud<pcl::PointXYZRGB> cld;
  for (auto && i : decomposed_cells) {

    auto plane_model = cost_regression_utils::fit_plane_to_cloud(i);
    auto rpy_from_plane_model = cost_regression_utils::absolute_rpy_from_plane(plane_model);

    auto pitch = rpy_from_plane_model[0] / MAX_ALLOWED_TILT * kMAX_COLOR_RANGE;
    auto roll = rpy_from_plane_model[1] / MAX_ALLOWED_TILT * kMAX_COLOR_RANGE;
    auto yaw = rpy_from_plane_model[2] / MAX_ALLOWED_TILT * kMAX_COLOR_RANGE;

    auto plane_fitted_cell =
      cost_regression_utils::set_cloud_color(
      i, std::vector<double>(
        {0 + std::max(pitch, roll),
          255 - std::max(pitch, roll),
          255 - std::max(pitch, roll)}));
    cld += *plane_fitted_cell;
  }
  pcl::io::savePCDFile("../data/decomposed_traversability_cloud.pcd", cld, false);
  std::cout << "===================================" << std::endl;
  std::cout << "Testing finished!" << std::endl;
  return EXIT_SUCCESS;
}
