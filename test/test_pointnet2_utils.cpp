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

int main()
{
  // check if cuda is there
  pointnet2_utils::check_avail_device();

  // Initialize some random
  int kB = 8; // Batch
  int kN = 4096; // Number of points in each batch
  int kFPS_SAMPLES = 128; // FPS going to sample kFPS_SAMPLES points
  int kC = 3; // number of channels
  int kMAX_N_POINTS_IN_RADIUS = 8;
  double kRADIUS = 0.04;

  c10::IntArrayRef test_tensor_shape = {kB, kN, kC};
  torch::Device cuda_device = torch::kCUDA;

  at::Tensor test_tensor = at::rand(test_tensor_shape, cuda_device);

  // Pass a real point cloud to pass through SA and FP stacks of layers
  /*auto test_tensor = pointnet2_utils::load_pcl_as_torch_tensor(
    "/home/atas/container_office_map.pcd", kN, cuda_device);*/
  std::cout << "test_tensor: \n" << test_tensor.data() << std::endl;


  // Since the tensor might to large, might nt fit to memory, so slie the first 4 Batches
  test_tensor = test_tensor.index(
    {at::indexing::Slice(at::indexing::None, kB)});

  // to test FPS(Furthest-point-sampleing algorithm) =============================
  at::Tensor fps_sampled_tensor_indices = pointnet2_utils::farthest_point_sample(
    test_tensor,
    kFPS_SAMPLES);

  at::Tensor fps_sampled_tensor = pointnet2_utils::index_points(
    test_tensor,
    fps_sampled_tensor_indices
  );

  //std::cout << "test_tensor: \n" << test_tensor << std::endl;
  //std::cout << "fps_sampled_tensor_indices: \n" << fps_sampled_tensor_indices << std::endl;
  //std::cout << "fps_sampled_tensor:  \n" << fps_sampled_tensor << std::endl;

  // Test Square distance function ================================================
  at::Tensor distance_tensor = pointnet2_utils::square_distance(fps_sampled_tensor, test_tensor);
  // std::cout << "distance_tensor:  \n" << distance_tensor << std::endl;

  // Test query_ball_point function ===============================================
  at::Tensor group_idx = pointnet2_utils::query_ball_point(
    kRADIUS, kMAX_N_POINTS_IN_RADIUS, test_tensor,
    fps_sampled_tensor);
  //std::cout << "test_tensor: \n" << test_tensor.sizes() << std::endl;
  //std::cout << "fps_sampled_tensor: \n" << fps_sampled_tensor.sizes() << std::endl;
  //std::cout << "group_idx:  \n" << group_idx << std::endl;
  //auto grouped_xyz = pointnet2_utils::extract_tensor_from_grouped_indices(&test_tensor, &group_idx);

  // Test Sample and Group ==========================================================
  auto new_xyz_and_points = pointnet2_utils::sample_and_group(
    kFPS_SAMPLES, kRADIUS, kMAX_N_POINTS_IN_RADIUS,
    test_tensor, test_tensor);

  // Visualize results =============================================================
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fps_sampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sampled_and_grouped_cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &test_tensor, full_cloud,
    std::vector<double>({255.0, 0, 0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &fps_sampled_tensor, fps_sampled_cloud,
    std::vector<double>({0.0, 255.0, 0}));

  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &new_xyz_and_points.first, sampled_and_grouped_cloud,
    std::vector<double>({0.0, 0, 255.0}));

  pcl::PointCloud<pcl::PointXYZRGB> merged_cloud = *fps_sampled_cloud;
  merged_cloud += *sampled_and_grouped_cloud;
  merged_cloud += *full_cloud;

  pcl::io::savePCDFile("../data/rand.pcd", merged_cloud, false);

  std::cout << "Pointnet2 utils test Successful." << std::endl;
  return EXIT_SUCCESS;
}
