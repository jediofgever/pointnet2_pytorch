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

#include <pointnet2_pytorch/pointnet2_core.hpp>


int main(int argc, char const * argv[])
{
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

  pointnet2_core::PointNetSetAbstraction sa1(2048, 0.1, 32,
    3 + 3, {32, 32, 64}, false);
  pointnet2_core::PointNetSetAbstraction sa2(512, 0.2, 32,
    64 + 3, {64, 64, 128}, false);
  pointnet2_core::PointNetSetAbstraction sa3(128, 0.4, 32,
    128 + 3, {128, 128, 256}, false);
  pointnet2_core::PointNetSetAbstraction sa4(32, 0.8, 32,
    256 + 3, {256, 256, 512}, false);

  auto tensor_from_cloud = pointnet2_utils::load_pcl_as_torch_tensor(
    "/home/ros2-foxy/Downloads/test49.pcd", 1600, device);

  tensor_from_cloud = tensor_from_cloud.permute({0, 2, 1});

  auto sliced_tensor = tensor_from_cloud.index(
    {torch::indexing::Slice(torch::indexing::None, 4)});

  auto l0_points = sliced_tensor;
  auto l0_xyz = sliced_tensor.index(
    {torch::indexing::Slice(),
      torch::indexing::Slice(torch::indexing::None, 3),
      torch::indexing::Slice()});

  auto l1_output = sa1.forward(&l0_xyz, &l0_points);
  auto l2_output = sa2.forward(&l1_output.first, &l1_output.second);
  auto l3_output = sa3.forward(&l2_output.first, &l2_output.second);
  auto l4_output = sa4.forward(&l3_output.first, &l3_output.second);

  // Visualize results =============================================================
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr l0(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr l1(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr l2(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr l3(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr l4(new pcl::PointCloud<pcl::PointXYZRGB>);

  auto l0t = l0_xyz.permute({0, 2, 1});
  auto l1t = l1_output.first.permute({0, 2, 1});
  auto l2t = l2_output.first.permute({0, 2, 1});
  auto l3t = l3_output.first.permute({0, 2, 1});
  auto l4t = l4_output.first.permute({0, 2, 1});

  auto ls0t = l0_points.permute({0, 2, 1});
  auto ls1t = l1_output.second.permute({0, 2, 1});
  auto ls2t = l2_output.second.permute({0, 2, 1});
  auto ls3t = l3_output.second.permute({0, 2, 1});
  auto ls4t = l4_output.second.permute({0, 2, 1});

  std::cout << l0t.sizes() << std::endl;
  std::cout << l1t.sizes() << std::endl;
  std::cout << l2t.sizes() << std::endl;
  std::cout << l3t.sizes() << std::endl;
  std::cout << l4t.sizes() << std::endl;
  std::cout << ls0t.sizes() << std::endl;
  std::cout << ls1t.sizes() << std::endl;
  std::cout << ls2t.sizes() << std::endl;
  std::cout << ls3t.sizes() << std::endl;
  std::cout << ls4t.sizes() << std::endl;

  pointnet2_utils::torch_tensor_to_pcl_cloud(&l0t, l0, std::vector<double>({0.0, 255.0, 0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(&l1t, l1, std::vector<double>({255.0, 0, 255.0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(&l2t, l2, std::vector<double>({255.0, 255, 0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(&l3t, l3, std::vector<double>({100.0, 0, 0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(&l4t, l4, std::vector<double>({50.0, 100, 0}));

  auto merged_cloud = *l4 + *l3 + *l2 + *l1 + *l0;
  pcl::io::savePCDFile("../data/sa_pass.pcd", merged_cloud, false);

  pointnet2_core::PointNetFeaturePropagation fp4(768, {256, 256});
  pointnet2_core::PointNetFeaturePropagation fp3(384, {256, 256});
  pointnet2_core::PointNetFeaturePropagation fp2(320, {256, 128});
  pointnet2_core::PointNetFeaturePropagation fp1(128, {128, 128, 256});


  l3_output.second = fp4.forward(
    &l3_output.first, &l4_output.first, &l3_output.second, &l4_output.second);

  l2_output.second = fp3.forward(
    &l2_output.first, &l3_output.first, &l2_output.second, &l3_output.second);

  l1_output.second = fp2.forward(
    &l1_output.first, &l2_output.first, &l1_output.second, &l2_output.second);

  auto final = fp1.forward(
    &l0_xyz, &l1_output.first, nullptr, &l1_output.second);

  std::cout << "Pointnet2 core test Successful." << std::endl;

  return EXIT_SUCCESS;
}
