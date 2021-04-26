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


  // Set abstraction layers
  pointnet2_core::PointNetSetAbstraction sa1(2048, 0.1, 32,
    3 + 3, {32, 32, 64}, false);
  pointnet2_core::PointNetSetAbstraction sa2(512, 0.2, 32,
    64 + 3, {64, 64, 128}, false);
  pointnet2_core::PointNetSetAbstraction sa3(128, 0.4, 32,
    128 + 3, {128, 128, 256}, false);
  pointnet2_core::PointNetSetAbstraction sa4(32, 0.8, 32,
    256 + 3, {256, 256, 512}, false);

  // Pass a real point cloud to pass through SA and FP stacks of layers
  auto tensor_from_cloud = pointnet2_utils::load_pcl_as_torch_tensor(
    "/home/ros2-foxy/Downloads/test49.pcd", 1600, device);

  // Permute the channels so that we have  : [B,C,N]
  tensor_from_cloud = tensor_from_cloud.permute({0, 2, 1});

  namespace toi = torch::indexing;

  // Since the tensor might to large, might nt fit to memory, so slie the first 4 Batches
  at::Tensor sliced_tensor = tensor_from_cloud.index(
    {toi::Slice(toi::None, 4)});

  at::Tensor input_points = sliced_tensor;
  at::Tensor input_xyz = sliced_tensor.index(
    {toi::Slice(),
      toi::Slice(toi::None, 3),
      toi::Slice()});

  std::pair<at::Tensor, at::Tensor> sa1_output = sa1.forward(&input_xyz, &input_points);
  std::pair<at::Tensor, at::Tensor> sa2_output = sa2.forward(&sa1_output.first, &sa1_output.second);
  std::pair<at::Tensor, at::Tensor> sa3_output = sa3.forward(&sa2_output.first, &sa2_output.second);
  std::pair<at::Tensor, at::Tensor> sa4_output = sa4.forward(&sa3_output.first, &sa3_output.second);

  // Visualize results =============================================================
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sa1_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sa2_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sa3_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sa4_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  at::Tensor input_xyz_tensor = input_xyz.permute({0, 2, 1});
  at::Tensor sa1_xyz = sa1_output.first.permute({0, 2, 1});
  at::Tensor sa2_xyz = sa2_output.first.permute({0, 2, 1});
  at::Tensor sa3_xyz = sa3_output.first.permute({0, 2, 1});
  at::Tensor sa4_xyz = sa4_output.first.permute({0, 2, 1});

  std::cout << "input_xyz_tensor.sizes() " << input_xyz_tensor.sizes() << std::endl;
  std::cout << "sa1_xyz.sizes() " << sa1_xyz.sizes() << std::endl;
  std::cout << "sa2_xyz.sizes() " << sa2_xyz.sizes() << std::endl;
  std::cout << "sa3_xyz.sizes() " << sa3_xyz.sizes() << std::endl;
  std::cout << "sa4_xyz.sizes() " << sa4_xyz.sizes() << std::endl;

  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &input_xyz_tensor, original_cloud, std::vector<double>({255.0, 0.0, 0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &sa1_xyz, sa1_cloud, std::vector<double>({0.0, 255.0, 0.0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &sa2_xyz, sa2_cloud, std::vector<double>({0.0, 0, 255.0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &sa3_xyz, sa3_cloud, std::vector<double>({100.0, 100, 0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &sa4_xyz, sa4_cloud, std::vector<double>({0.0, 255, 255}));

  auto merged_cloud = *sa4_cloud + *sa3_cloud + *sa2_cloud + *sa1_cloud + *original_cloud;
  pcl::io::savePCDFile("../data/sa_pass.pcd", merged_cloud, false);

  std::cout << "Saved a cloud pass from set abstraction layers to ../data/sa_pass.pcd " <<
    std::endl;

  pointnet2_core::PointNetFeaturePropagation fp4(768, {256, 256});
  pointnet2_core::PointNetFeaturePropagation fp3(384, {256, 256});
  pointnet2_core::PointNetFeaturePropagation fp2(320, {256, 128});
  pointnet2_core::PointNetFeaturePropagation fp1(128, {128, 128, 256});

  sa3_output.second = fp4.forward(
    &sa3_output.first, &sa4_output.first, &sa3_output.second, &sa4_output.second);
  sa2_output.second = fp3.forward(
    &sa2_output.first, &sa3_output.first, &sa2_output.second, &sa3_output.second);
  sa1_output.second = fp2.forward(
    &sa1_output.first, &sa2_output.first, &sa1_output.second, &sa2_output.second);
  auto final = fp1.forward(
    &input_xyz, &sa1_output.first, nullptr, &sa1_output.second);

  std::cout << "Pointnet2 core test Successful." << std::endl;

  return EXIT_SUCCESS;
}
