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
  pointnet2_core::PointNetSetAbstraction sa1(1024, 0.04, 32,
    3 + 3, {32, 32, 64}, false);
  pointnet2_core::PointNetSetAbstraction sa2(256, 0.08, 32,
    64 + 3, {64, 64, 128}, false);
  pointnet2_core::PointNetSetAbstraction sa3(64, 0.16, 32,
    128 + 3, {128, 128, 256}, false);
  pointnet2_core::PointNetSetAbstraction sa4(16, 0.32, 32,
    256 + 3, {256, 256, 512}, false);

  // Pass a real point cloud to pass through SA and FP stacks of layers
  auto tensor_from_cloud = pointnet2_utils::load_pcl_as_torch_tensor(
    "/home/ros2-foxy/pointnet2_pytorch/data/norm_train46.pcd", 2048, device);

  // Permute the channels so that we have  : [B,C,N]
  tensor_from_cloud = tensor_from_cloud.permute({0, 2, 1});

  namespace toi = torch::indexing;

  // Since the tensor might to large, might nt fit to memory, so slie the first 4 Batches
  at::Tensor sliced_tensor = tensor_from_cloud.index(
    {toi::Slice(toi::None, 1)});

  at::Tensor input_points = sliced_tensor;
  at::Tensor input_xyz = sliced_tensor.index(
    {toi::Slice(),
      toi::Slice(toi::None, 3),
      toi::Slice()});

  std::pair<at::Tensor, at::Tensor> sa1_output = sa1.forward(input_xyz, input_points);
  std::pair<at::Tensor, at::Tensor> sa2_output = sa2.forward(sa1_output.first, sa1_output.second);
  std::pair<at::Tensor, at::Tensor> sa3_output = sa3.forward(sa2_output.first, sa2_output.second);
  std::pair<at::Tensor, at::Tensor> sa4_output = sa4.forward(sa3_output.first, sa3_output.second);

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
  pointnet2_core::PointNetFeaturePropagation fp1(128, {128, 128, 128});

  sa3_output.second = fp4.forward(
    sa3_output.first, sa4_output.first, sa3_output.second, sa4_output.second);

  sa2_output.second = fp3.forward(
    sa2_output.first, sa3_output.first, sa2_output.second, sa3_output.second);

  sa1_output.second = fp2.forward(
    sa1_output.first, sa2_output.first, sa1_output.second, sa2_output.second);

  auto final_layer = fp1.forward(
    input_xyz, sa1_output.first, at::empty(0), sa1_output.second);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_layer_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fp4_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fp3_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fp2_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fp1_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &sa3_output.second, fp4_cloud, std::vector<double>({255.0, 0.0, 0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &sa2_output.second, fp3_cloud, std::vector<double>({0.0, 255.0, 0.0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &sa1_output.second, fp2_cloud, std::vector<double>({0.0, 0, 255.0}));
  pointnet2_utils::torch_tensor_to_pcl_cloud(
    &final_layer, final_layer_cloud, std::vector<double>({100.0, 100, 0}));

  auto fp_merged_cloud = *final_layer_cloud + *fp2_cloud + *fp3_cloud + *fp4_cloud;
  pcl::io::savePCDFile("../data/fp_pass.pcd", fp_merged_cloud, false);

  std::cout << "Saved a cloud pass from set abstraction layers to ../data/sa_pass.pcd " <<
    std::endl;

  auto conv1 = torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 128, 1));
  auto bn1 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(128));
  auto drop1 = torch::nn::Dropout(torch::nn::DropoutOptions(0.5));
  auto conv2 = torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 2, 1));

  conv1->to(tensor_from_cloud.device());
  bn1->to(tensor_from_cloud.device());
  drop1->to(tensor_from_cloud.device());
  conv2->to(tensor_from_cloud.device());

  auto x = torch::nn::functional::relu(bn1(conv1(final_layer)));
  x = conv2(x);
  x = torch::nn::functional::log_softmax(x, 1);
  x = x.permute({0, 2, 1});

  std::cout << "Test output size" << x.sizes() << std::endl;
  std::cout << "Pointnet2 core test Successful." << std::endl;

  return EXIT_SUCCESS;
}
