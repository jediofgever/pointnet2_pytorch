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

namespace pointnet2_core
{
PointNetSetAbstraction::PointNetSetAbstraction(
  int64_t npoint, float radius, int64_t nsample,
  int64_t in_channel, c10::IntArrayRef mlp, bool group_all)
{
  npoint_ = npoint;
  radius_ = radius;
  nsample_ = nsample;
  last_channel_ = in_channel;
  mlp_ = mlp;
  group_all_ = group_all;
  for (int i = 0; i < mlp.size(); i++) {
    mlp_convs_.push_back(
      torch::nn::Conv2d((torch::nn::Conv2dOptions(last_channel_, mlp_.at(i), 1))));
    mlp_bns_.push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(mlp_.at(i))));
    last_channel_ = mlp_.at(i);
  }
}


std::pair<at::Tensor, at::Tensor> PointNetSetAbstraction::forward(
  at::Tensor * xyz,
  at::Tensor * points)
{
  *xyz = xyz->permute({0, 2, 1});
  if (points) {
    *points = points->permute({0, 2, 1});
  }

  // new_xyz, new_points
  std::pair<at::Tensor, at::Tensor> samlpled_and_grouped =
    pointnet2_utils::sample_and_group(npoint_, radius_, nsample_, xyz, points);

  at::Tensor new_xyz = samlpled_and_grouped.first;
  // new_points shape :  [B, C+D, nsample,npoint]
  at::Tensor new_points = samlpled_and_grouped.second.permute({0, 3, 2, 1});

  for (size_t i = 0; i < mlp_convs_.size(); ++i) {
    auto crr_conv = mlp_convs_[i];
    auto crr_bn = mlp_bns_[i];
    crr_conv->to(new_points.device());
    crr_bn->to(new_points.device());

    new_points = torch::nn::functional::relu(crr_bn(crr_conv(new_points)));
  }
  new_points = std::get<0>(torch::max(new_points, 2));
  new_xyz = new_xyz.permute({0, 2, 1});

  return std::make_pair(new_xyz, new_points);
}
}  // namespace pointnet2_core


int main(int argc, char const * argv[])
{
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

  c10::IntArrayRef test_tensor_shape = {2, 3, 2000};

  at::Tensor test_tensor = at::rand(test_tensor_shape, device);

  pointnet2_core::PointNetSetAbstraction sa1(1024, 0.1, 32,
    3, {32, 32, 64}, false);
  pointnet2_core::PointNetSetAbstraction sa2(256, 0.2, 32,
    64 + 3, {64, 64, 128}, false);

  auto l1_output = sa1.forward(&test_tensor, nullptr);
  auto l2_output = sa2.forward(&l1_output.first, &l1_output.second);

  return 0;
}
