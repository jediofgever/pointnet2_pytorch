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
  group_all_ = group_all;
  for (int i = 0; i < mlp.size(); i++) {
    mlp_convs_.push_back(
      torch::nn::Conv2d((torch::nn::Conv2dOptions(last_channel_, mlp.at(i), 1))));
    mlp_bns_.push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(mlp.at(i))));
    last_channel_ = mlp.at(i);
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
  std::pair<at::Tensor, at::Tensor> sampled_and_grouped =
    pointnet2_utils::sample_and_group(npoint_, radius_, nsample_, xyz, points);

  at::Tensor new_xyz = sampled_and_grouped.first;
  // new_points shape :  [B, C+D, nsample,npoint]
  at::Tensor new_points = sampled_and_grouped.second.permute({0, 3, 2, 1});

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


PointNetFeaturePropagation::PointNetFeaturePropagation(
  int64_t in_channel, c10::IntArrayRef mlp)
{
  int64_t last_channel = in_channel;
  for (int i = 0; i < mlp.size(); i++) {
    mlp_convs_.push_back(
      torch::nn::Conv1d((torch::nn::Conv1dOptions(last_channel, mlp.at(i), 1))));
    mlp_bns_.push_back(torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(mlp.at(i))));
    last_channel = mlp.at(i);
  }
}

at::Tensor PointNetFeaturePropagation::forward(
  at::Tensor * xyz1, at::Tensor * xyz2,
  at::Tensor * points1, at::Tensor * points2)
{
  std::cerr << __PRETTY_FUNCTION__ << "Not implemnted yet" << std::endl;
  return *xyz1;
}
}  // namespace pointnet2_core
