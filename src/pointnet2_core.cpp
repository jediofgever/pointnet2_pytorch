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
  auto permuted_xyz = xyz->permute({0, 2, 1});
  at::Tensor permuted_points;

  if (points) {
    permuted_points = points->permute({0, 2, 1});
  }

  // new_xyz, new_points
  std::pair<at::Tensor, at::Tensor> sampled_and_grouped =
    pointnet2_utils::sample_and_group(npoint_, radius_, nsample_, &permuted_xyz, &permuted_points);

  // new_xyz: sampled points position data, [B, npoint, C]
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

  std::cout << "xyz1  " << xyz1->sizes() << std::endl;
  std::cout << "xyz2  " << xyz2->sizes() << std::endl;
  std::cout << "points2  .. " << points2->sizes() << std::endl;

  auto permuted_xyz1 = xyz1->permute({0, 2, 1});
  auto permuted_xyz2 = xyz2->permute({0, 2, 1});
  auto permuted_points2 = points2->permute({0, 2, 1});

  c10::IntArrayRef xyz1_shape = permuted_xyz1.sizes();
  c10::IntArrayRef xyz2_shape = permuted_xyz2.sizes();

  int B, N, C, S;
  B = xyz1_shape[0];
  N = xyz1_shape[1];
  C = xyz1_shape[2];
  S = xyz2_shape[1];

  at::Tensor interpolated_points;

  if (S == 1) {
    interpolated_points = permuted_points2.repeat({1, N, 1});
  } else {

    auto sqrdists = pointnet2_utils::square_distance(&permuted_xyz1, &permuted_xyz2);
    // this is a tuple , first element is distance itself, second idx
    auto dists_idx_tuple = sqrdists.sort(-1);

    auto dists = std::get<0>(dists_idx_tuple).index(
      {torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(torch::indexing::None, 3)});

    auto idx = std::get<1>(dists_idx_tuple).index(
      {torch::indexing::Slice(),
        torch::indexing::Slice(),
        torch::indexing::Slice(torch::indexing::None, 3)});

    // index smaller than  1e-10, set to  1e-10 to avoid division by zer0
    auto dist_recip = 1.0 / (dists + 1e-8);
    auto norm = torch::sum(dist_recip, 2, true);
    auto weight = dist_recip / norm;

    std::cout << "idx  " << idx.sizes() << std::endl;
    auto extracted_tensor = pointnet2_utils::extract_points_tensor_from_indices(
      &permuted_points2,
      &idx);
    std::cout << "extracted_tensor.. " << extracted_tensor.sizes() << std::endl;
    std::cout << "weight  " << weight.sizes() << std::endl;

    std::cout << "weight.view({B, N, 3, 1})  " << weight.view({B, N, C, 1}).sizes() << std::endl;
    auto ttbs = extracted_tensor * weight.view({B, N, C, 1});
    std::cout << "ttbs.. " << ttbs.sizes() << std::endl;

    interpolated_points =
      torch::sum(extracted_tensor, 2);

    std::cout << "interpolated_points" << interpolated_points.sizes() << std::endl;

  }
  at::Tensor new_points;

  if (points1 != nullptr) {
    std::cout << "points1  .. " << points1->sizes() << std::endl;
    auto permuted_points1 = points1->permute({0, 2, 1});
    new_points = torch::cat({permuted_points1, interpolated_points}, -1);
  } else {
    new_points = interpolated_points;
  }

  new_points = new_points.permute({0, 2, 1});

  for (size_t i = 0; i < mlp_convs_.size(); ++i) {
    auto crr_conv = mlp_convs_[i];
    auto crr_bn = mlp_bns_[i];
    crr_conv->to(new_points.device());
    crr_bn->to(new_points.device());
    new_points = torch::nn::functional::relu(crr_bn(crr_conv(new_points)));
  }

  return new_points;
}
}  // namespace pointnet2_core
