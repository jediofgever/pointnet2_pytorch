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

PointNetSetAbstraction::PointNetSetAbstraction(
  int64_t npoint, float radius, int64_t nsample,
  int64_t in_channel, c10::IntArrayRef mlp, bool group_all) :

conv1_(torch::nn::Conv2dOptions(in_channel, mlp[0], 1)),
batch_norm1_(torch::nn::BatchNorm2dOptions(mlp[0])),

conv2_(torch::nn::Conv2dOptions(mlp[0], mlp[1], 1)),
batch_norm2_(torch::nn::BatchNorm2dOptions(mlp[1])),

conv3_(torch::nn::Conv2dOptions(mlp[1], mlp[2], 1)),
batch_norm3_(torch::nn::BatchNorm2dOptions(mlp[2]))
{
  npoint_ = npoint;
  radius_ = radius;
  nsample_ = nsample;
  group_all_ = group_all;

  register_module("conv1_", conv1_);
  register_module("conv2_", conv2_);
  register_module("conv3_", conv3_);
  register_module("batch_norm1_", batch_norm1_);
  register_module("batch_norm2_", batch_norm2_);
  register_module("batch_norm3_", batch_norm3_);
}

std::pair<at::Tensor, at::Tensor> PointNetSetAbstraction::forward(
  at::Tensor xyz,
  at::Tensor points)
{
  xyz = xyz.permute({0, 2, 1});

  if (points.sizes().size()) {
    points = points.permute({0, 2, 1});
  }

  // new_xyz, new_points
  std::pair<at::Tensor, at::Tensor> sampled_and_grouped =
    pointnet2_utils::sample_and_group(npoint_, radius_, nsample_, xyz, points);

  // new_xyz: sampled points position data, [B, npoint, C], its from FPS algorithm
  at::Tensor new_xyz = sampled_and_grouped.first;
  // new_points shape :  [B, C+D, nsample,npoint]
  at::Tensor new_points = sampled_and_grouped.second.permute({0, 3, 2, 1});

  conv1_->to(new_points.device());
  batch_norm1_->to(new_points.device());
  new_points = torch::nn::functional::relu(batch_norm1_(conv1_(new_points)));

  conv2_->to(new_points.device());
  batch_norm2_->to(new_points.device());
  new_points = torch::nn::functional::relu(batch_norm2_(conv2_(new_points)));

  conv3_->to(new_points.device());
  batch_norm3_->to(new_points.device());
  new_points = torch::nn::functional::relu(batch_norm3_(conv3_(new_points)));

  new_points = std::get<0>(torch::max(new_points, 2));
  new_xyz = new_xyz.permute({0, 2, 1});

  return std::make_pair(new_xyz, new_points);
}


PointNetFeaturePropagation::PointNetFeaturePropagation(
  int64_t in_channel, c10::IntArrayRef mlp) :

conv1_(torch::nn::Conv1dOptions(in_channel, mlp[0], 1)),
batch_norm1_(torch::nn::BatchNorm1dOptions(mlp[0])),

conv2_(torch::nn::Conv1dOptions(mlp[0], mlp[1], 1)),
batch_norm2_(torch::nn::BatchNorm1dOptions(mlp[1])),

conv3_(mlp.size() == 3 ?
  torch::nn::Conv1dOptions(mlp[1], mlp[2], 1) : torch::nn::Conv1dOptions(1, 1, 1)),
batch_norm3_(mlp.size() == 3 ?
  torch::nn::BatchNorm1dOptions(mlp[2]) : torch::nn::BatchNorm1dOptions(1))
{
  num_mlp_channel_ = mlp.size();
  register_module("conv1_", conv1_);
  register_module("conv2_", conv2_);
  register_module("batch_norm1_", batch_norm1_);
  register_module("batch_norm2_", batch_norm2_);

  if (mlp.size() == 3) {
    register_module("conv3_", conv3_);
    register_module("batch_norm3_", batch_norm3_);
  }
}

at::Tensor PointNetFeaturePropagation::forward(
  at::Tensor xyz1, at::Tensor xyz2,
  at::Tensor points1, at::Tensor points2)
{
  xyz1 = xyz1.permute({0, 2, 1});
  xyz2 = xyz2.permute({0, 2, 1});
  points2 = points2.permute({0, 2, 1});
  c10::IntArrayRef xyz1_shape = xyz1.sizes();
  c10::IntArrayRef xyz2_shape = xyz2.sizes();
  int B, N, C, S;
  B = xyz1_shape[0];
  N = xyz1_shape[1];
  C = xyz1_shape[2];
  S = xyz2_shape[1];
  at::Tensor interpolated_points;
  if (S == 1) {
    interpolated_points = points2.repeat({1, N, 1});
  } else {
    auto sqrdists = pointnet2_utils::square_distance(xyz1, xyz2);

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
    auto extracted_tensor = pointnet2_utils::index_points(
      points2,
      idx);
    auto ttbs = extracted_tensor * weight.view({B, N, C, 1});
    interpolated_points =
      torch::sum(extracted_tensor, 2);
  }
  at::Tensor new_points;
  if (points1.sizes().size() > 1) {
    std::cout << "points1  .. " << points1.sizes() << std::endl;
    points1 = points1.permute({0, 2, 1});
    new_points = torch::cat({points1, interpolated_points}, -1);
  } else {
    new_points = interpolated_points;
  }
  new_points = new_points.permute({0, 2, 1});

  conv1_->to(new_points.device());
  batch_norm1_->to(new_points.device());
  new_points = torch::nn::functional::relu(batch_norm1_(conv1_(new_points)));

  conv2_->to(new_points.device());
  batch_norm2_->to(new_points.device());
  new_points = torch::nn::functional::relu(batch_norm2_(conv2_(new_points)));

  if (num_mlp_channel_ == 3) {
    conv3_->to(new_points.device());
    batch_norm3_->to(new_points.device());
    new_points = torch::nn::functional::relu(batch_norm3_(conv3_(new_points)));
  }

  return new_points;
}
