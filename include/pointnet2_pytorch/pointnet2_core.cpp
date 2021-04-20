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

namespace pointnet2_core
{

struct PointNetSetAbstraction : torch::nn::Module
{
  PointNetSetAbstraction(
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
      mlp_convs_->push_back(
        torch::nn::Conv2d((torch::nn::Conv2dOptions(last_channel_, mlp_.at(i), 1))));
      mlp_bns_->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(mlp_.at(i))));
      last_channel_ = mlp_.at(i);
    }
  }

  /**
   * @brief
         Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
   *
   * @param xyz
   * @param points
   * @return torch::Tensor
   */
  torch::Tensor forward(torch::Tensor xyz, at::Tensor points)
  {
    xyz = xyz.permute({0, 2, 1});
    if (!points.sizes().empty()) {
      points = points.permute({0, 2, 1});
    }

    // new_xyz, new_points
    std::pair<at::Tensor, at::Tensor> samlpled_and_grouped =
      pointnet2_utils::sample_and_group(npoint_, radius_, nsample_, &xyz, &points);

    // new_points shape :  [B, C+D, nsample,npoint]
    at::Tensor new_points = samlpled_and_grouped.second.permute({0, 3, 2, 1});

    for (int i = 0; i < mlp_convs_->size(); i++) {
    }


  }

  int64_t npoint_;
  float radius_;
  int64_t nsample_;
  int64_t last_channel_;
  c10::IntArrayRef mlp_;
  bool group_all_;

  torch::nn::ModuleList mlp_convs_;
  torch::nn::ModuleList mlp_bns_;

};


}  // namespace pointnet2_core
