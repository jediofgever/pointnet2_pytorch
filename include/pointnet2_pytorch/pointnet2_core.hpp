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

class PointNetSetAbstraction : public torch::nn::Module
{
public:

/**
 * @brief Construct a new Point Net Set Abstraction object
 * 
 *        Input:
       		npoint: Number of point for FPS sampling
       		radius: Radius for ball query
       		nsample: Number of point for each ball query
       		in_channel: the dimension of channel
       		mlp: A list for mlp input-output channel, such as [64, 64, 128]
       		group_all: bool type for group_all or not
 *
 * @param npoint
 * @param radius
 * @param nsample
 * @param in_channel
 * @param mlp
 * @param group_all
 */
  PointNetSetAbstraction(
    int64_t npoint, float radius, int64_t nsample,
    int64_t in_channel, c10::IntArrayRef mlp, bool group_all);


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
  std::pair<at::Tensor, at::Tensor> forward(at::Tensor * xyz, at::Tensor * points);

protected:
  int64_t npoint_;
  float radius_;
  int64_t nsample_;
  int64_t last_channel_;
  c10::IntArrayRef mlp_;
  bool group_all_;
  std::vector<torch::nn::Conv2d> mlp_convs_;
  std::vector<torch::nn::BatchNorm2d> mlp_bns_;
};

}  // namespace pointnet2_core
