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

#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/pcd_grabber.h>

/**
 * @brief TODO
 *
 */
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
  std::pair<at::Tensor, at::Tensor> forward(at::Tensor xyz, at::Tensor points);

private:
  int64_t npoint_;
  float radius_;
  int64_t nsample_;
  int64_t last_channel_;
  bool group_all_;

  torch::nn::Conv2d conv1_, conv2_, conv3_;
  torch::nn::BatchNorm2d batch_norm1_, batch_norm2_, batch_norm3_;
};

/**
 * @brief TODO
 *
 */
class PointNetFeaturePropagation : public torch::nn::Module
{
public:
  /**
   * @brief Construct a new Point Net Feature Propagation object
   *
   * @param in_channel
   * @param mlp
   */
  PointNetFeaturePropagation(
    int64_t in_channel, c10::IntArrayRef mlp);

  /**
   * @brief
   *        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
   *
   * @param xyz1
   * @param xyz2
   * @param points1
   * @param points2
   * @return at::Tensor
   */
  at::Tensor forward(
    at::Tensor xyz1, at::Tensor xyz2,
    at::Tensor points1, at::Tensor points2);

private:
  int num_mlp_channel_;
  torch::nn::Conv1d conv1_, conv2_, conv3_;
  torch::nn::BatchNorm1d batch_norm1_, batch_norm2_, batch_norm3_;
};
