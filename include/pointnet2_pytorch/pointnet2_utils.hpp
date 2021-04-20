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

#include <vector>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>

namespace pointnet2_utils
{

/**
 * @brief  Sample num_samples points from points
           according to farthest point sampling (FPS) algorithm.
 * @param input_tensor
 * @param num_samples
 * @return at::Tensor
 */
at::Tensor farthest_point_sample(at::Tensor * input_tensor, int num_samples, bool debug = false);

/**
 * @brief Extracts points specified in input_indices from input_tensor and resturns the as a newly consteructed tensor
 *
 * @param input_tensor
 * @param input_indices
 * @return at::Tensor
 */
at::Tensor extract_tensor_from_indices(at::Tensor * input_tensor, at::Tensor * input_indices);


at::Tensor extract_tensor_from_grouped_indices(
  at::Tensor * input_tensor,
  at::Tensor * input_indices);


/**
 * @brief checks and prints what device is available
 *
 */
void test_if_cuda_avail();

/**
 * @brief retruns a Tensor containing squared distance from each of point in source tensor to each point in target_point
 *
 * @param source_tensor
 * @param target_tensor
 * @return at::Tensor
 */
at::Tensor square_distance(at::Tensor * source_tensor, at::Tensor * target_tensor);

/**
 * @brief     Input:
              radius: local region radius
              nsample: max sample number in local region
              xyz: all points, [B, N, C]
              new_xyz: query points, [B, S, C]
              Return:
              group_idx: grouped points index, [B, S, nsample]
 *
 * @param radius
 * @param nsample
 * @param xyz
 * @param new_xyz
 * @return at::Tensor
 */
at::Tensor query_ball_point(double radius, int nsample, at::Tensor * xyz, at::Tensor * new_xyz);

/**
 * @brief     Input:
              npoint: Number of point for FPS
              radius: Radius of ball query
              nsample: Number of point for each ball query
              xyz: Old feature of points position data, [B, N, C]
              points: New feature of points data, [B, N, D]
              Return:
              new_xyz: sampled points position data, [B, npoint, C]
              new_points: sampled points data, [B, npoint, nsample, C+D]
 *
 * @param npoint
 * @param radius
 * @param nsample
 * @param xyz
 * @param points
 * @return std::pair<at::Tensor, at::Tensor>
 */
std::pair<at::Tensor, at::Tensor> sample_and_group(
  int npoint, double radius, int nsample,
  at::Tensor * xyz, at::Tensor * points);

/**
 * @brief
 *
 * @param input_tensor
 * @param cloud
 * @param point_color r,g,b
 */
void torch_tensor_to_pcl_cloud(
  const at::Tensor * input_tensor,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<double> point_color);

}  // namespace pointnet2_utils
