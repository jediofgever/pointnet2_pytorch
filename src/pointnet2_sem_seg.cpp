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

#include <pointnet2_pytorch/pointnet2_sem_seg.hpp>

namespace pointnet2_sem_seg
{
PointNet2SemSeg::PointNet2SemSeg()
{
  sa1_ =
    std::make_shared<pointnet2_core::PointNetSetAbstraction>(
    pointnet2_core::PointNetSetAbstraction(
      1024, 0.1, 32, 3 + 3, {32, 32, 64}, false));
  sa2_ =
    std::make_shared<pointnet2_core::PointNetSetAbstraction>(
    pointnet2_core::PointNetSetAbstraction(
      512, 0.2, 32,
      64 + 3, {64, 64, 128}, false));
  sa3_ =
    std::make_shared<pointnet2_core::PointNetSetAbstraction>(
    pointnet2_core::PointNetSetAbstraction(
      128, 0.4, 32,
      128 + 3, {128, 128, 256}, false));
  sa4_ =
    std::make_shared<pointnet2_core::PointNetSetAbstraction>(
    pointnet2_core::PointNetSetAbstraction(
      64, 0.8, 32,
      256 + 3, {256, 256, 512}, false));

  fp4_ =
    std::make_shared<pointnet2_core::PointNetFeaturePropagation>(
    pointnet2_core::PointNetFeaturePropagation(768, {256, 256}));
  fp3_ =
    std::make_shared<pointnet2_core::PointNetFeaturePropagation>(
    pointnet2_core::PointNetFeaturePropagation(384, {256, 256}));
  fp2_ =
    std::make_shared<pointnet2_core::PointNetFeaturePropagation>(
    pointnet2_core::PointNetFeaturePropagation(320, {256, 128}));
  fp1_ =
    std::make_shared<pointnet2_core::PointNetFeaturePropagation>(
    pointnet2_core::PointNetFeaturePropagation(128, {128, 128, 128}));
}

PointNet2SemSeg::~PointNet2SemSeg() {}

std::pair<at::Tensor, at::Tensor> PointNet2SemSeg::forward(at::Tensor * xyz)
{
  at::Tensor input_points = *xyz;

  at::Tensor input_xyz = xyz->index(
    {torch::indexing::Slice(),
      torch::indexing::Slice(torch::indexing::None, 3),
      torch::indexing::Slice()});

  std::pair<at::Tensor,
    at::Tensor> sa1_output = sa1_->forward(&input_xyz, &input_points);
  std::pair<at::Tensor,
    at::Tensor> sa2_output = sa2_->forward(&sa1_output.first, &sa1_output.second);
  std::pair<at::Tensor,
    at::Tensor> sa3_output = sa3_->forward(&sa2_output.first, &sa2_output.second);
  std::pair<at::Tensor,
    at::Tensor> sa4_output = sa4_->forward(&sa3_output.first, &sa3_output.second);

  sa3_output.second = fp4_->forward(
    &sa3_output.first, &sa4_output.first, &sa3_output.second, &sa4_output.second);

  sa2_output.second = fp3_->forward(
    &sa2_output.first, &sa3_output.first, &sa2_output.second, &sa3_output.second);

  sa1_output.second = fp2_->forward(
    &sa1_output.first, &sa2_output.first, &sa1_output.second, &sa2_output.second);

  auto final_layer = fp1_->forward(
    &input_xyz, &sa1_output.first, nullptr, &sa1_output.second);

  auto conv1 = torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 128, 1));
  auto bn1 = torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(128));
  auto drop1 = torch::nn::Dropout(torch::nn::DropoutOptions(0.5));
  auto conv2 = torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 2, 1));

  conv1->to(xyz->device());
  bn1->to(xyz->device());
  drop1->to(xyz->device());
  conv2->to(xyz->device());

  auto x = torch::nn::functional::relu(bn1(conv1(final_layer)));
  x = conv2(x);
  x = torch::nn::functional::log_softmax(x, 1);
  x = x.permute({0, 2, 1});

  return std::make_pair(x, sa4_output.second);
}


}  // namespace pointnet2_sem_seg
