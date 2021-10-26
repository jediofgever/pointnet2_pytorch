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
PointNet2SemSeg::PointNet2SemSeg(int num_class)
: num_class_(num_class),
  sa1_(PointNetSetAbstraction(1024, 0.05, 128,
    6 + 3, {32, 32, 64}, false)),
  sa2_(PointNetSetAbstraction(256, 0.1, 64,
    64 + 3, {64, 64, 128}, false)),
  sa3_(PointNetSetAbstraction(64, 0.2, 32,
    128 + 3, {128, 128, 256}, false)),
  sa4_(PointNetSetAbstraction(16, 0.4, 32,
    256 + 3, {256, 256, 512}, false)),

  fp4_(PointNetFeaturePropagation(768, {256, 256})),
  fp3_(PointNetFeaturePropagation(384, {256, 256})),
  fp2_(PointNetFeaturePropagation(320, {256, 128})),
  fp1_(PointNetFeaturePropagation(128, {128, 128, 128})),

  conv1_(torch::nn::Conv1dOptions(128, 128, 1)),
  batch_norm1_(torch::nn::BatchNorm1d(128)),
  drop1_(torch::nn::DropoutOptions(0.5)),
  conv2_(torch::nn::Conv1dOptions(128, num_class, 1))
{
  register_module("conv1_", conv1_);
  register_module("batch_norm1_", batch_norm1_);
  register_module("drop1_", drop1_);
  register_module("conv2_", conv2_);
}

PointNet2SemSeg::~PointNet2SemSeg() {}

std::pair<at::Tensor, at::Tensor> PointNet2SemSeg::forward(at::Tensor xyz)
{
  at::Tensor input_points = xyz;

  at::Tensor input_xyz = xyz.index(
    {torch::indexing::Slice(),
      torch::indexing::Slice(torch::indexing::None, 3),
      torch::indexing::Slice()});

  std::pair<at::Tensor, at::Tensor> sa1_output =
    sa1_.forward(input_xyz, input_points);
  std::pair<at::Tensor, at::Tensor> sa2_output =
    sa2_.forward(sa1_output.first, sa1_output.second);
  std::pair<at::Tensor, at::Tensor> sa3_output =
    sa3_.forward(sa2_output.first, sa2_output.second);
  std::pair<at::Tensor, at::Tensor> sa4_output =
    sa4_.forward(sa3_output.first, sa3_output.second);

  sa3_output.second = fp4_.forward(
    sa3_output.first, sa4_output.first, sa3_output.second, sa4_output.second);
  sa2_output.second = fp3_.forward(
    sa2_output.first, sa3_output.first, sa2_output.second, sa3_output.second);
  sa1_output.second = fp2_.forward(
    sa1_output.first, sa2_output.first, sa1_output.second, sa2_output.second);
  auto final_layer = fp1_.forward(
    input_xyz, sa1_output.first, at::empty(0), sa1_output.second);

  conv1_->to(xyz.device());
  batch_norm1_->to(xyz.device());
  drop1_->to(xyz.device());
  conv2_->to(xyz.device());

  namespace F = torch::nn::functional;
  auto x = F::relu(batch_norm1_(conv1_(final_layer)));
  x = drop1_(x);
  x = conv2_(x);
  x = F::log_softmax(x, F::LogSoftmaxFuncOptions(1));
  x = x.permute({0, 2, 1});
  return std::make_pair(x, sa3_output.second);
}
}  // namespace pointnet2_sem_seg
