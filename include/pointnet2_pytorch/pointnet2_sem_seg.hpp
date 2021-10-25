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

namespace pointnet2_sem_seg
{

/**
 * @brief TODO
 *
 */
class PointNet2SemSeg : public torch::nn::Module
{
public:
  PointNet2SemSeg(int num_class);
  ~PointNet2SemSeg();

  std::pair<at::Tensor, at::Tensor> forward(at::Tensor xyz);

protected:
  PointNetSetAbstraction sa1_;
  PointNetSetAbstraction sa2_;
  PointNetSetAbstraction sa3_;
  PointNetSetAbstraction sa4_;

  PointNetFeaturePropagation fp1_;
  PointNetFeaturePropagation fp2_;
  PointNetFeaturePropagation fp3_;
  PointNetFeaturePropagation fp4_;

  torch::nn::Conv1d conv1_;
  torch::nn::BatchNorm1d batch_norm1_;
  torch::nn::Dropout drop1_;
  torch::nn::Conv1d conv2_;

  int num_class_;
};


}  // namespace pointnet2_sem_seg
