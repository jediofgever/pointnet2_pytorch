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
  PointNet2SemSeg();
  ~PointNet2SemSeg();

  std::pair<at::Tensor, at::Tensor> forward(at::Tensor xyz);

protected:
  std::shared_ptr<pointnet2_core::PointNetSetAbstraction> sa1_;
  std::shared_ptr<pointnet2_core::PointNetSetAbstraction> sa2_;
  std::shared_ptr<pointnet2_core::PointNetSetAbstraction> sa3_;
  std::shared_ptr<pointnet2_core::PointNetSetAbstraction> sa4_;

  std::shared_ptr<pointnet2_core::PointNetFeaturePropagation> fp1_;
  std::shared_ptr<pointnet2_core::PointNetFeaturePropagation> fp2_;
  std::shared_ptr<pointnet2_core::PointNetFeaturePropagation> fp3_;
  std::shared_ptr<pointnet2_core::PointNetFeaturePropagation> fp4_;
};


}  // namespace pointnet2_sem_seg
