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


int main(int argc, char const * argv[])
{
  auto cuda_available = torch::cuda::is_available();
  torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);

  c10::IntArrayRef test_tensor_shape = {2, 3, 2000};

  at::Tensor test_tensor = at::rand(test_tensor_shape, device);

  pointnet2_core::PointNetSetAbstraction sa1(1024, 0.05, 32,
    3, {32, 32, 64}, false);

  pointnet2_core::PointNetSetAbstraction sa2(256, 0.1, 32,
    64 + 3, {64, 64, 128}, false);

  pointnet2_core::PointNetSetAbstraction sa3(64, 0.2, 32,
    128 + 3, {128, 128, 256}, false);

  pointnet2_core::PointNetSetAbstraction sa4(16, 0.4, 32,
    256 + 3, {256, 256, 512}, false);


  auto l1_output = sa1.forward(&test_tensor, nullptr);
  auto l2_output = sa2.forward(&l1_output.first, &l1_output.second);
  auto l3_output = sa3.forward(&l2_output.first, &l2_output.second);
  //auto l4_output = sa4.forward(&l3_output.first, &l3_output.second);


  return EXIT_SUCCESS;
}
