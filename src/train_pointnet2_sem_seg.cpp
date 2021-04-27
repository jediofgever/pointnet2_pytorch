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
#include <pointnet2_pytorch/uneven_ground_dataset.hpp>


int main()
{
  std::string root_dir = "/home/ros2-foxy/uneven_ground_dataset";
  torch::Device device = torch::kCPU;

  auto dataset = uneven_ground_dataset::UnevenGroudDataset(root_dir, device).map(
    torch::data::transforms::Stack<>());

  int batch_size = 4;
  auto dataset_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(dataset), 4);

  // In a for loop you can now use your data.
  for (auto & batch : *dataset_loader) {
    auto data = batch.data;
    auto label = batch.target;

    std::cout << " data " << data.sizes() << std::endl;
    std::cout << " label " << label.sizes() << std::endl;
  }

  std::cout << "Pointnet2 sem segmentation training Successful." << std::endl;
  return EXIT_SUCCESS;
}
