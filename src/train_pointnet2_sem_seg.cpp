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

  int batch_size = 4;
  torch::Device device = torch::kCUDA;
  std::string root_dir = "/home/ros2-foxy/uneven_ground_dataset/limited";

  auto net = std::make_shared<pointnet2_sem_seg::PointNet2SemSeg>();
  torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.001));
  net->to(device);

  auto dataset = uneven_ground_dataset::UnevenGroudDataset(root_dir, device).map(
    torch::data::transforms::Stack<>());
  auto dataset_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(dataset), 4);

  // In a for loop you can now use your data.
  for (auto & batch : *dataset_loader) {
    auto xyz = batch.data.to(device);
    auto labels = batch.target.to(device);
    xyz = xyz.to(torch::kF32);
    labels = labels.to(torch::kLong);

    // Permute the channels so that we have  : [B,C,N]
    xyz = xyz.permute({0, 2, 1});
    optimizer.zero_grad();
    auto net_output = net->forward(&xyz);

    net_output.first = net_output.first.reshape({4, 4400});
    labels = labels.reshape({4, 4400});

    std::cout << "net_output.first" << net_output.first.sizes() << std::endl;
    std::cout << "labels" << labels.sizes() << std::endl;

    auto loss = torch::nll_loss(torch::log_softmax(net_output.first, /*dim=*/ 1), labels);

    loss.backward();
    // Update the parameters based on the calculated gradients.
    optimizer.step();
    // Output the loss and checkpoint every 100 batches.
    std::cout << loss.item<float>() << std::endl;
  }
  std::cout << "Pointnet2 sem segmentation training Successful." << std::endl;
  return EXIT_SUCCESS;
}
