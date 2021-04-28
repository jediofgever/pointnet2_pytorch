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

  torch::Device cuda_device = torch::kCUDA;

  std::string root_dir = "/home/ros2-foxy/uneven_ground_dataset/limited";

  auto net = std::make_shared<pointnet2_sem_seg::PointNet2SemSeg>();
  torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(0.001));
  net->to(cuda_device);

  auto dataset = uneven_ground_dataset::UnevenGroudDataset(root_dir, cuda_device).map(
    torch::data::transforms::Stack<>());
  auto dataset_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(dataset), batch_size);

  // In a for loop you can now use your data.
  for (auto & batch : *dataset_loader) {
    auto xyz = batch.data.to(cuda_device).requires_grad_(true);
    auto labels = batch.target.to(cuda_device);
    xyz = xyz.to(torch::kF32);
    labels = labels.to(torch::kLong);

    // Permute the channels so that we have  : [B,C,N]
    xyz = xyz.permute({0, 2, 1});
    optimizer.zero_grad();
    auto net_output = net->forward(&xyz);

    at::IntArrayRef output_shape = net_output.first.sizes();
    at::IntArrayRef labels_shape = labels.sizes();

    // Out: [B * N, num_classes]
    // label: [B * N]
    net_output.first = net_output.first.reshape(
      {output_shape[0] * output_shape[1],
        output_shape[2]});
    labels = labels.reshape(
      {labels_shape[0] *
        labels_shape[1] *
        labels_shape[2]});

    torch::nn::NLLLoss criterion;
    auto loss = criterion->forward(net_output.first, labels);

    loss.backward();
    // Update the parameters based on the calculated gradients.
    optimizer.step();
    // Output the loss and checkpoint every 100 batches.
    std::cout << "Crr Loss at: " << loss.item<float>() << std::endl;
  }
  std::cout << "Pointnet2 sem segmentation training Successful." << std::endl;
  return EXIT_SUCCESS;
}
