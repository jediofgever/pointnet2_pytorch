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

  double downsample_voxel_size = 0.00;
  int batch_size = 1;
  int epochs = 50;
  int num_point_per_batch = 2048;
  double learning_rate = 0.001;
  bool use_normals_as_feature = false;

  torch::Device cuda_device = torch::kCUDA;
  std::string root_dir = "/home/ros2-foxy/pointnet2_pytorch/data";
  auto net = std::make_shared<pointnet2_sem_seg::PointNet2SemSeg>();
  net->to(cuda_device);

  torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(learning_rate));

  auto dataset = uneven_ground_dataset::UnevenGroudDataset(
    root_dir, cuda_device,
    num_point_per_batch, downsample_voxel_size, use_normals_as_feature).map(
    torch::data::transforms::Stack<>());

  auto dataset_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(dataset), batch_size);

  // Train the precious
  for (int i = 0; i < epochs; i++) {
    // In a for loop you can now use your data.
    float loss_numerical = 0.0;

    for (auto & batch : *dataset_loader) {

      auto xyz = batch.data.to(cuda_device);
      auto labels = batch.target.to(cuda_device);

      xyz = xyz.to(torch::kF32);
      labels = labels.to(torch::kLong);

      // Permute the channels so that we have  : [B,C,N]
      xyz = xyz.permute({0, 2, 1});

      auto net_output = net->forward(xyz);
      at::IntArrayRef output_shape = net_output.first.sizes();
      at::IntArrayRef labels_shape = labels.sizes();

      // Out: [B * N, num_classes]
      // label: [B * N]
      net_output.first = net_output.first.view(
        {output_shape[0] * output_shape[1],
          output_shape[2]});
          
      labels = labels.view(
        {labels_shape[0] *
          labels_shape[1] *
          labels_shape[2]});

      std::cout << net_output.first << std::endl;

      auto loss = torch::nll_loss(net_output.first, labels);

      optimizer.zero_grad();
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      loss_numerical += loss.item<float>();
    }

    std::cout << "===================================" << std::endl;
    std::cout << "========== Epoch %d =============== " << i << std::endl;
    std::cout << "Loss: " << loss_numerical << std::endl;
  }

  std::cout << "Pointnet2 sem segmentation training Successful." << std::endl;
  return EXIT_SUCCESS;
}
