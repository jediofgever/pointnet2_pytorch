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
  // inform whether CUDA is there
  pointnet2_utils::check_avail_device();

  // CONSTS
  double kPARTITION_STEP = 20.0;
  const double kDOWNSAMPLE_VOXEL_SIZE = 0.2;
  const double kNORMAL_ESTIMATION_RADIUS = 1.0;
  const int kBATCH_SIZE = 16;
  const int kEPOCHS = 16;
  int kN = 2048;
  bool kUSE_NORMALS = true;
  const int kNUM_CLASSES = 8;

  // use dynamic LR
  double learning_rate = 0.01;
  const size_t learning_rate_decay_frequency = 8;
  const double learning_rate_decay_factor = 1.0 / 5.0;

  torch::Device cuda_device = torch::kCUDA;

  uneven_ground_dataset::UnevenGroudDataset::Parameters params;
  params.root_dir = "/home/pc/pointnet2_pytorch/data";
  params.device = cuda_device;
  params.num_point_per_batch = kN;
  params.downsample_leaf_size = kDOWNSAMPLE_VOXEL_SIZE;
  params.use_normals_as_feature = kUSE_NORMALS;
  params.normal_estimation_radius = kNORMAL_ESTIMATION_RADIUS;
  params.partition_step_size = kPARTITION_STEP;
  params.split = "train";
  params.is_training = true;

  auto train_dataset = uneven_ground_dataset::UnevenGroudDataset(params).map(
    torch::data::transforms::Stack<>());

  auto train_dataset_loader =
    torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    std::move(train_dataset), kBATCH_SIZE);

  // initialize net and optimizer
  auto net = std::make_shared<pointnet2_sem_seg::PointNet2SemSeg>(kNUM_CLASSES);
  net->to(cuda_device);

  torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(
      learning_rate)
    .weight_decay(0.0001)
    .betas({0.9, 0.999}));

  auto current_learning_rate = learning_rate;

  double best_loss = INFINITY;


  // Train the precious
  for (int i = 0; i < kEPOCHS; i++) {
    // In a for loop you can now use your data.
    float loss_numerical = 0.0;
    double overall_batch_accu = 0.0;
    double num_correct_points = 0.0;
    int total_samples_in_batch = 0;
    int batch_counter = 0;

    for (auto & batch : *train_dataset_loader) {

      auto xyz = batch.data.to(cuda_device);
      auto labels = batch.target.to(cuda_device);
      labels = labels.to(torch::kLong);

      // Permute the channels so that we have  : [B,C,N]
      xyz = xyz.permute({0, 2, 1});

      auto net_output = net->forward(xyz);
      at::IntArrayRef output_shape = net_output.first.sizes();
      at::IntArrayRef labels_shape = labels.sizes();

      auto predicted_label = torch::max(net_output.first, 2);

      total_samples_in_batch += output_shape[0];
      auto correct_predictions = torch::eq(
        std::get<1>(predicted_label),
        labels.view(
          {labels_shape[0],
            labels_shape[1] *
            labels_shape[2]})).to(torch::kLong);

      num_correct_points += correct_predictions.count_nonzero().item<int>();

      // Out: [B * N, num_classes]
      // label: [B * N]
      net_output.first = net_output.first.reshape(
        {output_shape[0] * output_shape[1],
          output_shape[2]});

      labels = labels.reshape(
        {labels_shape[0] *
          labels_shape[1] *
          labels_shape[2]});

      auto loss = torch::nll_loss(net_output.first, labels);

      optimizer.zero_grad();
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      loss_numerical += loss.item<float>();
      batch_counter++;
      std::cout << "Current Batch: " << batch_counter << std::endl;
    }

    // Decay learning rate
    if ((i + 1) % learning_rate_decay_frequency == 0) {
      current_learning_rate *= learning_rate_decay_factor;
      static_cast<torch::optim::AdamOptions &>(optimizer.param_groups().front()
      .options()).lr(current_learning_rate);
    }

    overall_batch_accu = static_cast<double>(num_correct_points) /
      static_cast<double>(total_samples_in_batch * kN);

    std::cout << "===================================" << std::endl;
    std::cout << "========== Epoch: " << i << "===============" << std::endl;
    std::cout << "Loss: " << loss_numerical << std::endl;
    std::cout << "Overall Accuracy: " << overall_batch_accu << std::endl;

    if (loss_numerical < best_loss) {
      best_loss = loss_numerical;
      std::cout << "Found Best Loss at epoch: " << i << std::endl;
      std::cout << "Saving model and optimizer..." << std::endl;
      try {
        torch::save(net, "/home/pc/pointnet2_pytorch/log/best_loss_model.pt");
        torch::save(optimizer, "/home/pc/pointnet2_pytorch/log/best_optim_model.pt");
      } catch (const std::exception & e) {
        std::cout << "Failed to save model and optimizer..." << std::endl;
        std::cerr << e.what() << '\n';
      }
    }
  }
  std::cout << "Pointnet2 semantic segmentation training Successful." << std::endl;

  return EXIT_SUCCESS;
}
