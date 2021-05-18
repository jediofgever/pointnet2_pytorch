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
#include <torch/script.h> // One-stop header.

int main()
{
  // inform whether CUDA is there
  pointnet2_utils::check_avail_device();

  // CONSTS
  const double kDOWNSAMPLE_VOXEL_SIZE = 0.0;
  const int kBATCH_SIZE = 8;
  const int kEPOCHS = 32;
  int kN = 2048;
  bool kUSE_NORMALS = true;

  // use dynamic LR
  double learning_rate = 0.01;
  const size_t learning_rate_decay_frequency = 8;
  const double learning_rate_decay_factor = 1.0 / 5.0;

  torch::Device cuda_device = torch::kCUDA;

  // Training datset
  std::string train_root_dir = "/home/atas/pointnet2_pytorch/data";

  uneven_ground_dataset::UnevenGroudDataset::Parameters params;
  params.root_dir = "/home/atas/pointnet2_pytorch/data";
  params.device = cuda_device;
  params.num_point_per_batch = kN;
  params.downsample_leaf_size = kDOWNSAMPLE_VOXEL_SIZE;
  params.use_normals_as_feature = kUSE_NORMALS;
  params.normal_estimation_radius = 0.6;
  params.partition_step_size = 25.0;
  params.split = "test";
  params.is_training = true;

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("/home/atas/pointnet2_pytorch/log/best_loss_model.pt");
  } catch (const c10::Error & e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  auto test_dataset = uneven_ground_dataset::UnevenGroudDataset(params)
    .map(torch::data::transforms::Stack<>());

  auto test_dataset_loader =
    torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    std::move(test_dataset), kBATCH_SIZE);


  std::cout << "Beginning Testing." << std::endl;

  // Test the model
  torch::NoGradGuard no_grad;
  module.eval();

  double loss_numerical = 0.0;
  int total_samples_in_batch = 0;
  int num_correct_points = 0;
  double overall_batch_accu = 0.0;

  for (const auto & batch : *test_dataset_loader) {

    auto xyz = batch.data.to(cuda_device);
    auto labels = batch.target.to(cuda_device);
    labels = labels.to(torch::kLong);

    // Permute the channels so that we have  : [B,C,N]
    xyz = xyz.permute({0, 2, 1});

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(xyz);

    auto net_output = module.forward(inputs).toTensorVector();

    at::IntArrayRef output_shape = net_output[0].sizes();
    at::IntArrayRef labels_shape = labels.sizes();

    auto predicted_label = torch::max(net_output[0], 2);

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
    net_output[0] = net_output[0].reshape(
      {output_shape[0] * output_shape[1],
        output_shape[2]});

    labels = labels.reshape(
      {labels_shape[0] *
        labels_shape[1] *
        labels_shape[2]});

    auto loss = torch::nll_loss(net_output[0], labels);

    // Output the loss and checkpoint every 100 batches.
    loss_numerical += loss.item<float>();
  }

  overall_batch_accu = static_cast<double>(num_correct_points) /
    static_cast<double>(total_samples_in_batch * kN);

  std::cout << "===================================" << std::endl;
  std::cout << "Testing finished!" << std::endl;
  std::cout << "Loss: " << loss_numerical << std::endl;
  std::cout << "Overall Accuracy: " << overall_batch_accu << std::endl;

  return EXIT_SUCCESS;
}
