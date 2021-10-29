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
#include <pointnet2_pytorch/poss_dataset.hpp>
#include <torch/script.h> // One-stop header.

int main()
{
  // inform whether CUDA is there
  pointnet2_utils::check_avail_device();

  // CONSTS
  const double kDOWNSAMPLE_VOXEL_SIZE = 0.05;
  const double kNORMAL_ESTIMATION_RADIUS = 0.3;
  const int kBATCH_SIZE = 2;
  const int kEPOCHS = 100;
  int kN = 512;
  bool kUSE_NORMALS = true;
  const int kNUM_CLASSES = 14;
  const bool kRESUME_TRAINING = false;

  torch::Device cuda_device = torch::kCUDA;

  poss_dataset::POSSDataset::Parameters params;
  params.root_dir = "/home/fetulahatas1/poss_data";
  params.device = cuda_device;
  params.num_point_per_batch = kN;
  params.downsample_leaf_size = kDOWNSAMPLE_VOXEL_SIZE;
  params.use_normals_as_feature = kUSE_NORMALS;
  params.normal_estimation_radius = kNORMAL_ESTIMATION_RADIUS;
  params.split = "test";
  params.is_training = true;

  auto test_dataset = poss_dataset::POSSDataset(params)
    .map(torch::data::transforms::Stack<>());

  auto test_dataset_loader =
    torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    std::move(test_dataset), kBATCH_SIZE);

  std::cout << "Beginning Testing." << std::endl;

  // initialize net and optimizer
  pointnet2_sem_seg::PointNet2SemSeg net(kNUM_CLASSES);

  torch::serialize::InputArchive arc;
  arc.load_from("/home/fetulahatas1/pointnet2_pytorch/log/best_loss_model.pt", cuda_device);

  net.load(arc);
  net.to(cuda_device);

  // Test the model
  torch::NoGradGuard no_grad;
  net.eval();

  double loss_numerical = 0.0;
  int total_samples_in_batch = 0;
  int num_correct_points = 0;
  double overall_batch_accu = 0.0;

  for (const auto & batch : *test_dataset_loader) {
    auto original_input = batch.data.to(cuda_device);
    auto xyz = batch.data.to(cuda_device);
    auto labels = batch.target.to(cuda_device);
    labels = labels.to(torch::kLong);

    // Permute the channels so that we have  : [B,C,N]
    xyz = xyz.permute({0, 2, 1});

    auto net_output = net.forward(xyz);

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

    // Output the loss and checkpoint every 100 batches.
    loss_numerical += loss.item<float>();


    pcl::PointCloud<pcl::PointXYZRGB>::Ptr gt_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr predicted_cloud(
      new pcl::PointCloud<pcl::PointXYZRGB>());

    auto gt_labels = labels.view(
      {labels_shape[0], labels_shape[1] * labels_shape[2]}).to(torch::kLong);
    std::cout << "Processing a batch" << std::endl;
    pointnet2_utils::torch_tensor_to_pcl_cloud(&original_input, gt_cloud, &gt_labels);
    pointnet2_utils::torch_tensor_to_pcl_cloud(
      &original_input, predicted_cloud, &std::get<1>(predicted_label));

    pcl::PCDWriter wr;
    wr.writeASCII("/home/fetulahatas1/gt_cloud.pcd", *gt_cloud);
    wr.writeASCII("/home/fetulahatas1/predicted_cloud.pcd", *predicted_cloud);
  }

  overall_batch_accu = static_cast<double>(num_correct_points) /
    static_cast<double>(total_samples_in_batch * kN);

  std::cout << "===================================" << std::endl;
  std::cout << "Testing finished!" << std::endl;
  std::cout << "Loss: " << loss_numerical << std::endl;
  std::cout << "Overall Accuracy: " << overall_batch_accu << std::endl;

  return EXIT_SUCCESS;
}
