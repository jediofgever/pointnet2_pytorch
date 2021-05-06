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

  double downsample_voxel_size = 0.0;
  int batch_size = 4;
  int epochs = 50;
  int num_point_per_batch = 2048;
  double learning_rate = 0.01;
  bool use_normals_as_feature = true;
  const size_t learning_rate_decay_frequency = 10;  
  const double learning_rate_decay_factor = 1.0 / 5.0;  

  torch::Device cuda_device = torch::kCUDA;
  std::string root_dir = "/home/pc/pointnet2_pytorch/data";
  auto net = std::make_shared<pointnet2_sem_seg::PointNet2SemSeg>();
  net->to(cuda_device);
  torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(learning_rate));

  auto dataset = uneven_ground_dataset::UnevenGroudDataset(
    root_dir, cuda_device,
    num_point_per_batch, downsample_voxel_size, use_normals_as_feature).map(
    torch::data::transforms::Stack<>());
  auto dataset_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
    std::move(dataset), batch_size);

  auto current_learning_rate = learning_rate;

  // Train the precious
  for (int i = 0; i < epochs; i++) {
    // In a for loop you can now use your data.
    double loss_numerical = 0.0;
    double overall_batch_accu = 0.0;
    int num_correct_points = 0;
    int batch_counter = 0;
    int total_samples_in_batch = 0;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr segmented_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
   
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
      
 

      auto predicted_label = torch::max(net_output.first, 2);
 
      for (int k = 0; k < std::get<1>(predicted_label).sizes()[0]; k++) {
        for (int j = 0; j < std::get<1>(predicted_label).sizes()[1]; j++) {

          int predicted_value = std::get<1>(predicted_label).index({k, j}).item<int>();
          int gt_value = labels.index({k, j}).item<int>();

          pcl::PointXYZRGB point;

          point.x = xyz.index({k,0, j}).item<float>();
          point.y = xyz.index({k,1, j}).item<float>();
          point.z = xyz.index({k,2, j}).item<float>();

          if (predicted_value == gt_value) {
            num_correct_points += 1;
            point.g = 255;
          }
          else
          {
            point.r = 255;
          }

          segmented_cloud->points.push_back(point);
        }
      }
      total_samples_in_batch += output_shape[0];
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
      loss_numerical += loss.item<double>();
      std::cout << "Current Batch: " << batch_counter << std::endl;
      batch_counter++;
    }
    // Decay learning rate
    if ((i + 1) % learning_rate_decay_frequency == 0) {
        current_learning_rate *= learning_rate_decay_factor;
        static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
            .options()).lr(current_learning_rate);
    }
    overall_batch_accu = static_cast<double>(num_correct_points) / 
      static_cast<double>(total_samples_in_batch * num_point_per_batch) ;
    std::cout << "===================================" << std::endl;
    std::cout << "========== Epoch " << i << " ================" << std::endl;
    std::cout << "Loss: " << loss_numerical << std::endl;
    std::cout << "Overall Accuracy: " << overall_batch_accu << std::endl;

    segmented_cloud->width = segmented_cloud->points.size();
    segmented_cloud->height = 1;
    pcl::io::savePCDFile("../data/segmented_cloud.pcd", *segmented_cloud, false);
  }
  std::cout << "Pointnet2 sem segmentation training Successful." << std::endl;
  return EXIT_SUCCESS;
}
