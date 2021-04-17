#include <vector>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <iostream>

/**
 * @brief  Sample num_samples points from points
           according to farthest point sampling (FPS) algorithm.
 * @param input_tensor
 * @param num_samples
 * @return at::Tensor
 */
at::Tensor furthest_point_sampling(at::Tensor * input_tensor, int num_samples, bool debug = false)
{
  torch::Device device = input_tensor->device();
  c10::IntArrayRef input_shape = input_tensor->sizes();
  c10::IntArrayRef output_shape = {input_shape.front(), num_samples};

  //  options for indice tensors
  auto options =
    torch::TensorOptions()
    .dtype(torch::kLong)
    .device(device)
    .requires_grad(true);

  at::Tensor centroids = at::zeros(output_shape, options);
  at::Tensor distance =
    at::ones(input_shape.slice(0, 2), torch::dtype(torch::kFloat32).device(device)).multiply(1e10);
  at::Tensor farthest =
    at::randint(0, input_shape.slice(0, 2).back(), input_shape.front(), options);
  at::Tensor batch_indices = at::arange(0, input_shape.front(), options);

  for (int i = 0; i < num_samples; i++) {

    centroids.index_put_({torch::indexing::Slice(), i}, farthest);
    at::Tensor centroid =
      input_tensor->index(
      {batch_indices, farthest, torch::indexing::Slice()})
      .view({input_shape.front(), 1, 3});

    at::Tensor dist = torch::sum(
      (input_tensor->subtract(centroid))
      .pow(2), -1);

    at::Tensor mask = dist < distance;
    distance.index_put_({mask}, dist.index({mask}));
    farthest = std::get<1>(torch::max(distance, -1));
  }
  if (debug) {
    std::cout << "fps: resulting centroids." << centroids << std::endl;
    std::cout << "fps: input_shape." << input_shape << std::endl;
    std::cout << "fps: input_tensor." << *input_tensor << std::endl;
    std::cout << "fps: device." << device << std::endl;
  }
  return centroids;
}

at::Tensor extract_tensor_from_indices(at::Tensor * input_indices, at::Tensor * input_tensor)
{
  // This indices are usally
  c10::IntArrayRef input_shape = input_indices->sizes();
  at::Tensor extracted_tensor = at::zeros(
    {input_shape[0],
      input_shape[1], 3},
    input_indices->device());

  for (int i = 0; i < input_shape[0]; i++) {
    for (int j = 0; j < input_shape[1]; j++) {
      pcl::PointXYZRGB crr_point;
      int index = input_indices->index({i, j}).item<int>();

      std::cout << " crr indc " << index;

      at::Tensor sampled_point = input_tensor->index({i, index});
      extracted_tensor.index_put_({i, j}, sampled_point);
    }
  }
  return extracted_tensor;
}

/**
 * @brief checks and prints what device is available
 *
 */
void test_if_cuda_avail()
{
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU possible." << std::endl;
    device = torch::kCUDA;
  } else {
    std::cout << "CUDA is NOT available! Training on CPU possible." << std::endl;
  }
}

at::Tensor square_distance(at::Tensor * source_tensor, at::Tensor * target_tensor)
{
  c10::IntArrayRef source_tensor_shape = source_tensor->sizes();
  c10::IntArrayRef target_tensor_shape = target_tensor->sizes();

  auto dist = -2 * torch::matmul(*source_tensor, target_tensor->permute({0, 2, 1}));

  dist += torch::sum(source_tensor->pow(2), -1).view(
    {source_tensor_shape[0], source_tensor_shape[1], 1});

  dist += torch::sum(target_tensor->pow(2), -1).view(
    {source_tensor_shape[0], 1, target_tensor_shape[1]});

  return dist;
}

/**
 * @brief
 *
 * @param input_tensor
 * @param cloud
 * @param point_color r,g,b
 */
void torch_tensor_to_pcl_cloud(
  const at::Tensor * input_tensor,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, std::vector<double> point_color)
{
  c10::IntArrayRef input_shape = input_tensor->sizes();
  for (int i = 0; i < input_shape[0]; i++) {
    for (int j = 0; j < input_shape[1]; j++) {
      pcl::PointXYZRGB crr_point;
      crr_point.x = input_tensor->index({i, j, 0}).item<float>();
      crr_point.y = input_tensor->index({i, j, 1}).item<float>();
      crr_point.z = input_tensor->index({i, j, 2}).item<float>();
      crr_point.r = point_color.at(0);
      crr_point.g = point_color.at(1);
      crr_point.b = point_color.at(2);
      cloud->points.push_back(crr_point);
    }
  }
  cloud->width = 1;
  cloud->height = cloud->points.size();
}

int main()
{
  // check if cuda is there
  test_if_cuda_avail();

  // to test FPS(Furthest-point-sampleing algorithm)
  c10::IntArrayRef test_tensor_shape = {4, 512, 3};
  torch::Device cuda_device = torch::kCUDA;
  at::Tensor test_tensor = at::rand(test_tensor_shape, cuda_device);

  at::Tensor fps_sampled_tensor_indices = furthest_point_sampling(&test_tensor, 32, false);
  at::Tensor fps_sampled_tensor = extract_tensor_from_indices(
    &fps_sampled_tensor_indices,
    &test_tensor);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fps_sampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  std::cout << "test_tensor: \n" << test_tensor << std::endl;
  std::cout << "fps_sampled_tensor_indices: \n" << fps_sampled_tensor_indices << std::endl;
  std::cout << "fps_sampled_tensor:  \n" << fps_sampled_tensor << std::endl;

  torch_tensor_to_pcl_cloud(&test_tensor, full_cloud, std::vector<double>({255.0, 0, 0}));
  torch_tensor_to_pcl_cloud(
    &fps_sampled_tensor, fps_sampled_cloud,
    std::vector<double>({0.0, 255.0, 0}));

  auto merged_cloud = *fps_sampled_cloud + *full_cloud;

  pcl::io::savePCDFile("../data/rand.pcd", merged_cloud, false);

  // Test Square distance function
  at::Tensor source_tensor = at::rand({4, 4, 3}, cuda_device);
  at::Tensor target_tensor = at::zeros({4, 1, 3}, cuda_device);
  at::Tensor distance_tensor = square_distance(&source_tensor, &target_tensor);

  std::cout << "source_tensor: \n" << source_tensor << std::endl;
  std::cout << "target_tensor: \n" << target_tensor << std::endl;
  std::cout << "distance_tensor:  \n" << distance_tensor << std::endl;

  return EXIT_SUCCESS;

}
