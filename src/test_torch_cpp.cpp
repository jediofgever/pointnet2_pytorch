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
      at::Tensor sampled_point = input_tensor->index({i, index});
      extracted_tensor.index_put_({i}, sampled_point);
    }
  }
  return extracted_tensor;
}

/**
 * @brief checks whether cuda is avail for use
 *
 */
void test_if_cuda_avail()
{
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available! Training on GPU possible." << std::endl;
    device = torch::kCUDA;
  }
}

void torch_tensor_to_pcl_cloud(
  const at::Tensor * input_tensor,
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
  c10::IntArrayRef input_shape = input_tensor->sizes();
  for (int i = 0; i < input_shape[0]; i++) {
    for (int j = 0; j < input_shape[1]; j++) {
      pcl::PointXYZRGB crr_point;
      crr_point.x = input_tensor->index({i, j, 0}).item<float>() * 10.0;
      crr_point.y = input_tensor->index({i, j, 1}).item<float>() * 10.0;
      crr_point.z = input_tensor->index({i, j, 2}).item<float>() * 10.0;
      crr_point.g = 1.0;
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
  c10::IntArrayRef test_tensor_shape = {4, 4, 3};
  torch::Device cuda_device = torch::kCUDA;
  at::Tensor test_tensor = at::rand(test_tensor_shape, cuda_device);

  at::Tensor fps_sampled_tensor_indices = furthest_point_sampling(&test_tensor, 1, false);
  at::Tensor fps_sampled_tensor = extract_tensor_from_indices(
    &fps_sampled_tensor_indices,
    &test_tensor);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr full_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr fps_sampled_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

  std::cout << "test_tensor: \n" << test_tensor << std::endl;
  std::cout << "fps_sampled_tensor_indices: \n" << fps_sampled_tensor_indices << std::endl;
  std::cout << "fps_sampled_tensor:  \n" << fps_sampled_tensor << std::endl;

  torch_tensor_to_pcl_cloud(&test_tensor, full_cloud);
  torch_tensor_to_pcl_cloud(&fps_sampled_tensor, full_cloud);


  return EXIT_SUCCESS;

}
