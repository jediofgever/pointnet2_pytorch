#include <vector>
#include <torch/torch.h>
#include <torch/cuda.h>
#include <iostream>

/**
 * @brief  Sample num_samples points from points
           according to farthest point sampling (FPS) algorithm.
 * @param full_tensor
 * @param num_samples
 * @return at::Tensor
 */
at::Tensor fps(at::Tensor * full_tensor, int num_samples, bool debug = false)
{
  at::Tensor fps_sampled_tensor;
  torch::Device device = full_tensor->device();
  c10::IntArrayRef input_shape = full_tensor->sizes();

  if (debug) {
    std::cout << "fps: input_shape." << input_shape << std::endl;
    std::cout << "fps: full_tensor." << *full_tensor << std::endl;
    std::cout << "fps: device." << device << std::endl;
  }

  c10::IntArrayRef output_shape = {input_shape.front(), num_samples};

  // tensor options for indice tensors
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
      full_tensor->index({batch_indices, farthest, torch::indexing::Slice()}).view(
      {input_shape.front(), 1, 3});

    at::Tensor dist = torch::sum((full_tensor->subtract(centroid)).pow(2), -1);

    at::Tensor mask = dist < distance;

    distance.index_put_({mask}, dist.index({mask}));

    farthest = std::get<1>(torch::max(distance, -1));

  }
  if (debug) {
    std::cout << "fps: resulting centroids." << centroids << std::endl;
  }
  return centroids;
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

int main()
{
  // check if cuda is there
  test_if_cuda_avail();

  // to test FPS(Furthest-point-sampleing algorithm)
  c10::IntArrayRef test_tensor_shape = {20, 20, 3};
  torch::Device cuda_device = torch::kCUDA;
  at::Tensor test_tensor = at::rand(test_tensor_shape, cuda_device);

  fps(&test_tensor, 10, true);

  return EXIT_SUCCESS;
}
