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

  

  return fps_sampled_tensor;
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

  fps(&test_tensor, 50, true);

  return EXIT_SUCCESS;
}
