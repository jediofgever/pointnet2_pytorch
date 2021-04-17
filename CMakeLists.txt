cmake_minimum_required(VERSION 3.15)
project(pcl_learning)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(Torch_DIR "/opt/libtorch/share/cmake/Torch") 
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
option(USE_CUDA "Use CUDA" ON)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_60,code=sm_60)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

find_package(PCL REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${Torch_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS})

add_executable(test_pcl src/test_pcl.cpp)
target_link_libraries(test_pcl ${PCL_LIBRARIES})

add_executable(test_torch_cpp src/test_torch_cpp.cpp)
target_link_libraries(test_torch_cpp ${PCL_LIBRARIES} ${TORCH_LIBRARIES})

CUDA_ADD_EXECUTABLE(cuda_test src/cuda_test.cu)
target_link_libraries(cuda_test ${CUDA_LIBRARIES})

install(TARGETS test_pcl test_torch_cpp
                cuda_test
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS test_pcl test_torch_cpp cuda_test
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include/)

