cmake_minimum_required(VERSION 3.15)
project(pointnet2_pytorch)

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

add_executable(test_pointnet2_utils src/test_pointnet2_utils.cpp
                                    src/pointnet2_utils.cpp)
target_link_libraries(test_pointnet2_utils ${PCL_LIBRARIES} ${TORCH_LIBRARIES})


add_executable(test_pointnet2_core_modules src/test_pointnet2_core.cpp
                                 src/pointnet2_core.cpp
                                 src/pointnet2_utils.cpp)
target_link_libraries(test_pointnet2_core_modules ${PCL_LIBRARIES} ${TORCH_LIBRARIES})


add_executable(train_pointnet2_sem_seg  src/train_pointnet2_sem_seg.cpp
                                        src/pointnet2_sem_seg.cpp
                                       src/pointnet2_core.cpp
                                       src/pointnet2_utils.cpp
                                       src/uneven_ground_dataset.cpp)
target_link_libraries(train_pointnet2_sem_seg ${PCL_LIBRARIES} ${TORCH_LIBRARIES})


install(TARGETS test_pointnet2_utils test_pointnet2_core_modules train_pointnet2_sem_seg
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS test_pointnet2_utils test_pointnet2_core_modules train_pointnet2_sem_seg
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include/)
