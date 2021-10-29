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
find_package(CUDNN 8.0 REQUIRED)

#list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB")
#find_package(OpenVDB REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${Torch_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

add_executable(test_pointnet2_utils test/test_pointnet2_utils.cpp
                                    src/pointnet2_utils.cpp)
target_link_libraries(test_pointnet2_utils ${PCL_LIBRARIES} ${TORCH_LIBRARIES})


add_executable(test_pointnet2_core_modules      test/test_pointnet2_core.cpp
                                                src/pointnet2_core.cpp
                                                src/pointnet2_utils.cpp)
target_link_libraries(test_pointnet2_core_modules ${PCL_LIBRARIES} ${TORCH_LIBRARIES})

add_executable(train_pointnet2_sem_seg src/train_pointnet2_sem_seg.cpp
                                       src/pointnet2_sem_seg.cpp
                                       src/pointnet2_core.cpp
                                       src/pointnet2_utils.cpp
                                       src/uneven_ground_dataset.cpp)
target_link_libraries(train_pointnet2_sem_seg ${PCL_LIBRARIES} ${TORCH_LIBRARIES} stdc++fs OpenMP::OpenMP_CXX)

add_executable(train_pointnet2_sem_seg_poss src/train_pointnet2_sem_seg_poss.cpp
                                       src/pointnet2_sem_seg.cpp
                                       src/pointnet2_core.cpp
                                       src/pointnet2_utils.cpp
                                       src/poss_dataset.cpp)
target_link_libraries(train_pointnet2_sem_seg_poss ${PCL_LIBRARIES} ${TORCH_LIBRARIES} stdc++fs OpenMP::OpenMP_CXX)

add_executable(inference_pointnet2_sem_seg src/inference_pointnet2_sem_seg.cpp
                                       src/pointnet2_sem_seg.cpp
                                       src/pointnet2_core.cpp
                                       src/pointnet2_utils.cpp
                                       src/poss_dataset.cpp)
target_link_libraries(inference_pointnet2_sem_seg ${PCL_LIBRARIES} ${TORCH_LIBRARIES} stdc++fs OpenMP::OpenMP_CXX)

#add_executable(test_openvdb  test/test_openvdb.cpp)
#target_link_libraries(test_openvdb OpenVDB::openvdb ${PCL_LIBRARIES})

install(TARGETS test_pointnet2_utils 
                test_pointnet2_core_modules 
                train_pointnet2_sem_seg 
                train_pointnet2_sem_seg_poss
                inference_pointnet2_sem_seg
 #               test_openvdb
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS test_pointnet2_utils 
                test_pointnet2_core_modules 
                train_pointnet2_sem_seg 
                train_pointnet2_sem_seg_poss
                inference_pointnet2_sem_seg
 #               test_openvdb
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include/)
