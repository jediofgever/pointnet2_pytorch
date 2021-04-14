cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(pcl_learning)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(Torch_DIR "/opt/libtorch/share/cmake/Torch") 

find_package(PCL REQUIRED)
find_package(Torch REQUIRED)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${Torch_INCLUDE_DIRS})

add_executable(test_pcl src/test_pcl.cpp)
target_link_libraries(test_pcl ${PCL_LIBRARIES})

add_executable(test_torch_cpp src/test_torch_cpp.cpp)
target_link_libraries(test_torch_cpp ${TORCH_LIBRARIES})

install(TARGETS test_pcl test_torch_cpp
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS test_pcl test_torch_cpp
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include/)

