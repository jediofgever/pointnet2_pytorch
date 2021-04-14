cmake_minimum_required(VERSION 3.5)
project(pcl_learning)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PCL REQUIRED)
 
include_directories(
    include
    ${PCL_INCLUDE_DIRS})

add_executable(test_pcl src/test_pcl.cpp)
target_link_libraries(test_pcl ${PCL_LIBRARIES})

install(TARGETS test_pcl 
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

install(TARGETS test_pcl 
        RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY include/
        DESTINATION include/)

