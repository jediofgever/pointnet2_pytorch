#include <vector>
#include <iostream>
#include <pcl/io/pcd_io.h>

int main(int argc, char const * argv[])
{

  pcl::PointCloud<pcl::PointXYZRGB> test_cloud;
  if (!pcl::io::loadPCDFile("../data/error.pcd", test_cloud)) {
    std::cout << "Testing pcl, test cloud has:" << std::endl;
    std::cout << test_cloud.points.size() << " points" << std::endl;
  } else {
    std::cerr << "Could not read test cd file " << std::endl;
  }
  return 0;
}
