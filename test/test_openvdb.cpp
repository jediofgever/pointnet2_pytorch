// Copyright (c) 2020 Fetullah Atas, Norwegian University of Life Sciences
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <openvdb/openvdb.h>
#include <openvdb/points/PointConversion.h>
#include <openvdb/points/PointCount.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/VolumeToSpheres.h>
#include <openvdb/tools/FindActiveValues.h>
#include <openvdb/math/BBox.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

int main()
{

  openvdb::initialize();
  std::string segmneted_pcl_filename = "/home/atas/container_office_map.pcd";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);

  if (!pcl::io::loadPCDFile(segmneted_pcl_filename, *cloud)) {
    std::cout << "Gonna load a cloud with " << cloud->points.size() <<
      " points" <<
      std::endl;
  } else {
    std::cerr << "Could not read PCD file: " << segmneted_pcl_filename << std::endl;
  }

  //pcl::io::savePCDFile("../data/fp_pass.pcd", *cloud, false);
  pcl::io::loadPCDFile("", *cloud);
  pcl::io::savePCDFile("../data/rand.pcd", *cloud, false);

  std::vector<openvdb::Vec3R> points;

  for (auto && i : cloud->points) {
    points.push_back(openvdb::Vec3R(i.x, i.y, i.z));
  }

  float voxelSize = 0.05;

  // Create a transform using this voxel-size.
  openvdb::math::Transform::Ptr transform =
    openvdb::math::Transform::createLinearTransform(voxelSize);
  // Create a PointDataGrid containing these four points and using the
  // transform given. This function has two template parameters, (1) the codec
  // to use for storing the position, (2) the grid we want to create
  // (ie a PointDataGrid).
  // We use no compression here for the positions.
  openvdb::points::PointDataGrid::Ptr point_data_grid =
    openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
      openvdb::points::PointDataGrid>(points, *transform);

  // Set the name of the grid
  point_data_grid->setName("Points");

  openvdb::FloatGrid::Ptr float_grid =
    openvdb::FloatGrid::create(/*background value=*/ 0.0);
  float_grid->setTransform(transform);

  openvdb::FloatGrid::Accessor float_grid_accessor = float_grid->getAccessor();

  // Iterate over all the leaf nodes in the grid.
  for (auto leafIter = point_data_grid->tree().cbeginLeaf(); leafIter; ++leafIter) {
    const openvdb::points::AttributeArray & array =
      leafIter->constAttributeArray("P");
    // Create a read-only AttributeHandle. Position always uses Vec3f.
    openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);
    // Iterate over the point indices in the leaf.
    for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
      // Extract the voxel-space position of the point.
      openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
      // Extract the world-space position of the voxel.
      const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
      // Compute the world-space position of the point.
      openvdb::Vec3f worldPosition =
        point_data_grid->transform().indexToWorld(voxelPosition + xyz);
      // Verify the index and world-space position of the point
      float_grid_accessor.setValue(indexIter.getCoord(), 1);
    }
  }

  openvdb::tools::FindActiveValues<openvdb::FloatTree> op(float_grid->tree());
  auto bbx =
    openvdb::CoordBBox(
    openvdb::Coord(-1.0 / voxelSize, -1.0 / voxelSize, -1.0 / voxelSize),
    openvdb::Coord(1.0 / voxelSize, 1.0 / voxelSize, 1.0 / voxelSize));

  auto cube_center_coord =
    float_grid->transform().worldToIndexCellCentered(openvdb::Vec3R(139.592, 145.786, 0.583193));

  std::cout << op.anyActiveValues(bbx) << std::endl;

  //float_grid->fill(bbx, -1, false);
  //float_grid->clip(bbx);

  /*std::vector<float> tmpDistances;
  std::vector<openvdb::Vec3R> tmpPoints;
  tmpPoints.push_back(openvdb::Vec3R(10, 10, 4));
  tmpPoints.push_back(openvdb::Vec3R(12, 10, 3));
  auto csp = openvdb::tools::ClosestSurfacePoint<openvdb::FloatGrid>::create(*float_grid);
  if (csp->search(tmpPoints, tmpDistances)) {
    std::cout << "Dist to curr point is " << tmpDistances[0] << std::endl;
    std::cout << "Dist to curr point is " << tmpDistances[1] << std::endl;
  }*/


  // Create a VDB file object.
  openvdb::io::File file("/home/atas/mygrids.vdb");
  // Add the grid pointer to a container.
  openvdb::GridPtrVec grids;
  //grids.push_back(point_data_grid);
  grids.push_back(float_grid);
  // Write out the contents of the container.
  file.write(grids);
  file.close();

  return EXIT_SUCCESS;
}
