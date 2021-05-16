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
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>

int main()
{

  openvdb::initialize();
  std::string segmneted_pcl_filename = "/home/ros2-foxy/uneven_ground_dataset/train_0.pcd";

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
    new pcl::PointCloud<pcl::PointXYZRGB>);

  if (!pcl::io::loadPCDFile(segmneted_pcl_filename, *cloud)) {
    std::cout << "Gonna load a cloud with " << cloud->points.size() <<
      " points" <<
      std::endl;
  } else {
    std::cerr << "Could not read PCD file: " << segmneted_pcl_filename << std::endl;
  }

  std::vector<openvdb::Vec3R> position;

  // Compute the signed distance from the surface of the sphere of each
  // voxel within the bounding box and insert the value into the grid
  // if it is smaller in magnitude than the background value.

  for (auto && i : cloud->points) {
    position.push_back(openvdb::Vec3R(i.x, i.y, i.z));
  }

  openvdb::points::PointAttributeVector<openvdb::Vec3R> positionsWrapper(position);

  int pointsPerVoxel = 10;
  float voxelSize =
    openvdb::points::computeVoxelSize(positionsWrapper, pointsPerVoxel) / 2.0;

  // Create a transform using this voxel-size.
  openvdb::math::Transform::Ptr transform =
    openvdb::math::Transform::createLinearTransform(voxelSize);
  // Create a PointDataGrid containing these four points and using the
  // transform given. This function has two template parameters, (1) the codec
  // to use for storing the position, (2) the grid we want to create
  // (ie a PointDataGrid).
  // We use no compression here for the positions.
  openvdb::points::PointDataGrid::Ptr grid =
    openvdb::points::createPointDataGrid<openvdb::points::NullCodec,
      openvdb::points::PointDataGrid>(position, *transform);

  // Set the name of the grid
  grid->setName("Points");

  openvdb::FloatGrid::Ptr vox_grid =
    openvdb::FloatGrid::create(/*background value=*/ 0.0);

  vox_grid->setTransform(
    openvdb::math::Transform::createLinearTransform(/*voxel size=*/ voxelSize));

  openvdb::FloatGrid::Accessor accessor = vox_grid->getAccessor();

  // Iterate over all the leaf nodes in the grid.
  for (auto leafIter = grid->tree().cbeginLeaf(); leafIter; ++leafIter) {
    // Verify the leaf origin.
    std::cout << "Leaf" << leafIter->origin() << std::endl;
    // Extract the position attribute from the leaf by name (P is position).
    const openvdb::points::AttributeArray & array =
      leafIter->constAttributeArray("P");
    // Create a read-only AttributeHandle. Position always uses Vec3f.
    openvdb::points::AttributeHandle<openvdb::Vec3f> positionHandle(array);

    if (leafIter->origin().x() > 0.0) {
      accessor.setValue(leafIter->origin(), leafIter->origin().z() * 100.0);
    }
 
    // Iterate over the point indices in the leaf.
    for (auto indexIter = leafIter->beginIndexOn(); indexIter; ++indexIter) {
      // Extract the voxel-space position of the point.
      openvdb::Vec3f voxelPosition = positionHandle.get(*indexIter);
      // Extract the world-space position of the voxel.
      const openvdb::Vec3d xyz = indexIter.getCoord().asVec3d();
      // Compute the world-space position of the point.
      openvdb::Vec3f worldPosition =
        grid->transform().indexToWorld(voxelPosition + xyz);
      // Verify the index and world-space position of the point
    }
  }

  // Identify the grid as a level set.
  vox_grid->setGridClass(openvdb::GRID_FOG_VOLUME);
  // Name the grid "LevelSetSphere".
  vox_grid->setName("LevelSetSphere");

  // Create a VDB file object.
  openvdb::io::File file("/home/ros2-foxy/mygrids.vdb");
  // Add the grid pointer to a container.
  openvdb::GridPtrVec grids;
  grids.push_back(vox_grid);
  // Write out the contents of the container.
  file.write(grids);
  file.close();

  return EXIT_SUCCESS;
}
