# ovserver (GPL-3.0: links OpenVINS). Separate CMake project; links the installed ov_msckf_lib.
cmake_minimum_required(VERSION 3.16)
project(ovserver CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")   # test build: asserts live (no NDEBUG)
endif()

set(OV_PREFIX "$ENV{HOME}/.cache/dss/cpp" CACHE PATH "install prefix of OpenCV/Ceres/OpenVINS")
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs features2d video calib3d
             PATHS "${OV_PREFIX}/lib/cmake/opencv4" NO_DEFAULT_PATH)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Ceres REQUIRED PATHS "${OV_PREFIX}/lib/cmake/Ceres" NO_DEFAULT_PATH)

add_library(ov_msckf_lib SHARED IMPORTED)
set_target_properties(ov_msckf_lib PROPERTIES
  IMPORTED_LOCATION "${OV_PREFIX}/lib/libov_msckf_lib.dylib"
  INTERFACE_INCLUDE_DIRECTORIES "${OV_PREFIX}/include/open_vins"
  INTERFACE_COMPILE_DEFINITIONS "ROS_AVAILABLE=0;ENABLE_ARUCO_TAGS=0")
target_link_libraries(ov_msckf_lib INTERFACE Eigen3::Eigen ${OpenCV_LIBS} Boost::filesystem Ceres::ceres)

add_executable(ovserver src/main.cpp)
target_link_libraries(ovserver PRIVATE ov_msckf_lib)
set_target_properties(ovserver PROPERTIES BUILD_RPATH "${OV_PREFIX}/lib" INSTALL_RPATH "${OV_PREFIX}/lib")
install(TARGETS ovserver RUNTIME DESTINATION bin)
