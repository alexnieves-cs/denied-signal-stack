# Task 0.1 spike helpers (GPL-3.0: links OpenVINS). Separate project; never add_subdirectory(ov_msckf).
cmake_minimum_required(VERSION 3.16)
project(dss_spike CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")   # no NDEBUG: asserts stay live

set(OV_PREFIX "$ENV{HOME}/.cache/dss/cpp" CACHE PATH "install prefix of OpenCV/Ceres/OpenVINS")
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs features2d video calib3d
             PATHS "${OV_PREFIX}/lib/cmake/opencv4" NO_DEFAULT_PATH)
find_package(Boost REQUIRED COMPONENTS filesystem)
find_package(Ceres REQUIRED PATHS "${OV_PREFIX}/lib/cmake/Ceres" NO_DEFAULT_PATH)

# OpenVINS installs no CMake package: import its shared library by hand.
add_library(ov_msckf_lib SHARED IMPORTED)
set_target_properties(ov_msckf_lib PROPERTIES
  IMPORTED_LOCATION "${OV_PREFIX}/lib/libov_msckf_lib.dylib"
  INTERFACE_INCLUDE_DIRECTORIES "${OV_PREFIX}/include/open_vins"
  INTERFACE_COMPILE_DEFINITIONS "ROS_AVAILABLE=0;ENABLE_ARUCO_TAGS=0")
target_link_libraries(ov_msckf_lib INTERFACE Eigen3::Eigen ${OpenCV_LIBS} Boost::filesystem Ceres::ceres)

foreach(t sim_rmse klt_smoke)
  add_executable(${t} ${t}.cpp)
  target_link_libraries(${t} PRIVATE ov_msckf_lib)
  set_target_properties(${t} PROPERTIES BUILD_RPATH "${OV_PREFIX}/lib")
endforeach()
