# CMakeList.txt: CMake_RaytracingPractise 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project ("RaytracingPractise")

# Set to c++11
set ( CMAKE_CXX_STANDARD 11 )

set( ONE_WEEKEND
	src/camera.h
	src/color.h
	src/hittable.h
	src/hittable_list.h
	src/main.cc
	src/material.h
	src/ray.h
	src/sphere.h
	src/vec3.h
	src/math_constants.h
)

add_executable(RaytracingPractise ${ONE_WEEKEND})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build) 

# TODO: 如有需要，请添加测试并安装目标。
