cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(deploy_vit)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main src/main.cpp library/model.cpp library/utils.cpp)
target_link_libraries(main "${TORCH_LIBRARIES}" ${OpenCV_LIBS}) 
target_include_directories(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" ${OpenCV_INCLUDE_DIRS})
set_property(TARGET main PROPERTY CXX_STANDARD 17)