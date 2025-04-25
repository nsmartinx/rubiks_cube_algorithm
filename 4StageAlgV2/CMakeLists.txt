cmake_minimum_required(VERSION 3.20)
project(RubiksCudaSolver LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_ARCHITECTURES 89)

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_CPP src/*.cpp)
file(GLOB SRC_CU  src/*.cu)

add_executable(rubiks ${SRC_CPP} ${SRC_CU})

# Compile flags for host and device
target_compile_options(rubiks PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
  $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

# Link in the CUDA runtime
find_library(CUDART_LIBRARY cudart HINTS /usr/local/cuda/lib64)
if (CUDART_LIBRARY)
  target_link_libraries(rubiks PRIVATE ${CUDART_LIBRARY})
else()
  message(FATAL_ERROR "libcudart not found")
endif()
