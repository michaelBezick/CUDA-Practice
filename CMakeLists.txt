cmake_minimum_required(VERSION 3.18)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Let CMake generate the right flags for each compiler
# 86 = Ampere (RTX 3060). You can do a list like 75;86 if you really need multi-arch.
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(hello_cuda
  hello_cuda.cu
)

add_executable(vector_add
  vector_add.cu
)

add_executable(mat_vec
  matrix_vector.cu
)

# Always generate compile_commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
