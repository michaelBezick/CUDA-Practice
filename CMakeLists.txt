cmake_minimum_required(VERSION 3.18)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optional: make CUDA files show up as C++
set_source_files_properties(${CMAKE_SOURCE_DIR}/src/main.cu PROPERTIES LANGUAGE CUDA)

# Generate compile_commands.json for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(my_app
  src/main.cu
  src/whatever.cpp
)
# If you need specific arch flags:
# target_compile_options(my_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--generate-code=arch=compute_86,code=sm_86>)
