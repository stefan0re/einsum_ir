cmake_minimum_required(VERSION 3.14)
project(model)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_library(LIBXSMM_LIBRARY
    NAMES xsmm libxsmm
    PATHS ${CMAKE_SOURCE_DIR}/../../libxsmm/lib
    NO_DEFAULT_PATH
)

include_directories(
    ${CMAKE_SOURCE_DIR}/../../libxsmm/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(SOURCES
    src/bench_model.cpp
    src/common/common.cpp
    src/common/interpolation.cpp
    src/m4/bench_m4.cpp
    src/m4/model_m4.cpp
    src/zen5/bench_zen5.cpp
    src/zen5/model_zen5.cpp
    src/a76/bench_a76.cpp
    src/a76/model_a76.cpp
    src/generic/model_generic.cpp
)

add_executable(bench_model ${SOURCES})

target_link_libraries(bench_model ${LIBXSMM_LIBRARY})
