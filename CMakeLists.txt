cmake_minimum_required(VERSION 3.20)

project(dzeta_agi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(dzeta_core INTERFACE)
target_include_directories(dzeta_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dzeta
)

enable_testing()

add_executable(dzeta_smoke tests/smoke.cpp)
target_link_libraries(dzeta_smoke PRIVATE dzeta_core)
add_test(NAME dzeta_smoke COMMAND dzeta_smoke)
