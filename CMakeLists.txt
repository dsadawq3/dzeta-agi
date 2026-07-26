cmake_minimum_required(VERSION 3.20)

project(dzeta_agi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DZETA_NATIVE_SIMD "Enable native CPU SIMD/autovectorization flags for local benchmark builds" OFF)

add_library(dzeta_core INTERFACE)
target_include_directories(dzeta_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/src/dzeta
)
# Optimization flags are skipped for Debug configurations: MSVC hard-errors
# on /O2 together with the default Debug /RTC1 (D8016).
if(DZETA_NATIVE_SIMD)
    target_compile_definitions(dzeta_core INTERFACE DZETA_NATIVE_SIMD=1)
    if(MSVC)
        target_compile_options(dzeta_core INTERFACE $<$<NOT:$<CONFIG:Debug>>:/O2 /arch:AVX2>)
    elseif(WIN32)
        target_compile_options(dzeta_core INTERFACE $<$<NOT:$<CONFIG:Debug>>:-O3 -march=x86-64-v3>)
    else()
        target_compile_options(dzeta_core INTERFACE $<$<NOT:$<CONFIG:Debug>>:-O3 -march=native>)
    endif()
else()
    if(MSVC)
        target_compile_options(dzeta_core INTERFACE $<$<NOT:$<CONFIG:Debug>>:/O2>)
    else()
        target_compile_options(dzeta_core INTERFACE $<$<NOT:$<CONFIG:Debug>>:-O2>)
    endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(dzeta_core INTERFACE Threads::Threads)

enable_testing()

add_executable(dzeta_smoke tests/smoke.cpp)
target_link_libraries(dzeta_smoke PRIVATE dzeta_core)
add_test(NAME dzeta_smoke COMMAND dzeta_smoke)

add_executable(dzeta_learning tests/learning.cpp)
target_link_libraries(dzeta_learning PRIVATE dzeta_core)
add_test(NAME dzeta_learning COMMAND dzeta_learning)

add_executable(dzeta_parallel tests/parallel.cpp)
target_link_libraries(dzeta_parallel PRIVATE dzeta_core)
add_test(NAME dzeta_parallel COMMAND dzeta_parallel)

add_executable(dzeta_persistence tests/persistence.cpp)
target_link_libraries(dzeta_persistence PRIVATE dzeta_core)
add_test(NAME dzeta_persistence COMMAND dzeta_persistence)

add_executable(dzeta_stochastic tests/stochastic.cpp)
target_link_libraries(dzeta_stochastic PRIVATE dzeta_core)
add_test(NAME dzeta_stochastic COMMAND dzeta_stochastic)

add_executable(dzeta_tokenizer tests/tokenizer.cpp)
target_link_libraries(dzeta_tokenizer PRIVATE dzeta_core)
add_test(NAME dzeta_tokenizer COMMAND dzeta_tokenizer)

add_executable(dzeta_prompt_deflation tests/prompt_deflation.cpp)
target_link_libraries(dzeta_prompt_deflation PRIVATE dzeta_core)
add_test(NAME dzeta_prompt_deflation COMMAND dzeta_prompt_deflation)

add_executable(dzeta_determinism tests/determinism.cpp)
target_link_libraries(dzeta_determinism PRIVATE dzeta_core)
add_test(NAME dzeta_determinism COMMAND dzeta_determinism)

add_executable(dzeta_wave_invariance tests/wave_invariance.cpp)
target_link_libraries(dzeta_wave_invariance PRIVATE dzeta_core)
add_test(NAME dzeta_wave_invariance COMMAND dzeta_wave_invariance)

add_executable(dzeta_translation_transfer tests/translation_transfer.cpp)
target_link_libraries(dzeta_translation_transfer PRIVATE dzeta_core)
add_test(NAME dzeta_translation_transfer COMMAND dzeta_translation_transfer)

add_executable(dzeta_repetition tests/repetition.cpp)
target_link_libraries(dzeta_repetition PRIVATE dzeta_core)
add_test(NAME dzeta_repetition COMMAND dzeta_repetition)

add_executable(dzeta_persistence_v3 tests/persistence_v3.cpp)
target_link_libraries(dzeta_persistence_v3 PRIVATE dzeta_core)
add_test(NAME dzeta_persistence_v3 COMMAND dzeta_persistence_v3)

# Tests verify exclusively through assert(); keep it active in every
# configuration so a Release ctest run cannot pass vacuously (the built-in
# Release flags define NDEBUG, which compiles assert away).
set(DZETA_TEST_TARGETS
    dzeta_smoke dzeta_learning dzeta_parallel dzeta_persistence
    dzeta_stochastic dzeta_tokenizer dzeta_prompt_deflation dzeta_determinism
    dzeta_wave_invariance dzeta_translation_transfer dzeta_repetition
    dzeta_persistence_v3)
foreach(dzeta_test_target IN LISTS DZETA_TEST_TARGETS)
    if(MSVC)
        target_compile_options(${dzeta_test_target} PRIVATE /UNDEBUG)
    else()
        target_compile_options(${dzeta_test_target} PRIVATE -UNDEBUG)
    endif()
endforeach()

add_executable(dzeta_train_smoke benchmarks/train_smoke.cpp)
target_link_libraries(dzeta_train_smoke PRIVATE dzeta_core)

add_executable(dzeta_inspect_model benchmarks/inspect_model.cpp)
target_link_libraries(dzeta_inspect_model PRIVATE dzeta_core)
