cmake_minimum_required(VERSION 3.18)
project(ptcg_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  # /GL + /LTCG: whole-program (cross-TU) optimization — state.cpp/card_db/rl
  # helpers inline across translation-unit boundaries.
  # /arch:AVX2: vectorized float feature encoders (verified bit-exact by the
  # rollout-hash harness; drop this flag first if hashes ever diverge).
  add_compile_options(/O2 /W4 /permissive- /EHsc /Zc:__cplusplus /GL /arch:AVX2)
  add_link_options(/LTCG)
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /LTCG")
else()
  add_compile_options(-O3 -Wall -Wextra)
endif()

# --- Pure C++ engine core (no Python) -------------------------------------
# Static lib for the native engine replica and release utilities.
add_library(ptcg_core STATIC
  src/ptcg/version.cpp
  src/ptcg/card_db.cpp
  src/ptcg/state.cpp
  src/ptcg/rl.cpp
  src/search/debug_search.cpp
)
target_include_directories(ptcg_core PUBLIC src)

# --- Python binding -------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(ptcg_engine src/bindings.cpp)
target_link_libraries(ptcg_engine PRIVATE ptcg_core)
