#pragma once
#include <string>

namespace ptcg {

// Engine version string. Bumped as the rules surface grows.
std::string version();

// Smoke-test symbol, proves the C++ core links into the pybind module.
int add(int a, int b);

}  // namespace ptcg
