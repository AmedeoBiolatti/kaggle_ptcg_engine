#pragma once

#include <cstdint>
#include <vector>

#include "ptcg/state.hpp"

namespace ptcg {

constexpr int VECTOR_MAX_ACTIONS = 64;

int vector_obs_dim();
void vector_encode_obs(const GameState& st, float* out);
int vector_legal_mask(const GameState& st, uint8_t* mask);
void vector_step_action(GameState& st, int action, uint64_t& rng);

// The current decision's options, captured when the auto-advance loop stops so
// stepping and mask/observe never regenerate legal_main() for the same state.
// `descriptors` holds the FULL MAIN option list (empty for pending decisions,
// whose options live in GameState.pending); `count` is the raw option count.
struct VectorOptions {
  bool pending = false;
  int count = 0;
  std::vector<Descriptor> descriptors;
};

class VectorEnv {
 public:
  VectorEnv(std::vector<int> deck0, std::vector<int> deck1, int n, uint64_t seed);
  int size() const { return static_cast<int>(games_.size()); }

  void reset_all();
  void observe(float* obs, uint8_t* mask, int32_t* player, int32_t* result) const;
  void step(const int* actions, float* obs, float* reward, uint8_t* done,
            uint8_t* mask, int32_t* player, int32_t* result);

 private:
  void reset(int i);
  std::vector<GameState> games_;
  // Invariant: opts_[i] describes games_[i]'s current decision whenever
  // control is outside a member function.
  std::vector<VectorOptions> opts_;
  std::vector<int> deck0_, deck1_;
  uint64_t rng_;
};

}  // namespace ptcg
