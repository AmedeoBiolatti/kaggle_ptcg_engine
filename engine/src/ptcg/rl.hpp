#pragma once
#include <cstdint>
#include <vector>

#include "ptcg/state.hpp"

// RL-facing layer (S5): a fixed-size observation encoder, a unified single-select
// action interface (main actions + sub-decisions collapsed into one masked space),
// a vectorized batched environment, and a native random-rollout self-play actor.
// All raw-pointer / POD interfaces so ptcg_core stays free of pybind; bindings.cpp
// wraps these with numpy.
namespace ptcg {

constexpr int RL_MAX_ACTIONS = 64;  // cap on options at any single decision
constexpr int PPO_ACTION_FEAT_DIM = 128;
constexpr int PPO_CARD_SLOTS = 64;
constexpr int PPO_CARD_FEAT_DIM = 80;
constexpr int PPO_DECK_SLOTS = 60;
constexpr int PPO_BELIEF_SLOTS = 40;
constexpr int PPO_BELIEF_SUMMARY_DIM = 52;
constexpr int PPO_OPP_SELF = 0;
constexpr int PPO_OPP_RANDOM = 1;
constexpr int PPO_OPP_HEURISTIC = 2;
constexpr int PPO_OPP_940 = 3;
constexpr int PPO_OPP_BEAM_940 = 4;
constexpr int PPO_REWARD_TERMINAL = 0;
constexpr int PPO_REWARD_PRIZE_DELTA = 1;
constexpr int PPO_REWARD_TERMINAL_PLUS_DELTA = 2;

struct PpoStepProfile {
  int repeats = 0;
  int envs = 0;
  long long env_steps = 0;
  long long terminal_resets = 0;
  long long opponent_steps = 0;
  long long total_ns = 0;
  long long pre_options_ns = 0;
  long long learner_step_ns = 0;
  long long opponent_advance_ns = 0;
  long long opponent_options_ns = 0;
  long long opponent_action_ns = 0;
  long long opponent_step_ns = 0;
  long long auto_pending_ns = 0;
  long long auto_main_options_ns = 0;
  long long auto_main_apply_ns = 0;
  long long auto_pending_decisions = 0;
  long long auto_main_decisions = 0;
  long long reward_reset_ns = 0;
  long long obs_ns = 0;
  long long post_options_ns = 0;
  long long mask_ns = 0;
  long long action_features_ns = 0;
  long long card_features_ns = 0;
};

// Length of the encoded observation vector (acting-player POV).
int rl_obs_dim();

// Encode `st` from the acting player's (st.yourIndex) POV into out[0..rl_obs_dim()).
void rl_encode_obs(const GameState& st, float* out);

// Lightweight mover-POV heuristic used by native value MCTS.
float rl_heuristic_value(const GameState& st);
int rl_native_940_action(const GameState& st);
int rl_meta_beam_action(const GameState& st, int inner_mode, int beam_width,
                        int depth, uint64_t seed);

// Options for the CURRENT decision (pending sub-decision if any, else MAIN-phase).
// Fills mask[RL_MAX_ACTIONS] (1 = legal index). Returns the option count.
int rl_legal_mask(const GameState& st, uint8_t* mask);
void rl_action_features(const GameState& st, float* out);
void rl_card_features(const GameState& st, float* out);
void rl_deck_features(const std::vector<int>& deck, float* out);
void rl_belief_features(const GameState& st, float* out);
void rl_belief_summary(const GameState& st, float* out);

struct RlOptionSet {
  bool pending = false;
  int n = 0;
  std::vector<Descriptor> descriptors;  // main-phase options; empty for pending choices
};

// Build a reusable option set for search nodes. For MAIN decisions this stores
// compact Actions so child expansion does not need to regenerate legal_main().
RlOptionSet rl_options(const GameState& st);
RlOptionSet rl_options_from_descriptors(std::vector<Descriptor> descriptors);

// Advance one AGENT decision by choosing option `action` (index into the current
// option list), then auto-resolve forced (<=1 option) and multi-select sub-decisions
// with `rng` until the next single-select multi-option decision or game end.
// When `next` is non-null it receives exactly what rl_options() would return for
// the resulting state, so callers can skip regenerating legal_main(). `next` may
// alias `opts`: it is only written after `opts` is last read.
void rl_step(GameState& st, int action, uint64_t& rng);
void rl_step_cached(GameState& st, const RlOptionSet& opts, int action,
                    uint64_t& rng, RlOptionSet* next = nullptr);

struct MctsResult {
  bool terminal = false;
  int n_act = 0;
  std::vector<double> childN;
  std::vector<double> childW;
};

// Fully native uniform-prior MCTS using rl_heuristic_value at leaves.
MctsResult rl_value_mcts(const GameState& root_state, int n_sims, uint64_t seed,
                         double c_puct = 1.5);

// True when `st` is at a genuine agent decision (>=2 options, single-select).
bool rl_is_agent_decision(const GameState& st);

// --- vectorized batch environment -----------------------------------------
// N independent self-play games sharing one decklist pairing. One policy controls
// the player to move in every game (self-play); reward is from that mover's POV.
//
// RNG + threading model (all batch envs below): every game owns an
// independent RNG stream derived from (seed, game index), so each game's
// trajectory is a pure function of (seed, index) — independent of batch
// order and of `threads`. Stepping fans games out over a shared worker pool;
// per-game work touches only disjoint state/output slices and read-only
// global tables. `threads`: 0 = auto (hardware concurrency), 1 = serial.
class BatchEnv {
 public:
  BatchEnv(std::vector<int> deck0, std::vector<int> deck1, int n, uint64_t seed,
           int threads = 0);
  int size() const { return static_cast<int>(games_.size()); }

  void reset_all();
  // Apply actions[i] to game i, auto-reset finished games, and write the resulting
  // obs / reward / done / legal-mask for every game. Buffers are caller-owned:
  //   obs   : float [N * rl_obs_dim()]
  //   reward: float [N]   (+1 win / -1 loss / 0 draw|ongoing, mover's POV)
  //   done  : uint8 [N]
  //   mask  : uint8 [N * RL_MAX_ACTIONS]
  void step(const int* actions, float* obs, float* reward, uint8_t* done,
            uint8_t* mask);
  // obs + mask for all games without stepping (e.g. right after reset_all).
  void observe(float* obs, uint8_t* mask) const;

 private:
  void reset(int i);
  std::vector<GameState> games_;
  // Invariant: opts_[i] == rl_options(games_[i]) whenever control is outside a
  // member function, so step/observe never regenerate legal_main().
  std::vector<RlOptionSet> opts_;
  std::vector<uint64_t> rngs_;  // one independent stream per game
  std::vector<int> deck0_, deck1_;
  int threads_ = 0;
};

// Simplified public vectorized environment. Unlike PpoBatchEnv, it has no hidden
// opponent policy modes: each row exposes the current player and the caller
// supplies one legal action for that current player.
class VectorEnv {
 public:
  VectorEnv(std::vector<int> deck0, std::vector<int> deck1, int n,
            uint64_t seed, int threads = 0);
  int size() const { return static_cast<int>(games_.size()); }

  void reset_all();
  void observe(float* obs, uint8_t* mask, int32_t* player,
               int32_t* result) const;
  void observe_ids(int32_t* in_play, int32_t* zones, int32_t* player_counts,
                   int32_t* player_status, int32_t* global,
                   int32_t* action_meta, int32_t* action_options,
                   int32_t* action_deck, uint8_t* action_mask,
                   int32_t* player, int32_t* result) const;
  void step(const int* actions, float* obs, float* reward, uint8_t* done,
            uint8_t* mask, int32_t* player, int32_t* result);
  void step_ids(const int* actions, float* reward, uint8_t* done,
                int32_t* in_play, int32_t* zones,
                int32_t* player_counts, int32_t* player_status,
                int32_t* global, int32_t* action_meta,
                int32_t* action_options, int32_t* action_deck,
                uint8_t* action_mask, int32_t* player, int32_t* result);
  const GameState& state_at(int i) const { return games_[i]; }
  const RlOptionSet& options_at(int i) const { return opts_[i]; }

 private:
  void reset(int i);
  std::vector<GameState> games_;
  std::vector<RlOptionSet> opts_;
  std::vector<uint64_t> rngs_;  // one independent stream per game
  std::vector<int> deck0_, deck1_;
  int threads_ = 0;
};

class PpoBatchEnv {
 public:
  PpoBatchEnv(std::vector<int> deck0, std::vector<int> deck1, int n,
              uint64_t seed, int max_steps = 2000, double prize_weight = 0.0,
              int learner_seat = -1, int opponent_mode = PPO_OPP_SELF,
              int reward_mode = PPO_REWARD_TERMINAL, int threads = 0);
  int size() const { return static_cast<int>(games_.size()); }

  void reset_all();
  void observe(float* obs, uint8_t* mask, int32_t* player) const;
  void observe_ids(int32_t* in_play, int32_t* zones, int32_t* player_counts,
                   int32_t* player_status, int32_t* global,
                   int32_t* action_meta, int32_t* action_options,
                   int32_t* action_deck, uint8_t* action_mask,
                   int32_t* player, int32_t* result) const;
  void action_features(float* out) const;
  void card_features(float* out) const;
  void deck_features(float* out) const;
  void belief_features(float* out) const;
  void belief_summary(float* out) const;
  void step_rewards(const int* actions, float* reward, uint8_t* done,
                    int32_t* result, int32_t* episode_len);
  void step_card_features(const int* actions, float* obs, float* reward,
                          uint8_t* done, uint8_t* mask, int32_t* player,
                          int32_t* result, int32_t* episode_len,
                          float* action_features, float* card_features);
  PpoStepProfile profile_step_card_features(const int* actions, int repeats);
  void step(const int* actions, float* obs, float* reward, uint8_t* done,
            uint8_t* mask, int32_t* player, int32_t* result,
            int32_t* episode_len);
  void step_ids(const int* actions, float* reward, uint8_t* done,
                int32_t* result, int32_t* episode_len, int32_t* in_play,
                int32_t* zones, int32_t* player_counts,
                int32_t* player_status, int32_t* global,
                int32_t* action_meta, int32_t* action_options,
                int32_t* action_deck, uint8_t* action_mask,
                int32_t* player);

 private:
  void reset(int i);
  void advance_to_learner_or_done(int i);
  void advance_to_learner_or_done_profiled(int i, PpoStepProfile& profile);
  int opponent_action(const GameState& st, const RlOptionSet& opts,
                      uint64_t& rng);
  int opponent_action_cached(const GameState& st, const RlOptionSet& opts,
                             uint64_t& rng);
  int random_action(const GameState& st, uint64_t& rng);
  int heuristic_action(const GameState& st, const RlOptionSet& opts,
                       uint64_t rng);
  int reward_player(const GameState& st) const;
  double prize_score(const GameState& st, int player) const;
  float terminal_reward(const GameState& st, int player) const;
  float transition_reward(const GameState& st, int player, int i,
                          bool terminal, bool truncated) const;
  std::vector<GameState> games_;
  // Invariant: opts_[i] == rl_options(games_[i]) whenever control is outside a
  // member function, so step/observe/features never regenerate legal_main().
  std::vector<RlOptionSet> opts_;
  std::vector<uint64_t> rngs_;  // one independent stream per game
  std::vector<int> deck0_, deck1_;
  std::vector<int> episode_len_;
  std::vector<float> last_prize_score_;
  int threads_ = 0;
  int max_steps_ = 2000;
  double prize_weight_ = 0.0;
  int learner_seat_ = -1;
  int opponent_mode_ = PPO_OPP_SELF;
  int reward_mode_ = PPO_REWARD_TERMINAL;
};

// --- native random self-play actor (throughput / data generation) ----------
struct SelfplayResult {
  long p0_wins = 0, p1_wins = 0, draws = 0, unfinished = 0, total_steps = 0;
  double seconds = 0.0;
};
// Run `n` full games with a uniform-random policy entirely in C++ (no per-step
// Python), capping each game at `max_steps`.
SelfplayResult rl_selfplay(const std::vector<int>& deck0, const std::vector<int>& deck1,
                           int n, uint64_t seed, int max_steps);

}  // namespace ptcg
