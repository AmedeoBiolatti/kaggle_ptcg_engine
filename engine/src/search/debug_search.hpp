#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ptcg::search {

enum class DebugGameKind {
  Tictactoe,
  Kuhn,
};

struct DebugAction {
  int key = 0;
  std::string label;
};

struct DebugGameState {
  DebugGameKind kind = DebugGameKind::Tictactoe;
  int current_player = 0;
  int result = -1;  // -1 ongoing, 0/1 winner, 2 draw
  double payoff0 = 0.0;
  uint64_t rng = 1;

  std::array<int8_t, 9> ttt_board{};

  std::array<int8_t, 2> kuhn_cards{{-1, -1}};
  std::string kuhn_history;
};

struct DebugEvalInput {
  DebugGameKind kind = DebugGameKind::Tictactoe;
  int player = 0;
  std::vector<float> observation;
  std::vector<DebugAction> actions;
};

struct DebugEvalOutput {
  double value = 0.0;  // acting-player POV
  std::vector<double> logits;
};

struct DebugSearchStats {
  std::string game;
  std::string algo;
  int sims = 0;
  int dets = 0;
  int sampled_worlds = 0;
  int invalid_worlds = 0;
  int eval_calls = 0;
  int eval_rows = 0;
  int terminal_evals = 0;
};

struct DebugSearchResult {
  bool terminal = false;
  int root_player = 0;
  double value = 0.0;  // root-player POV
  std::vector<DebugAction> actions;
  std::vector<double> childN;
  std::vector<double> childW;  // root-player POV sums
  DebugSearchStats stats;
};

using DebugEvaluator =
    std::function<std::vector<DebugEvalOutput>(const std::vector<DebugEvalInput>&)>;

std::vector<std::string> debug_games();
DebugGameKind debug_game_kind_from_name(const std::string& name);
std::string debug_game_name(DebugGameKind kind);

DebugGameState debug_new_game(DebugGameKind kind, uint64_t seed);
std::vector<DebugAction> debug_legal_actions(const DebugGameState& state);
DebugGameState debug_step_state(const DebugGameState& state, int action_key,
                                uint64_t seed = 0);
DebugGameState debug_determinize(const DebugGameState& state, int perspective,
                                 uint64_t seed);
std::vector<float> debug_observe(const DebugGameState& state, int perspective = -1);
std::string debug_public_state(const DebugGameState& state);
double debug_terminal_value(const DebugGameState& state, int player);

DebugSearchResult debug_puct_search(const DebugGameState& root_state, int n_sims,
                                    uint64_t seed, const DebugEvaluator& evaluator);
DebugSearchResult debug_pimc_search(const DebugGameState& root_state, int n_sims,
                                    uint64_t seed, int dets,
                                    const DebugEvaluator& evaluator);
DebugSearchResult debug_maple_search(const DebugGameState& root_state, int n_sims,
                                     uint64_t seed, int dets,
                                     const DebugEvaluator& evaluator);
DebugSearchResult debug_search(const std::string& game, const DebugGameState& state,
                               const std::string& algo, int n_sims,
                               uint64_t seed, int dets,
                               const DebugEvaluator& evaluator);

}  // namespace ptcg::search
