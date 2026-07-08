#include "search/debug_search.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace ptcg::search {
namespace {

constexpr uint64_t kLcgMul = 6364136223846793005ULL;
constexpr uint64_t kLcgAdd = 1442695040888963407ULL;
constexpr double kCpuct = 1.25;

uint64_t lcg_next(uint64_t& rng) {
  rng = rng * kLcgMul + kLcgAdd;
  return rng;
}

int bounded_rand(uint64_t& rng, int n) {
  if (n <= 1) return 0;
  return static_cast<int>(lcg_next(rng) % static_cast<uint64_t>(n));
}

bool ieq(const std::string& a, const std::string& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    char ca = static_cast<char>(std::tolower(static_cast<unsigned char>(a[i])));
    char cb = static_cast<char>(std::tolower(static_cast<unsigned char>(b[i])));
    if (ca != cb) return false;
  }
  return true;
}

bool has_action(const std::vector<DebugAction>& actions, int key) {
  for (const auto& a : actions) {
    if (a.key == key) return true;
  }
  return false;
}

std::vector<double> softmax_priors(const std::vector<double>& logits, int n) {
  std::vector<double> priors(std::max(n, 0), 0.0);
  if (n <= 0) return priors;
  double max_logit = 0.0;
  if (!logits.empty()) {
    max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
      double v = i < static_cast<int>(logits.size()) ? logits[i] : 0.0;
      max_logit = std::max(max_logit, v);
    }
  }
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    double v = i < static_cast<int>(logits.size()) ? logits[i] : 0.0;
    priors[i] = std::exp(v - max_logit);
    sum += priors[i];
  }
  if (sum <= 0.0 || !std::isfinite(sum)) {
    std::fill(priors.begin(), priors.end(), 1.0 / static_cast<double>(n));
    return priors;
  }
  for (double& p : priors) p /= sum;
  return priors;
}

void normalize_in_place(std::vector<double>& values) {
  double sum = std::accumulate(values.begin(), values.end(), 0.0);
  if (sum <= 0.0 || !std::isfinite(sum)) {
    if (!values.empty())
      std::fill(values.begin(), values.end(), 1.0 / static_cast<double>(values.size()));
    return;
  }
  for (double& v : values) v /= sum;
}

int ttt_winner(const std::array<int8_t, 9>& b) {
  static constexpr int lines[8][3] = {
      {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6},
      {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6},
  };
  for (const auto& line : lines) {
    int8_t v = b[line[0]];
    if (v >= 0 && v == b[line[1]] && v == b[line[2]]) return v;
  }
  return -1;
}

bool ttt_full(const std::array<int8_t, 9>& b) {
  for (int8_t v : b) {
    if (v < 0) return false;
  }
  return true;
}

void settle_kuhn_showdown(DebugGameState& st, double win_value) {
  int winner = st.kuhn_cards[0] > st.kuhn_cards[1] ? 0 : 1;
  st.result = winner;
  st.payoff0 = winner == 0 ? win_value : -win_value;
}

void settle_kuhn_fold(DebugGameState& st, int winner) {
  st.result = winner;
  st.payoff0 = winner == 0 ? 0.5 : -0.5;
}

DebugEvalOutput default_eval(const DebugGameState& state) {
  DebugEvalOutput out;
  if (state.result >= 0) {
    out.value = debug_terminal_value(state, state.current_player);
    return out;
  }
  if (state.kind == DebugGameKind::Tictactoe) {
    double best = 0.0;
    for (const auto& a : debug_legal_actions(state)) {
      DebugGameState child = debug_step_state(state, a.key, state.rng);
      if (child.result == state.current_player) {
        best = 0.75;
        break;
      }
    }
    if (best == 0.0) {
      DebugGameState opp = state;
      opp.current_player = 1 - state.current_player;
      for (const auto& a : debug_legal_actions(opp)) {
        DebugGameState child = debug_step_state(opp, a.key, opp.rng);
        if (child.result == opp.current_player) {
          best = -0.5;
          break;
        }
      }
    }
    out.value = best;
  } else {
    out.value = 0.0;
  }
  out.logits.assign(debug_legal_actions(state).size(), 0.0);
  return out;
}

std::vector<DebugEvalOutput> eval_batch(
    const std::vector<DebugGameState>& states, const DebugEvaluator& evaluator,
    DebugSearchStats* stats) {
  std::vector<DebugEvalOutput> outs;
  outs.reserve(states.size());
  if (states.empty()) return outs;
  if (!evaluator) {
    for (const auto& st : states) outs.push_back(default_eval(st));
    return outs;
  }

  std::vector<DebugEvalInput> inputs;
  inputs.reserve(states.size());
  for (const auto& st : states) {
    DebugEvalInput in;
    in.kind = st.kind;
    in.player = st.current_player;
    in.observation = debug_observe(st, st.current_player);
    in.actions = debug_legal_actions(st);
    inputs.push_back(std::move(in));
  }
  if (stats) {
    ++stats->eval_calls;
    stats->eval_rows += static_cast<int>(inputs.size());
  }
  outs = evaluator(inputs);
  if (outs.size() != states.size())
    throw std::runtime_error("debug eval callback returned the wrong row count");
  return outs;
}

struct PuctNode {
  DebugGameState state;
  int mover = 0;
  bool terminal = false;
  std::vector<DebugAction> actions;
  std::vector<double> prior;
  std::vector<double> childN;
  std::vector<double> childW;
  std::vector<int> child;
  double totalN = 0.0;
  bool expanded = false;

  explicit PuctNode(DebugGameState s) : state(std::move(s)) {
    mover = state.current_player;
    terminal = state.result >= 0;
  }
};

double eval_and_expand_puct(PuctNode& node, const DebugEvaluator& evaluator,
                            DebugSearchStats* stats) {
  if (node.terminal) {
    ++stats->terminal_evals;
    return debug_terminal_value(node.state, node.mover);
  }
  node.actions = debug_legal_actions(node.state);
  int n = static_cast<int>(node.actions.size());
  node.childN.assign(n, 0.0);
  node.childW.assign(n, 0.0);
  node.child.assign(n, -1);
  std::vector<DebugGameState> states{node.state};
  DebugEvalOutput out = eval_batch(states, evaluator, stats)[0];
  node.prior = softmax_priors(out.logits, n);
  node.expanded = true;
  return out.value;
}

int select_puct_action(const PuctNode& node, int root_player) {
  int n = static_cast<int>(node.actions.size());
  double c = kCpuct * std::sqrt(node.totalN + 1.0);
  bool flip = node.mover != root_player;
  int best = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    double visits = node.childN[i];
    double q = visits > 0.0 ? node.childW[i] / visits : 0.0;
    if (flip) q = -q;
    double p = i < static_cast<int>(node.prior.size()) ? node.prior[i]
                                                       : 1.0 / std::max(n, 1);
    double score = q + c * p / (1.0 + visits);
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

DebugSearchResult make_empty_result(const DebugGameState& root_state,
                                    const std::string& algo, int n_sims,
                                    int dets) {
  DebugSearchResult out;
  out.terminal = root_state.result >= 0;
  out.root_player = root_state.current_player;
  out.value = out.terminal ? debug_terminal_value(root_state, out.root_player) : 0.0;
  out.actions = debug_legal_actions(root_state);
  out.childN.assign(out.actions.size(), 0.0);
  out.childW.assign(out.actions.size(), 0.0);
  out.stats.game = debug_game_name(root_state.kind);
  out.stats.algo = algo;
  out.stats.sims = n_sims;
  out.stats.dets = dets;
  return out;
}

void fill_result_from_puct_root(DebugSearchResult& out, const PuctNode& root) {
  out.terminal = root.terminal;
  out.root_player = root.mover;
  out.actions = root.actions;
  out.childN = root.childN;
  out.childW = root.childW;
  double total = std::accumulate(out.childN.begin(), out.childN.end(), 0.0);
  out.value = total > 0.0
                  ? std::accumulate(out.childW.begin(), out.childW.end(), 0.0) / total
                  : (root.terminal ? debug_terminal_value(root.state, root.mover) : 0.0);
}

std::vector<DebugGameState> sampled_worlds(const DebugGameState& root_state,
                                           int dets, uint64_t seed,
                                           int perspective) {
  int k = root_state.kind == DebugGameKind::Kuhn ? std::max(dets, 1) : 1;
  std::vector<DebugGameState> worlds;
  worlds.reserve(k);
  for (int i = 0; i < k; ++i) {
    uint64_t s = seed + static_cast<uint64_t>(i + 1) * 0x9E3779B97F4A7C15ULL;
    worlds.push_back(debug_determinize(root_state, perspective, s));
  }
  return worlds;
}

struct MapleNode {
  int mover = 0;
  bool expanded = false;
  bool terminal = false;
  std::string info_key;
  std::vector<DebugAction> actions;
  std::vector<double> prior;
  std::vector<double> childN;
  std::vector<double> childW;
  std::vector<int> child;
  double totalN = 0.0;
};

std::string info_key(const DebugGameState& st) {
  if (st.kind == DebugGameKind::Tictactoe) {
    std::string key = "ttt:";
    key.push_back(static_cast<char>('0' + st.current_player));
    key.push_back(':');
    for (int8_t v : st.ttt_board) key.push_back(static_cast<char>('0' + v + 1));
    return key;
  }
  std::string key = "kuhn:";
  key.push_back(static_cast<char>('0' + st.current_player));
  key.push_back(':');
  int card = st.kuhn_cards[st.current_player];
  key.push_back(static_cast<char>('0' + card));
  key.push_back(':');
  key += st.kuhn_history;
  return key;
}

int select_maple_action(const MapleNode& node, int root_player) {
  int n = static_cast<int>(node.actions.size());
  double c = kCpuct * std::sqrt(node.totalN + 1.0);
  bool flip = node.mover != root_player;
  int best = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    double visits = node.childN[i];
    double q = visits > 0.0 ? node.childW[i] / visits : 0.0;
    if (flip) q = -q;
    double p = i < static_cast<int>(node.prior.size()) ? node.prior[i]
                                                       : 1.0 / std::max(n, 1);
    double score = q + c * p / (1.0 + visits);
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

double expand_maple_node(MapleNode& node, const std::vector<DebugGameState>& worlds,
                         int root_player, const DebugEvaluator& evaluator,
                         DebugSearchStats* stats) {
  if (worlds.empty()) {
    node.terminal = true;
    node.expanded = true;
    return 0.0;
  }

  node.mover = worlds.front().current_player;
  node.info_key = info_key(worlds.front());

  std::vector<DebugGameState> eval_states;
  std::vector<int> eval_world_indices;
  double value_sum = 0.0;
  int value_count = 0;

  std::map<int, DebugAction> action_by_key;
  for (int wi = 0; wi < static_cast<int>(worlds.size()); ++wi) {
    const DebugGameState& w = worlds[wi];
    if (w.result >= 0) {
      ++stats->terminal_evals;
      value_sum += debug_terminal_value(w, root_player);
      ++value_count;
      continue;
    }
    for (const auto& a : debug_legal_actions(w)) action_by_key.emplace(a.key, a);
    eval_world_indices.push_back(wi);
    eval_states.push_back(w);
  }

  if (action_by_key.empty()) {
    node.terminal = true;
    node.expanded = true;
    return value_count > 0 ? value_sum / static_cast<double>(value_count) : 0.0;
  }

  for (const auto& [_, action] : action_by_key) node.actions.push_back(action);
  std::unordered_map<int, int> action_pos;
  for (int i = 0; i < static_cast<int>(node.actions.size()); ++i)
    action_pos[node.actions[i].key] = i;

  std::vector<double> prior_sum(node.actions.size(), 0.0);
  std::vector<int> prior_count(node.actions.size(), 0);
  std::vector<DebugEvalOutput> outs = eval_batch(eval_states, evaluator, stats);
  for (int i = 0; i < static_cast<int>(eval_states.size()); ++i) {
    const DebugGameState& st = eval_states[i];
    const DebugEvalOutput& out = outs[i];
    double v_me = st.current_player == root_player ? out.value : -out.value;
    value_sum += v_me;
    ++value_count;

    std::vector<DebugAction> legal = debug_legal_actions(st);
    std::vector<double> priors = softmax_priors(out.logits, static_cast<int>(legal.size()));
    for (int j = 0; j < static_cast<int>(legal.size()); ++j) {
      int pos = action_pos[legal[j].key];
      prior_sum[pos] += priors[j];
      ++prior_count[pos];
    }
  }

  node.prior.resize(node.actions.size(), 0.0);
  for (int i = 0; i < static_cast<int>(node.actions.size()); ++i) {
    node.prior[i] = prior_count[i] > 0
                        ? prior_sum[i] / static_cast<double>(prior_count[i])
                        : 0.0;
  }
  normalize_in_place(node.prior);
  node.childN.assign(node.actions.size(), 0.0);
  node.childW.assign(node.actions.size(), 0.0);
  node.child.assign(node.actions.size(), -1);
  node.expanded = true;
  return value_count > 0 ? value_sum / static_cast<double>(value_count) : 0.0;
}

}  // namespace

std::vector<std::string> debug_games() { return {"tictactoe", "kuhn"}; }

DebugGameKind debug_game_kind_from_name(const std::string& name) {
  if (ieq(name, "tictactoe") || ieq(name, "ttt")) return DebugGameKind::Tictactoe;
  if (ieq(name, "kuhn") || ieq(name, "kuhn_poker")) return DebugGameKind::Kuhn;
  throw std::runtime_error("unknown debug game: " + name);
}

std::string debug_game_name(DebugGameKind kind) {
  switch (kind) {
    case DebugGameKind::Tictactoe:
      return "tictactoe";
    case DebugGameKind::Kuhn:
      return "kuhn";
  }
  return "unknown";
}

DebugGameState debug_new_game(DebugGameKind kind, uint64_t seed) {
  DebugGameState st;
  st.kind = kind;
  st.rng = seed ? seed : 1;
  st.current_player = 0;
  st.result = -1;
  st.payoff0 = 0.0;
  st.ttt_board.fill(-1);
  if (kind == DebugGameKind::Kuhn) {
    std::array<int8_t, 3> deck{{0, 1, 2}};
    uint64_t rng = st.rng;
    for (int i = 2; i > 0; --i) {
      int j = bounded_rand(rng, i + 1);
      std::swap(deck[i], deck[j]);
    }
    st.kuhn_cards[0] = deck[0];
    st.kuhn_cards[1] = deck[1];
    st.kuhn_history.clear();
    st.rng = rng;
  }
  return st;
}

std::vector<DebugAction> debug_legal_actions(const DebugGameState& state) {
  if (state.result >= 0) return {};
  if (state.kind == DebugGameKind::Tictactoe) {
    std::vector<DebugAction> out;
    for (int i = 0; i < 9; ++i) {
      if (state.ttt_board[i] < 0) out.push_back({i, "cell" + std::to_string(i)});
    }
    return out;
  }

  const std::string& h = state.kuhn_history;
  if (h.empty()) return {{0, "check"}, {1, "bet"}};
  if (h == "p") return {{0, "check"}, {1, "bet"}};
  if (h == "b" || h == "pb") return {{2, "fold"}, {3, "call"}};
  return {};
}

DebugGameState debug_step_state(const DebugGameState& state, int action_key,
                                uint64_t seed) {
  DebugGameState out = state;
  if (out.result >= 0) return out;
  std::vector<DebugAction> legal = debug_legal_actions(out);
  if (!has_action(legal, action_key))
    throw std::runtime_error("illegal debug action: " + std::to_string(action_key));
  out.rng = seed ? seed : out.rng;

  if (out.kind == DebugGameKind::Tictactoe) {
    out.ttt_board[action_key] = static_cast<int8_t>(out.current_player);
    int winner = ttt_winner(out.ttt_board);
    if (winner >= 0) {
      out.result = winner;
      out.payoff0 = winner == 0 ? 1.0 : -1.0;
    } else if (ttt_full(out.ttt_board)) {
      out.result = 2;
      out.payoff0 = 0.0;
    } else {
      out.current_player = 1 - out.current_player;
    }
    return out;
  }

  std::string& h = out.kuhn_history;
  if (h.empty()) {
    h = action_key == 0 ? "p" : "b";
    out.current_player = 1;
  } else if (h == "p") {
    if (action_key == 0) {
      h = "pp";
      settle_kuhn_showdown(out, 0.5);
    } else {
      h = "pb";
      out.current_player = 0;
    }
  } else if (h == "b") {
    if (action_key == 2) {
      h = "bf";
      settle_kuhn_fold(out, 0);
    } else {
      h = "bc";
      settle_kuhn_showdown(out, 1.0);
    }
  } else if (h == "pb") {
    if (action_key == 2) {
      h = "pbf";
      settle_kuhn_fold(out, 1);
    } else {
      h = "pbc";
      settle_kuhn_showdown(out, 1.0);
    }
  }
  return out;
}

DebugGameState debug_determinize(const DebugGameState& state, int perspective,
                                 uint64_t seed) {
  DebugGameState out = state;
  if (out.kind != DebugGameKind::Kuhn) return out;
  int p = (perspective == 0 || perspective == 1) ? perspective : out.current_player;
  int opp = 1 - p;
  int own = out.kuhn_cards[p];
  uint64_t rng = seed ? seed : 1;
  if (own < 0) {
    std::array<int8_t, 3> deck{{0, 1, 2}};
    for (int i = 2; i > 0; --i) {
      int j = bounded_rand(rng, i + 1);
      std::swap(deck[i], deck[j]);
    }
    out.kuhn_cards[0] = deck[0];
    out.kuhn_cards[1] = deck[1];
  } else {
    std::vector<int8_t> remaining;
    for (int8_t c = 0; c < 3; ++c) {
      if (c != own) remaining.push_back(c);
    }
    out.kuhn_cards[opp] = remaining[bounded_rand(rng, static_cast<int>(remaining.size()))];
  }
  out.rng = rng;
  return out;
}

std::vector<float> debug_observe(const DebugGameState& state, int perspective) {
  int p = (perspective == 0 || perspective == 1) ? perspective : state.current_player;
  std::vector<float> obs;
  if (state.kind == DebugGameKind::Tictactoe) {
    obs.reserve(10);
    obs.push_back(state.current_player == p ? 1.0f : -1.0f);
    for (int8_t v : state.ttt_board) {
      if (v < 0) {
        obs.push_back(0.0f);
      } else {
        obs.push_back(v == p ? 1.0f : -1.0f);
      }
    }
    return obs;
  }

  obs.assign(8, 0.0f);
  obs[0] = state.current_player == p ? 1.0f : -1.0f;
  int card = state.kuhn_cards[p];
  if (card >= 0 && card < 3) obs[1 + card] = 1.0f;
  if (state.kuhn_history.empty()) obs[4] = 1.0f;
  else if (state.kuhn_history == "p") obs[5] = 1.0f;
  else if (state.kuhn_history == "b") obs[6] = 1.0f;
  else if (state.kuhn_history == "pb") obs[7] = 1.0f;
  return obs;
}

std::string debug_public_state(const DebugGameState& state) {
  if (state.kind == DebugGameKind::Tictactoe) {
    std::string s = "ttt p" + std::to_string(state.current_player) + " ";
    for (int8_t v : state.ttt_board) s.push_back(v < 0 ? '.' : static_cast<char>('0' + v));
    return s;
  }
  return "kuhn p" + std::to_string(state.current_player) + " h=" + state.kuhn_history;
}

double debug_terminal_value(const DebugGameState& state, int player) {
  if (state.result < 0) return 0.0;
  if (state.result == 2) return 0.0;
  return player == 0 ? state.payoff0 : -state.payoff0;
}

DebugSearchResult debug_puct_search(const DebugGameState& root_state, int n_sims,
                                    uint64_t seed, const DebugEvaluator& evaluator) {
  DebugSearchResult out = make_empty_result(root_state, "puct", n_sims, 0);
  std::vector<PuctNode> tree;
  tree.reserve(static_cast<size_t>(std::max(n_sims, 1)) + 1);
  tree.emplace_back(root_state);
  PuctNode& root = tree[0];
  if (root.terminal) return out;
  int root_player = root.mover;
  double root_v = eval_and_expand_puct(root, evaluator, &out.stats);
  (void)root_v;

  uint64_t rng = seed ? seed : 1;
  for (int sim = 0; sim < n_sims; ++sim) {
    int node_idx = 0;
    std::vector<std::pair<int, int>> path;
    double v_root = 0.0;
    while (true) {
      PuctNode& node = tree[node_idx];
      int a_idx = select_puct_action(node, root_player);
      path.emplace_back(node_idx, a_idx);
      int child_idx = node.child[a_idx];
      if (child_idx < 0) {
        DebugGameState child_state =
            debug_step_state(node.state, node.actions[a_idx].key, lcg_next(rng));
        tree.emplace_back(std::move(child_state));
        int leaf_idx = static_cast<int>(tree.size()) - 1;
        tree[node_idx].child[a_idx] = leaf_idx;
        PuctNode& leaf = tree[leaf_idx];
        if (leaf.terminal) {
          ++out.stats.terminal_evals;
          v_root = debug_terminal_value(leaf.state, root_player);
        } else {
          double v = eval_and_expand_puct(leaf, evaluator, &out.stats);
          v_root = leaf.mover == root_player ? v : -v;
        }
        break;
      }
      node_idx = child_idx;
      if (tree[node_idx].terminal) {
        ++out.stats.terminal_evals;
        v_root = debug_terminal_value(tree[node_idx].state, root_player);
        break;
      }
    }
    for (auto [idx, a] : path) {
      PuctNode& nd = tree[idx];
      nd.childN[a] += 1.0;
      nd.childW[a] += v_root;
      nd.totalN += 1.0;
    }
  }

  fill_result_from_puct_root(out, tree[0]);
  return out;
}

DebugSearchResult debug_pimc_search(const DebugGameState& root_state, int n_sims,
                                    uint64_t seed, int dets,
                                    const DebugEvaluator& evaluator) {
  DebugSearchResult out = make_empty_result(root_state, "pimc", n_sims, dets);
  if (out.terminal) return out;

  std::vector<DebugGameState> worlds =
      sampled_worlds(root_state, dets, seed, root_state.current_player);
  out.stats.sampled_worlds = static_cast<int>(worlds.size());
  out.childN.assign(out.actions.size(), 0.0);
  out.childW.assign(out.actions.size(), 0.0);
  std::unordered_map<int, int> root_pos;
  for (int i = 0; i < static_cast<int>(out.actions.size()); ++i)
    root_pos[out.actions[i].key] = i;

  for (int i = 0; i < static_cast<int>(worlds.size()); ++i) {
    DebugSearchResult part = debug_puct_search(
        worlds[i], n_sims, seed + static_cast<uint64_t>(i + 1) * 7919ULL, evaluator);
    out.stats.eval_calls += part.stats.eval_calls;
    out.stats.eval_rows += part.stats.eval_rows;
    out.stats.terminal_evals += part.stats.terminal_evals;
    for (int j = 0; j < static_cast<int>(part.actions.size()); ++j) {
      auto it = root_pos.find(part.actions[j].key);
      if (it == root_pos.end()) {
        ++out.stats.invalid_worlds;
        continue;
      }
      int pos = it->second;
      out.childN[pos] += part.childN[j];
      out.childW[pos] += part.childW[j];
    }
  }
  double total = std::accumulate(out.childN.begin(), out.childN.end(), 0.0);
  out.value = total > 0.0
                  ? std::accumulate(out.childW.begin(), out.childW.end(), 0.0) / total
                  : 0.0;
  return out;
}

DebugSearchResult debug_maple_search(const DebugGameState& root_state, int n_sims,
                                     uint64_t seed, int dets,
                                     const DebugEvaluator& evaluator) {
  DebugSearchResult out = make_empty_result(root_state, "maple", n_sims, dets);
  if (out.terminal) return out;
  int root_player = root_state.current_player;
  std::vector<DebugGameState> root_worlds =
      sampled_worlds(root_state, dets, seed, root_player);
  out.stats.sampled_worlds = static_cast<int>(root_worlds.size());

  std::vector<MapleNode> tree;
  tree.emplace_back();
  expand_maple_node(tree[0], root_worlds, root_player, evaluator, &out.stats);

  uint64_t rng = seed ? seed : 1;
  for (int sim = 0; sim < n_sims; ++sim) {
    int node_idx = 0;
    std::vector<DebugGameState> worlds = root_worlds;
    std::vector<std::pair<int, int>> path;
    double v_root = 0.0;

    while (true) {
      MapleNode& node = tree[node_idx];
      if (node.terminal || node.actions.empty()) {
        v_root = 0.0;
        break;
      }
      int a_idx = select_maple_action(node, root_player);
      int action_key = node.actions[a_idx].key;
      path.emplace_back(node_idx, a_idx);

      std::vector<DebugGameState> next_worlds;
      next_worlds.reserve(worlds.size());
      for (const DebugGameState& w : worlds) {
        if (w.result >= 0 || !has_action(debug_legal_actions(w), action_key)) {
          ++out.stats.invalid_worlds;
          continue;
        }
        next_worlds.push_back(debug_step_state(w, action_key, lcg_next(rng)));
      }
      if (next_worlds.empty()) {
        v_root = 0.0;
        break;
      }

      int child_idx = node.child[a_idx];
      if (child_idx < 0) {
        tree.emplace_back();
        child_idx = static_cast<int>(tree.size()) - 1;
        tree[node_idx].child[a_idx] = child_idx;
        v_root = expand_maple_node(tree[child_idx], next_worlds, root_player,
                                   evaluator, &out.stats);
        break;
      }

      node_idx = child_idx;
      worlds = std::move(next_worlds);
    }

    for (auto [idx, a] : path) {
      MapleNode& nd = tree[idx];
      nd.childN[a] += 1.0;
      nd.childW[a] += v_root;
      nd.totalN += 1.0;
    }
  }

  const MapleNode& root = tree[0];
  out.terminal = root.terminal;
  out.root_player = root_player;
  out.actions = root.actions;
  out.childN = root.childN;
  out.childW = root.childW;
  double total = std::accumulate(out.childN.begin(), out.childN.end(), 0.0);
  out.value = total > 0.0
                  ? std::accumulate(out.childW.begin(), out.childW.end(), 0.0) / total
                  : 0.0;
  return out;
}

DebugSearchResult debug_search(const std::string& game, const DebugGameState& state,
                               const std::string& algo, int n_sims,
                               uint64_t seed, int dets,
                               const DebugEvaluator& evaluator) {
  DebugGameKind requested = debug_game_kind_from_name(game);
  if (requested != state.kind)
    throw std::runtime_error("debug_search game does not match state kind");
  int sims = std::max(n_sims, 0);
  if (ieq(algo, "puct")) return debug_puct_search(state, sims, seed, evaluator);
  if (ieq(algo, "pimc")) return debug_pimc_search(state, sims, seed, dets, evaluator);
  if (ieq(algo, "maple")) return debug_maple_search(state, sims, seed, dets, evaluator);
  if (ieq(algo, "ismcts") || ieq(algo, "mccfr") || ieq(algo, "cfr") ||
      ieq(algo, "rebel")) {
    throw std::runtime_error("debug algorithm is reserved but unsupported in V1: " +
                             algo);
  }
  throw std::runtime_error("unknown debug algorithm: " + algo);
}

}  // namespace ptcg::search
