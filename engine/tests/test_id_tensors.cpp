#include "ptcg/id_tensors.hpp"
#include "ptcg/rl.hpp"
#include "ptcg/state.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

using namespace ptcg;

ActionIdView ids_from_options(const RlOptionSet& opts) {
  return ActionIdView{opts.pending, opts.n, &opts.descriptors};
}

std::vector<int> mega_lucario_deck() {
  std::vector<int> deck;
  auto add = [&](int card, int count) {
    for (int i = 0; i < count; ++i) deck.push_back(card);
  };
  add(673, 2);
  add(674, 2);
  add(675, 2);
  add(676, 3);
  add(677, 3);
  add(678, 4);
  add(1102, 4);
  add(1123, 2);
  add(1141, 4);
  add(1142, 4);
  add(1152, 4);
  add(1159, 1);
  add(1182, 2);
  add(1192, 4);
  add(1227, 4);
  add(1252, 2);
  add(6, 13);
  return deck;
}

void require(bool cond, const char* msg) {
  if (!cond) {
    std::cerr << "test_id_tensors failed: " << msg << "\n";
    std::exit(1);
  }
}

template <typename T>
void require_equal(const std::vector<T>& a, const std::vector<T>& b,
                   const char* msg) {
  require(a.size() == b.size(), msg);
  require(std::equal(a.begin(), a.end(), b.begin()), msg);
}

struct BatchIds {
  int n = 0;
  std::vector<int32_t> in_play;
  std::vector<int32_t> zones;
  std::vector<int32_t> counts;
  std::vector<int32_t> status;
  std::vector<int32_t> global;
  std::vector<int32_t> meta;
  std::vector<int32_t> options;
  std::vector<int32_t> deck;
  std::vector<uint8_t> mask;
  std::vector<int32_t> player;
  std::vector<int32_t> result;

  explicit BatchIds(int count) : n(count) {
    in_play.resize(n * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH);
    zones.resize(n * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS);
    counts.resize(n * 2 * 5);
    status.resize(n * 2 * 5);
    global.resize(n * STATE_GLOBAL_WIDTH);
    meta.resize(n * ACTION_META_WIDTH);
    options.resize(n * ACTION_MAX_OPTIONS * ACTION_OPTION_WIDTH);
    deck.resize(n * STATE_ZONE_SLOTS);
    mask.resize(n * ACTION_MAX_OPTIONS);
    player.resize(n);
    result.resize(n);
  }
};

void test_shapes_and_mask() {
  const auto deck = mega_lucario_deck();
  GameState st = new_game(deck, deck, 1);
  RlOptionSet opts = rl_options(st);
  ObservationIds state = make_observation_ids(st);
  ActionIds action = make_action_ids(st, ids_from_options(opts));

  require(state.in_play.size() == 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
          "state in_play size");
  require(state.zones.size() == 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
          "state zones size");
  require(state.player_counts.size() == 10, "state counts size");
  require(state.player_status.size() == 10, "state status size");
  require(state.global.size() == STATE_GLOBAL_WIDTH, "state global size");
  require(action.meta.size() == ACTION_META_WIDTH, "action meta size");
  require(action.options.size() == ACTION_MAX_OPTIONS * ACTION_OPTION_WIDTH,
          "action options size");
  require(action.deck.size() == STATE_ZONE_SLOTS, "action deck size");
  require(action.mask.size() == ACTION_MAX_OPTIONS, "action mask size");

  std::vector<uint8_t> legal(ACTION_MAX_OPTIONS);
  int n_legal = rl_legal_mask(st, legal.data());
  require(static_cast<int>(std::accumulate(action.mask.begin(),
                                           action.mask.end(), 0)) == opts.n,
          "action mask sum matches option set");
  require(opts.n == n_legal, "option set count matches rl_legal_mask count");
  require_equal(action.mask, legal, "action mask matches rl_legal_mask");

  IdTensorSpec spec = id_tensor_spec();
  require(spec.state_in_play_shape == std::vector<int>({2, STATE_INPLAY_SLOTS,
                                                        STATE_INPLAY_WIDTH}),
          "ID spec observation shape");
  require(spec.action_options_shape ==
              std::vector<int>({ACTION_MAX_OPTIONS, ACTION_OPTION_WIDTH}),
          "ID spec action shape");
}

void test_vector_env_observe_matches_helpers() {
  const auto deck = mega_lucario_deck();
  VectorEnv env(deck, deck, 4, 123, 1);
  BatchIds batch(env.size());
  env.observe_ids(batch.in_play.data(), batch.zones.data(), batch.counts.data(),
                  batch.status.data(), batch.global.data(),
                  batch.meta.data(), batch.options.data(), batch.deck.data(),
                  batch.mask.data(), batch.player.data(), batch.result.data());

  for (int i = 0; i < env.size(); ++i) {
    const GameState& st = env.state_at(i);
    ObservationIds state = make_observation_ids(st);
    ActionIds action = make_action_ids(st, ids_from_options(env.options_at(i)));
    require(std::equal(state.in_play.begin(), state.in_play.end(),
                       batch.in_play.begin() +
                           i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH),
            "observe_ids in_play equals helper");
    require(std::equal(state.zones.begin(), state.zones.end(),
                       batch.zones.begin() +
                           i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS),
            "observe_ids zones equals helper");
    require(std::equal(state.player_counts.begin(), state.player_counts.end(),
                       batch.counts.begin() + i * 10),
            "observe_ids counts equals helper");
    require(std::equal(state.player_status.begin(), state.player_status.end(),
                       batch.status.begin() + i * 10),
            "observe_ids status equals helper");
    require(std::equal(state.global.begin(), state.global.end(),
                       batch.global.begin() + i * STATE_GLOBAL_WIDTH),
            "observe_ids global equals helper");
    require(std::equal(action.meta.begin(), action.meta.end(),
                       batch.meta.begin() + i * ACTION_META_WIDTH),
            "observe_ids meta equals helper");
    require(std::equal(action.options.begin(), action.options.end(),
                       batch.options.begin() +
                           i * ACTION_MAX_OPTIONS * ACTION_OPTION_WIDTH),
            "observe_ids options equals helper");
    require(std::equal(action.deck.begin(), action.deck.end(),
                       batch.deck.begin() + i * STATE_ZONE_SLOTS),
            "observe_ids deck equals helper");
    require(std::equal(action.mask.begin(), action.mask.end(),
                       batch.mask.begin() + i * ACTION_MAX_OPTIONS),
            "observe_ids mask equals helper");
    require(batch.player[i] == st.yourIndex, "observe_ids player");
    require(batch.result[i] == st.result, "observe_ids result");
  }
}

void test_deterministic_fixed_seed() {
  const auto deck = mega_lucario_deck();
  VectorEnv a(deck, deck, 3, 999, 1);
  VectorEnv b(deck, deck, 3, 999, 1);
  BatchIds ids_a(a.size());
  BatchIds ids_b(b.size());
  a.observe_ids(ids_a.in_play.data(), ids_a.zones.data(), ids_a.counts.data(),
                ids_a.status.data(), ids_a.global.data(), ids_a.meta.data(),
                ids_a.options.data(), ids_a.deck.data(), ids_a.mask.data(),
                ids_a.player.data(), ids_a.result.data());
  b.observe_ids(ids_b.in_play.data(), ids_b.zones.data(), ids_b.counts.data(),
                ids_b.status.data(), ids_b.global.data(), ids_b.meta.data(),
                ids_b.options.data(), ids_b.deck.data(), ids_b.mask.data(),
                ids_b.player.data(), ids_b.result.data());
  require_equal(ids_a.in_play, ids_b.in_play, "deterministic in_play");
  require_equal(ids_a.zones, ids_b.zones, "deterministic zones");
  require_equal(ids_a.counts, ids_b.counts, "deterministic counts");
  require_equal(ids_a.status, ids_b.status, "deterministic status");
  require_equal(ids_a.global, ids_b.global, "deterministic global");
  require_equal(ids_a.meta, ids_b.meta, "deterministic meta");
  require_equal(ids_a.options, ids_b.options, "deterministic options");
  require_equal(ids_a.deck, ids_b.deck, "deterministic deck");
  require_equal(ids_a.mask, ids_b.mask, "deterministic mask");
  require_equal(ids_a.player, ids_b.player, "deterministic player");
  require_equal(ids_a.result, ids_b.result, "deterministic result");
}

void test_batch_step_ids_legal_action() {
  const auto deck = mega_lucario_deck();
  PpoBatchEnv env(deck, deck, 2, 321, 2000, 0.0, -1, PPO_OPP_SELF,
                  PPO_REWARD_TERMINAL, 1);
  BatchIds before(env.size());
  env.observe_ids(before.in_play.data(), before.zones.data(),
                  before.counts.data(), before.status.data(),
                  before.global.data(), before.meta.data(),
                  before.options.data(), before.deck.data(),
                  before.mask.data(), before.player.data(),
                  before.result.data());

  std::vector<int> actions(env.size(), 0);
  for (int i = 0; i < env.size(); ++i) {
    const uint8_t* row = before.mask.data() + i * ACTION_MAX_OPTIONS;
    auto it = std::find(row, row + ACTION_MAX_OPTIONS, uint8_t{1});
    require(it != row + ACTION_MAX_OPTIONS, "initial legal action exists");
    actions[i] = static_cast<int>(it - row);
  }

  std::vector<float> reward(env.size());
  std::vector<uint8_t> done(env.size());
  std::vector<int32_t> result(env.size());
  std::vector<int32_t> episode_len(env.size());
  BatchIds after(env.size());
  env.step_ids(actions.data(), reward.data(), done.data(), result.data(),
               episode_len.data(), after.in_play.data(), after.zones.data(),
               after.counts.data(), after.status.data(), after.global.data(),
               after.meta.data(), after.options.data(), after.deck.data(),
               after.mask.data(), after.player.data());

  for (int i = 0; i < env.size(); ++i) {
    const int mask_sum = std::accumulate(after.mask.begin() + i * ACTION_MAX_OPTIONS,
                                         after.mask.begin() +
                                             (i + 1) * ACTION_MAX_OPTIONS,
                                         0);
    require(mask_sum > 0, "step_ids next legal mask is non-empty");
    require(episode_len[i] >= 0, "step_ids episode length written");
  }
}

}  // namespace

int main() {
  test_shapes_and_mask();
  test_vector_env_observe_matches_helpers();
  test_deterministic_fixed_seed();
  test_batch_step_ids_legal_action();
  return 0;
}
