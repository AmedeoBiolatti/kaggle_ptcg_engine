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

void test_public_event_history_records_without_recursive_initialization() {
  const auto deck = mega_lucario_deck();
  PpoBatchEnv env(deck, deck, 1, 123, 2000, 0.0, -1, PPO_OPP_SELF,
                  PPO_REWARD_TERMINAL, 1, false, true);
  BatchIds before(env.size());
  env.observe_ids(before.in_play.data(), before.zones.data(),
                  before.counts.data(), before.status.data(),
                  before.global.data(), before.meta.data(),
                  before.options.data(), before.deck.data(),
                  before.mask.data(), before.player.data(),
                  before.result.data());
  auto legal = std::find(before.mask.begin(), before.mask.end(), uint8_t{1});
  require(legal != before.mask.end(), "public-event fixture has a legal action");
  int action = static_cast<int>(legal - before.mask.begin());

  float reward = 0.0f;
  uint8_t done = 0;
  int32_t result = -1;
  int32_t episode_len = 0;
  BatchIds after(env.size());
  std::vector<int32_t> public_events(
      PPO_PUBLIC_EVENT_SLOTS * PPO_PUBLIC_EVENT_WIDTH);
  env.step_ids(&action, &reward, &done, &result, &episode_len,
               after.in_play.data(), after.zones.data(), after.counts.data(),
               after.status.data(), after.global.data(), after.meta.data(),
               after.options.data(), after.deck.data(), after.mask.data(),
               after.player.data(), nullptr, public_events.data());

  int event_count = 0;
  for (int slot = 0; slot < PPO_PUBLIC_EVENT_SLOTS; ++slot) {
    const int32_t* event =
        public_events.data() + slot * PPO_PUBLIC_EVENT_WIDTH;
    if (event[0] == 0) continue;
    ++event_count;
    for (int column = 6; column <= 9; ++column)
      require(event[column] >= 0,
              "missing public-event locations encode as zero sentinels");
  }
  require(event_count > 0, "one legal step records a public event");
}

void test_int16_overflow_is_reported() {
  const auto deck = mega_lucario_deck();
  GameState st = new_game(deck, deck, 7);
  st.turn = 40000;
  std::vector<int16_t> in_play(2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH);
  std::vector<int16_t> zones(2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS);
  std::vector<int16_t> counts(2 * 5);
  std::vector<int16_t> status(2 * 5);
  std::vector<int16_t> global(STATE_GLOBAL_WIDTH);
  require(!fill_observation_ids16(st, in_play.data(), zones.data(),
                                  counts.data(), status.data(), global.data()),
          "int16 observation overflow is reported");
}

void test_observation_only_exposes_current_turn_attack_lock() {
  const auto deck = mega_lucario_deck();
  GameState st = new_game(deck, deck, 7);
  st.yourIndex = 0;
  st.turn = 10;
  Player& player = st.players[0];
  player.activePresent = true;
  player.activeKnown = true;
  player.active.id = 100;
  player.active.lockId = 77;
  player.active.lockTurn = 11;

  ObservationIds inactive = make_observation_ids(st);
  require(inactive.in_play[13] == 0,
          "future or expired one-turn attack lock is hidden");

  player.active.lockTurn = st.turn;
  ObservationIds active = make_observation_ids(st);
  require(active.in_play[13] == 77,
          "current-turn one-turn attack lock is exposed");
}

const int32_t* observation_inplay_row(const ObservationIds& obs, int pov,
                                      int slot) {
  return obs.in_play.data() +
         (pov * STATE_INPLAY_SLOTS + slot) * STATE_INPLAY_WIDTH;
}

int observation_effect_phase(const int32_t* row, int key) {
  if (key < 7)
    return (row[STATE_INPLAY_EFFECT_PHASES_0_COLUMN] >> (2 * key)) & 3;
  if (key < 14)
    return (row[STATE_INPLAY_EFFECT_PHASES_1_COLUMN] >> (2 * (key - 7))) & 3;
  return (row[STATE_INPLAY_EFFECT_META_COLUMN] >> (2 * (key - 14))) & 3;
}

void test_transient_effect_phases_values_and_stale_suppression() {
  GameState st;
  st.turn = 20;
  st.yourIndex = 0;
  Player& player = st.players[0];
  player.activePresent = true;
  player.activeKnown = true;
  player.active.id = 100;
  player.active.hp = 150;
  player.active.maxHp = 200;
  InPlay& pk = player.active;

  pk.dmgReduceTurn = 19;  // stale: both phase and residual value must vanish.
  pk.dmgReduce = 90;
  pk.attackCostModTurn = 20;
  pk.attackCostMod = 2;
  pk.retreatCostModTurn = 21;
  pk.retreatCostMod = 3;
  pk.delayedDamageTurn = 22;
  pk.delayedDamageCounters = 7;

  ObservationIds obs = make_observation_ids(st);
  const int32_t* row = observation_inplay_row(obs, 0, 0);
  require(observation_effect_phase(row, STATE_EFFECT_DAMAGE_REDUCTION) == 0,
          "stale transient phase is absent");
  require(observation_effect_phase(row, STATE_EFFECT_ATTACK_COST_MOD) == 1,
          "current transient phase is encoded");
  require(observation_effect_phase(row, STATE_EFFECT_RETREAT_COST_MOD) == 2,
          "next-turn transient phase is encoded");
  require(observation_effect_phase(row, STATE_EFFECT_DELAYED_DAMAGE) == 3,
          "two-turn transient phase is encoded");
  require((row[STATE_INPLAY_EFFECT_VALUES_0_COLUMN] & 63) == 0,
          "stale transient value is suppressed");
  require(((row[STATE_INPLAY_EFFECT_VALUES_0_COLUMN] >> 11) & 3) == 2 &&
              ((row[STATE_INPLAY_EFFECT_VALUES_0_COLUMN] >> 13) & 3) == 3,
          "simultaneous cost modifiers retain independent values");
  require(((row[STATE_INPLAY_EFFECT_VALUES_2_COLUMN] >> 9) & 63) == 7,
          "delayed damage counters are serialized");

  pk.delayedDamageTurn = 25;
  obs = make_observation_ids(st);
  row = observation_inplay_row(obs, 0, 0);
  require(observation_effect_phase(row, STATE_EFFECT_DELAYED_DAMAGE) == 3,
          "far-future phases saturate to the explicit later bucket");
}

void test_transient_effects_pack_simultaneously_and_fit_int16() {
  GameState st;
  st.turn = 30;
  st.yourIndex = 0;
  Player& player = st.players[0];
  player.activePresent = true;
  player.activeKnown = true;
  player.poisoned = true;
  player.poisonDamageCounters = 3;
  InPlay& pk = player.active;
  pk.id = 200;
  pk.hp = 120;
  pk.maxHp = 250;
  pk.preventDmgTurn = 30;
  pk.preventDmgCond = 6;
  pk.preventDmgValue = 120;
  pk.preventEffectsTurn = 31;
  pk.preventEffectsCond = 5;
  pk.preventEffectsValue = 80;
  pk.takeMoreDamageTurn = 30;
  pk.takeMoreDamage = 40;
  pk.nextAttackBonusTurn = 31;
  pk.nextAttackBonusId = 777;
  pk.nextAttackBonus = 50;
  pk.nextAttackSetBase = 130;
  pk.damagedByAttackCountersTurn = 30;
  pk.damagedByAttackCounters = 3;
  pk.damagedByAttackStatus = 4;
  pk.damagedByAttackEqualCountersTurn = 31;
  pk.energyAttachCountersTurn = 32;
  pk.energyAttachCounters = 4;
  pk.energyAttachCountersFromHandOnly = 1;
  pk.attackDmgReduceTurn = 30;
  pk.attackDmgReduce = 30;
  pk.attackBonusTurn = 31;
  pk.attackBonus = 60;
  pk.delayedKoTurn = 31;
  pk.delayedKoPromoteBeforePrize = true;
  pk.damagedByAttackTurn = 29;
  pk.damagedByAttackSide = 1;
  pk.damagedByAttackAmount = 90;
  pk.damagedByAttackBeforeHp = 210;

  ObservationIds obs = make_observation_ids(st);
  const int32_t* row = observation_inplay_row(obs, 0, 0);
  require(observation_effect_phase(row, STATE_EFFECT_PREVENT_DAMAGE) == 1 &&
              observation_effect_phase(row, STATE_EFFECT_PREVENT_EFFECTS) == 2,
          "simultaneous prevention effects keep separate phases");
  require(observation_effect_phase(row, STATE_EFFECT_REACTIVE_DAMAGE) == 1 &&
              observation_effect_phase(
                  row, STATE_EFFECT_REACTIVE_EQUAL_DAMAGE) == 2 &&
              observation_effect_phase(
                  row, STATE_EFFECT_ENERGY_ATTACH_DAMAGE) == 3,
          "simultaneous reactive effects keep separate phases");
  require(observation_effect_phase(
              row, STATE_EFFECT_ATTACK_DAMAGE_REDUCTION) == 1 &&
              observation_effect_phase(row, STATE_EFFECT_ATTACK_BONUS) == 2,
          "attack increase and reduction do not collapse");
  require(((row[STATE_INPLAY_EFFECT_VALUES_1_COLUMN] >> 3) & 63) == 12 &&
              ((row[STATE_INPLAY_EFFECT_VALUES_1_COLUMN] >> 9) & 63) == 4,
          "prevention threshold and extra damage are serialized");
  require((row[STATE_INPLAY_EFFECT_VALUES_4_COLUMN] & 2047) == 777 &&
              ((row[STATE_INPLAY_EFFECT_VALUES_4_COLUMN] >> 11) & 15) == 5,
          "named next-attack modifier keeps attack identity and amount");
  require((row[STATE_INPLAY_EFFECT_META_COLUMN] & (1 << 4)) != 0 &&
              (row[STATE_INPLAY_EFFECT_META_COLUMN] & (1 << 5)) != 0 &&
              ((row[STATE_INPLAY_EFFECT_META_COLUMN] >> 6) & 7) == 5 &&
              ((row[STATE_INPLAY_EFFECT_META_COLUMN] >> 9) & 3) == 2 &&
              (row[STATE_INPLAY_EFFECT_META_COLUMN] & (1 << 11)) != 0 &&
              (row[STATE_INPLAY_EFFECT_META_COLUMN] & (1 << 12)) != 0,
          "effect qualifiers, damage history, source, and severe poison survive");
  require((row[STATE_INPLAY_EFFECT_VALUES_3_COLUMN] & 63) == 9 &&
              ((row[STATE_INPLAY_EFFECT_VALUES_3_COLUMN] >> 6) & 63) == 21 &&
              ((row[STATE_INPLAY_EFFECT_VALUES_3_COLUMN] >> 12) & 7) == 3,
          "damage history and fixed reactive counters are serialized");

  std::vector<int16_t> in_play(2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH);
  std::vector<int16_t> zones(2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS);
  std::vector<int16_t> counts(2 * 5);
  std::vector<int16_t> status(2 * 5);
  std::vector<int16_t> global(STATE_GLOBAL_WIDTH);
  require(fill_observation_ids16(st, in_play.data(), zones.data(),
                                 counts.data(), status.data(), global.data()),
          "representative simultaneous effects fit losslessly in int16");
  for (int column : {STATE_INPLAY_EFFECT_PHASES_0_COLUMN,
                     STATE_INPLAY_EFFECT_PHASES_1_COLUMN,
                     STATE_INPLAY_EFFECT_META_COLUMN,
                     STATE_INPLAY_EFFECT_VALUES_0_COLUMN,
                     STATE_INPLAY_EFFECT_VALUES_1_COLUMN,
                     STATE_INPLAY_EFFECT_VALUES_2_COLUMN,
                     STATE_INPLAY_EFFECT_VALUES_3_COLUMN,
                     STATE_INPLAY_EFFECT_VALUES_4_COLUMN,
                     STATE_INPLAY_EFFECT_VALUES_5_COLUMN})
    require(in_play[column] == row[column],
            "int16 and int32 transient lanes are identical");
}

void test_global_transient_effects_are_pov_normalized() {
  GameState st;
  st.turn = 40;
  st.yourIndex = 1;
  for (int side = 0; side < 2; ++side) {
    st.players[side].activePresent = true;
    st.players[side].activeKnown = true;
    st.players[side].active.id = 300 + side;
  }
  st.players[1].active.dmgReduceTurn = 40;
  st.players[1].active.dmgReduce = 20;
  st.players[0].active.dmgReduceTurn = 41;
  st.players[0].active.dmgReduce = 30;
  st.noItemTurn[1] = 40;
  st.noSupporterTurn[1] = 41;
  st.noEvolveTurn[1] = 42;
  st.noStadiumTurn[1] = 39;  // stale
  st.noAttackEnergyLeTurn[1] = 41;
  st.noAttackEnergyLeThreshold[1] = 2;
  st.noItemTurn[0] = 41;
  st.noSupporterTurn[0] = 40;
  st.noAttackEnergyLeTurn[0] = 42;
  st.noAttackEnergyLeThreshold[0] = 3;
  st.discardHandEndTurn[1] = 40;
  st.discardHandEndThreshold[1] = 5;
  st.teamReduceTurn[1] = 41;
  st.teamReduceAmount[1] = 30;
  st.teamReduceType[1] = 6;
  st.activeExDamageBuffTurn[1] = 40;
  st.activeExDamageBuffAmount[1] = 40;
  st.prizeBonusTurn[1] = 41;
  st.prizeBonusAmount[1] = 3;
  st.prizeBonusKind[1] = 1;
  st.teamReduceTurn[0] = 40;
  st.teamReduceAmount[0] = 60;
  st.teamReduceType[0] = 3;
  st.activeExDamageBuffTurn[0] = 41;
  st.activeExDamageBuffAmount[0] = 30;
  st.prizeBonusTurn[0] = 40;
  st.prizeBonusAmount[0] = 1;
  st.prizeBonusKind[0] = 0;
  st.lunarUsedThisTurn = true;
  st.canariPlayed = true;
  st.tarragonPlayed = true;
  st.fightingBuff = 30;
  st.abilityGroupUsedTurn[1][3] = 40;

  ObservationIds side_one = make_observation_ids(st);
  require(observation_effect_phase(observation_inplay_row(side_one, 0, 0),
                                   STATE_EFFECT_DAMAGE_REDUCTION) == 1 &&
              observation_effect_phase(observation_inplay_row(side_one, 1, 0),
                                       STATE_EFFECT_DAMAGE_REDUCTION) == 2,
          "in-play transient rows are ordered by acting-player POV");
  const int self_restrictions =
      side_one.global[STATE_GLOBAL_RESTRICTIONS_SELF_COLUMN];
  const int opp_restrictions =
      side_one.global[STATE_GLOBAL_RESTRICTIONS_OPP_COLUMN];
  require((self_restrictions & 3) == 1 &&
              ((self_restrictions >> 2) & 3) == 2 &&
              ((self_restrictions >> 4) & 3) == 3 &&
              ((self_restrictions >> 6) & 3) == 0 &&
              ((self_restrictions >> 8) & 3) == 2 &&
              ((self_restrictions >> 10) & 7) == 3 &&
              (self_restrictions & (1 << 13)) != 0,
          "self restriction phases and low-energy threshold are structured");
  require((opp_restrictions & 3) == 2 &&
              ((opp_restrictions >> 2) & 3) == 1 &&
              ((opp_restrictions >> 8) & 3) == 3 &&
              ((opp_restrictions >> 10) & 7) == 4 &&
              (opp_restrictions & (1 << 13)) == 0,
          "opponent restrictions retain independent phases and threshold");
  require((self_restrictions & (1 << 14)) != 0 &&
              (opp_restrictions & (1 << 14)) != 0,
          "turn-scoped Lunar and Canari flags are represented");

  const int self_effects = side_one.global[STATE_GLOBAL_EFFECTS_SELF_COLUMN];
  require((self_effects & 3) == 1 && ((self_effects >> 2) & 3) == 2 &&
              ((self_effects >> 4) & 3) == 1 &&
              ((self_effects >> 6) & 3) == 2 &&
              ((self_effects >> 8) & 7) == 5 &&
              ((self_effects >> 11) & 15) == 7,
          "simultaneous self global effects keep phases and qualifiers");
  const int global_values =
      side_one.global[STATE_GLOBAL_EFFECT_VALUES_COLUMN];
  require((global_values & 1) == 0 &&
              ((global_values >> 1) & 1) == 1 &&
              ((global_values >> 2) & 3) == 1 &&
              ((global_values >> 4) & 3) == 0 &&
              ((global_values >> 6) & 3) == 3 &&
              ((global_values >> 8) & 3) == 1 &&
              ((global_values >> 10) & 1) == 1 &&
              ((global_values >> 11) & 1) == 0 &&
              ((global_values >> 12) & 1) == 1 &&
              (global_values & (1 << 13)) != 0 &&
              (global_values & (1 << 14)) == 0,
          "global effect amounts, prize kind, fighting buff, and Tarragon survive");

  st.yourIndex = 0;
  ObservationIds side_zero = make_observation_ids(st);
  require(observation_effect_phase(observation_inplay_row(side_zero, 0, 0),
                                   STATE_EFFECT_DAMAGE_REDUCTION) == 2 &&
              observation_effect_phase(observation_inplay_row(side_zero, 1, 0),
                                       STATE_EFFECT_DAMAGE_REDUCTION) == 1,
          "in-play transient rows swap with acting-player POV");
  require((side_zero.global[STATE_GLOBAL_RESTRICTIONS_SELF_COLUMN] & 0x3fff) ==
              (opp_restrictions & 0x3fff) &&
              (side_zero.global[STATE_GLOBAL_RESTRICTIONS_OPP_COLUMN] &
               0x3fff) == (self_restrictions & 0x3fff),
          "global restriction lanes swap with acting-player POV");
  require(side_zero.global[STATE_GLOBAL_EFFECTS_SELF_COLUMN] ==
              side_one.global[STATE_GLOBAL_EFFECTS_OPP_COLUMN] &&
              side_zero.global[STATE_GLOBAL_EFFECTS_OPP_COLUMN] == self_effects,
          "global secondary-effect lanes swap with acting-player POV");
}

void test_global_qualifier_clipping_sets_shared_saturation() {
  enum class Qualifier {
    LOW_ENERGY_THRESHOLD,
    DISCARD_THRESHOLD,
    TEAM_TYPE,
    PRIZE_KIND,
  };
  struct Case {
    Qualifier qualifier;
    const char* name;
    int packed_column;
    int shift;
    int mask;
    int expected;
  };
  const Case cases[] = {
      {Qualifier::LOW_ENERGY_THRESHOLD, "low-energy threshold",
       STATE_GLOBAL_RESTRICTIONS_SELF_COLUMN, 10, 7, 7},
      {Qualifier::DISCARD_THRESHOLD, "discard threshold",
       STATE_GLOBAL_EFFECTS_SELF_COLUMN, 8, 7, 7},
      {Qualifier::TEAM_TYPE, "team-reduction type",
       STATE_GLOBAL_EFFECTS_SELF_COLUMN, 11, 15, 15},
      {Qualifier::PRIZE_KIND, "prize kind",
       STATE_GLOBAL_EFFECT_VALUES_COLUMN, 10, 1, 1},
  };

  for (const Case& test : cases) {
    GameState st;
    st.turn = 70;
    st.yourIndex = 0;
    switch (test.qualifier) {
      case Qualifier::LOW_ENERGY_THRESHOLD:
        st.noAttackEnergyLeTurn[0] = st.turn;
        st.noAttackEnergyLeThreshold[0] = 99;
        break;
      case Qualifier::DISCARD_THRESHOLD:
        st.discardHandEndTurn[0] = st.turn;
        st.discardHandEndThreshold[0] = 99;
        break;
      case Qualifier::TEAM_TYPE:
        st.teamReduceTurn[0] = st.turn;
        st.teamReduceAmount[0] = 30;
        st.teamReduceType[0] = 99;
        break;
      case Qualifier::PRIZE_KIND:
        st.prizeBonusTurn[0] = st.turn;
        st.prizeBonusAmount[0] = 1;
        st.prizeBonusKind[0] = 99;
        break;
    }

    const ObservationIds obs = make_observation_ids(st);
    const int packed =
        (obs.global[test.packed_column] >> test.shift) & test.mask;
    const std::string clamp_message =
        std::string(test.name) + " clamps to its packed domain";
    require(packed == test.expected, clamp_message.c_str());
    const std::string saturation_message =
        std::string(test.name) + " reports shared saturation";
    require((obs.global[STATE_GLOBAL_EFFECT_VALUES_COLUMN] & (1 << 14)) != 0,
            saturation_message.c_str());
  }
}

void test_action_positions_are_semantic_and_raw_references_are_preserved() {
  GameState st;
  st.yourIndex = 0;
  st.players[0].hand = {202};
  st.players[0].handCount = 1;
  // Internal deck storage is bottom-first: 101 is the bottom, 103 the top.
  st.players[0].deck = {101, 102, 103};
  st.players[0].deckCount = 3;
  st.players[1].hand = {303};
  st.players[1].handCount = 1;
  std::vector<Descriptor> descriptors = {
      {Atom::S("PLAY"), Atom::I(202)},
      {Atom::S("CARD"), Atom::S("DECK"), Atom::I(0), Atom::I(101)},
      {Atom::S("CARD"), Atom::S("DECK"), Atom::I(2), Atom::I(103)},
      {Atom::S("CARD"), Atom::S("HAND"), Atom::I(0), Atom::I(303)},
  };

  ActionIds action = make_action_ids(
      st, ActionIdView{false, static_cast<int>(descriptors.size()), &descriptors});
  const int32_t* play = action.options.data();
  const int32_t* bottom = play + ACTION_OPTION_WIDTH;
  const int32_t* top = bottom + ACTION_OPTION_WIDTH;
  const int32_t* opponent_hand = top + ACTION_OPTION_WIDTH;

  require(play[3] == 2, "PLAY is represented as hand-sourced");
  require(bottom[4] == 2 && top[4] == 0,
          "deck option position is canonical distance from top");
  require(bottom[ACTION_RAW_REF_LOW_COLUMN] == 0 &&
              top[ACTION_RAW_REF_LOW_COLUMN] == 2,
          "canonical positions do not alter raw engine references");
  require(action.deck[0] == 101 && action.deck[2] == 103,
          "action deck reconstruction remains in raw engine order");
  require(opponent_hand[3] == 2 && opponent_hand[5] == 1,
          "opponent-hand choices keep hand semantics and POV ownership");
}

void test_attack_options_include_resolved_definition_keys() {
  GameState st;
  st.yourIndex = 0;
  st.players[0].activePresent = true;
  st.players[0].activeKnown = true;
  st.players[0].active.id = 481;  // Serperior ex, attack 680 in slot 0.

  std::vector<Descriptor> own_attack = {
      {Atom::S("ATTACK"), Atom::I(680)},
      {Atom::S("SKILL"), Atom::I(481), Atom::I(77)},
      {Atom::S("ATTACK"), Atom::I(999999)},
  };
  ActionIds action = make_action_ids(
      st, ActionIdView{false, static_cast<int>(own_attack.size()), &own_attack});
  const int32_t* own = action.options.data();
  const int32_t* skill = own + ACTION_OPTION_WIDTH;
  const int32_t* unresolved = skill + ACTION_OPTION_WIDTH;
  require(ACTION_ATTACK_DEFINITION_COLUMN == 12,
          "resolved attack key reuses contextual column 12");
  require(own[ACTION_ATTACK_DEFINITION_COLUMN] == 481 * 4 + 1,
          "own printed attack resolves to stable card-and-slot key");
  require(skill[ACTION_ATTACK_DEFINITION_COLUMN] == 77,
          "SKILL rows retain the legacy serial meaning of column 12");
  require(unresolved[ACTION_ATTACK_DEFINITION_COLUMN] == 0,
          "unknown attack sources use the reserved zero key");

  // Main-action copy attacks resolve to the copied card, not the card that
  // grants the copy behavior.
  st.players[0].active.id = 615;  // Zoroark: Foul Play.
  st.players[1].activePresent = true;
  st.players[1].activeKnown = true;
  st.players[1].active.id = 481;
  std::vector<Descriptor> copied = {{Atom::S("ATTACK"), Atom::I(680)}};
  action = make_action_ids(
      st, ActionIdView{false, static_cast<int>(copied.size()), &copied});
  require(action.options[ACTION_ATTACK_DEFINITION_COLUMN] == 481 * 4 + 1,
          "copied main attack resolves to opponent printed occurrence");

  // VM copied-attack descriptors can state the source card explicitly.
  std::vector<Descriptor> explicit_source = {{
      Atom::S("ATTACK"), Atom::I(1556), Atom::I(1556000), Atom::I(1180)}};
  action = make_action_ids(
      st, ActionIdView{false, static_cast<int>(explicit_source.size()),
                       &explicit_source});
  require(action.options[ACTION_ATTACK_DEFINITION_COLUMN] == 1180 * 4 + 1,
          "explicit copied-attack source resolves without board inference");
}

void test_full_deck_inspection_infers_exact_owner_prizes_without_pov_leak() {
  const auto deck = mega_lucario_deck();
  GameState st = new_game(deck, deck, 17);
  const int owner = st.yourIndex;
  Player& player = st.players[owner];

  require(!player.prizesKnown, "initial prizes are unknown");
  mark_full_deck_inspected(player);
  require(player.deckKnown && player.ownDeckInspected,
          "full inspection records exact owner deck knowledge");
  require(!player.prizesKnown && player.ownPrizesInferred,
          "owner inference stays isolated from generic Prize knowledge");

  ObservationIds own = make_observation_ids(st);
  std::vector<int> encoded_prizes;
  const int own_prize_base = 3 * STATE_ZONE_SLOTS;
  for (int i = 0; i < player.prizeCount; ++i)
    encoded_prizes.push_back(own.zones[own_prize_base + i]);
  std::vector<int> actual_prizes(player.prizes.begin(), player.prizes.end());
  std::sort(encoded_prizes.begin(), encoded_prizes.end());
  std::sort(actual_prizes.begin(), actual_prizes.end());
  require(encoded_prizes == actual_prizes,
          "owner Prize observation exactly matches hidden multiset");
  require(own.global[25] == 1, "owner fully-known Prize flag is set");

  st.yourIndex = 1 - owner;
  ObservationIds opponent = make_observation_ids(st);
  const int opponent_prize_base =
      (STATE_ZONE_COUNT + 3) * STATE_ZONE_SLOTS;
  for (int i = 0; i < player.prizeCount; ++i)
    require(opponent.zones[opponent_prize_base + i] == -1,
            "owner-only inferred prizes never leak to opponent POV");
  require(opponent.global[26] == 0,
          "opponent fully-known Prize flag does not leak");
}

void test_full_deck_inspection_fails_closed_without_exact_state() {
  Player replay_player;
  replay_player.deckCount = 20;
  replay_player.prizeCount = 6;
  mark_full_deck_inspected(replay_player);
  require(!replay_player.deckKnown && !replay_player.ownDeckInspected,
          "count-only deck cannot become exactly known");
  require(!replay_player.prizesKnown && !replay_player.ownPrizesInferred,
          "count-only prizes are never guessed");
}

void test_full_search_effect_uses_exact_inference_path() {
  auto deck = mega_lucario_deck();
  for (int& card : deck)
    if (card == 1102) card = 1086;  // Buddy-Buddy Poffin: full deck search.
  bool exercised = false;
  for (uint64_t seed = 1; seed <= 256 && !exercised; ++seed) {
    GameState st = new_game(deck, deck, seed);
    RlOptionSet opts = rl_options(st);
    ActionIds actions = make_action_ids(st, ids_from_options(opts));
    int search_action = -1;
    for (int action = 0; action < opts.n; ++action) {
      const int* row =
          actions.options.data() + action * ACTION_OPTION_WIDTH;
      if (actions.mask[action] && row[8] == 1086) {
        search_action = action;
        break;
      }
    }
    if (search_action < 0) continue;

    const int owner = st.yourIndex;
    uint64_t rng = seed + 1000;
    RlOptionSet next;
    rl_step_cached(st, opts, search_action, rng, &next);
    const Player& player = st.players[owner];
    const int known_deck_cards =
        player.deckKnown
            ? player.deckCount
            : static_cast<int>(player.deckKnownCards.size()) +
                  static_cast<int>(
                      std::count(player.deckKnownMask.begin(),
                                 player.deckKnownMask.end(), true));
    require(player.ownDeckInspected && known_deck_cards == player.deckCount,
            "full search records owner deck knowledge");
    require(player.ownPrizesInferred && !player.prizesKnown,
            "full search infers exact prizes without changing generic knowledge");
    const int saved_pov = st.yourIndex;
    st.yourIndex = owner;
    ObservationIds inferred_obs = make_observation_ids(st);
    st.yourIndex = saved_pov;
    std::vector<int> inferred;
    const int own_prize_base = 3 * STATE_ZONE_SLOTS;
    for (int i = 0; i < player.prizeCount; ++i)
      inferred.push_back(inferred_obs.zones[own_prize_base + i]);
    std::sort(inferred.begin(), inferred.end());
    std::vector<int> actual(player.prizes.begin(), player.prizes.end());
    std::sort(actual.begin(), actual.end());
    require(inferred == actual, "full search inferred Prize multiset is exact");
    exercised = true;
  }
  require(exercised, "deterministic seeds exercise a full deck search");
}

void test_known_hidden_zone_ids_remain_exact_across_random_trajectory() {
  const auto deck = mega_lucario_deck();
  GameState st = new_game(deck, deck, 91);
  mark_full_deck_inspected(st.players[0]);
  mark_full_deck_inspected(st.players[1]);
  uint64_t rng = 0x123456789abcdefULL;

  auto require_subset = [](std::vector<int> known, std::vector<int> actual,
                           const char* message) {
    std::sort(known.begin(), known.end());
    std::sort(actual.begin(), actual.end());
    for (size_t i = 0; i < known.size();) {
      const int card = known[i];
      const auto known_end =
          std::upper_bound(known.begin() + i, known.end(), card);
      const auto actual_range =
          std::equal_range(actual.begin(), actual.end(), card);
      require(known_end - (known.begin() + i) <=
                  actual_range.second - actual_range.first,
              message);
      i = static_cast<size_t>(known_end - known.begin());
    }
  };

  for (int step = 0; step < 400 && st.result < 0; ++step) {
    const Player& owner = st.players[st.yourIndex];
    ObservationIds obs = make_observation_ids(st);
    std::vector<int> known_deck;
    std::vector<int> known_prizes;
    for (int slot = 0; slot < owner.deckCount; ++slot) {
      const int card = obs.zones[STATE_ZONE_SLOTS + slot];
      if (card > 0) known_deck.push_back(card);
    }
    for (int slot = 0; slot < owner.prizeCount; ++slot) {
      const int card = obs.zones[3 * STATE_ZONE_SLOTS + slot];
      if (card > 0) known_prizes.push_back(card);
    }
    std::vector<int> actual_deck(owner.deck.begin(), owner.deck.end());
    std::vector<int> actual_prizes(owner.prizes.begin(), owner.prizes.end());
    require_subset(known_deck, actual_deck,
                   "known deck IDs are always an exact multiset subset");
    require_subset(known_prizes, actual_prizes,
                   "known Prize IDs are always an exact multiset subset");
    if (obs.global[23]) {
      std::sort(known_deck.begin(), known_deck.end());
      std::sort(actual_deck.begin(), actual_deck.end());
      require(known_deck == actual_deck,
              "fully-known deck multiset exactly matches hidden state");
    }
    if (obs.global[25]) {
      std::sort(known_prizes.begin(), known_prizes.end());
      std::sort(actual_prizes.begin(), actual_prizes.end());
      require(known_prizes == actual_prizes,
              "fully-known Prize multiset exactly matches hidden state");
    }

    RlOptionSet opts = rl_options(st);
    require(opts.n > 0, "random trajectory has a legal option");
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    const int action = static_cast<int>(rng % static_cast<uint64_t>(opts.n));
    rl_step_cached(st, opts, action, rng);
  }
}

}  // namespace

int main() {
  test_shapes_and_mask();
  test_vector_env_observe_matches_helpers();
  test_deterministic_fixed_seed();
  test_batch_step_ids_legal_action();
  test_public_event_history_records_without_recursive_initialization();
  test_int16_overflow_is_reported();
  test_observation_only_exposes_current_turn_attack_lock();
  test_transient_effect_phases_values_and_stale_suppression();
  test_transient_effects_pack_simultaneously_and_fit_int16();
  test_global_transient_effects_are_pov_normalized();
  test_global_qualifier_clipping_sets_shared_saturation();
  test_action_positions_are_semantic_and_raw_references_are_preserved();
  test_attack_options_include_resolved_definition_keys();
  test_full_deck_inspection_infers_exact_owner_prizes_without_pov_leak();
  test_known_hidden_zone_ids_remain_exact_across_random_trajectory();
  test_full_deck_inspection_fails_closed_without_exact_state();
  test_full_search_effect_uses_exact_inference_path();
  return 0;
}
