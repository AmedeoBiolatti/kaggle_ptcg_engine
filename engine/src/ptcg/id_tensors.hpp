#pragma once

#include <cstdint>
#include <vector>

#include "ptcg/state.hpp"

namespace ptcg {

constexpr int STATE_INPLAY_SLOTS = 6;  // active + 5 bench
constexpr int STATE_INPLAY_WIDTH = 64;
constexpr int STATE_ZONE_COUNT = 4;  // hand, deck, discard, prizes
constexpr int STATE_ZONE_SLOTS = 64;
constexpr int STATE_PRIZE_SLOTS = 6;
constexpr int STATE_MAX_ATTACHED_ENERGY = 16;
constexpr int STATE_MAX_TOOLS = 4;
constexpr int STATE_MAX_PRE_EVOS = 4;
constexpr int STATE_GLOBAL_WIDTH = 32;
// Fixed-shape transient-effect lanes.  Every packed word stays in the
// non-negative signed-int16 range so observe_ids() and observe_ids16() retain
// identical values.  Turn phases are 0=absent/stale, 1=current, 2=next turn,
// 3=two-or-more turns ahead.
constexpr int STATE_INPLAY_EFFECT_PHASES_0_COLUMN = 15;
constexpr int STATE_INPLAY_EFFECT_PHASES_1_COLUMN = 56;
constexpr int STATE_INPLAY_EFFECT_META_COLUMN = 57;
constexpr int STATE_INPLAY_EFFECT_VALUES_0_COLUMN = 58;
constexpr int STATE_INPLAY_EFFECT_VALUES_1_COLUMN = 59;
constexpr int STATE_INPLAY_EFFECT_VALUES_2_COLUMN = 60;
constexpr int STATE_INPLAY_EFFECT_VALUES_3_COLUMN = 61;
constexpr int STATE_INPLAY_EFFECT_VALUES_4_COLUMN = 62;
constexpr int STATE_INPLAY_EFFECT_VALUES_5_COLUMN = 63;
constexpr int STATE_EFFECT_PHASE_BITS = 2;
// phases_0: seven 2-bit phases for keys 0..6.
// phases_1: seven 2-bit phases for keys 7..13.
// meta: bits 0..1/2..3 are phases for keys 14/15; bit 4 delayed-KO
// promote order; bit 5 attach trigger is hand-only; bits 6..8 reactive
// status+1; bits 9..10 damage-history age; bit 11 damage source is the
// Pokemon owner's opponent; bit 12 severe poison; bit 13 next-attack set-base
// is present; bit 14 reports a saturated/non-decimal packed value.
// values_0: dmgReduce/10[6], attackDmgReduce/10[5], attackCost[2],
// retreatCost[2].
// values_1: preventDmgCond[3], preventDmgValue/10[6], takeMoreDamage/10[6].
// values_2: preventEffectsCond[3], preventEffectsValue/10[6],
// delayedDamageCounters[6].
// values_3: lastDamage/10[6], HP-before/10[6], reactive counters[3].
// values_4: named attack ID[11], named additive bonus/10[4].
// values_5: named set-base/10+1[6], direct attack bonus/10[5],
// energy-attach reactive counters[4].

enum StateInPlayEffectKey {
  STATE_EFFECT_DAMAGE_REDUCTION = 0,
  STATE_EFFECT_ATTACK_COST_MOD = 1,
  STATE_EFFECT_RETREAT_COST_MOD = 2,
  STATE_EFFECT_DELAYED_DAMAGE = 3,
  STATE_EFFECT_DELAYED_KO = 4,
  STATE_EFFECT_PREVENT_DAMAGE = 5,
  STATE_EFFECT_PREVENT_EFFECTS = 6,
  STATE_EFFECT_ATTACK_FLIP_FAIL = 7,
  STATE_EFFECT_NO_WEAKNESS = 8,
  STATE_EFFECT_TAKE_MORE_DAMAGE = 9,
  STATE_EFFECT_NEXT_ATTACK_BONUS = 10,
  STATE_EFFECT_REACTIVE_DAMAGE = 11,
  STATE_EFFECT_REACTIVE_EQUAL_DAMAGE = 12,
  STATE_EFFECT_ENERGY_ATTACH_DAMAGE = 13,
  STATE_EFFECT_ATTACK_DAMAGE_REDUCTION = 14,
  STATE_EFFECT_ATTACK_BONUS = 15,
};

constexpr int STATE_GLOBAL_RESTRICTIONS_SELF_COLUMN = 27;
constexpr int STATE_GLOBAL_RESTRICTIONS_OPP_COLUMN = 28;
constexpr int STATE_GLOBAL_EFFECTS_SELF_COLUMN = 29;
constexpr int STATE_GLOBAL_EFFECTS_OPP_COLUMN = 30;
constexpr int STATE_GLOBAL_EFFECT_VALUES_COLUMN = 31;
// restrictions self/opp: five 2-bit phases in enum order, low-energy
// threshold+1[3] at bits 10..12, shared ability-group-3-used at bit 13;
// bit 14 is Lunar-used for self and Canari-played for opponent.
// effects self/opp: four 2-bit phases in enum order, discard-hand threshold[3]
// at bits 8..10, typed-team-reduction EnergyType+1[4] at bits 11..14.
// global values: self/opp team-reduction-is-stacked[1] at bits 0..1
// (phase means 30, stacked means 60); self/opp Active-ex bonus code[2] at
// bits 2..5 (phase means 30 + 10*code); self/opp prize amount[2] at bits 6..9;
// self/opp prize kind at bits 10..11; fightingBuff-is-30 at bit 12;
// Tarragon-played at bit 13; saturated/non-domain value at bit 14.

enum StateGlobalRestrictionKey {
  STATE_RESTRICTION_ITEM = 0,
  STATE_RESTRICTION_SUPPORTER = 1,
  STATE_RESTRICTION_EVOLVE = 2,
  STATE_RESTRICTION_STADIUM = 3,
  STATE_RESTRICTION_LOW_ENERGY_ATTACK = 4,
};

enum StateGlobalEffectKey {
  STATE_GLOBAL_EFFECT_DISCARD_HAND = 0,
  STATE_GLOBAL_EFFECT_TEAM_REDUCTION = 1,
  STATE_GLOBAL_EFFECT_ACTIVE_EX_BONUS = 2,
  STATE_GLOBAL_EFFECT_PRIZE_BONUS = 3,
};
constexpr int STATE_SELECT_META_WIDTH = 16;
constexpr int STATE_SELECT_OPTION_WIDTH = 24;
constexpr int ACTION_MAX_OPTIONS = 64;
constexpr int ACTION_META_WIDTH = STATE_SELECT_META_WIDTH;
constexpr int ACTION_OPTION_WIDTH = STATE_SELECT_OPTION_WIDTH;
// Contextual option column: a stable printed-attack occurrence key for
// CG_OPTION_ATTACK rows.  The same column retains its legacy serial meaning
// for CG_OPTION_SKILL rows.  Keys are card_id * 4 + zero_based_slot + 1;
// zero means that the source occurrence could not be resolved.
constexpr int ACTION_ATTACK_DEFINITION_COLUMN = 12;
constexpr int ACTION_RAW_REF_LOW_COLUMN = 18;
constexpr int ACTION_RAW_REF_HIGH_COLUMN = 19;
constexpr int ACTION_RAW_REF_SHIFT = 15;

struct ObservationIds {
  std::vector<int32_t> in_play;        // [2, STATE_INPLAY_SLOTS, STATE_INPLAY_WIDTH]
  std::vector<int32_t> zones;          // [2, STATE_ZONE_COUNT, STATE_ZONE_SLOTS]
  std::vector<int32_t> player_counts;  // [2, 5]
  std::vector<int32_t> player_status;  // [2, 5]
  std::vector<int32_t> global;         // [STATE_GLOBAL_WIDTH]
};

struct ActionIds {
  std::vector<int32_t> meta;     // [ACTION_META_WIDTH]
  std::vector<int32_t> options;  // [ACTION_MAX_OPTIONS, ACTION_OPTION_WIDTH]
  std::vector<int32_t> deck;     // [STATE_ZONE_SLOTS]
  std::vector<uint8_t> mask;     // [ACTION_MAX_OPTIONS]
};

struct ActionIdView {
  bool pending = false;
  int n = 0;
  const std::vector<Descriptor>* descriptors = nullptr;
};

struct IdTensorSpec {
  std::vector<int> state_in_play_shape;
  std::vector<int> state_zones_shape;
  std::vector<int> state_player_counts_shape;
  std::vector<int> state_player_status_shape;
  std::vector<int> state_global_shape;
  std::vector<int> action_meta_shape;
  std::vector<int> action_options_shape;
  std::vector<int> action_deck_shape;
  std::vector<int> action_mask_shape;
};

void fill_observation_ids(const GameState& st, int32_t* in_play,
                           int32_t* zones, int32_t* player_counts,
                           int32_t* player_status, int32_t* global);
bool fill_observation_ids16(const GameState& st, int16_t* in_play,
                            int16_t* zones, int16_t* player_counts,
                            int16_t* player_status, int16_t* global);

ActionIdView action_id_view(const std::vector<Descriptor>& descriptors);

void fill_action_ids(const GameState& st, ActionIdView view, int32_t* meta,
                     int32_t* options, int32_t* deck, uint8_t* mask);
// Compact action options split the legacy int32 raw reference:
//   raw_ref = options[ACTION_RAW_REF_LOW_COLUMN]
//           + (options[ACTION_RAW_REF_HIGH_COLUMN] << ACTION_RAW_REF_SHIFT)
// All other columns retain their existing meaning. Returns false if any
// non-reference value cannot be represented exactly as signed int16.
bool fill_action_ids16(const GameState& st, ActionIdView view, int16_t* meta,
                       int16_t* options, int16_t* deck, uint8_t* mask);

ObservationIds make_observation_ids(const GameState& st);
ActionIds make_action_ids(const GameState& st, ActionIdView view);
ActionIds make_action_ids(const GameState& st);
IdTensorSpec id_tensor_spec();

}  // namespace ptcg
