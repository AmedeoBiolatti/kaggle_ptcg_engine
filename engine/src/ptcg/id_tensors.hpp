#pragma once

#include <cstdint>
#include <vector>

#include "ptcg/state.hpp"

namespace ptcg {

constexpr int STATE_INPLAY_SLOTS = 6;  // active + 5 bench
constexpr int STATE_INPLAY_WIDTH = 64;
constexpr int STATE_ZONE_COUNT = 4;  // hand, deck, discard, prizes
constexpr int STATE_ZONE_SLOTS = 64;
constexpr int STATE_GLOBAL_WIDTH = 32;
constexpr int STATE_SELECT_META_WIDTH = 16;
constexpr int STATE_SELECT_OPTION_WIDTH = 24;
constexpr int ACTION_MAX_OPTIONS = 64;
constexpr int ACTION_META_WIDTH = STATE_SELECT_META_WIDTH;
constexpr int ACTION_OPTION_WIDTH = STATE_SELECT_OPTION_WIDTH;

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

ActionIdView action_id_view(const std::vector<Descriptor>& descriptors);

void fill_action_ids(const GameState& st, ActionIdView view, int32_t* meta,
                     int32_t* options, int32_t* deck, uint8_t* mask);

ObservationIds make_observation_ids(const GameState& st);
ActionIds make_action_ids(const GameState& st, ActionIdView view);
ActionIds make_action_ids(const GameState& st);
IdTensorSpec id_tensor_spec();

}  // namespace ptcg
