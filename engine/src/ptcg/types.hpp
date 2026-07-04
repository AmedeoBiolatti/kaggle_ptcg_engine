#pragma once

namespace ptcg {

// Mirrors cg.api enums (only the values the engine needs).
enum CardType {
  POKEMON = 0, ITEM = 1, TOOL = 2, SUPPORTER = 3, STADIUM = 4,
  BASIC_ENERGY = 5, SPECIAL_ENERGY = 6,
};

enum EnergyType {
  COLORLESS = 0, GRASS = 1, FIRE = 2, WATER = 3, LIGHTNING = 4, PSYCHIC = 5,
  FIGHTING = 6, DARKNESS = 7, METAL = 8, DRAGON = 9, RAINBOW = 10,
  TEAM_ROCKET = 11,
};

enum AreaType {
  AREA_DECK = 1, AREA_HAND = 2, AREA_DISCARD = 3, AREA_ACTIVE = 4,
  AREA_BENCH = 5, AREA_PRIZE = 6, AREA_STADIUM = 7,
};

}  // namespace ptcg
