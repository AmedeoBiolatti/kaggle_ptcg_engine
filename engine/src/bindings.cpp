#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ptcg/card_db.hpp"
#include "ptcg/effect_vm.hpp"
#include "ptcg/id_tensors.hpp"
#include "ptcg/rl.hpp"
#include "ptcg/state.hpp"
#include "ptcg/version.hpp"
#include "search/debug_search.hpp"

namespace py = pybind11;

// Cast SmallVec exactly like std::vector (Python list) at the boundary.
namespace pybind11 {
namespace detail {
template <typename T, size_t N>
struct type_caster<ptcg::SmallVec<T, N>>
    : list_caster<ptcg::SmallVec<T, N>, T> {};
}  // namespace detail
}  // namespace pybind11

using namespace ptcg;
namespace dbg = ptcg::search;

static std::string_view atom_string(const Descriptor& d, size_t i);
static int atom_int(const Descriptor& d, size_t i, int fallback = 0);
static bool descriptor_equal(const Descriptor& a, const Descriptor& b);
static py::dict search_transients_to_py(const GameState& st);

static ActionIdView action_id_view_from_options(const RlOptionSet& opts) {
  return ActionIdView{opts.pending, opts.n, &opts.descriptors};
}

static constexpr int STATE_CARD_SLOTS = 64;

static py::dict rl_state_ids_py(const GameState& st) {
  using Shape = std::vector<py::ssize_t>;
  py::array_t<int32_t> in_play(
      Shape{2, STATE_INPLAY_SLOTS, STATE_INPLAY_WIDTH});
  py::array_t<int32_t> zones(Shape{2, STATE_ZONE_COUNT, STATE_ZONE_SLOTS});
  py::array_t<int32_t> counts(Shape{2, 5});
  py::array_t<int32_t> status(Shape{2, 5});
  py::array_t<int32_t> global(Shape{STATE_GLOBAL_WIDTH});
  ptcg::fill_observation_ids(st, in_play.mutable_data(), zones.mutable_data(),
                              counts.mutable_data(), status.mutable_data(),
                              global.mutable_data());
  py::dict out;
  out["in_play"] = in_play;
  out["zones"] = zones;
  out["player_counts"] = counts;
  out["player_status"] = status;
  out["global"] = global;
  out["empty_card_id"] = 0;
  out["unknown_card_id"] = -1;
  out["zone_hand"] = 0;
  out["zone_deck"] = 1;
  out["zone_discard"] = 2;
  out["zone_prizes"] = 3;
  return out;
}

static py::dict rl_state_observation_py(const GameState& st) {
  py::array_t<int32_t> card_id(STATE_CARD_SLOTS);
  py::array_t<int32_t> owner(STATE_CARD_SLOTS);
  py::array_t<int32_t> area(STATE_CARD_SLOTS);
  py::array_t<int32_t> index(STATE_CARD_SLOTS);
  py::array_t<uint8_t> known(STATE_CARD_SLOTS);
  py::array_t<int32_t> hp(STATE_CARD_SLOTS);
  py::array_t<int32_t> max_hp(STATE_CARD_SLOTS);
  py::array_t<int32_t> energy_count(STATE_CARD_SLOTS);
  py::array_t<int32_t> tool_count(STATE_CARD_SLOTS);

  std::fill(card_id.mutable_data(), card_id.mutable_data() + STATE_CARD_SLOTS, 0);
  std::fill(owner.mutable_data(), owner.mutable_data() + STATE_CARD_SLOTS, -1);
  std::fill(area.mutable_data(), area.mutable_data() + STATE_CARD_SLOTS, 0);
  std::fill(index.mutable_data(), index.mutable_data() + STATE_CARD_SLOTS, -1);
  std::fill(known.mutable_data(), known.mutable_data() + STATE_CARD_SLOTS, 0);
  std::fill(hp.mutable_data(), hp.mutable_data() + STATE_CARD_SLOTS, 0);
  std::fill(max_hp.mutable_data(), max_hp.mutable_data() + STATE_CARD_SLOTS, 0);
  std::fill(energy_count.mutable_data(),
            energy_count.mutable_data() + STATE_CARD_SLOTS, 0);
  std::fill(tool_count.mutable_data(),
            tool_count.mutable_data() + STATE_CARD_SLOTS, 0);

  int slot = 0;
  auto append_slot = [&](int cid, int slot_owner, int slot_area, int slot_index,
                         bool is_known, const InPlay* in_play = nullptr) {
    if (slot >= STATE_CARD_SLOTS) return;
    card_id.mutable_data()[slot] = is_known ? cid : 0;
    owner.mutable_data()[slot] = slot_owner;
    area.mutable_data()[slot] = slot_area;
    index.mutable_data()[slot] = slot_index;
    known.mutable_data()[slot] = is_known ? 1 : 0;
    if (in_play && is_known) {
      hp.mutable_data()[slot] = in_play->hp;
      max_hp.mutable_data()[slot] = in_play->maxHp;
      energy_count.mutable_data()[slot] =
          static_cast<int32_t>(in_play->energies.size());
      tool_count.mutable_data()[slot] =
          static_cast<int32_t>(in_play->tools.size());
    }
    ++slot;
  };

  auto append_player_inplay = [&](const Player& p, int owner_pov) {
    append_slot(p.active.id, owner_pov, AREA_ACTIVE, 0,
                p.activePresent && p.activeKnown, &p.active);
    for (int i = 0; i < 5; ++i) {
      if (i < static_cast<int>(p.bench.size())) {
        append_slot(p.bench[i].id, owner_pov, AREA_BENCH, i, true,
                    &p.bench[i]);
      } else {
        append_slot(0, owner_pov, AREA_BENCH, i, false);
      }
    }
  };

  const int me = st.yourIndex;
  append_player_inplay(st.players[me], 0);
  append_player_inplay(st.players[1 - me], 1);

  append_slot(st.stadium.empty() ? 0 : st.stadium[0],
              st.stadiumOwner == me ? 0 : (st.stadiumOwner == 1 - me ? 1 : -1),
              AREA_STADIUM, 0, !st.stadium.empty());

  auto append_card_list = [&](const SmallVec<int, 64>& cards, int owner_pov,
                              int slot_area, int count) {
    for (int i = 0; i < count; ++i) {
      const bool ok = i < static_cast<int>(cards.size());
      append_slot(ok ? cards[i] : 0, owner_pov, slot_area, i, ok);
    }
  };
  append_card_list(st.players[me].hand, 0, AREA_HAND, 8);
  append_card_list(st.players[1 - me].handKnownCards, 1, AREA_HAND, 8);
  append_card_list(st.players[me].discard, 0, AREA_DISCARD, 8);
  append_card_list(st.players[1 - me].discard, 1, AREA_DISCARD, 8);

  for (int i = 0; i < 16; ++i) {
    const Player& p = st.players[me];
    if (i < static_cast<int>(p.deck.size())) {
      int idx = static_cast<int>(p.deck.size()) - 1 - i;
      bool is_known = p.deckKnown ||
                      (idx < static_cast<int>(p.deckKnownMask.size()) &&
                       p.deckKnownMask[idx]);
      append_slot(p.deck[idx], 0, AREA_DECK, i, is_known);
    } else if (i < static_cast<int>(p.deckKnownCards.size())) {
      append_slot(p.deckKnownCards[i], 0, AREA_DECK, i, true);
    } else {
      append_slot(0, 0, AREA_DECK, i, false);
    }
  }

  for (int i = 0; i < 3; ++i) {
    const Player& p = st.players[me];
    bool is_known = p.prizesKnown ||
                    (i < static_cast<int>(p.prizesKnownMask.size()) &&
                     p.prizesKnownMask[i]);
    if (i < static_cast<int>(p.prizes.size())) {
      append_slot(p.prizes[i], 0, AREA_PRIZE, i, is_known);
    } else if (i < static_cast<int>(p.prizesKnownCards.size())) {
      append_slot(p.prizesKnownCards[i], 0, AREA_PRIZE, i, true);
    } else {
      append_slot(0, 0, AREA_PRIZE, i, false);
    }
  }

  py::array_t<int32_t> player_counts(std::vector<py::ssize_t>{2, 4});
  auto* counts = player_counts.mutable_data();
  for (int pov = 0; pov < 2; ++pov) {
    const Player& p = st.players[pov == 0 ? me : 1 - me];
    counts[pov * 4 + 0] = p.handCount;
    counts[pov * 4 + 1] = p.deckCount;
    counts[pov * 4 + 2] = p.prizeCount;
    counts[pov * 4 + 3] = static_cast<int32_t>(p.bench.size());
  }

  py::array_t<uint8_t> player_status(std::vector<py::ssize_t>{2, 5});
  auto* status = player_status.mutable_data();
  for (int pov = 0; pov < 2; ++pov) {
    const Player& p = st.players[pov == 0 ? me : 1 - me];
    status[pov * 5 + 0] = p.poisoned ? 1 : 0;
    status[pov * 5 + 1] = p.burned ? 1 : 0;
    status[pov * 5 + 2] = p.asleep ? 1 : 0;
    status[pov * 5 + 3] = p.paralyzed ? 1 : 0;
    status[pov * 5 + 4] = p.confused ? 1 : 0;
  }

  py::dict global;
  global["turn"] = st.turn;
  global["turn_action_count"] = st.turnActionCount;
  global["current_player"] = st.yourIndex;
  global["first_player"] = st.firstPlayer;
  global["result"] = st.result;
  global["supporter_played"] = st.supporterPlayed;
  global["stadium_played"] = st.stadiumPlayed;
  global["energy_attached"] = st.energyAttached;
  global["retreated"] = st.retreated;
  global["has_pending"] = st.has_pending();
  global["pending_context"] = st.has_pending() ? st.pending.context : -1;
  global["pending_min_count"] = st.has_pending() ? st.pending.minCount : 0;
  global["pending_max_count"] = st.has_pending() ? st.pending.maxCount : 0;

  py::dict out;
  out["card_id"] = card_id;
  out["owner"] = owner;
  out["area"] = area;
  out["index"] = index;
  out["known"] = known;
  out["hp"] = hp;
  out["max_hp"] = max_hp;
  out["energy_count"] = energy_count;
  out["tool_count"] = tool_count;
  out["player_counts"] = player_counts;
  out["player_status"] = player_status;
  out["global"] = global;
  return out;
}

// --- parse a cabt `current` state dict into our GameState -----------------

static InPlay parse_pokemon(const py::handle& obj) {
  py::dict d = obj.cast<py::dict>();
  InPlay p;
  p.id = d["id"].cast<int>();
  if (d.contains("serial") && !d["serial"].is_none())
    p.serial = d["serial"].cast<int>();
  p.hp = d["hp"].cast<int>();
  p.maxHp = d["maxHp"].cast<int>();
  if (d.contains("appearThisTurn"))
    p.appearThisTurn = d["appearThisTurn"].cast<bool>();
  if (d.contains("movedToActiveThisTurn"))
    p.movedToActiveThisTurn = d["movedToActiveThisTurn"].cast<bool>();
  if (d.contains("energies") && !d["energies"].is_none())
    for (auto e : d["energies"]) p.energies.push_back(e.cast<int>());
  if (d.contains("energyCards") && !d["energyCards"].is_none()) {
    bool saw_order = false;
    for (auto e : d["energyCards"]) {
      py::dict ed = e.cast<py::dict>();
      p.energyCardIds.push_back(ed["id"].cast<int>());
      int order = -1;
      if (ed.contains("attachOrder") && !ed["attachOrder"].is_none()) {
        order = ed["attachOrder"].cast<int>();
        saw_order = true;
      }
      p.energyCardOrders.push_back(order);
    }
    if (!saw_order) p.energyCardOrders.clear();
  }
  if (p.energies.empty() && !p.energyCardIds.empty()) {
    for (int cid : p.energyCardIds) {
      const CardInfo* ci = find_card(cid);
      if (ci && (ci->cardType == BASIC_ENERGY || ci->cardType == SPECIAL_ENERGY))
        p.energies.push_back(ci->energyType);
    }
  }
  if (d.contains("tools") && !d["tools"].is_none()) {
    bool saw_order = false;
    for (auto t : d["tools"]) {
      py::dict td = t.cast<py::dict>();
      p.tools.push_back(td["id"].cast<int>());
      int order = -1;
      if (td.contains("attachOrder") && !td["attachOrder"].is_none()) {
        order = td["attachOrder"].cast<int>();
        saw_order = true;
      } else if (td.contains("serial") && !td["serial"].is_none()) {
        order = 100000000 + td["serial"].cast<int>();
        saw_order = true;
      }
      p.toolOrders.push_back(order);
    }
    if (!saw_order) p.toolOrders.clear();
  }
  if (d.contains("preEvolution") && !d["preEvolution"].is_none())
    for (auto t : d["preEvolution"])
      p.preEvo.push_back(t.cast<py::dict>()["id"].cast<int>());
  if (d.contains("lockId"))
    p.lockId = d["lockId"].cast<int>();
  if (d.contains("lockTurn"))
    p.lockTurn = d["lockTurn"].cast<int>();
  if (d.contains("activeLockId"))
    p.activeLockId = d["activeLockId"].cast<int>();
  if (d.contains("noAttackTurn"))
    p.noAttackTurn = d["noAttackTurn"].cast<int>();
  if (d.contains("damagedByAttackTurn"))
    p.damagedByAttackTurn = d["damagedByAttackTurn"].cast<int>();
  if (d.contains("damagedByAttackSide"))
    p.damagedByAttackSide = d["damagedByAttackSide"].cast<int>();
  if (d.contains("damagedByAttackAmount"))
    p.damagedByAttackAmount = d["damagedByAttackAmount"].cast<int>();
  if (d.contains("dmgReduce"))
    p.dmgReduce = d["dmgReduce"].cast<int>();
  if (d.contains("dmgReduceTurn"))
    p.dmgReduceTurn = d["dmgReduceTurn"].cast<int>();
  if (d.contains("preventDmgTurn"))
    p.preventDmgTurn = d["preventDmgTurn"].cast<int>();
  if (d.contains("preventDmgCond"))
    p.preventDmgCond = d["preventDmgCond"].cast<int>();
  if (d.contains("preventDmgValue"))
    p.preventDmgValue = d["preventDmgValue"].cast<int>();
  if (d.contains("preventEffectsTurn"))
    p.preventEffectsTurn = d["preventEffectsTurn"].cast<int>();
  if (d.contains("preventEffectsCond"))
    p.preventEffectsCond = d["preventEffectsCond"].cast<int>();
  if (d.contains("preventEffectsValue"))
    p.preventEffectsValue = d["preventEffectsValue"].cast<int>();
  if (d.contains("takeMoreDamageTurn"))
    p.takeMoreDamageTurn = d["takeMoreDamageTurn"].cast<int>();
  if (d.contains("takeMoreDamage"))
    p.takeMoreDamage = d["takeMoreDamage"].cast<int>();
  if (d.contains("noRetreatTurn"))
    p.noRetreatTurn = d["noRetreatTurn"].cast<int>();
  if (d.contains("retreatCostModTurn"))
    p.retreatCostModTurn = d["retreatCostModTurn"].cast<int>();
  if (d.contains("retreatCostMod"))
    p.retreatCostMod = d["retreatCostMod"].cast<int>();
  if (d.contains("attackCostModTurn"))
    p.attackCostModTurn = d["attackCostModTurn"].cast<int>();
  if (d.contains("attackCostMod"))
    p.attackCostMod = d["attackCostMod"].cast<int>();
  if (d.contains("attackFlipFailTurn"))
    p.attackFlipFailTurn = d["attackFlipFailTurn"].cast<int>();
  if (d.contains("delayedDamageTurn"))
    p.delayedDamageTurn = d["delayedDamageTurn"].cast<int>();
  if (d.contains("delayedDamageCounters"))
    p.delayedDamageCounters = d["delayedDamageCounters"].cast<int>();
  if (d.contains("delayedKoTurn"))
    p.delayedKoTurn = d["delayedKoTurn"].cast<int>();
  if (d.contains("delayedKoPromoteBeforePrize"))
    p.delayedKoPromoteBeforePrize =
        d["delayedKoPromoteBeforePrize"].cast<bool>();
  if (d.contains("nextAttackBonusId"))
    p.nextAttackBonusId = d["nextAttackBonusId"].cast<int>();
  if (d.contains("nextAttackBonusTurn"))
    p.nextAttackBonusTurn = d["nextAttackBonusTurn"].cast<int>();
  if (d.contains("nextAttackBonus"))
    p.nextAttackBonus = d["nextAttackBonus"].cast<int>();
  if (d.contains("nextAttackSetBase"))
    p.nextAttackSetBase = d["nextAttackSetBase"].cast<int>();
  if (d.contains("attackDmgReduce"))
    p.attackDmgReduce = d["attackDmgReduce"].cast<int>();
  if (d.contains("attackDmgReduceTurn"))
    p.attackDmgReduceTurn = d["attackDmgReduceTurn"].cast<int>();
  if (d.contains("damagedByAttackCountersTurn"))
    p.damagedByAttackCountersTurn =
        d["damagedByAttackCountersTurn"].cast<int>();
  if (d.contains("damagedByAttackCounters"))
    p.damagedByAttackCounters = d["damagedByAttackCounters"].cast<int>();
  if (d.contains("damagedByAttackEqualCountersTurn"))
    p.damagedByAttackEqualCountersTurn =
        d["damagedByAttackEqualCountersTurn"].cast<int>();
  if (d.contains("damagedByAttackStatus"))
    p.damagedByAttackStatus = d["damagedByAttackStatus"].cast<int>();
  if (d.contains("energyAttachCountersTurn"))
    p.energyAttachCountersTurn = d["energyAttachCountersTurn"].cast<int>();
  if (d.contains("energyAttachCounters"))
    p.energyAttachCounters = d["energyAttachCounters"].cast<int>();
  if (d.contains("energyAttachCountersFromHandOnly"))
    p.energyAttachCountersFromHandOnly =
        d["energyAttachCountersFromHandOnly"].cast<int>();
  return p;
}

static Player parse_player(const py::dict& d) {
  Player p;
  py::object active = d["active"];
  if (!active.is_none()) {
    py::list al = active.cast<py::list>();
    if (py::len(al) >= 1) {
      p.activePresent = true;
      if (!al[0].is_none()) {
        p.activeKnown = true;
        p.active = parse_pokemon(al[0]);
      }
    }
  }
  for (auto b : d["bench"]) p.bench.push_back(parse_pokemon(b));
  p.benchMax = d["benchMax"].cast<int>();
  p.deckCount = d["deckCount"].cast<int>();
  p.handCount = d["handCount"].cast<int>();
  py::object hand = d["hand"];
  if (!hand.is_none()) {
    p.handKnown = true;
    for (auto c : hand) p.hand.push_back(c.cast<py::dict>()["id"].cast<int>());
  }
  if (d.contains("handKnownCards") && !d["handKnownCards"].is_none())
    for (auto c : d["handKnownCards"]) p.handKnownCards.push_back(c.cast<int>());
  for (auto c : d["discard"])
    p.discard.push_back(c.cast<py::dict>()["id"].cast<int>());
  bool explicit_deck_known = d.contains("deckKnown") && !d["deckKnown"].is_none();
  if (explicit_deck_known)
    p.deckKnown = d["deckKnown"].cast<bool>();
  if (d.contains("deck") && !d["deck"].is_none()) {  // optional: ordered deck (top = back)
    for (auto c : d["deck"]) p.deck.push_back(c.cast<int>());
    bool all_positive = true;
    for (int cid : p.deck)
      if (cid <= 0) {
        all_positive = false;
        break;
      }
    if (!explicit_deck_known)
      p.deckKnown = all_positive;
    if (p.deckKnown)
      p.deckKnownMask.assign(p.deck.size(), true);
  }
  if (d.contains("deckKnownMask") && !d["deckKnownMask"].is_none()) {
    p.deckKnownMask.clear();
    for (auto b : d["deckKnownMask"]) p.deckKnownMask.push_back(b.cast<bool>());
    p.deckKnown = !p.deckKnownMask.empty();
    for (bool known : p.deckKnownMask)
      if (!known) {
        p.deckKnown = false;
        break;
      }
  }
  if (d.contains("deckKnownCards") && !d["deckKnownCards"].is_none())
    for (auto c : d["deckKnownCards"]) p.deckKnownCards.push_back(c.cast<int>());
  py::object prize = d["prize"];
  p.prizeCount = prize.is_none() ? 0 : static_cast<int>(py::len(prize));
  if (!prize.is_none()) {
    for (auto c : prize) {
      if (!c.is_none()) {
        p.prizes.push_back(c.cast<py::dict>()["id"].cast<int>());
        p.prizesKnownMask.push_back(true);
      } else {
        p.prizes.push_back(0);
        p.prizesKnownMask.push_back(false);
      }
    }
  }
  bool explicit_prizes_known =
      d.contains("prizesKnown") && !d["prizesKnown"].is_none();
  if (explicit_prizes_known)
    p.prizesKnown = d["prizesKnown"].cast<bool>();
  if (!explicit_prizes_known && p.prizeCount > 0 &&
      static_cast<int>(p.prizes.size()) == p.prizeCount) {
    p.prizesKnown = true;
    for (int cid : p.prizes)
      if (cid <= 0) {
        p.prizesKnown = false;
        break;
      }
  }
  if (d.contains("prizesKnownMask") && !d["prizesKnownMask"].is_none()) {
    p.prizesKnownMask.clear();
    for (auto b : d["prizesKnownMask"]) p.prizesKnownMask.push_back(b.cast<bool>());
  }
  if (d.contains("prizesKnownCards") && !d["prizesKnownCards"].is_none())
    for (auto c : d["prizesKnownCards"]) p.prizesKnownCards.push_back(c.cast<int>());
  if (d.contains("prizeFaceUp") && !d["prizeFaceUp"].is_none()) {
    for (auto b : d["prizeFaceUp"]) p.prizeFaceUp.push_back(b.cast<bool>());
  }
  if (p.prizeFaceUp.size() < static_cast<size_t>(p.prizeCount))
    p.prizeFaceUp.resize(p.prizeCount, false);
  if (p.prizesKnownMask.size() < static_cast<size_t>(p.prizeCount))
    p.prizesKnownMask.resize(p.prizeCount, false);
  for (size_t i = 0; i < p.prizeFaceUp.size() && i < p.prizesKnownMask.size(); ++i)
    if (p.prizeFaceUp[i]) p.prizesKnownMask[i] = true;
  p.poisoned = d["poisoned"].cast<bool>();
  if (d.contains("poisonDamageCounters"))
    p.poisonDamageCounters = d["poisonDamageCounters"].cast<int>();
  p.burned = d["burned"].cast<bool>();
  p.asleep = d["asleep"].cast<bool>();
  p.paralyzed = d["paralyzed"].cast<bool>();
  p.confused = d["confused"].cast<bool>();
  return p;
}

static GameState load_state(const py::dict& cur) {
  GameState st;
  st.collectLogs = true;  // replay/oracle states keep native-log emission
  st.turn = cur["turn"].cast<int>();
  st.turnActionCount = cur["turnActionCount"].cast<int>();
  st.yourIndex = cur["yourIndex"].cast<int>();
  st.firstPlayer = cur["firstPlayer"].cast<int>();
  st.result = cur["result"].cast<int>();
  st.supporterPlayed = cur["supporterPlayed"].cast<bool>();
  if (cur.contains("teamRocketSupporterPlayed"))
    st.teamRocketSupporterPlayed = cur["teamRocketSupporterPlayed"].cast<bool>();
  if (cur.contains("ancientSupporterPlayed"))
    st.ancientSupporterPlayed = cur["ancientSupporterPlayed"].cast<bool>();
  if (cur.contains("canariPlayed"))
    st.canariPlayed = cur["canariPlayed"].cast<bool>();
  if (cur.contains("tarragonPlayed"))
    st.tarragonPlayed = cur["tarragonPlayed"].cast<bool>();
  st.stadiumPlayed = cur["stadiumPlayed"].cast<bool>();
  st.energyAttached = cur["energyAttached"].cast<bool>();
  st.retreated = cur["retreated"].cast<bool>();
  if (cur.contains("fightingBuff"))
    st.fightingBuff = cur["fightingBuff"].cast<int>();
  if (cur.contains("noItemTurn") && !cur["noItemTurn"].is_none()) {
    py::list turns = cur["noItemTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.noItemTurn[0] = turns[0].cast<int>();
      st.noItemTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("noSupporterTurn") && !cur["noSupporterTurn"].is_none()) {
    py::list turns = cur["noSupporterTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.noSupporterTurn[0] = turns[0].cast<int>();
      st.noSupporterTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("noEvolveTurn") && !cur["noEvolveTurn"].is_none()) {
    py::list turns = cur["noEvolveTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.noEvolveTurn[0] = turns[0].cast<int>();
      st.noEvolveTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("noStadiumTurn") && !cur["noStadiumTurn"].is_none()) {
    py::list turns = cur["noStadiumTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.noStadiumTurn[0] = turns[0].cast<int>();
      st.noStadiumTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("lastKoTurn") && !cur["lastKoTurn"].is_none()) {
    py::list turns = cur["lastKoTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.lastKoTurn[0] = turns[0].cast<int>();
      st.lastKoTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("lastAttackDamageKoTurn") &&
      !cur["lastAttackDamageKoTurn"].is_none()) {
    py::list turns = cur["lastAttackDamageKoTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.lastAttackDamageKoTurn[0] = turns[0].cast<int>();
      st.lastAttackDamageKoTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("lastTeamRocketKoTurn") &&
      !cur["lastTeamRocketKoTurn"].is_none()) {
    py::list turns = cur["lastTeamRocketKoTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.lastTeamRocketKoTurn[0] = turns[0].cast<int>();
      st.lastTeamRocketKoTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("lastEthanKoTurn") && !cur["lastEthanKoTurn"].is_none()) {
    py::list turns = cur["lastEthanKoTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.lastEthanKoTurn[0] = turns[0].cast<int>();
      st.lastEthanKoTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("lastAttackTurn") && !cur["lastAttackTurn"].is_none()) {
    py::list turns = cur["lastAttackTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.lastAttackTurn[0] = turns[0].cast<int>();
      st.lastAttackTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("lastAttackId") && !cur["lastAttackId"].is_none()) {
    py::list ids = cur["lastAttackId"].cast<py::list>();
    if (py::len(ids) >= 2) {
      st.lastAttackId[0] = ids[0].cast<int>();
      st.lastAttackId[1] = ids[1].cast<int>();
    }
  }
  if (cur.contains("lastAncientAttackTurn") &&
      !cur["lastAncientAttackTurn"].is_none()) {
    py::list turns = cur["lastAncientAttackTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.lastAncientAttackTurn[0] = turns[0].cast<int>();
      st.lastAncientAttackTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("lastAncientAttackCard") &&
      !cur["lastAncientAttackCard"].is_none()) {
    py::list ids = cur["lastAncientAttackCard"].cast<py::list>();
    if (py::len(ids) >= 2) {
      st.lastAncientAttackCard[0] = ids[0].cast<int>();
      st.lastAncientAttackCard[1] = ids[1].cast<int>();
    }
  }
  if (cur.contains("lastAncientAttackSerial") &&
      !cur["lastAncientAttackSerial"].is_none()) {
    py::list ids = cur["lastAncientAttackSerial"].cast<py::list>();
    if (py::len(ids) >= 2) {
      st.lastAncientAttackSerial[0] = ids[0].cast<int>();
      st.lastAncientAttackSerial[1] = ids[1].cast<int>();
    }
  }
  if (cur.contains("prizeTakenTurn") && !cur["prizeTakenTurn"].is_none()) {
    py::list turns = cur["prizeTakenTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.prizeTakenTurn[0] = turns[0].cast<int>();
      st.prizeTakenTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("prizeTakenCount") && !cur["prizeTakenCount"].is_none()) {
    py::list counts = cur["prizeTakenCount"].cast<py::list>();
    if (py::len(counts) >= 2) {
      st.prizeTakenCount[0] = counts[0].cast<int>();
      st.prizeTakenCount[1] = counts[1].cast<int>();
    }
  }
  if (cur.contains("teamReduceTurn") && !cur["teamReduceTurn"].is_none()) {
    py::list turns = cur["teamReduceTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.teamReduceTurn[0] = turns[0].cast<int>();
      st.teamReduceTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("teamReduceAmount") && !cur["teamReduceAmount"].is_none()) {
    py::list amounts = cur["teamReduceAmount"].cast<py::list>();
    if (py::len(amounts) >= 2) {
      st.teamReduceAmount[0] = amounts[0].cast<int>();
      st.teamReduceAmount[1] = amounts[1].cast<int>();
    }
  }
  if (cur.contains("teamReduceType") && !cur["teamReduceType"].is_none()) {
    py::list types = cur["teamReduceType"].cast<py::list>();
    if (py::len(types) >= 2) {
      st.teamReduceType[0] = types[0].cast<int>();
      st.teamReduceType[1] = types[1].cast<int>();
    }
  }
  if (cur.contains("activeExDamageBuffTurn") &&
      !cur["activeExDamageBuffTurn"].is_none()) {
    py::list turns = cur["activeExDamageBuffTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.activeExDamageBuffTurn[0] = turns[0].cast<int>();
      st.activeExDamageBuffTurn[1] = turns[1].cast<int>();
    }
  }
  if (cur.contains("activeExDamageBuffAmount") &&
      !cur["activeExDamageBuffAmount"].is_none()) {
    py::list amounts = cur["activeExDamageBuffAmount"].cast<py::list>();
    if (py::len(amounts) >= 2) {
      st.activeExDamageBuffAmount[0] = amounts[0].cast<int>();
      st.activeExDamageBuffAmount[1] = amounts[1].cast<int>();
    }
  }
  if (!cur["stadium"].is_none())
    for (auto c : cur["stadium"]) {
      py::dict card = c.cast<py::dict>();
      st.stadium.push_back(card["id"].cast<int>());
      if (card.contains("playerIndex") && !card["playerIndex"].is_none())
        st.stadiumOwner = card["playerIndex"].cast<int>();
    }
  if (cur.contains("stadiumAbilityUsed"))
    st.stadiumAbilityUsed = cur["stadiumAbilityUsed"].cast<bool>();
  py::list players = cur["players"].cast<py::list>();
  st.players[0] = parse_player(players[0].cast<py::dict>());
  st.players[1] = parse_player(players[1].cast<py::dict>());
  if (cur.contains("lastHopAttackDamageKoTurn") &&
      !cur["lastHopAttackDamageKoTurn"].is_none()) {
    py::list turns = cur["lastHopAttackDamageKoTurn"].cast<py::list>();
    if (py::len(turns) >= 2) {
      st.lastHopAttackDamageKoTurn[0] = turns[0].cast<int>();
      st.lastHopAttackDamageKoTurn[1] = turns[1].cast<int>();
    }
  }
  return st;
}

// --- emit the canonical projection (must match oracle.canonical_state) -----

static py::list sorted_ids(const auto& v) {
  std::vector<int> s = v;
  std::sort(s.begin(), s.end());
  py::list out;
  for (int x : s)
    if (x > 0) out.append(x);
  return out;
}

static py::dict pokemon_canon(const GameState& st, const InPlay& p, int ownerSide) {
  py::dict d;
  d["id"] = p.id;
  d["hp"] = p.hp;
  d["maxHp"] = p.maxHp;
  std::map<int, int> cnt;
  for (int e : provided_energy_units(p, &st, ownerSide)) ++cnt[e];
  py::list el;
  for (auto& kv : cnt) el.append(py::make_tuple(kv.first, kv.second));
  d["energies"] = el;
  d["tools"] = sorted_ids(p.tools);
  d["preEvo"] = sorted_ids(p.preEvo);
  d["appearThisTurn"] = p.appearThisTurn;
  return d;
}

static py::object player_canon(const GameState& st, int side, bool reveal_hand) {
  const Player& p = st.players[side];
  py::dict d;
  d["active"] = p.activeKnown ? py::object(pokemon_canon(st, p.active, side))
                              : py::object(py::none());
  py::list bench;
  for (auto& b : p.bench) bench.append(pokemon_canon(st, b, side));
  d["bench"] = bench;
  d["benchMax"] = p.benchMax;
  d["deckCount"] = p.deckCount;
  d["handCount"] = p.handCount;
  d["hand"] = (reveal_hand && p.handKnown) ? py::object(sorted_ids(p.hand))
                                           : py::object(py::none());
  d["discard"] = sorted_ids(p.discard);
  d["prizeCount"] = p.prizeCount;
  d["prize"] = sorted_ids(p.prizes);
  py::list faceUp;
  for (bool b : p.prizeFaceUp) faceUp.append(b);
  d["prizeFaceUp"] = faceUp;
  py::list cond;
  if (p.poisoned) cond.append("poisoned");
  if (p.burned) cond.append("burned");
  if (p.asleep) cond.append("asleep");
  if (p.paralyzed) cond.append("paralyzed");
  if (p.confused) cond.append("confused");
  d["conditions"] = cond;
  return d;
}

static py::dict canonical(const GameState& st) {
  py::dict d;
  d["turn"] = st.turn;
  d["turnActionCount"] = st.turnActionCount;
  d["yourIndex"] = st.yourIndex;
  d["firstPlayer"] = st.firstPlayer;
  py::dict flags;
  flags["supporterPlayed"] = st.supporterPlayed;
  flags["teamRocketSupporterPlayed"] = st.teamRocketSupporterPlayed;
  flags["ancientSupporterPlayed"] = st.ancientSupporterPlayed;
  flags["canariPlayed"] = st.canariPlayed;
  flags["tarragonPlayed"] = st.tarragonPlayed;
  flags["stadiumPlayed"] = st.stadiumPlayed;
  flags["energyAttached"] = st.energyAttached;
  flags["retreated"] = st.retreated;
  d["flags"] = flags;
  d["result"] = st.result;
  d["stadium"] = sorted_ids(st.stadium);
  py::list players;
  players.append(player_canon(st, 0, st.yourIndex == 0));
  players.append(player_canon(st, 1, st.yourIndex == 1));
  d["players"] = players;
  return d;
}

static py::dict debug_hidden_zones(const GameState& st) {
  py::dict out;
  py::list players;
  for (int side = 0; side < 2; ++side) {
    const Player& p = st.players[side];
    py::dict d;
    py::list deck;
    for (int cid : p.deck) deck.append(cid);
    py::list prizes;
    for (int cid : p.prizes) prizes.append(cid);
    py::list hand;
    for (int cid : p.hand) hand.append(cid);
    py::list deckMask;
    for (bool b : p.deckKnownMask) deckMask.append(b);
    py::list prizeMask;
    for (bool b : p.prizesKnownMask) prizeMask.append(b);
    py::list handKnownCards;
    for (int cid : p.handKnownCards) handKnownCards.append(cid);
    py::list deckKnownCards;
    for (int cid : p.deckKnownCards) deckKnownCards.append(cid);
    py::list prizesKnownCards;
    for (int cid : p.prizesKnownCards) prizesKnownCards.append(cid);
    d["deck"] = deck;
    d["deckKnown"] = p.deckKnown;
    d["deckKnownMask"] = deckMask;
    d["prizes"] = prizes;
    d["prizesKnown"] = p.prizesKnown;
    d["prizesKnownMask"] = prizeMask;
    d["hand"] = hand;
    d["handKnown"] = p.handKnown;
    d["handKnownCards"] = handKnownCards;
    d["deckKnownCards"] = deckKnownCards;
    d["prizesKnownCards"] = prizesKnownCards;
    players.append(d);
  }
  out["players"] = players;
  return out;
}

static py::dict native_log_to_py(const NativeLog& log) {
  py::dict d;
  d["type"] = log.type;
  if (log.playerIndex >= 0) d["playerIndex"] = log.playerIndex;
  if (log.cardId > 0) d["cardId"] = log.cardId;
  if (log.serial > 0) d["serial"] = log.serial;
  if (log.fromArea >= 0) d["fromArea"] = log.fromArea;
  if (log.toArea >= 0) d["toArea"] = log.toArea;
  if (log.cardIdActive > 0) d["cardIdActive"] = log.cardIdActive;
  if (log.serialActive > 0) d["serialActive"] = log.serialActive;
  if (log.cardIdBench > 0) d["cardIdBench"] = log.cardIdBench;
  if (log.serialBench > 0) d["serialBench"] = log.serialBench;
  if (log.cardIdTarget > 0) d["cardIdTarget"] = log.cardIdTarget;
  if (log.serialTarget > 0) d["serialTarget"] = log.serialTarget;
  if (log.attackId > 0) d["attackId"] = log.attackId;
  if (log.value != 0) d["value"] = log.value;
  if (log.result >= 0) d["result"] = log.result;
  if (log.reason != 0) d["reason"] = log.reason;
  if (log.hasBasicPokemon) d["hasBasicPokemon"] = log.hasBasicPokemon;
  if (log.putDamageCounter) d["putDamageCounter"] = log.putDamageCounter;
  if (log.isRecover) d["isRecover"] = log.isRecover;
  if (log.type == 22) d["head"] = log.head;
  return d;
}

static py::list native_logs_to_py(const GameState& st) {
  py::list out;
  for (const auto& log : st.nativeLogs)
    out.append(native_log_to_py(log));
  return out;
}

static void add_visible_card(std::map<int, int>& counts, int cid) {
  if (cid > 0) ++counts[cid];
}

static void add_visible_inplay(std::map<int, int>& counts, const InPlay& p) {
  add_visible_card(counts, p.id);
  for (int cid : p.energyCardIds) add_visible_card(counts, cid);
  for (int cid : p.tools) add_visible_card(counts, cid);
  for (int cid : p.preEvo) add_visible_card(counts, cid);
}

static std::map<int, int> visible_card_counts_for_player(
    const GameState& st, int player, int perspective) {
  const Player& p = st.players[player];
  std::map<int, int> counts;

  if (p.activePresent && p.activeKnown) add_visible_inplay(counts, p.active);
  for (const InPlay& b : p.bench) add_visible_inplay(counts, b);
  for (int cid : p.discard) add_visible_card(counts, cid);
  if (player == perspective && p.handKnown) {
    for (int cid : p.hand) add_visible_card(counts, cid);
  }
  for (int cid : p.handKnownCards) add_visible_card(counts, cid);
  if (p.deckKnown || !p.deckKnownMask.empty()) {
    for (size_t i = 0; i < p.deck.size(); ++i) {
      bool known = p.deckKnown ||
                   (i < p.deckKnownMask.size() && p.deckKnownMask[i]);
      if (known) add_visible_card(counts, p.deck[i]);
    }
  }
  for (int cid : p.deckKnownCards) add_visible_card(counts, cid);
  if (p.prizesKnown || !p.prizesKnownMask.empty()) {
    for (size_t i = 0; i < p.prizes.size(); ++i) {
      bool known = p.prizesKnown ||
                   (i < p.prizesKnownMask.size() && p.prizesKnownMask[i]);
      if (known) add_visible_card(counts, p.prizes[i]);
    }
  }
  for (int cid : p.prizesKnownCards) add_visible_card(counts, cid);
  if (st.stadiumOwner == player) {
    for (int cid : st.stadium) add_visible_card(counts, cid);
  }
  return counts;
}

static std::vector<int> hidden_pool_from_decklist(
    const std::vector<int>& decklist, const std::map<int, int>& visible) {
  std::map<int, int> remaining;
  for (int cid : decklist) {
    if (cid > 0) ++remaining[cid];
  }
  for (const auto& kv : visible) {
    auto it = remaining.find(kv.first);
    if (it != remaining.end()) it->second -= kv.second;
  }

  std::vector<int> pool;
  pool.reserve(decklist.size());
  for (const auto& kv : remaining) {
    for (int i = 0; i < kv.second; ++i) pool.push_back(kv.first);
  }
  return pool;
}

static std::vector<int> fill_known_slots(
    const std::vector<int>& current, const std::vector<bool>& mask,
    bool all_known, int count, const std::vector<int>& known_members,
    std::vector<int>& pool, size_t& pos, std::mt19937_64& rng) {
  std::vector<int> out;
  out.reserve(static_cast<size_t>(std::max(0, count)));
  int unknown_slots = 0;
  for (int i = 0; i < count; ++i) {
    bool known = all_known || (i < static_cast<int>(mask.size()) && mask[i]);
    if (!known) ++unknown_slots;
  }
  std::vector<int> unknown_cards;
  unknown_cards.reserve(static_cast<size_t>(std::max(0, unknown_slots)));
  for (int cid : known_members)
    if (cid > 0) unknown_cards.push_back(cid);
  while (static_cast<int>(unknown_cards.size()) < unknown_slots) {
    if (pos < pool.size())
      unknown_cards.push_back(pool[pos++]);
    else
      unknown_cards.push_back(1);
  }
  std::shuffle(unknown_cards.begin(), unknown_cards.end(), rng);
  size_t unknown_pos = 0;
  for (int i = 0; i < count; ++i) {
    bool known = all_known || (i < static_cast<int>(mask.size()) && mask[i]);
    if (known && i < static_cast<int>(current.size()) && current[i] > 0) {
      out.push_back(current[i]);
    } else if (unknown_pos < unknown_cards.size()) {
      out.push_back(unknown_cards[unknown_pos++]);
    } else {
      out.push_back(1);
    }
  }
  return out;
}

static void determinize_player_hidden_zones(
    GameState& out, const GameState& src, int player,
    const std::vector<int>& decklist, int perspective, std::mt19937_64& rng) {
  const Player& sp = src.players[player];
  Player& dp = out.players[player];
  const bool keep_hand = (player == perspective && sp.handKnown);
  auto visible = visible_card_counts_for_player(src, player, perspective);
  auto pool = hidden_pool_from_decklist(decklist, visible);
  const int known_hand_members = keep_hand ? 0 : static_cast<int>(sp.handKnownCards.size());
  const int hidden_hand_count = keep_hand ? 0 : std::max(0, sp.handCount - known_hand_members);
  const bool keep_deck = sp.deckKnown &&
                         static_cast<int>(sp.deck.size()) == sp.deckCount;
  const bool keep_prizes = sp.prizesKnown &&
                           static_cast<int>(sp.prizes.size()) == sp.prizeCount;
  const bool partial_deck = !keep_deck && !sp.deckKnownMask.empty() &&
                            static_cast<int>(sp.deck.size()) == sp.deckCount;
  const bool partial_prizes = !keep_prizes && !sp.prizesKnownMask.empty() &&
                              static_cast<int>(sp.prizes.size()) == sp.prizeCount;
  const bool infer_prizes = !keep_prizes && keep_deck && keep_hand &&
                            static_cast<int>(pool.size()) == sp.prizeCount;
  const int known_deck_slots = partial_deck ? static_cast<int>(
      std::count(sp.deckKnownMask.begin(), sp.deckKnownMask.end(), true)) : 0;
  const int known_prize_slots = partial_prizes ? static_cast<int>(
      std::count(sp.prizesKnownMask.begin(), sp.prizesKnownMask.end(), true)) : 0;
  const int known_deck_members = keep_deck ? 0 : static_cast<int>(sp.deckKnownCards.size());
  const int known_prize_members = keep_prizes ? 0 : static_cast<int>(sp.prizesKnownCards.size());
  const int hidden_deck_count = keep_deck ? 0 : std::max(0, sp.deckCount - known_deck_slots - known_deck_members);
  const int hidden_prize_count = (keep_prizes || infer_prizes) ? 0
      : std::max(0, sp.prizeCount - known_prize_slots - known_prize_members);
  const int needed = hidden_hand_count + hidden_prize_count + hidden_deck_count;
  if (static_cast<int>(pool.size()) < needed) pool.resize(needed, 1);
  std::shuffle(pool.begin(), pool.end(), rng);

  size_t pos = 0;
  if (keep_hand) {
    dp.hand = sp.hand;
    dp.handKnown = true;
  } else {
    dp.hand = fill_known_slots({}, {}, false, sp.handCount, sp.handKnownCards,
                               pool, pos, rng);
    dp.handKnown = true;
  }
  dp.handCount = static_cast<int>(dp.hand.size());
  if (keep_prizes) {
    dp.prizes = sp.prizes;
    dp.prizesKnownMask.assign(dp.prizes.size(), true);
    dp.prizesKnown = true;
  } else if (infer_prizes) {
    dp.prizes = pool;
    dp.prizesKnownMask.assign(dp.prizes.size(), true);
    dp.prizesKnown = true;
    pos = pool.size();
  } else {
    dp.prizes = fill_known_slots(sp.prizes, sp.prizesKnownMask, false,
                                 sp.prizeCount, sp.prizesKnownCards,
                                 pool, pos, rng);
    dp.prizesKnownMask = sp.prizesKnownMask;
    if (dp.prizesKnownMask.size() < dp.prizes.size())
      dp.prizesKnownMask.resize(dp.prizes.size(), false);
    dp.prizesKnown = false;
  }
  dp.prizeCount = static_cast<int>(dp.prizes.size());
  dp.prizeFaceUp = sp.prizeFaceUp;
  if (dp.prizeFaceUp.size() < static_cast<size_t>(dp.prizeCount))
    dp.prizeFaceUp.resize(dp.prizeCount, false);
  if (keep_deck) {
    dp.deck = sp.deck;
    dp.deckKnownMask.assign(dp.deck.size(), true);
    dp.deckKnown = true;
  } else {
    dp.deck = fill_known_slots(sp.deck, sp.deckKnownMask, false,
                               sp.deckCount, sp.deckKnownCards,
                               pool, pos, rng);
    dp.deckKnownMask = sp.deckKnownMask;
    if (dp.deckKnownMask.size() < dp.deck.size())
      dp.deckKnownMask.resize(dp.deck.size(), false);
    dp.deckKnown = false;
  }
  dp.deckCount = static_cast<int>(dp.deck.size());
}

static GameState rl_determinize_decklist(
    const GameState& state, const std::vector<int>& deck0,
    const std::vector<int>& deck1, int perspective, uint64_t seed) {
  if (perspective < 0) perspective = state.yourIndex;
  if (perspective != 0 && perspective != 1)
    throw std::runtime_error("perspective must be 0, 1, or -1");

  GameState out = state;
  std::mt19937_64 rng(seed ? seed : 1);
  determinize_player_hidden_zones(out, state, 0, deck0, perspective, rng);
  determinize_player_hidden_zones(out, state, 1, deck1, perspective, rng);
  out.rng = seed ? seed : state.rng;
  out.freeRunning = true;
  return out;
}

static py::object atom_to_py(const Atom& a) {
  if (a.is_none) return py::none();
  if (a.is_str) return py::cast(a.sym ? a.sym : "");
  return py::cast(a.i);
}

static py::tuple descriptor_to_py(const Descriptor& desc) {
  py::list t;
  for (const auto& a : desc)
    t.append(atom_to_py(a));
  return py::tuple(t);
}

static py::list descriptors_to_py(const std::vector<Descriptor>& descriptors) {
  py::list out;
  for (const auto& desc : descriptors)
    out.append(descriptor_to_py(desc));
  return out;
}

static py::array_t<double> vector_to_array(const std::vector<double>& values) {
  py::array_t<double> arr({static_cast<py::ssize_t>(values.size())});
  std::copy(values.begin(), values.end(), arr.mutable_data());
  return arr;
}

static py::list legal_main_py(const GameState& st) {
  return descriptors_to_py(legal_main(st));
}

static py::object pending_decision(const GameState& st) {
  if (!st.has_pending()) return py::none();
  py::dict d;
  d["context"] = st.pending.context;
  d["minCount"] = st.pending.minCount;
  d["maxCount"] = st.pending.maxCount;
  d["options"] = descriptors_to_py(st.pending.options);
  return d;
}

// --- CABT/cg observation serialization -----------------------------------
static constexpr int CG_AREA_DECK = 1;
static constexpr int CG_AREA_HAND = 2;
static constexpr int CG_AREA_DISCARD = 3;
static constexpr int CG_AREA_ACTIVE = 4;
static constexpr int CG_AREA_BENCH = 5;
static constexpr int CG_AREA_PRIZE = 6;
static constexpr int CG_AREA_STADIUM = 7;
static constexpr int CG_AREA_ENERGY = 8;
static constexpr int CG_AREA_TOOL = 9;
static constexpr int CG_AREA_PRE_EVOLUTION = 10;
static constexpr int CG_AREA_PLAYER = 11;
static constexpr int CG_AREA_LOOKING = 12;
static constexpr int CG_OPTION_NUMBER = 0;
static constexpr int CG_OPTION_YES = 1;
static constexpr int CG_OPTION_NO = 2;
static constexpr int CG_OPTION_CARD = 3;
static constexpr int CG_OPTION_TOOL_CARD = 4;
static constexpr int CG_OPTION_ENERGY_CARD = 5;
static constexpr int CG_OPTION_ENERGY = 6;
static constexpr int CG_OPTION_PLAY = 7;
static constexpr int CG_OPTION_ATTACH = 8;
static constexpr int CG_OPTION_EVOLVE = 9;
static constexpr int CG_OPTION_ABILITY = 10;
static constexpr int CG_OPTION_DISCARD = 11;
static constexpr int CG_OPTION_RETREAT = 12;
static constexpr int CG_OPTION_ATTACK = 13;
static constexpr int CG_OPTION_END = 14;
static constexpr int CG_OPTION_SKILL = 15;
static constexpr int CG_OPTION_SPECIAL_CONDITION = 16;
static constexpr int CG_SELECT_MAIN = 0;
static constexpr int CG_SELECT_CARD = 1;
static constexpr int CG_SELECT_ATTACHED_CARD = 2;
static constexpr int CG_SELECT_CARD_OR_ATTACHED_CARD = 3;
static constexpr int CG_SELECT_ENERGY = 4;
static constexpr int CG_SELECT_SKILL = 5;
static constexpr int CG_SELECT_ATTACK = 6;
static constexpr int CG_SELECT_EVOLVE = 7;
static constexpr int CG_SELECT_COUNT = 8;
static constexpr int CG_SELECT_YES_NO = 9;
static constexpr int CG_SELECT_SPECIAL_CONDITION = 10;
static constexpr int CG_CONTEXT_MAIN = 0;
static constexpr int CG_CONTEXT_DISCARD_ENERGY_CARD = 26;
static constexpr int CG_CONTEXT_DISCARD_TOOL_CARD = 27;
static constexpr int CG_CONTEXT_SWITCH_ENERGY_CARD = 28;
static constexpr int CG_CONTEXT_DISCARD_CARD_OR_ATTACHED_CARD = 29;

static int cg_area_code(std::string_view area) {
  if (area == "DECK") return CG_AREA_DECK;
  if (area == "HAND") return CG_AREA_HAND;
  if (area == "DISCARD") return CG_AREA_DISCARD;
  if (area == "ACTIVE") return CG_AREA_ACTIVE;
  if (area == "BENCH") return CG_AREA_BENCH;
  if (area == "PRIZE") return CG_AREA_PRIZE;
  if (area == "STADIUM") return CG_AREA_STADIUM;
  if (area == "ENERGY") return CG_AREA_ENERGY;
  if (area == "TOOL") return CG_AREA_TOOL;
  if (area == "PRE_EVOLUTION") return CG_AREA_PRE_EVOLUTION;
  if (area == "PLAYER") return CG_AREA_PLAYER;
  if (area == "LOOKING") return CG_AREA_LOOKING;
  return CG_AREA_HAND;
}

static int cg_area_index(std::string_view area, int index) {
  if (area == "ACTIVE" || area == "STADIUM") return 0;
  return index;
}

static int cg_energy_ref_inplay(int ref) {
  return ref / 1000 - 1;
}

static int cg_energy_ref_index(int ref) {
  return ref % 1000;
}

static const InPlay* cg_inplay_at(const GameState& st, int owner,
                                  int inplayIdx) {
  if (owner < 0 || owner >= 2) return nullptr;
  const Player& p = st.players[owner];
  if (inplayIdx < 0) return p.activeKnown ? &p.active : nullptr;
  return inplayIdx < static_cast<int>(p.bench.size()) ? &p.bench[inplayIdx]
                                                      : nullptr;
}

static int cg_attached_energy_slots(const InPlay& p) {
  return std::max(static_cast<int>(p.energies.size()),
                  static_cast<int>(p.energyCardIds.size()));
}

static int cg_attached_energy_card_id(const InPlay& p, int energyIdx) {
  return energyIdx >= 0 && energyIdx < static_cast<int>(p.energyCardIds.size())
             ? p.energyCardIds[energyIdx]
             : 0;
}

static bool cg_energy_slot_matches(const GameState& st, int owner,
                                   int inplayIdx, int energyIdx, int cardId) {
  const InPlay* p = cg_inplay_at(st, owner, inplayIdx);
  if (!p || energyIdx < 0 || energyIdx >= cg_attached_energy_slots(*p))
    return false;
  int attachedId = cg_attached_energy_card_id(*p, energyIdx);
  return cardId <= 0 || attachedId <= 0 || attachedId == cardId;
}

static int cg_infer_energy_owner(const GameState& st, int fallback,
                                 int inplayIdx, int energyIdx, int cardId) {
  int match = -1;
  for (int side = 0; side < 2; ++side) {
    if (!cg_energy_slot_matches(st, side, inplayIdx, energyIdx, cardId))
      continue;
    if (match >= 0) return fallback;
    match = side;
  }
  return match >= 0 ? match : fallback;
}

static int cg_energy_count(const GameState& st, int owner, int inplayIdx,
                           int energyIdx) {
  const InPlay* p = cg_inplay_at(st, owner, inplayIdx);
  if (!p) return 1;
  return std::max(1, provided_energy_units_for_card(*p, energyIdx, &st, owner));
}

static bool cg_decode_tool_ref(int ref, int me, int& owner, int& inplayIdx,
                               int& toolIdx) {
  if (ref >= 300000) {
    int local = (ref - 300000) % 100000;
    int ownerOffset = (ref - 300000) / 100000;
    owner = ownerOffset == 0 ? me : 1 - me;
    inplayIdx = local / 1000 - 1;
    toolIdx = local % 1000;
    return true;
  }
  if (ref >= 100000 && ref < 200000) {
    int local = ref - 100000;
    owner = 1 - me;
    inplayIdx = local / 1000 - 1;
    toolIdx = local % 1000;
    return true;
  }
  return false;
}

static int cg_infer_card_owner(const GameState& st, int fallback,
                               std::string_view area, int index, int cardId) {
  if (area == "STADIUM")
    return st.stadiumOwner >= 0 ? st.stadiumOwner : fallback;
  if (area != "ACTIVE" && area != "BENCH" && area != "PRIZE")
    return fallback;

  int match = -1;
  for (int side = 0; side < 2; ++side) {
    const Player& p = st.players[side];
    int cid = 0;
    if (area == "ACTIVE") {
      if (p.activeKnown) cid = p.active.id;
    } else if (area == "BENCH") {
      if (index >= 0 && index < static_cast<int>(p.bench.size()))
        cid = p.bench[index].id;
    } else if (area == "PRIZE") {
      if (index >= 0 && index < static_cast<int>(p.prizes.size()))
        cid = p.prizes[index];
    }
    if (cid <= 0 || (cardId > 0 && cid != cardId)) continue;
    if (match >= 0) return fallback;
    match = side;
  }
  return match >= 0 ? match : fallback;
}

static py::dict cg_card(int cid, int player, int serial = 0) {
  py::dict d;
  d["id"] = cid;
  d["name"] = "";
  if (const CardInfo* ci = find_card(cid))
    d["name"] = ci->name;
  d["playerIndex"] = player;
  d["serial"] = serial;
  int hp = 0;
  if (const CardInfo* ci = find_card(cid))
    hp = ci->hp;
  d["hp"] = hp;
  d["maxHp"] = hp;
  d["energies"] = py::list();
  d["energyCards"] = py::list();
  d["tools"] = py::list();
  d["preEvolution"] = py::list();
  d["appearThisTurn"] = false;
  return d;
}

static py::dict cg_pokemon(const GameState& st, const InPlay& p, int player,
                           int serial = 0) {
  py::dict d = cg_card(p.id, player, serial);
  d["hp"] = p.hp;
  d["maxHp"] = p.maxHp;
  d["appearThisTurn"] = p.appearThisTurn;
  py::list energy_cards;
  py::list energies;
  for (int etype : provided_energy_units(p, &st, player))
    energies.append(etype);
  d["energies"] = energies;
  for (size_t i = 0; i < p.energyCardIds.size(); ++i)
    energy_cards.append(cg_card(p.energyCardIds[i], player,
                                static_cast<int>(i)));
  d["energyCards"] = energy_cards;
  py::list tools;
  for (size_t i = 0; i < p.tools.size(); ++i)
    tools.append(cg_card(p.tools[i], player, static_cast<int>(i)));
  d["tools"] = tools;
  py::list pre_evo;
  for (size_t i = 0; i < p.preEvo.size(); ++i)
    pre_evo.append(cg_card(p.preEvo[i], player, static_cast<int>(i)));
  d["preEvolution"] = pre_evo;
  return d;
}

static py::dict cg_player(const GameState& st, int side, bool reveal_hand) {
  const Player& p = st.players[side];
  py::dict d;
  py::list active;
  if (p.activePresent) {
    if (p.activeKnown)
      active.append(cg_pokemon(st, p.active, side, p.active.serial));
    else
      active.append(py::none());
  }
  d["active"] = active;
  py::list bench;
  for (size_t i = 0; i < p.bench.size(); ++i)
    bench.append(cg_pokemon(st, p.bench[i], side, p.bench[i].serial));
  d["bench"] = bench;
  d["benchMax"] = p.benchMax;
  d["deck"] = py::list();
  d["deckCount"] = p.deckCount;
  py::list discard;
  for (size_t i = 0; i < p.discard.size(); ++i)
    discard.append(cg_card(p.discard[i], side, static_cast<int>(i)));
  d["discard"] = discard;
  py::list prize;
  for (size_t i = 0; i < p.prizes.size(); ++i) {
    bool face_up = i < p.prizeFaceUp.size() && p.prizeFaceUp[i];
    bool known = p.prizesKnown ||
                 (i < p.prizesKnownMask.size() && p.prizesKnownMask[i]) ||
                 face_up;
    if (known)
      prize.append(cg_card(p.prizes[i], side, static_cast<int>(i)));
    else
      prize.append(py::none());
  }
  while (static_cast<int>(py::len(prize)) < p.prizeCount)
    prize.append(py::none());
  d["prize"] = prize;
  py::list hand;
  if (reveal_hand && p.handKnown) {
    for (size_t i = 0; i < p.hand.size(); ++i)
      hand.append(cg_card(p.hand[i], side, static_cast<int>(i)));
    d["hand"] = hand;
  } else {
    d["hand"] = py::none();
  }
  d["handCount"] = p.handCount;
  d["poisoned"] = p.poisoned;
  d["burned"] = p.burned;
  d["asleep"] = p.asleep;
  d["paralyzed"] = p.paralyzed;
  d["confused"] = p.confused;
  return d;
}

struct CgHandIndexResolver {
  std::map<int, std::vector<int>> by_cid;
  std::map<int, int> cursor;
  std::map<int, std::vector<Descriptor>> seen;

  explicit CgHandIndexResolver(const Player& p) {
    for (size_t i = 0; i < p.hand.size(); ++i)
      by_cid[p.hand[i]].push_back(static_cast<int>(i));
  }

  int index_for(const Descriptor& desc, int cid) {
    auto it = by_cid.find(cid);
    if (it == by_cid.end() || it->second.empty()) return 0;
    auto& seen_for_card = seen[cid];
    bool already_seen = false;
    for (const Descriptor& old : seen_for_card) {
      if (descriptor_equal(old, desc)) {
        already_seen = true;
        break;
      }
    }
    int& cur = cursor[cid];
    if (already_seen && cur + 1 < static_cast<int>(it->second.size())) {
      ++cur;
      seen_for_card.clear();
    }
    seen_for_card.push_back(desc);
    return it->second[cur];
  }
};

static py::dict cg_option(const GameState& st, const Descriptor& d,
                          CgHandIndexResolver& hand, int selectType) {
  py::dict out;
  const int me = st.yourIndex;
  const std::string_view kind =
      d.empty() ? std::string_view("END") : atom_string(d, 0);
  if (kind == "END") {
    out["type"] = CG_OPTION_END;
  } else if (kind == "PLAY") {
    out["type"] = CG_OPTION_PLAY;
    out["index"] = hand.index_for(d, atom_int(d, 1));
  } else if (kind == "SETUP_ACTIVE") {
    out["type"] = CG_OPTION_CARD;
    out["area"] = CG_AREA_HAND;
    out["index"] = hand.index_for(d, atom_int(d, 1));
    out["playerIndex"] = me;
  } else if (kind == "ATTACH") {
    out["type"] = CG_OPTION_ATTACH;
    out["area"] = CG_AREA_HAND;
    out["index"] = hand.index_for(d, atom_int(d, 1));
    out["inPlayArea"] = cg_area_code(atom_string(d, 2));
    out["inPlayIndex"] = cg_area_index(atom_string(d, 2), atom_int(d, 3));
  } else if (kind == "EVOLVE") {
    out["type"] = CG_OPTION_EVOLVE;
    out["area"] = CG_AREA_HAND;
    out["index"] = hand.index_for(d, atom_int(d, 1));
    out["inPlayArea"] = cg_area_code(atom_string(d, 2));
    out["inPlayIndex"] = cg_area_index(atom_string(d, 2), atom_int(d, 3));
  } else if (kind == "ATTACK") {
    out["type"] = CG_OPTION_ATTACK;
    out["attackId"] = atom_int(d, 1);
  } else if (kind == "RETREAT") {
    out["type"] = CG_OPTION_RETREAT;
  } else if (kind == "ABILITY") {
    out["type"] = CG_OPTION_ABILITY;
    out["area"] = cg_area_code(atom_string(d, 1));
    out["index"] = cg_area_index(atom_string(d, 1), atom_int(d, 2));
  } else if (kind == "DISCARD") {
    out["type"] = CG_OPTION_DISCARD;
    out["area"] = cg_area_code(atom_string(d, 1));
    out["index"] = cg_area_index(atom_string(d, 1), atom_int(d, 2));
  } else if (kind == "CARD") {
    std::string_view area = atom_string(d, 1);
    int rawIndex = atom_int(d, 2);
    int cardId = atom_int(d, d.empty() ? 0 : d.size() - 1);
    int owner = me;
    int inplayIdx = 0;
    int toolIdx = 0;
    if (cg_decode_tool_ref(rawIndex, me, owner, inplayIdx, toolIdx)) {
      out["type"] = CG_OPTION_TOOL_CARD;
      out["area"] = cg_area_code(area);
      out["index"] = cg_area_index(area, inplayIdx);
      out["playerIndex"] = owner;
      out["toolIndex"] = toolIdx;
    } else {
      out["type"] = CG_OPTION_CARD;
      out["area"] = cg_area_code(area);
      out["index"] = cg_area_index(area, rawIndex);
      out["playerIndex"] = cg_infer_card_owner(st, me, area, rawIndex, cardId);
    }
  } else if (kind == "ENERGY") {
    std::string_view area = atom_string(d, 1);
    int ref = atom_int(d, 2);
    if (ref >= 200000) ref -= 200000;
    int inplayIdx = cg_energy_ref_inplay(ref);
    int energyIdx = cg_energy_ref_index(ref);
    int cardId = atom_int(d, d.empty() ? 0 : d.size() - 1);
    int owner = cg_infer_energy_owner(st, me, inplayIdx, energyIdx, cardId);
    bool attachedCardSelect =
        selectType == CG_SELECT_ATTACHED_CARD ||
        selectType == CG_SELECT_CARD_OR_ATTACHED_CARD;
    out["type"] = attachedCardSelect ? CG_OPTION_ENERGY_CARD : CG_OPTION_ENERGY;
    out["area"] = cg_area_code(area);
    out["index"] = cg_area_index(area, inplayIdx);
    out["playerIndex"] = owner;
    out["energyIndex"] = energyIdx;
    if (!attachedCardSelect)
      out["count"] = cg_energy_count(st, owner, inplayIdx, energyIdx);
  } else if (kind == "YES") {
    out["type"] = CG_OPTION_YES;
  } else if (kind == "NO") {
    out["type"] = CG_OPTION_NO;
  } else if (kind == "COUNT" || kind == "NUMBER") {
    out["type"] = CG_OPTION_NUMBER;
    out["number"] = atom_int(d, d.empty() ? 0 : d.size() - 1);
  } else if (kind == "SKILL") {
    out["type"] = CG_OPTION_SKILL;
    out["cardId"] = atom_int(d, 1);
    out["serial"] = atom_int(d, 2);
  } else if (kind == "SPECIAL_CONDITION") {
    out["type"] = CG_OPTION_SPECIAL_CONDITION;
    out["specialConditionType"] = atom_int(d, 1);
  } else {
    out["type"] = CG_OPTION_END;
  }
  return out;
}

static py::object cg_deck_window(const std::vector<Descriptor>& descriptors,
                                 int player) {
  int max_idx = -1;
  std::map<int, int> by_index;
  for (const Descriptor& d : descriptors) {
    if (d.size() >= 4 && atom_string(d, 0) == "CARD" &&
        atom_string(d, 1) == "DECK") {
      int idx = atom_int(d, 2);
      max_idx = std::max(max_idx, idx);
      by_index[idx] = atom_int(d, 3);
    }
  }
  if (max_idx < 0) return py::none();
  py::list out;
  for (int i = 0; i <= max_idx; ++i) {
    auto it = by_index.find(i);
    if (it == by_index.end())
      out.append(py::none());
    else
      out.append(cg_card(it->second, player, i));
  }
  return out;
}

struct CgActionView {
  int ctx = -1;
  int min_count = 1;
  int max_count = 1;
  std::vector<Descriptor> descriptors;
};

static CgActionView cg_action_view(const GameState& st) {
  CgActionView view;
  if (st.has_pending()) {
    view.ctx = st.pending.context;
    view.descriptors = st.pending.options;
    view.min_count = st.pending.minCount;
    view.max_count = st.pending.maxCount;
  } else {
    view.descriptors = legal_main(st);
  }
  return view;
}

static bool cg_all_descriptors_kind(const CgActionView& view,
                                    std::string_view kind) {
  if (view.descriptors.empty()) return false;
  for (const Descriptor& d : view.descriptors) {
    std::string_view cur = d.empty() ? std::string_view("END") : atom_string(d, 0);
    if (cur != kind) return false;
  }
  return true;
}

static bool cg_any_descriptor_kind(const CgActionView& view,
                                   std::string_view kind) {
  for (const Descriptor& d : view.descriptors) {
    std::string_view cur = d.empty() ? std::string_view("END") : atom_string(d, 0);
    if (cur == kind) return true;
  }
  return false;
}

static int cg_select_type_from_view(const CgActionView& view) {
  if (view.ctx < 0) return CG_SELECT_MAIN;
  if (view.ctx == CG_CONTEXT_DISCARD_CARD_OR_ATTACHED_CARD)
    return CG_SELECT_CARD_OR_ATTACHED_CARD;
  if (view.ctx == CG_CONTEXT_DISCARD_ENERGY_CARD ||
      view.ctx == CG_CONTEXT_DISCARD_TOOL_CARD ||
      view.ctx == CG_CONTEXT_SWITCH_ENERGY_CARD)
    return CG_SELECT_ATTACHED_CARD;
  if (cg_all_descriptors_kind(view, "YES") ||
      cg_all_descriptors_kind(view, "NO") ||
      (cg_any_descriptor_kind(view, "YES") && cg_any_descriptor_kind(view, "NO")))
    return CG_SELECT_YES_NO;
  if (cg_any_descriptor_kind(view, "SPECIAL_CONDITION"))
    return CG_SELECT_SPECIAL_CONDITION;
  if (cg_any_descriptor_kind(view, "COUNT") ||
      cg_any_descriptor_kind(view, "NUMBER"))
    return CG_SELECT_COUNT;
  if (cg_any_descriptor_kind(view, "SKILL")) return CG_SELECT_SKILL;
  if (cg_any_descriptor_kind(view, "ATTACK")) return CG_SELECT_ATTACK;
  if (cg_any_descriptor_kind(view, "EVOLVE")) return CG_SELECT_EVOLVE;
  if (cg_any_descriptor_kind(view, "ENERGY")) return CG_SELECT_ENERGY;
  return CG_SELECT_CARD;
}

static py::dict cg_select_from_view(const GameState& st, const CgActionView& view) {
  py::dict sel;
  int selectType = cg_select_type_from_view(view);
  sel["context"] = view.ctx < 0 ? CG_CONTEXT_MAIN : view.ctx;
  sel["type"] = selectType;
  sel["contextCard"] = py::none();
  sel["minCount"] = view.min_count;
  sel["maxCount"] = view.max_count;
  sel["remainDamageCounter"] = 0;
  sel["remainEnergyCost"] = 0;
  sel["effect"] = py::none();
  py::object deck = cg_deck_window(view.descriptors, st.yourIndex);
  sel["deck"] = deck;
  CgHandIndexResolver hand(st.players[st.yourIndex]);
  py::list options;
  for (const Descriptor& d : view.descriptors)
    options.append(cg_option(st, d, hand, selectType));
  sel["option"] = options;
  return sel;
}

static py::dict action_ids_dict(py::array_t<int32_t> meta,
                                py::array_t<int32_t> options,
                                py::array_t<int32_t> deck,
                                py::array_t<uint8_t> mask) {
  py::dict out;
  out["meta"] = meta;
  out["options"] = options;
  out["deck"] = deck;
  out["mask"] = mask;
  return out;
}

static py::dict rl_action_ids_py(const GameState& st) {
  using Shape = std::vector<py::ssize_t>;
  py::array_t<int32_t> meta(Shape{ACTION_META_WIDTH});
  py::array_t<int32_t> options(Shape{RL_MAX_ACTIONS, ACTION_OPTION_WIDTH});
  py::array_t<int32_t> deck(Shape{STATE_ZONE_SLOTS});
  py::array_t<uint8_t> mask(Shape{RL_MAX_ACTIONS});
  RlOptionSet opts = rl_options(st);
  ptcg::fill_action_ids(st, action_id_view_from_options(opts),
                        meta.mutable_data(), options.mutable_data(),
                        deck.mutable_data(), mask.mutable_data());
  return action_ids_dict(meta, options, deck, mask);
}

static py::dict rl_state_ids_complete_py(const GameState& st) {
  using Shape = std::vector<py::ssize_t>;
  py::dict out = rl_state_ids_py(st);

  py::array_t<int32_t> select_meta(Shape{STATE_SELECT_META_WIDTH});
  py::array_t<int32_t> select_options(
      Shape{RL_MAX_ACTIONS, STATE_SELECT_OPTION_WIDTH});
  py::array_t<int32_t> select_deck(Shape{STATE_ZONE_SLOTS});
  RlOptionSet opts = rl_options(st);
  ptcg::fill_action_ids(st, action_id_view_from_options(opts),
                        select_meta.mutable_data(),
                        select_options.mutable_data(),
                        select_deck.mutable_data(), nullptr);

  out["select_meta"] = select_meta;
  out["select_options"] = select_options;
  out["select_deck"] = select_deck;
  return out;
}

static py::dict cg_state(const GameState& st) {
  py::dict d;
  d["turn"] = st.turn;
  d["turnActionCount"] = st.turnActionCount;
  d["yourIndex"] = st.yourIndex;
  d["firstPlayer"] = st.firstPlayer;
  d["supporterPlayed"] = st.supporterPlayed;
  d["stadiumPlayed"] = st.stadiumPlayed;
  d["energyAttached"] = st.energyAttached;
  d["retreated"] = st.retreated;
  d["result"] = st.result;
  py::list stadium;
  std::vector<int> sorted_stadium = st.stadium;
  std::sort(sorted_stadium.begin(), sorted_stadium.end());
  for (size_t i = 0; i < sorted_stadium.size(); ++i)
    stadium.append(cg_card(sorted_stadium[i], st.stadiumOwner, static_cast<int>(i)));
  d["stadium"] = stadium;
  d["looking"] = py::none();
  py::list players;
  players.append(cg_player(st, 0, st.yourIndex == 0));
  players.append(cg_player(st, 1, st.yourIndex == 1));
  d["players"] = players;
  return d;
}

static py::dict cg_observation(const GameState& st) {
  CgActionView view = cg_action_view(st);
  py::dict obs;
  obs["current"] = cg_state(st);
  obs["select"] = cg_select_from_view(st, view);
  obs["logs"] = py::list();
  obs["search_begin_input"] = py::none();
  return obs;
}

static py::tuple cg_observation_with_view(const GameState& st) {
  CgActionView view = cg_action_view(st);
  py::dict obs;
  obs["current"] = cg_state(st);
  obs["select"] = cg_select_from_view(st, view);
  obs["logs"] = py::list();
  obs["search_begin_input"] = py::none();
  return py::make_tuple(obs, view.ctx, descriptors_to_py(view.descriptors));
}

static Action parse_action(const py::tuple& t) {
  Action a;
  std::string kind = t[0].cast<std::string>();
  if (kind == "END") {
    a.kind = ACT_END;
  } else if (kind == "ATTACH") {
    a.kind = ACT_ATTACH;
    a.cardId = t[1].cast<int>();
    a.targetArea = t[2].cast<std::string>() == "ACTIVE" ? AREA_ACTIVE : AREA_BENCH;
    a.targetIndex = t[3].cast<int>();
  } else if (kind == "PLAY") {
    int cid = t[1].cast<int>();
    const CardInfo* ci = find_card(cid);
    a.kind = (ci && ci->cardType == POKEMON && ci->basic) ? ACT_PLAY_BASIC
                                                          : ACT_PLAY_TRAINER;
    a.cardId = cid;
  } else if (kind == "EVOLVE") {
    a.kind = ACT_EVOLVE;
    a.cardId = t[1].cast<int>();
    a.targetArea = t[2].cast<std::string>() == "ACTIVE" ? AREA_ACTIVE : AREA_BENCH;
    a.targetIndex = t[3].cast<int>();
  } else if (kind == "ATTACK") {
    a.kind = ACT_ATTACK;
    a.attackId = t[1].cast<int>();
  } else if (kind == "RETREAT") {
    a.kind = ACT_RETREAT;
  } else if (kind == "ABILITY") {
    a.kind = ACT_ABILITY;
    std::string ar = t[1].cast<std::string>();
    a.targetArea = ar == "ACTIVE" ? AREA_ACTIVE : ar == "STADIUM" ? AREA_STADIUM
                                                                  : AREA_BENCH;
    a.targetIndex = t[2].cast<int>();
  } else if (kind == "DISCARD") {
    a.kind = ACT_DISCARD_INPLAY;
    a.targetArea = t[1].cast<std::string>() == "ACTIVE" ? AREA_ACTIVE : AREA_BENCH;
    a.targetIndex = t[2].cast<int>();
  } else if (kind == "SETUP_ACTIVE") {
    a.kind = ACT_SETUP_ACTIVE;
    a.cardId = t[1].cast<int>();
  } else {
    throw std::runtime_error("unsupported action: " + kind);
  }
  return a;
}

static py::tuple cg_select_step(GameState& st, const std::vector<int>& select) {
  st.collectLogs = true;  // cg bridge consumes per-step native logs
  st.nativeLogs.clear();
  if (st.has_pending()) {
    resolve(st, select);
  } else {
    if (select.size() != 1)
      throw std::runtime_error(
          "Must be Observation.select.minCount <= len(select) <= "
          "Observation.select.maxCount.");
    std::vector<Descriptor> descriptors = legal_main(st);
    int action = select[0];
    if (action < 0 || action >= static_cast<int>(descriptors.size()))
      throw std::out_of_range("action index out of range");
    apply(st, parse_action(descriptor_to_py(descriptors[action])));
  }

  CgActionView view = cg_action_view(st);
  py::dict obs;
  obs["current"] = cg_state(st);
  obs["select"] = cg_select_from_view(st, view);
  obs["logs"] = py::list();
  obs["search_begin_input"] = py::none();
  return py::make_tuple(
      obs,
      view.ctx,
      descriptors_to_py(view.descriptors),
      native_logs_to_py(st));
}

class NativeCgBattle {
 public:
  GameState state;
  uint64_t seed = 1;
  uint64_t generation = 0;
  bool active = false;

  py::tuple start(const std::vector<int>& deck0, const std::vector<int>& deck1,
                  uint64_t start_seed) {
    seed = start_seed;
    generation = 0;
    state = ptcg::new_game(deck0, deck1, seed, /*collectLogs=*/true);
    active = true;
    CgActionView view = cg_action_view(state);
    py::dict obs;
    obs["current"] = cg_state(state);
    obs["select"] = cg_select_from_view(state, view);
    obs["logs"] = py::list();
    obs["search_begin_input"] = py::none();
    return py::make_tuple(
        obs,
        view.ctx,
        descriptors_to_py(view.descriptors),
        py::list());
  }

  py::tuple observation() const {
    if (!active) throw std::runtime_error("battle has not started");
    CgActionView view = cg_action_view(state);
    py::dict obs;
    obs["current"] = cg_state(state);
    obs["select"] = cg_select_from_view(state, view);
    obs["logs"] = py::list();
    obs["search_begin_input"] = py::none();
    return py::make_tuple(
        obs,
        view.ctx,
        descriptors_to_py(view.descriptors),
        py::list());
  }

  py::tuple select(const std::vector<int>& selection) {
    if (!active) throw std::runtime_error("battle has not started");
    state.collectLogs = true;  // cg bridge consumes per-step native logs
    state.nativeLogs.clear();
    if (state.has_pending()) {
      resolve(state, selection);
    } else {
      if (selection.size() != 1)
        throw std::runtime_error(
            "Must be Observation.select.minCount <= len(select) <= "
            "Observation.select.maxCount.");
      std::vector<Descriptor> descriptors = legal_main(state);
      int action = selection[0];
      if (action < 0 || action >= static_cast<int>(descriptors.size()))
        throw std::out_of_range("action index out of range");
      apply(state, parse_action(descriptor_to_py(descriptors[action])));
    }
    seed = (seed * 6364136223846793005ULL + 1442695040888963407ULL);
    ++generation;

    CgActionView view = cg_action_view(state);
    py::dict obs;
    obs["current"] = cg_state(state);
    obs["select"] = cg_select_from_view(state, view);
    obs["logs"] = py::list();
    obs["search_begin_input"] = py::none();
    return py::make_tuple(
        obs,
        view.ctx,
        descriptors_to_py(view.descriptors),
        native_logs_to_py(state));
  }

  void finish() {
    active = false;
    ++generation;
  }

  GameState& mutable_state() { return state; }
  const GameState& readonly_state() const { return state; }
  GameState clone_state() const { return state; }
  py::object transient_snapshot() const { return search_transients_to_py(state); }
  py::object canonical_state() const { return canonical(state); }
};

struct PuctResult {
  bool terminal = false;
  int n_act = 0;
  std::vector<double> childN;
  std::vector<double> childW;
  GameState state;
  RlOptionSet step_opts;
  py::object canon = py::none();
  py::object opts = py::none();
  int ctx = -1;
};

struct PuctNode {
  GameState state;
  int mover = 0;
  bool terminal = false;
  bool expanded = false;
  int n_act = 0;
  double totalN = 0.0;
  RlOptionSet step_opts;
  std::vector<Descriptor> descriptors;
  py::object canon = py::none();
  py::list opts;
  int ctx = -1;
  bool py_ready = false;
  bool root_noised = false;
  std::vector<double> P;
  std::vector<double> childN;
  std::vector<double> childW;
  std::vector<int> child;

  explicit PuctNode(GameState s) : state(std::move(s)) {
    mover = state.yourIndex;
    terminal = state.result >= 0;
  }
};

static py::list action_index_lists(int n) {
  py::list actions;
  for (int i = 0; i < n; ++i) {
    py::list one;
    one.append(i);
    actions.append(one);
  }
  return actions;
}

static void ensure_puct_py_metadata(PuctNode& node) {
  if (node.py_ready) return;
  node.canon = canonical(node.state);
  node.opts = descriptors_to_py(node.descriptors);
  node.py_ready = true;
}

static void prepare_puct_node(PuctNode& node, bool py_metadata = true) {
  if (node.state.has_pending()) {
    node.ctx = node.state.pending.context;
    node.descriptors = node.state.pending.options;
    node.step_opts = rl_options(node.state);
  } else {
    node.descriptors = legal_main(node.state);
    node.ctx = -1;
    node.step_opts = rl_options_from_descriptors(node.descriptors);
  }
  if (static_cast<int>(node.descriptors.size()) > RL_MAX_ACTIONS)
    node.descriptors.resize(RL_MAX_ACTIONS);
  node.n_act = std::min(std::max(static_cast<int>(node.descriptors.size()), 1),
                        RL_MAX_ACTIONS);
  node.py_ready = false;
  node.canon = py::none();
  node.opts = py::list();
  if (py_metadata)
    ensure_puct_py_metadata(node);
  node.childN.assign(node.n_act, 0.0);
  node.childW.assign(node.n_act, 0.0);
  node.child.assign(node.n_act, -1);
}

static py::tuple eval_req(PuctNode& node, const py::object& deck) {
  ensure_puct_py_metadata(node);
  return py::make_tuple(node.canon, node.opts, deck,
                        action_index_lists(node.n_act), node.ctx);
}

static int feature_card_count() {
  static int n = [] {
    int mx = 0;
    for (const auto& c : card_db())
      mx = std::max(mx, c.id);
    return mx + 1;
  }();
  return n;
}

static int feature_attack_count() {
  static int n = [] {
    int mx = 0;
    for (const auto& c : card_db())
      for (int i = 0; i < c.n_attacks; ++i)
        mx = std::max(mx, c.attacks[i].id);
    return mx + 1;
  }();
  return n;
}

static constexpr int FEATURE_NUM_WORDS_ENCODER = 24;
static constexpr int DECODER_MAIN_FEATURE = 8;
static constexpr int DECODER_ATTACK_OFFSET = 14;
static constexpr int SELECT_CONTEXT_RECOVER_SPECIAL_CONDITION = 48;
static constexpr int CARD_TYPE_BLOCK = 7;
static constexpr int STAGE_BLOCK = 4;
static constexpr int ENERGY_TYPE_BLOCK = 12;
static constexpr int WEAKNESS_BLOCK = 13;
static constexpr int RESISTANCE_BLOCK = 13;
static constexpr int HP_BUCKET_BLOCK = 7;
static constexpr int RETREAT_BUCKET_BLOCK = 5;
static constexpr int ATTACK_COUNT_BLOCK = 5;
static constexpr int FLAG_BLOCK = 5;
static constexpr int FLAG_EX = 0;
static constexpr int FLAG_MEGA_EX = 1;
static constexpr int FLAG_TERA = 2;
static constexpr int FLAG_ACE_SPEC = 3;
static constexpr int FLAG_HAS_ABILITY = 4;

static int card_type_offset() { return feature_card_count(); }
static int stage_offset() { return card_type_offset() + CARD_TYPE_BLOCK; }
static int energy_type_offset() { return stage_offset() + STAGE_BLOCK; }
static int weakness_offset() { return energy_type_offset() + ENERGY_TYPE_BLOCK; }
static int resistance_offset() { return weakness_offset() + WEAKNESS_BLOCK; }
static int hp_bucket_offset() { return resistance_offset() + RESISTANCE_BLOCK; }
static int retreat_bucket_offset() { return hp_bucket_offset() + HP_BUCKET_BLOCK; }
static int attack_count_offset() { return retreat_bucket_offset() + RETREAT_BUCKET_BLOCK; }
static int flag_offset() { return attack_count_offset() + ATTACK_COUNT_BLOCK; }
static int card_feature_size() { return flag_offset() + FLAG_BLOCK; }

static int decoder_card_offset() {
  return DECODER_ATTACK_OFFSET + feature_attack_count();
}

static int decoder_pos_offset() {
  return decoder_card_offset() +
         (1 + DECODER_MAIN_FEATURE + SELECT_CONTEXT_RECOVER_SPECIAL_CONDITION) *
             card_feature_size();
}

struct SparseBuilder {
  std::vector<int32_t> index;
  std::vector<float> value;
  std::vector<int32_t> offset;
  int pos = 0;

  void reset() {
    index.clear();
    value.clear();
    offset.clear();
    pos = 0;
  }

  void add(int idx, float v) {
    if (v != 0.0f) {
      index.push_back(static_cast<int32_t>(pos + idx));
      value.push_back(v);
    }
  }
  void add_pos(int n) { pos += n; }
  void add_single(float v) {
    if (v != 0.0f) {
      index.push_back(static_cast<int32_t>(pos));
      value.push_back(v);
    }
    ++pos;
  }
  void word_start() { offset.push_back(static_cast<int32_t>(index.size())); }
};

static int basic_energy_cid(int etype) {
  return (etype >= 1 && etype <= 8) ? etype : -1;
}

static int energy_index(int value) {
  return value < 0 ? 0 : std::min(value + 1, 12);
}

static int hp_bucket(int hp) {
  if (hp <= 0) return 0;
  if (hp <= 70) return 1;
  if (hp <= 120) return 2;
  if (hp <= 180) return 3;
  if (hp <= 240) return 4;
  if (hp <= 300) return 5;
  return 6;
}

static int retreat_bucket(int retreat) {
  return std::min(std::max(retreat, 0), 4);
}

static int attack_count_bucket(int n) {
  return std::min(std::max(n, 0), 4);
}

static int stage_index(const CardInfo* c) {
  if (!c || c->cardType != POKEMON) return 0;
  if (c->stage2) return 3;
  if (c->stage1) return 2;
  if (c->basic) return 1;
  return 0;
}

static void add_card_features_at(SparseBuilder& sv, int base, int cid,
                                 float value) {
  if (cid <= 0) return;
  int fc = feature_card_count();
  if (cid >= fc) return;
  const CardInfo* c = find_card(cid);
  sv.add(base + cid, value);
  if (!c) return;
  sv.add(base + card_type_offset() +
             std::min(std::max(c->cardType, 0), CARD_TYPE_BLOCK - 1),
         value);
  sv.add(base + stage_offset() + stage_index(c), value);
  sv.add(base + energy_type_offset() +
             std::min(std::max(c->energyType, 0), ENERGY_TYPE_BLOCK - 1),
         value);
  sv.add(base + weakness_offset() + energy_index(c->weakness), value);
  sv.add(base + resistance_offset() + energy_index(c->resistance), value);
  sv.add(base + hp_bucket_offset() + hp_bucket(c->hp), value);
  sv.add(base + retreat_bucket_offset() + retreat_bucket(c->retreat), value);
  sv.add(base + attack_count_offset() + attack_count_bucket(c->n_attacks),
         value);
  if (c->ex) sv.add(base + flag_offset() + FLAG_EX, value);
  if (c->megaEx) sv.add(base + flag_offset() + FLAG_MEGA_EX, value);
  if (c->tera) sv.add(base + flag_offset() + FLAG_TERA, value);
  if (c->aceSpec) sv.add(base + flag_offset() + FLAG_ACE_SPEC, value);
}

static void add_card_id(SparseBuilder& sv, int cid) {
  add_card_features_at(sv, 0, cid, 1.0f);
  sv.add_pos(card_feature_size());
}

static void add_card_ids(SparseBuilder& sv, const auto& ids,
                         float value) {
  for (int cid : ids)
    add_card_features_at(sv, 0, cid, value);
  sv.add_pos(card_feature_size());
}

static void add_pokemon_features(SparseBuilder& sv, const InPlay* p) {
  if (!p) {
    sv.add_single(1.0f);
    sv.add_pos(1 + 3 * card_feature_size());
    return;
  }
  sv.add_single(0.0f);
  sv.add_single(static_cast<float>(p->hp) / 400.0f);
  add_card_id(sv, p->id);
  add_card_ids(sv, p->tools, 1.0f);
  for (int etype : p->energies) {
    int cid = basic_energy_cid(etype);
    if (cid > 0)
      add_card_features_at(sv, 0, cid, 0.5f);
  }
  sv.add_pos(card_feature_size());
}

static void add_player_features(SparseBuilder& sv, const Player& p) {
  sv.add_single(static_cast<float>(p.deckCount) / 60.0f);
  sv.add_single(static_cast<float>(p.discard.size()) / 60.0f);
  sv.add_single(static_cast<float>(p.handCount) / 8.0f);
  sv.add_single(static_cast<float>(p.bench.size()) / 5.0f);
  sv.add(p.prizeCount, 1.0f);
  sv.add_pos(7);
  sv.add_single(p.poisoned ? 1.0f : 0.0f);
  sv.add_single(p.burned ? 1.0f : 0.0f);
  sv.add_single(p.asleep ? 1.0f : 0.0f);
  sv.add_single(p.paralyzed ? 1.0f : 0.0f);
  sv.add_single(p.confused ? 1.0f : 0.0f);
  add_card_ids(sv, p.discard, 0.25f);
}

static void build_encoder_features(SparseBuilder& sv, const GameState& st,
                                   const std::vector<int>& deck) {
  int me = st.yourIndex;
  sv.reset();
  sv.index.reserve(160);
  sv.value.reserve(160);
  sv.offset.reserve(FEATURE_NUM_WORDS_ENCODER);
  for (int i = 0; i < 2; ++i) {
    const Player& ps = st.players[i ^ me];
    for (int j = 0; j < 8; ++j) {
      sv.word_start();
      int pos = sv.pos;
      const InPlay* p = j < static_cast<int>(ps.bench.size()) ? &ps.bench[j]
                                                               : nullptr;
      add_pokemon_features(sv, p);
      if (j != 7)
        sv.pos = pos;
    }
  }
  for (int i = 0; i < 2; ++i) {
    const Player& ps = st.players[i ^ me];
    sv.word_start();
    add_pokemon_features(sv, ps.activeKnown ? &ps.active : nullptr);
  }
  for (int i = 0; i < 2; ++i) {
    sv.word_start();
    add_player_features(sv, st.players[i ^ me]);
  }
  sv.word_start();
  if (st.players[me].handKnown)
    add_card_ids(sv, st.players[me].hand, 0.25f);
  else
    sv.add_pos(card_feature_size());
  sv.word_start();
  for (int cid : deck)
    add_card_features_at(sv, 0, cid, 0.25f);
  sv.add_pos(card_feature_size());
  sv.word_start();
  add_card_ids(sv, st.stadium, 1.0f);
  sv.word_start();
  sv.add_single(1.0f);
  sv.add_single(static_cast<float>(st.turn) / 10.0f);
  sv.add_single(st.firstPlayer == me ? 1.0f : 0.0f);
}

static SparseBuilder encoder_features(const GameState& st,
                                      const std::vector<int>& deck) {
  SparseBuilder sv;
  build_encoder_features(sv, st, deck);
  return sv;
}

static int area_code(std::string_view area) {
  if (area == "DECK") return AREA_DECK;
  if (area == "HAND") return AREA_HAND;
  if (area == "DISCARD") return AREA_DISCARD;
  if (area == "ACTIVE") return AREA_ACTIVE;
  if (area == "BENCH") return AREA_BENCH;
  if (area == "PRIZE") return AREA_PRIZE;
  if (area == "STADIUM") return AREA_STADIUM;
  return -1;
}

static int native_pos_index(int area, int index) {
  int raw = index;
  if (area == AREA_ACTIVE || area == AREA_STADIUM)
    return 0;
  if (area == AREA_BENCH) {
    if (raw >= 300000) raw = (raw - 300000) % 100000;
    if (raw >= 200000) raw -= 200000;
    if (raw >= 100000) raw -= 100000;
    if (raw >= 1000) return std::max(0, raw / 1000 - 1);
  }
  return std::max(0, raw);
}

static void dec_pos(SparseBuilder& sv, int area, int index) {
  if (area < 0) return;
  sv.add(decoder_pos_offset() + (area & 15) * 8 + std::min(index, 7), 1.0f);
}

static void dec_native_pos(SparseBuilder& sv, std::string_view area,
                           int index) {
  int ac = area_code(area);
  if (ac >= 0)
    dec_pos(sv, ac, native_pos_index(ac, index));
}

static void dec_main_id(SparseBuilder& sv, int feat, int cid) {
  add_card_features_at(sv, decoder_card_offset() + feat * card_feature_size(),
                       cid, 1.0f);
}

static void dec_card_id(SparseBuilder& sv, int ctx, int cid) {
  add_card_features_at(
      sv, decoder_card_offset() +
              (DECODER_MAIN_FEATURE + ctx) * card_feature_size(),
      cid, 1.0f);
}

static std::string_view atom_string(const Descriptor& d, size_t i) {
  return (i < d.size() && d[i].is_str) ? atom_sv(d[i]) : std::string_view();
}

static int atom_int(const Descriptor& d, size_t i, int fallback) {
  return (i < d.size() && !d[i].is_str && !d[i].is_none)
             ? static_cast<int>(d[i].i)
             : fallback;
}

static bool atom_int_value(const Atom& a, int& out) {
  if (a.is_str || a.is_none) return false;
  out = static_cast<int>(a.i);
  return true;
}

static int target_cid(const GameState& st, int me, std::string_view area,
                      int index) {
  const Player& p = st.players[me];
  if (area == "ACTIVE")
    return p.activeKnown ? p.active.id : 0;
  if (area == "BENCH")
    return (index >= 0 && index < static_cast<int>(p.bench.size()))
               ? p.bench[index].id
               : 0;
  if (area == "STADIUM")
    return st.stadium.empty() ? 0 : st.stadium[0];
  return 0;
}

static void add_decoder_empty(SparseBuilder& sv) { sv.add(0, 1.0f); }

static void add_decoder_action(SparseBuilder& sv, const GameState& st,
                               const std::vector<Descriptor>& descriptors,
                               int ctx, int action) {
  if (action < 0 || action >= static_cast<int>(descriptors.size())) {
    add_decoder_empty(sv);
    return;
  }
  const Descriptor& d = descriptors[action];
  if (d.empty() || !d[0].is_str) {
    add_decoder_empty(sv);
    return;
  }
  std::string_view kind = atom_sv(d[0]);
  int me = st.yourIndex;
  if (kind == "END") {
    sv.add(1, 1.0f);
  } else if (kind == "YES") {
    sv.add(2, 1.0f);
  } else if (kind == "NO") {
    sv.add(3, 1.0f);
  } else if (kind == "COUNT") {
    int n = atom_int(d, d.empty() ? 0 : d.size() - 1);
    sv.add(9 + std::min(std::max(n, 0), 4), 1.0f);
  } else if (kind == "ATTACK") {
    sv.add(DECODER_ATTACK_OFFSET + atom_int(d, 1), 1.0f);
  } else if (kind == "PLAY" || kind == "SETUP_ACTIVE") {
    dec_main_id(sv, 0, atom_int(d, 1));
  } else if (kind == "ATTACH") {
    int source = atom_int(d, 1);
    std::string_view area = atom_string(d, 2);
    int index = atom_int(d, 3);
    dec_main_id(sv, 1, source);
    dec_main_id(sv, 2, target_cid(st, me, area, index));
    dec_native_pos(sv, area, index);
  } else if (kind == "EVOLVE") {
    int source = atom_int(d, 1);
    std::string_view area = atom_string(d, 2);
    int index = atom_int(d, 3);
    dec_main_id(sv, 3, source);
    dec_main_id(sv, 4, target_cid(st, me, area, index));
    dec_native_pos(sv, area, index);
  } else if (kind == "ABILITY") {
    std::string_view area = atom_string(d, 1);
    int index = atom_int(d, 2);
    dec_main_id(sv, 5, target_cid(st, me, area, index));
    dec_native_pos(sv, area, index);
  } else if (kind == "RETREAT") {
    const Player& p = st.players[me];
    if (p.activeKnown)
      dec_main_id(sv, 7, p.active.id);
    dec_pos(sv, AREA_ACTIVE, 0);
  } else if (kind == "DISCARD") {
    std::string_view area = atom_string(d, 1);
    int index = atom_int(d, 2);
    dec_main_id(sv, 6, target_cid(st, me, area, index));
    dec_native_pos(sv, area, index);
  } else if (kind == "CARD" || kind == "ENERGY") {
    int cid = 0;
    if (!d.empty() && atom_int_value(d.back(), cid))
      dec_card_id(sv, ctx, cid);
    dec_native_pos(sv, atom_string(d, 1), atom_int(d, 2));
  }
}

static void build_decoder_features(SparseBuilder& sv, const GameState& st,
                                   const std::vector<Descriptor>& descriptors,
                                   int ctx, int n_act, int width) {
  sv.reset();
  sv.index.reserve(static_cast<size_t>(width) * 3);
  sv.value.reserve(static_cast<size_t>(width) * 3);
  sv.offset.reserve(width);
  for (int a = 0; a < width; ++a) {
    sv.word_start();
    if (a < n_act)
      add_decoder_action(sv, st, descriptors, ctx, a);
    else
      add_decoder_empty(sv);
  }
}

static SparseBuilder decoder_features(const GameState& st,
                                      const std::vector<Descriptor>& descriptors,
                                      int ctx, int n_act, int width) {
  SparseBuilder sv;
  build_decoder_features(sv, st, descriptors, ctx, n_act, width);
  return sv;
}

template <typename T>
static py::array_t<T> vector_to_numpy(const std::vector<T>& values) {
  py::array_t<T> arr({static_cast<py::ssize_t>(values.size())});
  std::copy(values.begin(), values.end(), arr.mutable_data());
  return arr;
}

struct FeaturePayloadBuilder {
  std::vector<int32_t> enc_i;
  std::vector<float> enc_v;
  std::vector<int32_t> enc_o;
  std::vector<int32_t> dec_i;
  std::vector<float> dec_v;
  std::vector<int32_t> dec_o;
  std::vector<int32_t> nacts;

  void clear() {
    enc_i.clear();
    enc_v.clear();
    enc_o.clear();
    dec_i.clear();
    dec_v.clear();
    dec_o.clear();
    nacts.clear();
  }

  void append_encoder(const SparseBuilder& sv) {
    int32_t base = static_cast<int32_t>(enc_i.size());
    enc_i.insert(enc_i.end(), sv.index.begin(), sv.index.end());
    enc_v.insert(enc_v.end(), sv.value.begin(), sv.value.end());
    for (int32_t o : sv.offset)
      enc_o.push_back(base + o);
  }

  void append_decoder(const SparseBuilder& sv) {
    int32_t base = static_cast<int32_t>(dec_i.size());
    dec_i.insert(dec_i.end(), sv.index.begin(), sv.index.end());
    dec_v.insert(dec_v.end(), sv.value.begin(), sv.value.end());
    for (int32_t o : sv.offset)
      dec_o.push_back(base + o);
  }

  py::tuple to_python() const {
    return py::make_tuple(vector_to_numpy(enc_i), vector_to_numpy(enc_v),
                          vector_to_numpy(enc_o), vector_to_numpy(dec_i),
                          vector_to_numpy(dec_v), vector_to_numpy(dec_o),
                          vector_to_numpy(nacts));
  }
};

struct FeaturePayloadScratch {
  FeaturePayloadBuilder payload;
  SparseBuilder enc;
  SparseBuilder dec;

  void clear() {
    payload.clear();
    enc.reset();
    dec.reset();
  }

  void reserve(size_t rows, int width) {
    payload.nacts.reserve(rows);
    payload.enc_i.reserve(rows * 160);
    payload.enc_v.reserve(rows * 160);
    payload.enc_o.reserve(rows * FEATURE_NUM_WORDS_ENCODER);
    payload.dec_i.reserve(rows * static_cast<size_t>(width) * 3);
    payload.dec_v.reserve(rows * static_cast<size_t>(width) * 3);
    payload.dec_o.reserve(rows * static_cast<size_t>(width));
  }

  void append(const GameState& state,
              const std::vector<Descriptor>& descriptors, int ctx, int n_act,
              int width, const std::vector<int>& deck) {
    build_encoder_features(enc, state, deck);
    build_decoder_features(dec, state, descriptors, ctx, n_act, width);
    payload.append_encoder(enc);
    payload.append_decoder(dec);
    payload.nacts.push_back(static_cast<int32_t>(n_act));
  }

  py::tuple to_python() const { return payload.to_python(); }
};

struct FeatureRecord {
  std::vector<int32_t> enc_i;
  std::vector<float> enc_v;
  std::vector<int32_t> enc_o;
  std::vector<int32_t> dec_i;
  std::vector<float> dec_v;
  std::vector<int32_t> dec_o;
  int n_act = 0;
};

static std::vector<Descriptor> descriptors_from_option_set(
    const GameState& st, const RlOptionSet& opts);

static py::tuple puct_feature_payload(const std::vector<const PuctNode*>& nodes,
                                      const std::vector<int>& deck, int width) {
  width = std::max(width, 1);
  FeaturePayloadScratch batch;
  batch.reserve(nodes.size(), width);
  for (const PuctNode* node : nodes) {
    batch.append(node->state, node->descriptors, node->ctx, node->n_act,
                 width, deck);
  }
  return batch.to_python();
}

static FeatureRecord feature_record_cached(const GameState& state,
                                           const RlOptionSet& opts, int ctx,
                                           int n_act,
                                           const std::vector<int>& deck,
                                           int width) {
  FeatureRecord rec;
  n_act = std::min(std::max(n_act, 1), RL_MAX_ACTIONS);
  width = std::max(width, 1);
  auto descriptors = descriptors_from_option_set(state, opts);
  SparseBuilder enc = encoder_features(state, deck);
  SparseBuilder dec = decoder_features(state, descriptors, ctx, n_act, width);
  rec.enc_i = std::move(enc.index);
  rec.enc_v = std::move(enc.value);
  rec.enc_o = std::move(enc.offset);
  rec.dec_i = std::move(dec.index);
  rec.dec_v = std::move(dec.value);
  rec.dec_o = std::move(dec.offset);
  rec.n_act = n_act;
  return rec;
}

static std::vector<FeatureRecord> feature_records_cached_batch(
    const py::sequence& reqs, const std::vector<int>& deck, int width) {
  std::vector<FeatureRecord> out;
  out.reserve(static_cast<size_t>(py::len(reqs)));
  width = std::max(width, 1);
  for (py::handle obj : reqs) {
    py::tuple t = obj.cast<py::tuple>();
    GameState state = t[0].cast<GameState>();
    const RlOptionSet& opts = t[1].cast<const RlOptionSet&>();
    int ctx = t[2].cast<int>();
    int n_act = t[3].cast<int>();
    out.push_back(feature_record_cached(state, opts, ctx, n_act, deck, width));
  }
  return out;
}

static py::tuple puct_feature_payload_for_indices(
    const std::vector<PuctNode>& tree, const std::vector<int>& node_indices,
    const std::vector<int>& deck) {
  std::vector<const PuctNode*> nodes;
  nodes.reserve(node_indices.size());
  int width = 1;
  for (int idx : node_indices) {
    nodes.push_back(&tree[idx]);
    width = std::max(width, tree[idx].n_act);
  }
  return puct_feature_payload(nodes, deck, width);
}

static py::tuple rl_feature_payload(const GameState& state,
                                    const std::vector<int>& deck, int width) {
  PuctNode node(state);
  if (node.terminal)
    return FeaturePayloadBuilder().to_python();
  prepare_puct_node(node, false);
  std::vector<const PuctNode*> nodes{&node};
  return puct_feature_payload(nodes, deck, width > 0 ? width : node.n_act);
}

static std::vector<Descriptor> descriptors_from_option_set(
    const GameState& st, const RlOptionSet& opts) {
  std::vector<Descriptor> descriptors =
      opts.pending ? st.pending.options : opts.descriptors;
  if (static_cast<int>(descriptors.size()) > RL_MAX_ACTIONS)
    descriptors.resize(RL_MAX_ACTIONS);
  return descriptors;
}

struct PreparedFeatureReq {
  GameState state;
  std::vector<Descriptor> descriptors;
  int ctx = -1;
  int n_act = 0;
};

static py::tuple rl_feature_payload_cached_batch(
    const py::sequence& reqs, const std::vector<int>& deck, int width) {
  std::vector<PreparedFeatureReq> items;
  items.reserve(static_cast<size_t>(py::len(reqs)));
  int batch_width = std::max(width, 0);
  for (py::handle obj : reqs) {
    py::tuple t = obj.cast<py::tuple>();
    PreparedFeatureReq r;
    r.state = t[0].cast<GameState>();
    const RlOptionSet& opts = t[1].cast<const RlOptionSet&>();
    r.ctx = t[2].cast<int>();
    r.n_act = std::min(std::max(t[3].cast<int>(), 1), RL_MAX_ACTIONS);
    r.descriptors = descriptors_from_option_set(r.state, opts);
    batch_width = std::max(batch_width, r.n_act);
    items.push_back(std::move(r));
  }

  batch_width = std::max(batch_width, 1);
  FeaturePayloadScratch batch;
  batch.reserve(items.size(), batch_width);
  for (const auto& r : items) {
    batch.append(r.state, r.descriptors, r.ctx, r.n_act, batch_width, deck);
  }
  return batch.to_python();
}

static void set_puct_priors(PuctNode& node, const py::handle& logits_obj) {
  py::sequence logits = logits_obj.cast<py::sequence>();
  node.P.assign(node.n_act, 0.0);
  double max_logit = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < node.n_act; ++i) {
    double v = logits[i].cast<double>();
    node.P[i] = v;
    if (v > max_logit) max_logit = v;
  }
  double sum = 0.0;
  for (int i = 0; i < node.n_act; ++i) {
    node.P[i] = std::exp(node.P[i] - max_logit);
    sum += node.P[i];
  }
  if (sum > 0.0) {
    for (double& p : node.P) p /= sum;
  } else {
    double u = 1.0 / static_cast<double>(node.n_act);
    std::fill(node.P.begin(), node.P.end(), u);
  }
  node.expanded = true;
}

static void apply_dirichlet_noise(PuctNode& node, double alpha, double eps,
                                  uint64_t seed) {
  if (eps <= 0.0 || node.n_act <= 1) return;
  std::mt19937_64 gen(seed ? seed : 1);
  std::gamma_distribution<double> gamma(alpha, 1.0);
  std::vector<double> noise(node.n_act, 0.0);
  double sum = 0.0;
  for (double& x : noise) {
    x = gamma(gen);
    sum += x;
  }
  if (sum <= 0.0) return;
  for (int i = 0; i < node.n_act; ++i)
    node.P[i] = (1.0 - eps) * node.P[i] + eps * (noise[i] / sum);
}

static int select_puct_action(const PuctNode& node, int me, double c_puct) {
  double c = c_puct * std::sqrt(node.totalN + 1.0);
  bool flip = node.mover != me;
  int best_i = 0;
  double best_u = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < node.n_act; ++i) {
    double n = node.childN[i];
    double q = n > 0.0 ? node.childW[i] / n : 0.0;
    if (flip) q = -q;
    double u = q + c * node.P[i] / (1.0 + n);
    if (u > best_u) {
      best_u = u;
      best_i = i;
    }
  }
  return best_i;
}

static double puct_term_value(const GameState& st, int me) {
  if (st.result == 2) return 0.0;
  return st.result == me ? 1.0 : -1.0;
}

static double virtual_loss_sign(const PuctNode& node, int me) {
  return node.mover == me ? -1.0 : 1.0;
}

struct PendingLeaf {
  int leaf_idx = -1;
  std::vector<std::vector<std::pair<int, int>>> paths;
};

static void add_pending_leaf(std::vector<PendingLeaf>& pending, int leaf_idx,
                             std::vector<std::pair<int, int>> path) {
  for (auto& p : pending) {
    if (p.leaf_idx == leaf_idx) {
      p.paths.push_back(std::move(path));
      return;
    }
  }
  PendingLeaf p;
  p.leaf_idx = leaf_idx;
  p.paths.push_back(std::move(path));
  pending.push_back(std::move(p));
}

static PuctResult puct_result_from_root(PuctNode& root) {
  PuctResult out;
  out.terminal = root.terminal;
  out.n_act = root.n_act;
  out.childN = root.childN;
  out.childW = root.childW;
  out.state = root.state;
  out.step_opts = root.step_opts;
  if (!root.terminal)
    ensure_puct_py_metadata(root);
  out.canon = root.canon;
  out.opts = root.opts;
  out.ctx = root.ctx;
  return out;
}

static PuctResult rl_puct_mcts_callback(
    const GameState& root_state, const py::object& deck, int n_sims,
    uint64_t seed, int batch, double vloss, double c_puct, double dir_alpha,
    double dir_eps, const py::function& eval_batch) {
  std::vector<PuctNode> tree;
  tree.reserve(std::max(1, n_sims) + 1);
  tree.emplace_back(root_state);
  PuctNode& root = tree[0];
  if (root.terminal) return puct_result_from_root(root);
  int me = root.mover;
  prepare_puct_node(root);
  {
    py::list reqs;
    reqs.append(eval_req(root, deck));
    py::sequence outs = eval_batch(reqs).cast<py::sequence>();
    py::tuple item = outs[0].cast<py::tuple>();
    set_puct_priors(root, item[1]);
    apply_dirichlet_noise(root, dir_alpha, dir_eps,
                          seed ^ 0x9e3779b97f4a7c15ULL);
  }

  uint64_t rng = seed ? seed : 1;
  int done = 0;
  auto backup = [&](const std::vector<std::pair<int, int>>& path, double v_me) {
    for (auto [idx, a] : path) {
      PuctNode& node = tree[idx];
      node.childN[a] += 1.0 - vloss;
      node.childW[a] += v_me - vloss * virtual_loss_sign(node, me);
      node.totalN += 1.0 - vloss;
    }
  };

  while (done < n_sims) {
    int target = std::min(std::max(batch, 1), n_sims - done);
    std::vector<PendingLeaf> pending;
    std::vector<std::pair<std::vector<std::pair<int, int>>, double>> terminals;
    int collected = 0;
    while (collected < target) {
      int node_idx = 0;
      std::vector<std::pair<int, int>> path;
      while (true) {
        PuctNode& node = tree[node_idx];
        int a = select_puct_action(node, me, c_puct);
        path.emplace_back(node_idx, a);
        node.childN[a] += vloss;
        node.childW[a] += vloss * virtual_loss_sign(node, me);
        node.totalN += vloss;

        int child_idx = node.child[a];
        if (child_idx < 0) {
          GameState cs = node.state;
          rng = (rng * 6364136223846793005ULL + 1442695040888963407ULL);
          cs.rng = rng;
          uint64_t step_rng = rng;
          rl_step_cached(cs, node.step_opts, a, step_rng);
          tree.emplace_back(std::move(cs));
          int leaf_idx = static_cast<int>(tree.size()) - 1;
          tree[node_idx].child[a] = leaf_idx;
          PuctNode& leaf = tree[leaf_idx];
          if (leaf.terminal) {
            terminals.emplace_back(std::move(path), puct_term_value(leaf.state, me));
          } else {
            prepare_puct_node(leaf);
            add_pending_leaf(pending, leaf_idx, std::move(path));
          }
          break;
        }
        node_idx = child_idx;
        if (tree[node_idx].terminal) {
          terminals.emplace_back(std::move(path),
                                 puct_term_value(tree[node_idx].state, me));
          break;
        }
        if (!tree[node_idx].expanded) {
          add_pending_leaf(pending, node_idx, std::move(path));
          break;
        }
      }
      ++collected;
    }

    if (!pending.empty()) {
      py::list reqs;
      for (const auto& p : pending)
        reqs.append(eval_req(tree[p.leaf_idx], deck));
      py::sequence outs = eval_batch(reqs).cast<py::sequence>();
      for (size_t i = 0; i < pending.size(); ++i) {
        PuctNode& leaf = tree[pending[i].leaf_idx];
        py::tuple item = outs[static_cast<py::ssize_t>(i)].cast<py::tuple>();
        double v = item[0].cast<double>();
        set_puct_priors(leaf, item[1]);
        double v_me = leaf.mover == me ? v : -v;
        for (const auto& path : pending[i].paths)
          backup(path, v_me);
      }
    }
    for (const auto& t : terminals)
      backup(t.first, t.second);
    done += collected;
  }
  return puct_result_from_root(tree[0]);
}

static PuctResult rl_puct_mcts_feature_callback(
    const GameState& root_state, const std::vector<int>& deck, int n_sims,
    uint64_t seed, int batch, double vloss, double c_puct, double dir_alpha,
    double dir_eps, const py::function& eval_payload) {
  std::vector<PuctNode> tree;
  tree.reserve(std::max(1, n_sims) + 1);
  tree.emplace_back(root_state);
  PuctNode& root = tree[0];
  if (root.terminal) return puct_result_from_root(root);
  int me = root.mover;
  prepare_puct_node(root, false);
  {
    std::vector<const PuctNode*> nodes{&root};
    py::sequence outs =
        eval_payload(puct_feature_payload(nodes, deck, root.n_act)).cast<py::sequence>();
    py::tuple item = outs[0].cast<py::tuple>();
    set_puct_priors(root, item[1]);
    apply_dirichlet_noise(root, dir_alpha, dir_eps,
                          seed ^ 0x9e3779b97f4a7c15ULL);
  }

  uint64_t rng = seed ? seed : 1;
  int done = 0;
  auto backup = [&](const std::vector<std::pair<int, int>>& path, double v_me) {
    for (auto [idx, a] : path) {
      PuctNode& node = tree[idx];
      node.childN[a] += 1.0 - vloss;
      node.childW[a] += v_me - vloss * virtual_loss_sign(node, me);
      node.totalN += 1.0 - vloss;
    }
  };

  while (done < n_sims) {
    int target = std::min(std::max(batch, 1), n_sims - done);
    std::vector<PendingLeaf> pending;
    std::vector<std::pair<std::vector<std::pair<int, int>>, double>> terminals;
    int collected = 0;
    while (collected < target) {
      int node_idx = 0;
      std::vector<std::pair<int, int>> path;
      while (true) {
        PuctNode& node = tree[node_idx];
        int a = select_puct_action(node, me, c_puct);
        path.emplace_back(node_idx, a);
        node.childN[a] += vloss;
        node.childW[a] += vloss * virtual_loss_sign(node, me);
        node.totalN += vloss;

        int child_idx = node.child[a];
        if (child_idx < 0) {
          GameState cs = node.state;
          rng = (rng * 6364136223846793005ULL + 1442695040888963407ULL);
          cs.rng = rng;
          uint64_t step_rng = rng;
          rl_step_cached(cs, node.step_opts, a, step_rng);
          tree.emplace_back(std::move(cs));
          int leaf_idx = static_cast<int>(tree.size()) - 1;
          tree[node_idx].child[a] = leaf_idx;
          PuctNode& leaf = tree[leaf_idx];
          if (leaf.terminal) {
            terminals.emplace_back(std::move(path), puct_term_value(leaf.state, me));
          } else {
            prepare_puct_node(leaf, false);
            add_pending_leaf(pending, leaf_idx, std::move(path));
          }
          break;
        }
        node_idx = child_idx;
        if (tree[node_idx].terminal) {
          terminals.emplace_back(std::move(path),
                                 puct_term_value(tree[node_idx].state, me));
          break;
        }
        if (!tree[node_idx].expanded) {
          add_pending_leaf(pending, node_idx, std::move(path));
          break;
        }
      }
      ++collected;
    }

    if (!pending.empty()) {
      std::vector<int> leaf_indices;
      leaf_indices.reserve(pending.size());
      for (const auto& p : pending)
        leaf_indices.push_back(p.leaf_idx);
      py::sequence outs =
          eval_payload(puct_feature_payload_for_indices(tree, leaf_indices, deck))
              .cast<py::sequence>();
      for (size_t i = 0; i < pending.size(); ++i) {
        PuctNode& leaf = tree[pending[i].leaf_idx];
        py::tuple item = outs[static_cast<py::ssize_t>(i)].cast<py::tuple>();
        double v = item[0].cast<double>();
        set_puct_priors(leaf, item[1]);
        double v_me = leaf.mover == me ? v : -v;
        for (const auto& path : pending[i].paths)
          backup(path, v_me);
      }
    }
    for (const auto& t : terminals)
      backup(t.first, t.second);
    done += collected;
  }
  return puct_result_from_root(tree[0]);
}

struct NativeTrainRecord {
  FeatureRecord features;
  std::vector<float> pi;
  int n_act = 0;
  int mover = 0;
  float hval = 0.0f;
};

struct NativePuctNode {
  GameState state;
  RlOptionSet step_opts;
  std::vector<Descriptor> descriptors;
  int32_t edge_offset = -1;
  int16_t ctx = -1;
  int8_t mover = 0;
  uint8_t n_act = 0;
  bool terminal = false;
  bool expanded = false;
  bool root_noised = false;
  float totalN = 0.0f;

  explicit NativePuctNode(GameState s) : state(std::move(s)) {
    mover = static_cast<int8_t>(state.yourIndex);
    terminal = state.result >= 0;
  }
};

struct NativePuctTree {
  std::vector<NativePuctNode> nodes;
  std::vector<float> prior;
  std::vector<float> childN;
  std::vector<float> childW;
  std::vector<int32_t> child;

  void clear() {
    nodes.clear();
    prior.clear();
    childN.clear();
    childW.clear();
    child.clear();
  }

  void reserve(size_t node_count, size_t edge_count) {
    nodes.reserve(node_count);
    prior.reserve(edge_count);
    childN.reserve(edge_count);
    childW.reserve(edge_count);
    child.reserve(edge_count);
  }

  int emplace(GameState state) {
    nodes.emplace_back(std::move(state));
    return static_cast<int>(nodes.size()) - 1;
  }

  void allocate_edges(int node_idx, int n_act) {
    NativePuctNode& node = nodes[node_idx];
    node.n_act = static_cast<uint8_t>(std::min(std::max(n_act, 1), RL_MAX_ACTIONS));
    node.edge_offset = static_cast<int32_t>(prior.size());
    prior.resize(prior.size() + node.n_act, 0.0f);
    childN.resize(childN.size() + node.n_act, 0.0f);
    childW.resize(childW.size() + node.n_act, 0.0f);
    child.resize(child.size() + node.n_act, -1);
  }

  int edge_index(int node_idx, int action) const {
    return nodes[node_idx].edge_offset + action;
  }

  float edge_prior(int node_idx, int action) const {
    return prior[edge_index(node_idx, action)];
  }

  float edge_visits(int node_idx, int action) const {
    return childN[edge_index(node_idx, action)];
  }

  float edge_value_sum(int node_idx, int action) const {
    return childW[edge_index(node_idx, action)];
  }

  int child_index(int node_idx, int action) const {
    return child[edge_index(node_idx, action)];
  }

  void set_edge_prior(int node_idx, int action, float value) {
    prior[edge_index(node_idx, action)] = value;
  }

  void set_edge_stats(int node_idx, int action, float visits,
                      float value_sum) {
    int edge = edge_index(node_idx, action);
    childN[edge] = visits;
    childW[edge] = value_sum;
  }

  void add_edge_stats(int node_idx, int action, float visits_delta,
                      float value_delta) {
    int edge = edge_index(node_idx, action);
    childN[edge] += visits_delta;
    childW[edge] += value_delta;
    nodes[node_idx].totalN += visits_delta;
  }

  void set_child_index(int node_idx, int action, int child_idx) {
    child[edge_index(node_idx, action)] = child_idx;
  }

  double visit_sum(int node_idx) const {
    const NativePuctNode& node = nodes[node_idx];
    double total = 0.0;
    for (int a = 0; a < static_cast<int>(node.n_act); ++a)
      total += edge_visits(node_idx, a);
    return total;
  }
};

struct NativeSelfplayGame {
  GameState state;
  uint64_t seed = 1;
  int step = 0;
  int root_idx = -1;
  int me = 0;
  std::mt19937 choice_rng;
  NativePuctTree tree;
  std::vector<NativeTrainRecord> records;
};

struct NativeEvalTarget {
  int game_idx = -1;
  int node_idx = -1;
};

struct NativePendingLeaf {
  int game_idx = -1;
  int leaf_idx = -1;
  int me = 0;
  size_t path_offset = 0;
  size_t path_len = 0;
};

struct NativeSelfplayScratch {
  std::vector<NativeEvalTarget> new_roots;
  std::vector<NativePendingLeaf> pending;
  std::vector<std::pair<int32_t, uint16_t>> pending_paths;
  std::vector<const NativePuctNode*> eval_nodes;
  std::vector<NativeSelfplayGame> next_pool;

  explicit NativeSelfplayScratch(int concurrent) {
    const size_t n = static_cast<size_t>(std::max(concurrent, 1));
    new_roots.reserve(n);
    pending.reserve(n);
    pending_paths.reserve(n * 16);
    eval_nodes.reserve(n);
    next_pool.reserve(n);
  }
};

static void prepare_native_puct_node(NativePuctTree& tree, int node_idx) {
  NativePuctNode& node = tree.nodes[node_idx];
  if (node.state.has_pending()) {
    node.ctx = static_cast<int16_t>(node.state.pending.context);
    node.descriptors = node.state.pending.options;
    node.step_opts = rl_options(node.state);
  } else {
    node.descriptors = legal_main(node.state);
    node.ctx = -1;
    node.step_opts = rl_options_from_descriptors(node.descriptors);
  }
  if (static_cast<int>(node.descriptors.size()) > RL_MAX_ACTIONS)
    node.descriptors.resize(RL_MAX_ACTIONS);
  tree.allocate_edges(node_idx, static_cast<int>(node.descriptors.size()));
}

static py::tuple native_puct_feature_payload(
    const std::vector<const NativePuctNode*>& nodes,
    const std::vector<int>& deck, int width) {
  width = std::max(width, 1);
  FeaturePayloadScratch batch;
  batch.reserve(nodes.size(), width);
  for (const NativePuctNode* node : nodes) {
    int n_act = static_cast<int>(node->n_act);
    batch.append(node->state, node->descriptors, node->ctx, n_act, width,
                 deck);
  }
  return batch.to_python();
}

static void set_native_puct_priors(NativePuctTree& tree, int node_idx,
                                   const py::handle& logits_obj) {
  py::sequence logits = logits_obj.cast<py::sequence>();
  NativePuctNode& node = tree.nodes[node_idx];
  int n = static_cast<int>(node.n_act);
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) {
    float v = logits[i].cast<float>();
    tree.set_edge_prior(node_idx, i, v);
    if (v > max_logit) max_logit = v;
  }
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    float p = std::exp(tree.edge_prior(node_idx, i) - max_logit);
    tree.set_edge_prior(node_idx, i, p);
    sum += p;
  }
  if (sum > 0.0f) {
    for (int i = 0; i < n; ++i)
      tree.set_edge_prior(node_idx, i, tree.edge_prior(node_idx, i) / sum);
  } else {
    float u = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i) tree.set_edge_prior(node_idx, i, u);
  }
  node.expanded = true;
}

static double virtual_loss_sign(const NativePuctNode& node, int me) {
  return node.mover == me ? -1.0 : 1.0;
}

struct NativeSelfplayStats {
  long long total_ns = 0;
  long long root_payload_ns = 0;
  long long root_callback_ns = 0;
  long long leaf_collect_ns = 0;
  long long leaf_payload_ns = 0;
  long long leaf_callback_ns = 0;
  long long leaf_backup_ns = 0;
  long long record_export_ns = 0;
  long long root_eval_calls = 0;
  long long leaf_eval_calls = 0;
  long long root_eval_rows = 0;
  long long leaf_eval_rows = 0;
  long long leaf_sims_collected = 0;
  long long configured_leaf_batch_rows = 0;
  long long row_cap_flushes = 0;
  long long peak_pending_rows = 0;
  long long root_payload_bytes = 0;
  long long leaf_payload_bytes = 0;
  long long peak_payload_bytes = 0;
  long long peak_pool_games = 0;
  long long peak_tree_nodes = 0;
  long long peak_tree_capacity = 0;
  long long peak_tree_edges = 0;
  long long peak_tree_edge_capacity = 0;
  long long peak_records_pending = 0;
  long long peak_scratch_pending = 0;
  long long peak_scratch_paths = 0;
  long long records = 0;
  long long games_finished = 0;
};

using NativeSelfplayClock = std::chrono::steady_clock;

static long long elapsed_native_ns(NativeSelfplayClock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             NativeSelfplayClock::now() - start)
      .count();
}

static py::dict native_selfplay_stats_to_dict(const NativeSelfplayStats& s) {
  py::dict d;
  d["total_ns"] = s.total_ns;
  d["root_payload_ns"] = s.root_payload_ns;
  d["root_callback_ns"] = s.root_callback_ns;
  d["leaf_collect_ns"] = s.leaf_collect_ns;
  d["leaf_payload_ns"] = s.leaf_payload_ns;
  d["leaf_callback_ns"] = s.leaf_callback_ns;
  d["leaf_backup_ns"] = s.leaf_backup_ns;
  d["record_export_ns"] = s.record_export_ns;
  d["root_eval_calls"] = s.root_eval_calls;
  d["leaf_eval_calls"] = s.leaf_eval_calls;
  d["root_eval_rows"] = s.root_eval_rows;
  d["leaf_eval_rows"] = s.leaf_eval_rows;
  d["leaf_sims_collected"] = s.leaf_sims_collected;
  d["leaf_batch_row_cap"] = s.configured_leaf_batch_rows;
  d["leaf_row_cap_flushes"] = s.row_cap_flushes;
  d["peak_pending_eval_rows"] = s.peak_pending_rows;
  d["root_payload_bytes"] = s.root_payload_bytes;
  d["leaf_payload_bytes"] = s.leaf_payload_bytes;
  d["peak_payload_bytes"] = s.peak_payload_bytes;
  d["peak_pool_games"] = s.peak_pool_games;
  d["peak_tree_nodes"] = s.peak_tree_nodes;
  d["peak_tree_capacity"] = s.peak_tree_capacity;
  d["peak_tree_edges"] = s.peak_tree_edges;
  d["peak_tree_edge_capacity"] = s.peak_tree_edge_capacity;
  d["peak_records_pending"] = s.peak_records_pending;
  d["peak_scratch_pending"] = s.peak_scratch_pending;
  d["peak_scratch_paths"] = s.peak_scratch_paths;
  d["records"] = s.records;
  d["games_finished"] = s.games_finished;
  return d;
}

static long long native_payload_nbytes(const py::tuple& payload) {
  long long total = 0;
  for (py::handle part : payload) {
    py::array arr = py::reinterpret_borrow<py::array>(part);
    total += static_cast<long long>(arr.nbytes());
  }
  return total;
}

static void observe_native_payload_stats(NativeSelfplayStats* stats,
                                         const py::tuple& payload,
                                         bool leaf_payload) {
  if (!stats) return;
  long long bytes = native_payload_nbytes(payload);
  if (leaf_payload)
    stats->leaf_payload_bytes += bytes;
  else
    stats->root_payload_bytes += bytes;
  stats->peak_payload_bytes = std::max(stats->peak_payload_bytes, bytes);
}

static void observe_native_memory_stats(
    const std::vector<NativeSelfplayGame>& pool,
    const NativeSelfplayScratch& scratch, NativeSelfplayStats* stats) {
  if (!stats) return;
  long long tree_nodes = 0;
  long long tree_capacity = 0;
  long long tree_edges = 0;
  long long tree_edge_capacity = 0;
  long long records_pending = 0;
  for (const auto& g : pool) {
    tree_nodes += static_cast<long long>(g.tree.nodes.size());
    tree_capacity += static_cast<long long>(g.tree.nodes.capacity());
    tree_edges += static_cast<long long>(g.tree.prior.size());
    tree_edge_capacity += static_cast<long long>(g.tree.prior.capacity());
    records_pending += static_cast<long long>(g.records.size());
  }
  stats->peak_pool_games = std::max(
      stats->peak_pool_games, static_cast<long long>(pool.size()));
  stats->peak_tree_nodes = std::max(stats->peak_tree_nodes, tree_nodes);
  stats->peak_tree_capacity =
      std::max(stats->peak_tree_capacity, tree_capacity);
  stats->peak_tree_edges = std::max(stats->peak_tree_edges, tree_edges);
  stats->peak_tree_edge_capacity =
      std::max(stats->peak_tree_edge_capacity, tree_edge_capacity);
  stats->peak_records_pending =
      std::max(stats->peak_records_pending, records_pending);
  stats->peak_scratch_pending = std::max(
      stats->peak_scratch_pending,
      static_cast<long long>(scratch.pending.size()));
  stats->peak_scratch_paths = std::max(
      stats->peak_scratch_paths,
      static_cast<long long>(scratch.pending_paths.size()));
}

static uint64_t lcg_next(uint64_t x) {
  return x * 6364136223846793005ULL + 1442695040888963407ULL;
}

static uint64_t ensure_native_agent(GameState& st, uint64_t seed) {
  while (st.result < 0 && !rl_is_agent_decision(st)) {
    if (st.has_pending()) {
      int k = std::min(st.pending.minCount,
                       static_cast<int>(st.pending.options.size()));
      std::vector<int> sel;
      sel.reserve(std::max(k, 0));
      for (int i = 0; i < k; ++i) sel.push_back(i);
      resolve(st, sel);
    } else {
      rl_step(st, 0, seed);
    }
  }
  return seed;
}

static size_t native_tree_node_reserve(int n_sims) {
  return static_cast<size_t>(std::max(96, n_sims * 3 + 32));
}

static size_t native_tree_edge_reserve(size_t node_reserve) {
  return node_reserve * 8;
}

static NativeSelfplayGame make_native_game(const std::vector<int>& deck,
                                           uint64_t base_seed, int index) {
  NativeSelfplayGame g;
  uint64_t gseed = base_seed + static_cast<uint64_t>(index) * 7919ULL + 1ULL;
  g.state = new_game(deck, deck, gseed ? gseed : 1ULL);
  g.seed = ensure_native_agent(g.state, gseed);
  g.choice_rng.seed(static_cast<uint32_t>(g.seed & 0x7fffffffULL));
  g.tree.reserve(native_tree_node_reserve(0),
                 native_tree_edge_reserve(native_tree_node_reserve(0)));
  return g;
}

static void apply_native_root_noise_once(NativePuctTree& tree, int node_idx,
                                         double alpha, double eps,
                                         uint64_t seed) {
  NativePuctNode& node = tree.nodes[node_idx];
  if (node.root_noised) return;
  if (eps <= 0.0 || node.n_act <= 1) {
    node.root_noised = true;
    return;
  }
  std::mt19937_64 gen(seed ? seed : 1);
  std::gamma_distribution<double> gamma(alpha, 1.0);
  int n = static_cast<int>(node.n_act);
  std::vector<float> noise(n, 0.0f);
  double sum = 0.0;
  for (float& x : noise) {
    x = static_cast<float>(gamma(gen));
    sum += x;
  }
  if (sum > 0.0) {
    for (int i = 0; i < n; ++i) {
      tree.set_edge_prior(
          node_idx, i,
          static_cast<float>((1.0 - eps) * tree.edge_prior(node_idx, i) +
                             eps * (noise[i] / sum)));
    }
  }
  node.root_noised = true;
}

static int select_native_puct_action(const NativePuctTree& tree, int node_idx,
                                     int me, double c_puct) {
  const NativePuctNode& node = tree.nodes[node_idx];
  double c = c_puct * std::sqrt(static_cast<double>(node.totalN) + 1.0);
  bool flip = node.mover != me;
  int best_i = 0;
  double best_u = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(node.n_act); ++i) {
    float n = tree.edge_visits(node_idx, i);
    double q = n > 0.0f
                   ? static_cast<double>(tree.edge_value_sum(node_idx, i) / n)
                   : 0.0;
    if (flip) q = -q;
    double u = q + c * static_cast<double>(tree.edge_prior(node_idx, i)) /
                       (1.0 + static_cast<double>(n));
    if (u > best_u) {
      best_u = u;
      best_i = i;
    }
  }
  return best_i;
}

static int copy_native_subtree(const NativePuctTree& src, int src_idx,
                               NativePuctTree& dst) {
  const NativePuctNode& sn = src.nodes[src_idx];
  int dst_idx = dst.emplace(sn.state);
  NativePuctNode& dn = dst.nodes[dst_idx];
  dn.step_opts = sn.step_opts;
  dn.descriptors = sn.descriptors;
  dn.ctx = sn.ctx;
  dn.mover = sn.mover;
  dn.terminal = sn.terminal;
  dn.expanded = sn.expanded;
  dn.root_noised = sn.root_noised;
  dn.totalN = sn.totalN;
  if (sn.edge_offset >= 0 && sn.n_act > 0) {
    dst.allocate_edges(dst_idx, static_cast<int>(sn.n_act));
    for (int a = 0; a < static_cast<int>(sn.n_act); ++a) {
      dst.set_edge_prior(dst_idx, a, src.edge_prior(src_idx, a));
      dst.set_edge_stats(dst_idx, a, src.edge_visits(src_idx, a),
                         src.edge_value_sum(src_idx, a));
      int child_idx = src.child_index(src_idx, a);
      dst.set_child_index(
          dst_idx, a,
          child_idx >= 0 ? copy_native_subtree(src, child_idx, dst) : -1);
    }
  }
  return dst_idx;
}

static std::pair<size_t, size_t> count_native_subtree(
    const NativePuctTree& tree, int root_idx) {
  size_t nodes = 0;
  size_t edges = 0;
  std::vector<int> stack{root_idx};
  while (!stack.empty()) {
    int idx = stack.back();
    stack.pop_back();
    const NativePuctNode& node = tree.nodes[idx];
    ++nodes;
    if (node.edge_offset < 0) continue;
    edges += static_cast<size_t>(node.n_act);
    for (int a = 0; a < static_cast<int>(node.n_act); ++a) {
      int ch = tree.child_index(idx, a);
      if (ch >= 0) stack.push_back(ch);
    }
  }
  return {nodes, edges};
}

static void compact_native_tree_to_subtree(NativePuctTree& tree, int root_idx) {
  auto [node_count, edge_count] = count_native_subtree(tree, root_idx);
  NativePuctTree compact;
  compact.reserve(node_count, edge_count);
  copy_native_subtree(tree, root_idx, compact);
  tree = std::move(compact);
}

static void rebase_native_tree_values(NativePuctTree& tree, int root_idx,
                                      double sign) {
  if (sign == 1.0 || root_idx < 0) return;
  std::vector<int> stack{root_idx};
  while (!stack.empty()) {
    int idx = stack.back();
    stack.pop_back();
    NativePuctNode& node = tree.nodes[idx];
    if (node.edge_offset < 0) continue;
    for (int a = 0; a < static_cast<int>(node.n_act); ++a) {
      tree.set_edge_stats(idx, a, tree.edge_visits(idx, a),
                          tree.edge_value_sum(idx, a) *
                              static_cast<float>(sign));
      int ch = tree.child_index(idx, a);
      if (ch >= 0) stack.push_back(ch);
    }
  }
}

static void backup_pending_path(
    NativeSelfplayGame& g,
    const std::vector<std::pair<int32_t, uint16_t>>& path,
    size_t offset, size_t len, int me, double v_me, double vloss) {
  const size_t end = offset + len;
  if (vloss == 0.0) {
    for (size_t i = offset; i < end; ++i) {
      auto [idx, a] = path[i];
      g.tree.add_edge_stats(idx, a, 1.0f, static_cast<float>(v_me));
    }
    return;
  }
  for (size_t i = offset; i < end; ++i) {
    auto [idx, a] = path[i];
    NativePuctNode& node = g.tree.nodes[idx];
    g.tree.add_edge_stats(
        idx, a, static_cast<float>(1.0 - vloss),
        static_cast<float>(v_me - vloss * virtual_loss_sign(node, me)));
  }
}

static void apply_native_virtual_loss(NativePuctTree& tree, int node_idx,
                                      int action, int me, double vloss) {
  if (vloss == 0.0) return;
  NativePuctNode& node = tree.nodes[node_idx];
  tree.add_edge_stats(node_idx, action, static_cast<float>(vloss),
                      static_cast<float>(vloss * virtual_loss_sign(node, me)));
}

static int choose_visit_action(const NativePuctTree& tree, int root_idx,
                               int step, int temp_moves, std::mt19937& rng) {
  const NativePuctNode& root = tree.nodes[root_idx];
  int n = root.n_act;
  if (n <= 0) return 0;
  double sum = tree.visit_sum(root_idx);
  if (step < temp_moves && sum > 0.0) {
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
      acc += tree.edge_visits(root_idx, i);
      if (r <= acc) return i;
    }
  }
  int best = 0;
  double best_n = -1.0;
  for (int i = 0; i < n; ++i) {
    if (tree.edge_visits(root_idx, i) > best_n) {
      best_n = tree.edge_visits(root_idx, i);
      best = i;
    }
  }
  return best;
}

static double native_root_visit_count(const NativePuctTree& tree, int root_idx) {
  return tree.visit_sum(root_idx);
}

static py::array_t<float> pi_to_array(const std::vector<float>& pi) {
  return vector_to_numpy(pi);
}

static void append_finished_game(py::list& data, std::map<int, long>& results,
                                 const NativeSelfplayGame& g, double vboot) {
  int r = g.state.result;
  results[r] += 1;
  for (const auto& rec : g.records) {
    double outcome = 0.0;
    if (r == 2)
      outcome = 0.0;
    else
      outcome = (r == rec.mover) ? 1.0 : -1.0;
    double z = (1.0 - vboot) * outcome + vboot * rec.hval;
    py::object rec_obj = py::cast(rec.features);
    data.append(py::make_tuple(rec_obj, rec_obj, rec.n_act,
                               pi_to_array(rec.pi), z));
  }
}

static void eval_native_targets(std::vector<NativeSelfplayGame>& pool,
                                const std::vector<NativeEvalTarget>& targets,
                                const std::vector<int>& deck,
                                const py::function& eval_payload,
                                std::vector<const NativePuctNode*>& nodes,
                                NativeSelfplayStats* stats) {
  if (targets.empty()) return;
  nodes.clear();
  nodes.reserve(targets.size());
  int width = 1;
  for (const auto& t : targets) {
    const NativePuctNode& node = pool[t.game_idx].tree.nodes[t.node_idx];
    nodes.push_back(&node);
    width = std::max(width, static_cast<int>(node.n_act));
  }
  auto t_payload = NativeSelfplayClock::now();
  py::tuple payload = native_puct_feature_payload(nodes, deck, width);
  observe_native_payload_stats(stats, payload, false);
  if (stats) stats->root_payload_ns += elapsed_native_ns(t_payload);

  auto t_callback = NativeSelfplayClock::now();
  py::sequence outs = eval_payload(payload).cast<py::sequence>();
  if (stats) {
    stats->root_callback_ns += elapsed_native_ns(t_callback);
    ++stats->root_eval_calls;
    stats->root_eval_rows += static_cast<long long>(targets.size());
  }
  for (size_t i = 0; i < targets.size(); ++i) {
    py::tuple item = outs[static_cast<py::ssize_t>(i)].cast<py::tuple>();
    set_native_puct_priors(pool[targets[i].game_idx].tree,
                           targets[i].node_idx, item[1]);
  }
}

static void prepare_native_roots(std::vector<NativeSelfplayGame>& pool,
                                 NativeSelfplayScratch& scratch,
                                 bool reuse_tree, int n_sims,
                                 double dir_alpha,
                                 double dir_eps, uint64_t seed) {
  scratch.new_roots.clear();
  for (int gi = 0; gi < static_cast<int>(pool.size()); ++gi) {
    NativeSelfplayGame& g = pool[gi];
    if (!reuse_tree || g.root_idx < 0) {
      g.tree.clear();
      size_t node_reserve = native_tree_node_reserve(n_sims);
      g.tree.reserve(node_reserve, native_tree_edge_reserve(node_reserve));
      g.tree.emplace(g.state);
      g.root_idx = 0;
    } else {
      g.tree.nodes[g.root_idx].root_noised = false;
    }

    NativePuctNode& root = g.tree.nodes[g.root_idx];
    g.me = root.mover;
    if (root.terminal) continue;
    if (!root.expanded) {
      prepare_native_puct_node(g.tree, g.root_idx);
      scratch.new_roots.push_back({gi, g.root_idx});
    } else {
      apply_native_root_noise_once(g.tree, g.root_idx, dir_alpha, dir_eps,
                                   seed ^ g.seed ^
                                       static_cast<uint64_t>(g.step));
    }
  }
}

static void eval_and_noise_native_roots(
    std::vector<NativeSelfplayGame>& pool, NativeSelfplayScratch& scratch,
    const std::vector<int>& deck, const py::function& eval_payload,
    double dir_alpha, double dir_eps, uint64_t seed,
    NativeSelfplayStats* stats) {
  eval_native_targets(pool, scratch.new_roots, deck, eval_payload,
                      scratch.eval_nodes, stats);
  for (const auto& t : scratch.new_roots) {
    NativeSelfplayGame& g = pool[t.game_idx];
    apply_native_root_noise_once(g.tree, t.node_idx, dir_alpha, dir_eps,
                                 seed ^ g.seed ^ 0x9e3779b97f4a7c15ULL);
  }
}

struct NativeCollectResult {
  int collected = 0;
  int pending_eval_rows = 0;
  bool exhausted = false;
  bool row_cap_reached = false;
};

struct NativeLeafBatchBudget {
  int row_cap = 0;
  int pending_width = 1;
  int pending_eval_rows = 0;
  bool row_cap_reached = false;

  explicit NativeLeafBatchBudget(int row_cap_) : row_cap(row_cap_) {}

  void observe_leaf(int n_act, int pending_count) {
    pending_width = std::max(pending_width, n_act);
    pending_eval_rows = pending_width * pending_count;
    if (row_cap > 0 && pending_eval_rows >= row_cap)
      row_cap_reached = true;
  }
};

static NativeCollectResult collect_native_leaf_batch(
    std::vector<NativeSelfplayGame>& pool, NativeSelfplayScratch& scratch,
    int n_sims, bool budget_reuse, int leaf_batch, double c_puct,
    double vloss, int leaf_batch_rows, int& sim_round) {
  scratch.pending.clear();
  scratch.pending_paths.clear();
  NativeCollectResult result;
  NativeLeafBatchBudget row_budget(leaf_batch_rows);

  auto add_pending = [&](NativePendingLeaf&& pe, const NativePuctNode& leaf) {
    scratch.pending.push_back(std::move(pe));
    row_budget.observe_leaf(static_cast<int>(leaf.n_act),
                            static_cast<int>(scratch.pending.size()));
    result.pending_eval_rows = row_budget.pending_eval_rows;
    result.row_cap_reached = row_budget.row_cap_reached;
  };

  while (result.collected < leaf_batch && !result.row_cap_reached) {
    if (!budget_reuse && sim_round >= n_sims) {
      result.exhausted = true;
      break;
    }

    bool round_selected = false;
    for (int gi = 0; gi < static_cast<int>(pool.size()); ++gi) {
      NativeSelfplayGame& g = pool[gi];
      if (g.root_idx < 0) continue;
      NativePuctNode& root = g.tree.nodes[g.root_idx];
      if (root.terminal || !root.expanded) continue;
      if (budget_reuse &&
          native_root_visit_count(g.tree, g.root_idx) >=
              static_cast<double>(n_sims))
        continue;

      round_selected = true;
      ++result.collected;
      int me = g.me;
      int node_idx = g.root_idx;
      const size_t path_offset = scratch.pending_paths.size();

      while (true) {
        NativePuctNode& node = g.tree.nodes[node_idx];
        int a = select_native_puct_action(g.tree, node_idx, me, c_puct);
        scratch.pending_paths.emplace_back(
            static_cast<int32_t>(node_idx), static_cast<uint16_t>(a));
        apply_native_virtual_loss(g.tree, node_idx, a, me, vloss);

        int child_idx = g.tree.child_index(node_idx, a);
        if (child_idx < 0) {
          GameState cs = node.state;
          g.seed = lcg_next(g.seed);
          cs.rng = g.seed;
          uint64_t step_rng = g.seed;
          rl_step_cached(cs, node.step_opts, a, step_rng);
          int leaf_idx = g.tree.emplace(std::move(cs));
          g.tree.set_child_index(node_idx, a, leaf_idx);
          NativePuctNode& leaf = g.tree.nodes[leaf_idx];
          if (leaf.terminal) {
            backup_pending_path(g, scratch.pending_paths, path_offset,
                                scratch.pending_paths.size() - path_offset,
                                me, puct_term_value(leaf.state, me), vloss);
            scratch.pending_paths.resize(path_offset);
          } else {
            prepare_native_puct_node(g.tree, leaf_idx);
            NativePendingLeaf pe;
            pe.game_idx = gi;
            pe.leaf_idx = leaf_idx;
            pe.me = me;
            pe.path_offset = path_offset;
            pe.path_len = scratch.pending_paths.size() - path_offset;
            add_pending(std::move(pe), leaf);
          }
          break;
        }

        node_idx = child_idx;
        if (g.tree.nodes[node_idx].terminal) {
          backup_pending_path(
              g, scratch.pending_paths, path_offset,
              scratch.pending_paths.size() - path_offset, me,
              puct_term_value(g.tree.nodes[node_idx].state, me), vloss);
          scratch.pending_paths.resize(path_offset);
          break;
        }
        if (!g.tree.nodes[node_idx].expanded) {
          NativePendingLeaf pe;
          pe.game_idx = gi;
          pe.leaf_idx = node_idx;
          pe.me = me;
          pe.path_offset = path_offset;
          pe.path_len = scratch.pending_paths.size() - path_offset;
          add_pending(std::move(pe), g.tree.nodes[node_idx]);
          break;
        }
      }
      if (result.row_cap_reached) break;
    }

    if (!round_selected) {
      result.exhausted = true;
      break;
    }
    if (!budget_reuse) ++sim_round;
  }

  return result;
}

static void eval_and_backup_native_pending(
    std::vector<NativeSelfplayGame>& pool, NativeSelfplayScratch& scratch,
    const std::vector<int>& deck, const py::function& eval_payload,
    double vloss, NativeSelfplayStats* stats) {
  if (scratch.pending.empty()) return;
  scratch.eval_nodes.clear();
  int width = 1;
  for (const auto& p : scratch.pending) {
    const NativePuctNode& leaf = pool[p.game_idx].tree.nodes[p.leaf_idx];
    scratch.eval_nodes.push_back(&leaf);
    width = std::max(width, static_cast<int>(leaf.n_act));
  }

  auto t_payload = NativeSelfplayClock::now();
  py::tuple payload = native_puct_feature_payload(scratch.eval_nodes, deck, width);
  observe_native_payload_stats(stats, payload, true);
  if (stats) stats->leaf_payload_ns += elapsed_native_ns(t_payload);

  auto t_callback = NativeSelfplayClock::now();
  py::sequence outs = eval_payload(payload).cast<py::sequence>();
  if (stats) {
    stats->leaf_callback_ns += elapsed_native_ns(t_callback);
    ++stats->leaf_eval_calls;
    stats->leaf_eval_rows += static_cast<long long>(scratch.pending.size());
  }

  auto t_backup = NativeSelfplayClock::now();
  for (size_t i = 0; i < scratch.pending.size(); ++i) {
    const NativePendingLeaf& p = scratch.pending[i];
    NativeSelfplayGame& g = pool[p.game_idx];
    NativePuctNode& leaf = g.tree.nodes[p.leaf_idx];
    py::tuple item = outs[static_cast<py::ssize_t>(i)].cast<py::tuple>();
    double v = item[0].cast<double>();
    set_native_puct_priors(g.tree, p.leaf_idx, item[1]);
    double v_me = leaf.mover == p.me ? v : -v;
    backup_pending_path(g, scratch.pending_paths, p.path_offset, p.path_len,
                        p.me, v_me, vloss);
  }
  if (stats) stats->leaf_backup_ns += elapsed_native_ns(t_backup);
}

static void run_native_selfplay_search(
    std::vector<NativeSelfplayGame>& pool, NativeSelfplayScratch& scratch,
    const std::vector<int>& deck, int n_sims, bool budget_reuse,
    int leaf_batch, int leaf_batch_rows, double c_puct, double vloss,
    const py::function& eval_payload, NativeSelfplayStats* stats) {
  int sim_round = 0;
  while (true) {
    auto t_collect = NativeSelfplayClock::now();
    NativeCollectResult r =
        collect_native_leaf_batch(pool, scratch, n_sims, budget_reuse,
                                  leaf_batch, c_puct, vloss, leaf_batch_rows,
                                  sim_round);
    if (stats) {
      stats->leaf_collect_ns += elapsed_native_ns(t_collect);
      stats->leaf_sims_collected += static_cast<long long>(r.collected);
      stats->peak_pending_rows = std::max(
          stats->peak_pending_rows,
          static_cast<long long>(r.pending_eval_rows));
      if (r.row_cap_reached) ++stats->row_cap_flushes;
    }
    observe_native_memory_stats(pool, scratch, stats);
    if (r.collected == 0) break;
    eval_and_backup_native_pending(pool, scratch, deck, eval_payload, vloss,
                                   stats);
    observe_native_memory_stats(pool, scratch, stats);
    if (r.exhausted) break;
  }
}

static NativeTrainRecord make_native_train_record(
    const NativePuctTree& tree, int root_idx, const NativeSelfplayGame& g,
    const std::vector<int>& deck) {
  const NativePuctNode& root = tree.nodes[root_idx];
  NativeTrainRecord rec;
  rec.features = feature_record_cached(root.state, root.step_opts, root.ctx,
                                       root.n_act, deck, RL_MAX_ACTIONS);
  rec.n_act = root.n_act;
  rec.mover = g.state.yourIndex;
  rec.hval = rl_heuristic_value(root.state);
  rec.pi.assign(RL_MAX_ACTIONS, 0.0f);

  double visit_sum = 0.0;
  for (int i = 0; i < root.n_act; ++i)
    visit_sum += tree.edge_visits(root_idx, i);
  if (visit_sum > 0.0) {
    for (int i = 0; i < root.n_act; ++i)
      rec.pi[i] = static_cast<float>(tree.edge_visits(root_idx, i) / visit_sum);
  } else {
    float u = 1.0f / static_cast<float>(root.n_act);
    for (int i = 0; i < root.n_act; ++i) rec.pi[i] = u;
  }
  return rec;
}

static void record_and_advance_native_games(
    std::vector<NativeSelfplayGame>& pool, NativeSelfplayScratch& scratch,
    py::list& data, std::map<int, long>& results, const std::vector<int>& deck,
    uint64_t seed, int& started, int total_games, int temp_moves, double vboot,
    bool reuse_tree, int max_steps, NativeSelfplayStats* stats) {
  auto t_record = NativeSelfplayClock::now();
  scratch.next_pool.clear();
  for (auto& g : pool) {
    NativePuctNode& root = g.tree.nodes[g.root_idx];
    if (root.terminal || root.n_act == 0) {
      append_finished_game(data, results, g, vboot);
      if (stats) ++stats->games_finished;
      if (started < total_games)
        scratch.next_pool.push_back(make_native_game(deck, seed, started++));
      continue;
    }

    g.records.push_back(make_native_train_record(g.tree, g.root_idx, g, deck));
    if (stats) ++stats->records;

    int a = choose_visit_action(g.tree, g.root_idx, g.step, temp_moves,
                                g.choice_rng);
    int next_root_idx =
        (reuse_tree && a >= 0 && a < static_cast<int>(root.n_act))
            ? g.tree.child_index(g.root_idx, a)
            : -1;
    if (next_root_idx >= 0) {
      double sign = g.tree.nodes[next_root_idx].mover != g.me ? -1.0 : 1.0;
      compact_native_tree_to_subtree(g.tree, next_root_idx);
      rebase_native_tree_values(g.tree, 0, sign);
      g.root_idx = 0;
      NativePuctNode& child = g.tree.nodes[0];
      child.root_noised = false;
      g.state = child.state;
      g.seed = g.state.rng;
    } else {
      rl_step(g.state, a, g.seed);
      g.root_idx = -1;
    }

    ++g.step;
    if (g.state.result >= 0 || g.step >= max_steps) {
      append_finished_game(data, results, g, vboot);
      if (stats) ++stats->games_finished;
      if (started < total_games)
        scratch.next_pool.push_back(make_native_game(deck, seed, started++));
    } else {
      scratch.next_pool.push_back(std::move(g));
    }
  }
  pool.swap(scratch.next_pool);
  if (stats) stats->record_export_ns += elapsed_native_ns(t_record);
  observe_native_memory_stats(pool, scratch, stats);
}

static py::tuple rl_selfplay_puct_callback(
    const std::vector<int>& deck, int total_games, int concurrent, int n_sims,
    uint64_t seed, int temp_moves, double vboot, bool reuse_tree,
    double c_puct, double dir_alpha, double dir_eps,
    const py::function& eval_payload, int max_steps, bool budget_reuse,
    int leaf_batch, double vloss, bool profile, int leaf_batch_rows) {
  NativeSelfplayStats stats;
  NativeSelfplayStats* stats_ptr = profile ? &stats : nullptr;
  auto t_total = NativeSelfplayClock::now();
  py::list data;
  std::map<int, long> results;
  if (total_games <= 0) {
    py::dict empty;
    if (profile) {
      stats.total_ns = elapsed_native_ns(t_total);
      return py::make_tuple(data, empty, native_selfplay_stats_to_dict(stats));
    }
    return py::make_tuple(data, empty);
  }
  concurrent = std::max(1, std::min(concurrent, total_games));
  n_sims = std::max(0, n_sims);
  max_steps = std::max(1, max_steps);
  leaf_batch = std::max(1, leaf_batch);
  leaf_batch_rows = std::max(0, leaf_batch_rows);
  vloss = std::max(0.0, vloss);
  if (leaf_batch <= 1) vloss = 0.0;
  if (stats_ptr) stats.configured_leaf_batch_rows = leaf_batch_rows;

  int started = 0;
  std::vector<NativeSelfplayGame> pool;
  pool.reserve(concurrent);
  for (; started < concurrent; ++started)
    pool.push_back(make_native_game(deck, seed, started));

  NativeSelfplayScratch scratch(concurrent);
  observe_native_memory_stats(pool, scratch, stats_ptr);

  while (!pool.empty()) {
    prepare_native_roots(pool, scratch, reuse_tree, n_sims,
                         dir_alpha, dir_eps, seed);
    observe_native_memory_stats(pool, scratch, stats_ptr);
    eval_and_noise_native_roots(pool, scratch, deck, eval_payload, dir_alpha,
                                dir_eps, seed, stats_ptr);
    run_native_selfplay_search(pool, scratch, deck, n_sims, budget_reuse,
                               leaf_batch, leaf_batch_rows, c_puct, vloss,
                               eval_payload, stats_ptr);
    record_and_advance_native_games(pool, scratch, data, results, deck, seed,
                                    started, total_games, temp_moves, vboot,
                                    reuse_tree, max_steps, stats_ptr);
  }

  py::dict res;
  for (const auto& kv : results)
    res[py::int_(kv.first)] = kv.second;
  if (profile) {
    stats.total_ns = elapsed_native_ns(t_total);
    return py::make_tuple(data, res, native_selfplay_stats_to_dict(stats));
  }
  return py::make_tuple(data, res);
}

static bool atom_equal(const Atom& a, const Atom& b) {
  // content compare — sym pointers differ between literals and interned pool
  return a.is_str == b.is_str && a.is_none == b.is_none &&
         atom_sv(a) == atom_sv(b) && a.i == b.i;
}

static bool descriptor_equal(const Descriptor& a, const Descriptor& b) {
  if (a.size() != b.size()) return false;
  if (a.size() >= 4 && atom_string(a, 0) == "ABILITY" &&
      atom_string(b, 0) == "ABILITY" && atom_string(a, 1) == "STADIUM" &&
      atom_string(b, 1) == "STADIUM") {
    for (size_t i = 0; i < 3; ++i)
      if (!atom_equal(a[i], b[i])) return false;
    return true;
  }
  for (size_t i = 0; i < a.size(); ++i)
    if (!atom_equal(a[i], b[i])) return false;
  return true;
}

static bool descriptors_equal(const std::vector<Descriptor>& a,
                              const std::vector<Descriptor>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!descriptor_equal(a[i], b[i])) return false;
  return true;
}

static Descriptor descriptor_from_py(const py::handle& obj) {
  py::sequence seq = obj.cast<py::sequence>();
  Descriptor out;
  out.reserve(static_cast<size_t>(py::len(seq)));
  for (py::handle item : seq) {
    py::object value = py::reinterpret_borrow<py::object>(item);
    if (value.is_none())
      out.push_back(Atom::N());
    else if (py::isinstance<py::str>(value))
      out.push_back(Atom::S(value.cast<std::string>()));
    else
      out.push_back(Atom::I(value.cast<long long>()));
  }
  return out;
}

static std::vector<Descriptor> descriptors_from_py(const py::object& obj) {
  std::vector<Descriptor> out;
  if (obj.is_none()) return out;
  py::sequence seq = obj.cast<py::sequence>();
  out.reserve(static_cast<size_t>(py::len(seq)));
  for (py::handle item : seq)
    out.push_back(descriptor_from_py(item));
  return out;
}

static py::list int_vector_to_py(const auto& values) {
  py::list out;
  for (int v : values) out.append(v);
  return out;
}

static std::vector<int> int_vector_from_py(const py::object& obj) {
  std::vector<int> out;
  if (obj.is_none()) return out;
  py::sequence seq = obj.cast<py::sequence>();
  out.reserve(static_cast<size_t>(py::len(seq)));
  for (py::handle item : seq) out.push_back(item.cast<int>());
  return out;
}

static py::dict effect_frame_to_py(const EffectFrame& fr) {
  py::dict d;
  d["effect"] = fr.effect;
  d["phase"] = fr.phase;
  d["a"] = fr.a;
  d["program"] = fr.program;
  d["pc"] = fr.pc;
  d["attackId"] = fr.attackId;
  d["attackCardId"] = fr.attackCardId;
  d["sourceCardId"] = fr.sourceCardId;
  d["copiedAttack"] = fr.copiedAttack;
  d["copiedAttackBaseDamage"] = fr.copiedAttackBaseDamage;
  d["loopRemain"] = fr.loopRemain;
  d["selfBench"] = fr.selfBench;
  d["savedSrc"] = fr.savedSrc;
  d["savedPhase"] = fr.savedPhase;
  d["topDeckCount"] = fr.topDeckCount;
  d["topDeckSelectedCount"] = fr.topDeckSelectedCount;
  d["topDeckOwner"] = fr.topDeckOwner;
  d["topDeckCountedOut"] = fr.topDeckCountedOut;
  d["topDeckStart"] = fr.topDeckStart;
  d["scratch"] = int_vector_to_py(fr.scratch);
  d["savedScratch"] = int_vector_to_py(fr.savedScratch);
  return d;
}

static EffectFrame effect_frame_from_py(const py::handle& obj) {
  py::dict d = py::reinterpret_borrow<py::dict>(obj);
  EffectFrame fr;
  fr.effect = d["effect"].cast<int>();
  fr.phase = d["phase"].cast<int>();
  fr.a = d["a"].cast<int>();
  fr.program = d["program"].cast<int>();
  fr.pc = d["pc"].cast<int>();
  fr.attackId = d["attackId"].cast<int>();
  fr.attackCardId = d["attackCardId"].cast<int>();
  fr.sourceCardId = d["sourceCardId"].cast<int>();
  fr.copiedAttack = d["copiedAttack"].cast<bool>();
  fr.copiedAttackBaseDamage = d["copiedAttackBaseDamage"].cast<int>();
  fr.loopRemain = d["loopRemain"].cast<int>();
  fr.selfBench = d["selfBench"].cast<int>();
  fr.savedSrc = d["savedSrc"].cast<int>();
  fr.savedPhase = d["savedPhase"].cast<int>();
  fr.topDeckCount = d["topDeckCount"].cast<int>();
  fr.topDeckSelectedCount = d["topDeckSelectedCount"].cast<int>();
  fr.topDeckOwner = d["topDeckOwner"].cast<int>();
  fr.topDeckCountedOut = d["topDeckCountedOut"].cast<int>();
  fr.topDeckStart = d["topDeckStart"].cast<int>();
  fr.scratch = int_vector_from_py(d["scratch"]);
  fr.savedScratch = int_vector_from_py(d["savedScratch"]);
  return fr;
}

static py::list effect_stack_to_py(const std::vector<EffectFrame>& frames) {
  py::list out;
  for (const EffectFrame& fr : frames) out.append(effect_frame_to_py(fr));
  return out;
}

static std::vector<EffectFrame> effect_stack_from_py(const py::object& obj) {
  std::vector<EffectFrame> out;
  if (obj.is_none()) return out;
  py::sequence seq = obj.cast<py::sequence>();
  out.reserve(static_cast<size_t>(py::len(seq)));
  for (py::handle item : seq) out.push_back(effect_frame_from_py(item));
  return out;
}

static py::dict search_transients_to_py(const GameState& st) {
  py::dict d;
  py::dict pending;
  pending["context"] = st.pending.context;
  pending["minCount"] = st.pending.minCount;
  pending["maxCount"] = st.pending.maxCount;
  pending["options"] = descriptors_to_py(st.pending.options);
  d["pending"] = pending;
  d["effectStack"] = effect_stack_to_py(st.effectStack);
  d["afterProgramQueue"] = effect_stack_to_py(st.afterProgramQueue);
  d["pendingTrainerDiscard"] = st.pendingTrainerDiscard;
  d["pendingTrainerOwner"] = st.pendingTrainerOwner;
  py::list pendingMeganiumAura;
  pendingMeganiumAura.append(st.pendingMeganiumAura[0]);
  pendingMeganiumAura.append(st.pendingMeganiumAura[1]);
  d["pendingMeganiumAura"] = pendingMeganiumAura;
  d["coinHeads"] = st.coinHeads;
  d["coinFlips"] = st.coinFlips;
  d["countReg"] = st.countReg;
  d["discardedCount"] = st.discardedCount;
  d["lastEffectCount"] = st.lastEffectCount;
  d["lastAttackDamage"] = st.lastAttackDamage;
  d["endTurnAfterProgram"] = st.endTurnAfterProgram;
  d["deferredPostAttack"] = st.deferredPostAttack;
  d["deferredPostAttackPlayer"] = st.deferredPostAttackPlayer;
  d["deferredPostAttackId"] = st.deferredPostAttackId;
  d["deferredPostAttackCard"] = st.deferredPostAttackCard;
  d["replacementKoPairs"] = int_vector_to_py(st.replacementKoPairs);
  d["replacementKoGroups"] = int_vector_to_py(st.replacementKoGroups);
  d["checkupNext"] = st.checkupNext;
  d["checkupKoFirst"] = st.checkupKoFirst;
  d["checkupPromoteBeforePrize"] = st.checkupPromoteBeforePrize;
  return d;
}

static void apply_search_transients_from_py(GameState& st, const py::dict& d) {
  py::dict pending = d["pending"].cast<py::dict>();
  st.pending.context = pending["context"].cast<int>();
  st.pending.minCount = pending["minCount"].cast<int>();
  st.pending.maxCount = pending["maxCount"].cast<int>();
  st.pending.options = descriptors_from_py(pending["options"]);
  st.effectStack = effect_stack_from_py(d["effectStack"]);
  st.afterProgramQueue = effect_stack_from_py(d["afterProgramQueue"]);
  st.pendingTrainerDiscard = d["pendingTrainerDiscard"].cast<int>();
  st.pendingTrainerOwner = d["pendingTrainerOwner"].cast<int>();
  if (d.contains("pendingMeganiumAura")) {
    py::sequence a = d["pendingMeganiumAura"].cast<py::sequence>();
    if (py::len(a) >= 2) {
      st.pendingMeganiumAura[0] = a[0].cast<bool>();
      st.pendingMeganiumAura[1] = a[1].cast<bool>();
    }
  }
  st.coinHeads = d["coinHeads"].cast<int>();
  st.coinFlips = d["coinFlips"].cast<int>();
  st.countReg = d["countReg"].cast<int>();
  st.discardedCount = d["discardedCount"].cast<int>();
  st.lastEffectCount = d["lastEffectCount"].cast<int>();
  st.lastAttackDamage = d["lastAttackDamage"].cast<int>();
  st.endTurnAfterProgram = d["endTurnAfterProgram"].cast<bool>();
  st.deferredPostAttack = d["deferredPostAttack"].cast<bool>();
  st.deferredPostAttackPlayer = d["deferredPostAttackPlayer"].cast<int>();
  st.deferredPostAttackId = d["deferredPostAttackId"].cast<int>();
  st.deferredPostAttackCard = d["deferredPostAttackCard"].cast<int>();
  st.replacementKoPairs = int_vector_from_py(d["replacementKoPairs"]);
  st.replacementKoGroups = int_vector_from_py(d["replacementKoGroups"]);
  st.checkupNext = d["checkupNext"].cast<int>();
  st.checkupKoFirst = d["checkupKoFirst"].cast<int>();
  st.checkupPromoteBeforePrize = d["checkupPromoteBeforePrize"].cast<bool>();
}

static py::dict debug_search_summary(const GameState& st) {
  py::dict out;
  out["turn"] = st.turn;
  out["turnActionCount"] = st.turnActionCount;
  out["yourIndex"] = st.yourIndex;
  out["result"] = st.result;
  out["hidden"] = debug_hidden_zones(st);
  out["transients"] = search_transients_to_py(st);
  out["hasPending"] = st.has_pending();
  out["effectStackSize"] = static_cast<int>(st.effectStack.size());
  out["afterProgramQueueSize"] = static_cast<int>(st.afterProgramQueue.size());
  return out;
}

static bool descriptor_list_contains(const std::vector<Descriptor>& haystack,
                                     const Descriptor& needle) {
  for (const Descriptor& desc : haystack)
    if (descriptor_equal(desc, needle)) return true;
  return false;
}

static bool mark_extra_ability_used(GameState& st, const Descriptor& desc) {
  if (atom_string(desc, 0) != "ABILITY") return false;
  std::string_view area = atom_string(desc, 1);
  if (area == "STADIUM") {
    if (st.stadiumAbilityUsed) return false;
    st.stadiumAbilityUsed = true;
    return true;
  }

  Player& me = st.players[st.yourIndex];
  InPlay* source = nullptr;
  if (area == "ACTIVE") {
    if (me.activeKnown) source = &me.active;
  } else if (area == "BENCH") {
    int index = atom_int(desc, 2, -1);
    if (index >= 0 && index < static_cast<int>(me.bench.size()))
      source = &me.bench[index];
  }
  if (!source) return false;

  if (source->reconstructedAbilityLocked &&
      source->reconstructedAbilityLockTurn == st.turn)
    return false;
  source->reconstructedAbilityPrevUsed = source->abilityUsedThisTurn;
  source->reconstructedAbilityLocked = true;
  source->reconstructedAbilityLockTurn = st.turn;
  source->abilityUsedThisTurn = true;
  return true;
}

static bool mark_extra_attack_locked(GameState& st, const Descriptor& desc) {
  if (atom_string(desc, 0) != "ATTACK") return false;
  int attackId = atom_int(desc, 1, 0);
  Player& me = st.players[st.yourIndex];
  if (attackId <= 0 || !me.activeKnown) return false;
  InPlay& active = me.active;
  if (active.reconstructedAttackLockTurn != st.turn) {
    active.reconstructedAttackLocks.clear();
    active.reconstructedAttackLockTurn = st.turn;
  }
  if (std::find(active.reconstructedAttackLocks.begin(),
                active.reconstructedAttackLocks.end(),
                attackId) == active.reconstructedAttackLocks.end()) {
    active.reconstructedAttackLocks.push_back(attackId);
    return true;
  }
  return false;
}

static int descriptor_source_card_id(const GameState& st,
                                     const Descriptor& desc) {
  std::string_view kind = atom_string(desc, 0);
  if (kind == "PLAY")
    return atom_int(desc, 1, 0);
  if (kind != "ABILITY" && kind != "DISCARD_INPLAY")
    return 0;
  std::string_view area = atom_string(desc, 1);
  const Player& me = st.players[st.yourIndex];
  if (area == "ACTIVE")
    return me.activeKnown ? me.active.id : 0;
  if (area == "BENCH") {
    int index = atom_int(desc, 2, -1);
    if (index >= 0 && index < static_cast<int>(me.bench.size()))
      return me.bench[index].id;
    return 0;
  }
  if (area == "STADIUM" && !st.stadium.empty())
    return st.stadium[0];
  return 0;
}

static void infer_ko_gate_transients_from_cabt_options(
    GameState& st, const std::vector<Descriptor>& cabt_options) {
  for (const Descriptor& desc : cabt_options) {
    std::string_view kind = atom_string(desc, 0);
    int source = descriptor_source_card_id(st, desc);
    if (kind == "PLAY" && (source == 1080 || source == 1193))
      st.lastKoTurn[st.yourIndex] = st.turn - 1;
    else if (kind == "PLAY" && source == 1217)
      st.lastTeamRocketKoTurn[st.yourIndex] = st.turn - 1;
    else if (kind == "ABILITY" && source == 140)
      st.lastKoTurn[st.yourIndex] = st.turn - 1;
  }
}

static void reconstruct_main_from_cabt_options(
    GameState& st, const std::vector<Descriptor>& cabt_options) {
  if (cabt_options.empty()) return;
  infer_ko_gate_transients_from_cabt_options(st, cabt_options);
  for (const Descriptor& desc : cabt_options) {
    if (atom_string(desc, 0) == "ABILITY" &&
        atom_string(desc, 1) == "STADIUM") {
      st.stadiumAbilityUsed = false;
      break;
    }
  }
  for (int pass = 0; pass < 4; ++pass) {
    bool changed = false;
    std::vector<Descriptor> native = legal_main(st);
    for (const Descriptor& desc : native) {
      if (descriptor_list_contains(cabt_options, desc)) continue;
      std::string_view kind = atom_string(desc, 0);
      if (kind == "ABILITY")
        changed = mark_extra_ability_used(st, desc) || changed;
      else if (kind == "ATTACK")
        changed = mark_extra_attack_locked(st, desc) || changed;
    }
    if (!changed) break;
  }
}

struct MapleNativeNode {
  std::vector<GameState> worlds;
  std::vector<Descriptor> descriptors;
  RlOptionSet step_opts;
  int ctx = -1;
  int mover = -1;
  int n_act = 0;
  bool terminal = false;
  bool expanded = false;
  double value = 0.0;  // root-player POV
  std::vector<double> P;
  std::vector<double> childN;
  std::vector<double> childW;
  std::vector<int> child;

  explicit MapleNativeNode(std::vector<GameState> w) : worlds(std::move(w)) {}
};

struct MapleNativeSearch {
  std::vector<MapleNativeNode> tree;
  int root_player = 0;
  uint64_t rng = 1;
};

struct MaplePendingNodeEval {
  int search_idx = -1;
  int node_idx = -1;
  double terminal_value_sum = 0.0;
  int terminal_value_count = 0;
  std::vector<std::pair<int, int>> path;
};

struct MapleBatchScratch {
  std::vector<MapleNativeSearch> searches;
  std::vector<MaplePendingNodeEval> pending;
  std::vector<PuctResult> roots;
  FeaturePayloadScratch payload;

  explicit MapleBatchScratch(int concurrent) {
    const size_t n = static_cast<size_t>(std::max(1, concurrent));
    searches.reserve(n);
    pending.reserve(n);
    roots.reserve(n);
  }

  void clear_for_roots() {
    searches.clear();
    pending.clear();
    roots.clear();
  }
};

static std::pair<int, std::vector<Descriptor>> maple_action_signature(
    const GameState& st) {
  if (st.has_pending())
    return {st.pending.context, st.pending.options};
  return {-1, legal_main(st)};
}

static py::tuple maple_feature_payload(const std::vector<GameState>& worlds,
                                       const std::vector<Descriptor>& desc,
                                       int ctx, int n_act,
                                       const std::vector<int>& deck) {
  FeaturePayloadScratch batch;
  batch.reserve(worlds.size(), n_act);
  for (const GameState& st : worlds)
    batch.append(st, desc, ctx, n_act, n_act, deck);
  return batch.to_python();
}

static void normalize_maple_prior(std::vector<double>& p) {
  double sum = 0.0;
  for (double x : p) sum += x;
  if (sum > 0.0 && std::isfinite(sum)) {
    for (double& x : p) x /= sum;
    return;
  }
  double u = p.empty() ? 0.0 : 1.0 / static_cast<double>(p.size());
  for (double& x : p) x = u;
}

static std::vector<double> softmax_logits(const py::handle& logits_obj,
                                          int n_act) {
  py::sequence logits = logits_obj.cast<py::sequence>();
  std::vector<double> out(n_act, 0.0);
  double mx = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < n_act; ++i) {
    out[i] = logits[i].cast<double>();
    mx = std::max(mx, out[i]);
  }
  double sum = 0.0;
  for (double& x : out) {
    x = std::exp(x - mx);
    sum += x;
  }
  if (sum > 0.0) {
    for (double& x : out) x /= sum;
  }
  return out;
}

static double expand_maple_native_node(std::vector<MapleNativeNode>& tree,
                                       int node_idx, int root_player,
                                       const std::vector<int>& deck,
                                       const py::function& eval_payload) {
  MapleNativeNode& node = tree[node_idx];
  if (node.expanded) return node.value;

  double value_sum = 0.0;
  int value_count = 0;
  int first_live = -1;
  for (int i = 0; i < static_cast<int>(node.worlds.size()); ++i) {
    const GameState& st = node.worlds[i];
    if (st.result >= 0) {
      value_sum += puct_term_value(st, root_player);
      ++value_count;
    } else if (first_live < 0) {
      first_live = i;
    }
  }

  if (first_live < 0) {
    node.terminal = true;
    node.expanded = true;
    node.value = value_count ? value_sum / value_count : 0.0;
    node.worlds.clear();
    return node.value;
  }

  auto [ctx, desc] = maple_action_signature(node.worlds[first_live]);
  if (desc.empty()) {
    node.terminal = true;
    node.expanded = true;
    node.value = value_count ? value_sum / value_count : 0.0;
    node.worlds.clear();
    return node.value;
  }
  if (static_cast<int>(desc.size()) > RL_MAX_ACTIONS) desc.resize(RL_MAX_ACTIONS);

  std::vector<GameState> compatible;
  compatible.reserve(node.worlds.size());
  for (GameState& st : node.worlds) {
    if (st.result >= 0) continue;
    auto [st_ctx, st_desc] = maple_action_signature(st);
    if (static_cast<int>(st_desc.size()) > RL_MAX_ACTIONS)
      st_desc.resize(RL_MAX_ACTIONS);
    if (st_ctx == ctx && descriptors_equal(st_desc, desc))
      compatible.push_back(std::move(st));
  }
  if (compatible.empty()) {
    node.terminal = true;
    node.expanded = true;
    node.value = value_count ? value_sum / value_count : 0.0;
    node.worlds.clear();
    return node.value;
  }

  node.worlds = std::move(compatible);
  node.ctx = ctx;
  node.descriptors = std::move(desc);
  node.step_opts = node.ctx >= 0 ? rl_options(node.worlds[0])
                                 : rl_options_from_descriptors(node.descriptors);
  node.mover = node.worlds[0].yourIndex;
  node.n_act = static_cast<int>(node.descriptors.size());
  node.P.assign(node.n_act, 0.0);
  node.childN.assign(node.n_act, 0.0);
  node.childW.assign(node.n_act, 0.0);
  node.child.assign(node.n_act, -1);

  py::sequence outs =
      eval_payload(maple_feature_payload(node.worlds, node.descriptors, node.ctx,
                                         node.n_act, deck))
          .cast<py::sequence>();
  for (size_t i = 0; i < node.worlds.size(); ++i) {
    const GameState& st = node.worlds[i];
    py::tuple item = outs[static_cast<py::ssize_t>(i)].cast<py::tuple>();
    double v = item[0].cast<double>();
    value_sum += (st.yourIndex == root_player) ? v : -v;
    ++value_count;
    std::vector<double> prior = softmax_logits(item[1], node.n_act);
    for (int a = 0; a < node.n_act; ++a) node.P[a] += prior[a];
  }
  normalize_maple_prior(node.P);
  node.value = value_count ? value_sum / value_count : 0.0;
  node.expanded = true;
  return node.value;
}

static int select_maple_native_action(const MapleNativeNode& node,
                                      int root_player, double c_puct) {
  double total_n = 0.0;
  for (double n : node.childN) total_n += n;
  double c = c_puct * std::sqrt(total_n + 1.0);
  bool flip = node.mover != root_player;
  int best_i = 0;
  double best_u = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < node.n_act; ++i) {
    double n = node.childN[i];
    double q = n > 0.0 ? node.childW[i] / n : 0.0;
    if (flip) q = -q;
    double u = q + c * node.P[i] / (1.0 + n);
    if (u > best_u) {
      best_u = u;
      best_i = i;
    }
  }
  return best_i;
}

static std::vector<GameState> step_maple_native_worlds(
    const std::vector<GameState>& worlds, const Descriptor& action_desc,
    uint64_t& rng) {
  std::vector<GameState> out;
  out.reserve(worlds.size());
  for (const GameState& st : worlds) {
    if (st.result >= 0) continue;
    auto [ctx, desc] = maple_action_signature(st);
    if (static_cast<int>(desc.size()) > RL_MAX_ACTIONS) desc.resize(RL_MAX_ACTIONS);
    int action = -1;
    for (int i = 0; i < static_cast<int>(desc.size()); ++i) {
      if (descriptor_equal(desc[i], action_desc)) {
        action = i;
        break;
      }
    }
    if (action < 0) continue;
    GameState cs = st;
    rng = lcg_next(rng);
    cs.rng = rng;
    uint64_t step_rng = rng;
    RlOptionSet opts = ctx >= 0 ? rl_options(cs) : rl_options_from_descriptors(desc);
    rl_step_cached(cs, opts, action, step_rng);
    out.push_back(std::move(cs));
  }
  return out;
}

static bool prepare_maple_node_for_eval(std::vector<MapleNativeNode>& tree,
                                        int node_idx, int root_player,
                                        MaplePendingNodeEval& pending) {
  MapleNativeNode& node = tree[node_idx];
  if (node.expanded) return false;

  pending.terminal_value_sum = 0.0;
  pending.terminal_value_count = 0;
  int first_live = -1;
  for (int i = 0; i < static_cast<int>(node.worlds.size()); ++i) {
    const GameState& st = node.worlds[i];
    if (st.result >= 0) {
      pending.terminal_value_sum += puct_term_value(st, root_player);
      ++pending.terminal_value_count;
    } else if (first_live < 0) {
      first_live = i;
    }
  }

  auto finish_terminal = [&]() {
    node.terminal = true;
    node.expanded = true;
    node.value = pending.terminal_value_count
                     ? pending.terminal_value_sum /
                           static_cast<double>(pending.terminal_value_count)
                     : 0.0;
    node.worlds.clear();
  };

  if (first_live < 0) {
    finish_terminal();
    return false;
  }

  auto [ctx, desc] = maple_action_signature(node.worlds[first_live]);
  if (desc.empty()) {
    finish_terminal();
    return false;
  }
  if (static_cast<int>(desc.size()) > RL_MAX_ACTIONS) desc.resize(RL_MAX_ACTIONS);

  std::vector<GameState> compatible;
  compatible.reserve(node.worlds.size());
  for (GameState& st : node.worlds) {
    if (st.result >= 0) continue;
    auto [st_ctx, st_desc] = maple_action_signature(st);
    if (static_cast<int>(st_desc.size()) > RL_MAX_ACTIONS)
      st_desc.resize(RL_MAX_ACTIONS);
    if (st_ctx == ctx && descriptors_equal(st_desc, desc))
      compatible.push_back(std::move(st));
  }
  if (compatible.empty()) {
    finish_terminal();
    return false;
  }

  node.worlds = std::move(compatible);
  node.ctx = ctx;
  node.descriptors = std::move(desc);
  node.step_opts = node.ctx >= 0 ? rl_options(node.worlds[0])
                                 : rl_options_from_descriptors(node.descriptors);
  node.mover = node.worlds[0].yourIndex;
  node.n_act = static_cast<int>(node.descriptors.size());
  node.P.assign(node.n_act, 0.0);
  node.childN.assign(node.n_act, 0.0);
  node.childW.assign(node.n_act, 0.0);
  node.child.assign(node.n_act, -1);
  return true;
}

static void eval_maple_pending_nodes(
    std::vector<MapleNativeSearch>& searches,
    std::vector<MaplePendingNodeEval>& pending,
    const std::vector<int>& deck, const py::function& eval_payload,
    FeaturePayloadScratch& batch, int row_cap) {
  if (pending.empty()) return;
  auto eval_range = [&](size_t begin, size_t end) {
    int width = 1;
    size_t rows = 0;
    for (size_t pi = begin; pi < end; ++pi) {
      const auto& p = pending[pi];
      const MapleNativeNode& node = searches[p.search_idx].tree[p.node_idx];
      width = std::max(width, node.n_act);
      rows += node.worlds.size();
    }

    batch.clear();
    batch.reserve(rows, width);
    for (size_t pi = begin; pi < end; ++pi) {
      const auto& p = pending[pi];
      const MapleNativeNode& node = searches[p.search_idx].tree[p.node_idx];
      for (const GameState& st : node.worlds)
        batch.append(st, node.descriptors, node.ctx, node.n_act, width, deck);
    }

    py::sequence outs = eval_payload(batch.to_python()).cast<py::sequence>();
    py::ssize_t row = 0;
    for (size_t pi = begin; pi < end; ++pi) {
      const auto& p = pending[pi];
      MapleNativeSearch& search = searches[p.search_idx];
      MapleNativeNode& node = search.tree[p.node_idx];
      double value_sum = p.terminal_value_sum;
      int value_count = p.terminal_value_count;
      std::fill(node.P.begin(), node.P.end(), 0.0);
      for (const GameState& st : node.worlds) {
        py::tuple item = outs[row++].cast<py::tuple>();
        double v = item[0].cast<double>();
        value_sum += (st.yourIndex == search.root_player) ? v : -v;
        ++value_count;
        std::vector<double> prior = softmax_logits(item[1], node.n_act);
        for (int a = 0; a < node.n_act; ++a) node.P[a] += prior[a];
      }
      normalize_maple_prior(node.P);
      node.value = value_count ? value_sum / static_cast<double>(value_count)
                               : 0.0;
      node.expanded = true;
    }
  };

  if (row_cap <= 0) {
    eval_range(0, pending.size());
    return;
  }

  size_t begin = 0;
  size_t rows = 0;
  for (size_t i = 0; i < pending.size(); ++i) {
    const MapleNativeNode& node =
        searches[pending[i].search_idx].tree[pending[i].node_idx];
    size_t node_rows = node.worlds.size();
    if (i > begin && rows + node_rows > static_cast<size_t>(row_cap)) {
      eval_range(begin, i);
      begin = i;
      rows = 0;
    }
    rows += node_rows;
  }
  if (begin < pending.size()) eval_range(begin, pending.size());
}

static PuctResult rl_maple_puct_callback(
    const GameState& root_state, const std::vector<int>& deck, int n_sims,
    uint64_t seed, int dets, double c_puct,
    const py::function& eval_payload) {
  PuctResult out;
  out.state = root_state;
  out.terminal = root_state.result >= 0;
  if (out.terminal) return out;

  int root_player = root_state.yourIndex;
  std::vector<GameState> worlds;
  int k = std::max(1, dets);
  worlds.reserve(k);
  for (int i = 0; i < k; ++i) {
    uint64_t dseed =
        (seed ? seed : 1ULL) + (static_cast<uint64_t>(i) + 1ULL) *
                                  0x9E3779B97F4A7C15ULL;
    worlds.push_back(rl_determinize_decklist(root_state, deck, deck,
                                             root_player, dseed));
  }

  std::vector<MapleNativeNode> tree;
  tree.reserve(static_cast<size_t>(std::max(8, n_sims + 1)));
  tree.emplace_back(std::move(worlds));
  expand_maple_native_node(tree, 0, root_player, deck, eval_payload);

  uint64_t rng = seed ? seed : 1ULL;
  for (int sim = 0; sim < std::max(0, n_sims); ++sim) {
    int node_idx = 0;
    std::vector<std::pair<int, int>> path;
    double v_root = 0.0;
    while (true) {
      MapleNativeNode& node = tree[node_idx];
      if (node.terminal || node.n_act <= 0) {
        v_root = node.value;
        break;
      }
      int action = select_maple_native_action(node, root_player, c_puct);
      path.emplace_back(node_idx, action);
      int child_idx = node.child[action];
      if (child_idx < 0) {
        Descriptor action_desc = node.descriptors[action];
        std::vector<GameState> child_worlds =
            step_maple_native_worlds(node.worlds, action_desc, rng);
        tree.emplace_back(std::move(child_worlds));
        child_idx = static_cast<int>(tree.size()) - 1;
        tree[node_idx].child[action] = child_idx;
        v_root =
            expand_maple_native_node(tree, child_idx, root_player, deck,
                                     eval_payload);
        break;
      }
      node_idx = child_idx;
    }
    for (auto [idx, action] : path) {
      tree[idx].childN[action] += 1.0;
      tree[idx].childW[action] += v_root;
    }
  }

  MapleNativeNode& root = tree[0];
  out.terminal = root.terminal;
  out.n_act = root.n_act;
  out.childN = root.childN;
  out.childW = root.childW;
  out.step_opts = root.step_opts;
  out.ctx = root.ctx;
  out.opts = descriptors_to_py(root.descriptors);
  out.canon = canonical(root_state);
  return out;
}

static NativeTrainRecord make_maple_train_record(const PuctResult& root,
                                                 const std::vector<int>& deck) {
  NativeTrainRecord rec;
  rec.features = feature_record_cached(root.state, root.step_opts, root.ctx,
                                       root.n_act, deck, RL_MAX_ACTIONS);
  rec.n_act = root.n_act;
  rec.mover = root.state.yourIndex;
  rec.hval = rl_heuristic_value(root.state);
  rec.pi.assign(RL_MAX_ACTIONS, 0.0f);

  double visit_sum = 0.0;
  for (int i = 0; i < root.n_act; ++i) visit_sum += root.childN[i];
  if (visit_sum > 0.0) {
    for (int i = 0; i < root.n_act; ++i)
      rec.pi[i] = static_cast<float>(root.childN[i] / visit_sum);
  } else if (root.n_act > 0) {
    float u = 1.0f / static_cast<float>(root.n_act);
    for (int i = 0; i < root.n_act; ++i) rec.pi[i] = u;
  }
  return rec;
}

static int choose_maple_visit_action(const PuctResult& root, int step,
                                     int temp_moves, std::mt19937& rng) {
  int n = root.n_act;
  if (n <= 0) return 0;
  double sum = 0.0;
  for (int i = 0; i < n; ++i) sum += root.childN[i];
  if (step < temp_moves && sum > 0.0) {
    std::uniform_real_distribution<double> dist(0.0, sum);
    double r = dist(rng);
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
      acc += root.childN[i];
      if (r <= acc) return i;
    }
  }
  int best = 0;
  double best_n = -1.0;
  for (int i = 0; i < n; ++i) {
    if (root.childN[i] > best_n) {
      best_n = root.childN[i];
      best = i;
    }
  }
  return best;
}

static void backup_maple_path(MapleNativeSearch& search,
                              const std::vector<std::pair<int, int>>& path,
                              double value_root) {
  for (auto [idx, action] : path) {
    MapleNativeNode& node = search.tree[idx];
    node.childN[action] += 1.0;
    node.childW[action] += value_root;
  }
}

static void collect_maple_sim_leaf(
    std::vector<MapleNativeSearch>& searches, int search_idx, double c_puct,
    std::vector<MaplePendingNodeEval>& pending) {
  MapleNativeSearch& search = searches[search_idx];
  int node_idx = 0;
  std::vector<std::pair<int, int>> path;
  while (true) {
    MapleNativeNode& node = search.tree[node_idx];
    if (node.terminal || node.n_act <= 0) {
      backup_maple_path(search, path, node.value);
      return;
    }

    int action = select_maple_native_action(node, search.root_player, c_puct);
    path.emplace_back(node_idx, action);
    int child_idx = node.child[action];
    if (child_idx < 0) {
      Descriptor action_desc = node.descriptors[action];
      std::vector<GameState> child_worlds =
          step_maple_native_worlds(node.worlds, action_desc, search.rng);
      search.tree.emplace_back(std::move(child_worlds));
      child_idx = static_cast<int>(search.tree.size()) - 1;
      search.tree[node_idx].child[action] = child_idx;

      MaplePendingNodeEval pe;
      pe.search_idx = search_idx;
      pe.node_idx = child_idx;
      pe.path = std::move(path);
      bool needs_eval = prepare_maple_node_for_eval(
          search.tree, child_idx, search.root_player, pe);
      if (needs_eval) {
        pending.push_back(std::move(pe));
      } else {
        backup_maple_path(search, pe.path, search.tree[child_idx].value);
      }
      return;
    }

    node_idx = child_idx;
    if (!search.tree[node_idx].expanded) {
      MaplePendingNodeEval pe;
      pe.search_idx = search_idx;
      pe.node_idx = node_idx;
      pe.path = std::move(path);
      bool needs_eval = prepare_maple_node_for_eval(
          search.tree, node_idx, search.root_player, pe);
      if (needs_eval) {
        pending.push_back(std::move(pe));
      } else {
        backup_maple_path(search, pe.path, search.tree[node_idx].value);
      }
      return;
    }
  }
}

static void run_batched_maple_roots(
    const std::vector<NativeSelfplayGame>& pool, const std::vector<int>& deck,
    int n_sims, uint64_t seed, int dets, double c_puct,
    const py::function& eval_payload, int row_cap,
    MapleBatchScratch& scratch) {
  scratch.clear_for_roots();
  std::vector<MapleNativeSearch>& searches = scratch.searches;
  std::vector<MaplePendingNodeEval>& pending = scratch.pending;

  for (int gi = 0; gi < static_cast<int>(pool.size()); ++gi) {
    const NativeSelfplayGame& g = pool[gi];
    MapleNativeSearch search;
    search.root_player = g.state.yourIndex;
    search.rng = g.seed ^ (static_cast<uint64_t>(g.step) * 2654435761ULL);
    int k = std::max(1, dets);
    std::vector<GameState> worlds;
    worlds.reserve(k);
    for (int i = 0; i < k; ++i) {
      uint64_t dseed =
          (seed ? seed : 1ULL) ^ g.seed ^
          ((static_cast<uint64_t>(gi) + 1ULL) * 0xBF58476D1CE4E5B9ULL) ^
          ((static_cast<uint64_t>(i) + 1ULL) * 0x9E3779B97F4A7C15ULL);
      worlds.push_back(rl_determinize_decklist(g.state, deck, deck,
                                               search.root_player, dseed));
    }
    search.tree.reserve(static_cast<size_t>(std::max(8, n_sims + 1)));
    search.tree.emplace_back(std::move(worlds));
    searches.push_back(std::move(search));

    MaplePendingNodeEval pe;
    pe.search_idx = gi;
    pe.node_idx = 0;
    bool needs_eval = prepare_maple_node_for_eval(
        searches.back().tree, 0, searches.back().root_player, pe);
    if (needs_eval) pending.push_back(std::move(pe));
  }

  eval_maple_pending_nodes(searches, pending, deck, eval_payload,
                           scratch.payload, row_cap);

  for (int sim = 0; sim < std::max(0, n_sims); ++sim) {
    pending.clear();
    for (int si = 0; si < static_cast<int>(searches.size()); ++si)
      collect_maple_sim_leaf(searches, si, c_puct, pending);
    eval_maple_pending_nodes(searches, pending, deck, eval_payload,
                             scratch.payload, row_cap);
    for (const auto& pe : pending)
      backup_maple_path(searches[pe.search_idx], pe.path,
                        searches[pe.search_idx].tree[pe.node_idx].value);
  }

  scratch.roots.clear();
  scratch.roots.reserve(searches.size());
  for (int si = 0; si < static_cast<int>(searches.size()); ++si) {
    const NativeSelfplayGame& g = pool[si];
    MapleNativeNode& root = searches[si].tree[0];
    PuctResult out;
    out.state = g.state;
    out.terminal = root.terminal;
    out.n_act = root.n_act;
    out.childN = root.childN;
    out.childW = root.childW;
    out.step_opts = root.step_opts;
    out.ctx = root.ctx;
    out.opts = descriptors_to_py(root.descriptors);
    out.canon = canonical(g.state);
    scratch.roots.push_back(std::move(out));
  }
}

static py::tuple rl_selfplay_maple_callback(
    const std::vector<int>& deck, int total_games, int concurrent, int n_sims,
    uint64_t seed, int temp_moves, double vboot, int dets, double c_puct,
    const py::function& eval_payload, int max_steps, int row_cap) {
  auto t_total = NativeSelfplayClock::now();
  py::list data;
  std::map<int, long> results;
  NativeSelfplayStats stats;
  if (total_games <= 0) {
    py::dict empty;
    stats.total_ns = elapsed_native_ns(t_total);
    return py::make_tuple(data, empty, native_selfplay_stats_to_dict(stats));
  }

  concurrent = std::max(1, std::min(concurrent, total_games));
  n_sims = std::max(0, n_sims);
  dets = std::max(1, dets);
  max_steps = std::max(1, max_steps);
  row_cap = std::max(0, row_cap);

  int started = 0;
  std::vector<NativeSelfplayGame> pool;
  pool.reserve(concurrent);
  for (; started < concurrent; ++started)
    pool.push_back(make_native_game(deck, seed, started));
  NativeSelfplayScratch scratch(concurrent);
  MapleBatchScratch maple_scratch(concurrent);

  while (!pool.empty()) {
    run_batched_maple_roots(pool, deck, n_sims, seed, dets, c_puct,
                            eval_payload, row_cap, maple_scratch);
    scratch.next_pool.clear();
    for (int gi = 0; gi < static_cast<int>(pool.size()); ++gi) {
      NativeSelfplayGame& g = pool[gi];
      const PuctResult& root = maple_scratch.roots[gi];
      if (root.terminal || root.n_act == 0) {
        append_finished_game(data, results, g, vboot);
        ++stats.games_finished;
        if (started < total_games)
          scratch.next_pool.push_back(make_native_game(deck, seed, started++));
        continue;
      }

      g.records.push_back(make_maple_train_record(root, deck));
      ++stats.records;
      int action = choose_maple_visit_action(root, g.step, temp_moves,
                                             g.choice_rng);
      rl_step(g.state, action, g.seed);
      ++g.step;

      if (g.state.result >= 0 || g.step >= max_steps) {
        if (g.state.result < 0) g.state.result = 2;
        append_finished_game(data, results, g, vboot);
        ++stats.games_finished;
        if (started < total_games)
          scratch.next_pool.push_back(make_native_game(deck, seed, started++));
      } else {
        scratch.next_pool.push_back(std::move(g));
      }
    }
    pool.swap(scratch.next_pool);
  }

  py::dict res;
  for (const auto& kv : results) res[py::int_(kv.first)] = kv.second;
  stats.total_ns = elapsed_native_ns(t_total);
  return py::make_tuple(data, res, native_selfplay_stats_to_dict(stats));
}

static py::dict debug_action_dict(const dbg::DebugAction& action) {
  py::dict d;
  d["key"] = action.key;
  d["label"] = action.label;
  return d;
}

static py::list debug_actions_to_dicts(
    const std::vector<dbg::DebugAction>& actions) {
  py::list out;
  for (const auto& action : actions) out.append(debug_action_dict(action));
  return out;
}

static std::vector<int> debug_action_keys(
    const std::vector<dbg::DebugAction>& actions) {
  std::vector<int> keys;
  keys.reserve(actions.size());
  for (const auto& action : actions) keys.push_back(action.key);
  return keys;
}

static py::array_t<float> debug_observation_array(const dbg::DebugGameState& st,
                                                  int perspective) {
  std::vector<float> obs = dbg::debug_observe(st, perspective);
  return vector_to_numpy(obs);
}

static py::list debug_returns_list(const dbg::DebugGameState& st) {
  py::list out;
  out.append(dbg::debug_terminal_value(st, 0));
  out.append(dbg::debug_terminal_value(st, 1));
  return out;
}

static dbg::DebugEvaluator make_debug_evaluator(const py::object& eval_callback) {
  if (eval_callback.is_none()) return dbg::DebugEvaluator();
  py::function callback = eval_callback.cast<py::function>();
  return [callback](const std::vector<dbg::DebugEvalInput>& inputs) {
    py::gil_scoped_acquire gil;
    py::list reqs;
    for (const auto& in : inputs) {
      py::dict req;
      req["game"] = dbg::debug_game_name(in.kind);
      req["player"] = in.player;
      req["observation"] = vector_to_numpy(in.observation);
      req["actions"] = debug_actions_to_dicts(in.actions);
      req["action_keys"] = debug_action_keys(in.actions);
      req["n_actions"] = static_cast<int>(in.actions.size());
      reqs.append(req);
    }

    py::sequence raw = callback(reqs).cast<py::sequence>();
    if (py::len(raw) != inputs.size())
      throw std::runtime_error("debug eval callback returned the wrong row count");

    std::vector<dbg::DebugEvalOutput> outs;
    outs.reserve(inputs.size());
    for (py::handle item : raw) {
      dbg::DebugEvalOutput out;
      py::object obj = py::reinterpret_borrow<py::object>(item);
      py::object logits_obj;
      if (py::isinstance<py::dict>(obj)) {
        py::dict d = obj.cast<py::dict>();
        out.value = d["value"].cast<double>();
        logits_obj = d.contains("logits")
                         ? py::reinterpret_borrow<py::object>(d["logits"])
                         : py::object(py::list());
      } else {
        py::sequence seq = obj.cast<py::sequence>();
        if (py::len(seq) != 2)
          throw std::runtime_error(
              "debug eval callback rows must be dicts or (value, logits)");
        out.value = seq[0].cast<double>();
        logits_obj = py::reinterpret_borrow<py::object>(seq[1]);
      }
      py::sequence logits = logits_obj.cast<py::sequence>();
      out.logits.reserve(static_cast<size_t>(py::len(logits)));
      for (py::handle v : logits) out.logits.push_back(v.cast<double>());
      outs.push_back(std::move(out));
    }
    return outs;
  };
}

PYBIND11_MODULE(ptcg_engine, m) {
  m.doc() = "PTCG fast engine - custom rules-equivalent C++ engine for RL self-play";
  m.def("version", &ptcg::version);
  m.def("add", &ptcg::add);

  py::class_<GameState>(m, "GameState")
      .def_readonly("turn", &GameState::turn)
      .def_readonly("turnActionCount", &GameState::turnActionCount)
      .def_readonly("yourIndex", &GameState::yourIndex)
      .def_readonly("result", &GameState::result)
      .def_readwrite("rng", &GameState::rng)  // settable so MCTS can reseed per branch
      .def_readwrite("collectLogs", &GameState::collectLogs)
      .def("has_pending", &GameState::has_pending);

  py::class_<NativeCgBattle>(m, "NativeCgBattle")
      .def(py::init<>())
      .def("start", &NativeCgBattle::start, py::arg("deck0"), py::arg("deck1"),
           py::arg("seed"),
           "Start a native cg-compatible battle and return "
           "(observation, context, descriptors, logs).")
      .def("observation", &NativeCgBattle::observation,
           "Return the current cg-compatible observation tuple.")
      .def("select", &NativeCgBattle::select, py::arg("selection"),
           "Apply one cg selection and return "
           "(observation, context, descriptors, logs).")
      .def("finish", &NativeCgBattle::finish)
      .def_property_readonly("state", &NativeCgBattle::mutable_state,
                             py::return_value_policy::reference_internal)
      .def_readonly("seed", &NativeCgBattle::seed)
      .def_readonly("generation", &NativeCgBattle::generation)
      .def_readonly("active", &NativeCgBattle::active)
      .def("clone_state", &NativeCgBattle::clone_state)
      .def("transient_snapshot", &NativeCgBattle::transient_snapshot)
      .def("canonical", &NativeCgBattle::canonical_state);

  m.def(
      "load_state",
      [](const py::dict& cur, uint64_t seed, py::object main_options) {
        GameState st = load_state(cur);
        if (!main_options.is_none())
          reconstruct_main_from_cabt_options(st, descriptors_from_py(main_options));
        if (seed) st.rng = seed;  // RNG for coin flips (leaves draws in replay mode)
        return st;
      },
      py::arg("current"), py::arg("seed") = 0,
      py::arg("main_options") = py::none(),
      "Parse a cabt `current` state dict into a GameState; optional seed sets "
      "the free-running RNG (coin flips). Optional MAIN descriptors from cabt "
      "can reconstruct transient one-turn ability usage that cabt omits from "
      "`current`.");
  m.def("new_game", &ptcg::new_game, py::arg("deck0"), py::arg("deck1"),
        py::arg("seed"), py::arg("collect_logs") = false,
        "Deal a fresh free-running game (shuffle/setup) at the first turn-1 MAIN.");
  m.def("canonical", &canonical, py::arg("state"),
        "Canonical projection of a GameState (matches oracle.canonical_state).");
  m.def("cg_observation", &cg_observation, py::arg("state"),
        "CABT/cg-shaped observation dict built directly from a native GameState.");
  m.def("cg_observation_with_view", &cg_observation_with_view, py::arg("state"),
        "Return (cg_observation, context, descriptors) using one legal-action pass.");
  m.def("cg_select_step", &cg_select_step, py::arg("state"), py::arg("selection"),
        "Apply/resolve one cg selection and return "
        "(cg_observation, context, descriptors, logs) in one native call.");
  m.def("native_logs", &native_logs_to_py, py::arg("state"),
        "Engine-emitted CABT-shaped logs since the last clear.");
  m.def("clear_native_logs", [](GameState& st) { st.nativeLogs.clear(); },
        py::arg("state"),
        "Clear engine-emitted native logs.");
  m.def("_debug_hidden_zones", &debug_hidden_zones, py::arg("state"),
        "Internal test helper exposing hidden-zone knownness and contents.");
  m.def("debug_search_summary", &debug_search_summary, py::arg("state"),
        "Internal test helper exposing hidden-zone and suspended-search "
        "transient summaries.");
  m.def("native_state_summary", &debug_search_summary, py::arg("state"),
        "Backward-compatible alias for debug_search_summary.");
  m.def("legal_main", &legal_main_py, py::arg("state"),
        "Structural MAIN-phase legal option descriptors for the acting player.");
  m.def(
      "action_view",
      [](const GameState& st) {
        CgActionView view = cg_action_view(st);
        return py::make_tuple(view.ctx, descriptors_to_py(view.descriptors));
      },
      py::arg("state"),
      "Return (context, descriptors) for the current native cg decision.");
  m.def(
      "reconstruct_main",
      [](GameState& st, py::object main_options) {
        reconstruct_main_from_cabt_options(st, descriptors_from_py(main_options));
      },
      py::arg("state"), py::arg("main_options"),
      "Reconstruct transient MAIN-only flags from cabt legal descriptors on an "
      "already-loaded GameState.");
  m.def(
      "copy_search_transients",
      [](GameState& dst, const GameState& src) {
        apply_search_transients_from_py(dst, search_transients_to_py(src));
      },
      py::arg("dst"), py::arg("src"),
      "Copy suspended native decision/effect transients onto a state rebuilt "
      "from CABT current plus search hidden-zone predictions.");
  m.def("search_transient_snapshot", &search_transients_to_py, py::arg("state"),
        "Serialize portable suspended native decision/effect transients for "
        "Observation.search_begin_input.");
  m.def("restore_search_transients", &apply_search_transients_from_py,
        py::arg("state"), py::arg("snapshot"),
        "Restore suspended native decision/effect transients from a portable "
        "search_begin_input snapshot.");
  m.def(
      "apply",
      [](GameState& st, const py::tuple& desc, const std::vector<int>& tape) {
        apply(st, parse_action(desc), tape);
      },
      py::arg("state"), py::arg("descriptor"), py::arg("tape") = std::vector<int>{},
      "Apply a semantic action descriptor to the state, mutating it in place. "
      "`tape` supplies drawn/revealed card ids and tagged coin outcomes for "
      "replay, plus optional tagged hand indices for duplicate-card replay.");
  m.def("clone", [](const GameState& s) { return s; }, py::arg("state"),
        "Deep-copy a GameState (cheap; used for search node expansion).");
  m.def(
      "bench",
      [](const GameState& base, int op, long n) {
        volatile long sink = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < n; ++i) {
          if (op == 0) {  // generate legal MAIN moves
            auto v = legal_main(base);
            sink += static_cast<long>(v.size());
          } else if (op == 1) {  // copy the state
            GameState s = base;
            sink += s.turn;
          } else if (op == 2) {  // copy + apply one step (END)
            GameState s = base;
            Action a;
            a.kind = ACT_END;
            apply(s, a);
            sink += s.turn;
          } else if (op == 3) {  // copy + legal + apply: a search-node expansion
            GameState s = base;
            auto v = legal_main(s);
            Action a;
            a.kind = ACT_END;
            apply(s, a);
            sink += static_cast<long>(v.size());
          }
        }
        auto t1 = std::chrono::steady_clock::now();
        double total_ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        return total_ns / static_cast<double>(n);
      },
      py::arg("state"), py::arg("op"), py::arg("n"),
      "Native ns/op for: 0=legal_main 1=clone 2=clone+apply 3=clone+legal+apply.");
  m.def("pending_decision", &pending_decision, py::arg("state"),
        "Current pending sub-decision {context,minCount,maxCount,options} or None.");
  m.def(
      "resolve",
      [](GameState& st, const std::vector<int>& sel, const std::vector<int>& tape) {
        resolve(st, sel, tape);
      },
      py::arg("state"), py::arg("selection"), py::arg("tape") = std::vector<int>{},
      "Resolve the pending sub-decision with chosen option indices; `tape` "
      "supplies drawn card ids for sub-decisions that draw.");

  // --- generic debug-game search lab --------------------------------------
  py::class_<dbg::DebugAction>(m, "DebugAction")
      .def_property_readonly("key", [](const dbg::DebugAction& a) { return a.key; })
      .def_property_readonly("label",
                             [](const dbg::DebugAction& a) { return a.label; })
      .def("__repr__", [](const dbg::DebugAction& a) {
        return "<DebugAction " + std::to_string(a.key) + ":" + a.label + ">";
      });

  py::class_<dbg::DebugSearchStats>(m, "DebugSearchStats")
      .def_readonly("game", &dbg::DebugSearchStats::game)
      .def_readonly("algo", &dbg::DebugSearchStats::algo)
      .def_readonly("sims", &dbg::DebugSearchStats::sims)
      .def_readonly("dets", &dbg::DebugSearchStats::dets)
      .def_readonly("sampled_worlds", &dbg::DebugSearchStats::sampled_worlds)
      .def_readonly("invalid_worlds", &dbg::DebugSearchStats::invalid_worlds)
      .def_readonly("eval_calls", &dbg::DebugSearchStats::eval_calls)
      .def_readonly("eval_rows", &dbg::DebugSearchStats::eval_rows)
      .def_readonly("terminal_evals", &dbg::DebugSearchStats::terminal_evals);

  py::class_<dbg::DebugSearchResult>(m, "DebugSearchResult")
      .def_property_readonly("terminal",
                             [](const dbg::DebugSearchResult& r) { return r.terminal; })
      .def_property_readonly("root_player",
                             [](const dbg::DebugSearchResult& r) { return r.root_player; })
      .def_property_readonly("value",
                             [](const dbg::DebugSearchResult& r) { return r.value; })
      .def_property_readonly("actions",
                             [](const dbg::DebugSearchResult& r) { return r.actions; })
      .def_property_readonly("childN",
                             [](const dbg::DebugSearchResult& r) {
                               return vector_to_array(r.childN);
                             })
      .def_property_readonly("childW",
                             [](const dbg::DebugSearchResult& r) {
                               return vector_to_array(r.childW);
                             })
      .def_property_readonly("stats",
                             [](const dbg::DebugSearchResult& r) { return r.stats; });

  py::class_<dbg::DebugGameState>(m, "DebugGameState",
                                  "Small native debug-game state with an "
                                  "OpenSpiel-like method subset.")
      .def_property_readonly("game",
                             [](const dbg::DebugGameState& st) {
                               return dbg::debug_game_name(st.kind);
                             })
      .def_property_readonly("currentPlayer",
                             [](const dbg::DebugGameState& st) {
                               return st.current_player;
                             })
      .def_property_readonly("result",
                             [](const dbg::DebugGameState& st) { return st.result; })
      .def_property_readonly("kuhn_history",
                             [](const dbg::DebugGameState& st) {
                               return st.kuhn_history;
                             })
      .def("clone", [](const dbg::DebugGameState& st) { return st; })
      .def("current_player",
           [](const dbg::DebugGameState& st) { return st.current_player; })
      .def("is_terminal",
           [](const dbg::DebugGameState& st) { return st.result >= 0; })
      .def("legal_actions",
           [](const dbg::DebugGameState& st) {
             return debug_action_keys(dbg::debug_legal_actions(st));
           })
      .def("action_view",
           [](const dbg::DebugGameState& st) { return dbg::debug_legal_actions(st); })
      .def("child",
           [](const dbg::DebugGameState& st, int action, uint64_t seed) {
             return dbg::debug_step_state(st, action, seed);
           },
           py::arg("action"), py::arg("seed") = 0)
      .def("apply_action",
           [](dbg::DebugGameState& st, int action, uint64_t seed) {
             st = dbg::debug_step_state(st, action, seed);
           },
           py::arg("action"), py::arg("seed") = 0)
      .def("observation_tensor", &debug_observation_array,
           py::arg("player") = -1)
      .def("information_state_tensor", &debug_observation_array,
           py::arg("player") = -1)
      .def("returns", &debug_returns_list)
      .def("player_return",
           [](const dbg::DebugGameState& st, int player) {
             return dbg::debug_terminal_value(st, player);
           })
      .def("resample_from_infostate",
           [](const dbg::DebugGameState& st, int player, uint64_t seed) {
             return dbg::debug_determinize(st, player, seed);
           },
           py::arg("player"), py::arg("seed") = 1)
      .def("action_to_string",
           [](const dbg::DebugGameState& st, int player, int action) {
             (void)player;
             const auto actions = dbg::debug_legal_actions(st);
             for (const auto& a : actions) {
               if (a.key == action) return a.label;
             }
             return std::string("<illegal ") + std::to_string(action) + ">";
           })
      .def("__str__", &dbg::debug_public_state)
      .def("__repr__", [](const dbg::DebugGameState& st) {
        return "<DebugGameState " + dbg::debug_public_state(st) + ">";
      });

  m.def("debug_games", &dbg::debug_games,
        "Registered native debug games.");
  m.def("debug_new_game",
        [](const std::string& game, uint64_t seed) {
          return dbg::debug_new_game(dbg::debug_game_kind_from_name(game), seed);
        },
        py::arg("game"), py::arg("seed") = 1,
        "Create a native debug-game initial state.");
  m.def("debug_action_view",
        [](const dbg::DebugGameState& st) { return dbg::debug_legal_actions(st); },
        py::arg("state"),
        "Return rich debug-game legal actions.");
  m.def("debug_step", &dbg::debug_step_state, py::arg("state"),
        py::arg("action"), py::arg("seed") = 0,
        "Return the child state after applying one debug-game action.");
  m.def("debug_observe", &debug_observation_array, py::arg("state"),
        py::arg("perspective") = -1,
        "Dense observation tensor for a debug-game state.");
  m.def("debug_determinize", &dbg::debug_determinize, py::arg("state"),
        py::arg("perspective"), py::arg("seed") = 1,
        "Sample hidden variables from the state's information set.");
  m.def("debug_search",
        [](const std::string& game, const dbg::DebugGameState& state,
           const std::string& algo, int sims, uint64_t seed, int dets, int batch,
           const py::object& eval_callback) {
          (void)batch;
          return dbg::debug_search(game, state, algo, sims, seed, dets,
                                   make_debug_evaluator(eval_callback));
        },
        py::arg("game"), py::arg("state"), py::arg("algo"),
        py::arg("sims"), py::arg("seed"), py::arg("dets") = 0,
        py::arg("batch") = 16, py::arg("eval_callback") = py::none(),
        "Run native debug-game search. Algorithms: puct, pimc, maple; "
        "ismcts/mccfr/rebel are reserved names that raise in V1.");

  // --- RL-facing layer (S5) ------------------------------------------------
  m.attr("RL_MAX_ACTIONS") = RL_MAX_ACTIONS;
  m.attr("STATE_CARD_SLOTS") = STATE_CARD_SLOTS;
  m.attr("STATE_INPLAY_SLOTS") = STATE_INPLAY_SLOTS;
  m.attr("STATE_INPLAY_WIDTH") = STATE_INPLAY_WIDTH;
  m.attr("STATE_ZONE_COUNT") = STATE_ZONE_COUNT;
  m.attr("STATE_ZONE_SLOTS") = STATE_ZONE_SLOTS;
  m.attr("STATE_PRIZE_SLOTS") = STATE_PRIZE_SLOTS;
  m.attr("STATE_MAX_ATTACHED_ENERGY") = STATE_MAX_ATTACHED_ENERGY;
  m.attr("STATE_MAX_TOOLS") = STATE_MAX_TOOLS;
  m.attr("STATE_MAX_PRE_EVOS") = STATE_MAX_PRE_EVOS;
  m.attr("STATE_GLOBAL_WIDTH") = STATE_GLOBAL_WIDTH;
  m.attr("STATE_SELECT_META_WIDTH") = STATE_SELECT_META_WIDTH;
  m.attr("STATE_SELECT_OPTION_WIDTH") = STATE_SELECT_OPTION_WIDTH;
  m.attr("ACTION_META_WIDTH") = ACTION_META_WIDTH;
  m.attr("ACTION_OPTION_WIDTH") = ACTION_OPTION_WIDTH;
  m.attr("PPO_ACTION_FEAT_DIM") = PPO_ACTION_FEAT_DIM;
  m.attr("PPO_CARD_SLOTS") = PPO_CARD_SLOTS;
  m.attr("PPO_CARD_FEAT_DIM") = PPO_CARD_FEAT_DIM;
  m.attr("PPO_DECK_SLOTS") = PPO_DECK_SLOTS;
  m.attr("PPO_BELIEF_SLOTS") = PPO_BELIEF_SLOTS;
  m.attr("PPO_BELIEF_SUMMARY_DIM") = PPO_BELIEF_SUMMARY_DIM;
  m.attr("PPO_OPP_SELF") = PPO_OPP_SELF;
  m.attr("PPO_OPP_RANDOM") = PPO_OPP_RANDOM;
  m.attr("PPO_OPP_HEURISTIC") = PPO_OPP_HEURISTIC;
  m.attr("PPO_OPP_940") = PPO_OPP_940;
  m.attr("PPO_OPP_BEAM_940") = PPO_OPP_BEAM_940;
  m.attr("PPO_REWARD_TERMINAL") = PPO_REWARD_TERMINAL;
  m.attr("PPO_REWARD_PRIZE_DELTA") = PPO_REWARD_PRIZE_DELTA;
  m.attr("PPO_REWARD_TERMINAL_PLUS_DELTA") = PPO_REWARD_TERMINAL_PLUS_DELTA;
  m.def("rl_obs_dim", &rl_obs_dim, "Length of the encoded observation vector.");
  using Shape = std::vector<py::ssize_t>;

  py::class_<RlOptionSet>(m, "RlOptionSet",
                          "Reusable native legal-action handle for tree search.")
      .def_property_readonly("n", [](const RlOptionSet& o) { return o.n; })
      .def_property_readonly("pending",
                             [](const RlOptionSet& o) { return o.pending; });
  py::class_<MctsResult>(m, "MctsResult",
                         "Root summary from fully native heuristic MCTS.")
      .def_property_readonly("terminal",
                             [](const MctsResult& r) { return r.terminal; })
      .def_property_readonly("n_act", [](const MctsResult& r) { return r.n_act; })
      .def_property_readonly("childN",
                             [](const MctsResult& r) { return vector_to_array(r.childN); })
      .def_property_readonly("childW",
                             [](const MctsResult& r) { return vector_to_array(r.childW); });
  py::class_<FeatureRecord>(m, "FeatureRecord",
                            "Native sparse encoder/decoder record for training.")
      .def_readonly("enc_i", &FeatureRecord::enc_i)
      .def_readonly("enc_v", &FeatureRecord::enc_v)
      .def_readonly("enc_o", &FeatureRecord::enc_o)
      .def_readonly("dec_i", &FeatureRecord::dec_i)
      .def_readonly("dec_v", &FeatureRecord::dec_v)
      .def_readonly("dec_o", &FeatureRecord::dec_o)
      .def_readonly("n_act", &FeatureRecord::n_act)
      .def(py::pickle(
          [](const FeatureRecord& r) {
            return py::make_tuple(r.enc_i, r.enc_v, r.enc_o, r.dec_i, r.dec_v,
                                  r.dec_o, r.n_act);
          },
          [](py::tuple t) {
            if (py::len(t) != 7)
              throw std::runtime_error("invalid FeatureRecord pickle state");
            FeatureRecord r;
            r.enc_i = t[0].cast<std::vector<int32_t>>();
            r.enc_v = t[1].cast<std::vector<float>>();
            r.enc_o = t[2].cast<std::vector<int32_t>>();
            r.dec_i = t[3].cast<std::vector<int32_t>>();
            r.dec_v = t[4].cast<std::vector<float>>();
            r.dec_o = t[5].cast<std::vector<int32_t>>();
            r.n_act = t[6].cast<int>();
            return r;
          }));
  py::class_<PuctResult>(m, "PuctResult",
                         "Root summary from C++ PUCT with Python batched leaf eval.")
      .def_property_readonly("terminal",
                             [](const PuctResult& r) { return r.terminal; })
      .def_property_readonly("n_act", [](const PuctResult& r) { return r.n_act; })
      .def_property_readonly("childN",
                             [](const PuctResult& r) { return vector_to_array(r.childN); })
      .def_property_readonly("childW",
                             [](const PuctResult& r) { return vector_to_array(r.childW); })
      .def_property_readonly("state", [](const PuctResult& r) { return r.state; })
      .def_property_readonly("step_opts",
                             [](const PuctResult& r) { return r.step_opts; })
      .def_property_readonly("canon", [](const PuctResult& r) { return r.canon; })
      .def_property_readonly("opts", [](const PuctResult& r) { return r.opts; })
      .def_property_readonly("ctx", [](const PuctResult& r) { return r.ctx; });

  // --- single-state tree-search API (clone + rl_step for native MCTS) -------
  // These let Python build/branch a search tree on one GameState (cabt has no
  // such API). clone() is already exposed; combine with reseeding state.rng per
  // branch for stochastic chance nodes.
  m.def("rl_encode_obs",
        [](const GameState& st) {
          py::array_t<float> obs(Shape{rl_obs_dim()});
          rl_encode_obs(st, obs.mutable_data());
          return obs;
        },
        py::arg("state"),
        "Encode one state -> obs[rl_obs_dim()] from the acting player's POV.");
  m.def("rl_state_observation", &rl_state_observation_py, py::arg("state"),
        "Structured mover-POV state observation for embedding models. Returns "
        "fixed slot arrays: card_id/owner/area/index/known plus in-play scalars.");
  m.def("rl_state_ids", &rl_state_ids_complete_py, py::arg("state"),
        "Structured mover-POV symbolic state for embedding models. Separates "
        "empty, unknown, and known cards. Returns packed int32 tensors: "
        "in_play[2,6,64], zones[2,4,64], player_counts[2,5], "
        "player_status[2,5], global[32], select_meta[16], "
        "select_options[64,16], select_deck[64].");
  m.def("observation_ids", &rl_state_ids_complete_py, py::arg("state"),
        "Symbolic mover-POV observation ID tensors.");
  m.def("rl_heuristic_value", &rl_heuristic_value, py::arg("state"),
        "Fast mover-POV prize/active-HP heuristic used by native value MCTS.");
  m.def("rl_native_940_action", &rl_native_940_action, py::arg("state"),
        "Fast native 940-style policy action index.");
  m.def("rl_meta_beam_action", &rl_meta_beam_action, py::arg("state"),
        py::arg("inner_mode") = PPO_OPP_940, py::arg("beam_width") = 4,
        py::arg("depth") = 6, py::arg("seed") = 0,
        py::call_guard<py::gil_scoped_release>(),
        "Generic native beam-search meta-agent over an inner native policy mode.");
  m.def("rl_determinize_decklist", &rl_determinize_decklist, py::arg("state"),
        py::arg("deck0"), py::arg("deck1"), py::arg("perspective") = -1,
        py::arg("seed") = 1,
        "Sample hidden hands/decks/prizes from full decklists minus visible cards.");
  m.def("rl_value_mcts", &rl_value_mcts, py::arg("state"), py::arg("n_sims"),
        py::arg("seed"), py::arg("c_puct") = 1.5,
        py::call_guard<py::gil_scoped_release>(),
        "Fully native uniform-prior MCTS using rl_heuristic_value at leaves.");
  m.def("rl_puct_mcts_callback", &rl_puct_mcts_callback, py::arg("state"),
        py::arg("deck"), py::arg("n_sims"), py::arg("seed"),
        py::arg("batch"), py::arg("vloss"), py::arg("c_puct"),
        py::arg("dir_alpha"), py::arg("dir_eps"), py::arg("eval_batch"),
        "C++ PUCT tree with batched Python callback for leaf value/prior eval.");
  m.def("rl_feature_payload", &rl_feature_payload, py::arg("state"),
        py::arg("deck"), py::arg("width") = 0,
        "Build native cabt-compatible sparse feature arrays for one decision.");
  m.def("rl_feature_payload_cached_batch", &rl_feature_payload_cached_batch,
        py::arg("reqs"), py::arg("deck"), py::arg("width") = 0,
        "Build native sparse features for prepared (state, options, ctx, n_act) "
        "requests without Python descriptor/action feature construction.");
  m.def("rl_feature_record_cached", &feature_record_cached, py::arg("state"),
        py::arg("options"), py::arg("ctx"), py::arg("n_act"), py::arg("deck"),
        py::arg("width") = RL_MAX_ACTIONS,
        "Build one native sparse feature record for training without NumPy "
        "array allocation during self-play.");
  m.def("rl_feature_records_cached_batch", &feature_records_cached_batch,
        py::arg("reqs"), py::arg("deck"), py::arg("width") = RL_MAX_ACTIONS,
        "Build many native sparse feature records for training in one pybind "
        "call.");
  m.def("rl_puct_mcts_feature_callback", &rl_puct_mcts_feature_callback,
        py::arg("state"), py::arg("deck"), py::arg("n_sims"), py::arg("seed"),
        py::arg("batch"), py::arg("vloss"), py::arg("c_puct"),
        py::arg("dir_alpha"), py::arg("dir_eps"), py::arg("eval_payload"),
        "C++ PUCT tree with native sparse feature batching and Python NN eval.");
  m.def("rl_maple_puct_callback", &rl_maple_puct_callback, py::arg("state"),
        py::arg("deck"), py::arg("n_sims"), py::arg("seed"),
        py::arg("dets"), py::arg("c_puct"), py::arg("eval_payload"),
        "Native MAPLE-style information-set PUCT over decklist determinizations "
        "with batched Python NN eval over compatible hidden worlds.");
  m.def("rl_selfplay_maple_callback", &rl_selfplay_maple_callback,
        py::arg("deck"), py::arg("total_games"), py::arg("concurrent"),
        py::arg("n_sims"), py::arg("seed"), py::arg("temp_moves"),
        py::arg("vboot"), py::arg("dets"), py::arg("c_puct"),
        py::arg("eval_payload"), py::arg("max_steps") = 2000,
        py::arg("row_cap") = 0,
        "C++-owned MAPLE AlphaZero self-play loop using native MAPLE roots "
        "and batched Python NN eval callbacks inside each information-set node.");
  m.def("rl_selfplay_puct_callback", &rl_selfplay_puct_callback,
        py::arg("deck"), py::arg("total_games"), py::arg("concurrent"),
        py::arg("n_sims"), py::arg("seed"), py::arg("temp_moves"),
        py::arg("vboot"), py::arg("reuse_tree"), py::arg("c_puct"),
        py::arg("dir_alpha"), py::arg("dir_eps"), py::arg("eval_payload"),
        py::arg("max_steps") = 2000, py::arg("budget_reuse") = true,
        py::arg("leaf_batch") = 16, py::arg("vloss") = 1.0,
        py::arg("profile") = false, py::arg("leaf_batch_rows") = 0,
        "Vectorized C++ AlphaZero self-play with batched Python NN eval; "
        "returns (records, result_counts), or adds stats when profile=True.");
  m.def("rl_options", &rl_options, py::arg("state"),
        "Reusable option set for the current decision; use with rl_step_cached.");
  m.def(
      "rl_action_view",
      [](const GameState& st) -> py::tuple {
        if (st.has_pending()) {
          return py::make_tuple(st.pending.context,
                                descriptors_to_py(st.pending.options),
                                rl_options(st));
        }
        auto desc = legal_main(st);
        return py::make_tuple(-1, descriptors_to_py(desc),
                              rl_options_from_descriptors(desc));
      },
      py::arg("state"),
      "Return (context, descriptors, option_set) from one legal-action pass.");
  m.def("rl_legal_mask",
        [](const GameState& st) {
          py::array_t<uint8_t> mask(Shape{RL_MAX_ACTIONS});
          int n = rl_legal_mask(st, mask.mutable_data());
          return py::make_tuple(n, mask);
        },
        py::arg("state"),
        "(n_options, legal_mask[RL_MAX_ACTIONS]) for the current decision; option "
        "indices align with rl_step's `action`.");
  m.def("rl_action_features",
        [](const GameState& st) {
          py::array_t<float> feat(Shape{RL_MAX_ACTIONS, PPO_ACTION_FEAT_DIM});
          rl_action_features(st, feat.mutable_data());
          return feat;
        },
        py::arg("state"),
        "Dense descriptor features [RL_MAX_ACTIONS,PPO_ACTION_FEAT_DIM] for PPO.");
  m.def("rl_action_ids", &rl_action_ids_py, py::arg("state"),
        "Packed symbolic legal-action tensors for embedding models and cg-select "
        "reconstruction. Returns meta[16], options[64,24], deck[64], mask[64].");
  m.def("action_ids", &rl_action_ids_py, py::arg("state"),
        "Symbolic legal-action ID tensors.");
  m.def("rl_card_features",
        [](const GameState& st) {
          py::array_t<float> feat(Shape{PPO_CARD_SLOTS, PPO_CARD_FEAT_DIM});
          rl_card_features(st, feat.mutable_data());
          return feat;
        },
        py::arg("state"),
        "Dense card-slot features [PPO_CARD_SLOTS,PPO_CARD_FEAT_DIM] for PPO.");
  m.def("rl_deck_features",
        [](std::vector<int> deck) {
          py::array_t<float> feat(Shape{PPO_DECK_SLOTS, PPO_CARD_FEAT_DIM});
          rl_deck_features(deck, feat.mutable_data());
          return feat;
        },
        py::arg("deck"),
        "Dense full-deck multiset features [PPO_DECK_SLOTS,PPO_CARD_FEAT_DIM].");
  m.def("rl_belief_features",
        [](const GameState& st) {
          py::array_t<float> feat(Shape{PPO_BELIEF_SLOTS, PPO_CARD_FEAT_DIM});
          rl_belief_features(st, feat.mutable_data());
          return feat;
        },
        py::arg("state"),
        "Dense known hidden-zone belief features "
        "[PPO_BELIEF_SLOTS,PPO_CARD_FEAT_DIM].");
  m.def("rl_belief_summary",
        [](const GameState& st) {
          py::array_t<float> feat(Shape{PPO_BELIEF_SUMMARY_DIM});
          rl_belief_summary(st, feat.mutable_data());
          return feat;
        },
        py::arg("state"),
        "Compact known hidden-zone belief/count summary for PPO.");
  m.def("rl_is_agent_decision", &rl_is_agent_decision, py::arg("state"),
        "True at a genuine single-select, multi-option agent decision.");
  m.def(
      "rl_step",
      [](GameState& st, int action, uint64_t seed) {
        uint64_t rng = seed ? seed : (st.rng ? st.rng : 0x9e3779b97f4a7c15ULL);
        rl_step(st, action, rng);
        return rng;
      },
      py::arg("state"), py::arg("action"), py::arg("seed") = 0,
      "Apply option `action`, then auto-resolve forced/multi-select sub-decisions "
      "until the next agent decision or game end (mutates state). Returns the "
      "advanced rng seed (thread it back in for the next call).");
  m.def(
      "rl_step_cached",
      [](GameState& st, const RlOptionSet& opts, int action, uint64_t seed) {
        uint64_t rng = seed ? seed : (st.rng ? st.rng : 0x9e3779b97f4a7c15ULL);
        rl_step_cached(st, opts, action, rng);
        return rng;
      },
      py::arg("state"), py::arg("options"), py::arg("action"),
      py::arg("seed") = 0,
      "Apply a cached option-set action, then auto-resolve to the next agent "
      "decision or game end.");

  py::class_<BatchEnv>(m, "BatchEnv",
                       "Vectorized self-play env: N games, one masked single-select "
                       "action space (MAIN + sub-decisions unified), numpy I/O.")
      .def(py::init<std::vector<int>, std::vector<int>, int, uint64_t, int>(),
           py::arg("deck0"), py::arg("deck1"), py::arg("n"), py::arg("seed") = 0,
           py::arg("threads") = 0)
      .def("size", &BatchEnv::size)
      .def("reset_all", &BatchEnv::reset_all,
           py::call_guard<py::gil_scoped_release>())
      .def("observe",
           [](BatchEnv& e) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D});
             py::array_t<uint8_t> mask(Shape{n, RL_MAX_ACTIONS});
             {
               py::gil_scoped_release rel;
               e.observe(obs.mutable_data(), mask.mutable_data());
             }
             return py::make_tuple(obs, mask);
           },
           "Return (obs[N,D], legal_mask[N,RL_MAX_ACTIONS]) for the current states.")
      .def("step",
           [](BatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D}), reward(Shape{n});
             py::array_t<uint8_t> done(Shape{n}), mask(Shape{n, RL_MAX_ACTIONS});
             {
               py::gil_scoped_release rel;
               e.step(actions.data(), obs.mutable_data(), reward.mutable_data(),
                      done.mutable_data(), mask.mutable_data());
             }
             return py::make_tuple(obs, reward, done, mask);
           },
           py::arg("actions"),
           "Step all games with actions[N]; auto-resets finished games. Returns "
           "(obs[N,D], reward[N], done[N], legal_mask[N,RL_MAX_ACTIONS]).");
  py::class_<VectorEnv>(m, "VectorEnv",
                        "Simplified vectorized env: N games, explicit current "
                        "player per row, one masked action per game.")
      .def(py::init<std::vector<int>, std::vector<int>, int, uint64_t, int>(),
           py::arg("deck0"), py::arg("deck1"), py::arg("n"), py::arg("seed") = 0,
           py::arg("threads") = 0)
      .def("size", &VectorEnv::size)
      .def("reset_all", &VectorEnv::reset_all,
           py::call_guard<py::gil_scoped_release>())
      .def("observe",
           [](VectorEnv& e) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D});
             py::array_t<uint8_t> mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n}), result(Shape{n});
             {
               py::gil_scoped_release rel;
               e.observe(obs.mutable_data(), mask.mutable_data(),
                         player.mutable_data(), result.mutable_data());
             }
             return py::make_tuple(obs, mask, player, result);
           },
           "Return (obs[N,D], legal_mask[N,RL_MAX_ACTIONS], player[N], result[N]).")
      .def("state_ids",
           [](VectorEnv& e) {
             int n = e.size();
             py::array_t<int32_t> in_play(
                 Shape{n, 2, STATE_INPLAY_SLOTS, STATE_INPLAY_WIDTH});
             py::array_t<int32_t> zones(
                 Shape{n, 2, STATE_ZONE_COUNT, STATE_ZONE_SLOTS});
             py::array_t<int32_t> counts(Shape{n, 2, 5});
             py::array_t<int32_t> status(Shape{n, 2, 5});
             py::array_t<int32_t> global(Shape{n, STATE_GLOBAL_WIDTH});
             py::array_t<int32_t> select_meta(
                 Shape{n, STATE_SELECT_META_WIDTH});
             py::array_t<int32_t> select_options(
                 Shape{n, RL_MAX_ACTIONS, STATE_SELECT_OPTION_WIDTH});
             py::array_t<int32_t> select_deck(
                 Shape{n, STATE_ZONE_SLOTS});

             for (int i = 0; i < n; ++i) {
               const GameState& st = e.state_at(i);
               ptcg::fill_observation_ids(
                   st,
                   in_play.mutable_data() +
                       i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
                   zones.mutable_data() +
                       i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
                   counts.mutable_data() + i * 2 * 5,
                   status.mutable_data() + i * 2 * 5,
                   global.mutable_data() + i * STATE_GLOBAL_WIDTH);
               ptcg::fill_action_ids(
                   st, action_id_view_from_options(e.options_at(i)),
                   select_meta.mutable_data() + i * STATE_SELECT_META_WIDTH,
                   select_options.mutable_data() +
                       i * RL_MAX_ACTIONS * STATE_SELECT_OPTION_WIDTH,
                   select_deck.mutable_data() + i * STATE_ZONE_SLOTS, nullptr);
             }

             py::dict out;
             out["in_play"] = in_play;
             out["zones"] = zones;
             out["player_counts"] = counts;
             out["player_status"] = status;
             out["global"] = global;
             out["select_meta"] = select_meta;
             out["select_options"] = select_options;
             out["select_deck"] = select_deck;
             out["empty_card_id"] = 0;
             out["unknown_card_id"] = -1;
             out["zone_hand"] = 0;
             out["zone_deck"] = 1;
             out["zone_discard"] = 2;
             out["zone_prizes"] = 3;
             return out;
           },
           "Return packed state tensors with a leading batch dimension: "
           "in_play[N,2,6,64], zones[N,2,4,64], player_counts[N,2,5], "
           "player_status[N,2,5], global[N,32], plus packed select_* fields.")
      .def("state_ids_into",
           [](VectorEnv& e, py::dict out) {
             using I32Array = py::array_t<int32_t, py::array::c_style>;
             int n = e.size();
             I32Array in_play = out["in_play"].cast<I32Array>();
             I32Array zones = out["zones"].cast<I32Array>();
             I32Array counts = out["player_counts"].cast<I32Array>();
             I32Array status = out["player_status"].cast<I32Array>();
             I32Array global = out["global"].cast<I32Array>();
             I32Array select_meta = out["select_meta"].cast<I32Array>();
             I32Array select_options = out["select_options"].cast<I32Array>();
             I32Array select_deck = out["select_deck"].cast<I32Array>();
             auto require_size = [](const py::array& a, py::ssize_t expected,
                                    const char* name) {
               if (a.size() != expected) {
                 throw std::runtime_error(std::string("bad ") + name +
                                          " buffer size");
               }
             };
             require_size(in_play, n * 2 * STATE_INPLAY_SLOTS *
                                       STATE_INPLAY_WIDTH, "in_play");
             require_size(zones, n * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
                          "zones");
             require_size(counts, n * 2 * 5, "player_counts");
             require_size(status, n * 2 * 5, "player_status");
             require_size(global, n * STATE_GLOBAL_WIDTH, "global");
             require_size(select_meta, n * STATE_SELECT_META_WIDTH,
                          "select_meta");
             require_size(select_options,
                          n * RL_MAX_ACTIONS * STATE_SELECT_OPTION_WIDTH,
                          "select_options");
             require_size(select_deck, n * STATE_ZONE_SLOTS, "select_deck");

             py::gil_scoped_release rel;
             for (int i = 0; i < n; ++i) {
               const GameState& st = e.state_at(i);
               ptcg::fill_observation_ids(
                   st,
                   in_play.mutable_data() +
                       i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
                   zones.mutable_data() +
                       i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
                   counts.mutable_data() + i * 2 * 5,
                   status.mutable_data() + i * 2 * 5,
                   global.mutable_data() + i * STATE_GLOBAL_WIDTH);
               ptcg::fill_action_ids(
                   st, action_id_view_from_options(e.options_at(i)),
                   select_meta.mutable_data() + i * STATE_SELECT_META_WIDTH,
                   select_options.mutable_data() +
                       i * RL_MAX_ACTIONS * STATE_SELECT_OPTION_WIDTH,
                   select_deck.mutable_data() + i * STATE_ZONE_SLOTS, nullptr);
             }
           },
           py::arg("out"),
           "Write packed state ids into an existing state_ids-style dict.")
      .def("action_ids",
           [](VectorEnv& e) {
             int n = e.size();
             py::array_t<int32_t> meta(Shape{n, ACTION_META_WIDTH});
             py::array_t<int32_t> options(
                 Shape{n, RL_MAX_ACTIONS, ACTION_OPTION_WIDTH});
             py::array_t<int32_t> deck(Shape{n, STATE_ZONE_SLOTS});
             py::array_t<uint8_t> mask(Shape{n, RL_MAX_ACTIONS});
             auto* meta_p = meta.mutable_data();
             auto* opt_p = options.mutable_data();
             auto* deck_p = deck.mutable_data();
             auto* mask_p = mask.mutable_data();
             for (int i = 0; i < n; ++i) {
               const GameState& st = e.state_at(i);
               ptcg::fill_action_ids(
                   st, action_id_view_from_options(e.options_at(i)),
                   meta_p + i * ACTION_META_WIDTH,
                   opt_p + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                   deck_p + i * STATE_ZONE_SLOTS, mask_p + i * RL_MAX_ACTIONS);
             }
             return action_ids_dict(meta, options, deck, mask);
           },
           "Return packed legal-action tensors: meta[N,16], "
           "options[N,64,24], deck[N,64], mask[N,64].")
      .def("action_ids_into",
           [](VectorEnv& e, py::dict out) {
             using I32Array = py::array_t<int32_t, py::array::c_style>;
             using U8Array = py::array_t<uint8_t, py::array::c_style>;
             int n = e.size();
             I32Array meta = out["meta"].cast<I32Array>();
             I32Array options = out["options"].cast<I32Array>();
             I32Array deck = out["deck"].cast<I32Array>();
             U8Array mask = out["mask"].cast<U8Array>();
             auto require_size = [](const py::array& a, py::ssize_t expected,
                                    const char* name) {
               if (a.size() != expected) {
                 throw std::runtime_error(std::string("bad ") + name +
                                          " buffer size");
               }
             };
             require_size(meta, n * ACTION_META_WIDTH, "meta");
             require_size(options, n * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                          "options");
             require_size(deck, n * STATE_ZONE_SLOTS, "deck");
             require_size(mask, n * RL_MAX_ACTIONS, "mask");

             py::gil_scoped_release rel;
             for (int i = 0; i < n; ++i) {
               const GameState& st = e.state_at(i);
               ptcg::fill_action_ids(
                   st, action_id_view_from_options(e.options_at(i)),
                   meta.mutable_data() + i * ACTION_META_WIDTH,
                   options.mutable_data() +
                       i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                   deck.mutable_data() + i * STATE_ZONE_SLOTS,
                   mask.mutable_data() + i * RL_MAX_ACTIONS);
             }
           },
           py::arg("out"),
           "Write packed action ids into an existing action_ids-style dict.")
      .def("observe_ids_into",
           [](VectorEnv& e, py::dict state_out, py::dict action_out,
              py::array_t<int32_t, py::array::c_style> player,
              py::array_t<int32_t, py::array::c_style> result) {
             using I32Array = py::array_t<int32_t, py::array::c_style>;
             using U8Array = py::array_t<uint8_t, py::array::c_style>;
             int n = e.size();
             I32Array in_play = state_out["in_play"].cast<I32Array>();
             I32Array zones = state_out["zones"].cast<I32Array>();
             I32Array counts = state_out["player_counts"].cast<I32Array>();
             I32Array status = state_out["player_status"].cast<I32Array>();
             I32Array global = state_out["global"].cast<I32Array>();
             I32Array select_meta = state_out["select_meta"].cast<I32Array>();
             I32Array select_options =
                 state_out["select_options"].cast<I32Array>();
             I32Array select_deck = state_out["select_deck"].cast<I32Array>();
             I32Array meta = action_out["meta"].cast<I32Array>();
             I32Array options = action_out["options"].cast<I32Array>();
             I32Array deck = action_out["deck"].cast<I32Array>();
             U8Array mask = action_out["mask"].cast<U8Array>();
             auto require_size = [](const py::array& a, py::ssize_t expected,
                                    const char* name) {
               if (a.size() != expected) {
                 throw std::runtime_error(std::string("bad ") + name +
                                          " buffer size");
               }
             };
             require_size(in_play, n * 2 * STATE_INPLAY_SLOTS *
                                       STATE_INPLAY_WIDTH, "in_play");
             require_size(zones, n * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
                          "zones");
             require_size(counts, n * 2 * 5, "player_counts");
             require_size(status, n * 2 * 5, "player_status");
             require_size(global, n * STATE_GLOBAL_WIDTH, "global");
             require_size(select_meta, n * STATE_SELECT_META_WIDTH,
                          "select_meta");
             require_size(select_options,
                          n * RL_MAX_ACTIONS * STATE_SELECT_OPTION_WIDTH,
                          "select_options");
             require_size(select_deck, n * STATE_ZONE_SLOTS, "select_deck");
             require_size(meta, n * ACTION_META_WIDTH, "meta");
             require_size(options, n * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                          "options");
             require_size(deck, n * STATE_ZONE_SLOTS, "deck");
             require_size(mask, n * RL_MAX_ACTIONS, "mask");
             require_size(player, n, "player");
             require_size(result, n, "result");

             py::gil_scoped_release rel;
             e.observe_ids(in_play.mutable_data(), zones.mutable_data(),
                           counts.mutable_data(), status.mutable_data(),
                           global.mutable_data(), meta.mutable_data(),
                           options.mutable_data(), deck.mutable_data(),
                           mask.mutable_data(), player.mutable_data(),
                           result.mutable_data());
             std::memcpy(select_meta.mutable_data(), meta.mutable_data(),
                         static_cast<size_t>(n) * STATE_SELECT_META_WIDTH *
                             sizeof(int32_t));
             std::memcpy(select_options.mutable_data(), options.mutable_data(),
                         static_cast<size_t>(n) * RL_MAX_ACTIONS *
                             STATE_SELECT_OPTION_WIDTH * sizeof(int32_t));
             std::memcpy(select_deck.mutable_data(), deck.mutable_data(),
                         static_cast<size_t>(n) * STATE_ZONE_SLOTS *
                             sizeof(int32_t));
           },
           py::arg("state_out"), py::arg("action_out"), py::arg("player"),
           py::arg("result"),
           "Write packed state/action ids plus player/result in one call.")
      .def("step",
           [](VectorEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D}), reward(Shape{n});
             py::array_t<uint8_t> done(Shape{n}), mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n}), result(Shape{n});
             {
               py::gil_scoped_release rel;
               e.step(actions.data(), obs.mutable_data(), reward.mutable_data(),
                      done.mutable_data(), mask.mutable_data(),
                      player.mutable_data(), result.mutable_data());
             }
             return py::make_tuple(obs, reward, done, mask, player, result);
           },
           py::arg("actions"),
           "Step all games with actions[N]; auto-resets finished games. Returns "
           "(obs[N,D], reward[N], done[N], legal_mask[N,RL_MAX_ACTIONS], "
           "player[N], result[N]).");
  py::class_<PpoBatchEnv>(m, "PpoBatchEnv",
                          "Fully native vector env for PPO rollouts. One policy "
                          "controls the current player in each self-play game; "
                          "finished games auto-reset after emitting terminal reward.")
      .def(py::init<std::vector<int>, std::vector<int>, int, uint64_t, int,
                    double, int, int, int, int>(),
           py::arg("deck0"), py::arg("deck1"), py::arg("n"), py::arg("seed") = 0,
           py::arg("max_steps") = 2000, py::arg("prize_weight") = 0.0,
           py::arg("learner_seat") = -1, py::arg("opponent_mode") = PPO_OPP_SELF,
           py::arg("reward_mode") = PPO_REWARD_TERMINAL,
           py::arg("threads") = 0)
      .def("size", &PpoBatchEnv::size)
      .def("reset_all", &PpoBatchEnv::reset_all,
           py::call_guard<py::gil_scoped_release>())
      .def("observe_into",
           [](PpoBatchEnv& e,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player) {
             py::gil_scoped_release rel;
             e.observe(obs.mutable_data(), mask.mutable_data(),
                       player.mutable_data());
           },
           py::arg("obs"), py::arg("legal_mask"), py::arg("player"),
           "Write current obs/legal_mask/player into caller-owned arrays.")
      .def("observe_all_features_into",
           [](PpoBatchEnv& e,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player,
              py::array_t<float, py::array::c_style> action_features,
              py::array_t<float, py::array::c_style> card_features,
              py::array_t<float, py::array::c_style> deck_features,
              py::array_t<float, py::array::c_style> belief_features,
              py::array_t<float, py::array::c_style> belief_summary) {
             py::gil_scoped_release rel;
             e.observe(obs.mutable_data(), mask.mutable_data(),
                       player.mutable_data());
             e.action_features(action_features.mutable_data());
             e.card_features(card_features.mutable_data());
             e.deck_features(deck_features.mutable_data());
             e.belief_features(belief_features.mutable_data());
             e.belief_summary(belief_summary.mutable_data());
           },
           py::arg("obs"), py::arg("legal_mask"), py::arg("player"),
           py::arg("action_features"), py::arg("card_features"),
           py::arg("deck_features"), py::arg("belief_features"),
           py::arg("belief_summary"),
           "Write current obs/action/card/deck/belief features into "
           "caller-owned arrays.")
      .def("observe_card_features_into",
           [](PpoBatchEnv& e,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player,
              py::array_t<float, py::array::c_style> action_features,
              py::array_t<float, py::array::c_style> card_features,
              py::array_t<float, py::array::c_style> deck_features) {
             py::gil_scoped_release rel;
             e.observe(obs.mutable_data(), mask.mutable_data(),
                       player.mutable_data());
             e.action_features(action_features.mutable_data());
             e.card_features(card_features.mutable_data());
             e.deck_features(deck_features.mutable_data());
           },
           py::arg("obs"), py::arg("legal_mask"), py::arg("player"),
           py::arg("action_features"), py::arg("card_features"),
           py::arg("deck_features"),
           "Write current obs/action/card/deck features into caller-owned arrays.")
      .def("observe_action_features_into",
           [](PpoBatchEnv& e,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player,
              py::array_t<float, py::array::c_style> action_features) {
             py::gil_scoped_release rel;
             e.observe(obs.mutable_data(), mask.mutable_data(),
                       player.mutable_data());
             e.action_features(action_features.mutable_data());
           },
           py::arg("obs"), py::arg("legal_mask"), py::arg("player"),
           py::arg("action_features"),
           "Write current obs/legal_mask/player/action_features into "
           "caller-owned arrays.")
      .def("observe",
           [](PpoBatchEnv& e) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D});
             py::array_t<uint8_t> mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n});
             {
               py::gil_scoped_release rel;
               e.observe(obs.mutable_data(), mask.mutable_data(),
                         player.mutable_data());
             }
             return py::make_tuple(obs, mask, player);
           },
           "Return (obs[N,D], legal_mask[N,RL_MAX_ACTIONS], player[N]).")
      .def("observe_with_action_features",
           [](PpoBatchEnv& e) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D});
             py::array_t<uint8_t> mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n});
             py::array_t<float> action_features(
                 Shape{n, RL_MAX_ACTIONS, PPO_ACTION_FEAT_DIM});
             {
               py::gil_scoped_release rel;
               e.observe(obs.mutable_data(), mask.mutable_data(),
                         player.mutable_data());
               e.action_features(action_features.mutable_data());
             }
             return py::make_tuple(obs, mask, player, action_features);
           },
           "Return (obs, legal_mask, player, action_features) in one pybind call.")
      .def("action_features",
           [](PpoBatchEnv& e) {
             int n = e.size();
             py::array_t<float> feat(
                 Shape{n, RL_MAX_ACTIONS, PPO_ACTION_FEAT_DIM});
             {
               py::gil_scoped_release rel;
               e.action_features(feat.mutable_data());
             }
             return feat;
           },
           "Return dense descriptor features "
           "[N,RL_MAX_ACTIONS,PPO_ACTION_FEAT_DIM].")
      .def("card_features",
           [](PpoBatchEnv& e) {
             int n = e.size();
             py::array_t<float> feat(
                 Shape{n, PPO_CARD_SLOTS, PPO_CARD_FEAT_DIM});
             {
               py::gil_scoped_release rel;
               e.card_features(feat.mutable_data());
             }
             return feat;
           },
           "Return dense card-slot features [N,PPO_CARD_SLOTS,PPO_CARD_FEAT_DIM].")
      .def("deck_features",
           [](PpoBatchEnv& e) {
             int n = e.size();
             py::array_t<float> feat(
                 Shape{n, PPO_DECK_SLOTS, PPO_CARD_FEAT_DIM});
             {
               py::gil_scoped_release rel;
               e.deck_features(feat.mutable_data());
             }
             return feat;
           },
           "Return dense full-deck features [N,PPO_DECK_SLOTS,PPO_CARD_FEAT_DIM].")
      .def("belief_features",
           [](PpoBatchEnv& e) {
             int n = e.size();
             py::array_t<float> feat(
                 Shape{n, PPO_BELIEF_SLOTS, PPO_CARD_FEAT_DIM});
             {
               py::gil_scoped_release rel;
               e.belief_features(feat.mutable_data());
             }
             return feat;
           },
           "Return known hidden-zone belief features "
           "[N,PPO_BELIEF_SLOTS,PPO_CARD_FEAT_DIM].")
      .def("belief_summary",
           [](PpoBatchEnv& e) {
             int n = e.size();
             py::array_t<float> feat(Shape{n, PPO_BELIEF_SUMMARY_DIM});
             {
               py::gil_scoped_release rel;
               e.belief_summary(feat.mutable_data());
             }
             return feat;
           },
           "Return compact known hidden-zone belief/count summary "
           "[N,PPO_BELIEF_SUMMARY_DIM].")
      .def("observe_with_all_features",
           [](PpoBatchEnv& e) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D});
             py::array_t<uint8_t> mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n});
             py::array_t<float> action_features(
                 Shape{n, RL_MAX_ACTIONS, PPO_ACTION_FEAT_DIM});
             py::array_t<float> card_features(
                 Shape{n, PPO_CARD_SLOTS, PPO_CARD_FEAT_DIM});
             py::array_t<float> deck_features(
                 Shape{n, PPO_DECK_SLOTS, PPO_CARD_FEAT_DIM});
             py::array_t<float> belief_features(
                 Shape{n, PPO_BELIEF_SLOTS, PPO_CARD_FEAT_DIM});
             py::array_t<float> belief_summary(Shape{n, PPO_BELIEF_SUMMARY_DIM});
             {
               py::gil_scoped_release rel;
               e.observe(obs.mutable_data(), mask.mutable_data(),
                         player.mutable_data());
               e.action_features(action_features.mutable_data());
               e.card_features(card_features.mutable_data());
               e.deck_features(deck_features.mutable_data());
               e.belief_features(belief_features.mutable_data());
               e.belief_summary(belief_summary.mutable_data());
             }
             return py::make_tuple(obs, mask, player, action_features,
                                   card_features, deck_features, belief_features,
                                   belief_summary);
           },
           "Return (obs, legal_mask, player, action_features, card_features, "
           "deck_features, belief_features, belief_summary).")
      .def("step",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D}), reward(Shape{n});
             py::array_t<uint8_t> done(Shape{n}), mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n}), result(Shape{n}),
                 episode_len(Shape{n});
             {
               py::gil_scoped_release rel;
               e.step(actions.data(), obs.mutable_data(), reward.mutable_data(),
                      done.mutable_data(), mask.mutable_data(),
                      player.mutable_data(), result.mutable_data(),
                      episode_len.mutable_data());
             }
             return py::make_tuple(obs, reward, done, mask, player, result,
                                   episode_len);
           },
           py::arg("actions"),
           "Step all games with actions[N]. Returns "
           "(obs, reward, done, legal_mask, player, result, episode_len).")
      .def("step_rewards_into",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> reward,
              py::array_t<uint8_t, py::array::c_style> done,
              py::array_t<int32_t, py::array::c_style> result,
              py::array_t<int32_t, py::array::c_style> episode_len) {
             py::gil_scoped_release rel;
             e.step_rewards(actions.data(), reward.mutable_data(),
                            done.mutable_data(), result.mutable_data(),
                            episode_len.mutable_data());
           },
           py::arg("actions"), py::arg("reward"), py::arg("done"),
           py::arg("result"), py::arg("episode_len"),
           "Advance all games and write only reward/done/result/episode_len.")
      .def("step_observe_into",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> reward,
              py::array_t<uint8_t, py::array::c_style> done,
              py::array_t<int32_t, py::array::c_style> result,
              py::array_t<int32_t, py::array::c_style> episode_len,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player) {
             py::gil_scoped_release rel;
             e.step_rewards(actions.data(), reward.mutable_data(),
                            done.mutable_data(), result.mutable_data(),
                            episode_len.mutable_data());
             e.observe(obs.mutable_data(), mask.mutable_data(),
                       player.mutable_data());
           },
           py::arg("actions"), py::arg("reward"), py::arg("done"),
           py::arg("result"), py::arg("episode_len"), py::arg("obs"),
           py::arg("legal_mask"), py::arg("player"),
           "Advance all games and write next obs/legal_mask/player.")
      .def("step_action_features_into",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> reward,
              py::array_t<uint8_t, py::array::c_style> done,
              py::array_t<int32_t, py::array::c_style> result,
              py::array_t<int32_t, py::array::c_style> episode_len,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player,
              py::array_t<float, py::array::c_style> action_features) {
             py::gil_scoped_release rel;
             e.step_rewards(actions.data(), reward.mutable_data(),
                            done.mutable_data(), result.mutable_data(),
                            episode_len.mutable_data());
             e.observe(obs.mutable_data(), mask.mutable_data(),
                       player.mutable_data());
             e.action_features(action_features.mutable_data());
           },
           py::arg("actions"), py::arg("reward"), py::arg("done"),
           py::arg("result"), py::arg("episode_len"), py::arg("obs"),
           py::arg("legal_mask"), py::arg("player"), py::arg("action_features"),
           "Advance all games and write next obs/action features.")
      .def("step_card_features_into",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> reward,
              py::array_t<uint8_t, py::array::c_style> done,
              py::array_t<int32_t, py::array::c_style> result,
              py::array_t<int32_t, py::array::c_style> episode_len,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player,
              py::array_t<float, py::array::c_style> action_features,
              py::array_t<float, py::array::c_style> card_features,
              py::array_t<float, py::array::c_style> deck_features) {
             py::gil_scoped_release rel;
             e.step_card_features(actions.data(), obs.mutable_data(),
                                  reward.mutable_data(), done.mutable_data(),
                                  mask.mutable_data(), player.mutable_data(),
                                  result.mutable_data(),
                                  episode_len.mutable_data(),
                                  action_features.mutable_data(),
                                  card_features.mutable_data());
             e.deck_features(deck_features.mutable_data());
           },
           py::arg("actions"), py::arg("reward"), py::arg("done"),
           py::arg("result"), py::arg("episode_len"), py::arg("obs"),
           py::arg("legal_mask"), py::arg("player"), py::arg("action_features"),
           py::arg("card_features"), py::arg("deck_features"),
           "Advance all games and write next obs/action/card/deck features.")
      .def("step_all_features_into",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions,
              py::array_t<float, py::array::c_style> reward,
              py::array_t<uint8_t, py::array::c_style> done,
              py::array_t<int32_t, py::array::c_style> result,
              py::array_t<int32_t, py::array::c_style> episode_len,
              py::array_t<float, py::array::c_style> obs,
              py::array_t<uint8_t, py::array::c_style> mask,
              py::array_t<int32_t, py::array::c_style> player,
              py::array_t<float, py::array::c_style> action_features,
              py::array_t<float, py::array::c_style> card_features,
              py::array_t<float, py::array::c_style> deck_features,
              py::array_t<float, py::array::c_style> belief_features,
              py::array_t<float, py::array::c_style> belief_summary) {
             py::gil_scoped_release rel;
             e.step_card_features(actions.data(), obs.mutable_data(),
                                  reward.mutable_data(), done.mutable_data(),
                                  mask.mutable_data(), player.mutable_data(),
                                  result.mutable_data(),
                                  episode_len.mutable_data(),
                                  action_features.mutable_data(),
                                  card_features.mutable_data());
             e.deck_features(deck_features.mutable_data());
             e.belief_features(belief_features.mutable_data());
             e.belief_summary(belief_summary.mutable_data());
           },
           py::arg("actions"), py::arg("reward"), py::arg("done"),
           py::arg("result"), py::arg("episode_len"), py::arg("obs"),
           py::arg("legal_mask"), py::arg("player"), py::arg("action_features"),
           py::arg("card_features"), py::arg("deck_features"),
           py::arg("belief_features"), py::arg("belief_summary"),
           "Advance all games and write next obs/action/card/deck/belief features.")
      .def("step_with_action_features",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D}), reward(Shape{n});
             py::array_t<uint8_t> done(Shape{n}), mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n}), result(Shape{n}),
                 episode_len(Shape{n});
             py::array_t<float> action_features(
                 Shape{n, RL_MAX_ACTIONS, PPO_ACTION_FEAT_DIM});
             {
               py::gil_scoped_release rel;
               e.step(actions.data(), obs.mutable_data(), reward.mutable_data(),
                      done.mutable_data(), mask.mutable_data(),
                      player.mutable_data(), result.mutable_data(),
                      episode_len.mutable_data());
               e.action_features(action_features.mutable_data());
             }
             return py::make_tuple(obs, reward, done, mask, player, result,
                                   episode_len, action_features);
           },
           py::arg("actions"),
           "Step all games and return action_features in the same pybind call.")
      .def("step_with_card_features",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D}), reward(Shape{n});
             py::array_t<uint8_t> done(Shape{n}), mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n}), result(Shape{n}),
                 episode_len(Shape{n});
             py::array_t<float> action_features(
                 Shape{n, RL_MAX_ACTIONS, PPO_ACTION_FEAT_DIM});
             py::array_t<float> card_features(
                 Shape{n, PPO_CARD_SLOTS, PPO_CARD_FEAT_DIM});
             {
               py::gil_scoped_release rel;
               e.step_card_features(
                   actions.data(), obs.mutable_data(), reward.mutable_data(),
                   done.mutable_data(), mask.mutable_data(), player.mutable_data(),
                   result.mutable_data(), episode_len.mutable_data(),
                   action_features.mutable_data(), card_features.mutable_data());
             }
             return py::make_tuple(obs, reward, done, mask, player, result,
                                   episode_len, action_features, card_features);
           },
           py::arg("actions"),
           "Step all games and return action/card features, excluding static deck "
           "and belief tensors.")
      .def("profile_step_card_features",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions,
              int repeats) {
             PpoStepProfile p;
             {
               py::gil_scoped_release rel;
               p = e.profile_step_card_features(actions.data(), repeats);
             }
             py::dict d;
             d["repeats"] = p.repeats;
             d["envs"] = p.envs;
             d["env_steps"] = p.env_steps;
             d["terminal_resets"] = p.terminal_resets;
             d["opponent_steps"] = p.opponent_steps;
             d["total_ns"] = p.total_ns;
             d["pre_options_ns"] = p.pre_options_ns;
             d["learner_step_ns"] = p.learner_step_ns;
             d["opponent_advance_ns"] = p.opponent_advance_ns;
             d["opponent_options_ns"] = p.opponent_options_ns;
             d["opponent_action_ns"] = p.opponent_action_ns;
             d["opponent_step_ns"] = p.opponent_step_ns;
             d["auto_pending_ns"] = p.auto_pending_ns;
             d["auto_main_options_ns"] = p.auto_main_options_ns;
             d["auto_main_apply_ns"] = p.auto_main_apply_ns;
             d["auto_pending_decisions"] = p.auto_pending_decisions;
             d["auto_main_decisions"] = p.auto_main_decisions;
             d["reward_reset_ns"] = p.reward_reset_ns;
             d["obs_ns"] = p.obs_ns;
             d["post_options_ns"] = p.post_options_ns;
             d["mask_ns"] = p.mask_ns;
             d["action_features_ns"] = p.action_features_ns;
             d["card_features_ns"] = p.card_features_ns;
             return d;
           },
           py::arg("actions"), py::arg("repeats") = 64,
           "Profile the native fused PPO step/card-feature path with reusable "
           "scratch buffers. Mutates the batch env like normal stepping.")
      .def("step_with_all_features",
           [](PpoBatchEnv& e,
              py::array_t<int, py::array::c_style | py::array::forcecast> actions) {
             int n = e.size(), D = rl_obs_dim();
             py::array_t<float> obs(Shape{n, D}), reward(Shape{n});
             py::array_t<uint8_t> done(Shape{n}), mask(Shape{n, RL_MAX_ACTIONS});
             py::array_t<int32_t> player(Shape{n}), result(Shape{n}),
                 episode_len(Shape{n});
             py::array_t<float> action_features(
                 Shape{n, RL_MAX_ACTIONS, PPO_ACTION_FEAT_DIM});
             py::array_t<float> card_features(
                 Shape{n, PPO_CARD_SLOTS, PPO_CARD_FEAT_DIM});
             py::array_t<float> deck_features(
                 Shape{n, PPO_DECK_SLOTS, PPO_CARD_FEAT_DIM});
             py::array_t<float> belief_features(
                 Shape{n, PPO_BELIEF_SLOTS, PPO_CARD_FEAT_DIM});
             py::array_t<float> belief_summary(Shape{n, PPO_BELIEF_SUMMARY_DIM});
             {
               py::gil_scoped_release rel;
               e.step(actions.data(), obs.mutable_data(), reward.mutable_data(),
                      done.mutable_data(), mask.mutable_data(),
                      player.mutable_data(), result.mutable_data(),
                      episode_len.mutable_data());
               e.action_features(action_features.mutable_data());
               e.card_features(card_features.mutable_data());
               e.deck_features(deck_features.mutable_data());
               e.belief_features(belief_features.mutable_data());
               e.belief_summary(belief_summary.mutable_data());
             }
             return py::make_tuple(obs, reward, done, mask, player, result,
                                   episode_len, action_features, card_features,
                                   deck_features, belief_features, belief_summary);
           },
           py::arg("actions"),
           "Step all games and return action/card/deck/belief features and summary "
           "in one pybind call.");
  m.def(
      "rl_selfplay",
      [](std::vector<int> d0, std::vector<int> d1, int n, uint64_t seed,
         int max_steps) {
        SelfplayResult r;
        {
          py::gil_scoped_release rel;
          r = rl_selfplay(d0, d1, n, seed, max_steps);
        }
        py::dict d;
        d["p0"] = r.p0_wins;
        d["p1"] = r.p1_wins;
        d["draw"] = r.draws;
        d["unfinished"] = r.unfinished;
        d["steps"] = r.total_steps;
        d["seconds"] = r.seconds;
        d["games_per_s"] = r.seconds > 0 ? n / r.seconds : 0.0;
        return d;
      },
      py::arg("deck0"), py::arg("deck1"), py::arg("n"), py::arg("seed") = 0,
      py::arg("max_steps") = 20000,
      "Run n random-policy games entirely in C++; returns a stats dict.");
}
