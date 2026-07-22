#include "ptcg/state.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "ptcg/card_db.hpp"
#include "ptcg/effect_vm.hpp"

namespace ptcg {

const char* intern_atom_string(const std::string& v) {
  // node-based set: element addresses are stable across inserts
  static std::mutex mu;
  static std::unordered_set<std::string> pool;
  std::lock_guard<std::mutex> lock(mu);
  return pool.insert(v).first->c_str();
}

static constexpr int CTX_MOVE_ENERGY_TO_ACTIVE = 33;
static constexpr int CTX_DISCARD_ENERGY = 30;
static constexpr int CTX_TO_BENCH = 5;
static constexpr int CTX_DISCARD = 8;
static constexpr int CTX_TO_DECK_BOTTOM = 10;
static constexpr int CTX_DECK_EVOLVE = 19;
static constexpr int CTX_ATTACK_TARGET = 15;
static constexpr int CTX_DAMAGE_COUNT = 39;
static constexpr int REPLAY_RANDOM_OPP_HAND_TO_DECK = -100000000;
static constexpr int REPLAY_COIN = -100000001;
static constexpr int REPLAY_HAND_INDEX = -100000002;
static constexpr int REPLAY_RANDOM_DISCARD_HAND = -100000003;
static constexpr int REPLAY_DRAW_PLAYER = -100000004;

static void emit_log(GameState& st, NativeLog log) {
  if (!st.collectLogs) return;  // opt-in: bridge/replay states only
  st.nativeLogs.push_back(log);
}

static void emit_card_log(GameState& st, int type, int player, int cardId) {
  NativeLog log;
  log.type = type;
  log.playerIndex = player;
  log.cardId = cardId;
  emit_log(st, log);
}

static void emit_move_log(GameState& st, int player, int cardId,
                          int fromArea, int toArea) {
  NativeLog log;
  log.type = 6;
  log.playerIndex = player;
  log.cardId = cardId;
  log.fromArea = fromArea;
  log.toArea = toArea;
  emit_log(st, log);
}

static void emit_result_log(GameState& st, int result, int reason = 0) {
  NativeLog log;
  log.type = 23;
  log.result = result;
  log.reason = reason;
  emit_log(st, log);
}

static void set_result(GameState& st, int result, int reason = 0) {
  if (st.result < 0 && result >= 0)
    emit_result_log(st, result, reason);
  st.result = result;
}

struct InPlaySnap {
  bool present = false;
  int id = 0;
  int hp = 0;
  SmallVec<int, 16> energyCards;
  SmallVec<int, 4> tools;
  SmallVec<int, 4> preEvo;
};

struct PlayerSnap {
  InPlaySnap active;
  SmallVec<InPlaySnap, 5> bench;
  SmallVec<int, 64> hand;
  int handCount = 0;
  SmallVec<int, 64> discard;
  SmallVec<int, 64> deck;
  int deckCount = 0;
  SmallVec<int, 8> prizes;
  int prizeCount = 0;
  bool poisoned = false;
  bool burned = false;
  bool asleep = false;
  bool paralyzed = false;
  bool confused = false;
};

struct StateSnap {
  PlayerSnap players[2];
  SmallVec<int, 1> stadium;
};

static InPlaySnap snap_inplay(const InPlay& p, bool present = true) {
  InPlaySnap out;
  out.present = present;
  out.id = p.id;
  out.hp = p.hp;
  out.energyCards = p.energyCardIds;
  out.tools = p.tools;
  out.preEvo = p.preEvo;
  return out;
}

static PlayerSnap snap_player(const Player& p) {
  PlayerSnap out;
  out.active = p.activeKnown ? snap_inplay(p.active, true) : InPlaySnap{};
  for (const auto& b : p.bench) out.bench.push_back(snap_inplay(b, true));
  out.hand = p.hand;
  out.handCount = p.handCount;
  out.discard = p.discard;
  out.deck = p.deck;
  out.deckCount = p.deckCount;
  out.prizes = p.prizes;
  out.prizeCount = p.prizeCount;
  out.poisoned = p.poisoned;
  out.burned = p.burned;
  out.asleep = p.asleep;
  out.paralyzed = p.paralyzed;
  out.confused = p.confused;
  return out;
}

static StateSnap snapshot_state(const GameState& st) {
  StateSnap out;
  out.players[0] = snap_player(st.players[0]);
  out.players[1] = snap_player(st.players[1]);
  out.stadium = st.stadium;
  return out;
}

template <typename Before, typename After>
static SmallVec<int, 64> multiset_added(const Before& before,
                                       const After& after) {
  SmallVec<int, 64> remaining;
  SmallVec<int, 64> out;
  for (int id : before)
    if (id > 0) remaining.push_back(id);
  std::sort(remaining.begin(), remaining.end());
  for (int id : after) {
    if (id <= 0) continue;
    auto it = std::lower_bound(remaining.begin(), remaining.end(), id);
    if (it != remaining.end() && *it == id)
      remaining.erase(it);
    else
      out.push_back(id);
  }
  std::sort(out.begin(), out.end());
  return out;
}

template <typename Before, typename After>
static SmallVec<int, 64> multiset_removed(const Before& before,
                                         const After& after) {
  return multiset_added(after, before);
}

static bool has_log_type_card_since(const GameState& st, size_t start,
                                    int type, int player, int cardId) {
  for (size_t i = start; i < st.nativeLogs.size(); ++i) {
    const NativeLog& log = st.nativeLogs[i];
    if (log.type == type && log.playerIndex == player &&
        (cardId <= 0 || log.cardId == cardId))
      return true;
  }
  return false;
}

static void emit_hp_delta(GameState& st, const InPlaySnap& before,
                          const InPlay* after, int owner,
                          bool putDamageCounter = false) {
  if (!before.present || !after || before.id != after->id ||
      before.hp == after->hp)
    return;
  NativeLog log;
  log.type = 16;
  log.playerIndex = owner;
  log.cardId = after->id;
  log.value = after->hp - before.hp;
  log.putDamageCounter = putDamageCounter;
  emit_log(st, log);
}

static void emit_inplay_attachment_deltas(GameState& st, const InPlaySnap& before,
                                          const InPlay* after, int owner,
                                          int area) {
  if (!after) return;
  for (int id : multiset_added(before.energyCards, after->energyCardIds))
    emit_move_log(st, owner, id, AREA_DISCARD, area);
  for (int id : multiset_removed(before.energyCards, after->energyCardIds))
    emit_move_log(st, owner, id, area, AREA_DISCARD);
  for (int id : multiset_added(before.tools, after->tools))
    emit_move_log(st, owner, id, AREA_HAND, area);
  for (int id : multiset_removed(before.tools, after->tools))
    emit_move_log(st, owner, id, area, AREA_DISCARD);
}

static void emit_condition_delta(GameState& st, bool before, bool after,
                                 int type, int player, int cardId) {
  if (before == after) return;
  NativeLog log;
  log.type = type;
  log.playerIndex = player;
  log.cardId = cardId;
  log.isRecover = !after;
  emit_log(st, log);
}

static void emit_player_delta_logs(GameState& st, const PlayerSnap& before,
                                   const Player& after, int player,
                                   size_t logStart) {
  for (int id : multiset_added(before.hand, after.hand)) {
    bool drew = after.deckCount < before.deckCount;
    if (drew && !has_log_type_card_since(st, logStart, 4, player, id))
      emit_card_log(st, 4, player, id);
  }
  for (int id : multiset_added(before.discard, after.discard)) {
    int from = AREA_HAND;
    auto removedDeck = multiset_removed(before.deck, after.deck);
    if (std::find(removedDeck.begin(), removedDeck.end(), id) != removedDeck.end())
      from = AREA_DECK;
    emit_move_log(st, player, id, from, AREA_DISCARD);
  }
  for (int id : multiset_added(before.prizes, after.prizes))
    emit_move_log(st, player, id, AREA_DECK, AREA_PRIZE);
  for (int id : multiset_removed(before.prizes, after.prizes))
    emit_move_log(st, player, id, AREA_PRIZE, AREA_HAND);

  const InPlay* afterActive = after.activeKnown ? &after.active : nullptr;
  emit_hp_delta(st, before.active, afterActive, player);
  emit_inplay_attachment_deltas(st, before.active, afterActive, player,
                                AREA_ACTIVE);
  const size_t nBench = std::min(before.bench.size(), after.bench.size());
  for (size_t i = 0; i < nBench; ++i) {
    emit_hp_delta(st, before.bench[i], &after.bench[i], player);
    emit_inplay_attachment_deltas(st, before.bench[i], &after.bench[i], player,
                                  AREA_BENCH);
  }
  if (after.activeKnown) {
    emit_condition_delta(st, before.poisoned, after.poisoned, 17, player,
                         after.active.id);
    emit_condition_delta(st, before.burned, after.burned, 18, player,
                         after.active.id);
    emit_condition_delta(st, before.asleep, after.asleep, 19, player,
                         after.active.id);
    emit_condition_delta(st, before.paralyzed, after.paralyzed, 20, player,
                         after.active.id);
    emit_condition_delta(st, before.confused, after.confused, 21, player,
                         after.active.id);
  }
}

static void emit_deep_delta_logs(GameState& st, const StateSnap& before,
                                 size_t logStart) {
  for (int p = 0; p < 2; ++p)
    emit_player_delta_logs(st, before.players[p], st.players[p], p, logStart);
  for (int id : multiset_added(before.stadium, st.stadium))
    emit_move_log(st, st.yourIndex, id, AREA_HAND, AREA_STADIUM);
  for (int id : multiset_removed(before.stadium, st.stadium))
    emit_move_log(st, 1 - st.yourIndex, id, AREA_STADIUM, AREA_DISCARD);
}

static bool is_ancient_supporter_card_id(int cardId);

// --- free-running RNG + deck helpers --------------------------------------

static inline uint64_t next_rand(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;  // xorshift64
  return s;
}

template <typename Cards>
static void shuffle_deck(Cards& d, uint64_t& rng) {
  for (size_t i = d.size(); i > 1; --i)
    std::swap(d[i - 1], d[next_rand(rng) % i]);
}

template <typename KnownList>
static void add_known_card(KnownList& known, int cid) {
  if (cid > 0) known.push_back(cid);
}

template <typename KnownList>
static bool consume_known_card(KnownList& known, int cid) {
  auto it = std::find(known.begin(), known.end(), cid);
  if (it == known.end()) return false;
  known.erase(it);
  return true;
}

static void refresh_deck_prize_union(Player& p) {
  if (p.deckPrizeKnownCards.empty()) return;
  if (static_cast<int>(p.deck.size()) != p.deckCount ||
      static_cast<int>(p.prizes.size()) != p.prizeCount)
    return;
  p.deckPrizeKnownCards.clear();
  for (int id : p.deck) p.deckPrizeKnownCards.push_back(id);
  for (int id : p.prizes) p.deckPrizeKnownCards.push_back(id);
  std::sort(p.deckPrizeKnownCards.begin(), p.deckPrizeKnownCards.end());
}

static void cap_deck_membership_to_unknown_positions(Player& p) {
  if (static_cast<int>(p.deck.size()) != p.deckCount) return;
  for (std::size_t index = 0; index < p.deckKnownCards.size();) {
    const int card = p.deckKnownCards[index];
    const int available = static_cast<int>(
        std::count(p.deck.begin(), p.deck.end(), card));
    int fixed = 0;
    const std::size_t slots =
        std::min(p.deck.size(), p.deckKnownMask.size());
    for (std::size_t slot = 0; slot < slots; ++slot) {
      if (p.deckKnownMask[slot] && p.deck[slot] == card) ++fixed;
    }
    const int accepted = static_cast<int>(std::count(
        p.deckKnownCards.begin(), p.deckKnownCards.begin() + index, card));
    if (fixed + accepted >= available) {
      p.deckKnownCards.erase(p.deckKnownCards.begin() + index);
    } else {
      ++index;
    }
  }
}

static void normalize_deck_knowledge(Player& p) {
  if (static_cast<int>(p.deck.size()) != p.deckCount) {
    p.deckKnown = false;
    p.deckKnownMask.clear();
    return;
  }
  if (p.deckKnownMask.size() < p.deck.size())
    p.deckKnownMask.resize(p.deck.size(), p.deckKnown);
  if (p.deckKnownMask.size() > p.deck.size())
    p.deckKnownMask.resize(p.deck.size());
  p.deckKnown = !p.deck.empty() || p.deckCount == 0;
  for (bool known : p.deckKnownMask)
    if (!known) {
      p.deckKnown = false;
      break;
    }
  // Exact positional knowledge already names every remaining deck card.
  // Retaining unordered membership as well represents the same physical
  // copies twice and makes a later shuffle append duplicates.
  if (p.deckKnown)
    p.deckKnownCards.clear();
  else
    cap_deck_membership_to_unknown_positions(p);
  refresh_deck_prize_union(p);
}

static void mark_deck_known(Player& p) {
  if (static_cast<int>(p.deck.size()) == p.deckCount) {
    p.deckKnownMask.assign(p.deck.size(), true);
    p.deckKnownCards.clear();
    p.deckKnown = true;
  }
}

void mark_full_deck_inspected(Player& p) {
  mark_deck_known(p);
  if (!p.deckKnown) return;
  p.ownDeckInspected = true;

  // A player knows their submitted decklist and every non-Prize card that is
  // outside the deck. Once the complete remaining deck has been inspected,
  // the remaining Prize multiset is therefore a deterministic deduction.
  // Only materialize it when the engine carries every exact identity.
  if (static_cast<int>(p.prizes.size()) != p.prizeCount ||
      !std::all_of(p.prizes.begin(), p.prizes.end(),
                   [](int card_id) { return card_id > 0; })) {
    return;
  }
  p.ownPrizesInferred = true;
  p.deckPrizeKnownCards.clear();
  for (int id : p.deck) p.deckPrizeKnownCards.push_back(id);
  for (int id : p.prizes) p.deckPrizeKnownCards.push_back(id);
  std::sort(p.deckPrizeKnownCards.begin(), p.deckPrizeKnownCards.end());
}

static void shuffle_deck_known(Player& p, uint64_t& rng) {
  if (p.deckKnown) {
    p.deckKnownCards.clear();
    for (int cid : p.deck) add_known_card(p.deckKnownCards, cid);
  } else if (p.deckKnownMask.size() == p.deck.size()) {
    for (size_t i = 0; i < p.deck.size(); ++i)
      if (p.deckKnownMask[i]) add_known_card(p.deckKnownCards, p.deck[i]);
  }
  shuffle_deck(p.deck, rng);
  p.deckKnownMask.clear();
  p.deckKnown = false;
  cap_deck_membership_to_unknown_positions(p);
}

static bool deck_card_known_at(const Player& p, int idx) {
  if (idx < 0 || idx >= static_cast<int>(p.deck.size())) return false;
  if (p.deckKnown) return true;
  return idx < static_cast<int>(p.deckKnownMask.size()) && p.deckKnownMask[idx];
}

static int erase_deck_at(Player& p, int idx, bool* knownOut = nullptr) {
  if (knownOut) *knownOut = false;
  if (idx < 0 || idx >= static_cast<int>(p.deck.size())) return 0;
  int id = p.deck[idx];
  bool known = deck_card_known_at(p, idx);
  auto kit = std::find(p.deckKnownCards.begin(), p.deckKnownCards.end(), id);
  if (kit != p.deckKnownCards.end()) {
    known = true;
    p.deckKnownCards.erase(kit);
  }
  if (idx < static_cast<int>(p.deckKnownMask.size()))
    p.deckKnownMask.erase(p.deckKnownMask.begin() + idx);
  p.deck.erase(p.deck.begin() + idx);
  p.deckCount -= 1;
  normalize_deck_knowledge(p);
  if (knownOut) *knownOut = known;
  return id;
}

static void push_deck_top(Player& p, int id, bool known) {
  p.deck.push_back(id);
  p.deckCount += 1;
  if (p.deckKnownMask.size() < p.deck.size() - 1)
    p.deckKnownMask.resize(p.deck.size() - 1, p.deckKnown);
  p.deckKnownMask.push_back(known);
  normalize_deck_knowledge(p);
}

static void push_deck_bottom(Player& p, int id, bool known) {
  p.deck.insert(p.deck.begin(), id);
  p.deckCount += 1;
  if (p.deckKnownMask.size() < p.deck.size() - 1)
    p.deckKnownMask.resize(p.deck.size() - 1, p.deckKnown);
  p.deckKnownMask.insert(p.deckKnownMask.begin(), known);
  normalize_deck_knowledge(p);
}

static void mark_completed_trainer_play(GameState& st, int cardId) {
  if (cardId == 1233)
    st.canariPlayed = true;
  if (cardId == 1238)
    st.tarragonPlayed = true;
  if (is_ancient_supporter_card_id(cardId))
    st.ancientSupporterPlayed = true;
}

static void discard_pending_trainer(GameState& st) {
  if (st.pendingTrainerDiscard < 0 || st.pendingTrainerOwner < 0) return;
  int cardId = st.pendingTrainerDiscard;
  st.players[st.pendingTrainerOwner].discard.push_back(cardId);
  mark_completed_trainer_play(st, cardId);
  st.pendingTrainerDiscard = -1;
  st.pendingTrainerOwner = -1;
}

static int discard_deck_top(Player& p, bool* knownOut = nullptr) {
  if (knownOut) *knownOut = false;
  if (p.deckCount <= 0) return 0;
  if (!p.deck.empty()) {
    return erase_deck_at(p, static_cast<int>(p.deck.size()) - 1, knownOut);
  }
  p.deckCount -= 1;
  p.deckKnown = false;
  p.deckKnownMask.clear();
  return 0;
}

static bool consume_replay_event(GameState& st, int tag, int& value);
static void remove_hand_card(GameState& st, Player& p, int cardId);

// One coin flip. In replay mode, consume cabt's recorded outcome from the
// per-action tape; otherwise fall back to the free-running RNG.
static inline bool flip_heads(GameState& st) {
  int replay = 0;
  bool heads = false;
  if (consume_replay_event(st, REPLAY_COIN, replay))
    heads = replay != 0;
  else
    heads = (next_rand(st.rng) & 1ULL) != 0;
  NativeLog log;
  log.type = 22;
  log.playerIndex = st.yourIndex;
  log.head = heads;
  emit_log(st, log);
  return heads;
}

// Special conditions live on the Player (its single Active). They all clear
// when the Active leaves play (switch / retreat / promote / KO).
static void clear_status(Player& p) {
  p.poisoned = p.burned = p.asleep = p.paralyzed = p.confused = false;
  p.poisonDamageCounters = 1;
}

static void clear_active_spot_locks(InPlay& p) {
  p.activeLockId = 0;
  p.delayedKoTurn = -1;
  p.delayedKoPromoteBeforePrize = false;
  p.attackBonusTurn = -1;
  p.attackBonus = 0;
}

static bool is_antique_fossil(int id) {
  return id == 1099 || id == 1136 || id == 1138 || id == 1150 || id == 1151;
}

// Inflict a condition. Poison/Burn are independent; Asleep/Paralyzed/Confused
// are mutually exclusive (a Pokemon holds at most one of the three).
static void apply_condition(Player& p, int status) {
  if (p.activeKnown && is_antique_fossil(p.active.id))
    return;  // Antique Fossils can't be affected by Special Conditions.
  if (status == ST_ASLEEP && p.activeKnown && p.active.id == 250)
    return;  // Hoothoot: Insomnia
  switch (status) {
    case ST_POISON: p.poisoned = true; p.poisonDamageCounters = 1; break;
    case ST_BURN: p.burned = true; break;
    case ST_ASLEEP: p.asleep = true; p.paralyzed = p.confused = false; break;
    case ST_PARALYZED: p.paralyzed = true; p.asleep = p.confused = false; break;
    case ST_CONFUSED: p.confused = true; p.asleep = p.paralyzed = false; break;
  }
}

static int in_play_count_id(const Player& p, int cardId) {
  int n = 0;
  if (p.activeKnown && p.active.id == cardId) ++n;
  for (const auto& b : p.bench)
    if (b.id == cardId) ++n;
  return n;
}

static bool festival_status_immunity(const GameState& st, int side);

static void apply_mismagius_switch_passive(GameState& st, int movingSide,
                                           int movedBenchIdx = -1) {
  if (movingSide != st.yourIndex) return;  // only during that player's own turn
  Player& controller = st.players[1 - movingSide];
  if (controller.activeKnown && controller.active.id == 813 &&
      !festival_status_immunity(st, movingSide))
    apply_condition(st.players[movingSide], ST_CONFUSED);
  int dugtrio = in_play_count_id(controller, 882);
  Player& moving = st.players[movingSide];
  if (dugtrio > 0 && movedBenchIdx >= 0 &&
      movedBenchIdx < static_cast<int>(moving.bench.size())) {
    InPlay& moved = moving.bench[movedBenchIdx];
    moved.hp -= 20 * dugtrio;
    if (moved.hp < 0) moved.hp = 0;
  }
}

static InPlay make_inplay(int cardId) {
  const CardInfo* c = find_card(cardId);
  InPlay p;
  p.id = cardId;
  p.maxHp = c ? c->hp : 0;
  p.hp = p.maxHp;
  return p;
}

// Draw n cards to hand. Free-running pops the real deck; replay takes from the
// per-action tape cursor (or leaves the hand unknown if neither is available).
static int draw_n(GameState& st, Player& p, int n) {
  int drawn = 0;
  int side = (&p == &st.players[0]) ? 0 : ((&p == &st.players[1]) ? 1 : -1);
  for (int i = 0; i < n; ++i) {
    if (p.deckCount <= 0) break;
    if (p.handPublicKnown) {
      p.handKnownCards.clear();
      for (int id : p.hand) add_known_card(p.handKnownCards, id);
      p.handPublicKnown = false;
    }
    p.deckCount -= 1;
    p.handCount += 1;
    ++drawn;
    int replayId = -1;
    bool hasReplayDraw = false;
    if (st.replayTapePos < static_cast<int>(st.replayTape.size())) {
      int tagOrId = st.replayTape[st.replayTapePos];
      if (tagOrId == REPLAY_DRAW_PLAYER &&
          st.replayTapePos + 2 < static_cast<int>(st.replayTape.size()) &&
          st.replayTape[st.replayTapePos + 1] == side) {
        replayId = st.replayTape[st.replayTapePos + 2];
        st.replayTapePos += 3;
        hasReplayDraw = true;
      } else if (tagOrId >= 0) {
        replayId = tagOrId;
        st.replayTapePos += 1;
        hasReplayDraw = true;
      }
    }
    int drawnId = 0;
    if (hasReplayDraw) {
      int id = replayId;
      drawnId = id;
      bool sourceKnown = p.deckKnown;
      p.hand.push_back(id);
      int deckIdx = -1;
      for (int j = static_cast<int>(p.deck.size()) - 1; j >= 0; --j) {
        if (p.deck[j] == id) {
          deckIdx = j;
          break;
        }
      }
      if (deckIdx >= 0) {
        if (p.deckKnownMask.size() == p.deck.size()) {
          sourceKnown = sourceKnown || p.deckKnownMask[deckIdx];
          p.deckKnownMask.erase(p.deckKnownMask.begin() + deckIdx);
        }
        p.deck.erase(p.deck.begin() + deckIdx);
      } else {
        p.deckKnown = false;
        p.deckKnownMask.clear();
      }
      auto kit = std::find(p.deckKnownCards.begin(), p.deckKnownCards.end(), id);
      if (kit != p.deckKnownCards.end()) {
        sourceKnown = true;
        p.deckKnownCards.erase(kit);
      }
      if (sourceKnown && !p.handKnown) add_known_card(p.handKnownCards, id);
    } else if (!p.deck.empty()) {
      int id = p.deck.back();
      drawnId = id;
      bool sourceKnown = p.deckKnown ||
                         (!p.deckKnownMask.empty() && p.deckKnownMask.back());
      auto kit = std::find(p.deckKnownCards.begin(), p.deckKnownCards.end(), id);
      if (kit != p.deckKnownCards.end()) {
        sourceKnown = true;
        p.deckKnownCards.erase(kit);
      }
      p.hand.push_back(p.deck.back());
      p.deck.pop_back();
      if (!p.deckKnownMask.empty()) p.deckKnownMask.pop_back();
      if (sourceKnown && !p.handKnown) add_known_card(p.handKnownCards, id);
    } else {
      p.deckKnown = false;
      p.deckKnownMask.clear();
      p.handKnown = false;
    }
    if (side >= 0 && drawnId > 0)
      emit_card_log(st, 4, side, drawnId);
    normalize_deck_knowledge(p);
  }
  return drawn;
}

static bool consume_replay_event(GameState& st, int tag, int& value) {
  if (st.replayTapePos + 1 >= static_cast<int>(st.replayTape.size()))
    return false;
  if (st.replayTape[st.replayTapePos] != tag) return false;
  value = st.replayTape[st.replayTapePos + 1];
  st.replayTapePos += 2;
  return true;
}

static void remove_hand_card(GameState& st, Player& p, int cardId) {
  bool removed = false;
  int handIdx = -1;
  if (consume_replay_event(st, REPLAY_HAND_INDEX, handIdx) &&
      handIdx >= 0 && handIdx < static_cast<int>(p.hand.size()) &&
      p.hand[handIdx] == cardId) {
    p.hand.erase(p.hand.begin() + handIdx);
    removed = true;
  }
  if (!removed) {
    auto it = std::find(p.hand.begin(), p.hand.end(), cardId);
    if (it != p.hand.end()) p.hand.erase(it);
  }
  // handKnownCards is observer knowledge, independent of whether the engine
  // carries the owner's exact raw hand. A publicly known card that leaves the
  // hand must consume exactly one unordered membership entry; otherwise the
  // same physical copy is counted again in its public destination zone.
  consume_known_card(p.handKnownCards, cardId);
  p.handCount -= 1;
}

static int discard_hand_to_discard(Player& p) {
  int discarded = p.handCount;
  for (int id : p.hand) p.discard.push_back(id);
  p.hand.clear();
  p.handCount = 0;
  p.handKnown = true;  // an empty hand is known even if discarded identities were not
  return discarded;
}

// --- effect VM (op interpreter) ------------------------------------------
template <typename Cards>
static void remove_one(Cards& v, int id);    // defined below
static void post_attack(GameState& st, int attacker);   // defined below
static void checkup_resolve(GameState& st);             // defined below
static int effective_max_hp(int cardId, const SmallVec<int, 4>& tools,
                            const GameState& st,
                            const SmallVec<int, 16>& energyCardIds = {},
                            const SmallVec<int, 16>& energies = {},
                            int ownerSide = -1);
static void refresh_inplay_max_hp(GameState& st, InPlay& p, int ownerSide = -1);
static int stadium_damage_mod(const GameState& st, int attackerId,
                              int defenderId, int dmg);
static void set_promote_pending(GameState& st, int who); // defined below
static void continue_active_ko_after_amulet(GameState& st, int taker, int owner,
                                            int prizeValue);
static int cur_stadium(const GameState& st);             // defined below
static void apply_stadium_hp_change(GameState& st, int oldId, int newId);
static void enforce_rotom_tool_limits(GameState& st);
static void enforce_area_zero_bench_limits(GameState& st);
static int discard_tools_from_inplay(GameState& st, Player& owner, InPlay& p,
                                     int count);
static bool in_play_has(const Player& p, int cardId);
static bool is_ancient_card_id(int id);
static bool discard_has_name(const Player& p, const char* sub);
static bool hand_has_basic_energy(const Player& p, int etype = -1);
static void enforce_festival_status_recovery(GameState& st);
static int retreat_cost(const GameState& st, int player);
static void clear_status_for_evolve(GameState& st, int side, bool activeSpot,
                                    InPlay* evolved = nullptr);
static bool grand_tree_basic_target_ok(const GameState& st, int side,
                                       const InPlay& p);
static bool grand_tree_has_stage1_option(const GameState& st, int side);
static bool start_grand_tree(GameState& st);
static void resolve_grand_tree(GameState& st, const std::vector<int>& sel);
static bool ogres_mask_available(const GameState& st, int side);
static void play_ogres_mask(GameState& st);
static void resolve_ogres_mask(GameState& st, const std::vector<int>& sel);
static void queue_palafin_zero_to_hero(GameState& st, int owner, int benchIdx,
                                       bool afterCurrentProgram);
static void resolve_palafin_zero_to_hero(GameState& st, const std::vector<int>& sel);
static void set_palafin_yesno_pending(GameState& st, EffectFrame& fr);
static void set_damage_trigger_order_pending_from_frame(GameState& st,
                                                        const EffectFrame& fr);
static void start_slowking_seek_inspiration(GameState& st);
static void resolve_slowking_seek_inspiration(GameState& st,
                                              const std::vector<int>& sel);
static void start_ninetales_shapeshifter(GameState& st);
static void play_bother_bot(GameState& st);
static void resolve_bother_bot(GameState& st, const std::vector<int>& sel);
static void play_hand_trimmer(GameState& st);
static void resolve_hand_trimmer(GameState& st, const std::vector<int>& sel);
static int encode_copied_attack_ref(int attackId, int damage);
static int copied_attack_damage(int ref);
static void advance_effect_stack(GameState& st, const std::vector<int>& tape);
static bool start_evolve_trigger_order(GameState& st, int actor, int targetIdx,
                                       int cardId);
static void resolve_evolve_trigger_order(GameState& st,
                                         const std::vector<int>& sel,
                                         const std::vector<int>& tape);
static bool start_on_play_basic_trigger_order(GameState& st, int actor,
                                              int benchIdx, int cardId);
static void resolve_on_play_basic_trigger_order(GameState& st,
                                                const std::vector<int>& sel,
                                                const std::vector<int>& tape);
static bool start_on_attach_trigger_order(GameState& st, int actor,
                                          int targetIdx, int targetCardId,
                                          int energyCardId);
static void resolve_on_attach_trigger_order(GameState& st,
                                            const std::vector<int>& sel,
                                            const std::vector<int>& tape);

static const AttackInfo* card_attack(int cardId, int attackId) {
  const CardInfo* c = find_card(cardId);
  if (c)
    for (int k = 0; k < c->n_attacks; ++k)
      if (c->attacks[k].id == attackId) return &c->attacks[k];
  return nullptr;
}

static bool direct_opp_active_copy_attack(int cardId, int attackId) {
  return (cardId == 615 && attackId == 886) ||
         (cardId == 958 && attackId == 1378);
}

static const AttackInfo* opponent_active_attack_info(const GameState& st,
                                                     int actor,
                                                     int attackId) {
  if (actor < 0 || actor > 1) return nullptr;
  const Player& opp = st.players[1 - actor];
  if (!opp.activeKnown) return nullptr;
  return card_attack(opp.active.id, attackId);
}

static int attack_base_damage(int cardId, int attackId) {
  if (const AttackInfo* at = card_attack(cardId, attackId))
    return at->damage;
  if (attackId == 1556) {
    if (const AttackInfo* at = card_attack(1180, attackId))  // Core Memory
      return at->damage;
  }
  return 0;
}

static void end_turn(GameState& st);
static int contextual_prize_value(GameState& st, int taker, int owner,
                                  const InPlay& knocked, bool activeKo);
static void note_pending_ko_aura(GameState& st, int owner,
                                 const InPlay& knocked);
static void clear_pending_ko_auras(GameState& st);
static void ko_active(Player& p);
static void ko_bench(Player& p, int j);
static void set_prize_pending(GameState& st, int attacker);
static bool has_zero_hp_pokemon(const GameState& st);
static bool resolve_main_action_zero_hp(GameState& st, int resumePlayer,
                                        int firstKoSide);
static void continue_ko_trigger_order(GameState& st);

// Common tail when a program completes: pop its frame, clear pending, run a
// trainer's deferred discard, and (for an attack) resolve the KO / end the turn.
static void restore_top_deck_counted_out(GameState& st, EffectFrame& fr) {
  if (fr.topDeckCountedOut <= 0) return;
  if (fr.topDeckOwner >= 0 && fr.topDeckOwner < 2)
    st.players[fr.topDeckOwner].deckCount += fr.topDeckCountedOut;
  fr.topDeckCountedOut = 0;
}

static void complete_attack_program(GameState& st, int attacker, int attackId,
                                    int attackCardId) {
  st.lastAttackTurn[attacker] = st.turn;
  st.lastAttackId[attacker] = attackId;
  if (is_ancient_card_id(attackCardId)) {
    st.lastAncientAttackTurn[attacker] = st.turn;
    st.lastAncientAttackCard[attacker] = attackCardId;
    st.lastAncientAttackSerial[attacker] =
        st.players[attacker].activeKnown ? st.players[attacker].active.serial : 0;
  }
  st.endTurnAfterProgram = false;
  st.yourIndex = attacker;
  post_attack(st, attacker);
}

static void finish_program(GameState& st) {
  restore_top_deck_counted_out(st, st.effectStack.back());
  EffectFrame done = st.effectStack.back();
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  bool deferPostAttack = done.attackId > 0 && !done.copiedAttack &&
                         !st.afterProgramQueue.empty();
  if (deferPostAttack) {
    st.deferredPostAttack = true;
    st.deferredPostAttackPlayer = done.a;
    st.deferredPostAttackId = done.attackId;
    st.deferredPostAttackCard = done.attackCardId;
  }
  if (!st.afterProgramQueue.empty()) {
    for (auto it = st.afterProgramQueue.rbegin();
         it != st.afterProgramQueue.rend(); ++it)
      st.effectStack.push_back(*it);
    st.afterProgramQueue.clear();
    if (!st.effectStack.empty() &&
        st.effectStack.back().effect == EFF_PALAFIN_ZERO_TO_HERO) {
      st.yourIndex = st.effectStack.back().a;
      set_palafin_yesno_pending(st, st.effectStack.back());
    }
    if (!st.effectStack.empty() &&
        st.effectStack.back().effect == EFF_DAMAGE_TRIGGER_ORDER) {
      set_damage_trigger_order_pending_from_frame(st, st.effectStack.back());
    }
  }
  bool shouldEndTurn = st.endTurnAfterProgram && done.attackId == 0 &&
                       st.effectStack.empty() && st.pendingTrainerDiscard >= 0;
  if (st.effectStack.empty() && st.pendingTrainerDiscard >= 0)
    discard_pending_trainer(st);
  if (done.attackId > 0 && !done.copiedAttack) {
    if (!deferPostAttack)
      complete_attack_program(st, done.a, done.attackId, done.attackCardId);
  } else if (shouldEndTurn) {
    st.endTurnAfterProgram = false;
    end_turn(st);
  } else if (!st.effectStack.empty() &&
             st.effectStack.back().effect == EFF_KO_RESUME) {
    st.effectStack.pop_back();
    checkup_resolve(st);
  } else if (!st.effectStack.empty() &&
             st.effectStack.back().effect == EFF_KO_TRIGGER_ORDER) {
    continue_ko_trigger_order(st);
  } else if (done.attackId == 0 && !st.deferredPostAttack &&
             st.effectStack.empty() && !st.has_pending() &&
             has_zero_hp_pokemon(st)) {
    const CardInfo* source =
        done.sourceCardId > 0 ? find_card(done.sourceCardId) : nullptr;
    if (source && source->cardType == POKEMON)
      resolve_main_action_zero_hp(st, done.a, done.a);
  }
  if (st.effectStack.empty() && !st.has_pending() && st.deferredPostAttack) {
    int attacker = st.deferredPostAttackPlayer;
    int attackId = st.deferredPostAttackId;
    int attackCardId = st.deferredPostAttackCard;
    st.deferredPostAttack = false;
    st.deferredPostAttackPlayer = -1;
    st.deferredPostAttackId = 0;
    st.deferredPostAttackCard = 0;
    complete_attack_program(st, attacker, attackId, attackCardId);
  }
}

// Case-insensitive substring (for PF_NAME, e.g. name contains "Iono's").
static bool name_contains(const char* name, const char* sub) {
  if (!name || !sub) return false;
  std::string n(name), s(sub);
  auto lower = [](std::string& x) {
    std::transform(x.begin(), x.end(), x.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  };
  lower(n);
  lower(s);
  return n.find(s) != std::string::npos;
}

static bool name_starts_with(const char* name, const char* prefix) {
  if (!name || !prefix) return false;
  std::string n(name), p(prefix);
  auto lower = [](std::string& x) {
    std::transform(x.begin(), x.end(), x.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  };
  lower(n);
  lower(p);
  return n.rfind(p, 0) == 0;
}

static bool is_ns_pokemon(const CardInfo* c) {
  return c && c->cardType == POKEMON && name_starts_with(c->name, "N's ");
}

static bool is_team_rocket_card_id(int cardId) {
  const CardInfo* c = find_card(cardId);
  return c && name_contains(c->name, "Team Rocket");
}

static bool is_ancient_supporter_card_id(int cardId) {
  return cardId == 1185;  // Explorer's Guidance
}

static bool is_ethan_card_id(int cardId) {
  const CardInfo* c = find_card(cardId);
  return c && name_contains(c->name, "Ethan");
}

static bool is_hop_card_id(int cardId) {
  const CardInfo* c = find_card(cardId);
  return c && name_contains(c->name, "Hop");
}

static bool player_has_tera(const Player& p) {
  auto tera = [](const InPlay& k) {
    const CardInfo* c = find_card(k.id);
    return c && c->tera;
  };
  if (p.activeKnown && tera(p.active)) return true;
  for (const auto& b : p.bench)
    if (tera(b)) return true;
  return false;
}

static void record_ko(GameState& st, int owner, int cardId) {
  st.lastKoTurn[owner] = st.turn;
  if (is_team_rocket_card_id(cardId))
    st.lastTeamRocketKoTurn[owner] = st.turn;
  if (is_ethan_card_id(cardId))
    st.lastEthanKoTurn[owner] = st.turn;
}

static void record_attack_damage_ko_marker(GameState& st, int owner,
                                           const InPlay& knocked) {
  if (knocked.damagedByAttackTurn != st.turn ||
      knocked.damagedByAttackSide != 1 - owner)
    return;
  st.lastAttackDamageKoTurn[owner] = st.turn;
  if (is_hop_card_id(knocked.id))
    st.lastHopAttackDamageKoTurn[owner] = st.turn;
}

static bool eval_pred_clause(const CardInfo& c, const PredClause& p) {
  if (p.field == PF_NAME) {
    bool contains = name_contains(c.name, pred_string(p.val));
    return p.cmp == PC_NE ? !contains : contains;
  }
  int lhs = 0;
  switch (p.field) {
    case PF_CARDTYPE: lhs = c.cardType; break;
    case PF_ENERGYTYPE: lhs = c.energyType; break;
    case PF_HP: lhs = c.hp; break;
    case PF_BASIC: lhs = c.basic; break;
    case PF_STAGE1: lhs = c.stage1; break;
    case PF_STAGE2: lhs = c.stage2; break;
    case PF_EX: lhs = c.ex; break;
    case PF_MEGAEX: lhs = c.megaEx; break;
    case PF_TERA: lhs = c.tera; break;
    case PF_ACESPEC: lhs = c.aceSpec; break;
    case PF_RULEBOX: lhs = (c.ex || c.megaEx || c.tera) ? 1 : 0; break;
    case PF_WEAKNESS: lhs = c.weakness; break;
    case PF_RESISTANCE: lhs = c.resistance; break;
    case PF_HAS_ABILITY: lhs = c.hasAbility; break;
    default: return false;
  }
  switch (p.cmp) {
    case PC_EQ: return lhs == p.val;
    case PC_NE: return lhs != p.val;
    case PC_LE: return lhs <= p.val;
    case PC_GE: return lhs >= p.val;
    case PC_LT: return lhs < p.val;
    case PC_GT: return lhs > p.val;
    default: return false;
  }
}

// Evaluate a generated predicate (OR of AND-clause groups over card definition).
static bool matches_predicate(int cardId, int predId) {
  const CardInfo* c = find_card(cardId);
  if (!c) return false;
  bool groupOk = true;
  bool sawClause = false;
  for (const PredClause* p = pred_clauses() + predicate_start(predId);
       p->field != -1; ++p) {
    if (p->field == -2) {
      if (groupOk && sawClause) return true;
      groupOk = true;
      sawClause = false;
      continue;
    }
    sawClause = true;
    if (groupOk && !eval_pred_clause(*c, *p)) groupOk = false;
  }
  return groupOk && sawClause;
}

static bool is_future_card(int id) {
  switch (id) {
    case 27:
    case 37:
    case 75:
    case 80:
    case 87:
    case 192:
    case 313:
    case 957:
    case 971:
      return true;
    default:
      return false;
  }
}

static bool is_ancient_card_id(int id) {
  switch (id) {
    case 35:
    case 46:
    case 56:
    case 58:
    case 61:
    case 62:
    case 63:
    case 171:
    case 226:
    case 312:
    case 969:
    case 986:
    case 1085:
    case 1185:
      return true;
    default:
      return false;
  }
}

static bool pokemon_has_type(int cardId, int type) {
  const CardInfo* c = find_card(cardId);
  if (!c || c->cardType != POKEMON) return false;
  if (cardId == 205)
    return type == GRASS || type == FIRE;  // Scovillain ex: Double Type
  if (cardId == 976)
    return type == FIGHTING || type == PSYCHIC;  // Carbink: Jewel Body
  return c->energyType == type;
}

static int effective_weakness(const GameState& st, int attackerSide,
                              const CardInfo* defender) {
  if (!defender) return -1;
  if (defender->energyType == DRAGON && in_play_has(st.players[attackerSide], 272))
    return PSYCHIC;  // Lillie's Clefairy ex: Fairy Zone
  return defender->weakness;
}

static bool matches_filter(int id, int filter) {
  if (filter >= FILTER_PREDICATE_BASE)
    return matches_predicate(id, filter - FILTER_PREDICATE_BASE);
  if (filter == F_ANY) return true;
  if (filter == F_FENERGY) return id == 6;  // Basic Fighting Energy
  const CardInfo* c = find_card(id);
  if (filter == F_POKEMON) return c && c->cardType == POKEMON;
  if (filter == F_FENERGY_OR_FBASIC)
    return id == 6 || (c && c->cardType == POKEMON && c->basic &&
                       c->energyType == FIGHTING);
  if (filter == F_POKEMON_NORULEBOX)
    return c && c->cardType == POKEMON && !c->ex && !c->megaEx && !c->tera;
  if (filter == F_SUPPORTER) return c && c->cardType == SUPPORTER;
  if (filter == F_TRAINER)
    return c && (c->cardType == ITEM || c->cardType == SUPPORTER ||
                 c->cardType == STADIUM || c->cardType == TOOL);
  if (filter == F_EVOLUTION)
    return c && c->cardType == POKEMON && (c->stage1 || c->stage2);
  if (filter == F_BASIC_ENERGY) return c && c->cardType == BASIC_ENERGY;
  if (filter == F_BASIC_POKEMON)
    return c && c->cardType == POKEMON && c->basic;
  if (filter == F_POKEMON_OR_BASIC_ENERGY)
    return c && (c->cardType == POKEMON || c->cardType == BASIC_ENERGY);
  if (filter == F_FUTURE_POKEMON)
    return c && c->cardType == POKEMON && is_future_card(id);
  if (filter == F_DAMAGED || filter == F_DAMAGE_COUNTERS_EQ_6)
    return false;  // in-play only
  return false;
}

static const InPlay* inplay_ref_const(const Player& p, int idx) {
  if (idx < 0) return p.activeKnown ? &p.active : nullptr;
  return idx < static_cast<int>(p.bench.size()) ? &p.bench[idx] : nullptr;
}

static const InPlay* effect_self(const GameState& st, const EffectFrame& fr) {
  const Player& me = st.players[fr.a];
  return inplay_ref_const(me, fr.selfBench);
}

static bool card_evolves_from(int evoId, int baseId, bool requireNoAbility) {
  if (evoId == 107) return false;  // Palafin ex: only Zero to Hero can put it in play.
  const CardInfo* evo = find_card(evoId);
  const CardInfo* base = find_card(baseId);
  return evo && base && evo->cardType == POKEMON && evo->evolvesFrom &&
         base->name && std::strcmp(evo->evolvesFrom, base->name) == 0 &&
         (!requireNoAbility || !evo->hasAbility);
}

static bool stage2_evolves_from_basic(int stage2Id, int basicId) {
  const CardInfo* stage2 = find_card(stage2Id);
  const CardInfo* basic = find_card(basicId);
  if (!stage2 || !basic || !stage2->stage2 || !basic->basic ||
      !stage2->evolvesFrom || !basic->name)
    return false;
  for (const CardInfo& mid : card_db()) {
    if (!mid.stage1 || !mid.name || !mid.evolvesFrom) continue;
    if (std::strcmp(stage2->evolvesFrom, mid.name) == 0 &&
        std::strcmp(mid.evolvesFrom, basic->name) == 0)
      return true;
  }
  return false;
}

static bool hand_has_stage2_for_basic(const Player& owner, int basicId) {
  for (int id : owner.hand)
    if (stage2_evolves_from_basic(id, basicId)) return true;
  return false;
}

static bool deck_has_evolution_for(const Player& owner, const InPlay& base,
                                   bool requireNoAbility) {
  for (int id : owner.deck)
    if (card_evolves_from(id, base.id, requireNoAbility)) return true;
  return false;
}

static bool inplay_name_matches(const Player& side, const char* name) {
  if (!name) return false;
  if (side.activeKnown) {
    const CardInfo* c = find_card(side.active.id);
    if (c && c->name && std::strcmp(c->name, name) == 0) return true;
  }
  for (const auto& b : side.bench) {
    const CardInfo* c = find_card(b.id);
    if (c && c->name && std::strcmp(c->name, name) == 0) return true;
  }
  return false;
}

static bool ns_pokemon_has_copyable_attack(int id) {
  const CardInfo* c = find_card(id);
  if (!is_ns_pokemon(c))
    return false;
  for (int i = 0; i < c->n_attacks; ++i)
    if (c->attacks[i].id != 403)
      return true;
  return false;
}

static bool matches_dynamic_card_filter(const GameState& st, const EffectFrame& fr,
                                        int id, int filter) {
  const Player& me = st.players[fr.a];
  const Player& opp = st.players[1 - fr.a];
  if (filter == F_EVOLVES_FROM_SELF) {
    const InPlay* self = effect_self(st, fr);
    return self && card_evolves_from(id, self->id, false);
  }
  if (filter == F_EVOLVES_FROM_SAVED_OWN_INPLAY ||
      filter == F_EVOLVES_FROM_SAVED_OWN_INPLAY_NO_ABILITY) {
    if (fr.savedScratch.empty()) return false;
    const InPlay* target = inplay_ref_const(me, fr.savedScratch[0]);
    return target && card_evolves_from(
                         id, target->id,
                         filter == F_EVOLVES_FROM_SAVED_OWN_INPLAY_NO_ABILITY);
  }
  if (filter == F_SAME_NAME_AS_OWN_INPLAY ||
      filter == F_SAME_NAME_AS_OPP_INPLAY) {
    const CardInfo* c = find_card(id);
    if (!c || c->cardType != POKEMON) return false;
    return inplay_name_matches(filter == F_SAME_NAME_AS_OWN_INPLAY ? me : opp,
                               c->name);
  }
  if (filter == F_STAGE2_EVOLVES_FROM_SAVED_BASIC) {
    if (fr.savedScratch.empty()) return false;
    const InPlay* target = inplay_ref_const(me, fr.savedScratch[0]);
    return target && stage2_evolves_from_basic(id, target->id);
  }
  if (filter == F_EVOLVES_FROM_OWN_INPLAY ||
      filter == F_EVOLVES_FROM_OWN_INPLAY_NO_ABILITY) {
    bool requireNoAbility = filter == F_EVOLVES_FROM_OWN_INPLAY_NO_ABILITY;
    if (me.activeKnown && card_evolves_from(id, me.active.id, requireNoAbility))
      return true;
    for (const auto& b : me.bench)
      if (card_evolves_from(id, b.id, requireNoAbility)) return true;
    return false;
  }
  if (filter == F_EVOLVES_FROM_OWN_BENCH) {
    for (const auto& b : me.bench)
      if (card_evolves_from(id, b.id, false)) return true;
    return false;
  }
  return matches_filter(id, filter);
}

static bool matches_dynamic_inplay_filter(const GameState& st, const EffectFrame& fr,
                                          const InPlay& p, int filter) {
  if (filter == F_DAMAGED) return p.hp < p.maxHp;
  if (filter == F_DAMAGE_COUNTERS_EQ_6)
    return (p.maxHp - p.hp) / 10 == 6;
  if (filter == F_DAMAGED_TEAM_ROCKET)
    return p.hp < p.maxHp && is_team_rocket_card_id(p.id);
  if (filter == F_REMAINING_HP_LE_30)
    return p.hp < p.maxHp && p.hp <= 30;
  if (filter == F_DAMAGED_PSYCHIC_INPLAY)
    return p.hp < p.maxHp && pokemon_has_type(p.id, PSYCHIC);
  if (filter == F_DAMAGED_MEGA_EX_INPLAY) {
    const CardInfo* c = find_card(p.id);
    return p.hp < p.maxHp && c && c->megaEx;
  }
  if (filter == F_NS_COPYABLE_ATTACK)
    return ns_pokemon_has_copyable_attack(p.id);
  if (filter == F_HAS_DECK_EVOLUTION ||
      filter == F_HAS_DECK_EVOLUTION_NO_ABILITY) {
    return deck_has_evolution_for(st.players[fr.a], p,
                                  filter == F_HAS_DECK_EVOLUTION_NO_ABILITY);
  }
  if (filter == F_EVOLVED_INPLAY)
    return !p.preEvo.empty();
  if (filter == F_EVOLVED_PSYCHIC_INPLAY)
    return !p.preEvo.empty() && pokemon_has_type(p.id, PSYCHIC);
  if (filter == F_BASIC_WITH_STAGE2_IN_HAND) {
    const CardInfo* c = find_card(p.id);
    return c && c->basic && p.preEvo.empty() && p.noEvolveTurn != st.turn &&
           !p.appearThisTurn &&
           hand_has_stage2_for_basic(st.players[fr.a], p.id);
  }
  // An in-play Antique Fossil plays AS a 60-HP Basic {C} Pokemon, so it matches
  // Pokemon / Basic-Pokemon in-play targets even though its printed card type is
  // Item. (In hand/deck it is an Item and must NOT match, so this is in-play only.)
  if (is_antique_fossil(p.id) &&
      (filter == F_POKEMON || filter == F_BASIC_POKEMON))
    return true;
  return matches_filter(p.id, filter);
}

static bool eval_cond(const GameState& st, int cond) {
  int actor = st.effectStack.back().a;
  const EffectFrame& fr = st.effectStack.back();
  const Player& me = st.players[actor];
  const Player& opp = st.players[1 - actor];
  auto opp_active_type = [&](int type) {
    if (!opp.activeKnown) return false;
    const CardInfo* c = find_card(opp.active.id);
    return c && c->energyType == type;
  };
  if (cond == COND_OPP_HAS_BENCH) return !opp.bench.empty();
  if (cond == COND_LUNATONE_ON_BENCH) {
    for (const auto& b : me.bench)
      if (b.id == 675) return true;
    return false;
  }
  if (cond == COND_UXIE_AZELF_ON_BENCH) {
    bool hasUxie = false;
    bool hasAzelf = false;
    for (const auto& b : me.bench) {
      if (b.id == 215) hasUxie = true;
      if (b.id == 217) hasAzelf = true;
    }
    return hasUxie && hasAzelf;
  }
  if (cond == COND_OPP_IS_EX) {
    if (!opp.activeKnown) return false;
    const CardInfo* c = find_card(opp.active.id);
    return c && (c->ex || c->megaEx);
  }
  if (cond == COND_STADIUM_IN_PLAY) {
    if (fr.attackId == 1032 && st.stadiumOwner >= 0)
      return st.stadiumOwner == actor;  // Dhelmise: "you have a Stadium in play"
    return !st.stadium.empty();
  }
  if (cond == COND_OPP_IS_EVOLUTION) {
    if (!opp.activeKnown) return false;
    const CardInfo* c = find_card(opp.active.id);
    return c && (c->stage1 || c->stage2);
  }
  if (cond == COND_OPP_ACTIVE_DAMAGED)
    return opp.activeKnown && opp.active.hp < opp.active.maxHp;
  if (cond == COND_OPP_POISONED) return opp.poisoned;
  if (cond == COND_OPP_BURNED) return opp.burned;
  if (cond == COND_SELF_HAS_TOOL)
    return me.activeKnown && !me.active.tools.empty();
  if (cond == COND_SELF_HAS_SPECIAL_ENERGY && me.activeKnown) {
    for (int id : me.active.energyCardIds) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == SPECIAL_ENERGY) return true;
    }
    return false;
  }
  if (cond == COND_SELF_HAS_TEAM_ROCKET_ENERGY && me.activeKnown) {
    for (int id : me.active.energyCardIds) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == SPECIAL_ENERGY &&
          name_contains(c->name, "Team Rocket"))
        return true;
    }
    return false;
  }
  if (cond == COND_OPP_STADIUM_IN_PLAY)
    return !st.stadium.empty() && st.stadiumOwner == 1 - actor;
  if (cond == COND_OPP_PRIZES_LE_3) return opp.prizeCount <= 3;
  if (cond == COND_OWN_HAND_GE_10) return me.handCount >= 10;
  if (cond == COND_ALL_OWN_INPLAY_TEAM_ROCKET) {
    auto teamRocket = [](const InPlay& p) {
      const CardInfo* c = find_card(p.id);
      return c && name_contains(c->name, "Team Rocket");
    };
    if (!me.activeKnown || !teamRocket(me.active)) return false;
    for (const auto& b : me.bench)
      if (!teamRocket(b)) return false;
    return true;
  }
  if (cond == COND_SELF_POISONED) return me.poisoned;
  if (cond == COND_OPP_IS_STAGE2) {
    if (!opp.activeKnown) return false;
    const CardInfo* c = find_card(opp.active.id);
    return c && c->stage2;
  }
  if (cond == COND_OWN_HAS_TERA) {
    auto tera = [](const InPlay& p) {
      const CardInfo* c = find_card(p.id);
      return c && c->tera;
    };
    if (me.activeKnown && tera(me.active)) return true;
    for (const auto& b : me.bench)
      if (tera(b)) return true;
    return false;
  }
  if (cond == COND_OWN_PRIZES_GT_OPP) return me.prizeCount > opp.prizeCount;
  if (cond == COND_ILLUMISE_ON_BENCH) {
    for (const auto& b : me.bench) {
      const CardInfo* c = find_card(b.id);
      if (c && name_contains(c->name, "Illumise")) return true;
    }
    return false;
  }
  if (cond == COND_SELF_EVOLVED_FROM_GIMMIGHOUL_THIS_TURN) {
    if (!me.activeKnown || !me.active.appearThisTurn) return false;
    for (int id : me.active.preEvo) {
      const CardInfo* c = find_card(id);
      if (c && name_contains(c->name, "Gimmighoul")) return true;
    }
    return false;
  }
  if (cond == COND_SELF_IS_TEAM_ROCKET) {
    if (!me.activeKnown) return false;
    const CardInfo* c = find_card(me.active.id);
    return c && name_contains(c->name, "Team Rocket");
  }
  if (cond == COND_LAST_EFFECT_POS) return st.lastEffectCount > 0;
  if (cond == COND_LAST_ATTACK_DAMAGE_POS) return st.lastAttackDamage > 0;
  if (cond == COND_CHOSE_ACTIVE_AND_EFFECT)
    return st.lastEffectCount > 0 &&
           std::find(st.effectStack.back().scratch.begin(),
                     st.effectStack.back().scratch.end(), -1) !=
               st.effectStack.back().scratch.end();
  if (cond == COND_PLAYED_TEAM_ROCKET_SUPPORTER)
    return st.teamRocketSupporterPlayed;
  if (cond == COND_PLAYED_ANCIENT_SUPPORTER)
    return st.ancientSupporterPlayed;
  if (cond == COND_OWN_KO_LAST_TURN)
    return st.lastKoTurn[actor] == st.turn - 1;
  if (cond == COND_OWN_ATTACK_DAMAGE_KO_LAST_TURN)
    return st.lastAttackDamageKoTurn[actor] == st.turn - 1;
  if (cond == COND_OWN_ETHAN_KO_LAST_TURN)
    return st.lastEthanKoTurn[actor] == st.turn - 1;
  if (cond == COND_OWN_HOP_ATTACK_DAMAGE_KO_LAST_TURN)
    return st.lastHopAttackDamageKoTurn[actor] == st.turn - 1;
  if (cond == COND_OPP_ACTIVE_PSYCHIC) return opp_active_type(PSYCHIC);
  if (cond == COND_OPP_ACTIVE_DRAGON) return opp_active_type(DRAGON);
  if (cond == COND_OPP_ACTIVE_DARK) return opp_active_type(DARKNESS);
  if (cond == COND_OPP_ACTIVE_TERA) {
    if (!opp.activeKnown) return false;
    const CardInfo* c = find_card(opp.active.id);
    return c && c->tera;
  }
  if (cond == COND_SELF_HAS_FIGHTING_ENERGY) {
    const InPlay* self = effect_self(st, fr);
    if (!self) return false;
    for (int e : self->energies)
      if (e == FIGHTING) return true;
    return false;
  }
  if (cond == COND_OPP_HAS_STATUS)
    return opp.poisoned || opp.burned || opp.asleep || opp.paralyzed ||
           opp.confused;
  if (cond == COND_OPP_PRIZES_3_OR_4)
    return opp.prizeCount == 3 || opp.prizeCount == 4;
  if (cond == COND_SELF_DAMAGED) {
    const InPlay* self = effect_self(st, fr);
    return self && self->hp < self->maxHp;
  }
  if (cond == COND_SELF_UNDAMAGED) {
    const InPlay* self = effect_self(st, fr);
    return self && self->hp == self->maxHp;
  }
  if (cond == COND_ACTIVE_ENERGY_COUNTS_EQUAL)
    return me.activeKnown && opp.activeKnown &&
           me.active.energies.size() == opp.active.energies.size();
  if (cond == COND_OWN_BENCH_GE_5)
    return me.bench.size() >= 5;
  if (cond == COND_OPP_HAND_LE_3)
    return opp.handCount <= 3;
  if (cond == COND_HAND_COUNTS_EQUAL)
    return me.handCount == opp.handCount;
  if (cond == COND_OPP_ACTIVE_FIGHTING_RESISTANCE && opp.activeKnown) {
    const CardInfo* c = find_card(opp.active.id);
    return c && c->resistance == FIGHTING;
  }
  if (cond == COND_SELF_BURNED_OR_POISONED)
    return me.burned || me.poisoned;
  if (cond == COND_OWN_BENCH_DAMAGED) {
    for (const auto& b : me.bench)
      if (b.hp < b.maxHp) return true;
    return false;
  }
  if (cond == COND_OWN_PRIZES_EQ_6)
    return me.prizeCount == 6;
  if (cond == COND_OPP_PRIZES_LE_2)
    return opp.prizeCount <= 2;
  if (cond == COND_LAST_DITCH_UNUSED)
    return st.abilityGroupUsedTurn[actor][3] != st.turn;
  if (cond == COND_SELF_HP_LE_50) {
    const InPlay* self = effect_self(st, fr);
    return self && self->hp <= 50;
  }
  if (cond == COND_SELF_HP_LE_0) {
    const InPlay* self = effect_self(st, fr);
    return self && self->hp <= 0;
  }
  if (cond == COND_OWN_HAND_EQ_7)
    return me.handCount == 7;
  if (cond == COND_DURANT_ON_BENCH) {
    for (const auto& b : me.bench) {
      const CardInfo* c = find_card(b.id);
      if (c && name_contains(c->name, "Durant")) return true;
    }
    return false;
  }
  if (cond == COND_BENCHED_PANCHAM_DAMAGED) {
    for (const auto& b : me.bench) {
      const CardInfo* c = find_card(b.id);
      if (c && name_contains(c->name, "Pancham") && b.hp < b.maxHp)
        return true;
    }
    return false;
  }
  if (cond == COND_OPP_ACTIVE_BASIC) {
    if (!opp.activeKnown) return false;
    const CardInfo* c = find_card(opp.active.id);
    return c && c->cardType == POKEMON && !c->stage1 && !c->stage2;
  }
  if (cond == COND_OPP_ACTIVE_DAMAGE_COUNTERS_EQ_6)
    return opp.activeKnown && (opp.active.maxHp - opp.active.hp) / 10 == 6;
  if (cond == COND_OWN_BENCH_STAGE2_DARK) {
    for (const auto& b : me.bench) {
      const CardInfo* c = find_card(b.id);
      if (c && c->stage2 && c->energyType == DARKNESS) return true;
    }
    return false;
  }
  if (cond == COND_SELF_EVOLVED_FROM_MAGNETON_THIS_TURN) {
    const InPlay* self = effect_self(st, fr);
    if (!self || !self->appearThisTurn) return false;
    for (int id : self->preEvo) {
      const CardInfo* c = find_card(id);
      if (c && name_contains(c->name, "Magneton")) return true;
    }
    return false;
  }
  if (cond == COND_OWN_OPP_TYPE_OVERLAP) {
    std::vector<const InPlay*> own, their;
    if (me.activeKnown) own.push_back(&me.active);
    for (const auto& b : me.bench) own.push_back(&b);
    if (opp.activeKnown) their.push_back(&opp.active);
    for (const auto& b : opp.bench) their.push_back(&b);
    for (const InPlay* a : own)
      for (const InPlay* b : their)
        for (int t = COLORLESS; t <= DRAGON; ++t)
          if (pokemon_has_type(a->id, t) && pokemon_has_type(b->id, t))
            return true;
    return false;
  }
  if (cond == COND_SELF_MOVED_FROM_BENCH_THIS_TURN) {
    const InPlay* self = effect_self(st, fr);
    return self && self->movedToActiveThisTurn;
  }
  if (cond == COND_OTHER_ANCIENT_ATTACKED_LAST_TURN) {
    const InPlay* self = effect_self(st, fr);
    return self && st.lastAncientAttackTurn[actor] == st.turn - 2 &&
           (st.lastAncientAttackCard[actor] != self->id ||
            (st.lastAncientAttackSerial[actor] != 0 && self->serial != 0 &&
             st.lastAncientAttackSerial[actor] != self->serial));
  }
  if (cond == COND_BELDUM_METANG_ON_BENCH) {
    bool beldum = false, metang = false;
    for (const auto& b : me.bench) {
      const CardInfo* c = find_card(b.id);
      beldum = beldum || (c && name_contains(c->name, "Beldum"));
      metang = metang || (c && name_contains(c->name, "Metang"));
    }
    return beldum && metang;
  }
  if (cond == COND_OWN_BENCH_HAS_TERA) {
    for (const auto& b : me.bench) {
      const CardInfo* c = find_card(b.id);
      if (c && c->tera) return true;
    }
    return false;
  }
  if (cond == COND_SELF_EVOLVED_FROM_MISTY_STARYU_THIS_TURN) {
    const InPlay* self = effect_self(st, fr);
    if (!self || !self->appearThisTurn) return false;
    for (int id : self->preEvo) {
      const CardInfo* c = find_card(id);
      if (c && name_contains(c->name, "Misty") && name_contains(c->name, "Staryu"))
        return true;
    }
    return false;
  }
  if (cond == COND_OWN_BENCH_HAS_NIDOKING) {
    for (const auto& b : me.bench) {
      const CardInfo* c = find_card(b.id);
      if (c && name_contains(c->name, "Nidoking")) return true;
    }
    return false;
  }
  if (cond == COND_SELF_HEALED_THIS_TURN) {
    const InPlay* self = effect_self(st, fr);
    return self && self->healedThisTurn;
  }
  if (cond == COND_OWN_DECK_LE_3) return me.deckCount <= 3;
  if (cond == COND_OWN_DISCARD_BASIC_FIGHTING_GE_10) {
    int n = 0;
    for (int id : me.discard) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == BASIC_ENERGY && c->energyType == FIGHTING)
        ++n;
    }
    return n >= 10;
  }
  if (cond == COND_OWN_DISCARD_ROSA_ENCOURAGEMENT)
    return discard_has_name(me, "Rosa's Encouragement");
  if (cond == COND_USED_SPIKY_ROLLING_LAST_TURN)
    return st.lastAttackTurn[actor] == st.turn - 2 &&
           st.lastAttackId[actor] == 1429;
  if (cond == COND_PLAYED_TARRAGON)
    return st.tarragonPlayed;
  if (cond == COND_SELF_HAS_LIGHTNING_ENERGY)
    return me.activeKnown &&
           std::find(me.active.energies.begin(), me.active.energies.end(),
                     LIGHTNING) != me.active.energies.end();
  if (cond == COND_SELF_IS_PSYCHIC) {
    const InPlay* self = effect_self(st, fr);
    return self && pokemon_has_type(self->id, PSYCHIC);
  }
  if (cond == COND_CAN_DRAW_UNTIL_6)
    return me.handCount < 6;
  if (cond == COND_OWN_HAS_BENCH) return !me.bench.empty();
  if (cond == COND_OPP_ACTIVE_HAS_ENERGY_AND_BENCH)
    return opp.activeKnown && !opp.active.energies.empty() && !opp.bench.empty();
  if (cond == COND_OWN_DECK_NONEMPTY) return me.deckCount > 0;
  return false;
}

static bool has_special_energy(const InPlay& p) {
  for (int id : p.energyCardIds) {
    const CardInfo* c = find_card(id);
    if (c && c->cardType == SPECIAL_ENERGY) return true;
  }
  return false;
}

static bool has_energy_type(const InPlay& p, int etype) {
  if ((etype == PSYCHIC || etype == DARKNESS) && is_team_rocket_card_id(p.id) &&
      std::find(p.energyCardIds.begin(), p.energyCardIds.end(), 15) !=
          p.energyCardIds.end())
    return true;
  for (int e : p.energies)
    if (e == etype || e == RAINBOW)
      return true;
  return false;
}

static bool has_tool(const InPlay& p, int toolId) {
  return std::find(p.tools.begin(), p.tools.end(), toolId) != p.tools.end();
}

static bool tool_effects_disabled(const GameState& st) {
  return cur_stadium(st) == 1246;  // Jamming Tower
}

static bool active_tool(const GameState& st, const InPlay& p, int toolId) {
  return !tool_effects_disabled(st) && has_tool(p, toolId);
}

static bool player_has_powerglass_source(const GameState& st, const Player& p) {
  if (p.activeKnown && active_tool(st, p.active, 1163))
    return true;
  for (const InPlay& b : p.bench)
    if (active_tool(st, b, 1163))
      return true;
  return false;
}

static void attach_tool_card(InPlay& p, int toolId, int order = -1) {
  p.tools.push_back(toolId);
  if (!p.toolOrders.empty() || order >= 0)
    p.toolOrders.push_back(order);
}

static void erase_tool_at(InPlay& p, int idx) {
  if (idx < 0 || idx >= static_cast<int>(p.tools.size())) return;
  p.tools.erase(p.tools.begin() + idx);
  if (!p.toolOrders.empty()) {
    if (idx < static_cast<int>(p.toolOrders.size()))
      p.toolOrders.erase(p.toolOrders.begin() + idx);
    else
      p.toolOrders.clear();
  }
}

static int pop_tool_card(InPlay& p) {
  if (p.tools.empty()) return 0;
  int id = p.tools.back();
  p.tools.pop_back();
  if (!p.toolOrders.empty()) {
    if (p.toolOrders.size() >= p.tools.size() + 1)
      p.toolOrders.pop_back();
    else
      p.toolOrders.clear();
  }
  return id;
}

static bool discard_tool(Player& owner, InPlay& p, int toolId) {
  auto it = std::find(p.tools.begin(), p.tools.end(), toolId);
  if (it == p.tools.end()) return false;
  int idx = static_cast<int>(std::distance(p.tools.begin(), it));
  owner.discard.push_back(*it);
  erase_tool_at(p, idx);
  return true;
}

static bool has_rule_box(const CardInfo* c) {
  return c && (c->ex || c->megaEx || c->tera);
}

static bool in_play_has(const Player& p, int cardId) {
  if (p.activeKnown && p.active.id == cardId) return true;
  for (const auto& b : p.bench)
    if (b.id == cardId) return true;
  return false;
}

static bool risky_ruins_bench_entry_applies(const GameState& st, int owner,
                                            int benchIdx) {
  if (cur_stadium(st) != 1260 || owner != st.yourIndex) return false;
  const Player& p = st.players[owner];
  if (benchIdx < 0 || benchIdx >= static_cast<int>(p.bench.size())) return false;
  const InPlay& pk = p.bench[benchIdx];
  const CardInfo* c = find_card(pk.id);
  return c && c->basic && c->energyType != DARKNESS;
}

static void apply_risky_ruins_bench_entry(GameState& st, int owner, int benchIdx) {
  if (!risky_ruins_bench_entry_applies(st, owner, benchIdx)) return;
  Player& p = st.players[owner];
  InPlay& pk = p.bench[benchIdx];
  pk.hp = std::max(0, pk.hp - 20);
}

static bool any_in_play_has(const GameState& st, int cardId) {
  return in_play_has(st.players[0], cardId) || in_play_has(st.players[1], cardId);
}

static bool all_n_team_in_play(const Player& p) {
  return in_play_has(p, 258) && in_play_has(p, 293) && in_play_has(p, 864) &&
         in_play_has(p, 296) && in_play_has(p, 303) && in_play_has(p, 906);
}

static bool ability_suppressed_by_self_ko_lock(const GameState& st, int id) {
  if (id != 132 && id != 133) return false;  // Cursed Blast abilities.
  return any_in_play_has(st, 858) || any_in_play_has(st, 859);
}

static bool ability_suppressed(const GameState& st, int ownerSide,
                               const InPlay& pk, bool isActive) {
  const CardInfo* c = find_card(pk.id);
  if (!c || !c->hasAbility) return false;

  if (cur_stadium(st) == 1256 && c->energyType == COLORLESS)
    return true;  // Team Rocket's Watchtower: Colorless Pokemon.

  for (int side = 0; side < 2; ++side) {
    const Player& p = st.players[side];
    if (p.activeKnown && p.active.id == 37 && has_rule_box(c) &&
        !is_future_card(pk.id))
      return true;  // Iron Thorns ex: rule-box Pokemon except Future.

    if (ownerSide == 1 - side && isActive &&
        p.activeKnown && p.active.id == 56 && pk.id != 56)
      return true;  // Flutter Mane: opponent's Active except Midnight Fluttering.

    if (c->stage2) {
      for (const auto& b : p.bench)
        if (b.id == 225 && !isActive)
          return true;  // Gastrodon: Benched Stage 2 Pokemon.
    }
  }
  return false;
}

static bool pokemon_has_active_ability(const GameState& st, int ownerSide,
                                       const InPlay& pk, bool isActive) {
  const CardInfo* c = find_card(pk.id);
  return c && c->hasAbility && !ability_suppressed(st, ownerSide, pk, isActive);
}

static bool owner_has_active_ability_card(const GameState& st, int ownerSide,
                                          int cardId) {
  if (ownerSide < 0 || ownerSide >= 2) return false;
  if (cardId == 710 && st.pendingMeganiumAura[ownerSide]) return true;
  const Player& p = st.players[ownerSide];
  if (p.activeKnown && p.active.id == cardId &&
      pokemon_has_active_ability(st, ownerSide, p.active, true))
    return true;
  for (const auto& b : p.bench)
    if (b.id == cardId && pokemon_has_active_ability(st, ownerSide, b, false))
      return true;
  return false;
}
static int active_ability_count_id(const GameState& st, int ownerSide,
                                   int cardId) {
  if (ownerSide < 0 || ownerSide >= 2) return 0;
  const Player& p = st.players[ownerSide];
  int count = 0;
  if (p.activeKnown && p.active.id == cardId &&
      pokemon_has_active_ability(st, ownerSide, p.active, true))
    ++count;
  for (const auto& b : p.bench)
    if (b.id == cardId &&
        pokemon_has_active_ability(st, ownerSide, b, false))
      ++count;
  return count;
}


static bool relicanth_memory_available(const GameState& st, int ownerSide) {
  const Player& p = st.players[ownerSide];
  if (p.activeKnown && p.active.id == 57 &&
      !ability_suppressed(st, ownerSide, p.active, true))
    return true;
  for (const auto& b : p.bench)
    if (b.id == 57 && !ability_suppressed(st, ownerSide, b, false))
      return true;
  return false;
}

static const AttackInfo* inherited_attack_info(const GameState& st, int ownerSide,
                                               const InPlay& attacker,
                                               int attackId) {
  if (!relicanth_memory_available(st, ownerSide) || attacker.preEvo.empty())
    return nullptr;
  for (int preId : attacker.preEvo) {
    const CardInfo* c = find_card(preId);
    if (!c) continue;
    for (int k = 0; k < c->n_attacks; ++k)
      if (c->attacks[k].id == attackId) return &c->attacks[k];
  }
  return nullptr;
}

static const AttackInfo* n_zoroark_benched_attack_info(const GameState& st,
                                                       int ownerSide,
                                                       int attackId) {
  if (ownerSide < 0 || ownerSide > 1) return nullptr;
  const Player& p = st.players[ownerSide];
  if (!p.activeKnown || p.active.id != 293) return nullptr;
  for (const auto& b : p.bench) {
    const CardInfo* c = find_card(b.id);
    if (!is_ns_pokemon(c)) continue;
    for (int k = 0; k < c->n_attacks; ++k)
      if (c->attacks[k].id == attackId) return &c->attacks[k];
  }
  return nullptr;
}

static bool card_name_contains(int cardId, const char* needle) {
  const CardInfo* c = find_card(cardId);
  return c && name_contains(c->name, needle);
}

static int owner_of_inplay(const GameState& st, const InPlay& pk) {
  for (int side = 0; side < 2; ++side) {
    const Player& p = st.players[side];
    if (p.activeKnown && &p.active == &pk) return side;
    for (const auto& b : p.bench)
      if (&b == &pk) return side;
  }
  return -1;
}

static int tool_capacity(const GameState& st, int ownerSide,
                         const InPlay& pk) {
  if (card_name_contains(pk.id, "Rotom") &&
      owner_has_active_ability_card(st, ownerSide, 806))
    return 2;  // Rotom ex: Multi Adapter
  return 1;
}

static void enforce_rotom_tool_limits(GameState& st) {
  for (int side = 0; side < 2; ++side) {
    Player& p = st.players[side];
    auto trim = [&st, side, &p](InPlay& k) {
      int cap = tool_capacity(st, side, k);
      while (static_cast<int>(k.tools.size()) > cap) {
        int damage = k.maxHp - k.hp;
        p.discard.push_back(pop_tool_card(k));
        k.maxHp = effective_max_hp(k.id, k.tools, st, k.energyCardIds,
                                   k.energies, side);
        k.hp = std::max(0, k.maxHp - damage);
      }
    };
    if (p.activeKnown) trim(p.active);
    for (auto& b : p.bench) trim(b);
  }
}

static int effective_bench_max(const GameState& st, int side) {
  return cur_stadium(st) == 1250 && player_has_tera(st.players[side]) ? 8 : 5;
}

static void enforce_area_zero_bench_limits(GameState& st) {
  for (int side = 0; side < 2; ++side) {
    Player& p = st.players[side];
    p.benchMax = effective_bench_max(st, side);
    while (static_cast<int>(p.bench.size()) > p.benchMax) {
      ko_bench(p, static_cast<int>(p.bench.size()) - 1);
    }
  }
}

static bool bench_has(const Player& p, int cardId) {
  for (const auto& b : p.bench)
    if (b.id == cardId) return true;
  return false;
}

static bool in_play_has_dark_mega_ex(const Player& p) {
  auto ok = [](const InPlay& pk) {
    const CardInfo* c = find_card(pk.id);
    return c && c->energyType == DARKNESS && c->megaEx;
  };
  if (p.activeKnown && ok(p.active)) return true;
  for (const auto& b : p.bench)
    if (ok(b)) return true;
  return false;
}

static bool in_play_has_ex_or_v(const Player& p) {
  auto is_ex_or_v = [](const InPlay& pk) {
    const CardInfo* c = find_card(pk.id);
    return c && (c->ex || c->megaEx);
  };
  if (p.activeKnown && is_ex_or_v(p.active)) return true;
  for (const auto& b : p.bench)
    if (is_ex_or_v(b)) return true;
  return false;
}

static int in_play_name_count(const Player& p, const char* sub) {
  int n = 0;
  if (p.activeKnown) {
    const CardInfo* c = find_card(p.active.id);
    if (c && name_contains(c->name, sub)) ++n;
  }
  for (const auto& b : p.bench) {
    const CardInfo* c = find_card(b.id);
    if (c && name_contains(c->name, sub)) ++n;
  }
  return n;
}

static bool discard_has_name(const Player& p, const char* sub) {
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if (c && name_contains(c->name, sub)) return true;
  }
  return false;
}

static int discard_name_count(const Player& p, const char* sub) {
  int n = 0;
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if (c && name_contains(c->name, sub)) ++n;
  }
  return n;
}

static void record_prize_taken(GameState& st, int player) {
  if (st.prizeTakenTurn[player] != st.turn) {
    st.prizeTakenTurn[player] = st.turn;
    st.prizeTakenCount[player] = 0;
  }
  st.prizeTakenCount[player] += 1;
}

static bool in_play_has_stage2(const Player& p) {
  if (p.activeKnown) {
    const CardInfo* c = find_card(p.active.id);
    if (c && c->stage2) return true;
  }
  for (const auto& b : p.bench) {
    const CardInfo* c = find_card(b.id);
    if (c && c->stage2) return true;
  }
  return false;
}

static bool pokemon_played_as_basic_from_hand(const GameState& st, int side,
                                              int cardId,
                                              const CardInfo* c = nullptr) {
  if (!c) c = find_card(cardId);
  if (!c || c->cardType != POKEMON) return false;
  if (c->basic) return true;
  return cardId == 167 && in_play_has_stage2(st.players[1 - side]);
}

static bool in_play_has_with_tool(const Player& p, int cardId) {
  if (p.activeKnown && p.active.id == cardId && !p.active.tools.empty())
    return true;
  for (const auto& b : p.bench)
    if (b.id == cardId && !b.tools.empty()) return true;
  return false;
}

static bool opponent_active_is(const GameState& st, int player, int cardId) {
  const Player& opp = st.players[1 - player];
  return opp.activeKnown && opp.active.id == cardId;
}

static bool item_locked(const GameState& st, int player) {
  return st.noItemTurn[player] == st.turn ||
         opponent_active_is(st, player, 290) ||   // Tyranitar: no Items
         opponent_active_is(st, player, 598);     // Jellicent ex: no Items/Tools
}

static bool supporter_locked(const GameState& st, int player) {
  return st.noSupporterTurn[player] == st.turn;
}

static bool tool_locked(const GameState& st, int player) {
  return opponent_active_is(st, player, 598);     // Jellicent ex
}

static bool stadium_locked(const GameState& st, int player) {
  return st.noStadiumTurn[player] == st.turn;
}

static bool evolution_locked(const GameState& st, int player) {
  return st.noEvolveTurn[player] == st.turn;
}

static bool ace_spec_locked(const GameState& st, int player) {
  return in_play_has_with_tool(st.players[1 - player], 142);  // Genesect
}

static bool pokemon_from_hand_blocked(const GameState& st, int player,
                                      const CardInfo* ci) {
  if (!ci || ci->cardType != POKEMON) return false;
  if (!opponent_active_is(st, player, 449)) return false;  // Team Rocket's Arbok
  return ci->hasAbility && !name_contains(ci->name, "Team Rocket");
}

static bool passive_free_retreat(const GameState& st, int player,
                                 const InPlay& pk) {
  const Player& owner = st.players[player];
  const CardInfo* ci = find_card(pk.id);
  if (cur_stadium(st) == 1253 && is_ns_pokemon(ci))
    return true;  // N's Castle: N's Pokemon
  if (active_tool(st, pk, 1157) && pk.hp <= 30)
    return true;  // Rescue Board: attached Pokemon at 30 HP or less
  if (in_play_has(owner, 184) && ci && ci->basic)
    return true;  // Latias ex: your Basic Pokemon
  if (in_play_has(owner, 170) && has_energy_type(pk, METAL))
    return true;  // Archaludon: your Pokemon with {M} Energy
  return false;
}

static bool damage_prevented(const GameState& st, const InPlay& defender,
                             int turn, const CardInfo* attacker,
                             const InPlay& attackerInPlay, int attackerSide,
                             int damage) {
  if (defender.preventDmgTurn != turn) return false;
  switch (defender.preventDmgCond) {
    case DPC_ALL:
      return true;
    case DPC_ATTACKER_BASIC:
      return attacker && attacker->basic;
    case DPC_ATTACKER_BASIC_NON_COLORLESS:
      return attacker && attacker->basic && attacker->energyType != COLORLESS;
    case DPC_ATTACKER_EX:
      return attacker && (attacker->ex || attacker->megaEx);
    case DPC_ATTACKER_HAS_ABILITY:
      return pokemon_has_active_ability(st, attackerSide, attackerInPlay, true);
    case DPC_DAMAGE_LE:
      return damage <= defender.preventDmgValue;
    case DPC_DAMAGE_GE:
      return damage >= defender.preventDmgValue;
    case DPC_ATTACKER_HAS_SPECIAL_ENERGY:
      return has_special_energy(attackerInPlay);
    default:
      return false;
  }
}

static bool prevention_condition_matches(const GameState& st, int cond,
                                         int value, const CardInfo* attacker,
                                         const InPlay& attackerInPlay,
                                         int attackerSide, int amount) {
  switch (cond) {
    case DPC_ALL:
      return true;
    case DPC_ATTACKER_BASIC:
      return attacker && attacker->basic;
    case DPC_ATTACKER_BASIC_NON_COLORLESS:
      return attacker && attacker->basic && attacker->energyType != COLORLESS;
    case DPC_ATTACKER_EX:
      return attacker && (attacker->ex || attacker->megaEx);
    case DPC_ATTACKER_HAS_ABILITY:
      return pokemon_has_active_ability(st, attackerSide, attackerInPlay, true);
    case DPC_DAMAGE_LE:
      return amount <= value;
    case DPC_DAMAGE_GE:
      return amount >= value;
    case DPC_ATTACKER_HAS_SPECIAL_ENERGY:
      return has_special_energy(attackerInPlay);
    default:
      return false;
  }
}

static bool attack_effects_prevented(const GameState& st, int targetSide,
                                     const InPlay& target) {
  if (st.effectStack.empty()) return false;
  const EffectFrame& fr = st.effectStack.back();
  if (targetSide == fr.a) return false;
  const Player& targetOwner = st.players[targetSide];
  const bool targetIsActive =
      targetOwner.activeKnown && &target == &targetOwner.active;
  const bool targetAbilityActive =
      !ability_suppressed(st, targetSide, target, targetIsActive);
  const CardInfo* src = fr.sourceCardId > 0 ? find_card(fr.sourceCardId) : nullptr;
  if (src && (src->cardType == ITEM || src->cardType == SUPPORTER) &&
      (target.id == 424 || target.id == 994) && targetAbilityActive)
    return true;  // Cetitan ex / Fraxure: opponent Item or Supporter effects.
  if (src && src->cardType == POKEMON && fr.attackId == 0 &&
      target.id == 1040 && targetAbilityActive)
    return true;  // Mega Clefable ex: opponent Pokemon Ability effects.
  if (fr.attackId > 0) {
    if (target.preventEffectsTurn == st.turn) {
      const Player& attackerOwner = st.players[fr.a];
      const InPlay* attackerInPlay = attackerOwner.activeKnown
                                         ? &attackerOwner.active : nullptr;
      const CardInfo* attacker = attackerInPlay ? find_card(attackerInPlay->id)
                                                : nullptr;
      if ((target.preventEffectsCond == DPC_ALL && !attackerInPlay) ||
          (attackerInPlay &&
           prevention_condition_matches(st, target.preventEffectsCond,
                                        target.preventEffectsValue, attacker,
                                        *attackerInPlay, fr.a, 0)))
        return true;
    }
    if ((target.id == 203 || target.id == 835 || target.id == 1136) &&
        targetAbilityActive)
      return true;  // Unaware / Emperor's Stance / Protective Cover
    if (!targetIsActive && (target.id == 28 || target.id == 362) &&
        targetAbilityActive)
      return true;  // Storehouse Hideaway / So Submerged: damage and effects.
    if (std::find(target.energyCardIds.begin(), target.energyCardIds.end(), 11) !=
        target.energyCardIds.end())
      return true;  // Mist Energy
    if (std::find(target.energyCardIds.begin(), target.energyCardIds.end(), 20) !=
            target.energyCardIds.end() &&
        pokemon_has_type(target.id, FIGHTING))
      return true;  // Rock Fighting Energy
    const CardInfo* tc = find_card(target.id);
    if (owner_has_active_ability_card(st, targetSide, 414) && tc && tc->basic &&
        name_contains(tc->name, "Team Rocket"))
      return true;  // Team Rocket's Articuno: Basic Team Rocket's Pokemon
  } else if (src && target.id == 1151) {
    if (src->cardType == SUPPORTER && targetAbilityActive)
      return true;  // Antique Sail Fossil: opponent Supporter effects
  }
  return false;
}

static bool source_is_opposing_pokemon_effect(const GameState& st, int targetSide) {
  if (st.effectStack.empty()) return false;
  const EffectFrame& fr = st.effectStack.back();
  if (targetSide == fr.a) return false;
  if (fr.attackId > 0) return true;
  if (fr.sourceCardId > 0) {
    const CardInfo* src = find_card(fr.sourceCardId);
    return src && src->cardType == POKEMON;
  }
  return false;
}

static bool bench_damage_counters_prevented(const GameState& st, int targetSide,
                                            bool isBench) {
  return isBench && cur_stadium(st) == 1264 &&
         source_is_opposing_pokemon_effect(st, targetSide);
}

static bool festival_status_immunity(const GameState& st, int side) {
  const Player& p = st.players[side];
  return cur_stadium(st) == 1245 && p.activeKnown && !p.active.energies.empty();
}

static void enforce_festival_status_recovery(GameState& st) {
  if (cur_stadium(st) != 1245) return;
  for (int side = 0; side < 2; ++side)
    if (festival_status_immunity(st, side))
      clear_status(st.players[side]);
}

static void apply_damaged_by_attack_reactive(GameState& st, InPlay& defender,
                                             int attackerSide, int damageDone,
                                             bool defenderBench = false,
                                             int beforeHp = -1) {
  (void)defenderBench;
  defender.damagedByAttackTurn = st.turn;
  defender.damagedByAttackSide = attackerSide;
  defender.damagedByAttackAmount = damageDone;
  defender.damagedByAttackBeforeHp = beforeHp;
  Player& attacker = st.players[attackerSide];
  if (!attacker.activeKnown) return;
  if (defender.damagedByAttackCountersTurn != st.turn &&
      defender.damagedByAttackStatus < 0 &&
      defender.damagedByAttackEqualCountersTurn != st.turn)
    return;
  if (defender.damagedByAttackCountersTurn == st.turn &&
      defender.damagedByAttackCounters > 0) {
    attacker.active.hp -= 10 * defender.damagedByAttackCounters;
    if (attacker.active.hp < 0) attacker.active.hp = 0;
  }
  if (defender.damagedByAttackEqualCountersTurn == st.turn && damageDone > 0) {
    attacker.active.hp -= damageDone;
    if (attacker.active.hp < 0) attacker.active.hp = 0;
  }
  if (defender.damagedByAttackStatus >= 0 &&
      defender.damagedByAttackCountersTurn == st.turn &&
      !festival_status_immunity(st, attackerSide))
    apply_condition(attacker, defender.damagedByAttackStatus);
}

static void apply_energy_attach_reactive(GameState& st, InPlay& target,
                                         bool fromHand) {
  if (target.energyAttachCountersTurn == st.turn && target.energyAttachCounters > 0 &&
      (!target.energyAttachCountersFromHandOnly || fromHand)) {
    target.hp -= 10 * target.energyAttachCounters;
    if (target.hp < 0) target.hp = 0;
  }
  enforce_festival_status_recovery(st);
}

static int attached_energy_unit_type(int cardId, const InPlay& target) {
  const CardInfo* ec = find_card(cardId);
  if (cardId == 16) {
    const CardInfo* pc = find_card(target.id);
    return (pc && pc->basic) ? RAINBOW : COLORLESS;
  }
  return ec ? ec->energyType : COLORLESS;
}

static bool passive_damage_prevented(const GameState& st,
                                     const InPlay& attackerInPlay, int defenderSide,
                                     const InPlay& defender, bool defenderBench,
                                     int damage) {
  const Player& owner = st.players[defenderSide];
  const CardInfo* attacker = find_card(attackerInPlay.id);
  const CardInfo* defenderCard = find_card(defender.id);
  const bool defenderIsActive =
      owner.activeKnown && &defender == &owner.active;
  const bool defenderAbilityActive =
      !ability_suppressed(st, defenderSide, defender, defenderIsActive);


  // Self/active-style prevention abilities.
  if (defender.id == 83 && defenderAbilityActive)  // Farigiraf ex
    return attacker && attacker->basic && (attacker->ex || attacker->megaEx);
  if (defender.id == 117 && defenderAbilityActive)  // Cornerstone Mask Ogerpon ex
    return pokemon_has_active_ability(st, 1 - defenderSide, attackerInPlay, true);
  if (defender.id == 158 && defenderAbilityActive)  // Drednaw
    return damage >= 200;
  if (defender.id == 207 && defenderAbilityActive)  // Milotic ex
    return attacker && attacker->tera;
  if ((defender.id == 330 || defender.id == 345) && defenderAbilityActive)
    return attacker && (attacker->ex || attacker->megaEx);
  if (defender.id == 504 && defenderAbilityActive)
    return has_special_energy(attackerInPlay);

  // Bench protection abilities. These prevent attack damage to Benched Pokemon;
  // they intentionally do not block OP_PLACE_DAMAGE damage-counter effects.
  if (defenderBench && defenderCard && defenderCard->tera)
    return true;
  if (defenderBench && (defender.id == 28 || defender.id == 362 ||
                        defender.id == 1138) && defenderAbilityActive)
    return true;
  if (defenderBench && owner_has_active_ability_card(st, defenderSide, 74))
    return true;
  if (defenderBench && owner_has_active_ability_card(st, defenderSide, 343) &&
      !has_rule_box(defenderCard))
    return true;  // Shaymin: Benched Pokemon without a Rule Box

  return false;
}

static int passive_damage_reduction(GameState& st,
                                    const InPlay& attackerInPlay, int defenderSide,
                                    InPlay& defender, int damage) {
  Player& owner = st.players[defenderSide];
  const CardInfo* attacker = find_card(attackerInPlay.id);
  const CardInfo* defenderCard = find_card(defender.id);
  int reduce = 0;
  bool defenderIsActive = owner.activeKnown && (&defender == &owner.active);

  if ((defender.id == 383 || defender.id == 631 || defender.id == 766) &&
      !ability_suppressed(st, defenderSide, defender, defenderIsActive))
    reduce += 30;  // Mud Coat / Bouffer / Diamond Coat
  if (defender.id == 799 &&
      !ability_suppressed(st, defenderSide, defender, defenderIsActive) && attacker &&
      (attacker->energyType == FIRE || attacker->energyType == WATER))
    reduce += 30;  // Dewgong: {R}/{W} attackers
  if (active_tool(st, defender, 1177) &&
      pokemon_has_active_ability(st, 1 - defenderSide, attackerInPlay, true))
    reduce += 30;  // Sacred Charm
  if (active_tool(st, defender, 1179) && defenderCard && defenderCard->energyType == DRAGON &&
      attacker && (attacker->energyType == GRASS || attacker->energyType == FIRE ||
                   attacker->energyType == WATER || attacker->energyType == LIGHTNING))
    reduce += 50;  // Thick Scale
  if (damage > 0 && active_tool(st, defender, 1164) && attacker &&
      attacker->energyType == PSYCHIC && discard_tool(owner, defender, 1164))
    reduce += 60;  // Payapa Berry
  if (damage > 0 && active_tool(st, defender, 1170) && attacker &&
      attacker->energyType == DRAGON && discard_tool(owner, defender, 1170))
    reduce += 60;  // Haban Berry
  if (owner_has_active_ability_card(st, defenderSide, 1033) &&
      has_energy_type(defender, WATER))
    reduce += 50;  // Aurorus: your Pokemon with {W} Energy
  if (has_energy_type(defender, METAL))
    reduce += 20 * active_ability_count_id(st, defenderSide, 623);
  if (bench_has(owner, 637) && owner_has_active_ability_card(st, defenderSide, 637) &&
      defenderCard &&
      name_contains(defenderCard->name, "Steven"))
    reduce += 30;  // Steven's Carbink on Bench: Steven's Pokemon
  if (cur_stadium(st) == 1258 && defenderCard &&
      name_contains(defenderCard->name, "Steven"))
    reduce += 30;  // Granite Cave: Steven's Pokemon
  if (st.teamReduceTurn[defenderSide] == st.turn && defenderCard &&
      defenderCard->energyType == st.teamReduceType[defenderSide])
    reduce += st.teamReduceAmount[defenderSide];
  if (owner_has_active_ability_card(st, defenderSide, 175) &&
      in_play_name_count(owner, "Bouffalant") >= 2 &&
      defenderCard && defenderCard->basic && defenderCard->energyType == COLORLESS)
    reduce += 60;  // Curly Wall; does not stack, so apply once.

  return reduce;
}

static void apply_delayed_attack_damage_ko_replacement(GameState& st,
                                                       int defenderSide,
                                                       InPlay& defender,
                                                       bool isActive,
                                                       int attackerSide) {
  if (defender.hp > 0) return;
  if (defender.damagedByAttackTurn != st.turn ||
      defender.damagedByAttackSide != attackerSide)
    return;
  int beforeHp = defender.damagedByAttackBeforeHp;
  if (beforeHp <= 0) return;
  Player& owner = st.players[defenderSide];
  if (active_tool(st, defender, 1155) && beforeHp == defender.maxHp &&
      discard_tool(owner, defender, 1155)) {
    defender.hp = 10;  // Survival Brace
    return;
  }
  (void)isActive;
}

static void apply_delayed_tenacious_body(GameState& st, int defenderSide,
                                         InPlay& defender, bool isActive,
                                         int attackerSide) {
  if (defender.id != 886 || defender.hp > 0) return;
  if (defender.damagedByAttackTurn != st.turn ||
      defender.damagedByAttackSide != attackerSide)
    return;
  if (!ability_suppressed(st, defenderSide, defender, isActive) &&
      flip_heads(st))
    defender.hp = 10;  // Mega Hawlucha ex: Tenacious Body
}

static int passive_attack_damage_delta(const GameState& st, int attackerSide,
                                       const InPlay& attackerInPlay,
                                       const InPlay& defender, bool defenderActive,
                                       int damage) {
  if (damage <= 0) return 0;
  const Player& owner = st.players[attackerSide];
  const Player& defenderOwner = st.players[1 - attackerSide];
  const CardInfo* attacker = find_card(attackerInPlay.id);
  const CardInfo* defenderCard = find_card(defender.id);
  if (!attacker || !defenderCard) return 0;
  int delta = 0;
  const bool attackerAbilityActive =
      !ability_suppressed(st, attackerSide, attackerInPlay, true);

  if (attackerInPlay.id == 116 && attackerAbilityActive &&
      has_energy_type(attackerInPlay, DARKNESS))
    delta += 100;  // Okidogi: any {D} Energy attached
  if (owner_has_active_ability_card(st, attackerSide, 155) &&
      (defenderCard->stage1 || defenderCard->stage2))
    delta += 30;   // Carracosta: opponent Active Evolution Pokemon
  if (attacker->energyType == FIRE && (attacker->stage1 || attacker->stage2))
    delta += 10 * active_ability_count_id(st, attackerSide, 202);
  if (owner_has_active_ability_card(st, attackerSide, 304) && name_contains(attacker->name, "Hop"))
    delta += 30;   // Hop's Snorlax: your Hop's Pokemon
  if (owner_has_active_ability_card(st, attackerSide, 322) &&
      (attacker->energyType == GRASS || attacker->energyType == FIRE))
    delta += 20 * active_ability_count_id(st, attackerSide, 322);
  if (attackerInPlay.id == 126 && attackerAbilityActive && defenderActive &&
      pokemon_has_active_ability(st, 1 - attackerSide, defender, true))
    delta += 50;   // Galvantula: into opponent Active Pokemon with an Ability
  if (owner_has_active_ability_card(st, attackerSide, 342) && name_contains(attacker->name, "Cynthia"))
    delta += 30;   // Cynthia's Roserade: your Cynthia's Pokemon
  if (defenderActive)
    delta += 20 * active_ability_count_id(st, attackerSide, 481);
  if (attackerInPlay.id == 439 && attackerAbilityActive &&
      (attackerInPlay.maxHp - attackerInPlay.hp) / 10 >= 2)
    delta += 120;  // Annihilape: this Pokemon has 2+ damage counters
  if (attacker->energyType == FIGHTING)
    delta += 30 * active_ability_count_id(st, attackerSide, 685);
  if (attackerInPlay.id == 829 && attackerAbilityActive && in_play_has_dark_mega_ex(owner))
    delta += 120;  // Seviper: you have any {D} Mega Evolution Pokemon ex
  if (active_tool(st, attackerInPlay, 1158) &&
      (defenderCard->ex || defenderCard->megaEx))
    delta += 50;   // Maximum Belt
  if (active_tool(st, attackerInPlay, 1162) && owner.poisoned)
    delta += 40;   // Binding Mochi
  if (active_tool(st, attackerInPlay, 1175) &&
      (defenderCard->ex || defenderCard->megaEx) && !has_rule_box(attacker))
    delta += 30;   // Brave Bangle
  if (active_tool(st, attackerInPlay, 1171) && name_contains(attacker->name, "Hop") &&
      defenderActive)
    delta += 30;   // Hop's Choice Band: attached Hop's Pokemon
  if (active_tool(st, attackerInPlay, 1178) && name_contains(attacker->name, "Pikachu") &&
      attacker->ex && (defenderCard->ex || defenderCard->megaEx) &&
      defenderActive)
    delta += 50;   // Light Ball: Pikachu ex into Active Pokemon ex
  if (cur_stadium(st) == 1255 && name_contains(attacker->name, "Hop") &&
      defenderActive)
    delta += 30;   // Postwick: Hop's Pokemon
  if (defenderActive && defenderOwner.activeKnown && defenderOwner.active.id == 716 &&
      pokemon_has_active_ability(st, 1 - attackerSide, defenderOwner.active, true))
    delta -= 30;   // Pyroar: opponent Active's attacks do 30 less damage

  return delta;
}

static int apply_defender_attack_damage_mods(GameState& st,
                                             const InPlay& attackerInPlay,
                                             int defenderSide, InPlay& defender,
                                             bool defenderBench, int damage) {
  int dmg = stadium_damage_mod(st, attackerInPlay.id, defender.id, damage);
  if (damage_prevented(st, defender, st.turn, find_card(attackerInPlay.id),
                       attackerInPlay, 1 - defenderSide, dmg) ||
      passive_damage_prevented(st, attackerInPlay, defenderSide,
                               defender, defenderBench, dmg)) {
    return 0;
  }
  bool defenderIsActive = st.players[defenderSide].activeKnown &&
                          &defender == &st.players[defenderSide].active;
  if (defender.id == 970 &&
      !ability_suppressed(st, defenderSide, defender, defenderIsActive) &&
      dmg > 0 && has_energy_type(defender, DARKNESS) &&
      flip_heads(st))
    return 0;  // Fezandipiti: prevent attack damage on heads with {D} Energy.
  if (defender.dmgReduceTurn == st.turn && dmg > 0)
    dmg = dmg > defender.dmgReduce ? dmg - defender.dmgReduce : 0;
  int passiveReduce =
      passive_damage_reduction(st, attackerInPlay, defenderSide, defender, dmg);
  if (passiveReduce > 0 && dmg > 0)
    dmg = dmg > passiveReduce ? dmg - passiveReduce : 0;
  if ((defender.id == 210 || defender.id == 533) &&
      !ability_suppressed(st, defenderSide, defender, defenderIsActive) &&
      defender.hp == defender.maxHp && dmg >= defender.hp)
    dmg = std::max(0, defender.hp - 10);  // Resolute Heart / Sturdy
  if (defender.takeMoreDamageTurn == st.turn && dmg > 0)
    dmg += defender.takeMoreDamage;
  return dmg;
}

// count(source) for AMT_BASE_PLUS_PER. `etype` (>=0) filters energy by resolved
// type; -1 counts any. Damage counters = (maxHp - hp) / 10; prizes taken = 6 - left.
static int count_source(const GameState& st, int actor, int source, int etype) {
  const Player& me = st.players[actor];
  const Player& opp = st.players[1 - actor];
  auto ecount = [etype](const InPlay& p) {
    if (etype < 0) return static_cast<int>(p.energies.size());
    int n = 0;
    for (int e : p.energies)
      if (e == etype || e == RAINBOW) ++n;
    return n;
  };
  switch (source) {
    case CS_OWN_ENERGY: return me.activeKnown ? ecount(me.active) : 0;
    case CS_OPP_ENERGY: return opp.activeKnown ? ecount(opp.active) : 0;
    case CS_OWN_BENCH: return static_cast<int>(me.bench.size());
    case CS_OPP_BENCH: return static_cast<int>(opp.bench.size());
    case CS_BOTH_BENCH:
      return static_cast<int>(me.bench.size() + opp.bench.size());
    case CS_OWN_DMG:
      return me.activeKnown ? (me.active.maxHp - me.active.hp) / 10 : 0;
    case CS_OPP_DMG:
      return opp.activeKnown ? (opp.active.maxHp - opp.active.hp) / 10 : 0;
    case CS_OWN_PRIZES_TAKEN: return me.prizeCount < 6 ? 6 - me.prizeCount : 0;
    case CS_OPP_PRIZES_TAKEN: return opp.prizeCount < 6 ? 6 - opp.prizeCount : 0;
    case CS_OPP_PRIZES_TAKEN_LAST_TURN:
      return st.prizeTakenTurn[1 - actor] == st.turn - 1
                 ? st.prizeTakenCount[1 - actor] : 0;
    case CS_OPP_RETREAT_COST:
      return retreat_cost(st, 1 - actor);
    case CS_OWN_HAND: return me.handCount;
    case CS_OPP_HAND: return opp.handCount;
    case CS_DISCARDED: return st.discardedCount;
    case CS_OWN_INPLAY_ENERGY: {  // energy of type etype across own in-play Pokemon
      auto en = [etype](const InPlay& p) {
        int n = 0;
        for (int e : p.energies)
          if (etype < 0 || e == etype || e == RAINBOW) ++n;
        return n;
      };
      int n = me.activeKnown ? en(me.active) : 0;
      for (const auto& b : me.bench) n += en(b);
      return n;
    }
    case CS_OPP_INPLAY_DMG: {
      int n = opp.activeKnown ? (opp.active.maxHp - opp.active.hp) / 10 : 0;
      for (const auto& b : opp.bench) n += (b.maxHp - b.hp) / 10;
      return n;
    }
    case CS_OPP_INPLAY_ENERGY: {
      auto en = [etype](const InPlay& p) {
        int n = 0;
        for (int e : p.energies)
          if (etype < 0 || e == etype || e == RAINBOW) ++n;
        return n;
      };
      int n = opp.activeKnown ? en(opp.active) : 0;
      for (const auto& b : opp.bench) n += en(b);
      return n;
    }
    case CS_LAST_EFFECT:
      return st.lastEffectCount;
    case CS_OWN_DISCARD_ENERGY: {
      int n = 0;
      for (int id : me.discard) {
        const CardInfo* c = find_card(id);
        if (c && (c->cardType == BASIC_ENERGY || c->cardType == SPECIAL_ENERGY) &&
            (etype < 0 || c->energyType == etype))
          ++n;
      }
      return n;
    }
    case CS_OWN_INPLAY:
    case CS_OPP_INPLAY: {
      const Player& p = (source == CS_OWN_INPLAY) ? me : opp;
      auto typed = [etype](const InPlay& k) {
        const CardInfo* c = find_card(k.id);
        return c && c->cardType == POKEMON &&
               (etype < 0 || pokemon_has_type(k.id, etype));
      };
      int n = (p.activeKnown && typed(p.active)) ? 1 : 0;
      for (const auto& b : p.bench)
        if (typed(b)) ++n;
      return n;
    }
    default: return 0;
  }
}

static bool energy_card_matches(int cardId, int filter) {
  const CardInfo* c = find_card(cardId);
  if (!c || (c->cardType != BASIC_ENERGY && c->cardType != SPECIAL_ENERGY))
    return false;
  if (filter == EF_SPECIAL) return c->cardType == SPECIAL_ENERGY;
  if (filter == EF_SPECIAL_WITH_TOOL) return false;
  if (filter == EF_BASIC) return c->cardType == BASIC_ENERGY;
  if (filter <= EF_BASIC_TYPE_BASE)
    return c->cardType == BASIC_ENERGY &&
           c->energyType == EF_BASIC_TYPE_BASE - filter;
  int wantType = filter - 1;  // filter 0 means any Energy card
  return filter == 0 || c->energyType == wantType;
}

static Player& zone_owner(GameState& st, int actor, int phase) {
  switch (phase) {
    case Z_OPP_HAND:
    case Z_OPP_HAND_BY_OPP:
      return st.players[1 - actor];
    default:
      return st.players[actor];
  }
}

static int zone_owner_index(int actor, int phase) {
  return (phase == Z_OPP_HAND || phase == Z_OPP_HAND_BY_OPP) ? 1 - actor
                                                              : actor;
}

static SmallVec<int, 64>* card_zone_for_phase(Player& p, int phase) {
  switch (phase) {
    case Z_HAND:
    case Z_HAND_ENERGY:
    case Z_OPP_HAND:
    case Z_OPP_HAND_BY_OPP:
      return &p.hand;
    case Z_DISCARD:
    case Z_DISCARD_ENERGY:
      return &p.discard;
    case Z_DECK:
    case Z_DECK_BOTTOM7:
    case Z_DECK_ENERGY:
      return &p.deck;
    default:
      return nullptr;
  }
}

static void adjust_card_zone_count(Player& p, const SmallVec<int, 64>* zone, int delta) {
  if (zone == &p.hand)
    p.handCount += delta;
  else if (zone == &p.deck)
    p.deckCount += delta;
}

static std::vector<int> take_cards_by_index(Player& p, int phase,
                                            const std::vector<int>& refs,
                                            bool energyOnly = false,
                                            bool decrementSourceCount = true,
                                            int* removedOut = nullptr) {
  SmallVec<int, 64>* src = card_zone_for_phase(p, phase);
  std::vector<int> out;
  if (removedOut) *removedOut = 0;
  if (!src) return out;
  for (int idx : refs) {
    if (idx < 0 || idx >= static_cast<int>(src->size())) continue;
    int id = (*src)[idx];
    if (!energyOnly || energy_card_matches(id, 0)) out.push_back(id);
  }
  std::vector<int> idxs = refs;
  std::sort(idxs.rbegin(), idxs.rend());
  idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
  int removed = 0;
  for (int idx : idxs) {
    if (idx < 0 || idx >= static_cast<int>(src->size())) continue;
    int id = (*src)[idx];
    if (energyOnly && !energy_card_matches(id, 0)) continue;
    src->erase(src->begin() + idx);
    ++removed;
  }
  if (decrementSourceCount)
    adjust_card_zone_count(p, src, -removed);
  if (removedOut) *removedOut = removed;
  return out;
}

static int move_card_refs_from_owner(Player& owner, int phase,
                                     const std::vector<int>& refs, int dest,
                                     bool decrementSourceCount = true) {
  SmallVec<int, 64>* srcp = card_zone_for_phase(owner, phase);
  if (!srcp) return 0;
  SmallVec<int, 64>& src = *srcp;
  std::vector<int> idxs = refs;
  std::sort(idxs.rbegin(), idxs.rend());  // erase from the back first
  idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
  int moved = 0;
  for (int idx : idxs) {
    if (idx < 0 || idx >= static_cast<int>(src.size())) continue;
    if (dest == D_BENCH &&
        static_cast<int>(owner.bench.size()) >= owner.benchMax)
      continue;
    int id = src[idx];
    if ((phase == Z_DISCARD || phase == Z_DISCARD_ENERGY) && id == 1096 &&
        (dest == D_HAND || dest == D_DECK || dest == D_DECK_BOTTOM))
      continue;  // Poke Vital A cannot be put into hand/deck from discard.
    bool sourceKnown = false;
    if (&src == &owner.hand) {
      sourceKnown = owner.handKnown;
      auto kit = std::find(owner.handKnownCards.begin(),
                           owner.handKnownCards.end(), id);
      if (kit != owner.handKnownCards.end()) {
        sourceKnown = true;
        owner.handKnownCards.erase(kit);
      }
    } else if (&src == &owner.deck) {
      sourceKnown = owner.deckKnown ||
                    (idx < static_cast<int>(owner.deckKnownMask.size()) &&
                     owner.deckKnownMask[idx]);
      auto kit = std::find(owner.deckKnownCards.begin(),
                           owner.deckKnownCards.end(), id);
      if (kit != owner.deckKnownCards.end()) {
        sourceKnown = true;
        owner.deckKnownCards.erase(kit);
      }
      if (idx < static_cast<int>(owner.deckKnownMask.size()))
        owner.deckKnownMask.erase(owner.deckKnownMask.begin() + idx);
    }
    src.erase(src.begin() + idx);
    if (decrementSourceCount)
      adjust_card_zone_count(owner, &src, -1);
    if (&src == &owner.deck)
      normalize_deck_knowledge(owner);
    if (dest == D_BENCH) {
      InPlay p = make_inplay(id);
      p.appearThisTurn = true;
      owner.bench.push_back(p);
    } else {
      SmallVec<int, 64>& dst = (dest == D_HAND) ? owner.hand
                            : (dest == D_DECK || dest == D_DECK_BOTTOM)
                                  ? owner.deck
                                  : owner.discard;
      if (dest == D_DECK_BOTTOM)
        dst.insert(dst.begin(), id);
      else
        dst.push_back(id);
      adjust_card_zone_count(owner, &dst, 1);
      if (&dst == &owner.hand && sourceKnown && !owner.handKnown)
        add_known_card(owner.handKnownCards, id);
      if (&dst == &owner.deck) {
        if (owner.deckKnownMask.size() < owner.deck.size() - 1)
          owner.deckKnownMask.resize(owner.deck.size() - 1, owner.deckKnown);
        if (dest == D_DECK_BOTTOM)
          owner.deckKnownMask.insert(owner.deckKnownMask.begin(), sourceKnown);
        else
          owner.deckKnownMask.push_back(sourceKnown);
        normalize_deck_knowledge(owner);
      }
    }
    ++moved;
  }
  return moved;
}

static int move_card_refs_to_bench_save(Player& owner, int phase,
                                        const std::vector<int>& refs,
                                        std::vector<int>& saved,
                                        bool decrementSourceCount = true) {
  SmallVec<int, 64>* srcp = card_zone_for_phase(owner, phase);
  if (!srcp) return 0;
  SmallVec<int, 64>& src = *srcp;
  std::vector<int> idxs = refs;
  std::sort(idxs.rbegin(), idxs.rend());
  idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
  int moved = 0;
  for (int idx : idxs) {
    if (idx < 0 || idx >= static_cast<int>(src.size())) continue;
    if (static_cast<int>(owner.bench.size()) >= owner.benchMax) continue;
    int id = src[idx];
    src.erase(src.begin() + idx);
    if (decrementSourceCount)
      adjust_card_zone_count(owner, &src, -1);
    InPlay p = make_inplay(id);
    p.appearThisTurn = true;
    int benchIdx = static_cast<int>(owner.bench.size());
    owner.bench.push_back(p);
    saved.push_back(benchIdx);
    ++moved;
  }
  return moved;
}

static int move_card_refs(GameState& st, int actor, int phase,
                          const std::vector<int>& refs, int dest,
                          bool decrementSourceCount = true) {
  Player& owner = zone_owner(st, actor, phase);
  return move_card_refs_from_owner(owner, phase, refs, dest,
                                   decrementSourceCount);
}

static bool is_counted_out_top_deck_source(const EffectFrame& fr, int ownerSide,
                                           int phase) {
  return fr.topDeckCountedOut > 0 && fr.topDeckOwner == ownerSide &&
         phase == Z_DECK;
}

static std::vector<int> top_deck_actual_refs(const EffectFrame& fr,
                                             const std::vector<int>& refs) {
  std::vector<int> out;
  out.reserve(refs.size());
  for (int ref : refs) {
    if (ref >= 0 && ref < fr.topDeckCount)
      out.push_back(fr.topDeckStart + (fr.topDeckCount - 1 - ref));
    else
      out.push_back(ref);
  }
  return out;
}

static void mark_top_deck_cards_removed(EffectFrame& fr, int moved) {
  if (moved <= 0 || fr.topDeckCountedOut <= 0) return;
  fr.topDeckCountedOut = std::max(0, fr.topDeckCountedOut - moved);
}

static int attached_energy_slots(const InPlay& p) {
  return std::max(static_cast<int>(p.energies.size()),
                  static_cast<int>(p.energyCardIds.size()));
}

static int attached_energy_card_id(const InPlay& p, int idx) {
  return idx >= 0 && idx < static_cast<int>(p.energyCardIds.size())
             ? p.energyCardIds[idx]
             : 0;
}

static int attached_energy_order(const InPlay& p, int idx) {
  return idx >= 0 && idx < static_cast<int>(p.energyCardOrders.size())
             ? p.energyCardOrders[idx]
             : -1;
}

static void push_attached_energy_card(InPlay& p, int cardId,
                                      int attachOrder = -1) {
  p.energyCardIds.push_back(cardId);
  if (!p.energyCardOrders.empty() || attachOrder >= 0) {
    while (p.energyCardOrders.size() + 1 < p.energyCardIds.size())
      p.energyCardOrders.push_back(-1);
    p.energyCardOrders.push_back(attachOrder);
  }
}

static void erase_attached_energy_card(InPlay& p, int idx) {
  if (idx >= 0 && idx < static_cast<int>(p.energyCardIds.size()))
    p.energyCardIds.erase(p.energyCardIds.begin() + idx);
  if (idx >= 0 && idx < static_cast<int>(p.energyCardOrders.size()))
    p.energyCardOrders.erase(p.energyCardOrders.begin() + idx);
}

static void pop_attached_energy_card(InPlay& p) {
  if (!p.energyCardIds.empty()) p.energyCardIds.pop_back();
  if (!p.energyCardOrders.empty()) p.energyCardOrders.pop_back();
}

static bool energy_unit_provides_type(int actualType, int wantType) {
  if (actualType == wantType) return true;
  if (actualType == RAINBOW) return true;
  return actualType == TEAM_ROCKET &&
         (wantType == PSYCHIC || wantType == DARKNESS);
}

static bool energy_unit_matches(const InPlay& p, int idx, int filter) {
  if (idx < 0 || idx >= attached_energy_slots(p)) return false;
  const CardInfo* attachedCard =
      idx < static_cast<int>(p.energyCardIds.size())
          ? find_card(p.energyCardIds[idx])
          : nullptr;
  int actualType = idx < static_cast<int>(p.energies.size())
                       ? p.energies[idx]
                       : (attachedCard ? attachedCard->energyType : COLORLESS);
  if (filter == EF_BASIC) {
    if (!attachedCard) return idx < static_cast<int>(p.energies.size());
    return attachedCard->cardType == BASIC_ENERGY;
  }
  if (filter <= EF_BASIC_TYPE_BASE) {
    int wantType = EF_BASIC_TYPE_BASE - filter;
    if (actualType != wantType) return false;
    if (!attachedCard) return idx < static_cast<int>(p.energies.size());
    return attachedCard->cardType == BASIC_ENERGY &&
           attachedCard->energyType == wantType;
  }
  if (filter == EF_SPECIAL) {
    return attachedCard && attachedCard->cardType == SPECIAL_ENERGY;
  }
  if (filter == EF_SPECIAL_WITH_TOOL) {
    return attachedCard &&
           (attachedCard->cardType == BASIC_ENERGY ||
            attachedCard->cardType == SPECIAL_ENERGY) &&
           !p.tools.empty();
  }
  int wantType = filter - 1;  // filter 0 means any energy unit
  return filter == 0 || energy_unit_provides_type(actualType, wantType);
}

static bool energy_slot_is_card_backed_or_legacy(const InPlay& p, int idx) {
  return p.energyCardIds.empty() ||
         idx < static_cast<int>(p.energyCardIds.size());
}

static bool energy_slot_can_move_to(const InPlay& src, int idx,
                                    const InPlay& dst,
                                    bool allowRestrictedSpecial = false) {
  if (!energy_slot_is_card_backed_or_legacy(src, idx)) return false;
  int cardId = idx < static_cast<int>(src.energyCardIds.size())
                   ? src.energyCardIds[idx]
                   : 0;
  return allowRestrictedSpecial || cardId != 15 || is_team_rocket_card_id(dst.id);
}

static int encode_energy_ref(int inplayIdx, int energyIdx) {
  return (inplayIdx + 1) * 1000 + energyIdx;  // Active = -1 -> 0..999
}

static int energy_ref_inplay(int ref) {
  return ref / 1000 - 1;
}

static int energy_ref_index(int ref) {
  return ref % 1000;
}

static InPlay* inplay_ref(Player& p, int idx) {
  if (idx < 0) return p.activeKnown ? &p.active : nullptr;
  return idx < static_cast<int>(p.bench.size()) ? &p.bench[idx] : nullptr;
}

static void push_program_frame(GameState& st, int prog, int actor, int selfBench) {
  if (prog < 0) return;
  EffectFrame ef;
  ef.effect = FLOW_PROGRAM;
  ef.program = prog;
  ef.a = actor;
  ef.selfBench = selfBench;
  st.effectStack.push_back(ef);
}

static void queue_program_frame(GameState& st, int prog, int actor, int selfBench,
                                bool afterCurrentProgram, int sourceCardId = 0) {
  if (prog < 0) return;
  EffectFrame ef;
  ef.effect = FLOW_PROGRAM;
  ef.program = prog;
  ef.a = actor;
  ef.selfBench = selfBench;
  ef.sourceCardId = sourceCardId;
  if (afterCurrentProgram && !st.effectStack.empty() &&
      st.effectStack.back().effect == FLOW_PROGRAM) {
    st.afterProgramQueue.push_back(ef);
  } else {
    st.effectStack.push_back(ef);
  }
}

using DamageTrigger = std::tuple<int, int, int, int>;  // program, actor, selfBench, sourceCardId

static void queue_deferred_program_frame(GameState& st, int prog, int actor,
                                         int selfBench, int sourceCardId) {
  if (prog < 0) return;
  EffectFrame ef;
  ef.effect = FLOW_PROGRAM;
  ef.program = prog;
  ef.a = actor;
  ef.selfBench = selfBench;
  ef.sourceCardId = sourceCardId;
  st.afterProgramQueue.push_back(ef);
}

static bool queue_damage_trigger_order_frame(GameState& st,
                                             const std::vector<DamageTrigger>& triggers) {
  if (triggers.size() < 2) return false;
  if (triggers.size() == 2) {
    int a = std::get<3>(triggers[0]);
    int b = std::get<3>(triggers[1]);
    if ((a == 1059 && b == 1176) || (a == 1176 && b == 1059))
      return false;
  }
  EffectFrame fr;
  fr.effect = EFF_DAMAGE_TRIGGER_ORDER;
  fr.phase = static_cast<int>(triggers.size());
  fr.a = std::get<1>(triggers.front());
  for (const auto& trigger : triggers) {
    fr.scratch.push_back(std::get<0>(trigger));
    fr.scratch.push_back(std::get<1>(trigger));
    fr.scratch.push_back(std::get<2>(trigger));
    fr.scratch.push_back(std::get<3>(trigger));
  }
  st.afterProgramQueue.push_back(fr);
  return true;
}

static bool on_damage_source_can_damage_attacker(int sourceCardId) {
  switch (sourceCardId) {
    case 14:    // Spiky Energy
    case 180:   // Bruxish
    case 255:   // Maractus
    case 688:   // Spiritomb
    case 993:   // Orthworm ex
    case 1167:  // Deluxe Bomb
    case 1176:  // Punk Helmet
      return true;
    default:
      return false;
  }
}

static bool queued_attacker_damage_trigger(const GameState& st) {
  for (const EffectFrame& ef : st.afterProgramQueue) {
    if (ef.effect == FLOW_PROGRAM &&
        on_damage_source_can_damage_attacker(ef.sourceCardId))
      return true;
    if (ef.effect == EFF_DAMAGE_TRIGGER_ORDER) {
      int n = static_cast<int>(ef.scratch.size()) / 4;
      for (int i = 0; i < n; ++i)
        if (on_damage_source_can_damage_attacker(ef.scratch[i * 4 + 3]))
          return true;
    }
  }
  return false;
}

static bool run_pre_promote_immediate_damage_hooks(GameState& st,
                                                   int promotePlayer = -1) {
  if (st.afterProgramQueue.empty()) return false;
  bool hasOrder = false;
  for (const EffectFrame& ef : st.afterProgramQueue) {
    if (ef.effect == EFF_DAMAGE_TRIGGER_ORDER) {
      hasOrder = true;
      break;
    }
  }
  if (hasOrder) {
    if (promotePlayer >= 0) {
      EffectFrame promote;
      promote.effect = EFF_ABILITY_PROMOTE;
      promote.a = promotePlayer;
      st.effectStack.push_back(promote);
    }
    for (auto it = st.afterProgramQueue.rbegin();
         it != st.afterProgramQueue.rend(); ++it)
      st.effectStack.push_back(*it);
    st.afterProgramQueue.clear();
    if (!st.effectStack.empty() &&
        st.effectStack.back().effect == EFF_DAMAGE_TRIGGER_ORDER) {
      set_damage_trigger_order_pending_from_frame(st, st.effectStack.back());
      return true;
    }
    return false;
  }
  std::vector<EffectFrame> keep;
  keep.reserve(st.afterProgramQueue.size());
  for (const EffectFrame& ef : st.afterProgramQueue) {
    if (ef.effect == FLOW_PROGRAM && ef.sourceCardId == 1156) {
      st.yourIndex = ef.a;
      st.lastEffectCount = draw_n(st, st.players[ef.a], 2);
    } else {
      keep.push_back(ef);
    }
  }
  st.afterProgramQueue.swap(keep);
  return false;
}

static void set_damage_trigger_order_pending_from_frame(GameState& st,
                                                        const EffectFrame& fr) {
  PendingDecision pd;
  pd.context = 34;  // SKILL order
  pd.minCount = fr.phase;
  pd.maxCount = fr.phase;
  int n = static_cast<int>(fr.scratch.size()) / 4;
  for (int i = 0; i < n; ++i) {
    int sourceCardId = fr.scratch[i * 4 + 3];
    pd.options.push_back({Atom::S("SKILL"), Atom::I(sourceCardId), Atom::N()});
  }
  st.yourIndex = fr.a;
  st.pending = pd;
}

static void apply_darkest_impulse(GameState& st, int actor, int targetIdx) {
  if (actor < 0 || actor > 1) return;
  if (in_play_count_id(st.players[1 - actor], 428) <= 0) return;
  InPlay* target = inplay_ref(st.players[actor], targetIdx);
  if (!target) return;
  target->hp -= 40;  // Team Rocket's Ampharos: Darkest Impulse, non-stacking.
  if (target->hp < 0) target->hp = 0;
}

static void set_evolve_trigger_order_pending(GameState& st,
                                             const EffectFrame& fr) {
  PendingDecision pd;
  pd.context = 34;  // SKILL order
  pd.minCount = 2;
  pd.maxCount = 2;
  pd.options.push_back({Atom::S("SKILL"), Atom::I(428), Atom::N()});
  pd.options.push_back({Atom::S("SKILL"), Atom::I(fr.sourceCardId), Atom::N()});
  st.yourIndex = fr.a;
  st.pending = pd;
}

static bool start_evolve_trigger_order(GameState& st, int actor, int targetIdx,
                                       int cardId) {
  if (in_play_count_id(st.players[1 - actor], 428) <= 0) return false;
  bool cardTrigger = card_program(cardId) >= 0;
  bool onEvolveTrigger = on_evolve_program(cardId) >= 0;
  InPlay* evolved = inplay_ref(st.players[actor], targetIdx);
  if (onEvolveTrigger && evolved &&
      ability_suppressed(st, actor, *evolved, targetIdx < 0))
    onEvolveTrigger = false;
  if (!cardTrigger && !onEvolveTrigger) return false;
  EffectFrame fr;
  fr.effect = EFF_EVOLVE_TRIGGER_ORDER;
  fr.phase = 2;
  fr.a = actor;
  fr.selfBench = targetIdx;
  fr.sourceCardId = cardId;
  st.effectStack.push_back(fr);
  set_evolve_trigger_order_pending(st, st.effectStack.back());
  return true;
}

static void set_on_play_basic_trigger_order_pending(GameState& st,
                                                    const EffectFrame& fr) {
  PendingDecision pd;
  pd.context = 34;  // SKILL order
  pd.minCount = 2;
  pd.maxCount = 2;
  pd.options.push_back({Atom::S("SKILL"), Atom::I(1260), Atom::N()});
  pd.options.push_back({Atom::S("SKILL"), Atom::I(fr.sourceCardId), Atom::N()});
  st.yourIndex = fr.a;
  st.pending = pd;
}

static bool start_on_play_basic_trigger_order(GameState& st, int actor,
                                              int benchIdx, int cardId) {
  if (on_play_program(cardId) < 0) return false;
  if (!risky_ruins_bench_entry_applies(st, actor, benchIdx)) return false;
  EffectFrame fr;
  fr.effect = EFF_ON_PLAY_BASIC_TRIGGER_ORDER;
  fr.phase = 2;
  fr.a = actor;
  fr.selfBench = benchIdx;
  fr.sourceCardId = cardId;
  st.effectStack.push_back(fr);
  set_on_play_basic_trigger_order_pending(st, st.effectStack.back());
  return true;
}

static void set_on_attach_trigger_order_pending(GameState& st,
                                                const EffectFrame& fr) {
  PendingDecision pd;
  pd.context = 34;  // SKILL order
  pd.minCount = 2;
  pd.maxCount = 2;
  pd.options.push_back({Atom::S("SKILL"), Atom::I(fr.savedSrc), Atom::N()});
  pd.options.push_back({Atom::S("SKILL"), Atom::I(fr.sourceCardId), Atom::N()});
  st.yourIndex = fr.a;
  st.pending = pd;
}

static bool start_on_attach_trigger_order(GameState& st, int actor,
                                          int targetIdx, int targetCardId,
                                          int energyCardId) {
  if (energyCardId != 13) return false;  // Enriching Energy draw trigger.
  const Player& me = st.players[actor];
  if (!me.activeKnown || me.active.id != 297) return false;
  if (on_attach_program(energyCardId) < 0) return false;
  EffectFrame fr;
  fr.effect = EFF_ON_ATTACH_TRIGGER_ORDER;
  fr.phase = 2;
  fr.a = actor;
  fr.selfBench = targetIdx;
  fr.savedSrc = me.active.id;
  fr.savedPhase = targetCardId;
  fr.sourceCardId = energyCardId;
  st.effectStack.push_back(fr);
  set_on_attach_trigger_order_pending(st, st.effectStack.back());
  return true;
}

static void set_palafin_yesno_pending(GameState& st, EffectFrame& fr) {
  (void)fr;
  PendingDecision pd;
  pd.context = 43;  // ACTIVATE.
  pd.minCount = 1;
  pd.maxCount = 1;
  pd.options.push_back({Atom::S("YES")});
  pd.options.push_back({Atom::S("NO")});
  st.pending = pd;
}

static void queue_palafin_zero_to_hero(GameState& st, int owner, int benchIdx,
                                       bool afterCurrentProgram) {
  Player& p = st.players[owner];
  if (benchIdx < 0 || benchIdx >= static_cast<int>(p.bench.size())) return;
  if (p.bench[benchIdx].id != 106 || p.bench[benchIdx].abilityUsedThisTurn)
    return;
  // Whether Palafin ex is actually in the hidden deck is learned only after
  // choosing to use the Ability and searching. It must not change the
  // pre-reveal YES/NO action set.
  EffectFrame ef;
  ef.effect = EFF_PALAFIN_ZERO_TO_HERO;
  ef.a = owner;
  ef.selfBench = benchIdx;
  ef.phase = 0;
  if (afterCurrentProgram && !st.effectStack.empty() &&
      st.effectStack.back().effect == FLOW_PROGRAM) {
    st.afterProgramQueue.push_back(ef);
  } else {
    st.effectStack.push_back(ef);
    st.yourIndex = owner;
    set_palafin_yesno_pending(st, st.effectStack.back());
  }
}

static void queue_move_triggers(GameState& st, int owner, int activeToBenchIdx,
                                bool benchToActive, bool afterCurrentProgram) {
  if (owner != st.yourIndex) return;  // these "when this moves" hooks are during your turn
  Player& p = st.players[owner];
  if (activeToBenchIdx >= 0 && activeToBenchIdx < static_cast<int>(p.bench.size())) {
    InPlay& moved = p.bench[activeToBenchIdx];
    if (moved.id == 106 && !moved.abilityUsedThisTurn)
      queue_palafin_zero_to_hero(st, owner, activeToBenchIdx, afterCurrentProgram);
    int prog = on_active_to_bench_program(moved.id);
    if (prog >= 0 && !moved.abilityUsedThisTurn)
      queue_program_frame(st, prog, owner, activeToBenchIdx, afterCurrentProgram);
  }
  if (benchToActive && p.activeKnown) {
    int prog = on_bench_to_active_program(p.active.id);
    if (prog >= 0 && !p.active.abilityUsedThisTurn)
      queue_program_frame(st, prog, owner, -1, afterCurrentProgram);
    int movedId = p.active.id;
    auto maybeObserver = [&](InPlay& observer, int selfBench) {
      int observerProg = on_bench_to_active_observer_program(observer.id);
      if (observerProg < 0 || observer.abilityUsedThisTurn) return;
      int watchedId = bench_to_active_observer_watch(observer.id);
      if (watchedId >= 0 && watchedId != movedId) return;
      queue_program_frame(st, observerProg, owner, selfBench, afterCurrentProgram);
    };
    maybeObserver(p.active, -1);
    for (int j = 0; j < static_cast<int>(p.bench.size()); ++j)
      maybeObserver(p.bench[j], j);
  }
}

static void set_palafin_deck_pending_or_finish(GameState& st, EffectFrame& fr) {
  Player& me = st.players[fr.a];
  mark_full_deck_inspected(me);
  PendingDecision pd;
  pd.context = 7;  // TO_HAND/deck-search style choice.
  pd.minCount = 1;
  pd.maxCount = 1;
  for (int i = 0; i < static_cast<int>(me.deck.size()); ++i) {
    if (me.deck[i] != 107) continue;
    pd.options.push_back({Atom::S("CARD"), Atom::S("DECK"), Atom::I(i),
                          Atom::I(107)});
  }
  if (pd.options.empty()) {
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    if (st.effectStack.empty() && st.pendingTrainerDiscard >= 0)
      discard_pending_trainer(st);
    return;
  }
  st.pending = pd;
}

static void resolve_palafin_zero_to_hero(GameState& st, const std::vector<int>& sel) {
  if (st.effectStack.empty()) return;
  EffectFrame& fr = st.effectStack.back();
  Player& me = st.players[fr.a];
  InPlay* target = inplay_ref(me, fr.selfBench);
  if (!target || target->id != 106) {
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    if (st.effectStack.empty() && st.pendingTrainerDiscard >= 0)
      discard_pending_trainer(st);
    return;
  }
  if (fr.phase == 0) {
    bool yes = !sel.empty() && sel[0] == 0;
    st.pending = PendingDecision();
    st.turnActionCount += 1;
    if (!yes) {
      st.effectStack.pop_back();
      if (st.effectStack.empty() && st.pendingTrainerDiscard >= 0)
        discard_pending_trainer(st);
      return;
    }
    target->abilityUsedThisTurn = true;
    fr.phase = 1;
    set_palafin_deck_pending_or_finish(st, fr);
    return;
  }

  if (!sel.empty() && sel[0] >= 0 &&
      sel[0] < static_cast<int>(st.pending.options.size())) {
    int deckIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
    if (deckIdx >= 0 && deckIdx < static_cast<int>(me.deck.size()) &&
        me.deck[deckIdx] == 107) {
      erase_deck_at(me, deckIdx);
      int damage = target->maxHp - target->hp;
      target->id = 107;
      target->maxHp = effective_max_hp(107, target->tools, st,
                                       target->energyCardIds,
                                       target->energies, fr.a);
      target->hp = std::max(0, target->maxHp - damage);
      target->abilityUsedThisTurn = true;
      push_deck_top(me, 106, true);
      shuffle_deck_known(me, st.rng);
    }
  }
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  st.turnActionCount += 1;
  if (st.effectStack.empty() && st.pendingTrainerDiscard >= 0)
    discard_pending_trainer(st);
}

static int move_chosen_to_top_after_shuffle(GameState& st, EffectFrame& fr) {
  Player& me = st.players[fr.a];
  std::vector<int> refs;
  for (int ref : fr.scratch) {
    if (ref < 0 || ref >= static_cast<int>(me.deck.size())) continue;
    if (std::find(refs.begin(), refs.end(), ref) == refs.end()) refs.push_back(ref);
  }
  std::vector<int> selected;
  std::vector<bool> selectedKnown;
  for (int ref : refs) {
    selected.push_back(me.deck[ref]);
    selectedKnown.push_back(deck_card_known_at(me, ref));
  }
  std::vector<int> eraseRefs = refs;
  std::sort(eraseRefs.rbegin(), eraseRefs.rend());
  for (int ref : eraseRefs) erase_deck_at(me, ref);
  shuffle_deck_known(me, st.rng);
  for (int i = static_cast<int>(selected.size()) - 1; i >= 0; --i)
    push_deck_top(me, selected[i], selectedKnown[i]);
  return static_cast<int>(selected.size());
}

static int apply_deck_evolve_chosen(GameState& st, EffectFrame& fr,
                                    int targetMode, bool requireNoAbility,
                                    int& targetIdx) {
  Player& me = st.players[fr.a];
  targetIdx = (targetMode == DET_SAVED_OWN_INPLAY && !fr.savedScratch.empty())
                  ? fr.savedScratch[0]
                  : fr.selfBench;
  int deckIdx = fr.scratch.empty() ? -1 : fr.scratch[0];
  if (targetMode == DET_AUTO_OWN_INPLAY ||
      targetMode == DET_AUTO_OWN_INPLAY_NO_ABILITY) {
    targetIdx = -999;
    if (deckIdx >= 0 && deckIdx < static_cast<int>(me.deck.size())) {
      int candidate = me.deck[deckIdx];
      if (me.activeKnown &&
          card_evolves_from(candidate, me.active.id, requireNoAbility)) {
        targetIdx = -1;
      } else {
        for (int i = 0; i < static_cast<int>(me.bench.size()); ++i) {
          if (card_evolves_from(candidate, me.bench[i].id, requireNoAbility)) {
            targetIdx = i;
            break;
          }
        }
      }
    }
  } else if (targetMode == DET_AUTO_OWN_BENCH) {
    targetIdx = -999;
    if (deckIdx >= 0 && deckIdx < static_cast<int>(me.deck.size())) {
      int candidate = me.deck[deckIdx];
      for (int i = 0; i < static_cast<int>(me.bench.size()); ++i) {
        if (card_evolves_from(candidate, me.bench[i].id, requireNoAbility)) {
          targetIdx = i;
          break;
        }
      }
    }
  }
  InPlay* target = inplay_ref(me, targetIdx);
  int evolvedId = 0;
  if (target && deckIdx >= 0 && deckIdx < static_cast<int>(me.deck.size())) {
    int candidate = me.deck[deckIdx];
    if (card_evolves_from(candidate, target->id, requireNoAbility)) {
      evolvedId = candidate;
      erase_deck_at(me, deckIdx);
      int damage = target->maxHp - target->hp;
      target->preEvo.push_back(target->id);
      target->id = candidate;
      target->maxHp = effective_max_hp(candidate, target->tools, st,
                                       target->energyCardIds, target->energies,
                                       fr.a);
      target->hp = target->maxHp - damage;
      target->appearThisTurn = true;
      clear_status_for_evolve(st, fr.a, targetIdx < 0, target);
    }
  }
  shuffle_deck_known(me, st.rng);
  return evolvedId;
}

static int apply_hand_stage2_evolve_chosen(GameState& st, EffectFrame& fr,
                                           int& targetIdx) {
  Player& me = st.players[fr.a];
  targetIdx = fr.savedScratch.empty() ? fr.selfBench : fr.savedScratch[0];
  InPlay* target = inplay_ref(me, targetIdx);
  int handIdx = fr.scratch.empty() ? -1 : fr.scratch[0];
  if (!target || handIdx < 0 || handIdx >= static_cast<int>(me.hand.size()))
    return 0;
  int candidate = me.hand[handIdx];
  if (!stage2_evolves_from_basic(candidate, target->id)) return 0;
  me.hand.erase(me.hand.begin() + handIdx);
  me.handCount -= 1;
  int damage = target->maxHp - target->hp;
  target->preEvo.push_back(target->id);
  target->id = candidate;
  target->maxHp = effective_max_hp(candidate, target->tools, st,
                                   target->energyCardIds, target->energies,
                                   fr.a);
  target->hp = std::max(0, target->maxHp - damage);
  target->appearThisTurn = true;
  clear_status_for_evolve(st, fr.a, targetIdx < 0, target);
  if (in_play_count_id(st.players[1 - fr.a], 428) > 0) {
    target->hp -= 40;  // Team Rocket's Ampharos: Darkest Impulse.
    if (target->hp < 0) target->hp = 0;
  }
  return candidate;
}

static int auto_deck_evolve_target(GameState& st, int actor, int targetIdx,
                                   bool requireNoAbility) {
  Player& me = st.players[actor];
  InPlay* target = inplay_ref(me, targetIdx);
  if (!target) return 0;
  for (int i = 0; i < static_cast<int>(me.deck.size()); ++i) {
    int candidate = me.deck[i];
    if (!card_evolves_from(candidate, target->id, requireNoAbility)) continue;
    erase_deck_at(me, i);
    int damage = target->maxHp - target->hp;
    target->preEvo.push_back(target->id);
    target->id = candidate;
    target->maxHp = effective_max_hp(candidate, target->tools, st,
                                     target->energyCardIds, target->energies,
                                     actor);
    target->hp = target->maxHp - damage;
    target->appearThisTurn = true;
    clear_status_for_evolve(st, actor, targetIdx < 0, target);
    return candidate;
  }
  return 0;
}

static void push_evolve_hooks(GameState& st, int actor,
                              const std::vector<std::pair<int, int>>& evolved) {
  for (auto it = evolved.rbegin(); it != evolved.rend(); ++it) {
    int evolvedId = it->first;
    int targetIdx = it->second;
    push_program_frame(st, on_evolve_program(evolvedId), actor, targetIdx);
    push_program_frame(st, card_program(evolvedId), actor, targetIdx);
  }
}

static void evolve_in_place_from_deck(GameState& st, int actor, int deckIdx,
                                      int targetIdx, int evolvedId) {
  Player& me = st.players[actor];
  InPlay* target = inplay_ref(me, targetIdx);
  if (!target || deckIdx < 0 || deckIdx >= static_cast<int>(me.deck.size()))
    return;
  if (me.deck[deckIdx] != evolvedId) return;
  erase_deck_at(me, deckIdx);
  int damage = target->maxHp - target->hp;
  target->preEvo.push_back(target->id);
  target->id = evolvedId;
  target->maxHp = effective_max_hp(evolvedId, target->tools, st,
                                   target->energyCardIds, target->energies,
                                   actor);
  target->hp = std::max(0, target->maxHp - damage);
  target->appearThisTurn = true;
  clear_status_for_evolve(st, actor, targetIdx < 0, target);
  if (in_play_count_id(st.players[1 - actor], 428) > 0) {
    target->hp -= 40;  // Team Rocket's Ampharos: Darkest Impulse.
    if (target->hp < 0) target->hp = 0;
  }
}

static void set_grand_tree_stage1_pending(GameState& st, EffectFrame& fr) {
  Player& me = st.players[fr.a];
  mark_full_deck_inspected(me);
  PendingDecision pd;
  pd.context = 7;  // TO_HAND/deck-search style choice.
  pd.minCount = 1;
  pd.maxCount = 1;
  auto add_for_target = [&](const InPlay& p, int targetIdx) {
    if (!grand_tree_basic_target_ok(st, fr.a, p)) return;
    for (int i = 0; i < static_cast<int>(me.deck.size()); ++i) {
      int id = me.deck[i];
      const CardInfo* c = find_card(id);
      if (!c || !c->stage1 || !card_evolves_from(id, p.id, false)) continue;
      pd.options.push_back({Atom::S("CARD"), Atom::S("DECK"), Atom::I(i),
                            Atom::I(id), Atom::I(targetIdx)});
    }
  };
  if (me.activeKnown) add_for_target(me.active, -1);
  for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
    add_for_target(me.bench[j], j);
  if (pd.options.empty()) {
    shuffle_deck_known(me, st.rng);
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    return;
  }
  st.pending = pd;
}

static void set_grand_tree_stage2_pending_or_finish(GameState& st,
                                                    EffectFrame& fr) {
  Player& me = st.players[fr.a];
  mark_full_deck_inspected(me);
  InPlay* target = inplay_ref(me, fr.selfBench);
  PendingDecision pd;
  pd.context = 7;
  pd.minCount = 0;
  pd.maxCount = 1;
  if (target) {
    for (int i = 0; i < static_cast<int>(me.deck.size()); ++i) {
      int id = me.deck[i];
      const CardInfo* c = find_card(id);
      if (!c || !c->stage2 || !card_evolves_from(id, target->id, false))
        continue;
      pd.options.push_back({Atom::S("CARD"), Atom::S("DECK"), Atom::I(i),
                            Atom::I(id), Atom::I(fr.selfBench)});
    }
  }
  if (pd.options.empty()) {
    shuffle_deck_known(me, st.rng);
    std::vector<std::pair<int, int>> evolved;
    if (fr.savedSrc > 0) evolved.push_back({fr.savedSrc, fr.selfBench});
    int actor = fr.a;
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    push_evolve_hooks(st, actor, evolved);
    if (!st.effectStack.empty()) run_program(st, {});
    return;
  }
  fr.phase = 1;
  st.pending = pd;
}

static bool start_grand_tree(GameState& st) {
  if (cur_stadium(st) != 1249 || st.stadiumAbilityUsed ||
      !grand_tree_has_stage1_option(st, st.yourIndex))
    return false;
  st.stadiumAbilityUsed = true;
  st.turnActionCount += 1;
  EffectFrame fr;
  fr.effect = EFF_GRAND_TREE;
  fr.a = st.yourIndex;
  fr.phase = 0;
  st.effectStack.push_back(fr);
  set_grand_tree_stage1_pending(st, st.effectStack.back());
  return true;
}

static void resolve_grand_tree(GameState& st, const std::vector<int>& sel) {
  if (st.effectStack.empty()) return;
  EffectFrame& fr = st.effectStack.back();
  Player& me = st.players[fr.a];
  std::vector<std::pair<int, int>> evolved;
  if (fr.phase == 0) {
    if (!sel.empty() && sel[0] >= 0 &&
        sel[0] < static_cast<int>(st.pending.options.size())) {
      const Descriptor& d = st.pending.options[sel[0]];
      int deckIdx = static_cast<int>(d[2].i);
      int evolvedId = static_cast<int>(d[3].i);
      int targetIdx = static_cast<int>(d[4].i);
      InPlay* target = inplay_ref(me, targetIdx);
      if (target && deckIdx >= 0 &&
          deckIdx < static_cast<int>(me.deck.size()) &&
          me.deck[deckIdx] == evolvedId &&
          grand_tree_basic_target_ok(st, fr.a, *target) &&
          card_evolves_from(evolvedId, target->id, false)) {
        evolve_in_place_from_deck(st, fr.a, deckIdx, targetIdx, evolvedId);
        fr.selfBench = targetIdx;
        fr.savedSrc = evolvedId;
        st.pending = PendingDecision();
        st.turnActionCount += 1;
        set_grand_tree_stage2_pending_or_finish(st, fr);
        return;
      }
    }
    shuffle_deck_known(me, st.rng);
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    st.turnActionCount += 1;
    return;
  }

  if (fr.savedSrc > 0) evolved.push_back({fr.savedSrc, fr.selfBench});
  if (!sel.empty() && sel[0] >= 0 &&
      sel[0] < static_cast<int>(st.pending.options.size())) {
    const Descriptor& d = st.pending.options[sel[0]];
    int deckIdx = static_cast<int>(d[2].i);
    int evolvedId = static_cast<int>(d[3].i);
    int targetIdx = static_cast<int>(d[4].i);
    InPlay* target = inplay_ref(me, targetIdx);
    if (target && deckIdx >= 0 && deckIdx < static_cast<int>(me.deck.size()) &&
        me.deck[deckIdx] == evolvedId && card_evolves_from(evolvedId, target->id, false)) {
      evolve_in_place_from_deck(st, fr.a, deckIdx, targetIdx, evolvedId);
      evolved.push_back({evolvedId, targetIdx});
    }
  }
  shuffle_deck_known(me, st.rng);
  int actor = fr.a;
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  st.turnActionCount += 1;
  push_evolve_hooks(st, actor, evolved);
  if (!st.effectStack.empty()) run_program(st, {});
}

static bool is_ogerpon_ex_card(int id) {
  const CardInfo* c = find_card(id);
  return c && c->cardType == POKEMON && c->ex && c->name &&
         name_contains(c->name, "Ogerpon");
}

static bool ogres_mask_available(const GameState& st, int side) {
  const Player& me = st.players[side];
  bool discardOgerpon = false;
  for (int id : me.discard) {
    if (is_ogerpon_ex_card(id)) {
      discardOgerpon = true;
      break;
    }
  }
  if (!discardOgerpon) return false;
  if (me.activeKnown && is_ogerpon_ex_card(me.active.id)) return true;
  for (const auto& b : me.bench)
    if (is_ogerpon_ex_card(b.id)) return true;
  return false;
}

static void set_ogres_mask_discard_pending(GameState& st, EffectFrame& fr) {
  Player& me = st.players[fr.a];
  PendingDecision pd;
  pd.context = 6;  // CABT uses DISCARD_FROM_DISCARD for the discard-pile pick.
  pd.minCount = 1;
  pd.maxCount = 1;
  for (int i = 0; i < static_cast<int>(me.discard.size()); ++i) {
    if (!is_ogerpon_ex_card(me.discard[i])) continue;
    pd.options.push_back({Atom::S("CARD"), Atom::S("DISCARD"), Atom::I(i),
                          Atom::I(me.discard[i])});
  }
  st.pending = pd;
}

static void set_ogres_mask_inplay_pending(GameState& st, EffectFrame& fr) {
  Player& me = st.players[fr.a];
  PendingDecision pd;
  pd.context = 43;  // ACTIVATE.
  pd.minCount = 1;
  pd.maxCount = 1;
  if (me.activeKnown && is_ogerpon_ex_card(me.active.id))
    pd.options.push_back({Atom::S("CARD"), Atom::S("ACTIVE"), Atom::I(-1),
                          Atom::I(me.active.id)});
  for (int j = 0; j < static_cast<int>(me.bench.size()); ++j) {
    if (!is_ogerpon_ex_card(me.bench[j].id)) continue;
    pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                          Atom::I(me.bench[j].id)});
  }
  st.pending = pd;
}

static void play_ogres_mask(GameState& st) {
  Player& me = st.players[st.yourIndex];
  if (!ogres_mask_available(st, st.yourIndex)) return;
  remove_hand_card(st, me, 1090);
  st.turnActionCount += 1;
  st.pendingTrainerDiscard = 1090;
  st.pendingTrainerOwner = st.yourIndex;
  EffectFrame fr;
  fr.effect = EFF_OGRES_MASK;
  fr.a = st.yourIndex;
  fr.phase = 0;
  st.effectStack.push_back(fr);
  set_ogres_mask_discard_pending(st, st.effectStack.back());
}

static void resolve_ogres_mask(GameState& st, const std::vector<int>& sel) {
  if (st.effectStack.empty()) return;
  EffectFrame& fr = st.effectStack.back();
  Player& me = st.players[fr.a];
  if (fr.phase == 0) {
    if (sel.empty() || sel[0] < 0 ||
        sel[0] >= static_cast<int>(st.pending.options.size())) {
      st.effectStack.pop_back();
      st.pending = PendingDecision();
      return;
    }
    const Descriptor& d = st.pending.options[sel[0]];
    int discardIdx = static_cast<int>(d[2].i);
    int cardId = static_cast<int>(d[3].i);
    if (discardIdx < 0 || discardIdx >= static_cast<int>(me.discard.size()) ||
        me.discard[discardIdx] != cardId || !is_ogerpon_ex_card(cardId)) {
      st.effectStack.pop_back();
      st.pending = PendingDecision();
      return;
    }
    fr.savedSrc = discardIdx;
    fr.sourceCardId = cardId;
    fr.phase = 1;
    st.pending = PendingDecision();
    st.turnActionCount += 1;
    set_ogres_mask_inplay_pending(st, fr);
    return;
  }

  int discardIdx = fr.savedSrc;
  int newId = fr.sourceCardId;
  int targetIdx = -2;
  if (!sel.empty() && sel[0] >= 0 &&
      sel[0] < static_cast<int>(st.pending.options.size()))
    targetIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
  InPlay* target = inplay_ref(me, targetIdx);
  if (target && discardIdx >= 0 &&
      discardIdx < static_cast<int>(me.discard.size()) &&
      me.discard[discardIdx] == newId && is_ogerpon_ex_card(target->id)) {
    int oldId = target->id;
    me.discard.erase(me.discard.begin() + discardIdx);
    int damage = target->maxHp - target->hp;
    target->id = newId;
    target->maxHp = effective_max_hp(newId, target->tools, st,
                                     target->energyCardIds, target->energies,
                                     fr.a);
    target->hp = std::max(0, target->maxHp - damage);
    me.discard.push_back(oldId);
  }
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  st.turnActionCount += 1;
  discard_pending_trainer(st);
}

static void discard_pending_trainer_if_any(GameState& st) {
  discard_pending_trainer(st);
}

static void finish_bother_bot(GameState& st) {
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  discard_pending_trainer_if_any(st);
}

static int hand_trimmer_needed(const Player& p) {
  return std::max(0, p.handCount - 5);
}

static void set_hand_trimmer_pending(GameState& st, int side) {
  Player& who = st.players[side];
  PendingDecision pd;
  pd.context = 8;  // DISCARD.
  pd.minCount = hand_trimmer_needed(who);
  pd.maxCount = pd.minCount;
  int n = static_cast<int>(who.hand.size()) == who.handCount
              ? static_cast<int>(who.hand.size())
              : who.handCount;
  for (int i = 0; i < n; ++i) {
    Atom card = i < static_cast<int>(who.hand.size()) ? Atom::I(who.hand[i])
                                                       : Atom::N();
    pd.options.push_back({Atom::S("CARD"), Atom::S("HAND"), Atom::I(i), card});
  }
  st.yourIndex = side;
  st.pending = pd;
}

static int discard_selected_hand_cards(Player& who, const PendingDecision& pd,
                                       const std::vector<int>& sel) {
  std::vector<int> refs;
  for (int s : sel) {
    if (s < 0 || s >= static_cast<int>(pd.options.size())) continue;
    refs.push_back(static_cast<int>(pd.options[s][2].i));
  }
  std::sort(refs.rbegin(), refs.rend());
  refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
  int discarded = 0;
  bool known = static_cast<int>(who.hand.size()) == who.handCount;
  for (int idx : refs) {
    if (idx < 0 || idx >= who.handCount) continue;
    if (known && idx < static_cast<int>(who.hand.size())) {
      who.discard.push_back(who.hand[idx]);
      who.hand.erase(who.hand.begin() + idx);
    } else {
      who.hand.clear();
      who.handKnown = false;
    }
    who.handCount -= 1;
    ++discarded;
  }
  return discarded;
}

static void finish_hand_trimmer(GameState& st) {
  int actor = st.effectStack.empty() ? st.yourIndex : st.effectStack.back().a;
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  st.yourIndex = actor;
  discard_pending_trainer_if_any(st);
}

static void advance_hand_trimmer(GameState& st, EffectFrame& fr) {
  int side = fr.phase == 0 ? (1 - fr.a) : fr.a;
  if (hand_trimmer_needed(st.players[side]) > 0) {
    set_hand_trimmer_pending(st, side);
    return;
  }
  if (fr.phase == 0) {
    fr.phase = 1;
    advance_hand_trimmer(st, fr);
    return;
  }
  finish_hand_trimmer(st);
}

static void play_hand_trimmer(GameState& st) {
  int actor = st.yourIndex;
  Player& me = st.players[actor];
  remove_hand_card(st, me, 1087);
  st.turnActionCount += 1;
  st.pendingTrainerDiscard = 1087;
  st.pendingTrainerOwner = actor;

  EffectFrame fr;
  fr.effect = EFF_HAND_TRIMMER;
  fr.a = actor;
  fr.phase = 0;
  st.effectStack.push_back(fr);
  advance_hand_trimmer(st, st.effectStack.back());
}

static void resolve_hand_trimmer(GameState& st, const std::vector<int>& sel) {
  if (st.effectStack.empty()) return;
  EffectFrame& fr = st.effectStack.back();
  int side = fr.phase == 0 ? (1 - fr.a) : fr.a;
  PendingDecision pd = st.pending;
  int discarded = discard_selected_hand_cards(st.players[side], pd, sel);
  st.discardedCount = discarded;
  st.lastEffectCount = discarded;
  st.turnActionCount += 1;
  st.pending = PendingDecision();
  if (fr.phase == 0)
    fr.phase = 1;
  advance_hand_trimmer(st, fr);
}

static void set_bother_bot_prize_pending(GameState& st, EffectFrame& fr) {
  Player& opp = st.players[1 - fr.a];
  PendingDecision pd;
  pd.context = 43;  // ACTIVATE.
  pd.minCount = opp.prizeCount > 0 ? 1 : 0;
  pd.maxCount = pd.minCount;
  for (int i = 0; i < opp.prizeCount; ++i) {
    if (i < static_cast<int>(opp.prizeFaceUp.size()) && opp.prizeFaceUp[i])
      continue;
    // The physical slot is selectable, but its face-down identity is not
    // public and must never enter a cached action descriptor.
    Atom card = Atom::N();
    pd.options.push_back({Atom::S("CARD"), Atom::S("PRIZE"),
                          Atom::I(i), card});
  }
  st.pending = pd;
}

static void set_bother_bot_yesno_pending(GameState& st) {
  PendingDecision pd;
  pd.context = 43;  // ACTIVATE.
  pd.minCount = 1;
  pd.maxCount = 1;
  pd.options.push_back({Atom::S("YES")});
  pd.options.push_back({Atom::S("NO")});
  st.pending = pd;
}

static void play_bother_bot(GameState& st) {
  Player& me = st.players[st.yourIndex];
  remove_hand_card(st, me, 1131);
  st.turnActionCount += 1;
  me.discard.push_back(1131);

  EffectFrame fr;
  fr.effect = EFF_BOTHER_BOT;
  fr.a = st.yourIndex;
  fr.phase = 0;
  st.effectStack.push_back(fr);
  set_bother_bot_prize_pending(st, st.effectStack.back());
  if (st.has_pending())
    st.turnActionCount += 1;
  if (st.pending.options.empty()) finish_bother_bot(st);
}

static void resolve_bother_bot(GameState& st, const std::vector<int>& sel) {
  if (st.effectStack.empty()) return;
  EffectFrame& fr = st.effectStack.back();
  Player& opp = st.players[1 - fr.a];
  if (fr.phase == 0) {
    int prizeIdx = -1;
    if (!sel.empty() && sel[0] >= 0 &&
        sel[0] < static_cast<int>(st.pending.options.size()))
      prizeIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
    bool knownHand = static_cast<int>(opp.hand.size()) == opp.handCount;
    if (prizeIdx < 0 || prizeIdx >= opp.prizeCount || opp.handCount <= 0 ||
        !knownHand || opp.hand.empty()) {
      finish_bother_bot(st);
      return;
    }
    fr.savedSrc = prizeIdx;
    if (static_cast<int>(opp.prizeFaceUp.size()) < opp.prizeCount)
      opp.prizeFaceUp.resize(opp.prizeCount, false);
    if (prizeIdx >= 0 && prizeIdx < static_cast<int>(opp.prizeFaceUp.size()))
      opp.prizeFaceUp[prizeIdx] = true;
    fr.sourceCardId = static_cast<int>(next_rand(st.rng) % opp.hand.size());
    fr.phase = 1;
    st.pending = PendingDecision();
    st.turnActionCount += 1;
    set_bother_bot_yesno_pending(st);
    return;
  }

  bool yes = !sel.empty() && sel[0] >= 0 &&
             sel[0] < static_cast<int>(st.pending.options.size()) &&
             atom_is(st.pending.options[sel[0]][0], "YES");
  int prizeIdx = fr.savedSrc;
  int handIdx = fr.sourceCardId;
  if (yes && prizeIdx >= 0 && prizeIdx < static_cast<int>(opp.prizes.size()) &&
      handIdx >= 0 && handIdx < static_cast<int>(opp.hand.size()))
    std::swap(opp.prizes[prizeIdx], opp.hand[handIdx]);
  finish_bother_bot(st);
}

static bool no_rule_box_pokemon_card(const CardInfo* c) {
  return c && c->cardType == POKEMON && !(c->ex || c->megaEx || c->tera);
}

static void start_slowking_seek_inspiration(GameState& st) {
  int actor = st.yourIndex;
  Player& p = st.players[actor];
  int topCard = 0;
  if (p.deckCount > 0) {
    topCard = discard_deck_top(p);
    if (topCard)
      p.discard.push_back(topCard);
    else
      p.handKnown = false;
  }
  const CardInfo* c = topCard ? find_card(topCard) : nullptr;
  if (!no_rule_box_pokemon_card(c) || c->n_attacks <= 0) {
    post_attack(st, actor);
    return;
  }

  EffectFrame fr;
  fr.effect = EFF_SLOWKING_SEEK_INSPIRATION;
  fr.a = actor;
  fr.attackId = 213;
  fr.attackCardId = p.activeKnown ? p.active.id : 0;
  fr.sourceCardId = topCard;
  st.effectStack.push_back(fr);

  PendingDecision pd;
  pd.context = 35;  // Choose copied attack.
  pd.minCount = 1;
  pd.maxCount = 1;
  for (int i = 0; i < c->n_attacks; ++i) {
    const AttackInfo& at = c->attacks[i];
    pd.options.push_back({Atom::S("ATTACK"), Atom::I(at.id)});
  }
  st.pending = pd;
}

static void resolve_slowking_seek_inspiration(GameState& st,
                                              const std::vector<int>& sel) {
  if (st.effectStack.empty()) return;
  EffectFrame fr = st.effectStack.back();
  int attackId = 0;
  int copiedDamage = -1;
  if (!sel.empty() && sel[0] >= 0 &&
      sel[0] < static_cast<int>(st.pending.options.size())) {
    const Descriptor& d = st.pending.options[sel[0]];
    attackId = static_cast<int>(d[1].i);
    if (d.size() > 2) {
      copiedDamage = copied_attack_damage(static_cast<int>(d[2].i));
    } else if (const AttackInfo* at = card_attack(fr.sourceCardId, attackId)) {
      copiedDamage = at->damage;
    }
  }
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  if (attackId <= 0 || copiedDamage < 0) {
    post_attack(st, fr.a);
    return;
  }

  int prog = attack_program(attackId);
  EffectFrame nested;
  nested.effect = FLOW_PROGRAM;
  nested.program = (prog >= 0) ? prog : vanilla_attack_program();
  nested.a = fr.a;
  nested.attackId = attackId;
  nested.attackCardId = fr.attackCardId;
  nested.copiedAttackBaseDamage = copiedDamage;
  st.effectStack.push_back(nested);
  run_program(st, {});
}

static void start_ninetales_shapeshifter(GameState& st) {
  int actor = st.yourIndex;
  Player& p = st.players[actor];
  int topCard = 0;
  if (p.deckCount > 0) {
    topCard = discard_deck_top(p);
    if (topCard)
      p.discard.push_back(topCard);
    else
      p.handKnown = false;
  }
  const CardInfo* c = topCard ? find_card(topCard) : nullptr;
  int prog = (c && c->cardType == SUPPORTER) ? card_program(topCard) : -1;
  if (prog < 0) {
    post_attack(st, actor);
    return;
  }

  EffectFrame fr;
  fr.effect = FLOW_PROGRAM;
  fr.program = prog;
  fr.a = actor;
  fr.attackId = 955;
  fr.attackCardId = p.activeKnown ? p.active.id : 0;
  fr.sourceCardId = topCard;
  st.effectStack.push_back(fr);
  run_program(st, {});
}

static void add_stack_card(Player& owner, int id, int dest, int& moved) {
  if (id == 0) return;
  if (dest == D_HAND) {
    owner.hand.push_back(id);
    owner.handCount += 1;
  } else if (dest == D_DECK || dest == D_DECK_BOTTOM) {
    if (dest == D_DECK_BOTTOM)
      push_deck_bottom(owner, id, true);
    else
      push_deck_top(owner, id, true);
  } else {
    owner.discard.push_back(id);
  }
  moved += 1;
}

static void add_inplay_stack(Player& owner, const InPlay& p, int dest,
                             int& moved) {
  add_stack_card(owner, p.id, dest, moved);
  for (int id : p.preEvo) add_stack_card(owner, id, dest, moved);
  for (int id : p.energyCardIds) add_stack_card(owner, id, dest, moved);
  for (int id : p.tools) add_stack_card(owner, id, dest, moved);
}

static bool side_has_zero_hp_pokemon(const Player& p) {
  if (p.activeKnown && p.active.hp <= 0)
    return true;
  for (const InPlay& b : p.bench)
    if (b.hp <= 0)
      return true;
  return false;
}

static int move_inplay_stack(GameState& st, int ownerIdx, int inplayIdx,
                             int dest, bool shuffleDeck, bool& needsPromote,
                             bool deferActiveOutcome = false) {
  needsPromote = false;
  if (dest == D_HAND && in_play_has(st.players[1 - ownerIdx], 102))
    return 0;  // Milotic: opponent's in-play Pokemon can't go to their hand.
  Player& owner = st.players[ownerIdx];
  int moved = 0;
  if (inplayIdx < 0) {
    if (!owner.activeKnown) return 0;
    InPlay leaving = owner.active;
    add_inplay_stack(owner, leaving, dest, moved);
    owner.active = InPlay();
    owner.activePresent = false;
    owner.activeKnown = false;
    clear_status(owner);
    if (owner.bench.empty() && !deferActiveOutcome)
      set_result(st, 1 - ownerIdx);
    else if (!owner.bench.empty())
      needsPromote = true;
  } else if (inplayIdx < static_cast<int>(owner.bench.size())) {
    InPlay leaving = owner.bench[inplayIdx];
    add_inplay_stack(owner, leaving, dest, moved);
    owner.bench.erase(owner.bench.begin() + inplayIdx);
  }
  if (shuffleDeck && moved > 0) shuffle_deck_known(owner, st.rng);
  return moved;
}

static void clear_attack_effects_for_evolve(InPlay& p) {
  p.lockId = 0;
  p.lockTurn = 0;
  p.activeLockId = 0;
  p.noAttackTurn = -1;
  p.noRetreatTurn = -1;
  p.dmgReduce = 0;
  p.dmgReduceTurn = -1;
  p.attackCostMod = 0;
  p.attackCostModTurn = -1;
  p.retreatCostMod = 0;
  p.retreatCostModTurn = -1;
  p.delayedDamageTurn = -1;
  p.delayedDamageCounters = 0;
  p.delayedKoTurn = -1;
  p.delayedKoPromoteBeforePrize = false;
  p.preventDmgTurn = -1;
  p.preventDmgCond = 0;
  p.preventDmgValue = 0;
  p.preventEffectsTurn = -1;
  p.preventEffectsCond = 0;
  p.preventEffectsValue = 0;
  p.attackFlipFailTurn = -1;
  p.noWeaknessTurn = -1;
  p.takeMoreDamageTurn = -1;
  p.takeMoreDamage = 0;
  p.nextAttackBonusId = 0;
  p.nextAttackBonusTurn = -1;
  p.nextAttackBonus = 0;
  p.nextAttackSetBase = -1;
  p.damagedByAttackCountersTurn = -1;
  p.damagedByAttackCounters = 0;
  p.damagedByAttackStatus = -1;
  p.damagedByAttackEqualCountersTurn = -1;
  p.damagedByAttackTurn = -1;
  p.damagedByAttackSide = -1;
  p.damagedByAttackAmount = 0;
  p.damagedByAttackBeforeHp = -1;
  p.energyAttachCountersTurn = -1;
  p.energyAttachCounters = 0;
  p.energyAttachCountersFromHandOnly = 0;
  p.attackDmgReduce = 0;
  p.attackDmgReduceTurn = -1;
  p.attackBonus = 0;
  p.attackBonusTurn = -1;
}

static void clear_status_for_evolve(GameState& st, int side, bool activeSpot,
                                    InPlay* evolved) {
  if (evolved) clear_attack_effects_for_evolve(*evolved);
  if (!activeSpot) return;
  Player& p = st.players[side];
  bool keepConfused = !st.stadium.empty() && st.stadium[0] == 1265 && p.confused;
  clear_status(p);
  if (keepConfused) p.confused = true;
}

static int devolve_one(GameState& st, int ownerIdx, InPlay& p, int dest,
                       int count, bool markNoEvolve, bool activeSpot) {
  Player& owner = st.players[ownerIdx];
  int moved = 0;
  int n = (count < 0) ? 1 : count;
  if (n <= 0) return 0;
  while (n-- > 0 && !p.preEvo.empty()) {
    int oldId = p.id;
    int damage = p.maxHp - p.hp;
    add_stack_card(owner, oldId, dest, moved);
    p.id = p.preEvo.back();
    p.preEvo.pop_back();
    p.maxHp = effective_max_hp(p.id, p.tools, st, p.energyCardIds, p.energies,
                               ownerIdx);
    p.hp = p.maxHp - damage;
    if (p.hp < 0) p.hp = 0;
  }
  if (moved > 0) {
    clear_status_for_evolve(st, ownerIdx, activeSpot, &p);
    if (markNoEvolve) p.noEvolveTurn = st.turn;
  }
  return moved;
}

static int devolve_selected(GameState& st, EffectFrame& fr, int zone, int dest,
                            int count, bool markNoEvolve) {
  Player& owner = (zone == Z_OWN_INPLAY || zone == Z_OWN_BENCH)
                      ? st.players[fr.a]
                      : st.players[1 - fr.a];
  int ownerIdx = (zone == Z_OWN_INPLAY || zone == Z_OWN_BENCH) ? fr.a : 1 - fr.a;
  int moved = 0;
  const SmallVec<int, 8>& refs =
      (fr.savedPhase == zone) ? fr.savedScratch : fr.scratch;
  for (int idx : refs) {
    if ((zone == Z_OWN_BENCH || zone == Z_OPP_BENCH) && idx < 0) continue;
    InPlay* p = inplay_ref(owner, idx);
    if (!p) continue;
    moved += devolve_one(st, ownerIdx, *p, dest, count, markNoEvolve, idx < 0);
  }
  if (dest == D_DECK && moved > 0) shuffle_deck_known(owner, st.rng);
  return moved;
}

static int devolve_all_matching(GameState& st, EffectFrame& fr, int zone,
                                int filter, int dest, int count) {
  Player& owner = (zone == Z_OWN_INPLAY || zone == Z_OWN_BENCH)
                      ? st.players[fr.a]
                      : st.players[1 - fr.a];
  int ownerIdx = (zone == Z_OWN_INPLAY || zone == Z_OWN_BENCH) ? fr.a : 1 - fr.a;
  int moved = 0;
  if (zone == Z_OWN_INPLAY || zone == Z_OPP_INPLAY) {
    if (owner.activeKnown && matches_dynamic_inplay_filter(st, fr, owner.active, filter))
      moved += devolve_one(st, ownerIdx, owner.active, dest, count, false, true);
  }
  if (zone == Z_OWN_INPLAY || zone == Z_OPP_INPLAY ||
      zone == Z_OWN_BENCH || zone == Z_OPP_BENCH) {
    for (auto& b : owner.bench)
      if (matches_dynamic_inplay_filter(st, fr, b, filter))
        moved += devolve_one(st, ownerIdx, b, dest, count, false, false);
  }
  if (dest == D_DECK && moved > 0) shuffle_deck_known(owner, st.rng);
  return moved;
}

static int return_inplay_pokemon_to_hand_discard_attachments(
    GameState& st, int ownerIdx, int inplayIdx, bool& needsPromote,
    bool deferActiveOutcome = false) {
  needsPromote = false;
  if (in_play_has(st.players[1 - ownerIdx], 102))
    return 0;  // Milotic blocks the Pokemon cards from returning to hand.
  Player& owner = st.players[ownerIdx];
  auto move_pokemon_cards = [&](const InPlay& p) {
    int moved = 0;
    add_stack_card(owner, p.id, D_HAND, moved);
    for (int id : p.preEvo) add_stack_card(owner, id, D_HAND, moved);
    for (int id : p.energyCardIds) add_stack_card(owner, id, D_DISCARD, moved);
    for (int id : p.tools) add_stack_card(owner, id, D_DISCARD, moved);
    return moved;
  };
  int moved = 0;
  if (inplayIdx < 0) {
    if (!owner.activeKnown) return 0;
    InPlay leaving = owner.active;
    moved = move_pokemon_cards(leaving);
    owner.active = InPlay();
    owner.activePresent = false;
    owner.activeKnown = false;
    clear_status(owner);
    if (owner.bench.empty() && !deferActiveOutcome)
      set_result(st, 1 - ownerIdx);
    else if (!owner.bench.empty())
      needsPromote = true;
  } else if (inplayIdx < static_cast<int>(owner.bench.size())) {
    InPlay leaving = owner.bench[inplayIdx];
    moved = move_pokemon_cards(leaving);
    owner.bench.erase(owner.bench.begin() + inplayIdx);
  }
  return moved;
}

static int attach_one_basic_energy_from(GameState& st, Player& owner,
                                        SmallVec<int, 64>& src,
                                        InPlay& target, int etype) {
  for (int i = static_cast<int>(src.size()) - 1; i >= 0; --i) {
    const CardInfo* c = find_card(src[i]);
    if (c && c->cardType == BASIC_ENERGY && (etype < 0 || c->energyType == etype)) {
      target.energies.push_back(c->energyType);
      push_attached_energy_card(target, src[i]);
      apply_energy_attach_reactive(st, target, false);
      adjust_card_zone_count(owner, &src, -1);
      src.erase(src.begin() + i);
      return 1;
    }
  }
  return 0;
}

static int attach_basic_energy_each_target(GameState& st, EffectFrame& fr,
                                           int source, int etype, int filt,
                                           int mode) {
  Player& me = st.players[fr.a];
  SmallVec<int, 64>& src = (source == AZ_HAND)      ? me.hand
                          : (source == AZ_DISCARD) ? me.discard
                                                   : me.deck;
  std::vector<InPlay*> targets;
  auto add_if = [&](InPlay& p) {
    if (p.id != 0 && (filt < 0 || matches_filter(p.id, filt))) targets.push_back(&p);
  };
  if (mode == AET_CHOSEN) {
    for (int idx : fr.scratch) {
      InPlay* p = inplay_ref(me, idx);
      if (p) add_if(*p);
    }
  } else {
    if (mode == AET_OWN_INPLAY && me.activeKnown) add_if(me.active);
    for (auto& b : me.bench) add_if(b);
  }
  int attached = 0;
  for (InPlay* p : targets)
    attached += attach_one_basic_energy_from(st, me, src, *p, etype);
  if (source == AZ_DECK) shuffle_deck_known(me, st.rng);
  return attached;
}

static bool basic_energy_matches_type(int cardId, int etype) {
  const CardInfo* c = find_card(cardId);
  return c && c->cardType == BASIC_ENERGY &&
         (etype < 0 || c->energyType == etype);
}

static InPlay* each_attach_target_ref(Player& me, EffectFrame& fr,
                                      int mode, int filt) {
  auto matches = [&](InPlay& p) {
    return p.id != 0 && (filt < 0 || matches_filter(p.id, filt));
  };
  if (mode == AET_CHOSEN) {
    while (fr.phase < static_cast<int>(fr.scratch.size())) {
      InPlay* p = inplay_ref(me, fr.scratch[fr.phase]);
      if (p && matches(*p)) return p;
      fr.phase += 1;
    }
    return nullptr;
  }
  if (mode == AET_OWN_INPLAY) {
    while (fr.phase <= static_cast<int>(me.bench.size())) {
      if (fr.phase == 0) {
        if (me.activeKnown && matches(me.active)) return &me.active;
      } else {
        int benchIdx = fr.phase - 1;
        if (benchIdx >= 0 && benchIdx < static_cast<int>(me.bench.size()) &&
            matches(me.bench[benchIdx]))
          return &me.bench[benchIdx];
      }
      fr.phase += 1;
    }
    return nullptr;
  }
  while (fr.phase < static_cast<int>(me.bench.size())) {
    if (matches(me.bench[fr.phase])) return &me.bench[fr.phase];
    fr.phase += 1;
  }
  return nullptr;
}

static bool set_attach_basic_energy_each_target_pending(GameState& st,
                                                        EffectFrame& fr,
                                                        int source,
                                                        int etype,
                                                        int filt,
                                                        int mode) {
  Player& me = st.players[fr.a];
  InPlay* target = each_attach_target_ref(me, fr, mode, filt);
  if (!target) {
    if (fr.savedSrc == AZ_DECK)
      shuffle_deck_known(me, st.rng);
    st.lastEffectCount = std::max(0, fr.loopRemain);
    fr.loopRemain = -1;
    fr.savedSrc = -2;
    fr.phase = 0;
    return false;
  }

  SmallVec<int, 64>& src = (source == AZ_HAND)      ? me.hand
                          : (source == AZ_DISCARD) ? me.discard
                                                   : me.deck;
  if (source == AZ_DECK)
    mark_full_deck_inspected(me);
  const char* label = (source == AZ_HAND)      ? "HAND"
                      : (source == AZ_DISCARD) ? "DISCARD"
                                               : "DECK";
  PendingDecision pd;
  pd.context = 22;  // CTX_ATTACH_TO
  pd.minCount = source == AZ_DECK ? 0 : 1;
  pd.maxCount = 1;
  for (int i = 0; i < static_cast<int>(src.size()); ++i) {
    if (basic_energy_matches_type(src[i], etype))
      pd.options.push_back({Atom::S("CARD"), Atom::S(label), Atom::I(i),
                            Atom::I(src[i])});
  }
  if (pd.options.empty()) {
    if (fr.savedSrc == AZ_DECK)
      shuffle_deck_known(me, st.rng);
    st.lastEffectCount = std::max(0, fr.loopRemain);
    fr.loopRemain = -1;
    fr.savedSrc = -2;
    fr.phase = 0;
    return false;
  }
  pd.maxCount = std::min(pd.maxCount, static_cast<int>(pd.options.size()));
  pd.minCount = std::min(pd.minCount, pd.maxCount);
  st.pending = pd;
  fr.savedSrc = source;
  return true;
}

static int resolve_attach_basic_energy_each_target_choice(
    GameState& st, EffectFrame& fr, const std::vector<int>& sel,
    int source, int etype, int filt, int mode) {
  Player& me = st.players[fr.a];
  InPlay* target = each_attach_target_ref(me, fr, mode, filt);
  int attached = 0;
  if (fr.loopRemain < 0) fr.loopRemain = 0;
  if (target) {
    SmallVec<int, 64>& src = (source == AZ_HAND)      ? me.hand
                            : (source == AZ_DISCARD) ? me.discard
                                                     : me.deck;
    std::vector<int> refs;
    for (int s : sel) {
      if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
      refs.push_back(static_cast<int>(st.pending.options[s][2].i));
    }
    std::sort(refs.rbegin(), refs.rend());
    refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
    for (int idx : refs) {
      if (idx < 0 || idx >= static_cast<int>(src.size())) continue;
      int id = src[idx];
      if (!basic_energy_matches_type(id, etype)) continue;
      const CardInfo* c = find_card(id);
      target->energies.push_back(c->energyType);
      push_attached_energy_card(*target, id);
      apply_energy_attach_reactive(st, *target, false);
      adjust_card_zone_count(me, &src, -1);
      src.erase(src.begin() + idx);
      attached += 1;
    }
  }
  fr.loopRemain += attached;
  fr.phase += 1;
  return attached;
}

static void refresh_energy_units_from_attached_cards(GameState& st, int ownerSide,
                                                     InPlay& p) {
  InPlay cardsOnly = p;
  cardsOnly.energies.clear();
  p.energies = provided_energy_units(cardsOnly, &st, ownerSide);
}

static int discard_special_energy_from(GameState& st, int ownerSide, InPlay& p) {
  Player& owner = st.players[ownerSide];
  int discarded = 0;
  for (int k = static_cast<int>(p.energyCardIds.size()) - 1; k >= 0; --k) {
    int id = p.energyCardIds[k];
    const CardInfo* c = find_card(id);
    if (!c || c->cardType != SPECIAL_ENERGY) continue;
    owner.discard.push_back(id);
    erase_attached_energy_card(p, k);
    ++discarded;
  }
  if (discarded > 0) {
    refresh_energy_units_from_attached_cards(st, ownerSide, p);
    refresh_inplay_max_hp(st, p, ownerSide);  // a discarded Special Energy may
                                              // have granted +maxHp (e.g. +20 {G})
  }
  return discarded;
}

static int discard_opp_tools_specials(GameState& st, bool includeStadium) {
  Player& opp = st.players[1 - st.effectStack.back().a];
  int oppSide = 1 - st.effectStack.back().a;
  int discarded = 0;
  if (opp.activeKnown && !attack_effects_prevented(st, oppSide, opp.active)) {
    discarded += discard_tools_from_inplay(st, opp, opp.active, -1);
    discarded += discard_special_energy_from(st, oppSide, opp.active);
  }
  for (auto& b : opp.bench) {
    if (attack_effects_prevented(st, oppSide, b)) continue;
    discarded += discard_tools_from_inplay(st, opp, b, -1);
    discarded += discard_special_energy_from(st, oppSide, b);
  }
  if (includeStadium && !st.stadium.empty()) {
    int oldStadium = cur_stadium(st);
    st.players[st.stadiumOwner >= 0 ? st.stadiumOwner : st.effectStack.back().a]
        .discard.push_back(st.stadium[0]);
    st.stadium.clear();
    st.stadiumOwner = -1;
    st.stadiumAbilityUsed = false;
    apply_stadium_hp_change(st, oldStadium, -1);
    enforce_area_zero_bench_limits(st);
    enforce_rotom_tool_limits(st);
    ++discarded;
  }
  return discarded;
}

static int shuffle_opp_bench_except_chosen(GameState& st, EffectFrame& fr) {
  Player& opp = st.players[1 - fr.a];
  std::vector<int> keep = fr.scratch;
  int movedPokemon = 0;
  for (int j = static_cast<int>(opp.bench.size()) - 1; j >= 0; --j) {
    if (std::find(keep.begin(), keep.end(), j) != keep.end()) continue;
    if (attack_effects_prevented(st, 1 - fr.a, opp.bench[j])) continue;
    InPlay p = opp.bench[j];
    push_deck_top(opp, p.id, true);
    for (int id : p.preEvo) push_deck_top(opp, id, true);
    for (int id : p.energyCardIds) push_deck_top(opp, id, true);
    for (int id : p.tools) push_deck_top(opp, id, true);
    opp.bench.erase(opp.bench.begin() + j);
    ++movedPokemon;
  }
  if (movedPokemon > 0) shuffle_deck_known(opp, st.rng);
  return movedPokemon;
}

static int encode_tool_ref(int inplayIdx, int toolIdx) {
  return 100000 + (inplayIdx + 1) * 1000 + toolIdx;
}

static int encode_any_tool_ref(int ownerOffset, int inplayIdx, int toolIdx) {
  return 300000 + ownerOffset * 100000 + (inplayIdx + 1) * 1000 + toolIdx;
}

static int any_tool_owner_offset(int ref) {
  return (ref - 300000) / 100000;
}

static int any_tool_inplay_idx(int ref) {
  int local = (ref - 300000) % 100000;
  return local / 1000 - 1;
}

static int any_tool_tool_idx(int ref) {
  int local = (ref - 300000) % 100000;
  return local % 1000;
}

static int encode_copied_attack_ref(int attackId, int damage) {
  return attackId * 1000 + std::max(0, damage);
}

static int copied_attack_id(int ref) {
  return ref / 1000;
}

static int copied_attack_damage(int ref) {
  return ref % 1000;
}

static int encode_special_energy_choice_ref(int inplayIdx, int energyIdx) {
  return 200000 + encode_energy_ref(inplayIdx, energyIdx);
}

static void add_attached_or_stadium_options(PendingDecision& pd, const GameState& st,
                                            int actor) {
  const Player& opp = st.players[1 - actor];
  if (!st.stadium.empty())
    pd.options.push_back({Atom::S("CARD"), Atom::S("STADIUM"), Atom::I(-1),
                          Atom::I(st.stadium[0])});
  auto add_for = [&](const InPlay& p, const char* area, int inplayIdx) {
    for (int i = 0; i < static_cast<int>(p.tools.size()); ++i)
      pd.options.push_back({Atom::S("CARD"), Atom::S(area),
                            Atom::I(encode_tool_ref(inplayIdx, i)),
                            Atom::I(p.tools[i])});
    for (int i = 0; i < static_cast<int>(p.energies.size()); ++i) {
      int id = (i < static_cast<int>(p.energyCardIds.size())) ? p.energyCardIds[i] : 0;
      const CardInfo* c = find_card(id);
      if (c && c->cardType == SPECIAL_ENERGY)
        pd.options.push_back({Atom::S("ENERGY"), Atom::S(area),
                              Atom::I(encode_special_energy_choice_ref(inplayIdx, i)),
                              Atom::I(id)});
    }
  };
  if (opp.activeKnown) add_for(opp.active, "ACTIVE", -1);
  for (int j = 0; j < static_cast<int>(opp.bench.size()); ++j)
    add_for(opp.bench[j], "BENCH", j);
}

static int discard_attached_or_stadium_choice(GameState& st, EffectFrame& fr) {
  if (fr.scratch.empty()) return 0;
  int ref = fr.scratch[0];
  if (ref == -1) {
    if (st.stadium.empty()) return 0;
    int oldStadium = cur_stadium(st);
    int owner = st.stadiumOwner >= 0 ? st.stadiumOwner : fr.a;
    st.players[owner].discard.push_back(st.stadium[0]);
    st.stadium.clear();
  st.stadiumOwner = -1;
  st.stadiumAbilityUsed = false;
  apply_stadium_hp_change(st, oldStadium, -1);
  enforce_area_zero_bench_limits(st);
  enforce_rotom_tool_limits(st);
  return 1;
  }
  Player& opp = st.players[1 - fr.a];
  if (ref >= 200000) {
    int eRef = ref - 200000;
    InPlay* p = inplay_ref(opp, energy_ref_inplay(eRef));
    int eIdx = energy_ref_index(eRef);
    if (!p || eIdx < 0 || eIdx >= static_cast<int>(p->energies.size())) return 0;
    if (attack_effects_prevented(st, 1 - fr.a, *p)) return 0;
    int id = (eIdx < static_cast<int>(p->energyCardIds.size())) ? p->energyCardIds[eIdx] : 0;
    const CardInfo* c = find_card(id);
    if (!c || c->cardType != SPECIAL_ENERGY) return 0;
    opp.discard.push_back(id);
    erase_attached_energy_card(*p, eIdx);
    p->energies.erase(p->energies.begin() + eIdx);
    return 1;
  }
  if (ref >= 100000) {
    int tRef = ref - 100000;
    int inplayIdx = tRef / 1000 - 1;
    int toolIdx = tRef % 1000;
    InPlay* p = inplay_ref(opp, inplayIdx);
    if (!p || toolIdx < 0 || toolIdx >= static_cast<int>(p->tools.size())) return 0;
    if (attack_effects_prevented(st, 1 - fr.a, *p)) return 0;
    int damage = p->maxHp - p->hp;
    opp.discard.push_back(p->tools[toolIdx]);
    erase_tool_at(*p, toolIdx);
    p->maxHp = effective_max_hp(p->id, p->tools, st,
                                p->energyCardIds, p->energies, 1 - fr.a);
    p->hp = std::max(0, p->maxHp - damage);
    return 1;
  }
  return 0;
}

static bool basic_energy_type(int id, int& etype) {
  const CardInfo* c = find_card(id);
  if (!c || c->cardType != BASIC_ENERGY) return false;
  etype = c->energyType;
  return true;
}

static void add_crispin_energy_options(PendingDecision& pd, const Player& owner,
                                       int stage, int handEnergyType) {
  for (int i = 0; i < static_cast<int>(owner.deck.size()); ++i) {
    int etype = -1;
    if (!basic_energy_type(owner.deck[i], etype)) continue;
    if (stage == 1 && (handEnergyType < 0 || etype == handEnergyType))
      continue;
    pd.options.push_back({Atom::S("CARD"), Atom::S("DECK"),
                          Atom::I(i), Atom::I(owner.deck[i])});
  }
}

static int take_crispin_hand_energy_choice(GameState& st, EffectFrame& fr) {
  if (fr.scratch.empty()) return 0;
  Player& me = st.players[fr.a];
  int handIdx = fr.scratch[0];
  if (handIdx < 0 || handIdx >= static_cast<int>(me.deck.size())) return 0;

  int handType = -1;
  int handId = me.deck[handIdx];
  if (!basic_energy_type(handId, handType)) return 0;

  bool known = false;
  erase_deck_at(me, handIdx, &known);
  me.hand.push_back(handId);
  me.handCount += 1;
  if (known && !me.handKnown) add_known_card(me.handKnownCards, handId);
  fr.savedSrc = handType;
  return 1;
}

static int take_crispin_attach_energy_choice(GameState& st, EffectFrame& fr) {
  int handType = fr.savedSrc;
  fr.savedSrc = -2;
  if (fr.scratch.empty()) return 0;
  Player& me = st.players[fr.a];
  int attachIdx = fr.scratch[0];
  if (attachIdx < 0 || attachIdx >= static_cast<int>(me.deck.size())) return 0;

  int attachType = -1;
  int attachId = me.deck[attachIdx];
  if (!basic_energy_type(attachId, attachType) || attachType == handType)
    return 0;

  erase_deck_at(me, attachIdx);
  fr.savedSrc = attachId;
  return 1;
}

static int take_crispin_energy_choice(GameState& st, EffectFrame& fr,
                                      int stage) {
  return stage == 1 ? take_crispin_attach_energy_choice(st, fr)
                    : take_crispin_hand_energy_choice(st, fr);
}

static int attach_crispin_saved_energy(GameState& st, EffectFrame& fr) {
  int cardId = fr.savedSrc;
  fr.savedSrc = -2;
  if (cardId <= 0 || fr.scratch.empty()) return 0;
  int etype = -1;
  if (!basic_energy_type(cardId, etype)) return 0;
  Player& me = st.players[fr.a];
  InPlay* target = inplay_ref(me, fr.scratch[0]);
  if (!target) return 0;
  target->energies.push_back(etype);
  push_attached_energy_card(*target, cardId);
  apply_energy_attach_reactive(st, *target, false);
  return 1;
}

static int saved_source_inplay_idx(const EffectFrame& fr);

struct EnergyOptionCandidate {
  int order = -1;
  int fallback = 0;
  Descriptor desc;
};

static void append_energy_options(PendingDecision& pd,
                                  std::vector<EnergyOptionCandidate> opts) {
  bool allOrdered = !opts.empty();
  for (const auto& opt : opts)
    if (opt.order < 0) {
      allOrdered = false;
      break;
    }
  if (allOrdered) {
    std::stable_sort(opts.begin(), opts.end(),
                     [](const EnergyOptionCandidate& a,
                        const EnergyOptionCandidate& b) {
                       if (a.order != b.order) return a.order < b.order;
                       return a.fallback < b.fallback;
                     });
  }
  for (auto& opt : opts) pd.options.push_back(std::move(opt.desc));
}

static Descriptor active_energy_desc(const InPlay& p, int k) {
  return {Atom::S("ENERGY"), Atom::S("ACTIVE"),
          Atom::I(encode_energy_ref(-1, k)),
          Atom::I(attached_energy_card_id(p, k))};
}

static Descriptor bench_energy_desc(const InPlay& p, int benchIdx, int k) {
  return {Atom::S("ENERGY"), Atom::S("BENCH"),
          Atom::I(encode_energy_ref(benchIdx, k)), Atom::I(benchIdx),
          Atom::I(attached_energy_card_id(p, k))};
}

static void add_energy_options(PendingDecision& pd, Player& owner, int filter,
                               bool activeOnly, int excludeInplayIdx = -2) {
  auto active_candidate = [&](int k, int fallback,
                              std::vector<EnergyOptionCandidate>& opts) {
    if (excludeInplayIdx == -1 || !owner.activeKnown ||
        k >= attached_energy_slots(owner.active) ||
        !energy_slot_is_card_backed_or_legacy(owner.active, k) ||
        !energy_unit_matches(owner.active, k, filter))
      return;
    opts.push_back({attached_energy_order(owner.active, k), fallback,
                    active_energy_desc(owner.active, k)});
  };
  if (activeOnly) {
    std::vector<EnergyOptionCandidate> opts;
    if (owner.activeKnown) {
      int fallback = 0;
      for (int k = 0; k < attached_energy_slots(owner.active); ++k)
        active_candidate(k, fallback++, opts);
    }
    append_energy_options(pd, std::move(opts));
    return;
  }
  int maxSlots = owner.activeKnown ? attached_energy_slots(owner.active) : 0;
  for (const InPlay& b : owner.bench)
    maxSlots = std::max(maxSlots, attached_energy_slots(b));
  std::vector<EnergyOptionCandidate> opts;
  int fallback = 0;
  for (int k = 0; k < maxSlots; ++k) {
    active_candidate(k, fallback++, opts);
    for (int j = 0; j < static_cast<int>(owner.bench.size()); ++j) {
      if (excludeInplayIdx == j ||
          k >= attached_energy_slots(owner.bench[j]) ||
          !energy_slot_is_card_backed_or_legacy(owner.bench[j], k) ||
          !energy_unit_matches(owner.bench[j], k, filter))
        continue;
      opts.push_back({attached_energy_order(owner.bench[j], k), fallback++,
                      bench_energy_desc(owner.bench[j], j, k)});
    }
  }
  append_energy_options(pd, std::move(opts));
}

static void add_unprevented_opp_energy_options(GameState& st,
                                               PendingDecision& pd,
                                               int ownerSide, int filter,
                                               bool activeOnly,
                                               bool benchOnly = false) {
  Player& owner = st.players[ownerSide];
  std::vector<EnergyOptionCandidate> opts;
  int fallback = 0;
  bool requirePhysicalEnergy =
      !st.effectStack.empty() && st.effectStack.back().attackId == 742;
  auto add_for = [&](const InPlay& p, int inplayIdx) {
    if (attack_effects_prevented(st, ownerSide, p)) return;
    for (int k = 0; k < attached_energy_slots(p); ++k) {
      if (!energy_slot_is_card_backed_or_legacy(p, k)) continue;
      if (requirePhysicalEnergy && attached_energy_card_id(p, k) <= 0) continue;
      if (!energy_unit_matches(p, k, filter)) continue;
      if (inplayIdx < 0) {
        opts.push_back({attached_energy_order(p, k), fallback++,
                        active_energy_desc(p, k)});
      } else {
        opts.push_back({attached_energy_order(p, k), fallback++,
                        bench_energy_desc(p, inplayIdx, k)});
      }
    }
  };
  if (!benchOnly && owner.activeKnown)
    add_for(owner.active, -1);
  if (!activeOnly)
    for (int j = 0; j < static_cast<int>(owner.bench.size()); ++j)
      add_for(owner.bench[j], j);
  append_energy_options(pd, std::move(opts));
}

static void add_bench_energy_options(PendingDecision& pd, Player& owner,
                                     int filter, const InPlay* moveTarget = nullptr,
                                     bool allowRestrictedSpecial = false) {
  std::vector<EnergyOptionCandidate> opts;
  int fallback = 0;
  for (int j = 0; j < static_cast<int>(owner.bench.size()); ++j)
    for (int k = 0; k < attached_energy_slots(owner.bench[j]); ++k)
      if (energy_slot_is_card_backed_or_legacy(owner.bench[j], k) &&
          energy_unit_matches(owner.bench[j], k, filter) &&
          (!moveTarget || energy_slot_can_move_to(owner.bench[j], k, *moveTarget,
                                                  allowRestrictedSpecial)))
        opts.push_back({attached_energy_order(owner.bench[j], k), fallback++,
                        bench_energy_desc(owner.bench[j], j, k)});
  append_energy_options(pd, std::move(opts));
}

static void add_saved_source_energy_options(PendingDecision& pd, Player& owner,
                                            int sourceIdx, int filter) {
  InPlay* src = inplay_ref(owner, sourceIdx);
  if (!src) return;
  for (int k = 0; k < attached_energy_slots(*src); ++k) {
    if (!energy_slot_is_card_backed_or_legacy(*src, k)) continue;
    if (!energy_unit_matches(*src, k, filter)) continue;
    if (sourceIdx < 0) {
      pd.options.push_back(active_energy_desc(*src, k));
    } else {
      pd.options.push_back(bench_energy_desc(*src, sourceIdx, k));
    }
  }
}

static int discard_hand_matching(Player& who, int filter) {
  std::vector<int> idxs;
  for (int i = 0; i < static_cast<int>(who.hand.size()); ++i)
    if (matches_filter(who.hand[i], filter)) idxs.push_back(i);
  std::sort(idxs.rbegin(), idxs.rend());
  int discarded = 0;
  for (int idx : idxs) {
    if (idx < 0 || idx >= static_cast<int>(who.hand.size())) continue;
    who.discard.push_back(who.hand[idx]);
    who.hand.erase(who.hand.begin() + idx);
    who.handCount -= 1;
    ++discarded;
  }
  return discarded;
}

static int discard_hand_to_size(Player& who, int keep) {
  int discarded = 0;
  keep = std::max(0, keep);
  bool known = static_cast<int>(who.hand.size()) == who.handCount;
  while (who.handCount > keep) {
    if (known && !who.hand.empty()) {
      who.discard.push_back(who.hand.back());
      who.hand.pop_back();
    } else {
      who.hand.clear();
      who.handKnown = false;
    }
    who.handCount -= 1;
    ++discarded;
  }
  return discarded;
}

static int random_discard_hand_to_size(GameState& st, Player& who, int keep) {
  int discarded = 0;
  keep = std::max(0, keep);
  bool known = static_cast<int>(who.hand.size()) == who.handCount;
  while (who.handCount > keep) {
    int replayId = 0;
    bool hasReplay = consume_replay_event(
        st, REPLAY_RANDOM_DISCARD_HAND, replayId);
    if (known && !who.hand.empty()) {
      int idx = -1;
      if (hasReplay) {
        auto it = std::find(who.hand.begin(), who.hand.end(), replayId);
        if (it != who.hand.end())
          idx = static_cast<int>(it - who.hand.begin());
      }
      if (idx < 0)
        idx = static_cast<int>(next_rand(st.rng) % who.hand.size());
      who.discard.push_back(who.hand[idx]);
      who.hand.erase(who.hand.begin() + idx);
    } else {
      if (hasReplay) {
        who.discard.push_back(replayId);
        auto it = std::find(who.handKnownCards.begin(),
                            who.handKnownCards.end(), replayId);
        if (it != who.handKnownCards.end())
          who.handKnownCards.erase(it);
      } else {
        who.hand.clear();
        who.handKnown = false;
      }
    }
    who.handCount -= 1;
    ++discarded;
  }
  return discarded;
}

static bool preserves_empty_choice_pending(const GameState& st,
                                           const EffectFrame& fr,
                                           const Op& o);

static bool preserves_unclamped_choice_count(const EffectFrame& fr, const Op& o) {
  if (fr.sourceCardId == 1128 && o.kind == OP_CHOOSE && o.p0 == Z_DECK &&
      o.p2 == 1 && o.p3 == 5)
    return true;
  // These attacks require an exact number of Basic Energy cards from hand. If
  // fewer are available, the attack does nothing rather than choosing all
  // available cards.
  return (fr.attackId == 557 || fr.attackId == 1151) &&
         o.kind == OP_CHOOSE && o.p0 == Z_HAND && o.p4 == CTX_DISCARD;
}

static void add_rare_candy_evolve_options(PendingDecision& pd,
                                          const GameState& st,
                                          const EffectFrame& fr) {
  const Player& me = st.players[fr.a];
  auto add_if_target = [&](int stage2Id, const InPlay& target,
                           const char* area, int displayIdx) {
    if (!stage2_evolves_from_basic(stage2Id, target.id)) return;
    pd.options.push_back({Atom::S("EVOLVE"), Atom::I(stage2Id),
                          Atom::S(area), Atom::I(displayIdx)});
  };
  for (int handIdx = 0; handIdx < static_cast<int>(me.hand.size()); ++handIdx) {
    int stage2Id = me.hand[handIdx];
    if (me.activeKnown &&
        matches_dynamic_inplay_filter(st, fr, me.active,
                                      F_BASIC_WITH_STAGE2_IN_HAND))
      add_if_target(stage2Id, me.active, "ACTIVE", 0);
    for (int j = 0; j < static_cast<int>(me.bench.size()); ++j) {
      if (matches_dynamic_inplay_filter(st, fr, me.bench[j],
                                        F_BASIC_WITH_STAGE2_IN_HAND))
        add_if_target(stage2Id, me.bench[j], "BENCH", j);
    }
  }
}

static int first_bench_index_with_deck_evolution(const Player& p) {
  for (int j = 0; j < static_cast<int>(p.bench.size()); ++j) {
    for (int id : p.deck) {
      if (card_evolves_from(id, p.bench[j].id, false))
        return j;
    }
  }
  return -1;
}

// Build the pending decision for an OP_CHOOSE (options from a zone + filter).
static void build_choose(GameState& st, const Op& o) {
  EffectFrame& fr = st.effectStack.back();
  Player& me = st.players[fr.a];
  PendingDecision pd;
  pd.context = o.p4;
  pd.minCount = o.p2;
  pd.maxCount = o.p3;
  if (pd.maxCount < 0)
    pd.maxCount = count_source(st, fr.a, -pd.maxCount - 1, -1);
  pd.maxCount = std::max(0, pd.maxCount);
  if (pd.minCount > pd.maxCount) pd.minCount = pd.maxCount;
  int requiredMin = pd.minCount;
  int zone = o.p0, filter = o.p1;
  if (fr.sourceCardId == 1079 && zone == Z_OWN_INPLAY &&
      filter == F_BASIC_WITH_STAGE2_IN_HAND) {
    add_rare_candy_evolve_options(pd, st, fr);
    pd.maxCount = std::min(pd.maxCount, static_cast<int>(pd.options.size()));
    pd.minCount = std::min(pd.minCount, pd.maxCount);
    st.pending = pd;
    fr.phase = zone;
    return;
  }
  if (zone == Z_OWN_BENCH || zone == Z_OPP_BENCH) {
    Player& src = (zone == Z_OWN_BENCH) ? me : st.players[1 - fr.a];
    for (int j = 0; j < static_cast<int>(src.bench.size()); ++j)
      if (matches_dynamic_inplay_filter(st, fr, src.bench[j], filter))
        pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                              Atom::I(src.bench[j].id)});
  } else if (zone == Z_HAND) {
    for (int i = 0; i < static_cast<int>(me.hand.size()); ++i)
      if (matches_dynamic_card_filter(st, fr, me.hand[i], filter))
        pd.options.push_back({Atom::S("CARD"), Atom::S("HAND"), Atom::I(i),
                              Atom::I(me.hand[i])});
  } else if (zone == Z_OPP_HAND || zone == Z_OPP_HAND_BY_OPP) {
    Player& src = st.players[1 - fr.a];
    if (static_cast<int>(src.hand.size()) == src.handCount)
      src.handKnown = true;
    for (int i = 0; i < static_cast<int>(src.hand.size()); ++i)
      if (matches_filter(src.hand[i], filter))
        pd.options.push_back({Atom::S("CARD"), Atom::S("HAND"), Atom::I(i),
                              Atom::I(src.hand[i])});
  } else if (zone == Z_DECK || zone == Z_DECK_BOTTOM7) {
    if (zone == Z_DECK) mark_full_deck_inspected(me);
    int n = (zone == Z_DECK_BOTTOM7) ? std::min(7, static_cast<int>(me.deck.size()))
                                     : static_cast<int>(me.deck.size());
    if (fr.attackId == 740 && zone == Z_DECK &&
        pd.context == CTX_DECK_EVOLVE && fr.selfBench < 0) {
      int firstTarget = first_bench_index_with_deck_evolution(me);
      if (firstTarget > 0)
        st.turnActionCount += firstTarget;
      fr.selfBench = firstTarget >= 0 ? firstTarget : 0;
    }
    for (int i = 0; i < n; ++i) {
      bool matches = matches_dynamic_card_filter(st, fr, me.deck[i], filter);
      if (fr.attackId == 740 && zone == Z_DECK &&
          pd.context == CTX_DECK_EVOLVE) {
        matches = fr.selfBench >= 0 &&
                  fr.selfBench < static_cast<int>(me.bench.size()) &&
                  card_evolves_from(me.deck[i], me.bench[fr.selfBench].id, false);
      }
      if (matches)
        pd.options.push_back({Atom::S("CARD"), Atom::S("DECK"), Atom::I(i),
                              Atom::I(me.deck[i])});
    }
  } else if (zone == Z_DISCARD) {
    for (int i = 0; i < static_cast<int>(me.discard.size()); ++i)
      if (matches_filter(me.discard[i], filter))
        pd.options.push_back({Atom::S("CARD"), Atom::S("DISCARD"), Atom::I(i),
                              Atom::I(me.discard[i])});
  } else if (zone == Z_HAND_ENERGY || zone == Z_DISCARD_ENERGY ||
             zone == Z_DECK_ENERGY) {
    SmallVec<int, 64>& src = (zone == Z_HAND_ENERGY)      ? me.hand
                            : (zone == Z_DISCARD_ENERGY) ? me.discard
                                                         : me.deck;
    if (zone == Z_DECK_ENERGY) mark_full_deck_inspected(me);
    const char* label = (zone == Z_HAND_ENERGY)      ? "HAND"
                        : (zone == Z_DISCARD_ENERGY) ? "DISCARD"
                                                     : "DECK";
    for (int i = 0; i < static_cast<int>(src.size()); ++i)
      if (energy_card_matches(src[i], filter))
        pd.options.push_back({Atom::S("CARD"), Atom::S(label), Atom::I(i),
                              Atom::I(src[i])});
  } else if (zone == Z_OWN_INPLAY || zone == Z_OPP_INPLAY) {  // Active(-1) + Bench(j)
    Player& side = (zone == Z_OWN_INPLAY) ? me : st.players[1 - fr.a];
    bool needEnergy = (filter == F_HAS_ENERGY);  // in-play Pokemon carrying energy
    bool needToolOrSpecialEnergy = (filter == F_HAS_TOOL_OR_SPECIAL_ENERGY);
    bool excludeSavedSource = (filter == F_NOT_SAVED_SOURCE);
    int excludedIdx = excludeSavedSource ? saved_source_inplay_idx(fr) : -2;
    int realFilter = (needEnergy || needToolOrSpecialEnergy || excludeSavedSource) ? F_ANY : filter;
    if (side.activeKnown &&
        (!excludeSavedSource || excludedIdx != -1) &&
        matches_dynamic_inplay_filter(st, fr, side.active, realFilter) &&
        (!needEnergy || !side.active.energies.empty()) &&
        (!needToolOrSpecialEnergy ||
         (!side.active.tools.empty() || has_special_energy(side.active))))
      pd.options.push_back({Atom::S("CARD"), Atom::S("ACTIVE"), Atom::I(-1),
                            Atom::I(side.active.id)});
    for (int j = 0; j < static_cast<int>(side.bench.size()); ++j)
      if ((!excludeSavedSource || excludedIdx != j) &&
          matches_dynamic_inplay_filter(st, fr, side.bench[j], realFilter) &&
          (!needEnergy || !side.bench[j].energies.empty()) &&
          (!needToolOrSpecialEnergy ||
           (!side.bench[j].tools.empty() || has_special_energy(side.bench[j]))))
        pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                              Atom::I(side.bench[j].id)});
  } else if (zone == Z_OWN_ACTIVE_ENERGY) {
    // One option per energy unit on the Active; o.p1 = type+1 (0 = any type).
    add_energy_options(pd, me, filter, true);
  } else if (zone == Z_OWN_INPLAY_ENERGY) {
    int excludeIdx = (fr.sourceCardId == 962) ? fr.selfBench : -2;
    add_energy_options(pd, me, filter, false, excludeIdx);
  } else if (zone == Z_OWN_BENCH_ENERGY) {
    const InPlay* moveTarget =
        (o.p4 == CTX_MOVE_ENERGY_TO_ACTIVE && me.activeKnown) ? &me.active : nullptr;
    add_bench_energy_options(pd, me, filter, moveTarget,
                             fr.sourceCardId == 1221);
  } else if (zone == Z_SAVED_OWN_INPLAY_ENERGY) {
    add_saved_source_energy_options(pd, me, saved_source_inplay_idx(fr), filter);
  } else if (zone == Z_SAVED_OPP_INPLAY_ENERGY) {
    add_saved_source_energy_options(pd, st.players[1 - fr.a],
                                    saved_source_inplay_idx(fr), filter);
  } else if (zone == Z_OPP_ACTIVE_ENERGY) {
    add_unprevented_opp_energy_options(st, pd, 1 - fr.a, filter, true);
  } else if (zone == Z_OPP_INPLAY_ENERGY) {
    add_unprevented_opp_energy_options(st, pd, 1 - fr.a, filter, false);
  } else if (zone == Z_OPP_BENCH_ENERGY) {
    add_unprevented_opp_energy_options(st, pd, 1 - fr.a, filter, false, true);
  } else if (zone == Z_SELF_ENERGY) {
    add_saved_source_energy_options(pd, me, fr.selfBench, filter);
  } else if (zone == Z_INPLAY_TOOLS) {
    struct ToolChoice {
      int toolId;
      int owner;
      int ownerOffset;
      int areaRank;
      int inplayIdx;
      int holderSerial;
      int toolOrder;
      int seq;
      Descriptor desc;
    };
    bool opponentToolExists = false;
    bool opponentActiveToolExists = false;
    int ownToolChoiceCount = 0;
    auto tool_choice_rank = [&](const ToolChoice& c) {
      if (c.ownerOffset == 1)
        return c.areaRank == 0 ? 0 : 1;  // Opponent Active, then opponent Bench.
      if (opponentToolExists)
        return 2;                         // Own tools by CABT holder order.
      return c.areaRank == 0 ? 0 : 1;    // Own Active, then own Bench.
    };
    std::vector<ToolChoice> choices;
    int seq = 0;
    auto add_for = [&](const InPlay& p, const char* area, int inplayIdx,
                       int owner, int areaRank, int ownerOffset) {
      for (int i = 0; i < static_cast<int>(p.tools.size()); ++i) {
        if (ownerOffset == 1)
          opponentToolExists = true;
        if (ownerOffset == 1 && areaRank == 0)
          opponentActiveToolExists = true;
        if (ownerOffset == 0)
          ownToolChoiceCount += 1;
        int toolOrder = -1;
        if (i < static_cast<int>(p.toolOrders.size()))
          toolOrder = p.toolOrders[i];
        choices.push_back({
            p.tools[i],
            owner,
            ownerOffset,
            areaRank,
            inplayIdx < 0 ? 0 : inplayIdx,
            p.serial,
            toolOrder,
            seq++,
            {Atom::S("CARD"), Atom::S(area),
             Atom::I(encode_any_tool_ref(ownerOffset, inplayIdx, i)),
             Atom::I(p.tools[i])}});
      }
    };
    for (int owner = 0; owner < 2; ++owner) {
      const Player& holder = st.players[owner];
      int ownerOffset = owner == fr.a ? 0 : 1;
      if (holder.activeKnown)
        add_for(holder.active, "ACTIVE", -1, owner, 0, ownerOffset);
      for (int j = 0; j < static_cast<int>(holder.bench.size()); ++j)
        add_for(holder.bench[j], "BENCH", j, owner, 1, ownerOffset);
    }
    bool useToolOrder = !choices.empty();
    for (const auto& c : choices) {
      if (c.toolOrder < 0) {
        useToolOrder = false;
        break;
      }
    }
    std::stable_sort(choices.begin(), choices.end(),
                     [&](const ToolChoice& a, const ToolChoice& b) {
                       if (useToolOrder) {
                         if (a.toolOrder != b.toolOrder)
                           return a.toolOrder < b.toolOrder;
                       }
                       int ar = tool_choice_rank(a);
                       int br = tool_choice_rank(b);
                       if (ar != br)
                         return ar < br;
                       if (ar == 1 && a.inplayIdx != b.inplayIdx) {
                         if (opponentActiveToolExists)
                           return a.inplayIdx < b.inplayIdx;
                         if (a.inplayIdx > 0 && b.inplayIdx > 0)
                           return a.inplayIdx < b.inplayIdx;
                         return a.inplayIdx > b.inplayIdx;
                       }
                       if (ar == 2 && opponentToolExists &&
                           !opponentActiveToolExists &&
                           ownToolChoiceCount == 2 &&
                           a.inplayIdx != b.inplayIdx)
                         return a.inplayIdx > b.inplayIdx;
                       if (ar == 2 && opponentToolExists &&
                           a.holderSerial != b.holderSerial)
                         return a.holderSerial > b.holderSerial;
                       if (a.inplayIdx != b.inplayIdx)
                         return a.inplayIdx < b.inplayIdx;
                       return a.seq < b.seq;
                     });
    for (const auto& c : choices)
      pd.options.push_back(c.desc);
  }
  if (fr.attackId == 95 && zone == Z_OPP_INPLAY &&
      pd.context == CTX_ATTACK_TARGET) {
    int targetCount = std::min(2, static_cast<int>(pd.options.size()));
    pd.minCount = targetCount;
    pd.maxCount = targetCount;
    requiredMin = targetCount;
  }
  if (zone == Z_OPP_HAND_BY_OPP && pd.context == CTX_DISCARD &&
      requiredMin > static_cast<int>(pd.options.size())) {
    int discardCount = static_cast<int>(pd.options.size());
    pd.minCount = discardCount;
    pd.maxCount = discardCount;
    requiredMin = discardCount;
  }
  if (fr.sourceCardId == 1239 && zone == Z_HAND && pd.context == CTX_DISCARD) {
    // Naveen: you may discard from hand, but must discard down to keep at most
    // 4 so the follow-up "draw until 5" draws at least one card (cabt models
    // this as select-type MaxCardUntil 4: minCount = handCount - 4).
    int mustDiscard = std::max(0, static_cast<int>(pd.options.size()) - 4);
    pd.minCount = mustDiscard;
    requiredMin = mustDiscard;
  }
  if (fr.attackId == 955 && fr.sourceCardId == 1184)
    pd.minCount = 0;  // Ninetales copying Lana's Aid can choose zero.
  if (!preserves_unclamped_choice_count(fr, o) &&
      (!preserves_empty_choice_pending(st, fr, o) || !pd.options.empty())) {
    if (pd.context == 5 &&
        (zone == Z_HAND || zone == Z_DISCARD || zone == Z_DECK ||
         zone == Z_DECK_BOTTOM7 || zone == Z_DECK_ENERGY)) {
      int benchRoom = std::max(
          0, effective_bench_max(st, fr.a) -
                 static_cast<int>(me.bench.size()));
      pd.maxCount = std::min(pd.maxCount, benchRoom);
    }
    pd.maxCount = std::min(pd.maxCount, static_cast<int>(pd.options.size()));
    if (requiredMin > pd.maxCount) {
      pd.options.clear();
      pd.minCount = 0;
      pd.maxCount = 0;
      st.pending = pd;
      return;
    }
    pd.minCount = std::min(pd.minCount, pd.maxCount);
  }
  if (fr.attackId == 1037 && zone == Z_OPP_ACTIVE_ENERGY &&
      pd.context == CTX_DISCARD_ENERGY)
    pd.minCount = pd.maxCount;
  st.pending = pd;
  fr.phase = zone;  // remember the source zone for the consuming op
  if (zone == Z_OPP_HAND_BY_OPP)
    st.yourIndex = 1 - fr.a;
}

static void build_choose_top_deck(GameState& st, const Op& o) {
  EffectFrame& fr = st.effectStack.back();
  int ownerIndex = (o.p0 < 0) ? (1 - fr.a) : fr.a;
  Player& owner = st.players[ownerIndex];
  PendingDecision pd;
  pd.context = o.p4;
  pd.minCount = o.p2;
  pd.maxCount = o.p3;
  int rawCount = (o.p0 < 0) ? -o.p0 : o.p0;
  int n = std::min(rawCount, static_cast<int>(owner.deck.size()));
  if (n == owner.deckCount && static_cast<int>(owner.deck.size()) == owner.deckCount)
    mark_full_deck_inspected(owner);
  bool sameWindow = fr.topDeckCount > 0 && fr.topDeckOwner == ownerIndex;
  if (!sameWindow) {
    int countedOut = std::min(n, std::max(0, owner.deckCount));
    owner.deckCount -= countedOut;
    fr.topDeckCountedOut = countedOut;
  }
  int start = static_cast<int>(owner.deck.size()) - n;
  if (n > 0 && static_cast<int>(owner.deck.size()) ==
                   owner.deckCount + fr.topDeckCountedOut) {
    if (owner.deckKnownMask.size() < owner.deck.size())
      owner.deckKnownMask.resize(owner.deck.size(), false);
    for (int i = start; i < static_cast<int>(owner.deck.size()); ++i)
      owner.deckKnownMask[i] = true;
    // deckCount temporarily excludes the revealed window. Normalizing here
    // would treat that intentional mismatch as replay uncertainty and erase
    // the positions we just learned; normalization happens when the window is
    // closed and deckCount is restored.
  }
  std::vector<int> exclude;
  if (fr.savedPhase == Z_DECK && fr.topDeckOwner == ownerIndex)
    exclude = fr.savedScratch;
  for (int localIdx = 0; localIdx < n; ++localIdx) {
    if (std::find(exclude.begin(), exclude.end(), localIdx) != exclude.end()) continue;
    int i = start + (n - 1 - localIdx);
    if (matches_filter(owner.deck[i], o.p1))
      pd.options.push_back({Atom::S("CARD"), Atom::S("LOOKING"), Atom::I(localIdx),
                            Atom::I(owner.deck[i])});
  }
  pd.maxCount = std::min(pd.maxCount, static_cast<int>(pd.options.size()));
  if (fr.sourceCardId == 1091 && ownerIndex == 1 - fr.a) {
    int benchRoom = std::max(
        0, effective_bench_max(st, ownerIndex) -
               static_cast<int>(owner.bench.size()));
    pd.maxCount = std::min(pd.maxCount, benchRoom);
  }
  pd.minCount = std::min(pd.minCount, pd.maxCount);
  st.pending = pd;
  fr.phase = Z_DECK;
  fr.topDeckCount = n;
  fr.topDeckOwner = ownerIndex;
  fr.topDeckStart = start;
  fr.topDeckSelectedCount = static_cast<int>(exclude.size());
}

static int saved_source_inplay_idx(const EffectFrame& fr) {
  if (fr.savedSrc != -2) return fr.savedSrc;
  if (fr.savedScratch.empty()) return -2;
  if (fr.savedPhase == Z_OWN_ACTIVE_ENERGY ||
      fr.savedPhase == Z_OWN_INPLAY_ENERGY ||
      fr.savedPhase == Z_OWN_BENCH_ENERGY ||
      fr.savedPhase == Z_OPP_ACTIVE_ENERGY ||
      fr.savedPhase == Z_OPP_INPLAY_ENERGY ||
      fr.savedPhase == Z_OPP_BENCH_ENERGY)
    return energy_ref_inplay(fr.savedScratch[0]);
  if (fr.savedPhase == Z_OWN_INPLAY || fr.savedPhase == Z_OPP_INPLAY ||
      fr.savedPhase == Z_OWN_BENCH || fr.savedPhase == Z_OPP_BENCH)
    return fr.savedScratch[0];
  return -2;
}

static void add_inplay_choice_options(PendingDecision& pd, const GameState& st,
                                      const EffectFrame& fr, const Player& owner,
                                      int zone, int repeats, int filter) {
  repeats = std::max(1, repeats);
  bool needEnergy = (filter == F_HAS_ENERGY);
  bool needToolOrSpecialEnergy = (filter == F_HAS_TOOL_OR_SPECIAL_ENERGY);
  bool excludeSavedSource = (filter == F_NOT_SAVED_SOURCE);
  int excludedIdx = excludeSavedSource ? saved_source_inplay_idx(fr) : -2;
  int realFilter = (needEnergy || needToolOrSpecialEnergy || excludeSavedSource) ? F_ANY : filter;
  auto add_if_ok = [&](const InPlay& p, const char* area, int idx) {
    if (excludeSavedSource && idx == excludedIdx) return;
    if (!matches_dynamic_inplay_filter(st, fr, p, realFilter)) return;
    if (needEnergy && p.energies.empty()) return;
    if (needToolOrSpecialEnergy && p.tools.empty() && !has_special_energy(p)) return;
    pd.options.push_back({Atom::S("CARD"), Atom::S(area), Atom::I(idx),
                          Atom::I(p.id)});
  };
  for (int r = 0; r < repeats; ++r) {
    if ((zone == Z_OWN_INPLAY || zone == Z_OPP_INPLAY) && owner.activeKnown)
      add_if_ok(owner.active, "ACTIVE", -1);
    if (zone == Z_OWN_INPLAY || zone == Z_OPP_INPLAY ||
        zone == Z_OWN_BENCH || zone == Z_OPP_BENCH) {
      for (int j = 0; j < static_cast<int>(owner.bench.size()); ++j)
        add_if_ok(owner.bench[j], "BENCH", j);
    }
  }
}

static void build_distribute_damage(GameState& st, const Op& o) {
  EffectFrame& fr = st.effectStack.back();
  PendingDecision pd;
  pd.context = o.p3;
  int count = (o.p1 == -1) ? (fr.scratch.empty() ? 0 : std::max(0, fr.scratch[0]))
                           : o.p1;
  int minCount = (o.p2 == -1) ? count : o.p2;
  pd.minCount = std::max(0, minCount);
  pd.maxCount = std::max(pd.minCount, count);
  fr.loopRemain = pd.maxCount;
  int zone = o.p0;
  Player& owner = (zone == Z_OWN_BENCH || zone == Z_OWN_INPLAY)
                      ? st.players[fr.a] : st.players[1 - fr.a];
  add_inplay_choice_options(pd, st, fr, owner, zone, std::max(1, count), o.p4);
  st.pending = pd;
  fr.phase = zone;
}

static void build_attach_targets(GameState& st, const Op& o) {
  EffectFrame& fr = st.effectStack.back();
  PendingDecision pd;
  int n = static_cast<int>(fr.savedScratch.size());
  pd.context = o.p2;
  pd.minCount = std::max(0, o.p3 < 0 ? n : o.p3);
  pd.maxCount = std::max(pd.minCount, o.p4 < 0 ? n : o.p4);
  Player& owner = st.players[fr.a];
  add_inplay_choice_options(pd, st, fr, owner, o.p0, std::max(1, n), o.p1);
  st.pending = pd;
  fr.phase = o.p0;
}

static void build_choose_distinct_basic_energy(GameState& st, const Op& o) {
  EffectFrame& fr = st.effectStack.back();
  Player& owner = zone_owner(st, fr.a, o.p0);
  SmallVec<int, 64>* src = card_zone_for_phase(owner, o.p0);
  PendingDecision pd;
  pd.context = o.p3;
  pd.minCount = std::max(0, o.p2);
  pd.maxCount = std::max(pd.minCount, o.p1);
  const char* label = "DECK";
  if (o.p0 == Z_HAND_ENERGY) label = "HAND";
  else if (o.p0 == Z_DISCARD_ENERGY) label = "DISCARD";
  bool seen[16] = {};
  if (src) {
    for (int i = 0; i < static_cast<int>(src->size()); ++i) {
      int etype = -1;
      if (!basic_energy_type((*src)[i], etype)) continue;
      if (etype < 0 || etype >= 16 || seen[etype]) continue;
      seen[etype] = true;
      pd.options.push_back({Atom::S("CARD"), Atom::S(label),
                            Atom::I(i), Atom::I((*src)[i])});
    }
  }
  st.pending = pd;
  fr.phase = o.p0;
}

static bool take_energy_ref(GameState& st, Player& owner, int ref, int& etype,
                            int& cardId, int* attachOrder = nullptr) {
  int inplayIdx = energy_ref_inplay(ref);
  int energyIdx = energy_ref_index(ref);
  InPlay* src = inplay_ref(owner, inplayIdx);
  if (!src || energyIdx < 0 || energyIdx >= static_cast<int>(src->energies.size()))
    return false;
  etype = src->energies[energyIdx];
  cardId = attached_energy_card_id(*src, energyIdx);
  if (attachOrder) *attachOrder = attached_energy_order(*src, energyIdx);
  erase_attached_energy_card(*src, energyIdx);
  src->energies.erase(src->energies.begin() + energyIdx);
  refresh_inplay_max_hp(st, *src);
  return true;
}

static int player_index_for_owner(GameState& st, Player& owner) {
  if (&owner == &st.players[0]) return 0;
  if (&owner == &st.players[1]) return 1;
  return -1;
}

static bool current_attack_effect_from_active(GameState& st, int ownerIdx,
                                              InPlay& p) {
  if (ownerIdx < 0 || st.effectStack.empty()) return false;
  const EffectFrame& fr = st.effectStack.back();
  return fr.attackId > 0 && fr.a == ownerIdx &&
         st.players[ownerIdx].activeKnown && &st.players[ownerIdx].active == &p;
}

static void discard_attached_energy_card(GameState& st, Player& owner,
                                         InPlay& p, int cardId) {
  int ownerIdx = player_index_for_owner(st, owner);
  owner.discard.push_back(cardId);
  if (cardId == 9 && current_attack_effect_from_active(st, ownerIdx, p))
    st.boomerangEnergyReturnCount[ownerIdx] += 1;
}

static void reattach_boomerang_energy_after_attack(GameState& st, int attacker) {
  if (attacker < 0 || attacker >= 2) return;
  Player& me = st.players[attacker];
  if (!me.activeKnown || st.boomerangEnergyReturnCount[attacker] <= 0) {
    st.boomerangEnergyReturnCount[attacker] = 0;
    return;
  }
  int returned = 0;
  for (int i = 0; i < st.boomerangEnergyReturnCount[attacker]; ++i) {
    auto it = std::find(me.discard.begin(), me.discard.end(), 9);
    if (it == me.discard.end()) break;
    me.discard.erase(it);
    push_attached_energy_card(me.active, 9);
    me.active.energies.push_back(COLORLESS);
    ++returned;
  }
  if (returned > 0) refresh_inplay_max_hp(st, me.active, attacker);
  st.boomerangEnergyReturnCount[attacker] = 0;
}

static int discard_energy_refs(GameState& st, Player& owner, std::vector<int> refs) {
  std::sort(refs.rbegin(), refs.rend());
  int n = 0;
  for (int ref : refs) {
    int etype = 0, cardId = 0;
    if (take_energy_ref(st, owner, ref, etype, cardId)) {
      InPlay* src = inplay_ref(owner, energy_ref_inplay(ref));
      if (cardId && src) discard_attached_energy_card(st, owner, *src, cardId);
      else if (cardId) owner.discard.push_back(cardId);
      ++n;
    }
  }
  return n;
}

static int discard_energy_from_inplay(GameState& st, Player& owner, InPlay& p,
                                      int count, int etype) {
  int n = 0;
  for (int k = static_cast<int>(p.energies.size()) - 1;
       k >= 0 && (count < 0 || n < count); --k) {
    if (etype >= 0 && p.energies[k] != etype) continue;
    if (k < static_cast<int>(p.energyCardIds.size())) {
      discard_attached_energy_card(st, owner, p, p.energyCardIds[k]);
      erase_attached_energy_card(p, k);
    }
    p.energies.erase(p.energies.begin() + k);
    ++n;
  }
  if (n > 0) refresh_inplay_max_hp(st, p);
  return n;
}

static bool discard_last_tool(GameState& st, Player& owner, InPlay& p) {
  if (p.tools.empty()) return false;
  int damage = p.maxHp - p.hp;
  owner.discard.push_back(pop_tool_card(p));
  p.maxHp = effective_max_hp(p.id, p.tools, st, p.energyCardIds, p.energies,
                             owner_of_inplay(st, p));
  p.hp = std::max(0, p.maxHp - damage);
  return true;
}

static int discard_tools_from_inplay(GameState& st, Player& owner, InPlay& p,
                                     int count) {
  int discarded = 0;
  while (!p.tools.empty() && (count < 0 || discarded < count)) {
    if (!discard_last_tool(st, owner, p)) break;
    ++discarded;
  }
  return discarded;
}

static InPlay* chosen_target(Player& me, const EffectFrame& fr, int target) {
  int nb = static_cast<int>(me.bench.size());
  if (target == AT_OWN_ACTIVE)
    return me.activeKnown ? &me.active : nullptr;
  if (target == AT_SELF)
    return (fr.selfBench < 0) ? (me.activeKnown ? &me.active : nullptr)
                              : (fr.selfBench < nb ? &me.bench[fr.selfBench] : nullptr);
  if (target == AT_CHOSEN_BENCH) {
    if (!fr.scratch.empty() && fr.scratch[0] >= 0 && fr.scratch[0] < nb)
      return &me.bench[fr.scratch[0]];
    return nullptr;
  }
  if (target == AT_CHOSEN_INPLAY && !fr.scratch.empty()) {
    int idx = fr.scratch[0];
    return inplay_ref(me, idx);
  }
  if (target == AT_SAVED_INPLAY) {
    int idx = saved_source_inplay_idx(fr);
    return inplay_ref(me, idx);
  }
  return nullptr;
}

static InPlay* chosen_target_any(Player& me, Player& opp,
                                 const EffectFrame& fr, int target) {
  int nb = static_cast<int>(opp.bench.size());
  if (target == AT_OPP_CHOSEN_BENCH) {
    if (!fr.scratch.empty() && fr.scratch[0] >= 0 && fr.scratch[0] < nb)
      return &opp.bench[fr.scratch[0]];
    return nullptr;
  }
  if (target == AT_OPP_CHOSEN_INPLAY && !fr.scratch.empty())
    return inplay_ref(opp, fr.scratch[0]);
  return chosen_target(me, fr, target);
}

// Stadium damage modifier applied to the DEFENDER (the opponent's Pokemon being
// hit by an attack): Full Metal Lab (-30 to {M}); Neutralization Zone prevents
// rule-box attackers from damaging non-rule-box Pokemon.
static int stadium_damage_mod(const GameState& st, int attackerId,
                              int defenderId, int dmg) {
  if (st.stadium.empty() || dmg <= 0) return dmg;
  const CardInfo* defender = find_card(defenderId);
  if (!defender) return dmg;
  switch (st.stadium[0]) {
    case 1244:
      return (defender->energyType == METAL) ? (dmg > 30 ? dmg - 30 : 0) : dmg;
    case 1247: {
      const CardInfo* attacker = find_card(attackerId);
      return (has_rule_box(attacker) && !has_rule_box(defender)) ? 0 : dmg;
    }
    default: return dmg;
  }
}

static void apply_stadium_hp_change(GameState& st, int oldId, int newId);  // below

static bool counts_empty_choice_step(const GameState& st, const EffectFrame& fr,
                                     const Op& o) {
  if (o.kind == OP_CHOOSE_TOP_DECK) return o.p3 > 0;
  if (o.kind != OP_CHOOSE) return false;
  if (o.p4 == CTX_TO_BENCH &&
      static_cast<int>(st.players[fr.a].bench.size()) >=
          effective_bench_max(st, fr.a))
    return false;
  if (fr.sourceCardId == 461 && o.p0 == Z_DECK && o.p4 == 5) {
    if (st.players[fr.a].deckCount <= 0)
      return false;
    if (static_cast<int>(st.players[fr.a].bench.size()) >=
        effective_bench_max(st, fr.a))
      return false;
  }
  if (fr.attackId == 1501 && o.p0 == Z_HAND && o.p4 == 8)
    return true;
  if ((fr.attackId == 39 || fr.attackId == 478 || fr.attackId == 620 ||
       fr.attackId == 740 || fr.attackId == 1262) &&
      o.p0 == Z_DECK && o.p4 == CTX_DECK_EVOLVE)
    return st.players[fr.a].deckCount > 0;
  if (fr.sourceCardId == 641 && o.p0 == Z_DECK)
    return matches_filter(8, o.p1);  // X-Boot: count only the second Metal miss.
  if (fr.sourceCardId == 1100 && o.p0 == Z_DECK)
    return false;  // Energy Search Pro: whiffed optional search is not a step.
  if (fr.sourceCardId == 898 && o.p0 == Z_DECK)
    return false;  // Pecharunt Parting Gift: empty deck skips without a step.
  if (fr.attackId == 189 && o.kind == OP_CHOOSE && o.p0 == Z_DECK &&
      o.p4 == 7)
    return false;  // Eevee: whiffed optional Energy search is not a step.
  if (fr.sourceCardId == 1135 && o.kind == OP_CHOOSE && o.p0 == Z_DECK &&
      o.p4 == 22)
    return st.players[fr.a].deckCount > 0;
  if (fr.sourceCardId == 182 && o.p0 == Z_HAND &&
      o.p4 == CTX_TO_DECK_BOTTOM)
    return true;  // Quaquaval: empty Up-Tempo cost still consumes a step.
  if (fr.attackId == 321 && o.kind == OP_CHOOSE && o.p0 == Z_DECK)
    return false;
  if (fr.attackId == 467 && o.kind == OP_CHOOSE && o.p0 == Z_DECK &&
      o.p4 == 22)
    return false;
  if (fr.attackId == 965 && o.kind == OP_CHOOSE && o.p0 == Z_DECK &&
      o.p4 == 22)
    return st.players[fr.a].deckCount > 0;
  if (fr.attackId == 53 && o.kind == OP_CHOOSE && o.p0 == Z_HAND_ENERGY)
    return true;
  if (fr.attackId == 1285 && o.kind == OP_CHOOSE && o.p0 == Z_HAND &&
      o.p4 == CTX_DISCARD)
    return true;
  if (fr.attackId == 522 && o.kind == OP_CHOOSE && o.p0 == Z_OWN_BENCH)
    return true;
  if (fr.attackId == 1453 && o.kind == OP_CHOOSE && o.p0 == Z_DECK &&
      st.lastEffectCount <= 0)
    return false;
  if (fr.attackId == 1547 && o.kind == OP_CHOOSE && o.p0 == Z_DECK &&
      o.p4 == 22 && st.lastEffectCount <= 0)
    return false;
  return o.p0 == Z_DECK || o.p0 == Z_DECK_BOTTOM7 ||
         o.p0 == Z_DECK_ENERGY || o.p0 == Z_OPP_HAND ||
         o.p0 == Z_OPP_HAND_BY_OPP;
}

static bool preserves_empty_choice_pending(const GameState& st,
                                           const EffectFrame& fr,
                                           const Op& o) {
  (void)st;
  return fr.sourceCardId == 357 && o.kind == OP_CHOOSE &&
         o.p0 == Z_OWN_BENCH && o.p4 == 22;
}

static bool suspends_empty_choice_pending(const GameState& st,
                                          const EffectFrame& fr,
                                          const Op& o) {
  return preserves_empty_choice_pending(st, fr, o);
}

static int next_program_end_pc(const EffectFrame& fr) {
  const Op* ops = effect_ops() + fr.program;
  for (int pc = fr.pc; pc < fr.pc + 64; ++pc)
    if (ops[pc].kind == OP_END) return pc;
  return fr.pc + 1;
}

static void skip_empty_choice(GameState& st, EffectFrame& fr, const Op& o) {
  st.pending = PendingDecision();
  bool countStep = counts_empty_choice_step(st, fr, o);
  fr.scratch.clear();
  st.lastEffectCount = 0;
  if (fr.attackId == 742 && o.kind == OP_CHOOSE &&
      o.p0 == Z_OPP_INPLAY_ENERGY) {
    fr.pc = next_program_end_pc(fr);
    return;
  }
  if (!countStep && fr.sourceCardId == 542 && o.kind == OP_CHOOSE &&
      o.p4 == CTX_TO_BENCH)
    countStep = true;
  if (countStep) st.turnActionCount += 1;
  fr.pc += 1;
}

void run_program(GameState& st, const std::vector<int>& tape) {
  (void)tape;
  while (true) {
    EffectFrame& fr = st.effectStack.back();
    st.yourIndex = fr.a;
    const Op& o = (effect_ops() + fr.program)[fr.pc];
    Player& me = st.players[fr.a];
    switch (o.kind) {
      case OP_END:
        finish_program(st);
        if (st.has_pending() || st.effectStack.empty()) return;
        if (st.effectStack.back().effect != FLOW_PROGRAM) return;
        break;
      case OP_CHOOSE:
        build_choose(st, o);
        if ((st.pending.options.empty() || st.pending.maxCount <= 0 ||
             static_cast<int>(st.pending.options.size()) < st.pending.minCount) &&
            !suspends_empty_choice_pending(st, fr, o)) {
          skip_empty_choice(st, fr, o);
          break;
        }
        return;  // suspend for the player's choice
      case OP_CHOOSE_TOP_DECK:
        build_choose_top_deck(st, o);
        if (st.pending.options.empty() || st.pending.maxCount <= 0) {
          skip_empty_choice(st, fr, o);
          break;
        }
        return;
      case OP_REPEAT_OPP_ACTIVE_ENERGY_DISCARD: {
        if (fr.loopRemain < 0)
          fr.loopRemain = std::max(0, st.lastEffectCount);
        if (fr.loopRemain <= 0) {
          fr.loopRemain = -1;
          fr.scratch.clear();
          st.lastEffectCount = 0;
          fr.pc += 3;  // skip DISCARD_CHOSEN and LOOP_BACK.
          break;
        }
        PendingDecision pd;
        pd.context = CTX_DISCARD_ENERGY;
        pd.minCount = 1;
        pd.maxCount = 1;
        add_unprevented_opp_energy_options(st, pd, 1 - fr.a, 0, true);
        if (pd.options.empty()) {
          fr.loopRemain = -1;
          fr.scratch.clear();
          st.lastEffectCount = 0;
          fr.pc += 3;
          break;
        }
        st.pending = pd;
        fr.phase = Z_OPP_ACTIVE_ENERGY;
        return;
      }
      case OP_CHOOSE_COUNT: {
        PendingDecision pd;
        pd.context = o.p1;
        pd.minCount = 1;
        pd.maxCount = 1;
        int mx = std::max(0, o.p0);
        for (int i = 0; i <= mx; ++i) {
          if (pd.context == CTX_DAMAGE_COUNT)
            pd.options.push_back({Atom::S("NUMBER"), Atom::I(i)});
          else
            pd.options.push_back({Atom::S("COUNT"), Atom::S("DAMAGE"),
                                  Atom::I(i)});
        }
        st.pending = pd;
        return;
      }
      case OP_CHOOSE_STATUS: {
        PendingDecision pd;
        pd.context = o.p0;
        pd.minCount = 1;
        pd.maxCount = 1;
        if (o.p0 == 48) {
          auto add_recover = [&](bool present, int status) {
            if (present)
              pd.options.push_back({Atom::S("SPECIAL_CONDITION"),
                                    Atom::I(status)});
          };
          add_recover(me.poisoned, ST_POISON);
          add_recover(me.burned, ST_BURN);
          add_recover(me.asleep, ST_ASLEEP);
          add_recover(me.paralyzed, ST_PARALYZED);
          add_recover(me.confused, ST_CONFUSED);
          if (pd.options.empty()) {
            st.lastEffectCount = 0;
            fr.pc += 1;
            break;
          }
        } else {
          pd.options.push_back({Atom::S("SPECIAL_CONDITION"),
                                Atom::I(ST_POISON)});
          pd.options.push_back({Atom::S("SPECIAL_CONDITION"),
                                Atom::I(ST_BURN)});
          pd.options.push_back({Atom::S("SPECIAL_CONDITION"),
                                Atom::I(ST_ASLEEP)});
          pd.options.push_back({Atom::S("SPECIAL_CONDITION"),
                                Atom::I(ST_PARALYZED)});
          pd.options.push_back({Atom::S("SPECIAL_CONDITION"),
                                Atom::I(ST_CONFUSED)});
        }
        st.pending = pd;
        return;
      }
      case OP_CHOOSE_DEVOLVE_COUNT: {
        PendingDecision pd;
        pd.context = o.p1;
        pd.minCount = 1;
        pd.maxCount = 1;
        int zone = o.p0;
        Player& owner = (zone == Z_OWN_INPLAY || zone == Z_OWN_BENCH)
                            ? st.players[fr.a]
                            : st.players[1 - fr.a];
        int idx = fr.savedScratch.empty() ? -2 : fr.savedScratch[0];
        InPlay* target = inplay_ref(owner, idx);
        int mx = target ? static_cast<int>(target->preEvo.size()) : 0;
        for (int i = 1; i <= mx; ++i)
          pd.options.push_back({Atom::S("COUNT"), Atom::S("DEVOLVE"),
                                Atom::I(i)});
        st.pending = pd;
        if (pd.options.empty()) {
          st.pending = PendingDecision();
          fr.scratch.clear();
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        return;
      }
      case OP_CHOOSE_DAMAGE_COUNTER_COUNT: {
        PendingDecision pd;
        pd.context = o.p1;
        pd.minCount = 1;
        pd.maxCount = 1;
        int zone = o.p0;
        Player& owner = (zone == Z_OWN_INPLAY || zone == Z_OWN_BENCH)
                            ? st.players[fr.a]
                            : st.players[1 - fr.a];
        int idx = fr.savedScratch.empty() ? -2 : fr.savedScratch[0];
        InPlay* target = inplay_ref(owner, idx);
        int mx = target ? std::max(0, (target->maxHp - target->hp) / 10) : 0;
        if (o.p2 > 0) mx = std::min(mx, o.p2);
        for (int i = 0; i <= mx; ++i)
          pd.options.push_back({Atom::S("COUNT"), Atom::S("DAMAGE"),
                                Atom::I(i)});
        st.pending = pd;
        if (pd.options.empty()) {
          st.pending = PendingDecision();
          fr.scratch.clear();
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        return;
      }
      case OP_DISTRIBUTE_DAMAGE:
        build_distribute_damage(st, o);
        if (st.pending.options.empty() || st.pending.maxCount <= 0) {
          st.pending = PendingDecision();
          fr.scratch.clear();
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        return;
      case OP_CHOOSE_ATTACH_TARGETS:
        build_attach_targets(st, o);
        if (st.pending.options.empty() || st.pending.maxCount <= 0) {
          st.pending = PendingDecision();
          fr.scratch.clear();
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        return;
      case OP_CHOOSE_DISTINCT_BASIC_ENERGY:
        build_choose_distinct_basic_energy(st, o);
        if (st.pending.options.empty() || st.pending.maxCount <= 0) {
          st.pending = PendingDecision();
          fr.scratch.clear();
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        return;
      case OP_CHOOSE_COPIED_ATTACK: {
        PendingDecision pd;
        pd.context = o.p0;
        pd.minCount = 1;
        pd.maxCount = 1;
        InPlay* src = fr.scratch.empty() ? nullptr : inplay_ref(me, fr.scratch[0]);
        int sourceCardId = src ? src->id : 0;
        if (sourceCardId == 0 && fr.phase == Z_DECK && fr.topDeckOwner >= 0 &&
            !fr.scratch.empty()) {
          Player& owner = st.players[fr.topDeckOwner];
          int idx = fr.scratch[0];
          if (idx >= 0 && idx < static_cast<int>(owner.deck.size()))
            sourceCardId = owner.deck[idx];
        }
        const CardInfo* c = sourceCardId ? find_card(sourceCardId) : nullptr;
        if (c) {
          for (int i = 0; i < c->n_attacks; ++i) {
            const AttackInfo& at = c->attacks[i];
            pd.options.push_back({Atom::S("ATTACK"), Atom::I(at.id),
                                  Atom::I(encode_copied_attack_ref(at.id, at.damage)),
                                  Atom::I(sourceCardId)});
          }
        }
        if (pd.options.empty()) {
          st.lastEffectCount = 0;
          fr.scratch.clear();
          fr.pc += 1;
          break;
        }
        st.pending = pd;
        return;
      }
      case OP_RUN_COPIED_ATTACK: {
        if (!fr.scratch.empty()) {
          int ref = fr.scratch[0];
          int attackId = copied_attack_id(ref);
          int prog = attack_program(attackId);
          EffectFrame nested;
          nested.effect = FLOW_PROGRAM;
          nested.program = (prog >= 0) ? prog : vanilla_attack_program();
          nested.a = fr.a;
          nested.attackId = attackId;
          nested.attackCardId = me.activeKnown ? me.active.id : 0;
          nested.copiedAttack = true;
          nested.copiedAttackBaseDamage = copied_attack_damage(ref);
          fr.pc += 1;
          st.effectStack.push_back(nested);
          break;
        }
        st.lastEffectCount = 0;
        fr.pc += 1;
        break;
      }
      case OP_RUN_CHOSEN_OPP_ACTIVE_ATTACK: {
        if (!fr.scratch.empty()) {
          int attackId = fr.scratch[0];
          Player& opp = st.players[1 - fr.a];
          const CardInfo* c = opp.activeKnown ? find_card(opp.active.id) : nullptr;
          const AttackInfo* chosen = nullptr;
          if (c) {
            for (int i = 0; i < c->n_attacks; ++i) {
              if (c->attacks[i].id == attackId) {
                chosen = &c->attacks[i];
                break;
              }
            }
          }
          if (chosen) {
            int prog = attack_program(attackId);
            EffectFrame nested;
            nested.effect = FLOW_PROGRAM;
            nested.program = (prog >= 0) ? prog : vanilla_attack_program();
            nested.a = fr.a;
            nested.attackId = attackId;
            nested.attackCardId = me.activeKnown ? me.active.id : 0;
            nested.copiedAttack = true;
            nested.copiedAttackBaseDamage = chosen->damage;
            fr.pc += 1;
            st.effectStack.push_back(nested);
            break;
          }
        }
        st.lastEffectCount = 0;
        fr.pc += 1;
        break;
      }
      case OP_CHOOSE_OPP_ACTIVE_ATTACK: {
        PendingDecision pd;
        pd.context = o.p0;
        pd.minCount = 1;
        pd.maxCount = 1;
        Player& opp = st.players[1 - fr.a];
        const CardInfo* c = opp.activeKnown ? find_card(opp.active.id) : nullptr;
        if (o.p0 == 36 && opp.activeKnown &&
            attack_effects_prevented(st, 1 - fr.a, opp.active)) {
          st.lastEffectCount = 0;
          fr.scratch.clear();
          fr.pc += 1;
          break;
        }
        if (c) {
          for (int i = 0; i < c->n_attacks; ++i) {
            const AttackInfo& at = c->attacks[i];
            if (o.p0 == 36) {
              pd.options.push_back({Atom::S("ATTACK"), Atom::I(at.id)});
            } else {
              pd.options.push_back({Atom::S("ATTACK"), Atom::I(at.id),
                                    Atom::I(at.id), Atom::I(opp.active.id)});
            }
          }
        }
        if (pd.options.empty()) {
          st.lastEffectCount = 0;
          fr.scratch.clear();
          fr.pc += 1;
          break;
        }
        st.pending = pd;
        return;
      }
      case OP_DISABLE_CHOSEN_ATTACK: {
        Player& opp = st.players[1 - fr.a];
        if (opp.activeKnown && !fr.scratch.empty()) {
          opp.active.lockId = fr.scratch[0];
          opp.active.lockTurn = st.turn + 1;
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_CHOOSE_OPP_ATTACHED_OR_STADIUM: {
        PendingDecision pd;
        pd.context = o.p0;
        pd.minCount = 1;
        pd.maxCount = 1;
        add_attached_or_stadium_options(pd, st, fr.a);
        if (pd.options.empty()) {
          st.lastEffectCount = 0;
          fr.scratch.clear();
          fr.pc += 1;
          break;
        }
        st.pending = pd;
        return;
      }
      case OP_CRISPIN_CHOOSE_ENERGIES: {
        // The option list enumerates every Basic Energy in the remaining
        // deck, so the player has legally inspected the complete deck.
        mark_full_deck_inspected(me);
        PendingDecision pd;
        pd.context = o.p0;
        pd.minCount = 0;
        pd.maxCount = 1;
        add_crispin_energy_options(pd, me, o.p1, fr.savedSrc);
        if (pd.options.empty()) {
          st.lastEffectCount = 0;
          fr.scratch.clear();
          st.turnActionCount += 1;
          fr.pc += 1;
          break;
        }
        st.pending = pd;
        return;
      }
      case OP_SWITCH_ACTIVE: {
        int owner = (o.p0 == S_OWN) ? fr.a : 1 - fr.a;
        Player& side = st.players[owner];
        if (!fr.scratch.empty()) {
          int benchIdx = fr.scratch[0];
          if (o.p0 == S_OPP && side.activeKnown &&
              attack_effects_prevented(st, 1 - fr.a, side.active)) {
            st.lastEffectCount = 0;
            fr.pc += 1;
            break;
          }
          if (o.p0 == S_OPP && benchIdx >= 0 &&
              benchIdx < static_cast<int>(side.bench.size()) &&
              attack_effects_prevented(st, 1 - fr.a, side.bench[benchIdx])) {
            st.lastEffectCount = 0;
            fr.pc += 1;
            break;
          }
          clear_active_spot_locks(side.active);
          std::swap(side.active, side.bench[benchIdx]);
          side.active.movedToActiveThisTurn = true;
          clear_status(side);  // switching cures special conditions
          apply_mismagius_switch_passive(st, owner, benchIdx);
          queue_move_triggers(st, owner, benchIdx, true, true);
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_SWITCH_ACTIVE_SAVE: {
        int owner = (o.p0 == S_OWN) ? fr.a : 1 - fr.a;
        Player& side = st.players[owner];
        fr.savedSrc = -2;
        if (!fr.scratch.empty()) {
          int benchIdx = fr.scratch[0];
          if (o.p0 == S_OPP && side.activeKnown &&
              attack_effects_prevented(st, 1 - fr.a, side.active)) {
            st.lastEffectCount = 0;
            fr.pc += 1;
            break;
          }
          if (o.p0 == S_OPP && benchIdx >= 0 &&
              benchIdx < static_cast<int>(side.bench.size()) &&
              attack_effects_prevented(st, 1 - fr.a, side.bench[benchIdx])) {
            st.lastEffectCount = 0;
            fr.pc += 1;
            break;
          }
          clear_active_spot_locks(side.active);
          std::swap(side.active, side.bench[benchIdx]);
          side.active.movedToActiveThisTurn = true;
          clear_status(side);
          apply_mismagius_switch_passive(st, owner, benchIdx);
          queue_move_triggers(st, owner, benchIdx, true, true);
          fr.savedSrc = benchIdx;
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_SWITCH_SELF_BENCH_ACTIVE: {
        int benchIdx = fr.selfBench;
        if (benchIdx >= 0 && benchIdx < static_cast<int>(me.bench.size()) &&
            me.activeKnown) {
          clear_active_spot_locks(me.active);
          std::swap(me.active, me.bench[benchIdx]);
          me.active.movedToActiveThisTurn = true;
          clear_status(me);
          apply_mismagius_switch_passive(st, fr.a, benchIdx);
          queue_move_triggers(st, fr.a, benchIdx, true, true);
          fr.selfBench = -1;
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_MOVE_CHOSEN: {
        int moved = 0;
        if ((fr.phase == Z_DECK || fr.phase == Z_DECK_BOTTOM7) &&
            fr.topDeckOwner >= 0) {
          int ownerSide = fr.topDeckOwner;
          int oldBench = static_cast<int>(st.players[ownerSide].bench.size());
          bool countedOut = is_counted_out_top_deck_source(fr, ownerSide, fr.phase);
          std::vector<int> refs =
              countedOut ? top_deck_actual_refs(fr, fr.scratch)
                         : std::vector<int>(fr.scratch.begin(),
                                            fr.scratch.end());
          moved = move_card_refs_from_owner(st.players[fr.topDeckOwner], fr.phase,
                                            refs, o.p0, !countedOut);
          if (countedOut)
            mark_top_deck_cards_removed(fr, moved);
          if (o.p0 == D_BENCH)
            for (int j = oldBench; j < oldBench + moved; ++j)
              apply_risky_ruins_bench_entry(st, ownerSide, j);
        } else {
          int ownerSide = zone_owner_index(fr.a, fr.phase);
          int oldBench = ownerSide >= 0
                             ? static_cast<int>(st.players[ownerSide].bench.size())
                             : 0;
          moved = move_card_refs(st, fr.a, fr.phase, fr.scratch, o.p0);
          if (o.p0 == D_BENCH && ownerSide >= 0)
            for (int j = oldBench; j < oldBench + moved; ++j)
              apply_risky_ruins_bench_entry(st, ownerSide, j);
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_CHOSEN_TO_DECK_BOTTOM_ORDERED: {
        int moved = 0;
        if (fr.phase == Z_HAND && static_cast<int>(me.hand.size()) == me.handCount) {
          std::vector<int> ids;
          std::vector<int> idxs;
          for (int idx : fr.scratch) {
            if (idx < 0 || idx >= static_cast<int>(me.hand.size())) continue;
            if (std::find(idxs.begin(), idxs.end(), idx) != idxs.end()) continue;
            idxs.push_back(idx);
            ids.push_back(me.hand[idx]);
          }
          std::vector<int> erase = idxs;
          std::sort(erase.rbegin(), erase.rend());
          for (int idx : erase)
            me.hand.erase(me.hand.begin() + idx);
          me.handCount -= static_cast<int>(ids.size());
          me.deck.insert(me.deck.begin(), ids.begin(), ids.end());
          if (me.deckKnownMask.size() < me.deck.size() - ids.size())
            me.deckKnownMask.resize(me.deck.size() - ids.size(), me.deckKnown);
          me.deckKnownMask.insert(me.deckKnownMask.begin(), ids.size(), true);
          me.deckCount += static_cast<int>(ids.size());
          normalize_deck_knowledge(me);
          moved = static_cast<int>(ids.size());
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_CHOSEN_TO_BENCH_SAVE: {
        std::vector<int> saved;
        Player& owner = ((fr.phase == Z_DECK || fr.phase == Z_DECK_BOTTOM7) &&
                         fr.topDeckOwner >= 0)
                            ? st.players[fr.topDeckOwner]
                            : zone_owner(st, fr.a, fr.phase);
        int ownerSide = ((fr.phase == Z_DECK || fr.phase == Z_DECK_BOTTOM7) &&
                         fr.topDeckOwner >= 0)
                            ? fr.topDeckOwner
                            : zone_owner_index(fr.a, fr.phase);
        bool countedOut = is_counted_out_top_deck_source(fr, ownerSide, fr.phase);
        std::vector<int> refs =
            countedOut ? top_deck_actual_refs(fr, fr.scratch)
                       : std::vector<int>(fr.scratch.begin(),
                                          fr.scratch.end());
        int moved = move_card_refs_to_bench_save(owner, fr.phase, refs, saved,
                                                 !countedOut);
        if (countedOut)
          mark_top_deck_cards_removed(fr, moved);
        fr.savedSrc = saved.empty() ? -2 : saved[0];
        fr.savedScratch = saved;
        fr.savedPhase = Z_OWN_INPLAY;
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_CHOSEN_TO_TOP: {
        st.lastEffectCount = move_chosen_to_top_after_shuffle(st, fr);
        fr.pc += 1;
        break;
      }
      case OP_DECK_EVOLVE_CHOSEN: {
        int targetIdx = fr.selfBench;
        int evolvedId = apply_deck_evolve_chosen(st, fr, o.p0, o.p1 != 0, targetIdx);
        st.lastEffectCount = evolvedId ? 1 : 0;
        if (fr.attackId == 740 && !evolvedId &&
            fr.selfBench + 1 < static_cast<int>(me.bench.size())) {
          fr.selfBench += 1;
          fr.scratch.clear();
          fr.pc -= 1;
          break;
        }
        fr.pc += 1;
        if (evolvedId) {
          int actor = fr.a;
          int onProg = on_evolve_program(evolvedId);
          int legacyProg = card_program(evolvedId);
          push_program_frame(st, onProg, actor, targetIdx);
          push_program_frame(st, legacyProg, actor, targetIdx);
        }
        break;
      }
      case OP_HAND_EVOLVE_CHOSEN: {
        int targetIdx = fr.selfBench;
        int evolvedId = apply_hand_stage2_evolve_chosen(st, fr, targetIdx);
        st.lastEffectCount = evolvedId ? 1 : 0;
        fr.pc += 1;
        if (evolvedId) {
          int actor = fr.a;
          push_program_frame(st, on_evolve_program(evolvedId), actor, targetIdx);
          push_program_frame(st, card_program(evolvedId), actor, targetIdx);
        }
        break;
      }
      case OP_END_TURN:
        st.endTurnAfterProgram = true;
        finish_program(st);
        return;
      case OP_JUMP:
        fr.pc += 1 + o.p0;
        break;
      case OP_RETURN_SELF_TO_HAND: {
        bool needsPromote = false;
        bool deferActiveOutcome =
            fr.attackId > 0 && side_has_zero_hp_pokemon(st.players[1 - fr.a]);
        int moved = move_inplay_stack(st, fr.a, fr.selfBench, D_HAND, false,
                                      needsPromote, deferActiveOutcome);
        st.lastEffectCount = moved;
        fr.pc += 1;
        if (needsPromote) {
          if (run_pre_promote_immediate_damage_hooks(st, fr.a)) return;
          int actor = fr.a;
          EffectFrame promote;
          promote.effect = EFF_ABILITY_PROMOTE;
          promote.a = actor;
          st.effectStack.push_back(promote);
          set_promote_pending(st, actor);
          return;
        }
        if (st.result >= 0) {
          if (run_pre_promote_immediate_damage_hooks(st)) return;
          st.effectStack.pop_back();
          st.pending = PendingDecision();
          return;
        }
        break;
      }
      case OP_RETURN_SELF_TO_HAND_DISCARD_ATTACHMENTS: {
        bool needsPromote = false;
        InPlay replacementKo;
        bool recordReplacementKo = false;
        InPlay* self = inplay_ref(me, fr.selfBench);
        if (st.deferredPostAttack && self && self->hp <= 0 &&
            self->damagedByAttackTurn == st.turn &&
            self->damagedByAttackSide == 1 - fr.a) {
          replacementKo = *self;
          recordReplacementKo = true;
        }
        bool deferActiveOutcome =
            recordReplacementKo ||
            (fr.attackId > 0 && side_has_zero_hp_pokemon(st.players[1 - fr.a]));
        int moved = return_inplay_pokemon_to_hand_discard_attachments(
            st, fr.a, fr.selfBench, needsPromote, deferActiveOutcome);
        if (recordReplacementKo && moved > 0) {
          int taker = 1 - fr.a;
          st.replacementKoPairs.push_back(taker);
          st.replacementKoPairs.push_back(
              contextual_prize_value(st, taker, fr.a, replacementKo,
                                     fr.selfBench < 0));
          st.replacementKoGroups.push_back(
              static_cast<int>(st.replacementKoGroups.size()));
          record_attack_damage_ko_marker(st, fr.a, replacementKo);
          record_ko(st, fr.a, replacementKo.id);
          needsPromote = false;
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        if (needsPromote) {
          if (run_pre_promote_immediate_damage_hooks(st, fr.a)) return;
          int actor = fr.a;
          EffectFrame promote;
          promote.effect = EFF_ABILITY_PROMOTE;
          promote.a = actor;
          st.effectStack.push_back(promote);
          set_promote_pending(st, actor);
          return;
        }
        if (st.result >= 0) {
          if (run_pre_promote_immediate_damage_hooks(st)) return;
          st.effectStack.pop_back();
          st.pending = PendingDecision();
          return;
        }
        break;
      }
      case OP_DISCARD_SELF_STACK: {
        int actor = fr.a;
        bool needsPromote = false;
        bool deferActiveOutcome =
            fr.attackId > 0 && side_has_zero_hp_pokemon(st.players[1 - actor]);
        int moved = move_inplay_stack(st, actor, fr.selfBench, D_DISCARD, false,
                                      needsPromote, deferActiveOutcome);
        st.lastEffectCount = moved;
        fr.pc += 1;
        if (needsPromote) {
          if (run_pre_promote_immediate_damage_hooks(st, actor)) return;
          EffectFrame promote;
          promote.effect = EFF_ABILITY_PROMOTE;
          promote.a = actor;
          st.effectStack.push_back(promote);
          set_promote_pending(st, actor);
          return;
        }
        if (st.result >= 0) {
          if (run_pre_promote_immediate_damage_hooks(st)) return;
          st.effectStack.pop_back();
          st.pending = PendingDecision();
          return;
        }
        break;
      }
      case OP_RETURN_CHOSEN_TO_HAND: {
        if (fr.scratch.empty()) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        int idx = fr.scratch[0];
        bool needsPromote = false;
        int moved = move_inplay_stack(st, fr.a, idx, D_HAND, false, needsPromote);
        st.lastEffectCount = moved;
        fr.pc += 1;
        if (needsPromote) {
          int actor = fr.a;
          EffectFrame promote;
          promote.effect = EFF_ABILITY_PROMOTE;
          promote.a = actor;
          st.effectStack.push_back(promote);
          set_promote_pending(st, actor);
          return;
        }
        if (st.result >= 0) {
          st.effectStack.pop_back();
          st.pending = PendingDecision();
          return;
        }
        break;
      }
      case OP_SHUFFLE_CHOSEN_INTO_DECK: {
        int ownerIdx = (o.p0 == S_OPP) ? 1 - fr.a : fr.a;
        if (fr.scratch.empty()) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        int idx = fr.scratch[0];
        bool needsPromote = false;
        InPlay* target = inplay_ref(st.players[ownerIdx], idx);
        if (ownerIdx == 1 - fr.a && target &&
            attack_effects_prevented(st, ownerIdx, *target)) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        int moved = move_inplay_stack(st, ownerIdx, idx, D_DECK, true, needsPromote);
        st.lastEffectCount = moved;
        fr.pc += 1;
        if (needsPromote) {
          EffectFrame promote;
          promote.effect = EFF_ABILITY_PROMOTE;
          promote.a = ownerIdx;
          st.effectStack.push_back(promote);
          set_promote_pending(st, ownerIdx);
          return;
        }
        if (st.result >= 0) {
          st.effectStack.pop_back();
          st.pending = PendingDecision();
          return;
        }
        break;
      }
      case OP_OPP_SWITCH_OUT: {
        Player& opp = st.players[1 - fr.a];
        if (opp.activeKnown && attack_effects_prevented(st, 1 - fr.a, opp.active)) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        if (opp.bench.empty()) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        PendingDecision pd;
        pd.context = 3;  // SWITCH
        pd.minCount = 1;
        pd.maxCount = 1;
        for (int j = 0; j < static_cast<int>(opp.bench.size()); ++j)
          pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                                Atom::I(opp.bench[j].id)});
        st.pending = pd;
        st.yourIndex = 1 - fr.a;
        return;
      }
      case OP_OPP_DISCARD_TO_N: {
        Player& opp = st.players[1 - fr.a];
        int keep = std::max(0, o.p0);
        int need = std::max(0, opp.handCount - keep);
        if (need <= 0) {
          st.discardedCount = 0;
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        PendingDecision pd;
        pd.context = 8;  // DISCARD
        pd.minCount = need;
        pd.maxCount = need;
        bool known = static_cast<int>(opp.hand.size()) == opp.handCount;
        for (int i = 0; i < opp.handCount; ++i) {
          if (known)
            pd.options.push_back({Atom::S("CARD"), Atom::S("HAND"),
                                  Atom::I(i), Atom::I(opp.hand[i])});
          else
            pd.options.push_back({Atom::S("CARD"), Atom::S("HAND"),
                                  Atom::I(i), Atom::N()});
        }
        st.pending = pd;
        st.yourIndex = 1 - fr.a;
        return;
      }
      case OP_SET_ACTIVE_EX_DAMAGE_BUFF:
        st.activeExDamageBuffTurn[fr.a] = st.turn;
        st.activeExDamageBuffAmount[fr.a] += o.p0;
        st.lastEffectCount = 1;
        fr.pc += 1;
        break;
      case OP_SET_PRIZE_BONUS:
        st.prizeBonusTurn[fr.a] = st.turn;
        st.prizeBonusAmount[fr.a] += o.p0;
        st.prizeBonusKind[fr.a] = o.p1;
        st.lastEffectCount = 1;
        fr.pc += 1;
        break;
      case OP_DEVOLVE_CHOSEN: {
        int count = (o.p2 == -1)
                        ? (fr.scratch.empty() ? 0 : std::max(0, fr.scratch[0]))
                        : o.p2;
        st.lastEffectCount = devolve_selected(st, fr, o.p0, o.p1, count,
                                              o.p3 != 0);
        fr.pc += 1;
        break;
      }
      case OP_DEVOLVE_ALL:
        st.lastEffectCount = devolve_all_matching(st, fr, o.p0, o.p1, o.p2,
                                                  o.p3);
        fr.pc += 1;
        break;
      case OP_SELF_ATTACK_BONUS: {
        InPlay* self = inplay_ref(st.players[fr.a], fr.selfBench);
        if (self) {
          self->attackBonusTurn = st.turn;
          self->attackBonus += o.p0;
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_DECK_EVOLVE_SAVED: {
        std::vector<int> targets = fr.savedScratch;
        std::sort(targets.begin(), targets.end());
        targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
        std::vector<std::pair<int, int>> evolved;
        for (int idx : targets) {
          int evolvedId = auto_deck_evolve_target(st, fr.a, idx, o.p0 != 0);
          if (evolvedId) evolved.push_back({evolvedId, idx});
        }
        shuffle_deck_known(me, st.rng);
        st.lastEffectCount = static_cast<int>(evolved.size());
        fr.pc += 1;
        push_evolve_hooks(st, fr.a, evolved);
        break;
      }
      case OP_DECK_EVOLVE_BENCH: {
        std::vector<std::pair<int, int>> evolved;
        for (int idx = 0; idx < static_cast<int>(me.bench.size()); ++idx) {
          int evolvedId = auto_deck_evolve_target(st, fr.a, idx, o.p0 != 0);
          if (evolvedId) evolved.push_back({evolvedId, idx});
        }
        shuffle_deck_known(me, st.rng);
        st.lastEffectCount = static_cast<int>(evolved.size());
        fr.pc += 1;
        push_evolve_hooks(st, fr.a, evolved);
        break;
      }
      case OP_SHUFFLE_DECK:
        shuffle_deck_known(me, st.rng);
        fr.pc += 1;
        break;
      case OP_TOP_DECK_REST: {
        Player& owner = (fr.topDeckOwner >= 0) ? st.players[fr.topDeckOwner] : me;
        int window = fr.topDeckCount > 0 ? fr.topDeckCount : o.p0;
        int selected = fr.topDeckCount > 0
                           ? fr.topDeckSelectedCount
                           : static_cast<int>(fr.scratch.size());
        int n = std::max(0, window - selected);
        n = std::min(n, static_cast<int>(owner.deck.size()));
        int countedOutBefore = fr.topDeckCountedOut;
        if (o.p1 == TR_DISCARD) {
          for (int i = 0; i < n; ++i) {
            int id = owner.deck.back();
            consume_known_card(owner.deckKnownCards, id);
            owner.discard.push_back(owner.deck.back());
            owner.deck.pop_back();
            if (!owner.deckKnownMask.empty()) owner.deckKnownMask.pop_back();
          }
          int alreadyOut = std::min(countedOutBefore, n);
          int decrement = n - alreadyOut;
          if (decrement > 0)
            owner.deckCount -= decrement;
          int stillOutInDeck = std::max(0, countedOutBefore - alreadyOut);
          if (stillOutInDeck > 0)
            owner.deckCount += stillOutInDeck;
        } else if (o.p1 == TR_BOTTOM) {
          for (int i = 0; i < n; ++i) {
            bool known = owner.deckKnown ||
                         (!owner.deckKnownMask.empty() &&
                          owner.deckKnownMask.back());
            int id = owner.deck.back();
            known = consume_known_card(owner.deckKnownCards, id) || known;
            owner.deck.pop_back();
            if (!owner.deckKnownMask.empty()) owner.deckKnownMask.pop_back();
            owner.deck.insert(owner.deck.begin(), id);
            owner.deckKnownMask.insert(owner.deckKnownMask.begin(), known);
          }
          owner.deckCount += countedOutBefore;
        } else if (o.p1 == TR_SHUFFLE_BOTTOM) {
          std::vector<int> rest;
          std::vector<bool> restKnown;
          for (int i = 0; i < n; ++i) {
            restKnown.push_back(owner.deckKnown ||
                                (!owner.deckKnownMask.empty() &&
                                 owner.deckKnownMask.back()));
            rest.push_back(owner.deck.back());
            if (consume_known_card(owner.deckKnownCards, owner.deck.back()))
              restKnown.back() = true;
            owner.deck.pop_back();
            if (!owner.deckKnownMask.empty()) owner.deckKnownMask.pop_back();
          }
          shuffle_deck(rest, st.rng);
          for (size_t i = 0; i < rest.size(); ++i)
            if (i < restKnown.size() && restKnown[i])
              add_known_card(owner.deckKnownCards, rest[i]);
          owner.deck.insert(owner.deck.begin(), rest.begin(), rest.end());
          owner.deckKnownMask.insert(owner.deckKnownMask.begin(), rest.size(), false);
          owner.deckCount += countedOutBefore;
        } else {
          shuffle_deck_known(owner, st.rng);
          owner.deckCount += countedOutBefore;
        }
        fr.topDeckCount = 0;
        fr.topDeckSelectedCount = 0;
        fr.topDeckOwner = -1;
        fr.topDeckCountedOut = 0;
        fr.topDeckStart = 0;
        if (fr.savedPhase == Z_DECK) {
          fr.savedPhase = -1;
          fr.savedScratch.clear();
        }
        fr.pc += 1;
        break;
      }
      case OP_SET_FLAG:
        if (o.p0 == FLAG_LUNAR_USED) st.lunarUsedThisTurn = true;
        fr.pc += 1;
        break;
      case OP_MARK_SELF_ABILITY_USED: {
        InPlay* self = chosen_target(me, fr, AT_SELF);
        if (self) {
          self->abilityUsedThisTurn = true;
          int group = ability_limit_group(self->id);
          if (group > 0 && group < 16)
            st.abilityGroupUsedTurn[fr.a][group] = st.turn;
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_YESNO: {
        PendingDecision pd;
        pd.context = o.p0;
        pd.options.push_back({Atom::S("YES")});
        pd.options.push_back({Atom::S("NO")});
        st.pending = pd;
        return;  // suspend for the YES/NO choice
      }
      case OP_IF_YES:
        if (!fr.scratch.empty() && fr.scratch[0] == 0)  // YES -> run the body
          fr.pc += 1;
        else  // NO -> skip the body
          fr.pc += 1 + o.p0;
        break;
      case OP_REQUIRE:
        if (eval_cond(st, o.p0)) {
          fr.pc += 1;
        } else {
          st.lastEffectCount = 0;
          fr.pc += 1 + o.p1;  // condition fails -> skip the rest
        }
        break;
      case OP_DAMAGE: {
        Player& opp = st.players[1 - fr.a];
        auto printed_base_damage = [&]() {
          return fr.copiedAttackBaseDamage >= 0
                     ? fr.copiedAttackBaseDamage
                     : attack_base_damage(me.active.id, fr.attackId);
        };
        int base;
        bool damageEffectApplies = true;
        switch (o.p0) {
          case AMT_CONST: base = o.p1; break;
          case AMT_COND_CONST:
            damageEffectApplies = eval_cond(st, o.p2);
            base = damageEffectApplies ? o.p1 : 0;
            break;
          case AMT_PER_HEADS: base = st.coinHeads * o.p1; break;
          case AMT_BASE_PLUS_HEADS:
            base = printed_base_damage() + st.coinHeads * o.p1;
            break;
          case AMT_BASE_PLUS_PER:  // base + per * count(source); p4 = energy-type filter
            base = printed_base_damage() + o.p1 * count_source(st, fr.a, o.p2, o.p4);
            break;
          case AMT_BASE_PLUS_IF:  // base + bonus if cond holds
            base = printed_base_damage() + (eval_cond(st, o.p2) ? o.p1 : 0);
            break;
          case AMT_BASE_PLUS_COUNTREG:  // base + p1 * st.countReg (from OP_COUNT)
            base = printed_base_damage() + o.p1 * st.countReg;
            break;
          case AMT_BASE_PLUS_IF_SOURCE_GE: {
            int threshold = o.p4 / 100;
            int etype = (o.p4 % 100) - 1;
            base = printed_base_damage() +
                   (count_source(st, fr.a, o.p2, etype) >= threshold ? o.p1 : 0);
            break;
          }
          case AMT_COND_BASE_OVERRIDE:
            base = eval_cond(st, o.p2) ? o.p1 : printed_base_damage();
            break;
          default:  // AMT_BASE
            base = printed_base_damage();
            break;
        }
        bool ownPrintedAttack = me.activeKnown &&
                                card_attack(me.active.id, fr.attackId) != nullptr;
        if (ownPrintedAttack &&
            me.active.nextAttackBonusTurn == st.turn &&
            me.active.nextAttackBonusId == fr.attackId) {
          if (me.active.nextAttackSetBase >= 0)
            base = me.active.nextAttackSetBase;
          base += me.active.nextAttackBonus;
        }
        if (fr.attackId == 1130 && me.activeKnown &&
            me.active.damagedByAttackTurn == st.turn - 1 &&
            me.active.damagedByAttackSide == 1 - fr.a &&
            me.active.damagedByAttackAmount > 0)
          base += me.active.damagedByAttackAmount;
        int dmg = base;
        st.lastAttackDamage = 0;
        const CardInfo* aci = find_card(me.active.id);
        if (base > 0 && opp.activeKnown && opp.active.id == 1150)
          dmg = dmg > 30 ? dmg - 30 : 0;  // Antique Jaw Fossil, before W/R.
        if (base > 0 && aci && aci->energyType == FIGHTING)
          dmg += st.fightingBuff;  // Premium Power Pro (before weakness/resist)
        if (base > 0 && me.active.attackBonusTurn == st.turn)
          dmg += me.active.attackBonus;
        if (base > 0 && aci && is_future_card(me.active.id) && me.active.id != 80)
          dmg += 20 * in_play_count_id(me, 80);  // Iron Crown ex: Cobalt Command
        if (base > 0 && me.active.id == 901)
          dmg += 30 * (opp.prizeCount < 6 ? 6 - opp.prizeCount : 0);
        if (base > 0 && st.activeExDamageBuffTurn[fr.a] == st.turn &&
            opp.activeKnown) {
          const CardInfo* dci = find_card(opp.active.id);
          if (dci && (dci->ex || dci->megaEx))
            dmg += st.activeExDamageBuffAmount[fr.a];
        }
        if (base > 0 && opp.activeKnown) {
          dmg += passive_attack_damage_delta(st, fr.a, me.active, opp.active,
                                             true, dmg);
          if (dmg < 0) dmg = 0;
        }
        if (me.active.attackDmgReduceTurn == st.turn && dmg > 0)
          dmg = dmg > me.active.attackDmgReduce ? dmg - me.active.attackDmgReduce : 0;
        if (base > 0 && !(o.p3 & DMG_IGNORE_WR) && opp.activeKnown) {
          const CardInfo* dci = find_card(opp.active.id);
          if (dci && aci) {
            int weakness = effective_weakness(st, fr.a, dci);
            if (!(o.p3 & DMG_IGNORE_WEAKNESS) &&
                opp.active.noWeaknessTurn != st.turn &&
                weakness >= 0 && pokemon_has_type(me.active.id, weakness))
              dmg *= 2;
            if (!(o.p3 & DMG_IGNORE_RESISTANCE) &&
                dci->resistance >= 0 && pokemon_has_type(me.active.id, dci->resistance))
              dmg = dmg > 30 ? dmg - 30 : 0;
          }
        }
        std::vector<DamageTrigger> damageTriggers;
        if (opp.activeKnown) {
          int defenderId = opp.active.id;
          if (damageEffectApplies && !(o.p3 & DMG_IGNORE_EFFECTS))
            dmg = apply_defender_attack_damage_mods(st, me.active, 1 - fr.a,
                                                    opp.active, false, dmg);
          if (dmg < 0) dmg = 0;
          if (damageEffectApplies && (o.p3 & DMG_IGNORE_EFFECTS) && dmg > 0) {
            const CardInfo* attackerInfo = find_card(me.active.id);
            if (attackerInfo && attackerInfo->energyType == PSYCHIC)
              discard_tool(opp, opp.active, 1164);
            if (attackerInfo && attackerInfo->energyType == DRAGON)
              discard_tool(opp, opp.active, 1170);
          }
          int beforeHp = opp.active.hp;
          opp.active.hp -= dmg;
          if (opp.active.hp < 0) opp.active.hp = 0;  // HP never shown negative
          st.lastAttackDamage = std::max(0, dmg);
          if (dmg > 0) {
            apply_damaged_by_attack_reactive(st, opp.active, fr.a, dmg, false,
                                             beforeHp);
            int defenderSide = 1 - fr.a;
            int prog = on_damage_program(defenderId);
            if (prog >= 0 &&
                !(defenderId == 1059 && opp.active.hp > 0))
              damageTriggers.push_back({prog, defenderSide, -1, defenderId});
            if (pokemon_has_type(opp.active.id, DARKNESS)) {
              for (int j = 0; j < static_cast<int>(opp.bench.size()); ++j) {
                InPlay& b = opp.bench[j];
                if (b.id != 688 ||
                    ability_suppressed(st, defenderSide, b, false))
                  continue;
                int spiritombProg = on_damage_program(688);
                if (spiritombProg >= 0)
                  damageTriggers.push_back({spiritombProg, defenderSide, j, 688});
              }
            }
            if (!tool_effects_disabled(st)) {
              for (int toolId : opp.active.tools) {
                if (toolId == 1176 && !pokemon_has_type(opp.active.id, DARKNESS))
                  continue;
                if (toolId == 1154 && player_has_powerglass_source(st, me))
                  continue;
                int toolProg = on_damage_program(toolId);
                if (toolProg >= 0)
                  damageTriggers.push_back({toolProg, defenderSide, -1, toolId});
              }
            }
            for (int energyId : opp.active.energyCardIds) {
              int energyProg = on_damage_program(energyId);
              if (energyProg >= 0)
                damageTriggers.push_back({energyProg, defenderSide, -1, energyId});
            }
          }
        }
        fr.pc += 1;
        if (!queue_damage_trigger_order_frame(st, damageTriggers)) {
          for (auto it = damageTriggers.rbegin(); it != damageTriggers.rend(); ++it)
            queue_program_frame(st, std::get<0>(*it), std::get<1>(*it),
                                std::get<2>(*it), true, std::get<3>(*it));
        }
        break;
      }
      case OP_HEAL_SELF_LAST_DAMAGE: {
        int healed = 0;
        if (me.activeKnown && st.lastAttackDamage > 0) {
          int before = me.active.hp;
          me.active.hp = std::min(me.active.maxHp,
                                  me.active.hp + st.lastAttackDamage);
          healed = me.active.hp - before;
          if (healed > 0) me.active.healedThisTurn = true;
        }
        st.lastEffectCount = healed / 10;
        fr.pc += 1;
        break;
      }
      case OP_RECOIL:
        if (me.activeKnown) {
          int beforeHp = me.active.hp;
          int recoil = std::max(0, o.p0);
          if (me.active.attackDmgReduceTurn == st.turn && recoil > 0)
            recoil = recoil > me.active.attackDmgReduce
                         ? recoil - me.active.attackDmgReduce
                         : 0;
          if (has_energy_type(me.active, METAL))
            recoil = std::max(0, recoil - 20 * in_play_count_id(me, 623));
          const CardInfo* selfCard = find_card(me.active.id);
          if (recoil > 0 && selfCard && selfCard->energyType == PSYCHIC)
            discard_tool(me, me.active, 1164);
          if (recoil > 0 && selfCard && selfCard->energyType == DRAGON)
            discard_tool(me, me.active, 1170);
          me.active.hp -= recoil;
          if (me.active.hp < 0) me.active.hp = 0;
          if (beforeHp > 0 && me.active.hp <= 0 &&
              queued_attacker_damage_trigger(st) && st.checkupKoFirst < 0)
            st.checkupKoFirst = fr.a;
        }
        fr.pc += 1;
        break;
      case OP_SELF_LOCK:
        if (me.activeKnown) {
          if (o.p0) {
            me.active.activeLockId = fr.attackId;  // until it leaves Active
          } else {
            me.active.lockId = fr.attackId;
            me.active.lockTurn = st.turn + 2;  // can't use it next turn
          }
        }
        fr.pc += 1;
        break;
      case OP_ATTACH_CHOSEN:  // attach one Basic {F} Energy from discard to bench
        if (!fr.scratch.empty()) {
          int benchIdx = fr.scratch[0];
          remove_one(me.discard, 6);
          me.bench[benchIdx].energies.push_back(FIGHTING);
          push_attached_energy_card(me.bench[benchIdx], 6);
          apply_energy_attach_reactive(st, me.bench[benchIdx], false);
        }
        fr.pc += 1;
        break;
      case OP_FOREACH_CHOSEN:
        if (fr.loopRemain < 0) fr.loopRemain = static_cast<int>(fr.scratch.size());
        if (fr.loopRemain == 0) {
          fr.loopRemain = -1;
          fr.pc += 2 + o.p0;  // skip the body + the LOOP_BACK
        } else {
          fr.loopRemain -= 1;
          fr.pc += 1;  // enter the body
        }
        break;
      case OP_LOOP_BACK:
        fr.pc -= o.p0;  // jump back to the FOREACH
        break;
      case OP_DISCARD_HAND:
        st.lastEffectCount = discard_hand_to_discard(me);
        fr.pc += 1;
        break;
      case OP_DISCARD_HAND_TO_N: {
        Player& who = (o.p0 == S_OPP) ? st.players[1 - fr.a] : me;
        int discarded = discard_hand_to_size(who, o.p1);
        st.discardedCount = discarded;
        st.lastEffectCount = discarded;
        fr.pc += 1;
        break;
      }
      case OP_RANDOM_DISCARD_HAND_TO_N: {
        Player& who = (o.p0 == S_OPP) ? st.players[1 - fr.a] : me;
        int discarded = random_discard_hand_to_size(st, who, o.p1);
        st.discardedCount = discarded;
        st.lastEffectCount = discarded;
        fr.pc += 1;
        break;
      }
      case OP_DRAW: {
        Player& who = (o.p2 == 1) ? st.players[1 - fr.a] : me;  // p2=1 -> opponent
        int n = 0;
        if (o.p0 == CNT_CONST) {
          n = o.p1;
        } else if (o.p0 == CNT_PRIZE_6_8) {
          n = who.prizeCount == 6 ? 8 : 6;
        } else if (o.p0 == CNT_SOURCE) {
          n = count_source(st, fr.a, o.p1, o.p3);
        } else if (o.p0 == CNT_COUNTREG) {
          n = st.countReg;
        }
        st.lastEffectCount = draw_n(st, who, n);
        fr.pc += 1;
        break;
      }
      case OP_DRAW_UNTIL_COUNTREG: {
        int need = st.countReg - me.handCount;
        st.lastEffectCount = need > 0 ? draw_n(st, me, need) : 0;
        fr.pc += 1;
        break;
      }
      case OP_CHOOSE_OPP_PRIZE: {
        Player& opp = st.players[1 - fr.a];
        PendingDecision pd;
        pd.context = 43; // CTX_ACTIVATE
        pd.minCount = opp.prizeCount > 0 ? 1 : 0;
        pd.maxCount = pd.minCount;
        for (int i = 0; i < opp.prizeCount; ++i) {
          if (i < static_cast<int>(opp.prizeFaceUp.size()) && opp.prizeFaceUp[i])
            continue;
          Atom card = Atom::N();
          pd.options.push_back({Atom::S("CARD"), Atom::S("PRIZE"),
                                Atom::I(i), card});
        }
        st.pending = pd;
        if (pd.options.empty()) {
          st.pending = PendingDecision();
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        return;
      }
      case OP_REDEEMABLE_TICKET: {
        int n = me.prizeCount;
        fr.savedScratch.clear();
        if (n <= 0) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        if (!me.prizes.empty()) {
          for (int id : me.prizes) push_deck_bottom(me, id, false);
          // Ticket shuffles the former Prizes into the whole deck before
          // dealing a new face-down Prize set.
          shuffle_deck(me.deck, st.rng);
        } else {
          me.deckCount += n;
        }
        me.prizes.clear();
        me.prizeFaceUp.clear();
        me.prizesKnownMask.clear();
        me.prizesKnownCards.clear();
        me.prizeCount = 0;
        me.prizesKnown = false;

        const int take = std::min(n, me.deckCount);
        const int represented =
            std::min(take, static_cast<int>(me.deck.size()));
        for (int i = 0; i < represented; ++i) {
          const int card_id =
              erase_deck_at(me, static_cast<int>(me.deck.size()) - 1);
          me.prizes.push_back(card_id);
          me.prizeFaceUp.push_back(false);
          me.prizesKnownMask.push_back(false);
        }
        me.deckCount -= take - represented;
        me.prizeCount = take;

        // The new partition between deck and Prizes is hidden. Previous exact
        // knowledge applies only to their union, so both zones fail closed.
        me.deckKnown = false;
        me.deckKnownMask.clear();
        me.deckKnownCards.clear();
        me.ownDeckInspected = false;
        me.ownPrizesInferred = false;
        refresh_deck_prize_union(me);
        st.lastEffectCount = take;
        fr.pc += 1;
        break;
      }
      case OP_SHUFFLE_HAND_INTO_DECK: {
        Player& who = (o.p0 == S_OPP) ? st.players[1 - fr.a] : me;
        int moved = who.handCount;
        if (static_cast<int>(who.hand.size()) == who.handCount) {
          for (int id : who.hand) push_deck_top(who, id, true);
          shuffle_deck_known(who, st.rng);
        } else {
          for (int id : who.handKnownCards)
            add_known_card(who.deckKnownCards, id);
          who.deckCount += who.handCount;
          who.deckKnown = false;
          who.deckKnownMask.clear();
        }
        who.hand.clear();
        who.handKnownCards.clear();
        who.handCount = 0;
        who.handKnown = true;
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_SET_BUFF:
        st.fightingBuff += o.p0;
        fr.pc += 1;
        break;
      case OP_FLIP: {
        int heads = 0;
        int flips = 0;
        if (o.p0 == FLIP_FIXED || o.p0 == FLIP_COUNT_SOURCE ||
            o.p0 == FLIP_COUNTREG) {
          flips = o.p1;
          if (o.p0 == FLIP_COUNT_SOURCE)
            flips = count_source(st, fr.a, o.p1, o.p2);
          else if (o.p0 == FLIP_COUNTREG)
            flips = st.countReg;
          for (int i = 0; i < flips; ++i)
            if (flip_heads(st)) ++heads;
        } else {  // FLIP_UNTIL_TAILS
          while (true) {
            ++flips;
            if (!flip_heads(st)) break;
            ++heads;
          }
        }
        st.coinHeads = heads;
        st.coinFlips = flips;
        st.lastEffectCount = heads;
        fr.pc += 1;
        break;
      }
      case OP_IF_HEADS:
        fr.pc += (st.coinHeads > 0) ? 1 : (1 + o.p0);  // skip body on 0 heads
        break;
      case OP_APPLY_STATUS: {
        if (fr.sourceCardId == 1154 && st.players[fr.a].activeKnown &&
            !is_team_rocket_card_id(st.players[fr.a].active.id)) {
          fr.pc += 1;
          break;
        }
        int targetSide = (o.p0 == S_OWN) ? fr.a : 1 - fr.a;
        Player& tgt = st.players[targetSide];
        if (tgt.activeKnown &&
            !festival_status_immunity(st, targetSide) &&
            !attack_effects_prevented(st, targetSide, tgt.active)) {
          apply_condition(tgt, o.p1);
          if (fr.attackId == 642 && o.p1 == ST_POISON)
            tgt.poisonDamageCounters = 8;
        }
        fr.pc += 1;
        break;
      }
      case OP_APPLY_CHOSEN_STATUS: {
        int targetSide = (o.p0 == S_OWN) ? fr.a : 1 - fr.a;
        Player& tgt = st.players[targetSide];
        int status = fr.scratch.empty() ? -1 : fr.scratch[0];
        if (status >= ST_POISON && status <= ST_CONFUSED && tgt.activeKnown &&
            !festival_status_immunity(st, targetSide) &&
            !attack_effects_prevented(st, targetSide, tgt.active)) {
          apply_condition(tgt, status);
        }
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_CHOSEN: {  // discard selected cards or energy refs
        std::vector<int> idxs = fr.scratch;
        std::sort(idxs.rbegin(), idxs.rend());  // erase from the back first
        int discarded = static_cast<int>(idxs.size());
        if (fr.phase == Z_OWN_ACTIVE_ENERGY ||
            fr.phase == Z_OWN_INPLAY_ENERGY ||
            fr.phase == Z_OWN_BENCH_ENERGY ||
            fr.phase == Z_SAVED_OWN_INPLAY_ENERGY ||
            fr.phase == Z_SELF_ENERGY) {
          discarded = discard_energy_refs(st, me, idxs);
        } else if (fr.phase == Z_OPP_ACTIVE_ENERGY ||
                   fr.phase == Z_OPP_INPLAY_ENERGY ||
                   fr.phase == Z_OPP_BENCH_ENERGY ||
                   fr.phase == Z_SAVED_OPP_INPLAY_ENERGY) {
          Player& oppOwner = st.players[1 - fr.a];
          std::vector<int> allowed;
          for (int ref : idxs) {
            InPlay* p = inplay_ref(oppOwner, energy_ref_inplay(ref));
            if (p && !attack_effects_prevented(st, 1 - fr.a, *p))
              allowed.push_back(ref);
          }
          discarded = discard_energy_refs(st, oppOwner, allowed);
        } else if (fr.phase == Z_HAND || fr.phase == Z_HAND_ENERGY ||
                   fr.phase == Z_OPP_HAND ||
                   fr.phase == Z_OPP_HAND_BY_OPP) {
          Player& owner = zone_owner(st, fr.a, fr.phase);
          for (int k : idxs)
            if (k < static_cast<int>(owner.hand.size())) {
              owner.discard.push_back(owner.hand[k]);
              owner.hand.erase(owner.hand.begin() + k);
              owner.handCount -= 1;
            }
        } else if (fr.phase == Z_INPLAY_TOOLS) {
          discarded = 0;
          for (int ref : idxs) {
            if (ref < 300000) continue;
            int ownerOffset = any_tool_owner_offset(ref);
            if (ownerOffset < 0 || ownerOffset > 1) continue;
            Player& owner = st.players[ownerOffset == 0 ? fr.a : 1 - fr.a];
            InPlay* p = inplay_ref(owner, any_tool_inplay_idx(ref));
            int toolIdx = any_tool_tool_idx(ref);
            if (!p || toolIdx < 0 || toolIdx >= static_cast<int>(p->tools.size()))
              continue;
            if (ownerOffset == 1 && attack_effects_prevented(st, 1 - fr.a, *p))
              continue;
            int damage = p->maxHp - p->hp;
            owner.discard.push_back(p->tools[toolIdx]);
            erase_tool_at(*p, toolIdx);
            p->maxHp = effective_max_hp(p->id, p->tools, st,
                                        p->energyCardIds, p->energies,
                                        ownerOffset == 0 ? fr.a : 1 - fr.a);
            p->hp = std::max(0, p->maxHp - damage);
            ++discarded;
          }
        }
        st.discardedCount = discarded;
        st.lastEffectCount = discarded;
        fr.pc += 1;
        break;
      }
      case OP_PLACE_DAMAGE: {  // counters = p1 (fixed/chosen) OR p1-per * count(source)
        int cnt = 0;
        if (o.p2 >= 0) {
          cnt = (o.p1 > 0 ? o.p1 : 1) * count_source(st, fr.a, o.p2, o.p4);
        } else if (o.p1 == -1) {
          cnt = fr.scratch.empty() ? 0 : std::max(0, fr.scratch[0]);
        } else if (o.p1 == -2) {
          cnt = std::max(0, st.countReg);
        } else if (o.p1 <= -100) {
          cnt = std::max(0, st.countReg) * (-100 - o.p1);
        } else {
          cnt = std::max(0, o.p1);
        }
        int dmg = 10 * cnt;
        Player& opp = st.players[1 - fr.a];
        int placed = 0;
        auto mark_trigger_ko_first = [&st, fr](int beforeHp, const InPlay& p) {
          if (beforeHp > 0 && p.hp <= 0 && st.deferredPostAttack &&
              fr.attackId == 0 && fr.sourceCardId > 0 &&
              st.checkupKoFirst < 0)
            st.checkupKoFirst = fr.a;
        };
        bool attackDamage = (o.p3 & PDMG_ATTACK) != 0;
        bool ignoreEffects = (o.p3 & PDMG_IGNORE_EFFECTS) != 0;
        auto own_hit = [&st, fr, &me, attackDamage, ignoreEffects,
                        &mark_trigger_ko_first, dmg, cnt, &placed](
                           InPlay& p, int ownerSide = -1, bool bench = false) {
          if (dmg <= 0) return;
          int d = dmg;
          if (attackDamage && !ignoreEffects && ownerSide >= 0 && me.activeKnown)
            d = apply_defender_attack_damage_mods(st, me.active, ownerSide,
                                                  p, bench, dmg);
          if (d <= 0) return;
          int beforeHp = p.hp;
          p.hp -= d;
          if (p.hp < 0) p.hp = 0;
          if (attackDamage && d > 0 && ownerSide >= 0)
            apply_damaged_by_attack_reactive(st, p, fr.a, d, bench, beforeHp);
          mark_trigger_ko_first(beforeHp, p);
          placed += d / 10;
        };
        auto opp_hit = [&st, fr, &me, dmg, attackDamage, ignoreEffects,
                        &mark_trigger_ko_first, &placed](InPlay& p, bool bench) {
          if (dmg <= 0) return;
          if (!attackDamage && bench_damage_counters_prevented(st, 1 - fr.a, bench))
            return;
          if (!attackDamage && attack_effects_prevented(st, 1 - fr.a, p))
            return;
          int d = attackDamage && !ignoreEffects
                      ? apply_defender_attack_damage_mods(st, me.active, 1 - fr.a,
                                                          p, bench, dmg)
                      : dmg;
          int beforeHp = p.hp;
          p.hp -= d;
          if (p.hp < 0) p.hp = 0;
          if (attackDamage && d > 0)
            apply_damaged_by_attack_reactive(st, p, fr.a, d, bench, beforeHp);
          mark_trigger_ko_first(beforeHp, p);
          placed += d / 10;
        };
        auto double_opp_counters = [&st, fr, &mark_trigger_ko_first,
                                    &placed](InPlay& p, bool bench) {
          if (bench_damage_counters_prevented(st, 1 - fr.a, bench)) return;
          if (attack_effects_prevented(st, 1 - fr.a, p)) return;
          int d = std::max(0, p.maxHp - p.hp);
          if (d <= 0) return;
          int beforeHp = p.hp;
          p.hp -= d;
          if (p.hp < 0) p.hp = 0;
          mark_trigger_ko_first(beforeHp, p);
          placed += d / 10;
        };
        auto direct_counters = [&st, fr, dmg, cnt, &mark_trigger_ko_first,
                                &placed](InPlay& p, int side = -1,
                                          bool bench = false) {
          if (dmg <= 0) return;
          if (side >= 0 && bench_damage_counters_prevented(st, side, bench)) return;
          if (side >= 0 && attack_effects_prevented(st, side, p)) return;
          int beforeHp = p.hp;
          p.hp -= dmg;
          if (p.hp < 0) p.hp = 0;
          mark_trigger_ko_first(beforeHp, p);
          placed += cnt;
        };
        auto has_ability_target = [&st](int side, const InPlay& p, bool isActive,
                                        bool exceptFroslass) {
          const CardInfo* c = find_card(p.id);
          if (!c || !c->hasAbility || ability_suppressed(st, side, p, isActive))
            return false;
          return !exceptFroslass || !name_contains(c->name, "Froslass");
        };
        if (o.p0 == PT_OPP_ACTIVE) {
          if (opp.activeKnown) opp_hit(opp.active, false);
        } else if (o.p0 == PT_SELF) {
          InPlay* self = inplay_ref(me, fr.selfBench);
          if (self) own_hit(*self);
        } else if (o.p0 == PT_CHOSEN_OPP_BENCH) {
          for (int idx : fr.scratch)
            if (idx >= 0 && idx < static_cast<int>(opp.bench.size()))
              opp_hit(opp.bench[idx], true);
        } else if (o.p0 == PT_EACH_OPP_BENCH) {
          for (auto& b : opp.bench) opp_hit(b, true);
        } else if (o.p0 == PT_CHOSEN_OPP_INPLAY) {
          for (int idx : fr.scratch) {
            if (idx < 0) {
              if (opp.activeKnown) opp_hit(opp.active, false);
            } else if (idx < static_cast<int>(opp.bench.size())) {
              opp_hit(opp.bench[idx], true);
            }
          }
        } else if (o.p0 == PT_SAVED_OPP_INPLAY) {
          for (int idx : fr.savedScratch) {
            if (idx < 0) {
              if (opp.activeKnown) opp_hit(opp.active, false);
            } else if (idx < static_cast<int>(opp.bench.size())) {
              opp_hit(opp.bench[idx], true);
            }
          }
        } else if (o.p0 == PT_DISTRIBUTE_OPP_BENCH) {
          // Deprecated approximate target. Use OP_DISTRIBUTE_DAMAGE plus
          // OP_APPLY_DISTRIBUTED_DAMAGE for exact "in any way you like" effects.
        } else if (o.p0 == PT_CHOSEN_OWN_BENCH) {
          for (int idx : fr.scratch)
            if (idx >= 0 && idx < static_cast<int>(me.bench.size()))
              own_hit(me.bench[idx], fr.a, true);
        } else if (o.p0 == PT_CHOSEN_OWN_INPLAY) {
          for (int idx : fr.scratch) {
            InPlay* p = inplay_ref(me, idx);
            if (p) own_hit(*p, fr.a, idx >= 0);
          }
        } else if (o.p0 == PT_EACH_OPP_INPLAY) {
          if (opp.activeKnown) opp_hit(opp.active, false);
          for (auto& b : opp.bench) opp_hit(b, true);
        } else if (o.p0 == PT_EACH_OWN_BENCH) {
          for (auto& b : me.bench) own_hit(b, fr.a, true);
        } else if (o.p0 == PT_EACH_OWN_INPLAY) {
          if (me.activeKnown) own_hit(me.active, fr.a, false);
          for (auto& b : me.bench) own_hit(b, fr.a, true);
        } else if (o.p0 == PT_EACH_BENCH) {
          for (auto& b : me.bench) own_hit(b, fr.a, true);
          for (auto& b : opp.bench) opp_hit(b, true);
        } else if (o.p0 == PT_EACH_DAMAGED_BENCH) {
          for (auto& b : me.bench)
            if (b.hp < b.maxHp) own_hit(b, fr.a, true);
          for (auto& b : opp.bench)
            if (b.hp < b.maxHp) opp_hit(b, true);
        } else if (o.p0 == PT_EACH_OPP_INPLAY_DOUBLE_DMG) {
          if (opp.activeKnown) double_opp_counters(opp.active, false);
          for (auto& b : opp.bench) double_opp_counters(b, true);
        } else if (o.p0 == PT_EACH_OPP_ABILITY_INPLAY) {
          if (opp.activeKnown && has_ability_target(1 - fr.a, opp.active, true,
                                                    false))
            direct_counters(opp.active, 1 - fr.a, false);
          for (auto& b : opp.bench)
            if (has_ability_target(1 - fr.a, b, false, false))
              direct_counters(b, 1 - fr.a, true);
        } else if (o.p0 == PT_EACH_ABILITY_INPLAY ||
                   o.p0 == PT_EACH_ABILITY_INPLAY_EXCEPT_FROSLASS) {
          bool exceptFroslass = o.p0 == PT_EACH_ABILITY_INPLAY_EXCEPT_FROSLASS;
          if (me.activeKnown && has_ability_target(fr.a, me.active, true,
                                                   exceptFroslass))
            direct_counters(me.active, -1, false);
          for (auto& b : me.bench)
            if (has_ability_target(fr.a, b, false, exceptFroslass))
              direct_counters(b, -1, true);
          if (opp.activeKnown && has_ability_target(1 - fr.a, opp.active, true,
                                                    exceptFroslass))
            direct_counters(opp.active, 1 - fr.a, false);
          for (auto& b : opp.bench)
            if (has_ability_target(1 - fr.a, b, false, exceptFroslass))
              direct_counters(b, 1 - fr.a, true);
        } else if (o.p0 == PT_OPP_ACTIVE_TO_HP10) {
          if (opp.activeKnown && opp.active.hp > 10) {
            int d = opp.active.hp - 10;
            opp.active.hp = 10;
            placed += d / 10;
          }
        } else if (o.p0 == PT_CHOSEN_OWN_BENCH_TO_OPP_ACTIVE) {
          if (opp.activeKnown) {
            for (int idx : fr.scratch) {
              if (idx < 0 || idx >= static_cast<int>(me.bench.size())) continue;
              InPlay& src = me.bench[idx];
              int d = std::max(0, src.maxHp - src.hp);
              if (d <= 0) continue;
              src.hp = src.maxHp;
              opp.active.hp -= d;
              if (opp.active.hp < 0) opp.active.hp = 0;
              placed += d / 10;
            }
          }
        } else if (o.p0 == PT_CHOSEN_OWN_BENCH_TO_CHOSEN_OPP_INPLAY) {
          if (!fr.savedScratch.empty() && !fr.scratch.empty()) {
            int srcIdx = fr.savedScratch[0];
            if (srcIdx >= 0 && srcIdx < static_cast<int>(me.bench.size())) {
              InPlay& src = me.bench[srcIdx];
              int d = std::max(0, src.maxHp - src.hp);
              if (d > 0) {
                for (int idx : fr.scratch) {
                  InPlay* dst = inplay_ref(opp, idx);
                  if (!dst) continue;
                  src.hp = src.maxHp;
                  dst->hp -= d;
                  if (dst->hp < 0) dst->hp = 0;
                  placed += d / 10;
                  break;
                }
              }
            }
          }
        }
        st.lastEffectCount = placed;
        fr.pc += 1;
        break;
      }
      case OP_ATTACK_DAMAGE_CHOSEN_OPP_BENCH_PER_TARGET_COUNTER: {
        Player& opp = st.players[1 - fr.a];
        int total = 0;
        for (int idx : fr.scratch) {
          if (idx < 0 || idx >= static_cast<int>(opp.bench.size())) continue;
          InPlay& target = opp.bench[idx];
          if (attack_effects_prevented(st, 1 - fr.a, target)) continue;
          int counters = std::max(0, target.maxHp - target.hp) / 10;
          int rawDamage = std::max(0, o.p0) * counters;
          int damage = apply_defender_attack_damage_mods(st, me.active, 1 - fr.a,
                                                         target, true, rawDamage);
          if (damage <= 0) continue;
          int beforeHp = target.hp;
          target.hp -= damage;
          if (target.hp < 0) target.hp = 0;
          apply_damaged_by_attack_reactive(st, target, fr.a, damage, true,
                                           beforeHp);
          total += damage / 10;
        }
        st.lastEffectCount = total;
        fr.pc += 1;
        break;
      }
      case OP_ATTACK_DAMAGE_EACH_OPP_INPLAY_FILTERED: {
        Player& opp = st.players[1 - fr.a];
        int total = 0;
        auto hit = [&](InPlay& p, bool bench) {
          if (!matches_filter(p.id, o.p1)) return;
          if ((o.p2 & DMG_IGNORE_EFFECTS) == 0 &&
              attack_effects_prevented(st, 1 - fr.a, p))
            return;
          int rawDamage = std::max(0, o.p0);
          int damage = (o.p2 & DMG_IGNORE_WR) != 0
                           ? rawDamage
                           : apply_defender_attack_damage_mods(st, me.active,
                                                               1 - fr.a, p,
                                                               bench, rawDamage);
          if (damage <= 0) return;
          int beforeHp = p.hp;
          p.hp -= damage;
          if (p.hp < 0) p.hp = 0;
          apply_damaged_by_attack_reactive(st, p, fr.a, damage, bench,
                                           beforeHp);
          total += damage / 10;
        };
        if (opp.activeKnown) hit(opp.active, false);
        for (auto& b : opp.bench) hit(b, true);
        st.lastEffectCount = total;
        fr.pc += 1;
        break;
      }
      case OP_DAMAGE_PER_TAILS_FROM_LAST_FLIP: {
        Player& opp = st.players[1 - fr.a];
        int tails = std::max(0, st.coinFlips - st.coinHeads);
        int rawDamage = std::max(0, o.p0) * tails;
        int damage = rawDamage;
        if (opp.activeKnown &&
            (o.p1 & DMG_IGNORE_EFFECTS) == 0 &&
            attack_effects_prevented(st, 1 - fr.a, opp.active)) {
          damage = 0;
        } else if (!(o.p1 & DMG_IGNORE_WR) && opp.activeKnown) {
          damage = apply_defender_attack_damage_mods(st, me.active, 1 - fr.a,
                                                     opp.active, false, rawDamage);
        }
        if (opp.activeKnown && damage > 0) {
          int beforeHp = opp.active.hp;
          opp.active.hp -= damage;
          if (opp.active.hp < 0) opp.active.hp = 0;
          apply_damaged_by_attack_reactive(st, opp.active, fr.a, damage, false,
                                           beforeHp);
        }
        st.lastEffectCount = damage / 10;
        fr.pc += 1;
        break;
      }
      case OP_PLACE_OPP_BENCH_COUNTERS_TO_HP: {
        Player& opp = st.players[1 - fr.a];
        int targetHp = std::max(0, o.p0);
        int placed = 0;
        for (auto& b : opp.bench) {
          if (bench_damage_counters_prevented(st, 1 - fr.a, true)) continue;
          if (attack_effects_prevented(st, 1 - fr.a, b)) continue;
          if (b.hp <= targetHp) continue;
          int damage = b.hp - targetHp;
          b.hp = targetHp;
          placed += damage / 10;
        }
        st.lastEffectCount = placed;
        fr.pc += 1;
        break;
      }
      case OP_ATTACK_DAMAGE_EACH_OPP_INPLAY_COIN: {
        Player& opp = st.players[1 - fr.a];
        int total = 0;
        auto hit = [&](InPlay& p, bool bench) {
          if (!flip_heads(st)) return;
          if ((o.p1 & DMG_IGNORE_EFFECTS) == 0 &&
              attack_effects_prevented(st, 1 - fr.a, p))
            return;
          int rawDamage = std::max(0, o.p0);
          int damage = (o.p1 & DMG_IGNORE_WR) != 0
                           ? rawDamage
                           : apply_defender_attack_damage_mods(st, me.active,
                                                               1 - fr.a, p,
                                                               bench, rawDamage);
          if (damage <= 0) return;
          int beforeHp = p.hp;
          p.hp -= damage;
          if (p.hp < 0) p.hp = 0;
          apply_damaged_by_attack_reactive(st, p, fr.a, damage, bench,
                                           beforeHp);
          total += damage / 10;
        };
        if (opp.activeKnown) hit(opp.active, false);
        for (auto& b : opp.bench) hit(b, true);
        st.lastEffectCount = total;
        fr.pc += 1;
        break;
      }
      case OP_APPLY_DISTRIBUTED_DAMAGE: {
        Player& owner = (o.p0 == Z_OWN_BENCH || o.p0 == Z_OWN_INPLAY)
                            ? me : st.players[1 - fr.a];
        int ownerSide = (o.p0 == Z_OWN_BENCH || o.p0 == Z_OWN_INPLAY)
                            ? fr.a : 1 - fr.a;
        int placed = 0;
        for (int idx : fr.scratch) {
          InPlay* p = inplay_ref(owner, idx);
          bool bench = idx >= 0;
          if (p && bench_damage_counters_prevented(st, ownerSide, bench)) continue;
          if (!p || attack_effects_prevented(st, ownerSide, *p)) continue;
          p->hp -= 10;
          if (p->hp < 0) p->hp = 0;
          ++placed;
        }
        st.lastEffectCount = placed;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_DAMAGE_COUNTERS_DISTRIBUTED: {
        Player& owner = (o.p0 == Z_OWN_BENCH || o.p0 == Z_OWN_INPLAY)
                            ? me : st.players[1 - fr.a];
        int ownerSide = (o.p0 == Z_OWN_BENCH || o.p0 == Z_OWN_INPLAY)
                            ? fr.a : 1 - fr.a;
        int srcIdx = fr.savedScratch.empty() ? -2 : fr.savedScratch[0];
        InPlay* src = inplay_ref(owner, srcIdx);
        int moved = 0;
        for (int dstIdx : fr.scratch) {
          InPlay* dst = inplay_ref(owner, dstIdx);
          if (!src || !dst || src == dst) continue;
          if (src->hp >= src->maxHp) break;
          if (attack_effects_prevented(st, ownerSide, *dst)) continue;
          src->hp = std::min(src->maxHp, src->hp + 10);
          dst->hp -= 10;
          if (dst->hp < 0) dst->hp = 0;
          ++moved;
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_DAMAGE_COUNTERS_OWN_TO_OPP: {
        Player& oppOwner = st.players[1 - fr.a];
        int srcIdx = fr.savedScratch.empty() ? -2 : fr.savedScratch[0];
        InPlay* src = inplay_ref(me, srcIdx);
        int moved = 0;
        for (int dstIdx : fr.scratch) {
          InPlay* dst = inplay_ref(oppOwner, dstIdx);
          if (!src || !dst) continue;
          if (src->hp >= src->maxHp) break;
          if (attack_effects_prevented(st, 1 - fr.a, *dst)) continue;
          src->hp = std::min(src->maxHp, src->hp + 10);
          src->healedThisTurn = true;
          dst->hp -= 10;
          if (dst->hp < 0) dst->hp = 0;
          ++moved;
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_DELAYED_DAMAGE_OPP_ACTIVE: {
        Player& opp = st.players[1 - fr.a];
        if (opp.activeKnown &&
            !attack_effects_prevented(st, 1 - fr.a, opp.active)) {
          opp.active.delayedDamageTurn = st.turn + 1;
          opp.active.delayedDamageCounters = std::max(0, o.p0);
          st.lastEffectCount = opp.active.delayedDamageCounters > 0 ? 1 : 0;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_DELAYED_KO_OPP_ACTIVE: {
        Player& opp = st.players[1 - fr.a];
        if (opp.activeKnown &&
            !attack_effects_prevented(st, 1 - fr.a, opp.active)) {
          opp.active.delayedKoTurn = st.turn + 1;
          opp.active.delayedKoPromoteBeforePrize = (o.p0 != 0);
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_STADIUM:  // send the in-play Stadium to its owner's discard
        if (!st.stadium.empty()) {
          int owner = (st.stadiumOwner >= 0) ? st.stadiumOwner : fr.a;
          int oldStadium = st.stadium[0];
          st.players[owner].discard.push_back(st.stadium[0]);
          st.stadium.clear();
          st.stadiumOwner = -1;
          apply_stadium_hp_change(st, oldStadium, -1);  // revert its HP passive
          enforce_area_zero_bench_limits(st);
          enforce_rotom_tool_limits(st);
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      case OP_MILL: {  // discard the top p1 cards of a deck (top = back of vector)
        auto match = [&o](int id) {
          const CardInfo* c = find_card(id);
          if (o.p2 == CM_ENERGY)
            return c && (c->cardType == BASIC_ENERGY || c->cardType == SPECIAL_ENERGY);
          if (o.p2 == CM_BASIC_ENERGY_TYPED)
            return c && c->cardType == BASIC_ENERGY && c->energyType == o.p3;
          if (o.p2 == CM_FILTER)
            return matches_filter(id, o.p3);
          return true;  // CM_NONE counts every milled card
        };
        int counted = 0;
        auto mill = [&](Player& p, int n) {
          for (int i = 0; i < n && !p.deck.empty(); ++i) {
            int id = erase_deck_at(p, static_cast<int>(p.deck.size()) - 1);
            p.discard.push_back(id);
            if (match(id)) ++counted;
          }
        };
        int n = (o.p1 == -3) ? st.coinHeads : o.p1;
        if (o.p0 == MILL_OWN) {
          mill(me, n);
        } else if (o.p0 == MILL_OPP) {
          mill(st.players[1 - fr.a], n);
        } else {  // MILL_EACH
          mill(me, n);
          mill(st.players[1 - fr.a], n);
        }
        st.discardedCount = counted;
        st.lastEffectCount = counted;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_ALL_TYPE: {  // discard every {p0}-type energy off the Active (p0<0=all)
        int cnt = me.activeKnown
                      ? discard_energy_from_inplay(st, me, me.active, -1, o.p0)
                      : 0;
        st.discardedCount = cnt;
        st.lastEffectCount = cnt;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_SELF_ENERGY: {
        InPlay* src = chosen_target(me, fr, AT_SELF);
        int cnt = src ? discard_energy_from_inplay(st, me, *src, o.p0, o.p1) : 0;
        st.discardedCount = cnt;
        st.lastEffectCount = cnt;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_SELF_ENERGY_CARD: {
        InPlay* src = chosen_target(me, fr, AT_SELF);
        int cnt = 0;
        if (src) {
          int want = std::max(1, o.p1);
          for (int k = static_cast<int>(src->energyCardIds.size()) - 1;
               k >= 0 && cnt < want; --k) {
            if (src->energyCardIds[k] != o.p0) continue;
            discard_attached_energy_card(st, me, *src, src->energyCardIds[k]);
            erase_attached_energy_card(*src, k);
            if (k < static_cast<int>(src->energies.size()))
              src->energies.erase(src->energies.begin() + k);
            ++cnt;
          }
          refresh_inplay_max_hp(st, *src);
        }
        st.discardedCount = cnt;
        st.lastEffectCount = cnt;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_SELF_TOOL_ID: {
        InPlay* self = inplay_ref(me, fr.selfBench);
        int cnt = (self && discard_tool(me, *self, o.p0)) ? 1 : 0;
        st.discardedCount = cnt;
        st.lastEffectCount = cnt;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_ENERGY_FROM_CHOSEN_INPLAY: {
        int cnt = 0;
        bool targetIsOpp = (fr.phase == Z_OPP_INPLAY ||
                            fr.savedPhase == Z_OPP_INPLAY);
        Player& owner = targetIsOpp ? st.players[1 - fr.a] : me;
        std::vector<int> refs = fr.scratch;
        std::sort(refs.rbegin(), refs.rend());
        refs.erase(std::unique(refs.begin(), refs.end()), refs.end());
        for (int idx : refs) {
          InPlay* src = inplay_ref(owner, idx);
          if (src) cnt += discard_energy_from_inplay(st, owner, *src, -1, -1);
        }
        st.discardedCount = cnt;
        st.lastEffectCount = cnt;
        fr.pc += 1;
        break;
      }
      case OP_TAKE_PRIZE: {
        int taken = 0;
        for (int i = 0; i < o.p0 && me.prizeCount > 0; ++i) {
          me.prizeCount -= 1;
          record_prize_taken(st, fr.a);
          me.handCount += 1;
          if (!me.prizes.empty()) {
            int cid = me.prizes.back();
            bool knownPrize = me.prizesKnown ||
                              (!me.prizesKnownMask.empty() &&
                               me.prizesKnownMask.back());
            auto kit = std::find(me.prizesKnownCards.begin(),
                                 me.prizesKnownCards.end(), cid);
            if (kit != me.prizesKnownCards.end()) {
              knownPrize = true;
              me.prizesKnownCards.erase(kit);
            }
            me.hand.push_back(cid);
            if (knownPrize && !me.handKnown) add_known_card(me.handKnownCards, cid);
            me.prizes.pop_back();
            if (!me.prizesKnownMask.empty()) me.prizesKnownMask.pop_back();
            if (!me.prizeFaceUp.empty()) me.prizeFaceUp.pop_back();
          } else {
            me.prizesKnown = false;
            me.prizesKnownMask.clear();
            me.handKnown = false;
          }
          ++taken;
        }
        if (me.prizeCount == 0) set_result(st, fr.a);
        st.lastEffectCount = taken;
        fr.pc += 1;
        break;
      }
      case OP_HEAL: {  // raise HP (capped at max) on own Pokemon
        int amt = o.p1;
        bool all = (o.p2 != 0);
        int healed = 0;
        std::vector<int> healedRefs;
        auto h = [amt, all, &healed, &healedRefs](InPlay& k, int idx) {
          if (k.id == 0) return;
          int before = k.hp;
          k.hp = all ? k.maxHp : std::min(k.maxHp, k.hp + amt);
          if (k.hp > before) {
            k.healedThisTurn = true;
            ++healed;
            healedRefs.push_back(idx);
          }
        };
        auto card_ok = [&o](const InPlay& k) {
          return k.id != 0 && (o.p3 < 0 || matches_filter(k.id, o.p3));
        };
        auto basic_ok = [](const InPlay& k) {
          const CardInfo* c = find_card(k.id);
          return c && c->basic;
        };
        auto self = [&]() -> InPlay* {
          int nb = static_cast<int>(me.bench.size());
          return (fr.selfBench < 0) ? (me.activeKnown ? &me.active : nullptr)
                                    : (fr.selfBench < nb ? &me.bench[fr.selfBench] : nullptr);
        };
        if (o.p0 == HEAL_SELF) {
          if (me.activeKnown && card_ok(me.active)) h(me.active, -1);
        } else if (o.p0 == HEAL_EACH_OWN) {
          if (me.activeKnown && card_ok(me.active)) h(me.active, -1);
          for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
            if (card_ok(me.bench[j])) h(me.bench[j], j);
        } else if (o.p0 == HEAL_EACH_OWN_BENCH) {
          for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
            if (card_ok(me.bench[j])) h(me.bench[j], j);
        } else if (o.p0 == HEAL_CHOSEN_OWN_BENCH) {
          if (!fr.scratch.empty() &&
              fr.scratch[0] >= 0 &&
              fr.scratch[0] < static_cast<int>(me.bench.size()) &&
              card_ok(me.bench[fr.scratch[0]]))
            h(me.bench[fr.scratch[0]], fr.scratch[0]);
        } else if (o.p0 == HEAL_CHOSEN_OWN_INPLAY) {
          InPlay* p = fr.scratch.empty() ? nullptr : inplay_ref(me, fr.scratch[0]);
          if (p && card_ok(*p) && (o.p4 <= 0 || p->hp <= o.p4))
            h(*p, fr.scratch[0]);
        } else if (o.p0 == HEAL_EACH_OWN_BASIC) {
          if (me.activeKnown && card_ok(me.active) && basic_ok(me.active))
            h(me.active, -1);
          for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
            if (card_ok(me.bench[j]) && basic_ok(me.bench[j])) h(me.bench[j], j);
        } else if (o.p0 == HEAL_SELF_FILTERED) {
          InPlay* p = self();
          if (p && card_ok(*p) &&
              (o.p4 <= 0 || static_cast<int>(p->energies.size()) >= o.p4))
            h(*p, fr.selfBench);
        }
        fr.scratch = healedRefs;
        st.lastEffectCount = healed;
        fr.pc += 1;
        break;
      }
      case OP_CLEAR_STATUS: {
        int had = (me.poisoned ? 1 : 0) + (me.burned ? 1 : 0) +
                  (me.asleep ? 1 : 0) + (me.paralyzed ? 1 : 0) +
                  (me.confused ? 1 : 0);
        clear_status(me);
        st.lastEffectCount = had > 0 ? 1 : 0;
        fr.pc += 1;
        break;
      }
      case OP_RETURN_CHOSEN_ENERGY: {
        bool energySource = (fr.phase == Z_OWN_ACTIVE_ENERGY ||
                             fr.phase == Z_OWN_INPLAY_ENERGY ||
                             fr.phase == Z_OWN_BENCH_ENERGY ||
                             fr.phase == Z_OPP_ACTIVE_ENERGY ||
                             fr.phase == Z_OPP_INPLAY_ENERGY ||
                             fr.phase == Z_OPP_BENCH_ENERGY ||
                             fr.phase == Z_SAVED_OWN_INPLAY_ENERGY);
        bool oppSource = (fr.phase == Z_OPP_BENCH || fr.phase == Z_OPP_INPLAY ||
                          fr.phase == Z_OPP_ACTIVE_ENERGY ||
                          fr.phase == Z_OPP_INPLAY_ENERGY ||
                          fr.phase == Z_OPP_BENCH_ENERGY);
        Player& owner = oppSource ? st.players[1 - fr.a] : me;
        int moved = 0;
        if (energySource) {
          std::vector<int> refs = fr.scratch;
          std::sort(refs.rbegin(), refs.rend());
          int want = (o.p1 <= 0) ? static_cast<int>(refs.size()) : o.p1;
          for (int ref : refs) {
            if (moved >= want) break;
            InPlay* src = inplay_ref(owner, energy_ref_inplay(ref));
            if (!src || (oppSource && attack_effects_prevented(st, 1 - fr.a, *src)))
              continue;
            int etype = 0, id = 0;
            if (!take_energy_ref(st, owner, ref, etype, id)) continue;
            if (id == 0) id = 6;  // basic energy fallback when replay lacks card ids
            if (o.p0 == D_DECK) {
              push_deck_top(owner, id, true);
            } else if (o.p0 == D_DECK_BOTTOM) {
              push_deck_bottom(owner, id, true);
            } else if (o.p0 == D_DISCARD) {
              owner.discard.push_back(id);
            } else {
              owner.hand.push_back(id);
              owner.handCount += 1;
            }
            ++moved;
          }
          if (o.p0 == D_DECK) shuffle_deck_known(owner, st.rng);
        } else {
          InPlay* src = fr.scratch.empty() ? nullptr : inplay_ref(owner, fr.scratch[0]);
          if (src && (!oppSource || !attack_effects_prevented(st, 1 - fr.a, *src))) {
          int want = (o.p1 <= 0) ? static_cast<int>(src->energies.size()) : o.p1;
          for (int k = static_cast<int>(src->energies.size()) - 1;
               k >= 0 && moved < want; --k) {
            int id = (k < static_cast<int>(src->energyCardIds.size()))
                         ? src->energyCardIds[k] : 6;  // 6 = a basic energy fallback
            if (o.p0 == D_DECK) {
              push_deck_top(owner, id, true);
            } else if (o.p0 == D_DECK_BOTTOM) {
              push_deck_bottom(owner, id, true);
            } else if (o.p0 == D_DISCARD) {
              owner.discard.push_back(id);
            } else {
              owner.hand.push_back(id);
              owner.handCount += 1;
            }
            erase_attached_energy_card(*src, k);
            src->energies.erase(src->energies.begin() + k);
            ++moved;
          }
          refresh_inplay_max_hp(st, *src);
          if (o.p0 == D_DECK) shuffle_deck_known(owner, st.rng);
          }
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_ATTACH_ENERGY: {  // attach up to p2 {p1}-type energy from a zone to a
        // target (p3): self (owner), a chosen bench, or a chosen in-play Pokemon.
        InPlay* tgt = chosen_target(me, fr, o.p3);
        int attached = 0;
        if (tgt) {
          SmallVec<int, 64>& src = (o.p0 == AZ_HAND)      ? me.hand
                                  : (o.p0 == AZ_DISCARD) ? me.discard
                                                         : me.deck;
          int want = o.p2;
          for (int i = static_cast<int>(src.size()) - 1; i >= 0 && attached < want; --i) {
            const CardInfo* c = find_card(src[i]);
            if (c && c->cardType == BASIC_ENERGY &&
              (o.p1 < 0 || c->energyType == o.p1)) {
              tgt->energies.push_back(c->energyType);
              push_attached_energy_card(*tgt, src[i]);
              apply_energy_attach_reactive(st, *tgt, false);
              if (&src == &me.hand) me.handCount -= 1;
              else if (&src == &me.deck) me.deckCount -= 1;
              src.erase(src.begin() + i);
              ++attached;
            }
          }
          if (o.p0 == AZ_DECK) shuffle_deck_known(me, st.rng);
        }
        st.lastEffectCount = attached;
        fr.pc += 1;
        break;
      }
      case OP_SAVE_CHOICE:
        fr.savedScratch = fr.scratch;
        fr.savedPhase = fr.phase;
        st.lastEffectCount = static_cast<int>(fr.savedScratch.size());
        fr.pc += 1;
        break;
      case OP_APPEND_CHOICE: {
        if (fr.savedPhase < 0) fr.savedPhase = fr.phase;
        if (fr.savedPhase == fr.phase) {
          for (int ref : fr.scratch)
            if (std::find(fr.savedScratch.begin(), fr.savedScratch.end(), ref) ==
                fr.savedScratch.end())
              fr.savedScratch.push_back(ref);
        }
        st.lastEffectCount = static_cast<int>(fr.savedScratch.size());
        if (fr.phase == Z_DECK && fr.topDeckCount > 0)
          fr.topDeckSelectedCount = st.lastEffectCount;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_SAVED_CHOSEN: {
        int moved = 0;
        if (fr.savedPhase >= 0) {
          if (fr.topDeckOwner >= 0 &&
              is_counted_out_top_deck_source(fr, fr.topDeckOwner, fr.savedPhase)) {
            std::vector<int> refs = top_deck_actual_refs(fr, fr.savedScratch);
            moved = move_card_refs_from_owner(st.players[fr.topDeckOwner],
                                              fr.savedPhase, refs, o.p0,
                                              false);
            mark_top_deck_cards_removed(fr, moved);
          } else {
            moved = move_card_refs(st, fr.a, fr.savedPhase, fr.savedScratch, o.p0);
          }
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_REORDER_TOP_DECK: {
        Player& owner = (fr.topDeckOwner >= 0) ? st.players[fr.topDeckOwner] : me;
        int n = std::min(fr.topDeckCount, static_cast<int>(owner.deck.size()));
        if (n > 0) {
          int start = static_cast<int>(owner.deck.size()) - n;
          std::vector<int> window(owner.deck.begin() + start, owner.deck.end());
          if (owner.deckKnownMask.size() < owner.deck.size())
            owner.deckKnownMask.resize(owner.deck.size(), false);
          std::vector<bool> maskWindow(owner.deckKnownMask.begin() + start,
                                       owner.deckKnownMask.end());
          for (size_t i = 0; i < window.size(); ++i)
            if (consume_known_card(owner.deckKnownCards, window[i]))
              maskWindow[i] = true;
          std::vector<int> nextTopToBottom;
          for (int ref : fr.scratch) {
            if (ref >= 0 && ref < n) {
              int local = n - 1 - ref;
              if (std::find(nextTopToBottom.begin(), nextTopToBottom.end(), local) ==
                  nextTopToBottom.end())
                nextTopToBottom.push_back(local);
            }
          }
          for (int i = n - 1; i >= 0; --i) {
            if (std::find(nextTopToBottom.begin(), nextTopToBottom.end(), i) ==
                nextTopToBottom.end())
              nextTopToBottom.push_back(i);
          }
          owner.deck.erase(owner.deck.begin() + start, owner.deck.end());
          owner.deckKnownMask.erase(owner.deckKnownMask.begin() + start,
                                    owner.deckKnownMask.end());
          for (auto it = nextTopToBottom.rbegin(); it != nextTopToBottom.rend(); ++it) {
            owner.deck.push_back(window[*it]);
            owner.deckKnownMask.push_back(maskWindow[*it]);
          }
        }
        normalize_deck_knowledge(owner);
        owner.deckCount += fr.topDeckCountedOut;
        normalize_deck_knowledge(owner);
        fr.topDeckCount = 0;
        fr.topDeckSelectedCount = 0;
        fr.topDeckOwner = -1;
        fr.topDeckCountedOut = 0;
        fr.topDeckStart = 0;
        fr.pc += 1;
        break;
      }
      case OP_REVEAL_HAND: {
        Player& who = (o.p0 == S_OPP) ? st.players[1 - fr.a] : me;
        if (static_cast<int>(who.hand.size()) == who.handCount) {
          who.handKnown = true;
          who.handPublicKnown = true;
        }
        st.lastEffectCount = who.handCount;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_HAND_MATCHING: {
        Player& who = (o.p0 == S_OPP) ? st.players[1 - fr.a] : me;
        int discarded = discard_hand_matching(who, o.p1);
        st.discardedCount = discarded;
        st.lastEffectCount = discarded;
        fr.pc += 1;
        break;
      }
      case OP_HAND_TO_BOTTOM_DRAW: {
        Player& who = (o.p0 == S_OPP) ? st.players[1 - fr.a] : me;
        int moved = who.handCount;
        if (!who.hand.empty()) {
          shuffle_deck(who.hand, st.rng);
          for (int id : who.hand) {
            add_known_card(who.deckKnownCards, id);
            push_deck_bottom(who, id, false);
          }
          who.hand.clear();
          who.handKnownCards.clear();
        } else {
          for (int cid : who.handKnownCards) add_known_card(who.deckKnownCards, cid);
          who.handKnownCards.clear();
          who.deckCount += moved;
        }
        normalize_deck_knowledge(who);
        who.handCount = 0;
        if (moved > 0 && o.p1 > 0) draw_n(st, who, o.p1);
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_HAND_TO_BOTTOM_DRAW_SAME: {
        Player& who = (o.p0 == S_OPP) ? st.players[1 - fr.a] : me;
        int moved = who.handCount;
        if (!who.hand.empty()) {
          shuffle_deck(who.hand, st.rng);
          for (int id : who.hand) {
            add_known_card(who.deckKnownCards, id);
            push_deck_bottom(who, id, false);
          }
          who.hand.clear();
          who.handKnownCards.clear();
        } else {
          for (int cid : who.handKnownCards) add_known_card(who.deckKnownCards, cid);
          who.handKnownCards.clear();
          who.deckCount += who.handCount;
        }
        normalize_deck_knowledge(who);
        who.handCount = 0;
        if (moved > 0) draw_n(st, who, moved);
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_OPP_CHOOSE_HAND_TO_DECK: {
        Player& opp = st.players[1 - fr.a];
        PendingDecision pd;
        pd.context = 9; // CTX_TO_DECK
        pd.minCount = std::min(std::max(0, o.p0), opp.handCount);
        pd.maxCount = pd.minCount;
        if (static_cast<int>(opp.hand.size()) == opp.handCount) {
          for (int i = 0; i < static_cast<int>(opp.hand.size()); ++i)
            pd.options.push_back({Atom::S("CARD"), Atom::S("HAND"), Atom::I(i),
                                  Atom::I(opp.hand[i])});
        } else {
          for (int i = 0; i < opp.handCount; ++i)
            pd.options.push_back({Atom::S("CARD"), Atom::S("HAND"), Atom::I(i),
                                  Atom::N()});
        }
        st.pending = pd;
        st.yourIndex = 1 - fr.a;
        return;
      }
      case OP_PRIZE_BARGAIN: {
        PendingDecision pd;
        pd.context = 43; // CTX_ACTIVATE
        pd.options.push_back({Atom::S("YES")});
        pd.options.push_back({Atom::S("NO")});
        st.pending = pd;
        st.yourIndex = 1 - fr.a;
        return;
      }
      case OP_DISCARD_OPP_ENERGY: {  // discard p0 energy from the opponent's Active
        Player& opp = st.players[1 - fr.a];
        int discarded = 0;
        if (opp.activeKnown && attack_effects_prevented(st, 1 - fr.a, opp.active)) {
          st.discardedCount = 0;
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        auto matches = [&](int idx) {
          if (o.p1 == EF_SPECIAL) {
            if (idx >= static_cast<int>(opp.active.energyCardIds.size())) return false;
            const CardInfo* c = find_card(opp.active.energyCardIds[idx]);
            return c && c->cardType == SPECIAL_ENERGY;
          }
          if (o.p1 > 0) return opp.active.energies[idx] == o.p1 - 1;
          return true;
        };
        if (opp.activeKnown) {
          for (int k = static_cast<int>(opp.active.energies.size()) - 1;
               k >= 0 && (o.p0 < 0 || discarded < o.p0); --k) {
            if (!matches(k)) continue;
            if (k < static_cast<int>(opp.active.energyCardIds.size())) {
              opp.discard.push_back(opp.active.energyCardIds[k]);
              erase_attached_energy_card(opp.active, k);
            }
            opp.active.energies.erase(opp.active.energies.begin() + k);
            ++discarded;
          }
        }
        st.discardedCount = discarded;
        st.lastEffectCount = discarded;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_OPP_HAND: {
        Player& opp = st.players[1 - fr.a];
        int want = std::max(0, o.p0);
        int discarded = 0;
        bool known = static_cast<int>(opp.hand.size()) == opp.handCount;
        while (discarded < want && opp.handCount > 0) {
          int replayId = 0;
          bool hasReplay = o.p1 != 0 &&
              consume_replay_event(st, REPLAY_RANDOM_DISCARD_HAND, replayId);
          if (known && !opp.hand.empty()) {
            int idx = -1;
            if (o.p1 != 0) {
              if (hasReplay) {
                auto it = std::find(opp.hand.begin(), opp.hand.end(), replayId);
                if (it != opp.hand.end())
                  idx = static_cast<int>(it - opp.hand.begin());
              }
              if (idx < 0)
                idx = static_cast<int>(next_rand(st.rng) % opp.hand.size());
            } else {
              idx = static_cast<int>(opp.hand.size()) - 1;
            }
            opp.discard.push_back(opp.hand[idx]);
            opp.hand.erase(opp.hand.begin() + idx);
          } else {
            if (hasReplay) {
              opp.discard.push_back(replayId);
              auto it = std::find(opp.handKnownCards.begin(),
                                  opp.handKnownCards.end(), replayId);
              if (it != opp.handKnownCards.end())
                opp.handKnownCards.erase(it);
            } else {
              opp.hand.clear();
              opp.handKnown = false;
            }
          }
          opp.handCount -= 1;
          ++discarded;
        }
        st.discardedCount = discarded;
        st.lastEffectCount = discarded;
        fr.pc += 1;
        break;
      }
      case OP_RANDOM_OPP_HAND_TO_DECK: {
        Player& opp = st.players[1 - fr.a];
        int moved = 0;
        int count = (o.p0 == -3) ? st.coinHeads : std::max(1, o.p0);
        for (int step = 0; step < count && opp.handCount > 0; ++step) {
          bool known = static_cast<int>(opp.hand.size()) == opp.handCount;
          int replayId = 0;
          bool hasReplay = consume_replay_event(
              st, REPLAY_RANDOM_OPP_HAND_TO_DECK, replayId);
          if (known && !opp.hand.empty()) {
            int idx = -1;
            if (hasReplay) {
              auto it = std::find(opp.hand.begin(), opp.hand.end(), replayId);
              if (it != opp.hand.end())
                idx = static_cast<int>(it - opp.hand.begin());
            }
            if (idx < 0)
              idx = static_cast<int>(next_rand(st.rng) % opp.hand.size());
            int id = opp.hand[idx];
            opp.hand.erase(opp.hand.begin() + idx);
            push_deck_top(opp, id, true);
            shuffle_deck_known(opp, st.rng);
          } else {
            if (hasReplay) {
              add_known_card(opp.deckKnownCards, replayId);
              auto it = std::find(opp.handKnownCards.begin(),
                                  opp.handKnownCards.end(), replayId);
              if (it != opp.handKnownCards.end())
                opp.handKnownCards.erase(it);
            } else if (!opp.handKnownCards.empty()) {
                add_known_card(opp.deckKnownCards, opp.handKnownCards.back());
                opp.handKnownCards.pop_back();
            }
            opp.hand.clear();
            opp.handKnown = false;
            opp.deckCount += 1;
          }
          opp.handCount -= 1;
          moved += 1;
        }
        st.discardedCount = 0;
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_SWAP_CHOSEN_HAND_WITH_TOP_DECK: {
        int swapped = 0;
        int handIdx = fr.scratch.empty() ? -1 : fr.scratch[0];
        if (handIdx >= 0 && handIdx < static_cast<int>(me.hand.size()) &&
            !me.deck.empty()) {
          int oldHand = me.hand[handIdx];
          int oldTop = me.deck.back();
          auto kit = std::find(me.deckKnownCards.begin(), me.deckKnownCards.end(),
                               oldTop);
          if (kit != me.deckKnownCards.end())
            me.deckKnownCards.erase(kit);
          me.hand[handIdx] = oldTop;
          me.deck.back() = oldHand;
          if (me.deckKnownMask.size() < me.deck.size())
            me.deckKnownMask.resize(me.deck.size(), me.deckKnown);
          me.deckKnownMask.back() = true;
          normalize_deck_knowledge(me);
          me.handKnown = true;
          swapped = 1;
        }
        st.lastEffectCount = swapped;
        fr.pc += 1;
        break;
      }
      case OP_SET_END_TURN_DISCARD_HAND:
        st.discardHandEndTurn[fr.a] = st.turn;
        st.discardHandEndThreshold[fr.a] = o.p0;
        st.lastEffectCount = 1;
        fr.pc += 1;
        break;
      case OP_BOTH_HAND_TO_BOTTOM_FLIP_DRAW: {
        Player& opp = st.players[1 - fr.a];
        auto hand_to_bottom = [&](Player& who) {
          int moved = who.handCount;
          if (static_cast<int>(who.hand.size()) == who.handCount) {
            shuffle_deck(who.hand, st.rng);
            for (int id : who.hand) {
              add_known_card(who.deckKnownCards, id);
              push_deck_bottom(who, id, false);
            }
          } else {
            for (int id : who.handKnownCards)
              add_known_card(who.deckKnownCards, id);
            who.deckCount += who.handCount;
            who.deckKnown = false;
            who.deckKnownMask.clear();
          }
          who.hand.clear();
          who.handKnownCards.clear();
          who.handCount = 0;
          who.handKnown = true;
          return moved;
        };
        int moved = hand_to_bottom(me) + hand_to_bottom(opp);
        int drawn = 0;
        if (moved > 0) {
          drawn += draw_n(st, me, flip_heads(st) ? 6 : 3);
          drawn += draw_n(st, opp, flip_heads(st) ? 6 : 3);
        }
        st.lastEffectCount = (moved > 0) ? drawn : 0;
        fr.pc += 1;
        break;
      }
      case OP_WIN_IF_OWN_PRIZES_EQ:
        if (me.prizeCount == o.p0) set_result(st, fr.a);
        st.lastEffectCount = (st.result == fr.a) ? 1 : 0;
        fr.pc += 1;
        break;
      case OP_SHUFFLE_SELF_INTO_DECK: {
        int moved = 0;
        int actor = fr.a;
        bool needsPromote = false;
        auto add = [&](int id) {
          if (id == 0) return;
          push_deck_top(me, id, true);
          moved += 1;
        };
        auto add_pokemon_stack = [&](const InPlay& p) {
          add(p.id);
          for (int id : p.preEvo) add(id);
          for (int id : p.energyCardIds) add(id);
          for (int id : p.tools) add(id);
        };
        if (fr.selfBench >= 0) {
          if (fr.selfBench < static_cast<int>(me.bench.size())) {
            add_pokemon_stack(me.bench[fr.selfBench]);
            me.bench.erase(me.bench.begin() + fr.selfBench);
          }
        } else if (me.activeKnown) {
          add_pokemon_stack(me.active);
          me.active = InPlay();
          me.activePresent = false;
          me.activeKnown = false;
          clear_status(me);
          if (me.bench.empty()) {
            set_result(st, 1 - actor);
          } else {
            needsPromote = true;
          }
        }
        if (moved > 0) shuffle_deck_known(me, st.rng);
        st.lastEffectCount = moved;
        fr.pc += 1;
        if (needsPromote) {
          if (run_pre_promote_immediate_damage_hooks(st, actor)) return;
          EffectFrame promote;
          promote.effect = EFF_ABILITY_PROMOTE;
          promote.a = actor;
          st.effectStack.push_back(promote);
          set_promote_pending(st, actor);
          return;
        }
        if (st.result >= 0) {
          if (run_pre_promote_immediate_damage_hooks(st)) return;
          st.effectStack.pop_back();
          st.pending = PendingDecision();
          return;
        }
        break;
      }
      case OP_DISCARD_BOTTOM_SELF_TO_TOP: {
        int moved = 0;
        if (!me.deck.empty()) {
          int bottomId = erase_deck_at(me, 0);
          me.discard.push_back(bottomId);
          InPlay* self = inplay_ref(me, fr.selfBench);
          if (self) {
            InPlay leaving = *self;
            if (fr.selfBench < 0) {
              me.active = InPlay();
              me.activePresent = false;
              me.activeKnown = false;
              clear_status(me);
            } else if (fr.selfBench < static_cast<int>(me.bench.size())) {
              me.bench.erase(me.bench.begin() + fr.selfBench);
            }
            auto top = [&](int id) {
              if (id == 0) return;
              push_deck_top(me, id, true);
              ++moved;
            };
            for (int id : leaving.tools) top(id);
            for (int id : leaving.energyCardIds) top(id);
            for (int id : leaving.preEvo) top(id);
            top(leaving.id);
          }
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_OPP_TOOL: {
        Player& opp = st.players[1 - fr.a];
        int targetIdx = -1;
        if (!fr.scratch.empty()) {
          targetIdx = fr.scratch[0];
          if (fr.phase == Z_OPP_ACTIVE_ENERGY ||
              fr.phase == Z_OPP_INPLAY_ENERGY ||
              fr.phase == Z_OPP_BENCH_ENERGY)
            targetIdx = energy_ref_inplay(fr.scratch[0]);
        }
        InPlay* p = inplay_ref(opp, targetIdx);
        int want = (o.p0 == 0) ? 1 : o.p0;
        int discarded = (p && !attack_effects_prevented(st, 1 - fr.a, *p))
                            ? discard_tools_from_inplay(st, opp, *p, want) : 0;
        st.discardedCount = discarded;
        st.lastEffectCount = discarded;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_CHOSEN_ENERGY: {
        Player& opp = st.players[1 - fr.a];
        InPlay* dst = chosen_target_any(me, opp, fr, o.p0);
        std::vector<int> refs = (fr.savedPhase >= 0) ? fr.savedScratch : fr.scratch;
        int srcPhase = (fr.savedPhase >= 0) ? fr.savedPhase : fr.phase;
        std::sort(refs.rbegin(), refs.rend());
        bool srcIsOpp = (srcPhase == Z_OPP_ACTIVE_ENERGY ||
                         srcPhase == Z_OPP_INPLAY_ENERGY ||
                         srcPhase == Z_OPP_BENCH_ENERGY);
        bool dstIsOpp = (o.p0 == AT_OPP_CHOSEN_BENCH ||
                         o.p0 == AT_OPP_CHOSEN_INPLAY);
        Player& srcOwner = srcIsOpp ? st.players[1 - fr.a] : me;
        int moved = 0;
        if (dst && (!dstIsOpp || !attack_effects_prevented(st, 1 - fr.a, *dst))) {
          for (int ref : refs) {
            InPlay* src = inplay_ref(srcOwner, energy_ref_inplay(ref));
            if (srcIsOpp && src && attack_effects_prevented(st, 1 - fr.a, *src))
              continue;
            if (src && !energy_slot_can_move_to(*src, energy_ref_index(ref), *dst,
                                                fr.sourceCardId == 1221))
              continue;
            int etype = 0, cardId = 0, attachOrder = -1;
            if (take_energy_ref(st, srcOwner, ref, etype, cardId, &attachOrder)) {
              dst->energies.push_back(etype);
              push_attached_energy_card(*dst, cardId, attachOrder);
              ++moved;
            }
          }
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_IF_EFFECT:
        fr.pc += (st.lastEffectCount >= std::max(1, o.p1)) ? 1 : (1 + o.p0);
        break;
      case OP_SCALE_LAST_EFFECT:
        st.lastEffectCount *= std::max(0, o.p0);
        fr.pc += 1;
        break;
      case OP_ATTACH_CHOSEN_ENERGY: {
        std::vector<int> refs = (fr.savedPhase >= 0) ? fr.savedScratch : fr.scratch;
        int srcPhase = (fr.savedPhase >= 0) ? fr.savedPhase : fr.phase;
        std::vector<InPlay*> targets;
        if (o.p0 == AT_EACH_CHOSEN_INPLAY) {
          for (int idx : fr.scratch) {
            InPlay* p = inplay_ref(me, idx);
            if (p) targets.push_back(p);
          }
        } else if (InPlay* p = chosen_target(me, fr, o.p0)) {
          targets.push_back(p);
        }
        int attached = 0;
        if (!targets.empty()) {
          int removed = 0;
          bool countedOut = is_counted_out_top_deck_source(fr, fr.a, srcPhase);
          std::vector<int> actualRefs = countedOut ? top_deck_actual_refs(fr, refs)
                                                   : refs;
          std::vector<int> ids = take_cards_by_index(me, srcPhase, actualRefs, true,
                                                     !countedOut, &removed);
          if (countedOut)
            mark_top_deck_cards_removed(fr, removed);
          for (int id : ids) {
            const CardInfo* c = find_card(id);
            if (!c || (c->cardType != BASIC_ENERGY && c->cardType != SPECIAL_ENERGY))
              continue;
            InPlay* tgt = targets[attached % static_cast<int>(targets.size())];
            tgt->energies.push_back(c->energyType);
            push_attached_energy_card(*tgt, id);
            apply_energy_attach_reactive(st, *tgt, false);
            ++attached;
          }
        }
        st.lastEffectCount = attached;
        fr.pc += 1;
        break;
      }
      case OP_ATTACH_CHOSEN_TOOL: {
        std::vector<int> refs = (fr.savedPhase >= 0) ? fr.savedScratch : fr.scratch;
        int srcPhase = (fr.savedPhase >= 0) ? fr.savedPhase : fr.phase;
        InPlay* target = chosen_target(me, fr, o.p0);
        int attached = 0;
        if (target) {
          int removed = 0;
          bool countedOut = is_counted_out_top_deck_source(fr, fr.a, srcPhase);
          std::vector<int> actualRefs = countedOut ? top_deck_actual_refs(fr, refs)
                                                   : refs;
          std::vector<int> ids = take_cards_by_index(me, srcPhase, actualRefs, false,
                                                     !countedOut, &removed);
          if (countedOut)
            mark_top_deck_cards_removed(fr, removed);
          for (int id : ids) {
            const CardInfo* c = find_card(id);
            if (!c || c->cardType != TOOL) continue;
            int damage = target->maxHp - target->hp;
            attach_tool_card(*target, id);
            target->maxHp = effective_max_hp(target->id, target->tools, st,
                                             target->energyCardIds,
                                             target->energies, fr.a);
            target->hp = std::max(0, target->maxHp - damage);
            ++attached;
          }
        }
        st.lastEffectCount = attached;
        fr.pc += 1;
        break;
      }
      case OP_ATTACH_SAVED_ENERGY_DISTRIBUTED: {
        if (fr.savedPhase < 0 || fr.scratch.empty()) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        bool countedOut = is_counted_out_top_deck_source(fr, fr.a, fr.savedPhase);
        std::vector<int> actualRefs =
            countedOut ? top_deck_actual_refs(fr, fr.savedScratch)
                       : std::vector<int>(fr.savedScratch.begin(),
                                          fr.savedScratch.end());
        int removed = 0;
        std::vector<int> ids = (fr.savedPhase >= 0)
                                   ? take_cards_by_index(me, fr.savedPhase,
                                                         actualRefs, true,
                                                         !countedOut, &removed)
                                   : std::vector<int>();
        if (countedOut)
          mark_top_deck_cards_removed(fr, removed);
        int attached = 0;
        int n = std::min(static_cast<int>(ids.size()),
                         static_cast<int>(fr.scratch.size()));
        for (int i = 0; i < n; ++i) {
          InPlay* tgt = inplay_ref(me, fr.scratch[i]);
          const CardInfo* c = find_card(ids[i]);
          if (!tgt || !c ||
              (c->cardType != BASIC_ENERGY && c->cardType != SPECIAL_ENERGY))
            continue;
          tgt->energies.push_back(c->energyType);
          push_attached_energy_card(*tgt, ids[i]);
          refresh_inplay_max_hp(st, *tgt);
          apply_energy_attach_reactive(st, *tgt, false);
          ++attached;
        }
        st.lastEffectCount = attached;
        fr.pc += 1;
        break;
      }
      case OP_ATTACH_BASIC_ENERGY_EACH_TARGET: {
        if ((o.p0 == AZ_DECK && o.p3 == AET_OWN_BENCH) ||
            (o.p0 == AZ_DISCARD && o.p3 == AET_OWN_INPLAY)) {
          if (set_attach_basic_energy_each_target_pending(
                  st, fr, o.p0, o.p1, o.p2, o.p3))
            return;
          fr.pc += 1;
          break;
        }
        st.lastEffectCount =
            attach_basic_energy_each_target(st, fr, o.p0, o.p1, o.p2, o.p3);
        fr.pc += 1;
        break;
      }
      case OP_DISCARD_OPP_TOOLS_SPECIALS:
        st.lastEffectCount = discard_opp_tools_specials(st, o.p0 != 0);
        fr.pc += 1;
        break;
      case OP_DISCARD_CHOSEN_OPP_ATTACHED_OR_STADIUM:
        st.lastEffectCount = discard_attached_or_stadium_choice(st, fr);
        fr.pc += 1;
        break;
      case OP_CRISPIN_TAKE_HAND_SAVE_ATTACH:
        st.lastEffectCount = take_crispin_energy_choice(st, fr, o.p0);
        fr.pc += 1;
        break;
      case OP_CRISPIN_ATTACH_SAVED:
        st.lastEffectCount = attach_crispin_saved_energy(st, fr);
        fr.pc += 1;
        break;
      case OP_SHUFFLE_OPP_BENCH_EXCEPT_CHOSEN:
        st.lastEffectCount = shuffle_opp_bench_except_chosen(st, fr);
        fr.pc += 1;
        break;
      case OP_KO_OPP_ACTIVE: {
        Player& opp = st.players[1 - fr.a];
        if (opp.activeKnown &&
            !attack_effects_prevented(st, 1 - fr.a, opp.active)) {
          opp.active.hp = 0;
          if (fr.attackId > 0 && fr.attackId != 305 && st.checkupKoFirst < 0)
            st.checkupKoFirst = 1 - fr.a;
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_MARK_SELF_KO: {
        int selfIdx = fr.attackId > 0 ? -1 : fr.selfBench;
        InPlay* self = inplay_ref(me, selfIdx);
        if (self) {
          self->hp = 0;
          st.lastEffectCount = 1;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_KO_SELF_ACTIVE: {
        if (fr.attackId > 0) {
          if (me.activeKnown) {
            me.active.hp = 0;
            st.lastEffectCount = 1;
          } else {
            st.lastEffectCount = 0;
          }
          fr.pc += 1;
          break;
        }
        int selfIdx = fr.selfBench;
        InPlay* self = inplay_ref(me, selfIdx);
        if (self) {
          InPlay knocked = *self;
          int taker = 1 - fr.a;
          int pv = contextual_prize_value(st, taker, fr.a, knocked,
                                          selfIdx < 0);
          int koId = knocked.id;
          if (selfIdx < 0) {
            note_pending_ko_aura(st, fr.a, knocked);
            ko_active(me);
          } else {
            ko_bench(me, selfIdx);
          }
          record_ko(st, fr.a, koId);
          st.lastEffectCount = 1;
          fr.pc += 1;
          EffectFrame ko;
          ko.effect = EFF_ABILITY_KO;
          ko.a = fr.a;          // owner of the Knocked Out Pokemon
          ko.savedSrc = taker;  // prize taker
          ko.phase = pv;
          st.effectStack.push_back(ko);
          if (pv > 0 && st.players[taker].prizeCount > 0) {
            st.yourIndex = taker;
            set_prize_pending(st, taker);
          }
          return;
        } else {
          st.lastEffectCount = 0;
        }
        fr.pc += 1;
        break;
      }
      case OP_KO_CHOSEN_OPP_INPLAY: {
        Player& opp = st.players[1 - fr.a];
        int knocked = 0;
        for (int idx : fr.scratch) {
          InPlay* p = inplay_ref(opp, idx);
          if (!p) continue;
          if (attack_effects_prevented(st, 1 - fr.a, *p)) continue;
          p->hp = 0;
          ++knocked;
        }
        st.lastEffectCount = knocked;
        fr.pc += 1;
        break;
      }
      case OP_KO_OPP_INPLAY_HP_LE: {
        Player& opp = st.players[1 - fr.a];
        int knocked = 0;
        auto maybe_ko = [&st, fr, &knocked, limit = o.p0](InPlay& p) {
          if (p.hp <= 0 || p.hp > limit) return;
          if (attack_effects_prevented(st, 1 - fr.a, p)) return;
          p.hp = 0;
          ++knocked;
        };
        if (opp.activeKnown) maybe_ko(opp.active);
        for (auto& b : opp.bench) maybe_ko(b);
        st.lastEffectCount = knocked;
        fr.pc += 1;
        break;
      }
      case OP_KO_LOWEST_HP_EXCEPT_SELF: {
        struct TargetRef { int side; int idx; int hp; int serial; int seq; };
        std::vector<TargetRef> refs;
        int sideOrder[2] = {fr.a, 1 - fr.a};
        int seq = 0;
        for (int side : sideOrder) {
          Player& p = st.players[side];
          auto add = [&](InPlay& k, int idx) {
            if (side == fr.a && idx == fr.selfBench) return;
            if (k.hp > 0) refs.push_back({side, idx, k.hp, k.serial, seq++});
          };
          if (p.activeKnown) add(p.active, -1);
          for (int j = 0; j < static_cast<int>(p.bench.size()); ++j)
            add(p.bench[j], j);
        }
        int minHp = 1000000;
        for (const auto& r : refs) minHp = std::min(minHp, r.hp);
        std::vector<TargetRef> tied;
        for (const auto& r : refs)
          if (r.hp == minHp) tied.push_back(r);
        if (tied.empty()) {
          st.lastEffectCount = 0;
          fr.pc += 1;
          break;
        }
        PendingDecision pd;
        pd.context = 8;  // DISCARD target
        pd.minCount = 1;
        pd.maxCount = 1;
        fr.savedScratch.clear();
        for (const auto& r : tied) {
          InPlay* p = inplay_ref(st.players[r.side], r.idx);
          if (!p) continue;
          pd.options.push_back({Atom::S("CARD"),
                                Atom::S(r.idx < 0 ? "ACTIVE" : "BENCH"),
                                Atom::I(r.idx), Atom::I(p->id)});
          fr.savedScratch.push_back(r.side);
        }
        st.pending = pd;
        return;
      }
      case OP_SELF_REDUCE: {  // -p0 damage to "this Pokemon" during opp's next turn
        InPlay* self = (fr.selfBench < 0)
                           ? (me.activeKnown ? &me.active : nullptr)
                           : (fr.selfBench < static_cast<int>(me.bench.size())
                                  ? &me.bench[fr.selfBench] : nullptr);
        if (self) {
          self->dmgReduce = o.p0;
          self->dmgReduceTurn = st.turn + 1;  // the opponent's next turn
        }
        fr.pc += 1;
        break;
      }
      case OP_DRAW_UNTIL: {  // draw until the hand holds p0 cards
        int need = o.p0 - me.handCount;
        st.lastEffectCount = need > 0 ? draw_n(st, me, need) : 0;
        fr.pc += 1;
        break;
      }
      case OP_COUNT: {  // count(domain, filter, etype) -> st.countReg
        Player& opp = st.players[1 - fr.a];
        int filt = o.p1, et = o.p2;
        auto pk = [&](const InPlay& k) {
          return k.id != 0 && (filt < 0 || matches_filter(k.id, filt));
        };
        auto ecount = [et](const InPlay& k) {
          int c = 0;
          for (int e : k.energies)
            if (et < 0 || e == et) ++c;
          return c;
        };
        int n = 0;
        switch (o.p0) {
          case CD_OWN_INPLAY:
            if (me.activeKnown && pk(me.active)) ++n;
            for (auto& b : me.bench) if (pk(b)) ++n;
            break;
          case CD_OPP_INPLAY:
            if (opp.activeKnown && pk(opp.active)) ++n;
            for (auto& b : opp.bench) if (pk(b)) ++n;
            break;
          case CD_BOTH_INPLAY:
            if (me.activeKnown && pk(me.active)) ++n;
            for (auto& b : me.bench) if (pk(b)) ++n;
            if (opp.activeKnown && pk(opp.active)) ++n;
            for (auto& b : opp.bench) if (pk(b)) ++n;
            break;
          case CD_OWN_BENCH:
            for (auto& b : me.bench) if (pk(b)) ++n;
            break;
          case CD_OWN_DISCARD:
            for (int id : me.discard) if (filt < 0 || matches_filter(id, filt)) ++n;
            break;
          case CD_OWN_DECK:
            for (int id : me.deck) if (filt < 0 || matches_filter(id, filt)) ++n;
            break;
          case CD_OWN_DAMAGED:
            if (me.activeKnown && pk(me.active) && me.active.hp < me.active.maxHp) ++n;
            for (auto& b : me.bench) if (pk(b) && b.hp < b.maxHp) ++n;
            break;
          case CD_OPP_DAMAGED:
            if (opp.activeKnown && pk(opp.active) && opp.active.hp < opp.active.maxHp) ++n;
            for (auto& b : opp.bench) if (pk(b) && b.hp < b.maxHp) ++n;
            break;
          case CD_OWN_INPLAY_ENERGY:
            if (me.activeKnown && pk(me.active)) n += ecount(me.active);
            for (auto& b : me.bench) if (pk(b)) n += ecount(b);
            break;
          case CD_BOTH_ACTIVE_ENERGY:
            if (me.activeKnown) n += ecount(me.active);
            if (opp.activeKnown) n += ecount(opp.active);
            break;
          case CD_OPP_STATUS:
            n = (opp.poisoned ? 1 : 0) + (opp.burned ? 1 : 0) + (opp.asleep ? 1 : 0) +
                (opp.paralyzed ? 1 : 0) + (opp.confused ? 1 : 0);
            break;
          case CD_OPP_HAND:
            if (filt < 0 || opp.hand.empty()) {
              n = (filt < 0) ? opp.handCount : 0;
            } else {
              for (int id : opp.hand)
                if (matches_filter(id, filt)) ++n;
            }
            break;
          case CD_OPP_DISCARD:
            for (int id : opp.discard) if (filt < 0 || matches_filter(id, filt)) ++n;
            break;
          case CD_OWN_TOOLS:
            if (me.activeKnown) n += static_cast<int>(me.active.tools.size());
            for (const auto& b : me.bench) n += static_cast<int>(b.tools.size());
            break;
          case CD_BOTH_TOOLS:
            if (me.activeKnown) n += static_cast<int>(me.active.tools.size());
            for (const auto& b : me.bench) n += static_cast<int>(b.tools.size());
            if (opp.activeKnown) n += static_cast<int>(opp.active.tools.size());
            for (const auto& b : opp.bench) n += static_cast<int>(b.tools.size());
            break;
          case CD_OWN_BENCH_DAMAGE:
            for (auto& b : me.bench)
              if (pk(b)) n += (b.maxHp - b.hp) / 10;
            break;
          case CD_OWN_DISCARD_ANCIENT:
            for (int id : me.discard) if (is_ancient_card_id(id)) ++n;
            break;
          case CD_OWN_INPLAY_ANCIENT:
            if (me.activeKnown && is_ancient_card_id(me.active.id)) ++n;
            for (auto& b : me.bench) if (is_ancient_card_id(b.id)) ++n;
            break;
          case CD_OWN_DISCARD_BASIC_ENERGY_TYPED:
            for (int id : me.discard) {
              const CardInfo* c = find_card(id);
              if (c && c->cardType == BASIC_ENERGY && (et < 0 || c->energyType == et))
                ++n;
            }
            break;
          default: break;
        }
        st.countReg = n;
        fr.pc += 1;
        break;
      }
      case OP_SHUFFLE_DISCARD_MATCHING: {
        int filt = o.p1, et = o.p2;
        std::vector<int> kept;
        std::vector<int> moved;
        kept.reserve(me.discard.size());
        auto matches = [&](int id) {
          const CardInfo* c = find_card(id);
          if (o.p0 == CD_OWN_DISCARD_BASIC_ENERGY_TYPED)
            return c && c->cardType == BASIC_ENERGY &&
                   (et < 0 || c->energyType == et);
          if (o.p0 == CD_OWN_DISCARD_ANCIENT)
            return is_ancient_card_id(id);
          if (o.p0 == CD_OWN_DISCARD)
            return filt < 0 || matches_filter(id, filt);
          return false;
        };
        for (int id : me.discard) {
          if (matches(id)) moved.push_back(id);
          else kept.push_back(id);
        }
        me.discard = kept;
        for (int id : moved) push_deck_top(me, id, true);
        if (!moved.empty()) {
          shuffle_deck_known(me, st.rng);
        }
        st.lastEffectCount = static_cast<int>(moved.size());
        fr.pc += 1;
        break;
      }
      case OP_TURN_EFFECT: {  // set a turn-scoped flag on a target
        int when = st.turn + (o.p2 == TSC_OPP_NEXT ? 1 : 2);
        InPlay* t = nullptr;
        if (o.p0 == TT_SELF)
          t = (fr.selfBench < 0) ? (me.activeKnown ? &me.active : nullptr)
              : (fr.selfBench < static_cast<int>(me.bench.size())
                     ? &me.bench[fr.selfBench] : nullptr);
        else if (o.p0 == TT_OPP_ACTIVE) {
          Player& opp = st.players[1 - fr.a];
          if (opp.activeKnown &&
              !(fr.attackId != 0 &&
                attack_effects_prevented(st, 1 - fr.a, opp.active)))
            t = &opp.active;
        } else if (o.p0 == TT_CHOSEN_OWN_INPLAY) {
          int idx = fr.scratch.empty() ? -2 : fr.scratch[0];
          t = inplay_ref(me, idx);
        }
        if (t) {
          if (o.p1 == TK_NO_ATTACK) t->noAttackTurn = when;
          else if (o.p1 == TK_NO_RETREAT) t->noRetreatTurn = when;
          else if (o.p1 == TK_PREVENT_DMG) {
            t->preventDmgTurn = when;
            t->preventDmgCond = o.p3;
            t->preventDmgValue = o.p4;
          } else if (o.p1 == TK_ATTACK_DMG_REDUCE) {
            t->attackDmgReduceTurn = when;
            t->attackDmgReduce = o.p3;
          } else if (o.p1 == TK_ATTACK_COST_MORE) {
            t->attackCostModTurn = when;
            t->attackCostMod = o.p3;
          } else if (o.p1 == TK_RETREAT_COST_MORE) {
            t->retreatCostModTurn = when;
            t->retreatCostMod = o.p3;
          } else if (o.p1 == TK_PREVENT_EFFECTS) {
            t->preventEffectsTurn = when;
            t->preventEffectsCond = o.p3;
            t->preventEffectsValue = o.p4;
          } else if (o.p1 == TK_ATTACK_FLIP_FAIL) {
            t->attackFlipFailTurn = when;
          } else if (o.p1 == TK_TAKE_MORE_DAMAGE) {
            t->takeMoreDamageTurn = when;
            t->takeMoreDamage = o.p3;
          } else if (o.p1 == TK_NO_WEAKNESS) {
            t->noWeaknessTurn = when;
          } else if (o.p1 == TK_NAMED_ATTACK_BONUS) {
            t->nextAttackBonusId = o.p3;
            t->nextAttackBonusTurn = when;
            t->nextAttackSetBase = (o.p4 / 1000) - 1;
            t->nextAttackBonus = o.p4 % 1000;
          } else if (o.p1 == TK_IF_DAMAGED_COUNTERS) {
            t->damagedByAttackCountersTurn = when;
            t->damagedByAttackCounters = o.p3;
          } else if (o.p1 == TK_IF_DAMAGED_STATUS) {
            t->damagedByAttackCountersTurn = when;
            t->damagedByAttackStatus = o.p3;
          } else if (o.p1 == TK_ENERGY_ATTACH_COUNTERS) {
            t->energyAttachCountersTurn = when;
            t->energyAttachCounters = o.p3;
            t->energyAttachCountersFromHandOnly = o.p4;
          } else if (o.p1 == TK_IF_DAMAGED_DAMAGE_COUNTERS) {
            t->damagedByAttackEqualCountersTurn = when;
          }
        }
        fr.pc += 1;
        break;
      }
      case OP_OPP_NO_ITEMS:
        if (o.p1 == OTL_SUPPORTERS)
          st.noSupporterTurn[1 - fr.a] = st.turn + (o.p0 == TSC_OPP_NEXT ? 1 : 2);
        else if (o.p1 == OTL_EVOLVE)
          st.noEvolveTurn[1 - fr.a] = st.turn + (o.p0 == TSC_OPP_NEXT ? 1 : 2);
        else if (o.p1 == OTL_STADIUM)
          st.noStadiumTurn[1 - fr.a] = st.turn + (o.p0 == TSC_OPP_NEXT ? 1 : 2);
        else
          st.noItemTurn[1 - fr.a] = st.turn + (o.p0 == TSC_OPP_NEXT ? 1 : 2);
        fr.pc += 1;
        break;
      case OP_OPP_NO_ATTACK_ENERGY_LE:
        st.noAttackEnergyLeTurn[1 - fr.a] =
            st.turn + (o.p0 == TSC_OPP_NEXT ? 1 : 2);
        st.noAttackEnergyLeThreshold[1 - fr.a] = o.p1;
        fr.pc += 1;
        break;
      case OP_TEAM_REDUCE:
        if (st.teamReduceTurn[fr.a] == st.turn + 1 &&
            st.teamReduceType[fr.a] == o.p1) {
          st.teamReduceAmount[fr.a] += o.p0;
        } else {
          st.teamReduceTurn[fr.a] = st.turn + 1;
          st.teamReduceAmount[fr.a] = o.p0;
          st.teamReduceType[fr.a] = o.p1;
        }
        fr.pc += 1;
        break;
      case OP_MOVE_ENERGY: {  // move p0 {p1}-energy between own Pokemon. src (p2):
        // 0 = "this Pokemon", 1 = the saved source. dest (p3): 0 = the chosen Pokemon
        // (scratch[0]), 1 = "this Pokemon". In-play index: -1 Active, else bench idx.
        int nb = static_cast<int>(me.bench.size());
        auto inplay = [&](int idx) -> InPlay* {
          if (idx <= -2) return nullptr;
          return (idx < 0) ? (me.activeKnown ? &me.active : nullptr)
                           : (idx < nb ? &me.bench[idx] : nullptr);
        };
        int srcIdx = (o.p2 == 1) ? fr.savedSrc : fr.selfBench;
        int dstIdx = (o.p3 == 1) ? fr.selfBench
                     : (fr.scratch.empty() ? -2 : fr.scratch[0]);
        InPlay* src = inplay(srcIdx);
        InPlay* dst = inplay(dstIdx);
        int moved = 0;
        if (src && dst && src != dst) {
          int want = o.p0;
          for (int k = static_cast<int>(src->energies.size()) - 1;
               k >= 0 && moved < want; --k) {
            if (o.p1 >= 0 && src->energies[k] != o.p1) continue;
            dst->energies.push_back(src->energies[k]);
            if (k < static_cast<int>(src->energyCardIds.size())) {
              push_attached_energy_card(*dst, src->energyCardIds[k],
                                        attached_energy_order(*src, k));
              erase_attached_energy_card(*src, k);
            }
            src->energies.erase(src->energies.begin() + k);
            ++moved;
          }
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_MOVE_DAMAGE_COUNTER: {
        int srcIdx = fr.savedScratch.empty() ? -2 : fr.savedScratch[0];
        int dstIdx = fr.scratch.empty() ? -2 : fr.scratch[0];
        InPlay* src = inplay_ref(me, srcIdx);
        InPlay* dst = inplay_ref(me, dstIdx);
        int moved = 0;
        if (src && dst && src != dst && src->hp < src->maxHp) {
          src->hp = std::min(src->maxHp, src->hp + 10);
          src->healedThisTurn = true;
          dst->hp -= 10;
          if (dst->hp < 0) dst->hp = 0;
          moved = 1;
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_RETURN_ENERGY: {  // move p1 energy off "this Pokemon" to hand/deck
        int nb = static_cast<int>(me.bench.size());
        InPlay* src = (fr.selfBench < 0) ? (me.activeKnown ? &me.active : nullptr)
                      : (fr.selfBench < nb ? &me.bench[fr.selfBench] : nullptr);
        int moved = 0;
        if (src) {
          int want = (o.p1 <= 0) ? static_cast<int>(src->energies.size()) : o.p1;
          for (int k = static_cast<int>(src->energies.size()) - 1;
               k >= 0 && moved < want; --k) {
            int id = (k < static_cast<int>(src->energyCardIds.size()))
                         ? src->energyCardIds[k] : 6;  // 6 = a basic energy fallback
            if (o.p0 == D_DECK) { push_deck_top(me, id, true); }
            else { me.hand.push_back(id); me.handCount += 1; }
            erase_attached_energy_card(*src, k);
            src->energies.erase(src->energies.begin() + k);
            ++moved;
          }
          if (o.p0 == D_DECK) shuffle_deck_known(me, st.rng);
        }
        st.lastEffectCount = moved;
        fr.pc += 1;
        break;
      }
      case OP_SAVE_SRC:  // stash the chosen in-play index for a later OP_MOVE_ENERGY
        fr.savedSrc = fr.scratch.empty() ? -2 : fr.scratch[0];
        fr.pc += 1;
        break;
      default:
        fr.pc += 1;  // op not yet implemented (decision / attack ops)
        break;
    }
  }
}

static void advance_effect_stack(GameState& st, const std::vector<int>& tape) {
  while (!st.has_pending() && !st.effectStack.empty()) {
    EffectFrame fr = st.effectStack.back();
    if (fr.effect == FLOW_PROGRAM) {
      run_program(st, tape);
      continue;
    }
    if (fr.effect == EFF_RISKY_RUINS_BENCH_ENTRY) {
      st.effectStack.pop_back();
      st.yourIndex = fr.a;
      apply_risky_ruins_bench_entry(st, fr.a, fr.selfBench);
      continue;
    }
    if (fr.effect == EFF_PALAFIN_ZERO_TO_HERO) {
      st.yourIndex = fr.a;
      set_palafin_yesno_pending(st, st.effectStack.back());
      return;
    }
    if (fr.effect == EFF_ABILITY_PROMOTE) {
      st.yourIndex = fr.a;
      set_promote_pending(st, fr.a);
      return;
    }
    if (fr.effect == EFF_DAMAGE_TRIGGER_ORDER) {
      set_damage_trigger_order_pending_from_frame(st, st.effectStack.back());
      return;
    }
    if (fr.effect == EFF_EVOLVE_TRIGGER_ORDER) {
      set_evolve_trigger_order_pending(st, st.effectStack.back());
      return;
    }
    if (fr.effect == EFF_DARKEST_IMPULSE) {
      st.effectStack.pop_back();
      st.yourIndex = fr.a;
      apply_darkest_impulse(st, fr.a, fr.selfBench);
      continue;
    }
    if (fr.effect == EFF_ON_PLAY_BASIC_TRIGGER_ORDER) {
      set_on_play_basic_trigger_order_pending(st, st.effectStack.back());
      return;
    }
    if (fr.effect == EFF_ON_ATTACH_TRIGGER_ORDER) {
      set_on_attach_trigger_order_pending(st, st.effectStack.back());
      return;
    }
    return;
  }
}

// Can `energies` (resolved types) pay for an attack's cost? Colorless (0)
// requirements are paid by any leftover energy; Rainbow (10) pays any specific.
static constexpr int FLEX_PSYCHIC_DARKNESS = TEAM_ROCKET;

static bool meganium_grass_energy_doubled(const GameState* st, int ownerSide,
                                          int energyCardId, int fallbackType) {
  if (!st || ownerSide < 0 || ownerSide >= 2) return false;
  int basicType = -1;
  if (basic_energy_type(energyCardId, basicType))
    return basicType == GRASS && owner_has_active_ability_card(*st, ownerSide, 710);
  return energyCardId <= 0 && fallbackType == GRASS &&
         owner_has_active_ability_card(*st, ownerSide, 710);
}

SmallVec<int, 24> provided_energy_units(const InPlay& pk,
                                        const GameState* st,
                                        int ownerSide) {
  const CardInfo* pc = find_card(pk.id);
  SmallVec<int, 24> units;
  for (int i = 0; i < static_cast<int>(pk.energyCardIds.size()); ++i) {
    int id = pk.energyCardIds[i];
    const CardInfo* ec = find_card(id);
    int fallback = (i < static_cast<int>(pk.energies.size()))
                       ? pk.energies[i]
                       : (ec ? ec->energyType : COLORLESS);
    if (!ec || (ec->cardType != BASIC_ENERGY && ec->cardType != SPECIAL_ENERGY)) {
      units.push_back(fallback);
      continue;
    }
    if (meganium_grass_energy_doubled(st, ownerSide, id, fallback)) {
      units.push_back(GRASS);
      units.push_back(GRASS);
      continue;
    }
    switch (id) {
      case 15:  // Team Rocket's Energy: two units, each {P} or {D}, on TR Pokemon.
        if (is_team_rocket_card_id(pk.id)) {
          units.push_back(FLEX_PSYCHIC_DARKNESS);
          units.push_back(FLEX_PSYCHIC_DARKNESS);
        }
        break;
      case 10:  // Neo Upper Energy: {C}, or two of any type on Stage 2.
        if (pc && pc->stage2) {
          units.push_back(RAINBOW);
          units.push_back(RAINBOW);
        } else {
          units.push_back(COLORLESS);
        }
        break;
      case 16:  // Prism Energy: {C}, or any type on Basic Pokemon.
        units.push_back((pc && pc->basic) ? RAINBOW : COLORLESS);
        break;
      case 17:  // Ignition Energy: {C}, or {C}{C}{C} on Evolution Pokemon.
        units.push_back(COLORLESS);
        if (pc && (pc->stage1 || pc->stage2)) {
          units.push_back(COLORLESS);
          units.push_back(COLORLESS);
        }
        break;
      default:
        units.push_back(ec->energyType);
        break;
    }
  }
  if (pk.energyCardIds.empty()) {
    units = pk.energies;  // fallback for legacy constructed states.
    if (st && ownerSide >= 0 && owner_has_active_ability_card(*st, ownerSide, 710)) {
      std::vector<int> doubled;
      for (int e : units) {
        doubled.push_back(e);
        if (e == GRASS) doubled.push_back(GRASS);
      }
      units = doubled;
    }
  }
  return units;
}

int provided_energy_units_for_card(const InPlay& pk, int energyIdx,
                                   const GameState* st,
                                   int ownerSide) {
  if (energyIdx < 0 || energyIdx >= static_cast<int>(pk.energyCardIds.size()))
    return 1;
  int id = pk.energyCardIds[energyIdx];
  const CardInfo* pc = find_card(pk.id);
  int fallback = (energyIdx < static_cast<int>(pk.energies.size()))
                     ? pk.energies[energyIdx]
                     : COLORLESS;
  if (meganium_grass_energy_doubled(st, ownerSide, id, fallback)) return 2;
  if (id == 15 && is_team_rocket_card_id(pk.id)) return 2;
  if (id == 10 && pc && pc->stage2) return 2;
  if (id == 17 && pc && (pc->stage1 || pc->stage2)) return 3;
  return 1;
}

static bool can_flex_psychic_darkness_pay(int required) {
  return required == PSYCHIC || required == DARKNESS;
}

static bool affordable_units(const SmallVec<int, 24>& energies,
                             const SmallVec<int, 8>& typed,
                             int colorless) {
  SmallVec<int, 24> pool = energies;
  for (int r : typed) {
    auto it = std::find(pool.begin(), pool.end(), r);
    if (it == pool.end()) it = std::find(pool.begin(), pool.end(), RAINBOW);
    if (it == pool.end() && can_flex_psychic_darkness_pay(r))
      it = std::find(pool.begin(), pool.end(), FLEX_PSYCHIC_DARKNESS);
    if (it == pool.end()) return false;
    pool.erase(it);
  }
  return static_cast<int>(pool.size()) >= std::max(0, colorless);
}

static bool affordable(const SmallVec<int, 24>& energies, const AttackInfo& at,
                       int extraColorless = 0, int anyDiscount = 0) {
  SmallVec<int, 8> units;
  int colorless = extraColorless;
  for (int c = 0; c < at.n_cost; ++c) {
    int r = at.cost[c];
    if (r == COLORLESS) {
      ++colorless;
      continue;
    }
    units.push_back(r);
  }
  colorless = std::max(0, colorless);
  if (anyDiscount <= 0)
    return affordable_units(energies, units, colorless);

  for (int i = 0; i < colorless; ++i) units.push_back(COLORLESS);
  int n = static_cast<int>(units.size());
  if (anyDiscount >= n) return true;
  int limit = 1 << n;
  for (int mask = 0; mask < limit; ++mask) {
    int removed = 0;
    SmallVec<int, 8> typed;
    int cneed = 0;
    for (int i = 0; i < n; ++i) {
      if (mask & (1 << i)) {
        ++removed;
        continue;
      }
      if (units[i] == COLORLESS)
        ++cneed;
      else
        typed.push_back(units[i]);
    }
    if (removed <= anyDiscount && affordable_units(energies, typed, cneed))
      return true;
  }
  return false;
}

static bool affordable(const InPlay& attacker, const AttackInfo& at,
                       int extraColorless = 0, int anyDiscount = 0,
                       const GameState* st = nullptr, int ownerSide = -1) {
  return affordable(provided_energy_units(attacker, st, ownerSide), at, extraColorless,
                    anyDiscount);
}

static int retreat_cost(const GameState& st, int player) {
  const Player& me = st.players[player];
  if (!me.activeKnown) return 0;
  if (is_antique_fossil(me.active.id)) return 999;
  const CardInfo* aci = find_card(me.active.id);
  bool freeRetreat = (aci && aci->freeRetreat && me.active.energies.empty()) ||
                     passive_free_retreat(st, player, me.active);
  int cost = (aci && !freeRetreat) ? aci->retreat : 0;
  if (!freeRetreat) {
    if (active_tool(st, me.active, 1157)) cost -= 1;  // Rescue Board
    if (active_tool(st, me.active, 1174)) cost -= 2;  // Air Balloon
  }
  const Player& opp = st.players[1 - player];
  if (!freeRetreat &&
      ((me.activeKnown && active_tool(st, me.active, 1166)) ||
       (opp.activeKnown && active_tool(st, opp.active, 1166))))
    cost += 1;  // Gravity Gemstone does not override "no Retreat Cost" effects.
  if (me.active.retreatCostModTurn == st.turn)
    cost += me.active.retreatCostMod;
  return std::max(0, cost);
}

static bool first_player_turn1(const GameState& st) {
  return st.firstPlayer >= 0 && st.yourIndex == st.firstPlayer && st.turn == 1;
}

static bool second_player_turn1(const GameState& st) {
  return st.firstPlayer >= 0 && st.yourIndex != st.firstPlayer &&
         (st.turn == 1 || st.turn == 2);
}

static bool actor_first_turn(const GameState& st) {
  return first_player_turn1(st) || second_player_turn1(st);
}

static bool attack_gate_ok(const GameState& st, int attackId) {
  switch (attack_gate(attackId)) {
    case ATG_ALLOW_FIRST_PLAYER_TURN1:
      return true;
    case ATG_NOT_SECOND_PLAYER_TURN1:
      return !second_player_turn1(st);
    case ATG_ONLY_SECOND_PLAYER_TURN1:
      return second_player_turn1(st);
    case ATG_TEAM_ROCKET_INPLAY_GE_4:
      return in_play_name_count(st.players[st.yourIndex], "Team Rocket") >= 4;
    case ATG_OPP_HAS_EX_OR_V_INPLAY:
      return in_play_has_ex_or_v(st.players[1 - st.yourIndex]);
    case ATG_NOT_USED_LAST_TURN:
      return !(st.lastAttackTurn[st.yourIndex] == st.turn - 2 &&
               st.lastAttackId[st.yourIndex] == attackId);
    default:
      return true;
  }
}

static bool attack_turn_ok(const GameState& st, int attackId) {
  if (!first_player_turn1(st)) return true;
  return attack_gate(attackId) == ATG_ALLOW_FIRST_PLAYER_TURN1;
}

static bool attack_effect_available(const GameState& st, int attackId) {
  const Player& me = st.players[st.yourIndex];
  switch (attackId) {
    case 157:
      return hand_has_basic_energy(me);
    case 242:
      return !me.bench.empty();
    default:
      return true;
  }
}

static bool ability_repeatable(int id) {
  return id == 269 || id == 436 || id == 569 || id == 795 || id == 962 ||
         id == 1029 || id == 652;  // "as often as you like" abilities
}

static bool ability_group_available(const GameState& st, int player, int cardId) {
  int group = ability_limit_group(cardId);
  return group <= 0 || group >= 16 ||
         st.abilityGroupUsedTurn[player][group] != st.turn;
}

static void mark_ability_group_used(GameState& st, int player, int cardId) {
  int group = ability_limit_group(cardId);
  if (group > 0 && group < 16)
    st.abilityGroupUsedTurn[player][group] = st.turn;
}

static bool supporter_turn1_exception(int cardId) {
  return cardId == 1192 || cardId == 1220;  // Carmine / Team Rocket's Proton
}

static bool rare_candy_gate_ok(const GameState& st, const Player& me) {
  if (actor_first_turn(st) || evolution_locked(st, st.yourIndex)) return false;
  auto ok_target = [&me, &st](const InPlay& p) {
    const CardInfo* c = find_card(p.id);
    return c && c->basic && p.preEvo.empty() && !p.appearThisTurn &&
           p.noEvolveTurn != st.turn && hand_has_stage2_for_basic(me, p.id);
  };
  if (me.activeKnown && ok_target(me.active)) return true;
  for (const auto& b : me.bench)
    if (ok_target(b)) return true;
  return false;
}

static bool grand_tree_basic_target_ok(const GameState& st, int side,
                                       const InPlay& p) {
  const CardInfo* c = find_card(p.id);
  return c && c->basic && p.preEvo.empty() && !p.appearThisTurn &&
         p.noEvolveTurn != st.turn && !actor_first_turn(st) &&
         !evolution_locked(st, side);
}

static bool grand_tree_has_stage1_option(const GameState& st, int side) {
  const Player& me = st.players[side];
  auto has_stage1_for = [&me](const InPlay& p) {
    for (int id : me.deck) {
      const CardInfo* c = find_card(id);
      if (c && c->stage1 && card_evolves_from(id, p.id, false)) return true;
    }
    return false;
  };
  if (me.activeKnown && grand_tree_basic_target_ok(st, side, me.active) &&
      has_stage1_for(me.active))
    return true;
  for (const auto& b : me.bench)
    if (grand_tree_basic_target_ok(st, side, b) && has_stage1_for(b))
      return true;
  return false;
}

static bool is_damaged(const InPlay& p) {
  return p.id != 0 && p.hp < p.maxHp;
}

static int own_inplay_count(const Player& p) {
  return (p.activeKnown ? 1 : 0) + static_cast<int>(p.bench.size());
}

static bool player_has_damaged_inplay(const Player& p) {
  if (p.activeKnown && is_damaged(p.active)) return true;
  for (const auto& b : p.bench)
    if (is_damaged(b)) return true;
  return false;
}

static bool player_has_status(const Player& p) {
  return p.poisoned || p.burned || p.asleep || p.paralyzed || p.confused;
}

static bool active_is_damaged(const Player& p) {
  return p.activeKnown && is_damaged(p.active);
}

static bool active_has_type(const Player& p, int etype) {
  if (!p.activeKnown) return false;
  return pokemon_has_type(p.active.id, etype);
}

static bool active_energy_count_ge(const Player& p, int n) {
  return p.activeKnown &&
         static_cast<int>(provided_energy_units(p.active).size()) >= n;
}

static bool card_is_basic_energy_type(int cardId, int etype = -1) {
  int actual = -1;
  return basic_energy_type(cardId, actual) && (etype < 0 || actual == etype);
}

static bool discard_has_basic_energy(const Player& p, int etype = -1) {
  for (int id : p.discard)
    if (card_is_basic_energy_type(id, etype)) return true;
  return false;
}

static bool hand_has_basic_energy(const Player& p, int etype) {
  for (int id : p.hand)
    if (card_is_basic_energy_type(id, etype)) return true;
  return false;
}

static bool hand_has_energy_card(const Player& p) {
  for (int id : p.hand) {
    const CardInfo* c = find_card(id);
    if (c && (c->cardType == BASIC_ENERGY || c->cardType == SPECIAL_ENERGY))
      return true;
  }
  return false;
}

static bool discard_has_pokemon_or_basic_energy(const Player& p) {
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if ((c && c->cardType == POKEMON) || card_is_basic_energy_type(id))
      return true;
  }
  return false;
}

static bool discard_has_no_rulebox_pokemon_or_basic_energy(const Player& p) {
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if ((c && c->cardType == POKEMON && !has_rule_box(c)) ||
        card_is_basic_energy_type(id))
      return true;
  }
  return false;
}

static bool discard_has_type_pokemon_or_basic_energy(const Player& p, int etype) {
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if ((c && c->cardType == POKEMON && pokemon_has_type(id, etype)) ||
        card_is_basic_energy_type(id, etype))
      return true;
  }
  return false;
}

static bool discard_has_pokemon(const Player& p) {
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if (c && c->cardType == POKEMON) return true;
  }
  return false;
}

static bool discard_has_supporter(const Player& p) {
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if (c && c->cardType == SUPPORTER) return true;
  }
  return false;
}

static bool inplay_has_any_energy(const InPlay& p) {
  return !p.energies.empty() || !p.energyCardIds.empty();
}

static bool player_has_any_energy_inplay(const Player& p) {
  if (p.activeKnown && inplay_has_any_energy(p.active)) return true;
  for (const auto& b : p.bench)
    if (inplay_has_any_energy(b)) return true;
  return false;
}

static bool inplay_has_basic_energy_card(const InPlay& p, int etype = -1) {
  if (!p.energyCardIds.empty()) {
    for (int id : p.energyCardIds)
      if (card_is_basic_energy_type(id, etype)) return true;
    return false;
  }
  for (int e : p.energies)
    if (etype < 0 || e == etype) return true;
  return false;
}

static bool player_has_basic_energy_inplay(const Player& p, int etype = -1) {
  if (p.activeKnown && inplay_has_basic_energy_card(p.active, etype)) return true;
  for (const auto& b : p.bench)
    if (inplay_has_basic_energy_card(b, etype)) return true;
  return false;
}

static bool player_has_special_energy_inplay(const Player& p) {
  auto has_special = [](const InPlay& pk) {
    for (int id : pk.energyCardIds) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == SPECIAL_ENERGY) return true;
    }
    return false;
  };
  if (p.activeKnown && has_special(p.active)) return true;
  for (const auto& b : p.bench)
    if (has_special(b)) return true;
  return false;
}

static bool player_has_tool_inplay(const Player& p) {
  if (p.activeKnown && !p.active.tools.empty()) return true;
  for (const auto& b : p.bench)
    if (!b.tools.empty()) return true;
  return false;
}

static bool player_has_ancient_inplay(const Player& p) {
  if (p.activeKnown && is_ancient_card_id(p.active.id)) return true;
  for (const auto& b : p.bench)
    if (is_ancient_card_id(b.id)) return true;
  return false;
}

static bool player_has_future_inplay(const Player& p) {
  if (p.activeKnown && is_future_card(p.active.id)) return true;
  for (const auto& b : p.bench)
    if (is_future_card(b.id)) return true;
  return false;
}

static bool bench_has_type(const Player& p, int etype) {
  for (const auto& b : p.bench) {
    if (pokemon_has_type(b.id, etype)) return true;
  }
  return false;
}

static bool player_has_type_inplay(const Player& p, int etype) {
  if (p.activeKnown && pokemon_has_type(p.active.id, etype)) return true;
  for (const auto& b : p.bench)
    if (pokemon_has_type(b.id, etype)) return true;
  return false;
}

static bool player_has_damaged_type_inplay(const Player& p, int etype) {
  if (p.activeKnown && is_damaged(p.active) && pokemon_has_type(p.active.id, etype))
    return true;
  for (const auto& b : p.bench)
    if (is_damaged(b) && pokemon_has_type(b.id, etype)) return true;
  return false;
}

static bool player_has_low_hp_damaged_inplay(const Player& p, int maxHp) {
  if (p.activeKnown && is_damaged(p.active) && p.active.hp <= maxHp)
    return true;
  for (const auto& b : p.bench)
    if (is_damaged(b) && b.hp <= maxHp) return true;
  return false;
}

static bool player_has_damaged_mega_ex_inplay(const Player& p) {
  auto ok = [](const InPlay& pk) {
    const CardInfo* c = find_card(pk.id);
    return c && c->megaEx && is_damaged(pk);
  };
  if (p.activeKnown && ok(p.active)) return true;
  for (const auto& b : p.bench)
    if (ok(b)) return true;
  return false;
}

static bool deck_has_salvatore_target(const Player& me) {
  return me.deckCount > 0 && own_inplay_count(me) > 0;
}

static bool deck_has_same_name_as_opp_inplay_candidate(const GameState& st) {
  const Player& me = st.players[st.yourIndex];
  if (me.deckCount <= 0) return false;
  // Loaded oracle states may not carry enough hidden deck identity to prove
  // the same-name source card, while CABT can still offer the search.
  return true;
}

static bool deck_has_type_pokemon(const Player& p, int etype) {
  (void)etype;
  if (p.deckCount <= 0) return false;
  return true;
}

static bool deck_has_mega_ex(const Player& p) {
  if (p.deckCount <= 0) return false;
  if (!p.deckKnown)
    return true;
  if (!p.deck.empty()) {
    for (int id : p.deck) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == POKEMON && c->megaEx) return true;
    }
    return false;
  }
  return true;
}

static bool deck_has_tool(const Player& p) {
  if (p.deckCount <= 0) return false;
  if (!p.deckKnown)
    return true;
  if (!p.deck.empty()) {
    for (int id : p.deck) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == TOOL) return true;
    }
    return false;
  }
  return true;
}

static bool deck_has_team_rocket_great_ball_target(const Player& p) {
  if (p.deckCount <= 0) return false;
  if (!p.deckKnown)
    return true;
  if (!p.deck.empty()) {
    for (int id : p.deck) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == POKEMON && is_team_rocket_card_id(id) &&
          (c->basic || c->stage1 || c->stage2))
        return true;
    }
    return false;
  }
  return true;
}

static bool deck_has_colress_target(const Player& p) {
  if (p.deckCount <= 0) return false;
  if (!p.deckKnown)
    return true;
  if (!p.deck.empty()) {
    for (int id : p.deck) {
      const CardInfo* c = find_card(id);
      if ((c && c->cardType == STADIUM) || card_is_basic_energy_type(id))
        return true;
    }
    return false;
  }
  return true;
}

static bool secret_box_playable(const Player& p) {
  return p.handCount >= 4 && p.deckCount > 0;
}

static bool bench_has_dark_non_pecharunt_ex(const Player& p) {
  for (const auto& b : p.bench) {
    const CardInfo* c = find_card(b.id);
    if (c && c->energyType == DARKNESS && b.id != 141) return true;
  }
  return false;
}

static bool player_has_bench_energy(const Player& p) {
  for (const auto& b : p.bench)
    if (inplay_has_any_energy(b)) return true;
  return false;
}

static bool player_has_bench_energy_type(const Player& p, int etype) {
  for (const auto& b : p.bench)
    for (int k = 0; k < attached_energy_slots(b); ++k)
      if (energy_unit_matches(b, k, etype + 1)) return true;
  return false;
}

static bool player_has_tool_or_special_energy_inplay(const Player& p) {
  auto ok = [](const InPlay& pk) {
    if (!pk.tools.empty()) return true;
    for (int id : pk.energyCardIds) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == SPECIAL_ENERGY) return true;
    }
    return false;
  };
  if (p.activeKnown && ok(p.active)) return true;
  for (const auto& b : p.bench)
    if (ok(b)) return true;
  return false;
}

static bool player_has_basic_bench(const Player& p) {
  for (const auto& b : p.bench) {
    if (is_antique_fossil(b.id)) return true;  // in play it is a Basic {C} Pokemon
    const CardInfo* c = find_card(b.id);
    if (c && c->cardType == POKEMON && c->basic) return true;
  }
  return false;
}

static bool hand_has_pokemon(const Player& p) {
  for (int id : p.hand) {
    const CardInfo* c = find_card(id);
    if (c && c->cardType == POKEMON) return true;
  }
  return false;
}

static bool all_own_inplay_team_rocket(const Player& p) {
  auto ok = [](const InPlay& pk) { return is_team_rocket_card_id(pk.id); };
  if (!p.activeKnown || !ok(p.active)) return false;
  for (const auto& b : p.bench)
    if (!ok(b)) return false;
  return true;
}

static bool active_and_bench_team_rocket(const Player& p) {
  if (!p.activeKnown || !is_team_rocket_card_id(p.active.id)) return false;
  for (const auto& b : p.bench)
    if (is_team_rocket_card_id(b.id)) return true;
  return false;
}

static bool discard_has_basic_pokemon_hp_le(const Player& p, int maxHp) {
  for (int id : p.discard) {
    const CardInfo* c = find_card(id);
    if (c && c->cardType == POKEMON && c->basic && c->hp <= maxHp) return true;
  }
  return false;
}

static bool player_has_evolved_psychic(const Player& p) {
  auto ok = [](const InPlay& pk) {
    const CardInfo* c = find_card(pk.id);
    return c && c->energyType == PSYCHIC && !pk.preEvo.empty();
  };
  if (p.activeKnown && ok(p.active)) return true;
  for (const auto& b : p.bench)
    if (ok(b)) return true;
  return false;
}

static bool deck_has_basic_bench_candidate(const GameState& st, int cardId) {
  const Player& me = st.players[st.yourIndex];
  if (static_cast<int>(me.bench.size()) >= effective_bench_max(st, st.yourIndex))
    return false;
  if (cardId == 1086 || cardId == 1115 || cardId == 1126)
    return me.deckCount > 0;
  if (!me.deckKnown)
    return me.deckCount > 0;
  auto matches = [cardId](int id) {
    const CardInfo* c = find_card(id);
    if (!c || c->cardType != POKEMON || !c->basic) return false;
    if (cardId == 1086) return c->hp <= 70;  // Buddy-Buddy Poffin
    return true;  // Precious Trolley and generic Basic-to-Bench searches.
  };
  if (!me.deck.empty()) {
    for (int id : me.deck)
      if (matches(id)) return true;
    return false;
  }
  return me.deckCount > 0;
}

static bool opponent_top_deck_has_basic_bench_candidate(const GameState& st,
                                                        int revealCount) {
  (void)revealCount;
  int oppIndex = 1 - st.yourIndex;
  const Player& opp = st.players[oppIndex];
  if (static_cast<int>(opp.bench.size()) >= effective_bench_max(st, oppIndex))
    return false;
  return opp.deckCount > 0;
}

static bool deck_has_type_pokemon_top_bench_candidate(const GameState& st,
                                                      int etype,
                                                      int revealCount) {
  (void)etype;
  (void)revealCount;
  const Player& me = st.players[st.yourIndex];
  if (static_cast<int>(me.bench.size()) >= effective_bench_max(st, st.yourIndex))
    return false;
  return me.deckCount > 0 || !me.deck.empty();
}

static bool mystery_garden_available(const GameState& st, int player) {
  return hand_has_energy_card(st.players[player]);
}

static bool ability_effect_available(const GameState& st, int cardId,
                                     bool isActive) {
  const Player& me = st.players[st.yourIndex];
  switch (cardId) {
    case 96:   // Teal Mask Ogerpon ex: attach Basic Grass Energy from hand.
      return hand_has_basic_energy(me, GRASS);
    case 141:  // Pecharunt ex: switch a Benched Darkness non-Pecharunt ex.
      return bench_has_dark_non_pecharunt_ex(me);
    case 326:  // Blaziken ex: attach a Basic Energy from discard.
      return discard_has_basic_energy(me);
    case 357:  // Ethan's Ho-Oh ex: cabt offers when Fire Energy is available.
      return hand_has_basic_energy(me, FIRE);
    case 505:  // Alomomola: put a small Basic from discard onto the Bench.
      return isActive &&
             static_cast<int>(me.bench.size()) <
                 effective_bench_max(st, st.yourIndex) &&
             discard_has_basic_pokemon_hp_le(me, 70);
    case 569:  // Emboar: attach Basic Fire Energy from hand.
      return hand_has_basic_energy(me, FIRE);
    case 711:  // Shuckle: heal 30 damage from one of your Pokemon.
      return player_has_damaged_inplay(me);
    case 112:  // Munkidori: move counters from one damaged own Pokemon.
      return player_has_damaged_inplay(me);
    case 1009:  // Larry's Komala: attach an Energy card from hand.
      return hand_has_energy_card(me);
    case 1029:  // Dewgong: move a Water Energy from Bench to Active.
      return me.activeKnown && player_has_bench_energy_type(me, WATER);
    default:
      return true;
  }
}

static bool trainer_effect_playable(const GameState& st, int cardId) {
  const Player& me = st.players[st.yourIndex];
  const Player& opp = st.players[1 - st.yourIndex];
  switch (cardId) {
    case 1081:
      return player_has_special_energy_inplay(opp);
    case 1085:
      return player_has_ancient_inplay(me);
    case 1086:
    case 1115:
    case 1126:
      return deck_has_basic_bench_candidate(st, cardId);
    case 1088:
      return !opp.bench.empty();
    case 1087:
      return me.handCount > 6 || opp.handCount > 5;
    case 1089:
      return discard_has_basic_energy(me) && player_has_future_inplay(me);
    case 1091:
      return opponent_top_deck_has_basic_bench_candidate(st, 5);
    case 1092:
      return secret_box_playable(me);
    case 1083:
      return deck_has_same_name_as_opp_inplay_candidate(st);
    case 1077:
    case 1082:
    case 1094:
    case 1106:
    case 1121:
    case 1122:
    case 1127:
    case 1134:
    case 1119:
    case 1198:
    case 1202:
    case 1205:
    case 1210:
    case 1215:
    case 1219:
    case 1220:
    case 1225:
    case 1231:
    case 1235:
      return me.deckCount > 0;
    case 1111:
      return deck_has_tool(me);
    case 1125:
      return me.deckCount > 0;
    case 1132:
      return deck_has_team_rocket_great_ball_target(me);
    case 1096:
      return player_has_damaged_inplay(me);
    case 1098:
      return discard_has_basic_energy(me) && bench_has_type(me, COLORLESS);
    case 1104:
      return !st.stadium.empty() || player_has_tool_inplay(opp) ||
             player_has_special_energy_inplay(opp);
    case 1107:
      return !me.bench.empty();
    case 1109:
      return discard_has_supporter(me);
    case 1097:
    case 1110:
      return discard_has_pokemon_or_basic_energy(me);
    case 1105:
      return active_has_type(me, DRAGON) && active_is_damaged(me);
    case 1108:
      return second_player_turn1(st) && player_has_any_energy_inplay(opp);
    case 1112:
    case 1117:
      return player_has_damaged_inplay(me);
    case 1113:
      return discard_has_basic_energy(me);
    case 1116:
      return own_inplay_count(me) >= 2 && player_has_basic_energy_inplay(me);
    case 1118:
    case 1139:
      return discard_has_basic_energy(me);
    case 1120:
      return player_has_any_energy_inplay(opp);
    case 1124:
    case 1143:
      return !opp.bench.empty();
    case 1131:
      return opp.handCount > 0 && opp.prizeCount > 0;
    case 1129:
      return discard_has_pokemon(me);
    case 1130:
      return active_is_damaged(me);
    case 1137:
      return player_has_tool_inplay(me) || player_has_tool_inplay(opp);
    case 1144:
      return player_has_evolved_psychic(me);
    case 1145:
      return deck_has_mega_ex(me);
    case 1146:
      return discard_has_basic_energy(me, PSYCHIC) && bench_has_type(me, PSYCHIC);
    case 1147:
      return active_is_damaged(me) && active_energy_count_ge(me, 3);
    case 1148:
      return hand_has_basic_energy(me, FIRE) &&
             (!st.stadium.empty() || player_has_tool_inplay(opp) ||
              player_has_special_energy_inplay(opp));
    case 1153:
      return active_is_damaged(me) || player_has_status(me);
    case 1183:
      return hand_has_pokemon(me);
    case 1184:
      return discard_has_no_rulebox_pokemon_or_basic_energy(me);
    case 1187:
      return !opp.bench.empty() && me.deckCount > 0;
    case 1188:
      return me.deckCount > 0;
    case 1189:
      return deck_has_salvatore_target(me);
    case 1190:
      return player_has_low_hp_damaged_inplay(me, 30);
    case 1194:
      return deck_has_colress_target(me);
    case 1195:
      return player_has_type_inplay(me, DARKNESS) &&
             me.deckCount > 0;
    case 1197:
      return opp.handCount > 3;
    case 1203:
      return !me.bench.empty();
    case 1204:
      return player_has_basic_bench(opp);
    case 1206:
      return me.deckCount > 0 || me.handCount > 1;
    case 1208:
      return me.handCount >= 2 && me.handCount <= 7;
    case 1209:
      return player_has_tool_or_special_energy_inplay(opp);
    case 1212:
      return active_is_damaged(me);
    case 1216: {
      int drawTo = all_own_inplay_team_rocket(me) ? 8 : 5;
      return me.deckCount > 0 && me.handCount <= drawTo;
    }
    case 1218:
      return active_and_bench_team_rocket(me) && !opp.bench.empty();
    case 1221:
      return player_has_bench_energy(me);
    case 1222:
      return player_has_damaged_inplay(me);
    case 1229:
      return player_has_damaged_mega_ex_inplay(me);
    case 1230:
      return deck_has_type_pokemon_top_bench_candidate(st, DARKNESS, 7);
    case 1232:
      return me.deckCount > 0;
    case 1233:
      return me.handCount >= 2 && deck_has_type_pokemon(me, LIGHTNING);
    case 1238:
      return discard_has_type_pokemon_or_basic_energy(me, FIGHTING);
    case 1240:
      return discard_has_basic_energy(me) && in_play_has_stage2(me);
    case 1241:
      return player_has_damaged_type_inplay(me, PSYCHIC);
    default:
      return true;
  }
}

static bool card_gate_ok(const GameState& st, int cardId) {
  if (cardId == 1090)
    return ogres_mask_available(st, st.yourIndex);
  if (cardId == 1193)
    return st.lastAttackDamageKoTurn[st.yourIndex] == st.turn - 1;
  const Player& me = st.players[st.yourIndex];
  const Player& opp = st.players[1 - st.yourIndex];
  switch (card_gate(cardId)) {
    case CG_NONE:
      return true;
    case CG_KO_LAST_TURN:
      return st.lastKoTurn[st.yourIndex] == st.turn - 1;
    case CG_TEAM_ROCKET_KO_LAST_TURN:
      return st.lastTeamRocketKoTurn[st.yourIndex] == st.turn - 1;
    case CG_SECOND_PLAYER_TURN1:
      return second_player_turn1(st);
    case CG_LAST_CARD_IN_HAND:
      return me.handCount == 1;
    case CG_MORE_PRIZES_THAN_OPP:
      return me.prizeCount > opp.prizeCount;
    case CG_TERA_IN_PLAY:
      return player_has_tera(me);
    case CG_OTHER_CARD_IN_HAND:
      return me.handCount >= 2;
    case CG_OWN_HAND_GE_3:
      return me.handCount >= 3;
    case CG_NOT_ACTOR_FIRST_TURN:
      return !actor_first_turn(st);
    case CG_OPP_PRIZES_EQ_2:
      return opp.prizeCount == 2;
    case CG_ALL_N_TEAM:
      return all_n_team_in_play(me);
    case CG_OWN_DECK_GT_0:
      return me.deckCount > 0;
    case CG_OPP_PRIZES_LE_2:
      return opp.prizeCount <= 2;
    case CG_RARE_CANDY:
      return rare_candy_gate_ok(st, me);
    case CG_OPP_HAND_GT_0:
      return opp.handCount > 0;
    default:
      return false;
  }
}

std::vector<Descriptor> legal_main(const GameState& st) {
  std::vector<Descriptor> out;
  if (st.result >= 0) return out;
  const Player& me = st.players[st.yourIndex];
  int benchLimit = effective_bench_max(st, st.yourIndex);
  // Loop-invariant lock checks, hoisted out of the per-hand-card loop.
  const bool itemLocked = item_locked(st, st.yourIndex);
  const bool supporterLocked = supporter_locked(st, st.yourIndex);
  const bool stadiumLocked = stadium_locked(st, st.yourIndex);
  const bool toolLocked = tool_locked(st, st.yourIndex);
  const bool aceSpecLocked = ace_spec_locked(st, st.yourIndex);
  const bool evolutionLocked = evolution_locked(st, st.yourIndex);

  if (st.turn <= 0 && !me.activePresent && me.handKnown &&
      std::find(me.hand.begin(), me.hand.end(), 666) != me.hand.end())
    out.push_back({Atom::S("SETUP_ACTIVE"), Atom::I(666)});

  // In-play targets in cabt index order: active (0), then bench (0..).
  SmallVec<std::tuple<const char*, int, const InPlay*>, 6> targets;
  if (me.activePresent && me.activeKnown)
    targets.emplace_back("ACTIVE", 0, &me.active);
  for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
    targets.emplace_back("BENCH", j, &me.bench[j]);

  // Playing / evolving from hand.
  for (int hid : me.hand) {
    const CardInfo* ci = find_card(hid);
    if (!ci) continue;
    switch (ci->cardType) {
      case BASIC_ENERGY:
      case SPECIAL_ENERGY:
        if (!st.energyAttached) {
          for (auto& [area, idx, pk] : targets) {
            if (hid == 15 && (!pk || !is_team_rocket_card_id(pk->id)))
              continue;
            if (ci->aceSpec && aceSpecLocked) continue;
            out.push_back({Atom::S("ATTACH"), Atom::I(hid), Atom::S(area),
                           Atom::I(idx)});
          }
        }
        break;
      case POKEMON:
        if (ci->basic && pokemon_played_as_basic_from_hand(st, st.yourIndex, hid, ci)) {
          if (static_cast<int>(me.bench.size()) < benchLimit &&
              !pokemon_from_hand_blocked(st, st.yourIndex, ci))
            out.push_back({Atom::S("PLAY"), Atom::I(hid)});
        }
        if (!ci->basic && hid != 107 &&
            !evolutionLocked &&
            !pokemon_from_hand_blocked(st, st.yourIndex, ci)) {
          for (auto& [area, idx, pk] : targets) {
            bool boostedEevee =
                pk->id == 317 && std::strcmp(area, "ACTIVE") == 0;
            bool stimulatedKarrablast = pk->id == 487 && in_play_has(me, 564);
            bool stimulatedShelmet = pk->id == 564 && in_play_has(me, 487);
            bool fightingRoarLuxio = false;
            if (pk->id == 1036) {
              const Player& opp = st.players[1 - st.yourIndex];
              const CardInfo* oc =
                  opp.activeKnown ? find_card(opp.active.id) : nullptr;
              fightingRoarLuxio = oc && (oc->ex || oc->megaEx);
            }
            bool forestVitality = false;
            if (cur_stadium(st) == 1261 && !actor_first_turn(st) &&
                !ci->basic) {
              const CardInfo* pc = find_card(pk->id);
              forestVitality =
                  pc && pc->energyType == GRASS && ci->energyType == GRASS;
            }
            bool timingOk = (st.turn >= 3 && !pk->appearThisTurn &&
                             pk->noEvolveTurn != st.turn) || boostedEevee ||
                            stimulatedKarrablast || stimulatedShelmet ||
                            fightingRoarLuxio || forestVitality;
            const CardInfo* pci = find_card(pk->id);
            bool rainbowDna =
                pk->id == 249 && ci->ex && ci->evolvesFrom &&
                std::strcmp(ci->evolvesFrom, "Eevee") == 0;
            if (pci && ci->evolvesFrom && pci->name &&
                ((std::strcmp(ci->evolvesFrom, pci->name) == 0) ||
                 rainbowDna) && timingOk)
              out.push_back({Atom::S("EVOLVE"), Atom::I(hid), Atom::S(area),
                             Atom::I(idx)});
          }
        }
        if (!ci->basic &&
            pokemon_played_as_basic_from_hand(st, st.yourIndex, hid, ci)) {
          if (static_cast<int>(me.bench.size()) < benchLimit &&
              !pokemon_from_hand_blocked(st, st.yourIndex, ci))
            out.push_back({Atom::S("PLAY"), Atom::I(hid)});
        }
        break;
      case ITEM: {
        bool ok = !itemLocked;
        if (ci->aceSpec && aceSpecLocked) ok = false;
        if (!card_gate_ok(st, hid)) ok = false;
        if (!trainer_effect_playable(st, hid)) ok = false;
        if (is_antique_fossil(hid) &&
            static_cast<int>(me.bench.size()) >= benchLimit)
          ok = false;  // Fossils are played as Basic Pokemon onto the Bench.
        if (hid == 1123 && me.bench.empty())
          ok = false;  // Switch needs a Benched Pokemon
        if ((hid == 1102 || hid == 1142 || hid == 1152) && me.deckCount == 0)
          ok = false;  // Dusk Ball / Fighting Gong / Poke Pad need a non-empty deck
        if (ok) out.push_back({Atom::S("PLAY"), Atom::I(hid)});
        break;
      }
      case SUPPORTER: {
        // The player going first can't play a Supporter on turn 1 (Carmine and
        // Team Rocket's Proton are exceptions).
        bool turn_ok = st.turn != 1 || supporter_turn1_exception(hid);
        bool target_ok =
            hid != 1182 || !st.players[1 - st.yourIndex].bench.empty();
        bool lock_ok = !supporterLocked &&
                       !(ci->aceSpec && aceSpecLocked);
        if (!st.supporterPlayed && turn_ok && target_ok && lock_ok &&
            card_gate_ok(st, hid) && trainer_effect_playable(st, hid))
          out.push_back({Atom::S("PLAY"), Atom::I(hid)});
        break;
      }
      case STADIUM:
        if (!st.stadiumPlayed && !stadiumLocked &&
            !(ci->aceSpec && aceSpecLocked) &&
            (st.stadium.empty() || st.stadium[0] != hid) &&
            card_gate_ok(st, hid))
          out.push_back({Atom::S("PLAY"), Atom::I(hid)});
        break;
      case TOOL:
        // Tools attach to an in-play Pokemon (one tool per Pokemon).
        for (auto& [area, idx, pk] : targets)
          if (static_cast<int>(pk->tools.size()) <
                  tool_capacity(st, st.yourIndex, *pk) &&
              !toolLocked &&
              !(ci->aceSpec && aceSpecLocked) &&
              card_gate_ok(st, hid))
            out.push_back({Atom::S("ATTACH"), Atom::I(hid), Atom::S(area),
                           Atom::I(idx)});
        break;
      default:
        break;
    }
  }

  // Activated abilities: Lunatone Lunar Cycle (needs Solrock in play + a Basic
  // {F} Energy in hand + not used yet this turn).
  if (!st.lunarUsedThisTurn) {
    bool solrock = me.activeKnown && me.active.id == 676;
    for (auto& b : me.bench)
      if (b.id == 676) solrock = true;
    bool fEnergy = false;
    for (int id : me.hand)
      if (id == 6) fEnergy = true;
    if (solrock && fEnergy) {
      if (me.activeKnown && me.active.id == 675 &&
          !ability_suppressed(st, st.yourIndex, me.active, true) &&
          !(me.active.reconstructedAbilityLocked &&
            me.active.reconstructedAbilityLockTurn == st.turn))
        out.push_back({Atom::S("ABILITY"), Atom::S("ACTIVE"), Atom::I(0), Atom::N()});
      for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
        if (me.bench[j].id == 675 &&
            !ability_suppressed(st, st.yourIndex, me.bench[j], false) &&
            !(me.bench[j].reconstructedAbilityLocked &&
              me.bench[j].reconstructedAbilityLockTurn == st.turn))
          out.push_back({Atom::S("ABILITY"), Atom::S("BENCH"), Atom::I(j), Atom::N()});
    }
  }

  // General activated once-per-turn abilities (data-driven via ability_program):
  // offer for each in-play Pokemon that has one, hasn't used it this turn, and
  // whose gate passes (if-self-active / if-self-has-energy).
  auto gate_ok = [&st](int id, bool isActive, const InPlay& pk) {
    switch (ability_gate(id)) {
      case AG_NONE: return true;
      case AG_SELF_ACTIVE: return isActive;
      case AG_SELF_HAS_ENERGY: return !pk.energies.empty();
      case AG_STADIUM: return !st.stadium.empty();
      case AG_PLAYED_SUPPORTER: return st.supporterPlayed;
      case AG_KO_LAST_TURN: return st.lastKoTurn[st.yourIndex] == st.turn - 1;
      case AG_TERA: {
        const CardInfo* c = find_card(id);
        return c && c->tera;
      }
      case AG_SELF_BENCH:
        return !isActive;
      case AG_SELF_BENCH_ACTIVE_LARRY: {
        if (isActive || !st.players[st.yourIndex].activeKnown) return false;
        const CardInfo* c = find_card(st.players[st.yourIndex].active.id);
        return c && name_contains(c->name, "Larry");
      }
      case AG_HAS_FIRE_MEGA_EX: {
        const Player& me = st.players[st.yourIndex];
        auto ok = [](const InPlay& p) {
          const CardInfo* c = find_card(p.id);
          return c && c->energyType == FIRE && c->megaEx;
        };
        if (me.activeKnown && ok(me.active)) return true;
        for (const auto& b : me.bench)
          if (ok(b)) return true;
        return false;
      }
      case AG_PLAYED_TEAM_ROCKET_SUPPORTER:
        return st.teamRocketSupporterPlayed;
      case AG_ACTIVE_HAS_FESTIVAL_LEAD: {
        const Player& me = st.players[st.yourIndex];
        if (!me.activeKnown) return false;
        return me.active.id == 93 || me.active.id == 100 ||
               me.active.id == 240 || me.active.id == 247;
      }
      case AG_OWN_HAND_GE_1:
        return st.players[st.yourIndex].handCount >= 1;
      case AG_OWN_HAND_GE_2:
        return st.players[st.yourIndex].handCount >= 2;
      case AG_FIRST_TURN:
        return actor_first_turn(st);
      case AG_SELF_HAS_GRASS_ENERGY:
        return has_energy_type(pk, GRASS);
      case AG_OWN_HAS_GRASS_MEGA_EX: {
        const Player& me = st.players[st.yourIndex];
        auto ok = [](const InPlay& p) {
          const CardInfo* c = find_card(p.id);
          return c && c->energyType == GRASS && c->megaEx;
        };
        if (me.activeKnown && ok(me.active)) return true;
        for (const auto& b : me.bench)
          if (ok(b)) return true;
        return false;
      }
      case AG_SELF_BENCH_OWN_HAS_MEGA_EX: {
        if (isActive) return false;
        const Player& me = st.players[st.yourIndex];
        auto ok = [](const InPlay& p) {
          const CardInfo* c = find_card(p.id);
          return c && c->megaEx;
        };
        if (me.activeKnown && ok(me.active)) return true;
        for (const auto& b : me.bench)
          if (ok(b)) return true;
        return false;
      }
      case AG_PLAYED_CANARI:
        return st.canariPlayed;
      case AG_SELF_HAS_DARKNESS_ENERGY:
        return has_energy_type(pk, DARKNESS);
      case AG_SELF_HAS_BASIC_LIGHTNING_ENERGY:
        return inplay_has_basic_energy_card(pk, LIGHTNING);
      case AG_OWN_HAND_HAS_ENERGY: {
        const Player& me = st.players[st.yourIndex];
        for (int handCardId : me.hand) {
          const CardInfo* c = find_card(handCardId);
          if (c && (c->cardType == BASIC_ENERGY ||
                    c->cardType == SPECIAL_ENERGY))
            return true;
        }
        return false;
      }
      default: return false;
    }
  };
    if (me.activeKnown && ability_program(me.active.id) >= 0 &&
        !ability_suppressed(st, st.yourIndex, me.active, true) &&
        !ability_suppressed_by_self_ko_lock(st, me.active.id) &&
        !(me.active.reconstructedAbilityLocked &&
          me.active.reconstructedAbilityLockTurn == st.turn) &&
        (!me.active.abilityUsedThisTurn || ability_repeatable(me.active.id)) &&
        ability_group_available(st, st.yourIndex, me.active.id) &&
        ability_effect_available(st, me.active.id, true) &&
        gate_ok(me.active.id, true, me.active))
      out.push_back({Atom::S("ABILITY"), Atom::S("ACTIVE"), Atom::I(0), Atom::N()});
    for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
      if (ability_program(me.bench[j].id) >= 0 &&
          !ability_suppressed(st, st.yourIndex, me.bench[j], false) &&
          !ability_suppressed_by_self_ko_lock(st, me.bench[j].id) &&
          !(me.bench[j].reconstructedAbilityLocked &&
            me.bench[j].reconstructedAbilityLockTurn == st.turn) &&
          (!me.bench[j].abilityUsedThisTurn || ability_repeatable(me.bench[j].id)) &&
          ability_group_available(st, st.yourIndex, me.bench[j].id) &&
          ability_effect_available(st, me.bench[j].id, false) &&
          gate_ok(me.bench[j].id, false, me.bench[j]))
        out.push_back({Atom::S("ABILITY"), Atom::S("BENCH"), Atom::I(j), Atom::N()});

  // Stadium once-per-turn ability (each player may use the in-play Stadium's).
  auto stadium_gate_ok = [&st]() {
    if (st.stadium.empty()) return false;
    if (st.stadium[0] == 1254)
      return discard_has_basic_energy(st.players[st.yourIndex], LIGHTNING);
    if (st.stadium[0] == 1262)
      return active_has_type(st.players[st.yourIndex], WATER) &&
             bench_has_type(st.players[st.yourIndex], WATER);
    if (st.stadium[0] == 1263)
      return mystery_garden_available(st, st.yourIndex);
    if (st.stadium[0] == 1267)
      return deck_has_basic_bench_candidate(st, 1267);
    if (st.stadium[0] == 1259)
      return st.players[st.yourIndex].deckCount > 0;
    if (st.stadium[0] == 1242)
      return st.supporterPlayed &&
             player_has_damaged_inplay(st.players[st.yourIndex]);
    switch (ability_gate(st.stadium[0])) {
      case AG_NONE: return true;
      case AG_STADIUM: return !st.stadium.empty();
      case AG_PLAYED_SUPPORTER: return st.supporterPlayed;
      case AG_PLAYED_TEAM_ROCKET_SUPPORTER: return st.teamRocketSupporterPlayed;
      case AG_KO_LAST_TURN: return st.lastKoTurn[st.yourIndex] == st.turn - 1;
      case AG_OWN_HAND_GE_1: return st.players[st.yourIndex].handCount >= 1;
      case AG_OWN_HAND_GE_2: return st.players[st.yourIndex].handCount >= 2;
      case AG_PLAYED_CANARI: return st.canariPlayed;
      case AG_SELF_HAS_DARKNESS_ENERGY: return false;
      case AG_SELF_HAS_BASIC_LIGHTNING_ENERGY: return false;
      case AG_OWN_HAND_HAS_ENERGY: {
        const Player& me = st.players[st.yourIndex];
        for (int handCardId : me.hand) {
          const CardInfo* c = find_card(handCardId);
          if (c && (c->cardType == BASIC_ENERGY ||
                    c->cardType == SPECIAL_ENERGY))
            return true;
        }
        return false;
      }
      default: return false;
    }
  };
  if (!st.stadium.empty() && stadium_ability_program(st.stadium[0]) >= 0 &&
      !st.stadiumAbilityUsed && stadium_gate_ok())
    out.push_back({Atom::S("ABILITY"), Atom::S("STADIUM"), Atom::I(0), Atom::N()});
  if (cur_stadium(st) == 1249 && !st.stadiumAbilityUsed &&
      grand_tree_has_stage1_option(st, st.yourIndex))
    out.push_back({Atom::S("ABILITY"), Atom::S("STADIUM"), Atom::I(0), Atom::N()});

  // Antique Fossils can be discarded from play during their controller's turn.
  if (me.activeKnown && is_antique_fossil(me.active.id) && !me.bench.empty())
    out.push_back({Atom::S("DISCARD"), Atom::S("ACTIVE"), Atom::I(0)});
  for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
    if (is_antique_fossil(me.bench[j].id))
      out.push_back({Atom::S("DISCARD"), Atom::S("BENCH"), Atom::I(j)});

  // Attacking (the player going first cannot attack on turn 1; an Asleep or
  // Paralyzed Active cannot attack; nor one under a "can't use attacks" effect).
  bool energyLeAttackLocked =
      st.noAttackEnergyLeTurn[st.yourIndex] == st.turn &&
      st.noAttackEnergyLeThreshold[st.yourIndex] >= 0 &&
      static_cast<int>(me.active.energies.size()) <=
          st.noAttackEnergyLeThreshold[st.yourIndex];
  if (me.activePresent && me.activeKnown &&
      !me.asleep && !me.paralyzed && me.active.noAttackTurn != st.turn &&
      !energyLeAttackLocked) {
    const CardInfo* aci = find_card(me.active.id);
    auto attack_affordable = [&st, &me](const AttackInfo& at) {
      int extra = me.active.attackCostModTurn == st.turn ? me.active.attackCostMod : 0;
      int anyDiscount = 0;
      const Player& opp = st.players[1 - st.yourIndex];
      const CardInfo* aci = find_card(me.active.id);
      if (me.active.id == 79)
        extra -= static_cast<int>(opp.bench.size());  // Incineroar ex: Hustle Play
      if (me.active.id == 1022 && opp.handCount == 4)
        extra -= static_cast<int>(std::count(at.cost, at.cost + at.n_cost,
                                             COLORLESS));
      if (me.active.id == 44 && at.id == 41)
        extra -= opp.prizeCount < 6 ? 6 - opp.prizeCount : 0;  // Bloodmoon ex
      if (me.active.id == 156 || me.active.id == 159)
        extra -= discard_name_count(me, "Kofu");
      if (opp.activeKnown && opp.active.id == 1099 && aci && aci->basic)
        extra += 1;  // Antique Root Fossil: opponent's Basic attacks cost {C} more.
      if (aci && aci->tera && active_tool(st, me.active, 1165))
        anyDiscount += 1;  // Sparkling Crystal: one Energy of any type less.
      if (active_tool(st, me.active, 1168) &&
          me.prizeCount > opp.prizeCount)
        extra -= 1;  // Counter Gain: {C} less when behind on prizes.
      if (aci && name_contains(aci->name, "Hop") && active_tool(st, me.active, 1171))
        extra -= 1;  // Hop's Choice Band: Hop's Pokemon attacks cost {C} less.
      if (aci && aci->tera && cur_stadium(st) == 1266)
        extra += 1;  // Nighttime Mine: Tera Pokemon attacks cost {C} more.
      if (affordable(me.active, at, extra, anyDiscount, &st, st.yourIndex))
        return true;
      if (me.active.id == 686 && at.id == 993 &&
          me.active.hp < me.active.maxHp) {
        for (int e : me.active.energies)
          if (e == DARKNESS || e == RAINBOW) return true;
      }
      if (me.active.id == 115 && at.id == 146 &&
          (me.poisoned || me.burned || me.asleep || me.paralyzed || me.confused))
        return true;
      if (me.active.id == 315 && at.id == 439 &&
          player_has_tera(st.players[st.yourIndex]) &&
          has_energy_type(me.active, PSYCHIC))
        return true;
      return me.active.id == 144 && at.id == 188 && !me.active.energies.empty() &&
             discard_has_name(st.players[1 - st.yourIndex], "Colress");
    };
    SmallVec<int, 8> offeredAttackIds;
    auto push_attack_id = [&](int attackId, bool dedupe = true) {
      if (dedupe &&
          std::find(offeredAttackIds.begin(), offeredAttackIds.end(), attackId) !=
              offeredAttackIds.end())
        return;
      out.push_back({Atom::S("ATTACK"), Atom::I(attackId)});
      if (dedupe)
        offeredAttackIds.push_back(attackId);
    };
    auto reconstructed_attack_locked = [&](int attackId) {
      return me.active.reconstructedAttackLockTurn == st.turn &&
             std::find(me.active.reconstructedAttackLocks.begin(),
                       me.active.reconstructedAttackLocks.end(),
                       attackId) != me.active.reconstructedAttackLocks.end();
    };
    auto offer_attack = [&](const AttackInfo& at) {
      if (attack_turn_ok(st, at.id) && attack_gate_ok(st, at.id) &&
          attack_effect_available(st, at.id) &&
          attack_affordable(at) &&
          !(me.active.lockTurn == st.turn && me.active.lockId == at.id) &&
          me.active.activeLockId != at.id &&
          !reconstructed_attack_locked(at.id)) {
        if (direct_opp_active_copy_attack(me.active.id, at.id)) {
          const Player& opp = st.players[1 - st.yourIndex];
          const CardInfo* oc = opp.activeKnown ? find_card(opp.active.id) : nullptr;
          if (oc) {
            for (int k = 0; k < oc->n_attacks; ++k)
              if (!reconstructed_attack_locked(oc->attacks[k].id))
                push_attack_id(oc->attacks[k].id, false);
          }
        } else if (me.active.id == 293 && at.id == 403) {
          for (const auto& b : me.bench) {
            const CardInfo* bc = find_card(b.id);
            if (!is_ns_pokemon(bc)) continue;
            for (int k = 0; k < bc->n_attacks; ++k)
              push_attack_id(bc->attacks[k].id, false);
          }
        } else {
          push_attack_id(at.id);
        }
      }
    };
    if (aci) {
      for (int k = 0; k < aci->n_attacks; ++k)
        offer_attack(aci->attacks[k]);
      if (relicanth_memory_available(st, st.yourIndex) &&
          !me.active.preEvo.empty()) {
        for (int preId : me.active.preEvo) {
          const CardInfo* pre = find_card(preId);
          if (!pre) continue;
          for (int k = 0; k < pre->n_attacks; ++k)
            offer_attack(pre->attacks[k]);
        }
      }
      if (me.active.id == 1056 && active_tool(st, me.active, 1180)) {
        if (const AttackInfo* geobuster = card_attack(1180, 1556))
          offer_attack(*geobuster);
      }
    }
  }

  // Retreating (an Asleep or Paralyzed Active cannot retreat; nor one locked).
  if (me.activePresent && me.activeKnown && !st.retreated &&
      !me.bench.empty() && !me.asleep && !me.paralyzed &&
      me.active.noRetreatTurn != st.turn &&
      !is_antique_fossil(me.active.id)) {
    int rc = retreat_cost(st, st.yourIndex);
    if (static_cast<int>(provided_energy_units(me.active, &st, st.yourIndex).size()) >= rc)
      out.push_back({Atom::S("RETREAT")});
  }

  out.push_back({Atom::S("END")});
  return out;
}

// --- transitions ----------------------------------------------------------

template <typename Cards>
static void remove_one(Cards& v, int id) {
  auto it = std::find(v.begin(), v.end(), id);
  if (it != v.end()) v.erase(it);
}

// HP added by attached tools. Placeholder for the S3 HP-modifier framework;
// only Hero's Cape (+100) is in the current pool.
static int tool_hp_bonus(int cardId, const std::vector<int>& tools,
                         bool toolEffectsActive) {
  if (!toolEffectsActive) return 0;
  int b = 0;
  const CardInfo* c = find_card(cardId);
  for (int t : tools) {
    if (t == 1159) b += 100;  // Hero's Cape
    if (t == 1173 && c && name_contains(c->name, "Cynthia"))
      b += 70;  // Cynthia's Power Weight
  }
  return b;
}

static int owner_hp_bonus(const GameState& st, int ownerSide) {
  if (owner_has_active_ability_card(st, ownerSide, 262))
    return 40;  // Ludicolo: non-stacking +40 HP to your Pokemon in play.
  return 0;
}

// HP modifier a given stadium applies to a Pokemon (affects both players):
// Gravity Mountain (-30 to Stage 2), Lively Stadium (+30 to Basics).
static int stadium_hp_delta(int cardId, int stadiumId) {
  if (stadiumId < 0) return 0;
  const CardInfo* c = find_card(cardId);
  if (!c) return 0;
  switch (stadiumId) {
    case 1252: return c->stage2 ? -30 : 0;  // Gravity Mountain
    case 1251: return c->basic ? 30 : 0;    // Lively Stadium
    default: return 0;
  }
}

static int cur_stadium(const GameState& st) {
  return st.stadium.empty() ? -1 : st.stadium[0];
}

// A Pokemon's max HP = printed HP + tools + the active stadium's modifier
// (used when a Pokemon first enters play under the current stadium).
static int attached_energy_hp_bonus(int cardId,
                                    const std::vector<int>& energyCardIds,
                                    const std::vector<int>& energies) {
  if (cardId == 116) {
    for (int e : energies)
      if (e == DARKNESS || e == RAINBOW)
        return 100;  // Okidogi: +100 HP with any {D} Energy.
  }
  if (cardId == 1054) {
    for (int id : energyCardIds) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == SPECIAL_ENERGY)
        return 150;  // Tyrantrum: +150 HP with any Special Energy.
    }
  }
  if (cardId == 530) {
    int n = 0;
    for (int e : energies)
      if (e == FIGHTING) ++n;
    return 40 * n;  // Conkeldurr: +40 HP for each {F} Energy attached.
  }
  if (pokemon_has_type(cardId, GRASS)) {
    int growGrass = 0;
    for (int id : energyCardIds)
      if (id == 18) ++growGrass;  // Grow Grass Energy: +20 per copy on a {G} Pokemon
    if (growGrass > 0) return 20 * growGrass;
  }
  return 0;
}

static int effective_max_hp(int cardId, const SmallVec<int, 4>& tools,
                            const GameState& st,
                            const SmallVec<int, 16>& energyCardIds,
                            const SmallVec<int, 16>& energies,
                            int ownerSide) {
  const CardInfo* c = find_card(cardId);
  return (c ? c->hp : 0) +
         tool_hp_bonus(cardId, tools, !tool_effects_disabled(st)) +
         stadium_hp_delta(cardId, cur_stadium(st)) +
         attached_energy_hp_bonus(cardId, energyCardIds, energies) +
         owner_hp_bonus(st, ownerSide);
}

static void refresh_inplay_max_hp(GameState& st, InPlay& p, int ownerSide) {
  if (ownerSide < 0) ownerSide = owner_of_inplay(st, p);
  int damage = p.maxHp - p.hp;
  p.maxHp = effective_max_hp(p.id, p.tools, st, p.energyCardIds, p.energies,
                             ownerSide);
  p.hp = std::max(0, p.maxHp - damage);
}

static void refresh_player_inplay_max_hp(GameState& st, int ownerSide) {
  if (ownerSide < 0 || ownerSide >= 2) return;
  Player& p = st.players[ownerSide];
  if (p.activeKnown) refresh_inplay_max_hp(st, p.active, ownerSide);
  for (auto& b : p.bench) refresh_inplay_max_hp(st, b, ownerSide);
}

// Apply the NET HP change to every in-play Pokemon when the stadium goes
// oldId -> newId (delta-based, so it preserves each Pokemon's existing max HP
// baseline and damage). A Pokemon dropped to <=0 is KO'd by the next sweep.
static void apply_stadium_hp_change(GameState& st, int oldId, int newId) {
  if (oldId == newId) return;
  for (int s = 0; s < 2; ++s) {
    Player& p = st.players[s];
    auto fix = [oldId, newId](InPlay& k) {
      if (k.id == 0) return;
      int d = stadium_hp_delta(k.id, newId) - stadium_hp_delta(k.id, oldId);
      k.maxHp += d;
      k.hp += d;  // damage counters preserved
      if (k.hp < 0) k.hp = 0;
    };
    if (p.activeKnown) fix(p.active);
    for (auto& b : p.bench) fix(b);
  }
}

static void clear_appear_this_turn_flags(GameState& st) {
  for (int p = 0; p < 2; ++p) {
    if (st.players[p].activeKnown)
      st.players[p].active.appearThisTurn = false;
    for (auto& b : st.players[p].bench)
      b.appearThisTurn = false;
  }
}

static void clear_attack_end_turn_flags(GameState& st) {
  st.supporterPlayed = false;
  st.stadiumPlayed = false;
  st.energyAttached = false;
  st.retreated = false;
  clear_appear_this_turn_flags(st);
}

static void clear_stale_attack_damage_markers(GameState& st) {
  auto clear_one = [&st](InPlay& p) {
    // Effects such as Juggernaut Horn may inspect damage taken during the
    // opponent's immediately preceding turn. Keep that one-turn history, but
    // remove it before it can leak into a later turn.
    if (p.damagedByAttackTurn >= st.turn - 1) return;
    p.damagedByAttackTurn = -1;
    p.damagedByAttackSide = -1;
    p.damagedByAttackAmount = 0;
    p.damagedByAttackBeforeHp = -1;
  };
  for (Player& p : st.players) {
    if (p.activeKnown) clear_one(p.active);
    for (InPlay& b : p.bench) clear_one(b);
  }
}

// Start `player`'s turn: make them the actor, bump the turn, reset per-turn
// flags + appear-this-turn marks, and draw for the turn start.
static void start_turn(GameState& st, int player) {
  st.yourIndex = player;
  NativeLog start;
  start.type = 2;
  start.playerIndex = player;
  emit_log(st, start);
  clear_pending_ko_auras(st);
  st.turn += 1;
  clear_stale_attack_damage_markers(st);
  st.turnActionCount = 1;  // turn-start draw counts as the first action
  st.supporterPlayed = st.stadiumPlayed = st.energyAttached = st.retreated = false;
  st.teamRocketSupporterPlayed = false;
  st.ancientSupporterPlayed = false;
  st.canariPlayed = false;
  st.tarragonPlayed = false;
  st.fightingBuff = 0;
  st.lunarUsedThisTurn = false;
  st.stadiumAbilityUsed = false;
  for (int p = 0; p < 2; ++p) {
    if (st.players[p].activeKnown) {
      st.players[p].active.appearThisTurn = false;
      st.players[p].active.abilityUsedThisTurn = false;
      st.players[p].active.movedToActiveThisTurn = false;
      st.players[p].active.healedThisTurn = false;
    }
    for (auto& b : st.players[p].bench) {
      b.appearThisTurn = false;
      b.abilityUsedThisTurn = false;
      b.movedToActiveThisTurn = false;
      b.healedThisTurn = false;
    }
  }
  enforce_rotom_tool_limits(st);
  enforce_area_zero_bench_limits(st);
  Player& nx = st.players[player];
  if (nx.deckCount <= 0) {
    set_result(st, 1 - player, 2);  // deck-out: the drawing player loses
  } else {
    draw_n(st, nx, 1);  // real deck (free-running) or counts/unknown (replay)
  }
}

static void pokemon_checkup(GameState& st, int endingPlayer,
                            bool deferPoisonDamage = false);
static void checkup_resolve(GameState& st);
static void checkup_step(GameState& st);
static std::vector<std::pair<int, int>> checkup_damage_order_triggers(
    const GameState& st);
static void apply_poison_checkup_damage(GameState& st, int poisonedSide);
static void apply_one_freezing_shroud(GameState& st, int sourceSide = -1);
static void apply_one_team_rocket_tyranitar_checkup(GameState& st,
                                                    int sourceSide);
static bool set_froslass_checkup_pending(GameState& st,
    const std::vector<std::pair<int, int>>& sources);
static void finish_end_turn_after_optional_hooks(GameState& st, int endingPlayer);

static void apply_delayed_end_turn_damage(GameState& st, int endingPlayer) {
  Player& p = st.players[endingPlayer];
  auto apply = [&](InPlay& k) {
    if (k.delayedDamageTurn != st.turn || k.delayedDamageCounters <= 0) return;
    k.hp -= 10 * k.delayedDamageCounters;
    if (k.hp < 0) k.hp = 0;
    k.delayedDamageTurn = -1;
    k.delayedDamageCounters = 0;
  };
  auto applyKo = [&](InPlay& k) {
    if (k.delayedKoTurn != st.turn) return;
    k.hp = 0;
    // Keep the marker until KO resolution so prize scoring can distinguish a
    // delayed effect KO from a normal damage KO.
  };
  if (p.activeKnown) apply(p.active);
  if (p.activeKnown) applyKo(p.active);
  for (auto& b : p.bench) apply(b);
  for (auto& b : p.bench) applyKo(b);
}

static bool apply_delayed_end_turn_discards(GameState& st, int endingPlayer) {
  Player& p = st.players[endingPlayer];
  bool discarded = false;
  if (p.activeKnown && p.active.delayedKoTurn == st.turn &&
      p.active.delayedKoPromoteBeforePrize) {
    ko_active(p);
    discarded = true;
  }
  for (int j = static_cast<int>(p.bench.size()) - 1; j >= 0; --j) {
    if (p.bench[j].delayedKoTurn == st.turn &&
        p.bench[j].delayedKoPromoteBeforePrize) {
      ko_bench(p, j);
      discarded = true;
    }
  }
  return discarded;
}

static bool delayed_end_turn_would_ko(const GameState& st, int endingPlayer) {
  const Player& p = st.players[endingPlayer];
  auto would_ko = [&](const InPlay& k) {
    if (k.delayedKoTurn == st.turn)
      return true;
    if (k.delayedDamageTurn == st.turn && k.delayedDamageCounters > 0)
      return k.hp - 10 * k.delayedDamageCounters <= 0;
    return false;
  };
  if (p.activeKnown && would_ko(p.active))
    return true;
  for (const auto& b : p.bench)
    if (would_ko(b))
      return true;
  return false;
}

static bool has_zero_hp_pokemon(const GameState& st) {
  for (int side = 0; side < 2; ++side) {
    const Player& p = st.players[side];
    if (p.activeKnown && p.active.hp <= 0)
      return true;
    for (const auto& b : p.bench)
      if (b.hp <= 0)
        return true;
  }
  return false;
}

static void discard_ignition_energy_end_turn(GameState& st, int endingPlayer) {
  Player& p = st.players[endingPlayer];
  auto discard_from = [&st, &p, endingPlayer](InPlay& k) {
    bool changed = false;
    for (int i = static_cast<int>(k.energyCardIds.size()) - 1; i >= 0; --i) {
      if (k.energyCardIds[i] != 17) continue;
      p.discard.push_back(k.energyCardIds[i]);
      erase_attached_energy_card(k, i);
      InPlay cardsOnly = k;
      cardsOnly.energies.clear();
      k.energies = provided_energy_units(cardsOnly, &st, endingPlayer);
      changed = true;
    }
    if (changed) refresh_inplay_max_hp(st, k);
  };
  if (p.activeKnown) discard_from(p.active);
  for (auto& b : p.bench) discard_from(b);
}

static std::vector<int> powerglass_basic_energy_indices(const Player& p) {
  std::vector<int> indices;
  for (int i = 0; i < static_cast<int>(p.discard.size()); ++i) {
    const CardInfo* c = find_card(p.discard[i]);
    if (c && c->cardType == BASIC_ENERGY)
      indices.push_back(i);
  }
  return indices;
}

static std::vector<int> powerglass_sources(const GameState& st, const Player& p) {
  std::vector<int> sources;
  if (p.activeKnown && active_tool(st, p.active, 1163))
    sources.push_back(-1);
  for (int i = 0; i < static_cast<int>(p.bench.size()); ++i)
    if (active_tool(st, p.bench[i], 1163))
      sources.push_back(i);
  return sources;
}

static InPlay* powerglass_holder(Player& p, int source) {
  if (source < 0)
    return p.activeKnown ? &p.active : nullptr;
  if (source >= 0 && source < static_cast<int>(p.bench.size()))
    return &p.bench[source];
  return nullptr;
}

static void set_powerglass_energy_pending(GameState& st, int endingPlayer,
                                          int source,
                                          const std::vector<int>& indices) {
  Player& p = st.players[endingPlayer];
  PendingDecision pd;
  pd.context = 22;  // attach-to choice for a discard Basic Energy
  pd.minCount = 0;
  pd.maxCount = 1;
  for (int i : indices) {
    pd.options.push_back({Atom::S("CARD"), Atom::S("DISCARD"), Atom::I(i),
                          Atom::I(p.discard[i])});
  }
  st.yourIndex = endingPlayer;
  st.pending = pd;
  if (!st.effectStack.empty())
    st.effectStack.back().selfBench = source;
}

static bool set_powerglass_pending(GameState& st, int endingPlayer) {
  Player& p = st.players[endingPlayer];
  std::vector<int> sources = powerglass_sources(st, p);
  if (sources.empty()) return false;
  std::vector<int> indices = powerglass_basic_energy_indices(p);
  if (sources.size() == 1) {
    if (sources[0] >= 0 || indices.empty()) return false;
  }
  EffectFrame fr;
  fr.effect = EFF_POWERGLASS;
  fr.a = endingPlayer;
  fr.phase = sources.size() > 1 ? 0 : 1;
  fr.selfBench = sources[0];
  fr.scratch = sources;
  st.effectStack.push_back(fr);
  if (fr.phase == 0) {
    PendingDecision pd;
    pd.context = 34;  // CABT skill order before resolving Powerglass.
    pd.minCount = static_cast<int>(sources.size());
    pd.maxCount = static_cast<int>(sources.size());
    for (size_t i = 0; i < sources.size(); ++i)
      pd.options.push_back({Atom::S("SKILL"), Atom::I(1163), Atom::N()});
    st.yourIndex = endingPlayer;
    st.pending = pd;
  } else {
    set_powerglass_energy_pending(st, endingPlayer, sources[0], indices);
  }
  return true;
}

// A normal turn end runs the between-turns Pokemon Checkup (poison/burn damage,
// sleep/burn coin recovery, paralysis clear), then resolves any Checkup KOs
// (prizes/tie/promote -- possibly leaving a pending decision) and finally hands
// the turn to the other player (checkup_resolve handles the no-KO fast path).
static void end_turn(GameState& st) {
  int endingPlayer = st.yourIndex;
  NativeLog end;
  end.type = 3;
  end.playerIndex = endingPlayer;
  emit_log(st, end);
  if (st.discardHandEndTurn[endingPlayer] == st.turn) {
    Player& p = st.players[endingPlayer];
    if (p.handCount >= st.discardHandEndThreshold[endingPlayer])
      discard_hand_to_discard(p);
    st.discardHandEndTurn[endingPlayer] = -1;
    st.discardHandEndThreshold[endingPlayer] = 0;
  }
  if (set_powerglass_pending(st, endingPlayer)) return;
  finish_end_turn_after_optional_hooks(st, endingPlayer);
}

static void finish_end_turn_after_optional_hooks(GameState& st, int endingPlayer) {
  discard_ignition_energy_end_turn(st, endingPlayer);
  const Player& ending = st.players[endingPlayer];
  bool delayedKoPromoteActive =
      ending.activeKnown && ending.active.delayedKoTurn == st.turn &&
      ending.active.delayedKoPromoteBeforePrize;
  if (apply_delayed_end_turn_discards(st, endingPlayer)) {
    st.checkupNext = 1 - endingPlayer;
    EffectFrame fr;
    fr.effect = EFF_CHECKUP;
    fr.phase = 2;
    fr.a = 0;
    st.effectStack.push_back(fr);
    checkup_step(st);
    return;
  }
  apply_delayed_end_turn_damage(st, endingPlayer);
  if (has_zero_hp_pokemon(st)) {
    st.checkupNext = 1 - endingPlayer;
    st.checkupPromoteBeforePrize = delayedKoPromoteActive;
    checkup_resolve(st);
    return;
  }
  std::vector<std::pair<int, int>> checkupDamageTriggers =
      checkup_damage_order_triggers(st);
  bool hasSkillCheckupTrigger =
      std::any_of(checkupDamageTriggers.begin(), checkupDamageTriggers.end(),
                  [](const auto& source) { return source.first != 0; });
  bool orderCheckupDamage =
      hasSkillCheckupTrigger && checkupDamageTriggers.size() > 1;
  pokemon_checkup(st, endingPlayer, hasSkillCheckupTrigger);
  st.checkupNext = 1 - endingPlayer;
  if (checkupDamageTriggers.size() == 1) {
    if (checkupDamageTriggers[0].first == 104)
      apply_one_freezing_shroud(st, checkupDamageTriggers[0].second);
    else if (checkupDamageTriggers[0].first == 442)
      apply_one_team_rocket_tyranitar_checkup(st,
                                              checkupDamageTriggers[0].second);
  } else if (orderCheckupDamage &&
             set_froslass_checkup_pending(st, checkupDamageTriggers)) {
    return;
  }
  checkup_resolve(st);
}

static int prize_value(const CardInfo* c) {
  if (c && c->megaEx) return 3;  // Mega Evolution Pokemon ex
  if (c && c->ex) return 2;      // Pokemon ex
  return 1;
}

static int contextual_prize_value(GameState& st, int taker, int owner,
                                  const InPlay& knocked, bool activeKo) {
  int pv = prize_value(find_card(knocked.id));
  const Player& takerPlayer = st.players[taker];
  Player& ownerPlayer = st.players[owner];
  const InPlay* attacker = takerPlayer.activeKnown ? &takerPlayer.active : nullptr;
  const CardInfo* attackerCard = attacker ? find_card(attacker->id) : nullptr;
  const CardInfo* knockedCard = find_card(knocked.id);
  bool attackKo = st.lastAttackTurn[taker] == st.turn;
  bool attackDamageKo = knocked.damagedByAttackTurn == st.turn &&
                        knocked.damagedByAttackSide == taker;

  if (activeKo && in_play_has(takerPlayer, 214) && flip_heads(st))
    pv += 1;  // Togekiss: Wonder Kiss, non-stacking.
  if (attackKo && activeKo && attacker && attacker->id == 618 &&
      knockedCard && knockedCard->basic)
    pv += 1;  // Hydreigon ex.
  if (attackKo && activeKo && st.prizeBonusTurn[taker] == st.turn && attackerCard) {
    if (st.prizeBonusKind[taker] == PB_TERA_ATTACKER && attackerCard->tera)
      pv += st.prizeBonusAmount[taker];
    if (st.prizeBonusKind[taker] == PB_N_ATTACKER &&
        is_ns_pokemon(attackerCard))
      pv += st.prizeBonusAmount[taker];
  }

  int reduce = 0;
  if (attackDamageKo && !st.legacyEnergyPrizeUsed[owner]) {
    for (int id : knocked.energyCardIds) {
      if (id == 12) {
        reduce += 1;
        st.legacyEnergyPrizeUsed[owner] = true;
        break;
      }
    }
  }
  if (attackDamageKo && knocked.id == 139 && in_play_has(ownerPlayer, 141))
    reduce += 1;  // Munkidori ex + Pecharunt ex in play.
  if (active_tool(st, knocked, 1172) && knockedCard &&
      name_contains(knockedCard->name, "Lillie"))
    reduce += 1;  // Lillie's Pearl.
  if (attackKo && attackerCard && (attackerCard->ex || attackerCard->megaEx)) {
    if (knocked.id == 748)
      pv = 0;  // Shedinja: no Prize cards.
    if (in_play_has(ownerPlayer, 772) && knockedCard &&
        knockedCard->energyType == DARKNESS)
      reduce += 1;  // Mega Gengar ex: Shadowy Concealment, non-stacking.
  }
  return std::max(0, pv - reduce);
}

static int current_prize_take_count(const GameState& st, int attacker) {
  int count = 1;
  bool one_at_a_time = false;
  if (!st.effectStack.empty()) {
    const EffectFrame& fr = st.effectStack.back();
    if (fr.effect == EFF_PRIZE || fr.effect == EFF_ABILITY_KO) {
      count = fr.phase;
    } else if ((fr.effect == EFF_CHECKUP ||
                fr.effect == EFF_MAIN_ACTION_KO) &&
               fr.phase == 0) {
      count = fr.loopRemain;
      one_at_a_time = fr.topDeckSelectedCount > 1;
    }
  }
  count = std::min(std::max(0, count), st.players[attacker].prizeCount);
  return one_at_a_time && count > 0 ? 1 : count;
}

static int pending_prize_take_count(const GameState& st, int taker,
                                    int remaining) {
  int count = st.pending.maxCount > 0 ? st.pending.maxCount : 1;
  count = std::min(count, std::max(0, remaining));
  count = std::min(count, st.players[taker].prizeCount);
  return std::max(0, count);
}

// Present a prize-take decision: one option per (face-down) prize card.
static void set_prize_pending(GameState& st, int attacker) {
  st.yourIndex = attacker;
  PendingDecision pd;
  pd.context = 7;  // TO_HAND
  pd.minCount = current_prize_take_count(st, attacker);
  pd.maxCount = pd.minCount;
  const Player& p = st.players[attacker];
  for (int i = 0; i < p.prizeCount; ++i) {
    bool known = p.prizesKnown ||
                 (i < static_cast<int>(p.prizesKnownMask.size()) &&
                  p.prizesKnownMask[i]);
    int cid = (i < static_cast<int>(p.prizes.size())) ? p.prizes[i] : 0;
    pd.options.push_back({Atom::S("CARD"), Atom::S("PRIZE"), Atom::I(i),
                          known && cid > 0 ? Atom::I(cid) : Atom::N()});
  }
  st.pending = pd;
}

static int take_selected_prizes(GameState& st, int taker,
                                const std::vector<int>& sel, int requested) {
  Player& me = st.players[taker];
  int wanted = std::min(std::max(0, requested), me.prizeCount);
  if (wanted <= 0) return 0;

  std::vector<int> idxs;
  auto add_unique = [&](int idx) {
    if (idx < 0 || idx >= me.prizeCount) return;
    if (std::find(idxs.begin(), idxs.end(), idx) == idxs.end())
      idxs.push_back(idx);
  };
  for (int s : sel) {
    if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
    add_unique(static_cast<int>(st.pending.options[s][2].i));
  }
  for (int i = 0; static_cast<int>(idxs.size()) < wanted && i < me.prizeCount; ++i)
    add_unique(i);
  if (static_cast<int>(idxs.size()) > wanted)
    idxs.resize(wanted);

  std::sort(idxs.rbegin(), idxs.rend());
  int taken = 0;
  for (int idx : idxs) {
    if (idx < 0 || idx >= me.prizeCount) continue;
    me.prizeCount -= 1;
    record_prize_taken(st, taker);
    me.handCount += 1;
    if (!me.prizes.empty() && idx < static_cast<int>(me.prizes.size())) {
      int cid = me.prizes[idx];
      bool knownPrize = me.prizesKnown ||
                        (idx < static_cast<int>(me.prizesKnownMask.size()) &&
                         me.prizesKnownMask[idx]);
      if (!knownPrize) {
        auto kit = std::find(me.prizesKnownCards.begin(),
                             me.prizesKnownCards.end(), cid);
        if (kit != me.prizesKnownCards.end()) {
          knownPrize = true;
          me.prizesKnownCards.erase(kit);
        }
      }
      me.hand.push_back(cid);
      if (knownPrize && !me.handKnown) add_known_card(me.handKnownCards, cid);
      me.prizes.erase(me.prizes.begin() + idx);
      if (idx < static_cast<int>(me.prizesKnownMask.size()))
        me.prizesKnownMask.erase(me.prizesKnownMask.begin() + idx);
    } else {
      me.prizesKnown = false;
      me.prizesKnownMask.clear();
      me.handKnown = false;
    }
    if (idx < static_cast<int>(me.prizeFaceUp.size()))
      me.prizeFaceUp.erase(me.prizeFaceUp.begin() + idx);
    ++taken;
  }
  return taken;
}

static bool set_amulet_hope_pending(GameState& st, int owner, int taker,
                                    int prizeValue,
                                    bool returnToKoOrder = false) {
  Player& p = st.players[owner];
  if (p.deckCount <= 0) return false;
  if (static_cast<int>(p.deck.size()) == p.deckCount)
    mark_full_deck_inspected(p);
  PendingDecision pd;
  pd.context = 7;  // TO_HAND
  pd.minCount = 1;
  pd.maxCount = std::min(3, p.deckCount);
  int n = p.deck.empty() ? p.deckCount : static_cast<int>(p.deck.size());
  for (int i = 0; i < n; ++i) {
    Descriptor d = {Atom::S("CARD"), Atom::S("DECK"), Atom::I(i)};
    if (i < static_cast<int>(p.deck.size()))
      d.push_back(Atom::I(p.deck[i]));
    else
      d.push_back(Atom::N());
    pd.options.push_back(d);
  }
  if (pd.options.empty()) return false;
  EffectFrame fr;
  fr.effect = EFF_AMULET_HOPE;
  fr.a = taker;
  fr.phase = prizeValue;
  fr.savedSrc = owner;
  fr.loopRemain = returnToKoOrder ? 1 : 0;
  st.effectStack.push_back(fr);
  st.yourIndex = owner;
  st.pending = pd;
  return true;
}

static bool has_basic_water_energy_card(const InPlay& p) {
  for (int id : p.energyCardIds) {
    const CardInfo* c = find_card(id);
    if (c && c->cardType == BASIC_ENERGY && c->energyType == WATER)
      return true;
  }
  return false;
}

static int move_basic_water_energy_to_hand(GameState& st, int ownerIdx,
                                           InPlay& p) {
  Player& owner = st.players[ownerIdx];
  int moved = 0;
  for (int i = static_cast<int>(p.energyCardIds.size()) - 1; i >= 0; --i) {
    int id = p.energyCardIds[i];
    const CardInfo* c = find_card(id);
    if (!c || c->cardType != BASIC_ENERGY || c->energyType != WATER)
      continue;
    owner.hand.push_back(id);
    owner.handCount += 1;
    erase_attached_energy_card(p, i);
    if (i < static_cast<int>(p.energies.size()))
      p.energies.erase(p.energies.begin() + i);
    moved += 1;
  }
  return moved;
}

static bool huntail_trigger_available(const GameState& st, int owner,
                                      const InPlay& knocked) {
  return knocked.damagedByAttackTurn == st.turn &&
         knocked.damagedByAttackSide == 1 - owner &&
         pokemon_has_type(knocked.id, WATER) &&
         has_basic_water_energy_card(knocked) &&
         in_play_has(st.players[owner], 416);
}

static bool ko_hook_trigger_available(const GameState& st, int owner,
                                      const InPlay& knocked, int targetIdx) {
  if (on_ko_program(knocked.id) < 0) return false;
  if (knocked.damagedByAttackTurn != st.turn ||
      knocked.damagedByAttackSide != 1 - owner)
    return false;
  if (knocked.id == 824 && targetIdx >= 0)
    return false;  // Flygon triggers only from the Active Spot.
  return true;
}

static bool set_ko_hook_pending(GameState& st, int owner, int targetIdx) {
  Player& p = st.players[owner];
  InPlay* target = inplay_ref(p, targetIdx);
  if (!target || !ko_hook_trigger_available(st, owner, *target, targetIdx))
    return false;
  record_attack_damage_ko_marker(st, owner, *target);
  target->damagedByAttackTurn = -1;
  target->damagedByAttackSide = -1;

  EffectFrame resume;
  resume.effect = EFF_KO_RESUME;
  st.effectStack.push_back(resume);

  EffectFrame hook;
  hook.effect = FLOW_PROGRAM;
  hook.program = on_ko_program(target->id);
  hook.a = owner;
  hook.selfBench = targetIdx;
  hook.sourceCardId = target->id;
  st.yourIndex = owner;
  st.effectStack.push_back(hook);
  run_program(st, {});
  return true;
}

static bool amulet_hope_trigger_available(const GameState& st, int owner,
                                          const InPlay& knocked) {
  return active_tool(st, knocked, 1169) &&
         knocked.damagedByAttackTurn == st.turn &&
         knocked.damagedByAttackSide == 1 - owner &&
         st.players[owner].deckCount > 0;
}

static bool ko_skill_order_source_available(const GameState& st, int owner,
                                            const InPlay& knocked,
                                            int targetIdx) {
  if (ko_hook_trigger_available(st, owner, knocked, targetIdx))
    return true;
  return knocked.id == 139 &&
         knocked.damagedByAttackTurn == st.turn &&
         knocked.damagedByAttackSide == 1 - owner;
}

static void set_ko_trigger_order_pending(GameState& st, const EffectFrame& fr) {
  PendingDecision pd;
  pd.context = 34;  // SKILL order
  pd.minCount = 2;
  pd.maxCount = 2;
  pd.options.push_back({Atom::S("SKILL"), Atom::I(fr.sourceCardId), Atom::N()});
  pd.options.push_back({Atom::S("SKILL"), Atom::I(1169), Atom::N()});
  st.yourIndex = fr.a;
  st.pending = pd;
}

static bool set_ko_trigger_order_pending(GameState& st, int owner, int taker,
                                         int targetIdx, int prizeValue,
                                         int sourceCardId) {
  EffectFrame fr;
  fr.effect = EFF_KO_TRIGGER_ORDER;
  fr.a = taker;
  fr.savedSrc = owner;
  fr.selfBench = targetIdx;
  fr.phase = prizeValue;
  fr.sourceCardId = sourceCardId;
  fr.pc = -1;
  st.effectStack.push_back(fr);
  set_ko_trigger_order_pending(st, st.effectStack.back());
  return true;
}

static void finish_ko_trigger_order(GameState& st, const EffectFrame& fr) {
  int owner = fr.savedSrc;
  int taker = fr.a;
  Player& p = st.players[owner];
  if (fr.selfBench < 0) {
    if (p.activeKnown && p.active.hp <= 0) {
      InPlay knocked = p.active;
      int koId = knocked.id;
      record_attack_damage_ko_marker(st, owner, knocked);
      note_pending_ko_aura(st, owner, knocked);
      ko_active(p);
      record_ko(st, owner, koId);
    }
    st.yourIndex = taker;
    continue_active_ko_after_amulet(st, taker, owner, fr.phase);
    return;
  }

  checkup_resolve(st);
}

static void continue_ko_trigger_order(GameState& st) {
  if (st.effectStack.empty() ||
      st.effectStack.back().effect != EFF_KO_TRIGGER_ORDER)
    return;
  EffectFrame& fr = st.effectStack.back();
  if (fr.pc < 0)
    fr.pc = 0;
  if (fr.pc >= static_cast<int>(fr.scratch.size())) {
    EffectFrame done = fr;
    st.effectStack.pop_back();
    finish_ko_trigger_order(st, done);
    return;
  }

  int sourceCardId = fr.scratch[fr.pc++];
  int owner = fr.savedSrc;
  int taker = fr.a;
  int targetIdx = fr.selfBench;
  if (sourceCardId == 1169) {
    if (!set_amulet_hope_pending(st, owner, taker, fr.phase, true))
      continue_ko_trigger_order(st);
    return;
  }

  int prog = on_ko_program(sourceCardId);
  if (prog < 0) {
    continue_ko_trigger_order(st);
    return;
  }
  EffectFrame hook;
  hook.effect = FLOW_PROGRAM;
  hook.program = prog;
  hook.a = owner;
  hook.selfBench = targetIdx;
  hook.sourceCardId = sourceCardId;
  st.yourIndex = owner;
  st.effectStack.push_back(hook);
  run_program(st, {});
}

static bool set_huntail_pending(GameState& st, int owner, int taker,
                                int targetIdx, int prizeValue,
                                bool festivalFollowup = false) {
  Player& p = st.players[owner];
  InPlay* target = inplay_ref(p, targetIdx);
  if (!target || !huntail_trigger_available(st, owner, *target))
    return false;
  record_attack_damage_ko_marker(st, owner, *target);
  PendingDecision pd;
  pd.context = 43;  // ACTIVATE yes/no
  pd.minCount = 1;
  pd.maxCount = 1;
  pd.options.push_back({Atom::S("YES")});
  pd.options.push_back({Atom::S("NO")});
  EffectFrame fr;
  fr.effect = EFF_HUNTAIL;
  fr.a = taker;
  fr.phase = prizeValue;
  fr.savedSrc = owner;
  fr.selfBench = targetIdx;
  fr.loopRemain = festivalFollowup ? 1 : 0;
  st.effectStack.push_back(fr);
  st.yourIndex = owner;
  st.pending = pd;
  return true;
}

static bool set_heavy_baton_pending(GameState& st, int owner, int taker,
                                    int prizeValue,
                                    bool festivalFollowup = false) {
  Player& p = st.players[owner];
  if (!p.activeKnown || p.bench.empty() || !active_tool(st, p.active, 1160))
    return false;
  if (p.active.damagedByAttackTurn != st.turn ||
      p.active.damagedByAttackSide != taker)
    return false;
  if (retreat_cost(st, owner) != 4) return false;
  record_attack_damage_ko_marker(st, owner, p.active);
  PendingDecision pd;
  pd.context = 17;  // optional attached Energy choice
  pd.minCount = 0;
  pd.maxCount = 0;
  for (int i = 0; i < static_cast<int>(p.active.energyCardIds.size()); ++i) {
    int etype = -1;
    int id = p.active.energyCardIds[i];
    if (!basic_energy_type(id, etype)) continue;
    pd.options.push_back({Atom::S("ENERGY"), Atom::S("ACTIVE"),
                          Atom::I(encode_energy_ref(-1, i)), Atom::I(id)});
    pd.maxCount = std::min(3, pd.maxCount + 1);
  }
  if (pd.options.empty() || pd.maxCount <= 0) return false;
  EffectFrame fr;
  fr.effect = EFF_HEAVY_BATON;
  fr.a = taker;
  fr.phase = prizeValue;
  fr.savedSrc = owner;
  fr.selfBench = -1;
  fr.loopRemain = festivalFollowup ? 1 : 0;
  fr.topDeckSelectedCount = 0;
  st.effectStack.push_back(fr);
  st.yourIndex = owner;
  st.pending = pd;
  return true;
}

static void note_pending_ko_aura(GameState& st, int owner, const InPlay& knocked) {
  if (owner >= 0 && owner < 2 && knocked.id == 710)
    st.pendingMeganiumAura[owner] = true;
}

static void clear_pending_ko_auras(GameState& st) {
  st.pendingMeganiumAura[0] = false;
  st.pendingMeganiumAura[1] = false;
}

// Move a KO'd Active (and its attached cards) to the discard pile and empty the
// Active Spot; its special conditions clear with it.
static void ko_active(Player& p) {
  p.discard.push_back(p.active.id);
  for (int x : p.active.preEvo) p.discard.push_back(x);
  for (int x : p.active.energyCardIds) p.discard.push_back(x);
  for (int x : p.active.tools) p.discard.push_back(x);
  p.active = InPlay();
  p.activePresent = false;
  p.activeKnown = false;
  clear_status(p);
}

// Move a KO'd Benched Pokemon (and its attached cards) to the discard pile and
// remove it from the Bench. No promotion follows (the Active Spot is unaffected).
static void ko_bench(Player& p, int j) {
  InPlay& b = p.bench[j];
  p.discard.push_back(b.id);
  for (int x : b.preEvo) p.discard.push_back(x);
  for (int x : b.energyCardIds) p.discard.push_back(x);
  for (int x : b.tools) p.discard.push_back(x);
  p.bench.erase(p.bench.begin() + j);
}

// Between-turns Pokemon Checkup. Per TCG the turn player's Active is processed
// first. Poison deals 10; Burn deals 20 then a coin (heads removes Burn); Asleep
// flips (heads wakes). Paralysis on the player whose turn just ended is removed.
static int poison_checkup_counters(const GameState& st, int poisonedSide) {
  const Player& poisoned = st.players[poisonedSide];
  if (!poisoned.activeKnown) return 0;
  int counters = std::max(1, poisoned.poisonDamageCounters);
  const Player& opp = st.players[1 - poisonedSide];
  if (opp.activeKnown && opp.active.id == 230)
    counters += 5;  // Pecharunt: five more Poison counters on opponent.
  if (cur_stadium(st) == 1243 &&
      !pokemon_has_type(poisoned.active.id, DARKNESS))
    counters += 2;  // Perilous Jungle: two more on non-Dark Pokemon.
  return counters;
}

static void apply_one_team_rocket_tyranitar_checkup(GameState& st, int sourceSide) {
  if (sourceSide < 0 || sourceSide >= 2) return;
  const Player& source = st.players[sourceSide];
  if (!source.activeKnown || source.active.id != 442 ||
      ability_suppressed(st, sourceSide, source.active, true))
    return;
  auto place_on_basic = [](InPlay& pk) {
    const CardInfo* c = find_card(pk.id);
    if (!c || c->cardType != POKEMON || !c->basic) return;
    pk.hp -= 20;
    if (pk.hp < 0) pk.hp = 0;
  };
  Player& opp = st.players[1 - sourceSide];
  if (opp.activeKnown) place_on_basic(opp.active);
  for (auto& b : opp.bench) place_on_basic(b);
}

static void apply_team_rocket_tyranitar_checkup(GameState& st) {
  for (int side = 0; side < 2; ++side) {
    apply_one_team_rocket_tyranitar_checkup(st, side);
  }
}

static std::vector<std::pair<int, int>> checkup_damage_order_triggers(
    const GameState& st) {
  std::vector<std::pair<int, int>> sources;
  for (int s = 0; s < 2; ++s) {
    const Player& p = st.players[s];
    if (p.activeKnown && p.active.id == 104) sources.push_back({104, s});
    for (const auto& b : p.bench)
      if (b.id == 104) sources.push_back({104, s});
  }
  for (int s = 0; s < 2; ++s) {
    const Player& p = st.players[s];
    if (p.activeKnown && p.active.id == 442 &&
        !ability_suppressed(st, s, p.active, true))
      sources.push_back({442, s});
  }
  for (int s = 0; s < 2; ++s) {
    const Player& p = st.players[s];
    if (p.activeKnown && p.poisoned)
      sources.push_back({0, s});
  }
  return sources;
}

static void apply_poison_checkup_damage(GameState& st, int poisonedSide) {
  Player& p = st.players[poisonedSide];
  if (!p.activeKnown || !p.poisoned) return;
  p.active.hp -= 10 * poison_checkup_counters(st, poisonedSide);
  if (p.active.hp < 0) p.active.hp = 0;
}

static void apply_one_freezing_shroud(GameState& st, int sourceSide) {
  int firstKoSide = -1;
  auto place_on_ability_pokemon = [&st, &firstKoSide](int side, InPlay& k,
                                        bool isActive) {
    const CardInfo* c = find_card(k.id);
    if (!c || !c->hasAbility || ability_suppressed(st, side, k, isActive) ||
        name_contains(c->name, "Froslass"))
      return;
    int before = k.hp;
    k.hp -= 10;
    if (k.hp < 0) k.hp = 0;
    if (before > 0 && k.hp <= 0 && firstKoSide < 0)
      firstKoSide = side;
  };
  for (int s = 0; s < 2; ++s) {
    Player& p = st.players[s];
    if (p.activeKnown) place_on_ability_pokemon(s, p.active, true);
    for (auto& b : p.bench) place_on_ability_pokemon(s, b, false);
  }
  if (sourceSide >= 0 && sourceSide < 2 && firstKoSide >= 0 &&
      st.checkupKoFirst < 0)
    st.checkupKoFirst = firstKoSide;
}

static bool set_froslass_checkup_pending(GameState& st,
    const std::vector<std::pair<int, int>>& sources) {
  if (sources.size() < 2) return false;
  clear_attack_end_turn_flags(st);
  EffectFrame fr;
  fr.effect = EFF_FROSLASS_CHECKUP;
  fr.phase = static_cast<int>(sources.size());
  fr.a = st.checkupNext;
  for (const auto& source : sources) {
    fr.scratch.push_back(source.first);
    fr.scratch.push_back(source.second);
  }
  st.effectStack.push_back(fr);

  PendingDecision pd;
  pd.context = 34;  // SKILL order
  pd.minCount = static_cast<int>(sources.size());
  pd.maxCount = static_cast<int>(sources.size());
  for (const auto& source : sources)
    pd.options.push_back({Atom::S("SKILL"), Atom::I(source.first), Atom::N()});
  st.yourIndex = st.checkupNext;
  st.pending = pd;
  return true;
}

static void pokemon_checkup(GameState& st, int endingPlayer,
                            bool deferPoisonDamage) {
  // cabt resolves the Checkup in phases across BOTH Active Pokemon, in game
  // first-player order: status-immunity clears + Poison damage, then every Burn
  // (damage + cure coin), then every Sleep (wake coin), then Paralysis. Keeping
  // the coin flips phase-separated and first-player-ordered matches the order
  // cabt consumes coins, so replayed coin tapes stay aligned.
  int first = st.firstPlayer >= 0 ? st.firstPlayer : endingPlayer;
  int order[2] = {first, 1 - first};
  for (int who : order) {
    Player& p = st.players[who];
    if (!p.activeKnown) continue;
    if (is_antique_fossil(p.active.id))
      clear_status(p);
    if (festival_status_immunity(st, who))
      clear_status(p);
    if (p.poisoned && !deferPoisonDamage)
      apply_poison_checkup_damage(st, who);
    if (p.active.hp < 0) p.active.hp = 0;
  }
  for (int who : order) {  // Burn: 20 (+extra) damage, then a cure coin
    Player& p = st.players[who];
    if (!p.activeKnown || !p.burned) continue;
    int extraCounters = 3 * in_play_count_id(st.players[1 - who], 256);
    p.active.hp -= 20 + 10 * extraCounters;
    if (p.active.hp < 0) p.active.hp = 0;
    if (flip_heads(st)) p.burned = false;  // heads cures Burn
  }
  for (int who : order) {  // Sleep: a wake coin
    Player& p = st.players[who];
    if (!p.activeKnown || !p.asleep) continue;
    if (flip_heads(st)) p.asleep = false;  // heads wakes up
  }
  st.players[endingPlayer].paralyzed = false;  // clears at its owner's Checkup
  if (!deferPoisonDamage)
    apply_team_rocket_tyranitar_checkup(st);
}

// Build a "promote a new Active" decision for `who` (its Active was KO'd).
static void set_promote_pending(GameState& st, int who) {
  st.yourIndex = who;
  PendingDecision pd;
  pd.context = 4;  // TO_ACTIVE
  Player& d = st.players[who];
  for (int j = 0; j < static_cast<int>(d.bench.size()); ++j)
    pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                          Atom::I(d.bench[j].id)});
  st.pending = pd;
}

// After ALL Checkup prizes are taken, score both win conditions independently.
// A player satisfying both conditions beats an opponent satisfying only one;
// equal positive scores are a draw in the competition API.
// Returns true iff the game is over.
static bool checkup_endgame(GameState& st) {
  bool aOut = !st.players[0].activeKnown && st.players[0].bench.empty();
  bool bOut = !st.players[1].activeKnown && st.players[1].bench.empty();
  int aScore = (st.players[0].prizeCount == 0 ? 1 : 0) + (bOut ? 1 : 0);
  int bScore = (st.players[1].prizeCount == 0 ? 1 : 0) + (aOut ? 1 : 0);
  if (aScore == 0 && bScore == 0) return false;
  set_result(st, aScore == bScore ? 2 : (aScore > bScore ? 0 : 1));
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  st.checkupNext = -1;
  return true;
}

// Drive the Checkup/post-attack KO frame. Stage 0 takes Prizes for every KO'd
// Pokemon (Active OR Bench) and DEFERS the win/tie check until they are ALL taken
// (so a simultaneous double-KO ties when both players would win); stage 1 promotes
// any side whose Active Spot is now empty (Bench KOs need no promotion); stage 2
// is a narrow delayed-KO mode that promotes first, then returns to stage 0.
// Suspends whenever it needs input.
// Frame: phase = stage; a = prize-pair cursor (stage 0) / side cursor (stage 1);
// loopRemain = Prizes left in the current take; scratch = flattened
// [takerSide, prizeValue, ...] (one pair per KO'd Pokemon); savedScratch =
// one grouping id per KO pair.
static void checkup_step(GameState& st) {
  EffectFrame& fr = st.effectStack.back();
  int n = static_cast<int>(fr.scratch.size()) / 2;
  auto group_id = [&fr](int idx) {
    return (idx >= 0 && idx < static_cast<int>(fr.savedScratch.size()))
               ? fr.savedScratch[idx]
               : idx;
  };
  if (fr.phase == 2) {  // ---- delayed-KO promotion before prizes ----
    while (fr.a < 2) {
      int side = (fr.a == 0) ? st.checkupNext : (1 - st.checkupNext);
      if (!st.players[side].activeKnown && !st.players[side].bench.empty()) {
        set_promote_pending(st, side);
        return;
      }
      fr.a += 1;
    }
    fr.phase = 0;
    fr.a = 0;
  }
  if (fr.phase == 0) {  // ---- prize-taking ----
    while (fr.a < n) {
      int taker = fr.scratch[fr.a * 2];
      int group = group_id(fr.a);
      int pv = 0;
      int next = fr.a;
      while (next < n && fr.scratch[next * 2] == taker &&
             group_id(next) == group) {
        pv += fr.scratch[next * 2 + 1];
        next += 1;
      }
      if (pv > 0 && st.players[taker].prizeCount > 0) {
        fr.loopRemain = pv;
        fr.topDeckSelectedCount = next - fr.a;
        fr.savedSrc = next;
        set_prize_pending(st, taker);
        st.yourIndex = taker;
        return;  // suspend: the taker picks a Prize
      }
      fr.a = next;
    }
    fr.savedSrc = -2;
    clear_pending_ko_auras(st);
    if (checkup_endgame(st)) return;  // game over (wins / ties)
    fr.phase = 1;
    fr.a = 0;
  }
  while (fr.a < 2) {  // ---- promotion (next player chooses first, per the rules) ----
    int side = (fr.a == 0) ? st.checkupNext : (1 - st.checkupNext);
    if (!st.players[side].activeKnown && !st.players[side].bench.empty()) {
      set_promote_pending(st, side);
      return;  // suspend: that side promotes
    }
    fr.a += 1;
  }
  int effect = fr.effect;
  int resumePlayer = fr.savedPhase;
  int nxt = st.checkupNext;
  st.checkupNext = -1;
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  if (effect == EFF_MAIN_ACTION_KO) {
    if (resumePlayer >= 0 && resumePlayer < 2)
      st.yourIndex = resumePlayer;
    return;
  }
  start_turn(st, nxt);
}

// Entry from end_turn / post_attack: KO every Pokemon (Active or Bench) the
// preceding damage brought to 0 HP, awarding each KO's Prize to the OPPONENT of
// its owner; then resolve prizes/tie/promotion. No KO -> just start the next turn.
static void checkup_resolve(GameState& st) {
  int first = st.checkupKoFirst >= 0 ? st.checkupKoFirst : 1 - st.checkupNext;
  st.checkupKoFirst = -1;
  bool promoteBeforePrize = st.checkupPromoteBeforePrize;
  st.checkupPromoteBeforePrize = false;
  int order[2] = {first, 1 - first};
  for (int who : order) {
    Player& p = st.players[who];
    int taker = 1 - who;
    for (int j = 0; j < static_cast<int>(p.bench.size()); ++j) {
      if (p.bench[j].hp <= 0 && set_ko_hook_pending(st, who, j))
        return;
      if (p.bench[j].hp <= 0 &&
          set_huntail_pending(st, who, taker, j, -1))
        return;
    }
    if (p.activeKnown && p.active.hp <= 0 && set_ko_hook_pending(st, who, -1))
      return;
    if (p.activeKnown && p.active.hp <= 0 &&
        set_heavy_baton_pending(st, who, taker, -1))
      return;
    if (p.activeKnown && p.active.hp <= 0 &&
        set_huntail_pending(st, who, taker, -1, -1))
      return;
    if (p.activeKnown && p.active.hp <= 0 &&
        active_tool(st, p.active, 1169) &&
        p.active.damagedByAttackTurn == st.turn) {
      p.active.damagedByAttackTurn = -1;
      if (set_amulet_hope_pending(st, who, taker, -1))
        return;
    }
    for (auto& b : p.bench) {
      if (b.hp <= 0 && active_tool(st, b, 1169) && b.damagedByAttackTurn == st.turn) {
        b.damagedByAttackTurn = -1;
        if (set_amulet_hope_pending(st, who, taker, -1))
          return;
      }
    }
  }
  std::vector<int> koPairs;   // [takerSide, prizeValue, ...]
  std::vector<int> koGroups;  // same id means one combined Prize decision
  int nextKoGroup = 0;
  int attackDamageGroup[2] = {-1, -1};
  auto push_ko = [&](int taker, int owner, const InPlay& knocked,
                     bool activeKo) {
    bool attackDamageKo = knocked.damagedByAttackTurn == st.turn &&
                          knocked.damagedByAttackSide == taker;
    int group = -1;
    if (attackDamageKo && !activeKo) {
      if (attackDamageGroup[owner] < 0)
        attackDamageGroup[owner] = nextKoGroup++;
      group = attackDamageGroup[owner];
    } else {
      group = nextKoGroup++;
    }
    koPairs.push_back(taker);
    koPairs.push_back(contextual_prize_value(st, taker, owner, knocked,
                                             activeKo));
    koGroups.push_back(group);
  };
  for (int who : order) {
    Player& p = st.players[who];
    int taker = 1 - who;
    if (p.activeKnown && p.active.hp <= 0) {  // Active KO
      InPlay knocked = p.active;
      int koId = knocked.id;
      push_ko(taker, who, knocked, true);
      record_attack_damage_ko_marker(st, who, knocked);
      note_pending_ko_aura(st, who, knocked);
      ko_active(p);
      record_ko(st, who, koId);
    }
    for (int j = static_cast<int>(p.bench.size()) - 1; j >= 0; --j)  // Bench KOs
      if (p.bench[j].hp <= 0) {
        InPlay knocked = p.bench[j];
        int koId = knocked.id;
        push_ko(taker, who, knocked, false);
        record_attack_damage_ko_marker(st, who, knocked);
        ko_bench(p, j);
        record_ko(st, who, koId);
      }
  }
  if (koPairs.empty()) {
    int nxt = st.checkupNext;
    st.checkupNext = -1;
    st.checkupKoFirst = -1;
    start_turn(st, nxt);
    return;
  }
  EffectFrame fr;
  fr.effect = EFF_CHECKUP;
  fr.phase = promoteBeforePrize ? 2 : 0;
  fr.a = 0;
  fr.scratch = koPairs;
  fr.savedScratch = koGroups;
  st.effectStack.push_back(fr);
  checkup_step(st);
}

// Main-action passives such as Gravity Mountain can leave Pokemon at 0 HP
// without ending the turn. Resolve those KOs through the same prize/promotion
// flow as Checkup, then restore the acting player.
static bool resolve_main_action_zero_hp(GameState& st, int resumePlayer,
                                        int firstKoSide) {
  int first = (firstKoSide >= 0 && firstKoSide < 2) ? firstKoSide : resumePlayer;
  int order[2] = {first, 1 - first};
  std::vector<int> koPairs;   // [takerSide, prizeValue, ...]
  std::vector<int> koGroups;  // same id means one combined Prize decision
  int nextKoGroup = 0;
  int attackDamageGroup[2] = {-1, -1};
  auto push_ko = [&](int taker, int owner, const InPlay& knocked,
                     bool activeKo) {
    bool attackDamageKo = knocked.damagedByAttackTurn == st.turn &&
                          knocked.damagedByAttackSide == taker;
    int group = -1;
    if (attackDamageKo && !activeKo) {
      if (attackDamageGroup[owner] < 0)
        attackDamageGroup[owner] = nextKoGroup++;
      group = attackDamageGroup[owner];
    } else {
      group = nextKoGroup++;
    }
    koPairs.push_back(taker);
    koPairs.push_back(contextual_prize_value(st, taker, owner, knocked,
                                             activeKo));
    koGroups.push_back(group);
  };
  for (int who : order) {
    Player& p = st.players[who];
    int taker = 1 - who;
    if (p.activeKnown && p.active.hp <= 0) {
      InPlay knocked = p.active;
      int koId = knocked.id;
      push_ko(taker, who, knocked, true);
      note_pending_ko_aura(st, who, knocked);
      ko_active(p);
      record_ko(st, who, koId);
    }
    for (int j = static_cast<int>(p.bench.size()) - 1; j >= 0; --j) {
      if (p.bench[j].hp > 0) continue;
      InPlay knocked = p.bench[j];
      int koId = knocked.id;
      push_ko(taker, who, knocked, false);
      ko_bench(p, j);
      record_ko(st, who, koId);
    }
  }
  if (koPairs.empty()) return false;

  EffectFrame fr;
  fr.effect = EFF_MAIN_ACTION_KO;
  fr.phase = 0;
  fr.a = 0;
  fr.savedPhase = resumePlayer;
  fr.scratch = koPairs;
  fr.savedScratch = koGroups;
  st.effectStack.push_back(fr);
  st.checkupNext = resumePlayer;
  checkup_step(st);
  return true;
}

// Resolve a pending Checkup decision: a Prize pick (stage 0) or a promotion
// (stage 1), then advance the state machine.
static void resolve_checkup(GameState& st, const std::vector<int>& sel) {
  EffectFrame& fr = st.effectStack.back();
  if (fr.phase == 0) {  // a Prize was taken by st.yourIndex
    Player& me = st.players[st.yourIndex];
    int taken = take_selected_prizes(
        st, st.yourIndex, sel,
        pending_prize_take_count(st, st.yourIndex, fr.loopRemain));
    st.turnActionCount += 1;
    fr.loopRemain -= taken;
    if (fr.loopRemain <= 0) clear_pending_ko_auras(st);
    if (fr.loopRemain > 0 && me.prizeCount > 0) {
      set_prize_pending(st, st.yourIndex);  // more Prizes for this KO
      return;
    }
    fr.topDeckSelectedCount = 0;
    fr.a = (fr.savedSrc > fr.a) ? fr.savedSrc : fr.a + 1;
    fr.savedSrc = -2;
    st.pending = PendingDecision();
    checkup_step(st);
  } else {  // a promotion choice
    int koSide = st.yourIndex;
    if (!sel.empty()) {
      int benchIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
      Player& d = st.players[koSide];
      d.active = d.bench[benchIdx];
      d.bench.erase(d.bench.begin() + benchIdx);
      d.activePresent = true;
      d.activeKnown = true;
      d.active.movedToActiveThisTurn = true;
      clear_active_spot_locks(d.active);
      clear_status(d);
    }
    fr.a += 1;
    st.pending = PendingDecision();
    checkup_step(st);
  }
}

static void resolve_damage_trigger_order(GameState& st,
                                         const std::vector<int>& sel,
                                         const std::vector<int>& tape) {
  EffectFrame fr = st.effectStack.back();
  int n = static_cast<int>(fr.scratch.size()) / 4;
  std::vector<int> order;
  auto add_unique = [&](int idx) {
    if (idx < 0 || idx >= n) return;
    if (std::find(order.begin(), order.end(), idx) == order.end())
      order.push_back(idx);
  };
  for (int s : sel) add_unique(s);
  for (int i = 0; static_cast<int>(order.size()) < n && i < n; ++i)
    add_unique(i);
  if (static_cast<int>(order.size()) > n) order.resize(n);

  for (int idx : order) {
    int base = idx * 4;
    queue_deferred_program_frame(st, fr.scratch[base], fr.scratch[base + 1],
                                 fr.scratch[base + 2], fr.scratch[base + 3]);
  }
  st.turnActionCount += 1;
  st.pending = PendingDecision();
  st.effectStack.pop_back();
  if (!st.afterProgramQueue.empty()) {
    for (auto it = st.afterProgramQueue.rbegin();
         it != st.afterProgramQueue.rend(); ++it)
      st.effectStack.push_back(*it);
    st.afterProgramQueue.clear();
  }
  if (!st.effectStack.empty() && st.effectStack.back().effect == FLOW_PROGRAM)
    run_program(st, tape);
}

static void resolve_evolve_trigger_order(GameState& st,
                                         const std::vector<int>& sel,
                                         const std::vector<int>& tape) {
  EffectFrame fr = st.effectStack.back();
  int n = 2;
  std::vector<int> order;
  auto add_unique = [&](int idx) {
    if (idx < 0 || idx >= n) return;
    if (std::find(order.begin(), order.end(), idx) == order.end())
      order.push_back(idx);
  };
  for (int s : sel) add_unique(s);
  for (int i = 0; static_cast<int>(order.size()) < n && i < n; ++i)
    add_unique(i);

  std::vector<EffectFrame> frames;
  auto add_program = [&](int prog) {
    if (prog < 0) return;
    EffectFrame next;
    next.effect = FLOW_PROGRAM;
    next.program = prog;
    next.a = fr.a;
    next.selfBench = fr.selfBench;
    next.sourceCardId = fr.sourceCardId;
    frames.push_back(next);
  };
  for (int idx : order) {
    if (idx == 0) {
      EffectFrame next;
      next.effect = EFF_DARKEST_IMPULSE;
      next.a = fr.a;
      next.selfBench = fr.selfBench;
      next.sourceCardId = 428;
      frames.push_back(next);
    } else {
      add_program(card_program(fr.sourceCardId));
      add_program(on_evolve_program(fr.sourceCardId));
    }
  }

  st.turnActionCount += 1;
  st.pending = PendingDecision();
  st.effectStack.pop_back();
  for (auto it = frames.rbegin(); it != frames.rend(); ++it)
    st.effectStack.push_back(*it);
  advance_effect_stack(st, tape);
  if (!st.has_pending() && st.effectStack.empty())
    resolve_main_action_zero_hp(st, fr.a, fr.a);
}

static void resolve_on_play_basic_trigger_order(GameState& st,
                                                const std::vector<int>& sel,
                                                const std::vector<int>& tape) {
  EffectFrame fr = st.effectStack.back();
  int n = 2;
  std::vector<int> order;
  auto add_unique = [&](int idx) {
    if (idx < 0 || idx >= n) return;
    if (std::find(order.begin(), order.end(), idx) == order.end())
      order.push_back(idx);
  };
  for (int s : sel) add_unique(s);
  for (int i = 0; static_cast<int>(order.size()) < n && i < n; ++i)
    add_unique(i);

  std::vector<EffectFrame> frames;
  for (int idx : order) {
    EffectFrame next;
    if (idx == 0) {
      next.effect = EFF_RISKY_RUINS_BENCH_ENTRY;
      next.a = fr.a;
      next.selfBench = fr.selfBench;
    } else {
      int prog = on_play_program(fr.sourceCardId);
      if (prog < 0) continue;
      next.effect = FLOW_PROGRAM;
      next.program = prog;
      next.a = fr.a;
      next.selfBench = fr.selfBench;
      next.sourceCardId = fr.sourceCardId;
    }
    frames.push_back(next);
  }

  st.turnActionCount += 1;
  st.pending = PendingDecision();
  st.effectStack.pop_back();
  for (auto it = frames.rbegin(); it != frames.rend(); ++it)
    st.effectStack.push_back(*it);
  advance_effect_stack(st, tape);
}

static void resolve_on_attach_trigger_order(GameState& st,
                                            const std::vector<int>& sel,
                                            const std::vector<int>& tape) {
  EffectFrame fr = st.effectStack.back();
  int n = 2;
  std::vector<int> order;
  auto add_unique = [&](int idx) {
    if (idx < 0 || idx >= n) return;
    if (std::find(order.begin(), order.end(), idx) == order.end())
      order.push_back(idx);
  };
  for (int s : sel) add_unique(s);
  for (int i = 0; static_cast<int>(order.size()) < n && i < n; ++i)
    add_unique(i);

  std::vector<EffectFrame> frames;
  auto queue_draw = [&]() {
    int prog = on_attach_program(fr.sourceCardId);
    if (prog < 0) return;
    EffectFrame next;
    next.effect = FLOW_PROGRAM;
    next.program = prog;
    next.a = fr.a;
    next.selfBench = fr.selfBench;
    next.sourceCardId = fr.sourceCardId;
    frames.push_back(next);
  };
  for (int idx : order) {
    if (idx == 0) {
      Player& p = st.players[fr.a];
      InPlay* target = inplay_ref(p, fr.selfBench);
      if (target) {
        int before = target->hp;
        target->hp = std::min(target->maxHp, target->hp + 90);
        if (target->hp > before) target->healedThisTurn = true;
      }
    } else {
      queue_draw();
    }
  }

  st.turnActionCount += 1;
  st.pending = PendingDecision();
  st.effectStack.pop_back();
  for (auto it = frames.rbegin(); it != frames.rend(); ++it)
    st.effectStack.push_back(*it);
  advance_effect_stack(st, tape);
  if (!st.has_pending() && st.effectStack.empty())
    resolve_main_action_zero_hp(st, fr.a, fr.a);
}

static void resolve_ko_trigger_order(GameState& st,
                                     const std::vector<int>& sel) {
  EffectFrame& fr = st.effectStack.back();
  std::vector<int> order;
  auto add_unique = [&](int idx) {
    if (idx < 0 || idx >= 2) return;
    if (std::find(order.begin(), order.end(), idx) == order.end())
      order.push_back(idx);
  };
  for (int s : sel) add_unique(s);
  for (int i = 0; static_cast<int>(order.size()) < 2 && i < 2; ++i)
    add_unique(i);

  fr.scratch.clear();
  for (int idx : order)
    fr.scratch.push_back(idx == 0 ? fr.sourceCardId : 1169);
  fr.pc = 0;
  st.turnActionCount += 1;
  st.pending = PendingDecision();
  continue_ko_trigger_order(st);
}

static void resolve_froslass_checkup(GameState& st,
                                     const std::vector<int>& sel) {
  EffectFrame fr = st.effectStack.back();
  int n = static_cast<int>(fr.scratch.size()) / 2;
  std::vector<int> order;
  auto add_unique = [&](int idx) {
    if (idx < 0 || idx >= n) return;
    if (std::find(order.begin(), order.end(), idx) == order.end())
      order.push_back(idx);
  };
  for (int s : sel) add_unique(s);
  for (int i = 0; static_cast<int>(order.size()) < n && i < n; ++i)
    add_unique(i);

  for (int idx : order) {
    int sourceCardId = fr.scratch[idx * 2];
    int side = fr.scratch[idx * 2 + 1];
    if (sourceCardId == 104)
      apply_one_freezing_shroud(st, side);
    else if (sourceCardId == 442)
      apply_one_team_rocket_tyranitar_checkup(st, side);
    else if (sourceCardId == 0)
      apply_poison_checkup_damage(st, side);
  }
  st.turnActionCount += 1;
  st.pending = PendingDecision();
  st.effectStack.pop_back();
  checkup_resolve(st);
}

// Present an energy-discard decision: one option per energy unit on the Active.
static void set_discard_energy_pending(GameState& st) {
  Player& me = st.players[st.yourIndex];
  PendingDecision pd;
  pd.context = 30;  // DISCARD_ENERGY
  int cardBackedUnits = 0;
  for (int k = 0; k < static_cast<int>(me.active.energyCardIds.size()); ++k) {
    int cardId = me.active.energyCardIds[k];
    cardBackedUnits += std::max(
        1, provided_energy_units_for_card(me.active, k, &st, st.yourIndex));
    pd.options.push_back({Atom::S("ENERGY"), Atom::S("ACTIVE"), Atom::I(k),
                          Atom::I(cardId)});
  }
  int legacyUnits = me.active.energyCardIds.empty()
                        ? static_cast<int>(me.active.energies.size())
                        : std::max(0, static_cast<int>(me.active.energies.size()) -
                                          cardBackedUnits);
  if (legacyUnits > 0) {
    int idx = static_cast<int>(me.active.energyCardIds.size());
    pd.options.push_back({Atom::S("ENERGY"), Atom::S("ACTIVE"), Atom::I(idx),
                          Atom::I(0)});
  }
  st.pending = pd;
}

// Present the retreat's switch decision (choose a Benched Pokemon).
static void set_own_switch_pending(GameState& st) {
  Player& me = st.players[st.yourIndex];
  PendingDecision pd;
  pd.context = 3;  // SWITCH
  for (int j = 0; j < static_cast<int>(me.bench.size()); ++j)
    pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                          Atom::I(me.bench[j].id)});
  st.pending = pd;
}

// --- effects (decision-flow) ----------------------------------------------

static void begin_program(GameState& st, int prog, int actor,
                          const std::vector<int>& tape, int selfBench = -1,
                          int sourceCardId = 0) {
  if (prog < 0) return;
  st.lastEffectCount = 0;
  EffectFrame fr;
  fr.effect = FLOW_PROGRAM;
  fr.program = prog;
  fr.a = actor;
  fr.selfBench = selfBench;
  fr.sourceCardId = sourceCardId;
  st.effectStack.push_back(fr);
  run_program(st, tape);
}

// Dispatch a card's effect by id through the VM (trainer play / legacy on-evolve).
static void begin_effect(GameState& st, int cardId, const std::vector<int>& tape) {
  begin_program(st, card_program(cardId), st.yourIndex, tape, -1, cardId);
}

// Dispatch an activated Pokemon ability through the VM (no draws during begin).
static void begin_ability(GameState& st, int cardId) {
  begin_effect(st, cardId, {});
}

static void set_same_player_promote_pending(GameState& st, int who) {
  st.yourIndex = who;
  PendingDecision pd;
  pd.context = 4;  // TO_ACTIVE
  Player& p = st.players[who];
  for (int j = 0; j < static_cast<int>(p.bench.size()); ++j)
    pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                          Atom::I(p.bench[j].id)});
  st.pending = pd;
  EffectFrame promote;
  promote.effect = EFF_ABILITY_PROMOTE;
  st.effectStack.push_back(promote);
}

static void play_antique_fossil(GameState& st, int cardId) {
  Player& me = st.players[st.yourIndex];
  int selfBench = static_cast<int>(me.bench.size());
  InPlay p = make_inplay(cardId);
  p.appearThisTurn = true;
  me.bench.push_back(p);
  remove_hand_card(st, me, cardId);
  st.turnActionCount += 1;
  begin_program(st, on_play_program(cardId), st.yourIndex, {}, selfBench);
}

static void discard_antique_fossil_from_play(GameState& st, int area, int idx) {
  Player& me = st.players[st.yourIndex];
  if (area == AREA_ACTIVE) {
    if (!me.activeKnown || !is_antique_fossil(me.active.id) || me.bench.empty())
      return;
    ko_active(me);  // no prize / KO record; just move the stack to discard.
    st.turnActionCount += 1;
    set_same_player_promote_pending(st, st.yourIndex);
    return;
  }
  if (area != AREA_BENCH || idx < 0 || idx >= static_cast<int>(me.bench.size()))
    return;
  if (!is_antique_fossil(me.bench[idx].id)) return;
  ko_bench(me, idx);  // no prize / KO record; just move the stack to discard.
  st.turnActionCount += 1;
}

// Play a trainer from hand: move it (stadium -> stadium zone, else discard),
// set the per-turn flag, count the action, then begin its effect.
static void play_trainer(GameState& st, int cardId, const std::vector<int>& tape) {
  Player& me = st.players[st.yourIndex];
  const CardInfo* ci = find_card(cardId);
  if (is_antique_fossil(cardId)) {
    play_antique_fossil(st, cardId);
    return;
  }
  if (cardId == 1090) {
    play_ogres_mask(st);
    return;
  }
  if (cardId == 1087) {
    play_hand_trimmer(st);
    return;
  }
  if (cardId == 1131) {
    play_bother_bot(st);
    return;
  }
  remove_hand_card(st, me, cardId);
  st.turnActionCount += 1;
  if (ci && ci->cardType == SUPPORTER) {
    st.supporterPlayed = true;
    if (is_team_rocket_card_id(cardId))
      st.teamRocketSupporterPlayed = true;
  }
  if (ci && ci->cardType == STADIUM) {
    st.stadiumPlayed = true;
    int oldStadium = cur_stadium(st);
    int oldOwner = st.stadiumOwner >= 0 ? st.stadiumOwner : st.yourIndex;
    if (oldStadium >= 0)
      st.players[oldOwner].discard.push_back(oldStadium);
    st.stadium.clear();
    st.stadium.push_back(cardId);
    st.stadiumOwner = st.yourIndex;
    st.stadiumAbilityUsed = false;
    apply_stadium_hp_change(st, oldStadium, cardId);  // swap HP passives
    enforce_area_zero_bench_limits(st);
    enforce_rotom_tool_limits(st);
    enforce_festival_status_recovery(st);
    if (resolve_main_action_zero_hp(st, st.yourIndex, 1 - st.yourIndex))
      return;
    begin_effect(st, cardId, tape);
    return;
  }
  begin_effect(st, cardId, tape);
  // The card hits the discard only once its effect fully resolves; cabt keeps
  // it out of every zone while a sub-decision is pending.
  if (st.has_pending()) {
    st.pendingTrainerDiscard = cardId;
    st.pendingTrainerOwner = st.yourIndex;
  } else {
    bool zeroHpKo = resolve_main_action_zero_hp(st, st.yourIndex,
                                                st.yourIndex);
    me.discard.push_back(cardId);
    mark_completed_trainer_play(st, cardId);
    if (zeroHpKo)
      return;
    if (st.endTurnAfterProgram) {
      st.endTurnAfterProgram = false;
      end_turn(st);
    }
  }
}

static bool festival_lead_active_id(int id) {
  return id == 93 || id == 100 || id == 240 || id == 247;
}

static bool festival_lead_followup_available(const GameState& st, int attacker) {
  const Player& me = st.players[attacker];
  return cur_stadium(st) == 1245 && me.activeKnown &&
         festival_lead_active_id(me.active.id) &&
         st.festivalLeadAttackTurn[attacker] != st.turn;
}

static void append_zero_hp_kos_to_replacement_flow(GameState& st,
                                                   int firstOwner) {
  int nextGroup = 0;
  for (int group : st.replacementKoGroups)
    nextGroup = std::max(nextGroup, group + 1);
  int attackDamageGroup[2] = {-1, -1};
  auto push_ko = [&](int taker, int owner, const InPlay& knocked,
                     bool activeKo) {
    bool attackDamageKo = knocked.damagedByAttackTurn == st.turn &&
                          knocked.damagedByAttackSide == taker;
    int group = -1;
    if (attackDamageKo && !activeKo) {
      if (attackDamageGroup[owner] < 0)
        attackDamageGroup[owner] = nextGroup++;
      group = attackDamageGroup[owner];
    } else {
      group = nextGroup++;
    }
    st.replacementKoPairs.push_back(taker);
    st.replacementKoPairs.push_back(contextual_prize_value(st, taker, owner,
                                                           knocked, activeKo));
    st.replacementKoGroups.push_back(group);
  };

  int first = (firstOwner >= 0 && firstOwner < 2) ? firstOwner : 0;
  int order[2] = {first, 1 - first};
  for (int who : order) {
    Player& p = st.players[who];
    int taker = 1 - who;
    if (p.activeKnown && p.active.hp <= 0) {
      InPlay knocked = p.active;
      int koId = knocked.id;
      push_ko(taker, who, knocked, true);
      record_attack_damage_ko_marker(st, who, knocked);
      note_pending_ko_aura(st, who, knocked);
      ko_active(p);
      record_ko(st, who, koId);
    }
    for (int j = static_cast<int>(p.bench.size()) - 1; j >= 0; --j) {
      if (p.bench[j].hp > 0) continue;
      InPlay knocked = p.bench[j];
      int koId = knocked.id;
      push_ko(taker, who, knocked, false);
      record_attack_damage_ko_marker(st, who, knocked);
      ko_bench(p, j);
      record_ko(st, who, koId);
    }
  }
}

// After an attack program completes, process the deferred KO(s) or end the turn.
// The common case (only the defender's Active KO'd) uses the bespoke prize/promote
// path. If the attacker's OWN Active is also/instead KO'd -- recoil or self-placed
// damage counters -- the Checkup machinery resolves it (it separates the promoter
// from the next-turn player and scores a simultaneous double-KO / tie correctly).
static void post_attack(GameState& st, int attacker) {
  Player& opp = st.players[1 - attacker];
  Player& me = st.players[attacker];
  reattach_boomerang_energy_after_attack(st, attacker);
  bool festivalFollowup = festival_lead_followup_available(st, attacker);
  if (opp.activeKnown)
    apply_delayed_attack_damage_ko_replacement(st, 1 - attacker, opp.active,
                                               true, attacker);
  if (me.activeKnown)
    apply_delayed_attack_damage_ko_replacement(st, attacker, me.active, true,
                                               1 - attacker);
  for (InPlay& b : opp.bench)
    apply_delayed_attack_damage_ko_replacement(st, 1 - attacker, b, false,
                                               attacker);
  for (InPlay& b : me.bench)
    apply_delayed_attack_damage_ko_replacement(st, attacker, b, false,
                                               1 - attacker);
  if (opp.activeKnown)
    apply_delayed_tenacious_body(st, 1 - attacker, opp.active, true, attacker);
  if (me.activeKnown)
    apply_delayed_tenacious_body(st, attacker, me.active, true, attacker);
  for (InPlay& b : opp.bench)
    apply_delayed_tenacious_body(st, 1 - attacker, b, false, attacker);
  for (InPlay& b : me.bench)
    apply_delayed_tenacious_body(st, attacker, b, false, attacker);
  bool oppKO = opp.activeKnown && opp.active.hp <= 0;
  bool selfKO = me.activeKnown && me.active.hp <= 0;
  bool benchKO = false;  // damage-counter / spread attacks can KO Benched Pokemon
  for (int s = 0; s < 2 && !benchKO; ++s)
    for (const auto& b : st.players[s].bench)
      if (b.hp <= 0) { benchKO = true; break; }
  if (selfKO && oppKO && st.lastAttackId[attacker] == 98)
    st.checkupKoFirst = -1;
  if (!st.replacementKoPairs.empty()) {
    append_zero_hp_kos_to_replacement_flow(st, attacker);
    st.checkupNext = 1 - attacker;
    st.checkupKoFirst = -1;
    EffectFrame fr;
    fr.effect = EFF_CHECKUP;
    fr.phase = 0;
    fr.a = 0;
    fr.scratch = st.replacementKoPairs;
    fr.savedScratch = st.replacementKoGroups;
    st.replacementKoPairs.clear();
    st.replacementKoGroups.clear();
    st.effectStack.push_back(fr);
    checkup_step(st);
    return;
  }
  if (selfKO || benchKO) {  // anything beyond a lone defender-Active KO -> sweep
    st.checkupNext = 1 - attacker;  // the defender plays next regardless
    if (st.checkupKoFirst < 0) {
      if (selfKO && oppKO && in_play_has(opp, 214)) {
        st.checkupKoFirst = attacker;
      } else if (selfKO && oppKO && in_play_has(me, 214)) {
        st.checkupKoFirst = 1 - attacker;
      } else if (selfKO && oppKO) {
        const CardInfo* selfCard = me.activeKnown ? find_card(me.active.id) : nullptr;
        if (opp.activeKnown &&
            opp.active.damagedByAttackEqualCountersTurn == st.turn) {
          st.checkupKoFirst = 1 - attacker;
        } else if (st.lastAttackId[attacker] == 305) {
          st.checkupKoFirst =
              (me.activeKnown && active_tool(st, me.active, 1156))
                  ? 1 - attacker
                  : attacker;
        } else if (st.lastAttackId[attacker] == 98 && opp.activeKnown) {
          int oppActivePrizeValue =
              contextual_prize_value(st, attacker, 1 - attacker,
                                     opp.active, true);
          st.checkupKoFirst =
              (me.prizeCount <= oppActivePrizeValue) ? attacker : 1 - attacker;
        } else if (st.lastAttackId[attacker] == 1338 && opp.activeKnown) {
          int oppActivePrizeValue =
              contextual_prize_value(st, attacker, 1 - attacker,
                                     opp.active, true);
          st.checkupKoFirst =
              (me.prizeCount < 6 || me.prizeCount <= oppActivePrizeValue)
                  ? 1 - attacker
                  : attacker;
        } else {
          st.checkupKoFirst =
              (st.lastAttackId[attacker] == 1446 &&
               me.activeKnown && active_tool(st, me.active, 1156))
                  ? 1 - attacker
                  : (st.lastAttackId[attacker] == 1276)
                  ? 1 - attacker
                  : (st.lastAttackId[attacker] == 537 ||
                     st.lastAttackId[attacker] == 1446 ||
                     (selfCard && selfCard->megaEx && !opp.bench.empty()))
                        ? attacker
                        : 0;
        }
      } else if (oppKO && benchKO && !selfKO) {
        int oppActivePrizeValue =
            opp.activeKnown
                ? contextual_prize_value(st, attacker, 1 - attacker,
                                         opp.active, true)
                : 1;
        if (st.lastAttackId[attacker] == 399) {
          st.checkupKoFirst =
              (me.activeKnown && active_tool(st, me.active, 1156))
                  ? attacker
                  : ((me.prizeCount <= oppActivePrizeValue)
                         ? attacker
                         : 1 - attacker);
        } else {
          st.checkupKoFirst =
              (me.prizeCount <= oppActivePrizeValue)
                  ? 1 - attacker
                  : (st.lastAttackId[attacker] == 1341
                         ? 1 - attacker
                         : (st.lastAttackId[attacker] == 764
                                ? attacker
                                : 1 - attacker));
        }
      } else {
        st.checkupKoFirst =
            (selfKO && !oppKO && !benchKO)
                ? attacker
                : (st.lastAttackId[attacker] == 295 ? 1 - attacker : 0);
      }
    }
    checkup_resolve(st);
  } else if (oppKO) {
    st.checkupKoFirst = -1;
    InPlay knocked = opp.active;
    int pv = contextual_prize_value(st, attacker, 1 - attacker, knocked, true);
    if (st.lastAttackId[attacker] == 986 && me.activeKnown &&
        knocked.damagedByAttackTurn == st.turn &&
        knocked.damagedByAttackSide == attacker) {
      me.active.preventDmgTurn = st.turn + 1;
      me.active.preventEffectsTurn = st.turn + 1;
    }
    if (festivalFollowup) {
      st.festivalLeadAttackTurn[attacker] = st.turn;
      st.festivalLeadResumeTurn[attacker] = st.turn;
    }
    if (ko_skill_order_source_available(st, 1 - attacker, knocked, -1) &&
        amulet_hope_trigger_available(st, 1 - attacker, knocked)) {
      set_ko_trigger_order_pending(st, 1 - attacker, attacker, -1, pv,
                                   knocked.id);
      return;
    }
    if (ko_hook_trigger_available(st, 1 - attacker, knocked, -1)) {
      st.checkupNext = 1 - attacker;
      if (set_ko_hook_pending(st, 1 - attacker, -1))
        return;
    }
    if (set_heavy_baton_pending(st, 1 - attacker, attacker, pv, festivalFollowup))
      return;
    if (set_huntail_pending(st, 1 - attacker, attacker, -1, pv, festivalFollowup))
      return;
    if (active_tool(st, knocked, 1169) && knocked.damagedByAttackTurn == st.turn &&
        set_amulet_hope_pending(st, 1 - attacker, attacker, pv))
      return;
    int koId = knocked.id;
    record_attack_damage_ko_marker(st, 1 - attacker, knocked);
    note_pending_ko_aura(st, 1 - attacker, knocked);
    ko_active(opp);
    record_ko(st, 1 - attacker, koId);
    continue_active_ko_after_amulet(st, attacker, 1 - attacker, pv);
  } else {
    st.checkupKoFirst = -1;
    if (festivalFollowup) {
      st.festivalLeadAttackTurn[attacker] = st.turn;
      st.yourIndex = attacker;
      return;
    }
    bool delayedEndTurnKo = delayed_end_turn_would_ko(st, attacker);
    end_turn(st);
    if (st.has_pending() && st.pending.context == 7 && !delayedEndTurnKo)
      clear_attack_end_turn_flags(st);
  }
}

static void clear_reconstructed_attack_locks(Player& p) {
  auto clear_one = [](InPlay& pk) {
    if (pk.reconstructedAbilityLocked)
      pk.abilityUsedThisTurn = pk.reconstructedAbilityPrevUsed;
    pk.reconstructedAbilityLocked = false;
    pk.reconstructedAbilityPrevUsed = false;
    pk.reconstructedAbilityLockTurn = -1;
    pk.reconstructedAttackLockTurn = -1;
    pk.reconstructedAttackLocks.clear();
  };
  if (p.activeKnown) clear_one(p.active);
  for (InPlay& b : p.bench)
    clear_one(b);
}

static void clear_reconstructed_attack_locks(GameState& st) {
  clear_reconstructed_attack_locks(st.players[0]);
  clear_reconstructed_attack_locks(st.players[1]);
}

void apply(GameState& st, const Action& a, const std::vector<int>& tape) {
  StateSnap before;
  if (st.collectLogs) before = snapshot_state(st);
  size_t logStart = st.nativeLogs.size();
  clear_reconstructed_attack_locks(st);
  st.replayTape = tape;
  st.replayTapePos = 0;
  Player& me = st.players[st.yourIndex];
  switch (a.kind) {
    case ACT_END: {
      st.turnActionCount += 1;
      bool delayedEndTurnKo = delayed_end_turn_would_ko(st, st.yourIndex);
      end_turn(st);
      if (st.has_pending() && st.pending.context == 7 && !delayedEndTurnKo)
        clear_attack_end_turn_flags(st);
      break;
    }
    case ACT_ATTACH: {
      InPlay& tgt =
          (a.targetArea == AREA_ACTIVE) ? me.active : me.bench[a.targetIndex];
      NativeLog attachLog;
      attachLog.type = 11;
      attachLog.playerIndex = st.yourIndex;
      attachLog.cardId = a.cardId;
      attachLog.cardIdTarget = tgt.id;
      emit_log(st, attachLog);
      int attachTargetIdx =
          (a.targetArea == AREA_BENCH) ? a.targetIndex : -1;
      int attachTargetCardId = tgt.id;
      const CardInfo* ci = find_card(a.cardId);
      bool attachedEnergyFromHand = false;
      if (ci && ci->cardType == TOOL) {
        int damage = tgt.maxHp - tgt.hp;
        attach_tool_card(tgt, a.cardId);
        tgt.maxHp = effective_max_hp(tgt.id, tgt.tools, st, tgt.energyCardIds,
                                     tgt.energies, st.yourIndex);
        tgt.hp = std::max(0, tgt.maxHp - damage);
      } else {
        tgt.energies.push_back(attached_energy_unit_type(a.cardId, tgt));
        push_attached_energy_card(tgt, a.cardId);
        if (a.cardId == 15 && !is_team_rocket_card_id(tgt.id)) {
          tgt.energies.pop_back();
          pop_attached_energy_card(tgt);
          me.discard.push_back(a.cardId);
          refresh_inplay_max_hp(st, tgt);
          st.energyAttached = true;
        } else {
          refresh_inplay_max_hp(st, tgt);
          st.energyAttached = true;
          if (me.activeKnown && me.active.id == 297 && a.cardId != 13) {
            int before = tgt.hp;
            tgt.hp = std::min(tgt.maxHp, tgt.hp + 90);
            if (tgt.hp > before) tgt.healedThisTurn = true;
          }
          apply_energy_attach_reactive(st, tgt, true);
          attachedEnergyFromHand = true;
        }
      }
      remove_hand_card(st, me, a.cardId);
      st.turnActionCount += 1;
      enforce_rotom_tool_limits(st);
      enforce_area_zero_bench_limits(st);
      if (!attachedEnergyFromHand ||
          !start_on_attach_trigger_order(st, st.yourIndex, attachTargetIdx,
                                         attachTargetCardId, a.cardId)) {
        begin_program(st, on_attach_program(a.cardId), st.yourIndex, {},
                      attachTargetIdx, a.cardId);
      }
      if (!st.has_pending())
        resolve_main_action_zero_hp(st, st.yourIndex, st.yourIndex);
      break;
    }
    case ACT_PLAY_BASIC: {
      int selfBench = static_cast<int>(me.bench.size());
      InPlay p;
      p.id = a.cardId;
      p.maxHp = effective_max_hp(a.cardId, {}, st, {}, {}, st.yourIndex);
      p.hp = p.maxHp;
      p.appearThisTurn = true;
      me.bench.push_back(p);
      emit_move_log(st, st.yourIndex, a.cardId, AREA_HAND, AREA_BENCH);
      remove_hand_card(st, me, a.cardId);
      st.turnActionCount += 1;
      enforce_rotom_tool_limits(st);
      enforce_area_zero_bench_limits(st);
      if (!start_on_play_basic_trigger_order(st, st.yourIndex, selfBench,
                                             a.cardId)) {
        apply_risky_ruins_bench_entry(st, st.yourIndex, selfBench);
        begin_program(st, on_play_program(a.cardId), st.yourIndex, {},
                      selfBench, a.cardId);
      }
      break;
    }
    case ACT_SETUP_ACTIVE: {
      if (st.turn <= 0 && !me.activePresent && me.handKnown &&
          std::find(me.hand.begin(), me.hand.end(), a.cardId) != me.hand.end() &&
          a.cardId == 666) {
        me.activePresent = true;
        me.activeKnown = false;
        emit_move_log(st, st.yourIndex, a.cardId, AREA_HAND, AREA_ACTIVE);
        remove_hand_card(st, me, a.cardId);
      }
      break;
    }
    case ACT_PLAY_TRAINER: {
      const CardInfo* ci = find_card(a.cardId);
      if (pokemon_played_as_basic_from_hand(st, st.yourIndex, a.cardId, ci)) {
        int selfBench = static_cast<int>(me.bench.size());
        InPlay p;
        p.id = a.cardId;
        p.maxHp = effective_max_hp(a.cardId, {}, st, {}, {}, st.yourIndex);
        p.hp = p.maxHp;
        p.appearThisTurn = true;
        me.bench.push_back(p);
        emit_move_log(st, st.yourIndex, a.cardId, AREA_HAND, AREA_BENCH);
        remove_hand_card(st, me, a.cardId);
        st.turnActionCount += 1;
        enforce_rotom_tool_limits(st);
        enforce_area_zero_bench_limits(st);
        if (!start_on_play_basic_trigger_order(st, st.yourIndex, selfBench,
                                               a.cardId)) {
          apply_risky_ruins_bench_entry(st, st.yourIndex, selfBench);
          begin_program(st, on_play_program(a.cardId), st.yourIndex, {},
                        selfBench, a.cardId);
        }
      } else {
        emit_card_log(st, 10, st.yourIndex, a.cardId);
        play_trainer(st, a.cardId, tape);
      }
      break;
    }
    case ACT_EVOLVE: {
      int targetIdx = a.targetArea == AREA_BENCH ? a.targetIndex : -1;
      InPlay& tgt =
          (a.targetArea == AREA_ACTIVE) ? me.active : me.bench[a.targetIndex];
      NativeLog evoLog;
      evoLog.type = 12;
      evoLog.playerIndex = st.yourIndex;
      evoLog.cardId = a.cardId;
      evoLog.cardIdTarget = tgt.id;
      emit_log(st, evoLog);
      int damage = tgt.maxHp - tgt.hp;
      tgt.preEvo.push_back(tgt.id);
      tgt.id = a.cardId;
      tgt.maxHp = effective_max_hp(a.cardId, tgt.tools, st, tgt.energyCardIds,
                                   tgt.energies, st.yourIndex);
      tgt.hp = tgt.maxHp - damage;
      tgt.appearThisTurn = true;
      clear_status_for_evolve(st, st.yourIndex, a.targetArea == AREA_ACTIVE,
                              &tgt);
      if (a.cardId == 262)
        refresh_player_inplay_max_hp(st, st.yourIndex);
      bool darkestImpulse = in_play_count_id(st.players[1 - st.yourIndex], 428) > 0;
      remove_hand_card(st, me, a.cardId);
      st.turnActionCount += 1;
      enforce_rotom_tool_limits(st);
      enforce_area_zero_bench_limits(st);
      if (start_evolve_trigger_order(st, st.yourIndex, targetIdx, a.cardId))
        break;
      if (darkestImpulse)
        apply_darkest_impulse(st, st.yourIndex, targetIdx);
      int evoProg = card_program(a.cardId);  // on-evolve program (e.g. Hariyama)
      if (evoProg >= 0) {
        EffectFrame ef;
        ef.effect = FLOW_PROGRAM;
        ef.program = evoProg;
        ef.a = st.yourIndex;
        st.effectStack.push_back(ef);
        run_program(st, {});
      }
      if (!st.has_pending())
        begin_program(st, on_evolve_program(a.cardId), st.yourIndex, {},
                      targetIdx);
      if (!st.has_pending() && st.effectStack.empty())
        resolve_main_action_zero_hp(st, st.yourIndex, st.yourIndex);
      break;
    }
    case ACT_ATTACK: {
      NativeLog attackLog;
      attackLog.type = 15;
      attackLog.playerIndex = st.yourIndex;
      attackLog.cardId = me.activeKnown ? me.active.id : 0;
      attackLog.attackId = a.attackId;
      emit_log(st, attackLog);
      st.checkupKoFirst = -1;
      st.turnActionCount += 1;  // the attack counts as an action
      // Festival Lead's second attack re-enters here; cabt's second-attack path
      // skips the pre-attack coin flips, so suppress them on the follow-up.
      bool festivalFollowupAttack =
          st.festivalLeadAttackTurn[st.yourIndex] == st.turn;
      // cabt order: flip the "flip or the attack fails" coin FIRST; Confusion is
      // only checked if it passes. A failed attack-coin ends the attack cleanly
      // (no Confusion self-damage).
      if (!festivalFollowupAttack &&
          me.active.attackFlipFailTurn == st.turn && !flip_heads(st)) {
        post_attack(st, st.yourIndex);
        break;
      }
      if (!festivalFollowupAttack && me.confused &&
          !flip_heads(st)) {         // Confusion: tails -> attack fails, and the
        if (me.activeKnown) {        // Active hits itself for 30
          me.active.hp -= 30;
          if (me.active.hp < 0) me.active.hp = 0;
        }
        post_attack(st, 1 - st.yourIndex);  // a self-KO gives the opponent a prize
        break;
      }
      st.discardedCount = 0;
      st.lastEffectCount = 0;
      if (me.activeKnown && me.active.id == 163 && a.attackId == 213) {
        start_slowking_seek_inspiration(st);
        break;
      }
      if (me.activeKnown && me.active.id == 660 && a.attackId == 955) {
        start_ninetales_shapeshifter(st);
        break;
      }
      if (me.activeKnown) {
        bool copiedDirect = false;
        if (const CardInfo* aci = find_card(me.active.id)) {
          for (int k = 0; k < aci->n_attacks; ++k) {
            if (direct_opp_active_copy_attack(me.active.id, aci->attacks[k].id) &&
                opponent_active_attack_info(st, st.yourIndex, a.attackId)) {
              copiedDirect = true;
              break;
            }
          }
        }
        if (copiedDirect) {
          const AttackInfo* copied =
              opponent_active_attack_info(st, st.yourIndex, a.attackId);
          int prog = attack_program(a.attackId);
          EffectFrame fr;
          fr.effect = FLOW_PROGRAM;
          fr.program = (prog >= 0) ? prog : vanilla_attack_program();
          fr.a = st.yourIndex;
          fr.attackId = a.attackId;
          fr.attackCardId = me.active.id;
          fr.copiedAttackBaseDamage = copied ? copied->damage : 0;
          st.effectStack.push_back(fr);
          run_program(st, {});
          break;
        }
        if (const AttackInfo* copied =
                n_zoroark_benched_attack_info(st, st.yourIndex, a.attackId)) {
          int prog = attack_program(a.attackId);
          EffectFrame fr;
          fr.effect = FLOW_PROGRAM;
          fr.program = (prog >= 0) ? prog : vanilla_attack_program();
          fr.a = st.yourIndex;
          fr.attackId = a.attackId;
          fr.attackCardId = me.active.id;
          fr.copiedAttackBaseDamage = copied->damage;
          st.effectStack.push_back(fr);
          run_program(st, {});
          break;
        }
      }
      int prog = (me.activeKnown && me.active.id == 481 && a.attackId == 680)
                     ? attack_program(100680)
                     : attack_program(a.attackId);
      EffectFrame fr;
      fr.effect = FLOW_PROGRAM;
      fr.program = (prog >= 0) ? prog : vanilla_attack_program();
      fr.a = st.yourIndex;
      fr.attackId = a.attackId;
      fr.attackCardId = me.activeKnown ? me.active.id : 0;
      if (me.activeKnown) {
        if (const AttackInfo* inherited =
                inherited_attack_info(st, st.yourIndex, me.active, a.attackId))
          fr.copiedAttackBaseDamage = inherited->damage;
      }
      st.effectStack.push_back(fr);
      run_program(st, {});  // DAMAGE/effects, then finish_program -> post_attack
      break;
    }
    case ACT_RETREAT: {
      st.retreated = true;
      st.turnActionCount += 1;
      int cost = retreat_cost(st, st.yourIndex);
      EffectFrame retreat;
      retreat.effect = EFF_RETREAT;
      retreat.phase = cost;
      st.effectStack.push_back(retreat);
      if (cost > 0)
        set_discard_energy_pending(st);  // pay the cost first
      else
        set_own_switch_pending(st);  // free retreat -> straight to the switch
      break;
    }
    case ACT_ABILITY: {
      if (a.targetArea == AREA_STADIUM) {  // the in-play Stadium's once/turn ability
        if (st.stadium.empty()) break;
        if (st.stadium[0] == 1249) {
          start_grand_tree(st);
          break;
        }
        int prog = stadium_ability_program(st.stadium[0]);
        if (prog < 0) break;
        st.stadiumAbilityUsed = true;
        st.turnActionCount += 1;
        EffectFrame fr;
        fr.effect = FLOW_PROGRAM;
        fr.program = prog;
        fr.a = st.yourIndex;
        fr.sourceCardId = st.stadium[0];
        st.effectStack.push_back(fr);
        run_program(st, {});
        break;
      }
      InPlay* who = nullptr;
      if (a.targetArea == AREA_ACTIVE && me.activeKnown)
        who = &me.active;
      else if (a.targetArea == AREA_BENCH &&
               a.targetIndex < static_cast<int>(me.bench.size()))
        who = &me.bench[a.targetIndex];
      if (!who) break;
      bool isActiveAbility = a.targetArea == AREA_ACTIVE;
      if (ability_suppressed(st, st.yourIndex, *who, isActiveAbility)) break;
      int abilityCardId = who->id;
      int beforeAbilityActionCount = st.turnActionCount;
      st.turnActionCount += 1;
      int prog = ability_program(abilityCardId);
      if (prog >= 0) {  // general activated once-per-turn ability
        if (!ability_repeatable(abilityCardId)) who->abilityUsedThisTurn = true;
        mark_ability_group_used(st, st.yourIndex, abilityCardId);
        EffectFrame fr;
        fr.effect = FLOW_PROGRAM;
        fr.program = prog;
        fr.a = st.yourIndex;
        fr.selfBench = (a.targetArea == AREA_BENCH) ? a.targetIndex : -1;
        fr.sourceCardId = abilityCardId;
        st.effectStack.push_back(fr);
        run_program(st, {});
      } else {
        begin_ability(st, abilityCardId);  // Lunatone (bespoke card_program path)
      }
      if (abilityCardId == 109 && st.result >= 0)
        st.turnActionCount = beforeAbilityActionCount;
      break;
    }
    case ACT_DISCARD_INPLAY: {
      discard_antique_fossil_from_play(st, a.targetArea, a.targetIndex);
      break;
    }
  }
  if (st.collectLogs) emit_deep_delta_logs(st, before, logStart);
}

// Take prizes after a KO; chain to promotion or a win.
static void resolve_prize(GameState& st, const std::vector<int>& sel) {
  EffectFrame& fr = st.effectStack.back();
  int attacker = fr.a;
  Player& me = st.players[attacker];
  int taken = take_selected_prizes(
      st, attacker, sel, pending_prize_take_count(st, attacker, fr.phase));
  st.turnActionCount += 1;
  fr.phase -= taken;
  if (fr.phase <= 0) clear_pending_ko_auras(st);
  if (me.prizeCount == 0) {  // took the last prize -> win
    set_result(st, attacker, 1);
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    return;
  }
  if (fr.phase > 0) {  // more prizes to take (ex / mega-ex)
    set_prize_pending(st, attacker);
    return;
  }
  int defender = 1 - attacker;
  if (st.players[defender].bench.empty()) {  // nothing to promote -> win
    set_result(st, attacker, 3);
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    return;
  }
  st.yourIndex = defender;  // hand off: the defender promotes a new Active
  PendingDecision pd;
  pd.context = 4;  // TO_ACTIVE
  for (int j = 0; j < static_cast<int>(st.players[defender].bench.size()); ++j)
    pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                          Atom::I(st.players[defender].bench[j].id)});
  st.pending = pd;
  fr.effect = EFF_PROMOTE;
}

static void finish_ability_ko(GameState& st, EffectFrame& fr,
                              const std::vector<int>& tape) {
  int owner = fr.a;
  int taker = fr.savedSrc;
  if (!st.players[owner].activeKnown) {
    if (st.players[owner].bench.empty()) {
      set_result(st, taker, 3);
      st.effectStack.pop_back();
      st.pending = PendingDecision();
      return;
    }
    st.yourIndex = owner;
    fr.effect = EFF_ABILITY_PROMOTE;
    set_promote_pending(st, owner);
    return;
  }
  st.yourIndex = owner;
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  if (!st.effectStack.empty() && st.effectStack.back().effect == FLOW_PROGRAM)
    run_program(st, tape);
}

static void resolve_ability_ko(GameState& st, const std::vector<int>& sel,
                               const std::vector<int>& tape) {
  EffectFrame& fr = st.effectStack.back();
  int taker = fr.savedSrc;
  Player& prizeTaker = st.players[taker];
  if (fr.phase > 0 && prizeTaker.prizeCount > 0) {
    int taken = take_selected_prizes(
        st, taker, sel, pending_prize_take_count(st, taker, fr.phase));
    st.turnActionCount += 1;
    fr.phase -= taken;
    if (fr.phase <= 0) clear_pending_ko_auras(st);
    if (prizeTaker.prizeCount == 0) {
      set_result(st, taker, 3);
      st.effectStack.pop_back();
      st.pending = PendingDecision();
      return;
    }
    if (fr.phase > 0) {
      set_prize_pending(st, taker);
      return;
    }
  }
  finish_ability_ko(st, fr, tape);
}

// Defender promotes a Benched Pokemon into the empty Active Spot; then the
// attacker's turn ends and the defender's turn begins.
static void resolve_promote(GameState& st, const std::vector<int>& sel) {
  int defender = st.yourIndex;
  if (!sel.empty()) {
    int benchIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
    Player& d = st.players[defender];
    int oldActive = d.activeKnown ? d.active.id : 0;
    int newActive = d.bench[benchIdx].id;
    d.active = d.bench[benchIdx];
    d.bench.erase(d.bench.begin() + benchIdx);
    d.activePresent = true;
    d.activeKnown = true;
    d.active.movedToActiveThisTurn = true;
    clear_active_spot_locks(d.active);
    clear_status(d);  // a freshly promoted Active has no conditions
    NativeLog log;
    log.type = 8;
    log.playerIndex = defender;
    log.cardIdActive = oldActive;
    log.cardIdBench = newActive;
    emit_log(st, log);
  }
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  int attacker = 1 - defender;
  if (st.festivalLeadResumeTurn[attacker] == st.turn) {
    st.festivalLeadResumeTurn[attacker] = -1;
    st.yourIndex = attacker;
    return;
  }
  start_turn(st, defender);  // post-attack KO: attacker ends, defender begins
}

// Same-player promotion after an effect removes its owner's Active without a KO
// prize flow (for example Dudunsparce's Run Away Draw), then resume that effect.
static void resolve_ability_promote(GameState& st, const std::vector<int>& sel,
                                    const std::vector<int>& tape) {
  int who = st.yourIndex;
  bool promoted = false;
  if (!sel.empty()) {
    int benchIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
    Player& p = st.players[who];
    int oldActive = p.activeKnown ? p.active.id : 0;
    int newActive = p.bench[benchIdx].id;
    p.active = p.bench[benchIdx];
    p.bench.erase(p.bench.begin() + benchIdx);
    p.activePresent = true;
    p.activeKnown = true;
    p.active.movedToActiveThisTurn = true;
    clear_status(p);
    NativeLog log;
    log.type = 8;
    log.playerIndex = who;
    log.cardIdActive = oldActive;
    log.cardIdBench = newActive;
    emit_log(st, log);
    promoted = true;
  }
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  if (promoted)
    queue_move_triggers(st, who, -1, true, true);
  if (!st.effectStack.empty() && st.effectStack.back().effect == FLOW_PROGRAM)
    run_program(st, tape);
}

// Retreat: phase>0 = an energy-discard step (pay the cost); phase==0 = the
// switch step. Energies are mono-type here, so any unit discards equivalently.
static void resolve_retreat(GameState& st, const std::vector<int>& sel) {
  EffectFrame& fr = st.effectStack.back();
  Player& me = st.players[st.yourIndex];
  if (fr.phase > 0) {
    int energyIdx = -1;
    if (!sel.empty() && sel[0] >= 0 &&
        sel[0] < static_cast<int>(st.pending.options.size()))
      energyIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
    if (energyIdx < 0)
      energyIdx = static_cast<int>(std::max(me.active.energies.size(),
                                            me.active.energyCardIds.size())) - 1;
    if (energyIdx >= 0 &&
        (energyIdx < static_cast<int>(me.active.energies.size()) ||
         energyIdx < static_cast<int>(me.active.energyCardIds.size()))) {
      int paid = provided_energy_units_for_card(me.active, energyIdx, &st,
                                                st.yourIndex);
      if (energyIdx < static_cast<int>(me.active.energies.size()))
        me.active.energies.erase(me.active.energies.begin() + energyIdx);
      if (energyIdx < static_cast<int>(me.active.energyCardIds.size())) {
        me.discard.push_back(me.active.energyCardIds[energyIdx]);
        erase_attached_energy_card(me.active, energyIdx);
      }
      refresh_inplay_max_hp(st, me.active);
      fr.phase -= std::max(1, paid);
    } else {
      fr.phase -= 1;
    }
    st.turnActionCount += 1;
    if (fr.phase > 0)
      set_discard_energy_pending(st);
    else
      set_own_switch_pending(st);
  } else {
    bool switched = false;
    int movedBenchIdx = -1;
    if (!sel.empty()) {
      int benchIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
      clear_active_spot_locks(me.active);
      std::swap(me.active, me.bench[benchIdx]);
      me.active.movedToActiveThisTurn = true;
      clear_status(me);  // retreating cures special conditions
      apply_mismagius_switch_passive(st, st.yourIndex, benchIdx);
      switched = true;
      movedBenchIdx = benchIdx;
    }
    st.turnActionCount += 1;
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    if (switched) {
      queue_move_triggers(st, st.yourIndex, movedBenchIdx, true, false);
      if (!st.effectStack.empty() && st.effectStack.back().effect == FLOW_PROGRAM)
        run_program(st, {});
    }
  }
}

static void resolve_powerglass(GameState& st, const std::vector<int>& sel) {
  EffectFrame& top = st.effectStack.back();
  if (top.phase == 0) {
    int who = top.a;
    Player& p = st.players[who];
    std::vector<int> indices = powerglass_basic_energy_indices(p);
    std::vector<int> sources;
    auto add_source = [&](int idx) {
      if (idx < 0 || idx >= static_cast<int>(top.scratch.size())) return;
      int source = top.scratch[idx];
      if (std::find(sources.begin(), sources.end(), source) == sources.end())
        sources.push_back(source);
    };
    for (int s : sel) add_source(s);
    for (int i = 0; static_cast<int>(sources.size()) < static_cast<int>(top.scratch.size()) &&
                    i < static_cast<int>(top.scratch.size()); ++i)
      add_source(i);
    st.turnActionCount += 1;
    top.scratch = sources;
    top.pc = 0;
    if (!indices.empty()) {
      for (int i = 0; i < static_cast<int>(top.scratch.size()); ++i) {
        int source = top.scratch[i];
        InPlay* holder = powerglass_holder(p, source);
        if (source < 0 && holder && active_tool(st, *holder, 1163)) {
          top.pc = i;
          top.phase = 1;
          set_powerglass_energy_pending(st, who, source, indices);
          return;
        }
      }
    }
    st.effectStack.pop_back();
    st.pending = PendingDecision();
    st.yourIndex = who;
    finish_end_turn_after_optional_hooks(st, who);
    return;
  }
  EffectFrame fr = top;
  int who = fr.a;
  Player& p = st.players[who];
  InPlay* holder = powerglass_holder(p, fr.selfBench);
  if (!sel.empty() && holder && active_tool(st, *holder, 1163)) {
    int opt = sel[0];
    if (opt >= 0 && opt < static_cast<int>(st.pending.options.size())) {
      int discardIdx = static_cast<int>(st.pending.options[opt][2].i);
      if (discardIdx >= 0 && discardIdx < static_cast<int>(p.discard.size())) {
        const CardInfo* c = find_card(p.discard[discardIdx]);
        if (c && c->cardType == BASIC_ENERGY) {
          holder->energies.push_back(c->energyType);
          push_attached_energy_card(*holder, p.discard[discardIdx]);
          apply_energy_attach_reactive(st, *holder, false);
          adjust_card_zone_count(p, &p.discard, -1);
          p.discard.erase(p.discard.begin() + discardIdx);
        }
      }
    }
  }
  st.turnActionCount += 1;
  st.pending = PendingDecision();
  st.yourIndex = who;
  EffectFrame& cur = st.effectStack.back();
  for (int i = cur.pc + 1; i < static_cast<int>(cur.scratch.size()); ++i) {
    int source = cur.scratch[i];
    InPlay* nextHolder = powerglass_holder(p, source);
    if (source < 0 && nextHolder && active_tool(st, *nextHolder, 1163)) {
      std::vector<int> indices = powerglass_basic_energy_indices(p);
      if (!indices.empty()) {
        cur.pc = i;
        set_powerglass_energy_pending(st, who, source, indices);
        return;
      }
    }
  }
  st.effectStack.pop_back();
  finish_end_turn_after_optional_hooks(st, who);
}

static void continue_active_ko_after_amulet(GameState& st, int taker, int owner,
                                            int prizeValue) {
  if (prizeValue > 0) {
    EffectFrame prize;
    prize.effect = EFF_PRIZE;
    prize.phase = prizeValue;
    prize.a = taker;
    st.effectStack.push_back(prize);
    set_prize_pending(st, taker);
  } else if (st.players[owner].bench.empty()) {
    set_result(st, taker, 3);
  } else {
    st.yourIndex = owner;
    EffectFrame fr;
    fr.effect = EFF_PROMOTE;
    fr.a = taker;
    st.effectStack.push_back(fr);
    set_promote_pending(st, owner);
  }
}

static void resolve_amulet_hope(GameState& st, const std::vector<int>& sel) {
  EffectFrame fr = st.effectStack.back();
  int owner = fr.savedSrc;
  int taker = fr.a;
  Player& p = st.players[owner];
  std::vector<int> idxs;
  for (int s : sel) {
    if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
    idxs.push_back(static_cast<int>(st.pending.options[s][2].i));
  }
  std::sort(idxs.rbegin(), idxs.rend());
  idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
  int moved = 0;
  if (!p.deck.empty()) {
    for (int idx : idxs) {
      if (idx < 0 || idx >= static_cast<int>(p.deck.size())) continue;
      bool known = false;
      int id = erase_deck_at(p, idx, &known);
      p.hand.push_back(id);
      p.handCount += 1;
      if ((known || p.deckKnown) && !p.handKnown)
        add_known_card(p.handKnownCards, id);
      moved += 1;
    }
    shuffle_deck_known(p, st.rng);
  } else {
    moved = std::min(static_cast<int>(idxs.size()), p.deckCount);
    p.handKnown = false;
    p.handCount += moved;
    p.deckCount -= moved;
  }
  st.lastEffectCount = moved;
  st.turnActionCount += 1;
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  if (fr.loopRemain == 1) {
    continue_ko_trigger_order(st);
    return;
  }
  if (fr.phase < 0) {
    checkup_resolve(st);
    return;
  }
  if (p.activeKnown && p.active.hp <= 0) {
    InPlay knocked = p.active;
    int koId = knocked.id;
    record_attack_damage_ko_marker(st, owner, knocked);
    note_pending_ko_aura(st, owner, knocked);
    ko_active(p);
    record_ko(st, owner, koId);
  }
  st.yourIndex = taker;
  continue_active_ko_after_amulet(st, taker, owner, fr.phase);
}

static void resolve_huntail(GameState& st, const std::vector<int>& sel) {
  EffectFrame fr = st.effectStack.back();
  int owner = fr.savedSrc;
  int taker = fr.a;
  Player& p = st.players[owner];
  bool yes = !sel.empty() && sel[0] == 0;
  InPlay* target = inplay_ref(p, fr.selfBench);
  if (yes && target)
    move_basic_water_energy_to_hand(st, owner, *target);
  if (target)
    target->damagedByAttackTurn = -1;
  st.turnActionCount += 1;
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  if (fr.phase < 0) {
    st.yourIndex = taker;
    checkup_resolve(st);
    return;
  }
  if (fr.selfBench < 0 && p.activeKnown) {
    InPlay knocked = p.active;
    int koId = knocked.id;
    note_pending_ko_aura(st, owner, knocked);
    ko_active(p);
    record_ko(st, owner, koId);
  } else if (fr.selfBench >= 0 &&
             fr.selfBench < static_cast<int>(p.bench.size())) {
    int koId = p.bench[fr.selfBench].id;
    ko_bench(p, fr.selfBench);
    record_ko(st, owner, koId);
  }
  if (fr.loopRemain > 0) {
    st.festivalLeadAttackTurn[taker] = st.turn;
    st.festivalLeadResumeTurn[taker] = st.turn;
  }
  st.yourIndex = taker;
  continue_active_ko_after_amulet(st, taker, owner, fr.phase);
}

static void resume_after_heavy_baton(GameState& st, const EffectFrame& fr) {
  int owner = fr.savedSrc;
  int taker = fr.a;
  st.effectStack.pop_back();
  st.pending = PendingDecision();
  if (fr.phase < 0) {
    st.yourIndex = taker;
    checkup_resolve(st);
    return;
  }
  Player& p = st.players[owner];
  if (p.activeKnown) {
    InPlay knocked = p.active;
    int koId = knocked.id;
    note_pending_ko_aura(st, owner, knocked);
    ko_active(p);
    record_ko(st, owner, koId);
  }
  if (fr.loopRemain > 0) {
    st.festivalLeadAttackTurn[taker] = st.turn;
    st.festivalLeadResumeTurn[taker] = st.turn;
  }
  st.yourIndex = taker;
  continue_active_ko_after_amulet(st, taker, owner, fr.phase);
}

static void resolve_heavy_baton(GameState& st, const std::vector<int>& sel) {
  EffectFrame& fr = st.effectStack.back();
  int owner = fr.savedSrc;
  if (owner < 0 || owner > 1) {
    EffectFrame done = fr;
    resume_after_heavy_baton(st, done);
    return;
  }
  Player& p = st.players[owner];
  if (!p.activeKnown) {
    EffectFrame done = fr;
    resume_after_heavy_baton(st, done);
    return;
  }
  if (fr.topDeckSelectedCount == 0) {
    fr.savedScratch.clear();
    for (int s : sel) {
      if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
      int cardId = static_cast<int>(st.pending.options[s][3].i);
      int etype = -1;
      if (basic_energy_type(cardId, etype))
        fr.savedScratch.push_back(cardId);
      if (static_cast<int>(fr.savedScratch.size()) >= 3) break;
    }
    if (fr.savedScratch.empty()) {
      EffectFrame done = fr;
      resume_after_heavy_baton(st, done);
      return;
    }
    PendingDecision pd;
    pd.context = 18;  // attach target
    pd.minCount = static_cast<int>(fr.savedScratch.size());
    pd.maxCount = pd.minCount;
    for (int r = 0; r < pd.maxCount; ++r) {
      for (int j = 0; j < static_cast<int>(p.bench.size()); ++j)
        pd.options.push_back({Atom::S("CARD"), Atom::S("BENCH"), Atom::I(j),
                              Atom::I(p.bench[j].id)});
    }
    fr.topDeckSelectedCount = 1;
    st.yourIndex = owner;
    st.pending = pd;
    return;
  }

  std::vector<int> targets;
  for (int s : sel) {
    if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
    targets.push_back(static_cast<int>(st.pending.options[s][2].i));
  }
  int moved = 0;
  int n = std::min(static_cast<int>(fr.savedScratch.size()),
                   static_cast<int>(targets.size()));
  for (int i = 0; i < n; ++i) {
    int cardId = fr.savedScratch[i];
    int etype = -1;
    if (!basic_energy_type(cardId, etype)) continue;
    auto it = std::find(p.active.energyCardIds.begin(),
                        p.active.energyCardIds.end(), cardId);
    if (it == p.active.energyCardIds.end()) continue;
    int eidx = static_cast<int>(it - p.active.energyCardIds.begin());
    int attachOrder = attached_energy_order(p.active, eidx);
    if (eidx < static_cast<int>(p.active.energies.size()))
      p.active.energies.erase(p.active.energies.begin() + eidx);
    erase_attached_energy_card(p.active, eidx);
    int benchIdx = targets[i];
    if (benchIdx < 0 || benchIdx >= static_cast<int>(p.bench.size())) {
      p.discard.push_back(cardId);
      continue;
    }
    InPlay& dst = p.bench[benchIdx];
    dst.energies.push_back(etype);
    push_attached_energy_card(dst, cardId, attachOrder);
    refresh_inplay_max_hp(st, dst);
    apply_energy_attach_reactive(st, dst, false);
    ++moved;
  }
  refresh_inplay_max_hp(st, p.active);
  st.lastEffectCount = moved;
  st.turnActionCount += 1;
  EffectFrame done = fr;
  resume_after_heavy_baton(st, done);
}

// Resume a suspended VM program after a CHOOSE: record the chosen zone-indices,
// advance past the op, and continue running.
static void resolve_program(GameState& st, const std::vector<int>& sel,
                            const std::vector<int>& tape) {
  EffectFrame& fr = st.effectStack.back();
  const Op& cur = (effect_ops() + fr.program)[fr.pc];
  fr.scratch.clear();
  if (cur.kind == OP_YESNO) {
    fr.scratch.push_back(sel.empty() ? 1 : sel[0]);  // 0 = YES, 1 = NO
    st.lastEffectCount = 0;
  } else if (cur.kind == OP_PRIZE_BARGAIN) {
    bool yes = !sel.empty() && sel[0] == 0;
    st.yourIndex = fr.a;
    st.lastEffectCount = 0;
    if (yes) {
      for (int side = 0; side < 2; ++side) {
        if (st.players[side].prizeCount > 0) {
          Player& pl = st.players[side];
          st.players[side].prizeCount -= 1;
          record_prize_taken(st, side);
          if (!pl.prizes.empty()) {
            int cid = pl.prizes.back();
            consume_known_card(pl.prizesKnownCards, cid);
            pl.prizes.pop_back();
          } else {
            pl.prizesKnown = false;
          }
          if (!pl.prizesKnownMask.empty()) pl.prizesKnownMask.pop_back();
          if (!pl.prizeFaceUp.empty()) pl.prizeFaceUp.pop_back();
          st.lastEffectCount += 1;
        }
      }
    } else {
      st.lastEffectCount = draw_n(st, st.players[fr.a], 4);
    }
  } else if (cur.kind == OP_ATTACH_BASIC_ENERGY_EACH_TARGET &&
             ((cur.p0 == AZ_DECK && cur.p3 == AET_OWN_BENCH) ||
              (cur.p0 == AZ_DISCARD && cur.p3 == AET_OWN_INPLAY))) {
    resolve_attach_basic_energy_each_target_choice(
        st, fr, sel, cur.p0, cur.p1, cur.p2, cur.p3);
    st.lastEffectCount = std::max(0, fr.loopRemain);
    st.turnActionCount += 1;
    st.pending = PendingDecision();
    run_program(st, tape);
    return;
  } else if (cur.kind == OP_OPP_CHOOSE_HAND_TO_DECK) {
    Player& opp = st.players[1 - fr.a];
    std::vector<int> idxs;
    std::vector<int> cardIds;
    for (int s : sel) {
      if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
      idxs.push_back(static_cast<int>(st.pending.options[s][2].i));
      if (st.pending.options[s].size() > 3 && !st.pending.options[s][3].is_none)
        cardIds.push_back(static_cast<int>(st.pending.options[s][3].i));
    }
    std::sort(idxs.begin(), idxs.end());
    idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
    int moved = 0;
    int required = std::max(3, std::max(std::max(0, cur.p0),
                                        std::max(st.pending.minCount,
                                                 st.pending.maxCount)));
    bool known = static_cast<int>(opp.hand.size()) == opp.handCount;
    if (known) {
      for (int id : cardIds) {
        auto it = std::find(opp.hand.begin(), opp.hand.end(), id);
        if (it == opp.hand.end()) continue;
        push_deck_top(opp, *it, true);
        opp.hand.erase(it);
        ++moved;
      }
      while (moved < required && !opp.hand.empty()) {
        int id = opp.hand.front();
        push_deck_top(opp, id, true);
        opp.hand.erase(opp.hand.begin());
        ++moved;
      }
      opp.handCount = static_cast<int>(opp.hand.size());
      if (moved > 0) shuffle_deck_known(opp, st.rng);
    } else {
      moved = std::min(required, opp.handCount);
      int knownMove = std::min(moved, static_cast<int>(opp.handKnownCards.size()));
      for (int i = 0; i < knownMove; ++i)
        add_known_card(opp.deckKnownCards, opp.handKnownCards[i]);
      opp.handKnownCards.erase(opp.handKnownCards.begin(),
                               opp.handKnownCards.begin() + knownMove);
      opp.hand.clear();
      opp.handKnown = false;
      opp.handCount -= moved;
      opp.deckCount += moved;
    }
    st.lastEffectCount = moved;
    st.yourIndex = fr.a;
  } else if (cur.kind == OP_OPP_SWITCH_OUT) {
    Player& opp = st.players[1 - fr.a];
    int switchedBenchIdx = -1;
    if (!sel.empty()) {
      int benchIdx = static_cast<int>(st.pending.options[sel[0]][2].i);
      if (benchIdx >= 0 && benchIdx < static_cast<int>(opp.bench.size())) {
        clear_active_spot_locks(opp.active);
        std::swap(opp.active, opp.bench[benchIdx]);
        opp.active.movedToActiveThisTurn = true;
        clear_status(opp);
        switchedBenchIdx = benchIdx;
        fr.scratch.push_back(benchIdx);
        st.lastEffectCount = 1;
      } else {
        st.lastEffectCount = 0;
      }
    } else {
      st.lastEffectCount = 0;
    }
    st.yourIndex = fr.a;
    if (switchedBenchIdx >= 0)
      queue_move_triggers(st, 1 - fr.a, switchedBenchIdx, true, true);
  } else if (cur.kind == OP_OPP_DISCARD_TO_N) {
    Player& opp = st.players[1 - fr.a];
    std::vector<int> idxs;
    for (int s : sel)
      idxs.push_back(static_cast<int>(st.pending.options[s][2].i));
    std::sort(idxs.rbegin(), idxs.rend());
    idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());
    int discarded = 0;
    bool known = static_cast<int>(opp.hand.size()) == opp.handCount;
    if (known) {
      for (int idx : idxs) {
        if (idx < 0 || idx >= static_cast<int>(opp.hand.size())) continue;
        opp.discard.push_back(opp.hand[idx]);
        opp.hand.erase(opp.hand.begin() + idx);
        opp.handCount -= 1;
        ++discarded;
      }
    } else {
      discarded = std::min(static_cast<int>(idxs.size()), opp.handCount);
      opp.hand.clear();
      opp.handKnown = false;
      opp.handCount -= discarded;
    }
    st.discardedCount = discarded;
    st.lastEffectCount = discarded;
    st.yourIndex = fr.a;
  } else if (cur.kind == OP_CRISPIN_CHOOSE_ENERGIES) {
    for (int s : sel)
      fr.scratch.push_back(static_cast<int>(st.pending.options[s][2].i));
    st.lastEffectCount = static_cast<int>(fr.scratch.size());
  } else if (cur.kind == OP_REPEAT_OPP_ACTIVE_ENERGY_DISCARD) {
    for (int s : sel) {
      if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
      fr.scratch.push_back(static_cast<int>(st.pending.options[s][2].i));
    }
    if (!fr.scratch.empty() && fr.loopRemain > 0)
      fr.loopRemain -= 1;
    else if (fr.scratch.empty())
      fr.loopRemain = 0;
    st.lastEffectCount = static_cast<int>(fr.scratch.size());
  } else if (cur.kind == OP_CHOOSE_STATUS) {
    for (int s : sel) {
      if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
      const Descriptor& d = st.pending.options[s];
      if (!d.empty() && atom_is(d[0], "SPECIAL_CONDITION") &&
          d.size() > 1)
        fr.scratch.push_back(static_cast<int>(d[1].i));
      else if (d.size() > 2)
        fr.scratch.push_back(static_cast<int>(d[2].i));
    }
    st.lastEffectCount = static_cast<int>(fr.scratch.size());
  } else if (cur.kind == OP_CHOOSE_OPP_ACTIVE_ATTACK) {
    for (int s : sel) {
      if (s < 0 || s >= static_cast<int>(st.pending.options.size())) continue;
      const Descriptor& d = st.pending.options[s];
      if (d.size() > 1)
        fr.scratch.push_back(static_cast<int>(d[1].i));
    }
    st.lastEffectCount = static_cast<int>(fr.scratch.size());
  } else if (cur.kind == OP_CHOOSE && fr.sourceCardId == 1079 &&
             cur.p0 == Z_OWN_INPLAY &&
             cur.p1 == F_BASIC_WITH_STAGE2_IN_HAND) {
    Player& me = st.players[fr.a];
    if (!sel.empty() && sel[0] >= 0 &&
        sel[0] < static_cast<int>(st.pending.options.size())) {
      const Descriptor& d = st.pending.options[sel[0]];
      int stage2Id = d.size() > 1 ? static_cast<int>(d[1].i) : 0;
      std::string_view area =
          (d.size() > 2 && d[2].is_str) ? atom_sv(d[2]) : std::string_view();
      int displayIdx = d.size() > 3 ? static_cast<int>(d[3].i) : 0;
      int targetIdx = (area == "ACTIVE") ? -1 : displayIdx;
      InPlay* target = inplay_ref(me, targetIdx);
      int handIdx = -1;
      if (target) {
        for (int i = 0; i < static_cast<int>(me.hand.size()); ++i) {
          if (me.hand[i] == stage2Id &&
              stage2_evolves_from_basic(me.hand[i], target->id)) {
            handIdx = i;
            break;
          }
        }
      }
      if (handIdx >= 0) {
        fr.scratch.push_back(handIdx);
        fr.savedScratch = {targetIdx};
      }
    }
    st.lastEffectCount = static_cast<int>(fr.scratch.size());
  } else if (cur.kind == OP_KO_LOWEST_HP_EXCEPT_SELF) {
    int knocked = 0;
    if (!sel.empty()) {
      int choice = sel[0];
      const Descriptor& d = st.pending.options[choice];
      int idx = static_cast<int>(d[2].i);
      int side = (choice >= 0 && choice < static_cast<int>(fr.savedScratch.size()))
                     ? fr.savedScratch[choice]
                     : 1 - fr.a;
      InPlay* p = inplay_ref(st.players[side], idx);
      if (p) {
        p->hp = 0;
        knocked = 1;
      }
    }
    st.lastEffectCount = knocked;
  } else {  // OP_CHOOSE: record the chosen zone-indices
    for (int s : sel)
      fr.scratch.push_back(static_cast<int>(st.pending.options[s][2].i));
    st.lastEffectCount = static_cast<int>(fr.scratch.size());
    if (cur.kind == OP_CHOOSE_TOP_DECK && fr.topDeckCount > 0)
      fr.topDeckSelectedCount = static_cast<int>(fr.scratch.size());
  }
  st.turnActionCount += 1;  // each decision advances the action counter
  fr.pc += 1;               // past the decision op
  st.pending = PendingDecision();
  run_program(st, tape);
}

void resolve(GameState& st, const std::vector<int>& selection,
             const std::vector<int>& tape) {
  StateSnap before;
  if (st.collectLogs) before = snapshot_state(st);
  size_t logStart = st.nativeLogs.size();
  clear_reconstructed_attack_locks(st);
  st.replayTape = tape;
  st.replayTapePos = 0;
  if (st.effectStack.empty()) return;
  switch (st.effectStack.back().effect) {
    case EFF_PRIZE:
      resolve_prize(st, selection);
      break;
    case EFF_PROMOTE:
      resolve_promote(st, selection);
      break;
    case EFF_ABILITY_PROMOTE:
      resolve_ability_promote(st, selection, tape);
      break;
    case EFF_RETREAT:
      resolve_retreat(st, selection);
      break;
    case FLOW_PROGRAM:
      resolve_program(st, selection, tape);
      break;
    case EFF_CHECKUP:
      resolve_checkup(st, selection);
      break;
    case EFF_POWERGLASS:
      resolve_powerglass(st, selection);
      break;
    case EFF_AMULET_HOPE:
      resolve_amulet_hope(st, selection);
      break;
    case EFF_HUNTAIL:
      resolve_huntail(st, selection);
      break;
    case EFF_HEAVY_BATON:
      resolve_heavy_baton(st, selection);
      break;
    case EFF_GRAND_TREE:
      resolve_grand_tree(st, selection);
      break;
    case EFF_OGRES_MASK:
      resolve_ogres_mask(st, selection);
      break;
    case EFF_PALAFIN_ZERO_TO_HERO:
      resolve_palafin_zero_to_hero(st, selection);
      break;
    case EFF_SLOWKING_SEEK_INSPIRATION:
      resolve_slowking_seek_inspiration(st, selection);
      break;
    case EFF_BOTHER_BOT:
      resolve_bother_bot(st, selection);
      break;
    case EFF_HAND_TRIMMER:
      resolve_hand_trimmer(st, selection);
      break;
    case EFF_DAMAGE_TRIGGER_ORDER:
      resolve_damage_trigger_order(st, selection, tape);
      break;
    case EFF_EVOLVE_TRIGGER_ORDER:
      resolve_evolve_trigger_order(st, selection, tape);
      break;
    case EFF_ON_PLAY_BASIC_TRIGGER_ORDER:
      resolve_on_play_basic_trigger_order(st, selection, tape);
      break;
    case EFF_ON_ATTACH_TRIGGER_ORDER:
      resolve_on_attach_trigger_order(st, selection, tape);
      break;
    case EFF_KO_TRIGGER_ORDER:
      resolve_ko_trigger_order(st, selection);
      break;
    case EFF_FROSLASS_CHECKUP:
      resolve_froslass_checkup(st, selection);
      break;
    case EFF_MAIN_ACTION_KO:
      resolve_checkup(st, selection);
      break;
    case EFF_ABILITY_KO:
      resolve_ability_ko(st, selection, tape);
      break;
    case EFF_KO_RESUME:
      st.effectStack.pop_back();
      st.pending = PendingDecision();
      checkup_resolve(st);
      break;
    default:
      st.effectStack.pop_back();
      st.pending = PendingDecision();
      break;
  }
  if (!st.has_pending())
    advance_effect_stack(st, tape);
  if (st.collectLogs) emit_deep_delta_logs(st, before, logStart);
}

// --- free-running setup ---------------------------------------------------

static void setup_player(GameState& st, int p) {
  constexpr int kMaxSetupAttempts = 1024;
  Player& pl = st.players[p];
  // Mulligan: draw 7 until a Basic Pokemon is present.
  bool found_basic = false;
  for (int attempt = 0; attempt < kMaxSetupAttempts; ++attempt) {
    pl.hand.clear();
    for (int i = 0; i < 7 && !pl.deck.empty(); ++i) {
      pl.hand.push_back(pl.deck.back());
      pl.deck.pop_back();
    }
    bool basic = false;
    for (int id : pl.hand) {
      const CardInfo* c = find_card(id);
      if (c && c->cardType == POKEMON && c->basic) {
        basic = true;
        break;
      }
    }
    if (basic) {
      found_basic = true;
      break;
    }
    for (int id : pl.hand) pl.deck.push_back(id);
    pl.hand.clear();
    shuffle_deck_known(pl, st.rng);
  }
  if (!found_basic) {
    throw std::runtime_error(
        "setup failed to draw a Basic Pokemon within 1024 attempts");
  }
  // Place the first Basic as Active; the rest of the Basics on the Bench.
  for (size_t i = 0; i < pl.hand.size(); ++i) {
    const CardInfo* c = find_card(pl.hand[i]);
    if (c && c->cardType == POKEMON && c->basic) {
      pl.active = make_inplay(pl.hand[i]);
      pl.activePresent = true;
      pl.activeKnown = true;
      pl.hand.erase(pl.hand.begin() + i);
      break;
    }
  }
  for (size_t i = 0; i < pl.hand.size() && (int)pl.bench.size() < pl.benchMax;) {
    const CardInfo* c = find_card(pl.hand[i]);
    if (c && c->cardType == POKEMON && c->basic) {
      pl.bench.push_back(make_inplay(pl.hand[i]));
      pl.hand.erase(pl.hand.begin() + i);
    } else {
      ++i;
    }
  }
  // 6 prizes off the top of the deck.
  for (int i = 0; i < 6 && !pl.deck.empty(); ++i) {
    pl.prizes.push_back(pl.deck.back());
    pl.prizeFaceUp.push_back(false);
    pl.deck.pop_back();
  }
  pl.deckCount = static_cast<int>(pl.deck.size());
  pl.prizeCount = static_cast<int>(pl.prizes.size());
  pl.handCount = static_cast<int>(pl.hand.size());
  pl.handKnown = true;
}

GameState new_game(const std::vector<int>& deck0, const std::vector<int>& deck1,
                   uint64_t seed, bool collectLogs) {
  GameState st;
  st.collectLogs = collectLogs;  // before dealing so setup logs are captured
  st.freeRunning = true;
  st.rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
  st.players[0].deck = deck0;
  st.players[1].deck = deck1;
  for (int p = 0; p < 2; ++p) {
    shuffle_deck(st.players[p].deck, st.rng);
    setup_player(st, p);
  }
  st.firstPlayer = static_cast<int>(next_rand(st.rng) & 1);
  st.result = -1;
  st.turn = 0;
  start_turn(st, st.firstPlayer);  // turn 1: the first player draws
  return st;
}

}  // namespace ptcg
