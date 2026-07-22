#include "ptcg/id_tensors.hpp"

#include "ptcg/card_db.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <string_view>

namespace ptcg {
namespace {

template <typename T>
class IdWriter {
 public:
  class Reference {
   public:
    Reference(T* value, bool* ok) : value_(value), ok_(ok) {}

    template <typename V>
    Reference& operator=(V value) {
      const int64_t wide = static_cast<int64_t>(value);
      if constexpr (sizeof(T) < sizeof(int32_t)) {
        if (wide < std::numeric_limits<T>::min() ||
            wide > std::numeric_limits<T>::max()) {
          *value_ = 0;
          *ok_ = false;
          return *this;
        }
      }
      *value_ = static_cast<T>(wide);
      return *this;
    }

   private:
    T* value_;
    bool* ok_;
  };

  IdWriter(T* data, bool* ok) : data_(data), ok_(ok) {}

  Reference operator[](size_t index) const {
    return Reference(data_ + index, ok_);
  }
  IdWriter operator+(size_t offset) const {
    return IdWriter(data_ + offset, ok_);
  }
  void clear(size_t count) const {
    std::fill(data_, data_ + count, static_cast<T>(0));
  }

 private:
  T* data_;
  bool* ok_;
};

template <typename T>
IdWriter<T> id_writer(T* data, bool* ok) {
  return IdWriter<T>(data, ok);
}

constexpr int CG_AREA_DECK = 1;
constexpr int CG_AREA_HAND = 2;
constexpr int CG_AREA_DISCARD = 3;
constexpr int CG_AREA_ACTIVE = 4;
constexpr int CG_AREA_BENCH = 5;
constexpr int CG_AREA_PRIZE = 6;
constexpr int CG_AREA_STADIUM = 7;
constexpr int CG_AREA_ENERGY = 8;
constexpr int CG_AREA_TOOL = 9;
constexpr int CG_AREA_PRE_EVOLUTION = 10;
constexpr int CG_AREA_PLAYER = 11;
constexpr int CG_AREA_LOOKING = 12;
constexpr int CG_OPTION_NUMBER = 0;
constexpr int CG_OPTION_YES = 1;
constexpr int CG_OPTION_NO = 2;
constexpr int CG_OPTION_CARD = 3;
constexpr int CG_OPTION_TOOL_CARD = 4;
constexpr int CG_OPTION_ENERGY_CARD = 5;
constexpr int CG_OPTION_ENERGY = 6;
constexpr int CG_OPTION_PLAY = 7;
constexpr int CG_OPTION_ATTACH = 8;
constexpr int CG_OPTION_EVOLVE = 9;
constexpr int CG_OPTION_ABILITY = 10;
constexpr int CG_OPTION_DISCARD = 11;
constexpr int CG_OPTION_RETREAT = 12;
constexpr int CG_OPTION_ATTACK = 13;
constexpr int CG_OPTION_END = 14;
constexpr int CG_OPTION_SKILL = 15;
constexpr int CG_OPTION_SPECIAL_CONDITION = 16;
constexpr int CG_SELECT_MAIN = 0;
constexpr int CG_SELECT_CARD = 1;
constexpr int CG_SELECT_ATTACHED_CARD = 2;
constexpr int CG_SELECT_CARD_OR_ATTACHED_CARD = 3;
constexpr int CG_SELECT_ENERGY = 4;
constexpr int CG_SELECT_SKILL = 5;
constexpr int CG_SELECT_ATTACK = 6;
constexpr int CG_SELECT_EVOLVE = 7;
constexpr int CG_SELECT_COUNT = 8;
constexpr int CG_SELECT_YES_NO = 9;
constexpr int CG_SELECT_SPECIAL_CONDITION = 10;
constexpr int CG_CONTEXT_MAIN = 0;
constexpr int CG_CONTEXT_DISCARD_ENERGY_CARD = 26;
constexpr int CG_CONTEXT_DISCARD_TOOL_CARD = 27;
constexpr int CG_CONTEXT_SWITCH_ENERGY_CARD = 28;
constexpr int CG_CONTEXT_DISCARD_CARD_OR_ATTACHED_CARD = 29;

std::string_view desc_str(const Descriptor& d, size_t i) {
  return (i < d.size() && d[i].is_str) ? atom_sv(d[i]) : std::string_view();
}

int desc_int(const Descriptor& d, size_t i, int fallback = 0) {
  return (i < d.size() && !d[i].is_str && !d[i].is_none)
             ? static_cast<int>(d[i].i)
             : fallback;
}

bool atom_equal_value(const Atom& a, const Atom& b) {
  return a.is_str == b.is_str && a.is_none == b.is_none &&
         atom_sv(a) == atom_sv(b) && a.i == b.i;
}

bool descriptor_equal_value(const Descriptor& a, const Descriptor& b) {
  if (a.size() != b.size()) return false;
  if (a.size() >= 4 && desc_str(a, 0) == "ABILITY" &&
      desc_str(b, 0) == "ABILITY" && desc_str(a, 1) == "STADIUM" &&
      desc_str(b, 1) == "STADIUM") {
    for (size_t i = 0; i < 3; ++i)
      if (!atom_equal_value(a[i], b[i])) return false;
    return true;
  }
  for (size_t i = 0; i < a.size(); ++i)
    if (!atom_equal_value(a[i], b[i])) return false;
  return true;
}

const InPlay* state_inplay_at_slot(const Player& p, int slot) {
  if (slot == 0) return p.activePresent ? &p.active : nullptr;
  int bench_idx = slot - 1;
  return bench_idx >= 0 && bench_idx < static_cast<int>(p.bench.size())
             ? &p.bench[bench_idx]
             : nullptr;
}

int cg_area_code(std::string_view area) {
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

int cg_area_index(std::string_view area, int index) {
  if (area == "ACTIVE" || area == "STADIUM") return 0;
  return index;
}

int cg_energy_ref_inplay(int ref) { return ref / 1000 - 1; }
int cg_energy_ref_index(int ref) { return ref % 1000; }

const InPlay* cg_inplay_at(const GameState& st, int owner, int inplayIdx) {
  if (owner < 0 || owner >= 2) return nullptr;
  const Player& p = st.players[owner];
  if (inplayIdx < 0) return p.activeKnown ? &p.active : nullptr;
  return inplayIdx < static_cast<int>(p.bench.size()) ? &p.bench[inplayIdx]
                                                      : nullptr;
}

int cg_attached_energy_slots(const InPlay& p) {
  return std::max(static_cast<int>(p.energies.size()),
                  static_cast<int>(p.energyCardIds.size()));
}

int cg_attached_energy_card_id(const InPlay& p, int energyIdx) {
  return energyIdx >= 0 && energyIdx < static_cast<int>(p.energyCardIds.size())
             ? p.energyCardIds[energyIdx]
             : 0;
}

bool cg_energy_slot_matches(const GameState& st, int owner, int inplayIdx,
                            int energyIdx, int cardId) {
  const InPlay* p = cg_inplay_at(st, owner, inplayIdx);
  if (!p || energyIdx < 0 || energyIdx >= cg_attached_energy_slots(*p))
    return false;
  int attachedId = cg_attached_energy_card_id(*p, energyIdx);
  return cardId <= 0 || attachedId <= 0 || attachedId == cardId;
}

int cg_infer_energy_owner(const GameState& st, int fallback, int inplayIdx,
                          int energyIdx, int cardId) {
  int match = -1;
  for (int side = 0; side < 2; ++side) {
    if (!cg_energy_slot_matches(st, side, inplayIdx, energyIdx, cardId))
      continue;
    if (match >= 0) return fallback;
    match = side;
  }
  return match >= 0 ? match : fallback;
}

int cg_energy_count(const GameState& st, int owner, int inplayIdx,
                    int energyIdx) {
  const InPlay* p = cg_inplay_at(st, owner, inplayIdx);
  if (!p) return 1;
  return std::max(1, provided_energy_units_for_card(*p, energyIdx, &st, owner));
}

bool cg_decode_tool_ref(int ref, int me, int& owner, int& inplayIdx,
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

int cg_infer_card_owner(const GameState& st, int fallback,
                        std::string_view area, int index, int cardId) {
  if (area == "STADIUM")
    return st.stadiumOwner >= 0 ? st.stadiumOwner : fallback;
  if (area != "ACTIVE" && area != "BENCH" && area != "PRIZE" &&
      area != "HAND" && area != "DISCARD" && area != "DECK" &&
      area != "LOOKING")
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
    } else if (area == "HAND") {
      if (index >= 0 && index < static_cast<int>(p.hand.size()))
        cid = p.hand[index];
    } else if (area == "DISCARD") {
      if (index >= 0 && index < static_cast<int>(p.discard.size()))
        cid = p.discard[index];
    } else if (area == "DECK") {
      if (index >= 0 && index < static_cast<int>(p.deck.size()))
        cid = p.deck[index];
    } else if (area == "LOOKING") {
      const int deck_index = static_cast<int>(p.deck.size()) - 1 - index;
      if (deck_index >= 0 && deck_index < static_cast<int>(p.deck.size()))
        cid = p.deck[deck_index];
    }
    if (cid <= 0 || (cardId > 0 && cid != cardId)) continue;
    if (match >= 0) return fallback;
    match = side;
  }
  return match >= 0 ? match : fallback;
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
      if (descriptor_equal_value(old, desc)) {
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

struct CgActionView {
  int ctx = -1;
  int min_count = 1;
  int max_count = 1;
  std::vector<Descriptor> descriptors;
};

CgActionView action_view_from_id_view(const GameState& st, ActionIdView ids) {
  CgActionView view;
  if (ids.pending) {
    view.ctx = st.has_pending() ? st.pending.context : -1;
    view.min_count = st.has_pending() ? st.pending.minCount : 1;
    view.max_count = st.has_pending() ? st.pending.maxCount : 1;
    view.descriptors = st.has_pending() ? st.pending.options
                                        : std::vector<Descriptor>{};
  } else {
    if (ids.descriptors) view.descriptors = *ids.descriptors;
  }
  const int n = std::clamp(ids.n, 0, ACTION_MAX_OPTIONS);
  if (static_cast<int>(view.descriptors.size()) > n) view.descriptors.resize(n);
  return view;
}

bool cg_all_descriptors_kind(const CgActionView& view, std::string_view kind) {
  if (view.descriptors.empty()) return false;
  for (const Descriptor& d : view.descriptors) {
    std::string_view cur = d.empty() ? std::string_view("END") : desc_str(d, 0);
    if (cur != kind) return false;
  }
  return true;
}

bool cg_any_descriptor_kind(const CgActionView& view, std::string_view kind) {
  for (const Descriptor& d : view.descriptors) {
    std::string_view cur = d.empty() ? std::string_view("END") : desc_str(d, 0);
    if (cur == kind) return true;
  }
  return false;
}

int cg_select_type_from_view(const CgActionView& view) {
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

int action_kind_code(std::string_view kind) {
  if (kind == "END") return 1;
  if (kind == "PLAY") return 2;
  if (kind == "ATTACH") return 3;
  if (kind == "EVOLVE") return 4;
  if (kind == "ABILITY") return 5;
  if (kind == "ATTACK") return 6;
  if (kind == "RETREAT") return 7;
  if (kind == "DISCARD") return 8;
  if (kind == "SETUP_ACTIVE") return 9;
  if (kind == "CARD") return 10;
  if (kind == "ENERGY") return 11;
  if (kind == "YES") return 12;
  if (kind == "NO") return 13;
  if (kind == "COUNT" || kind == "NUMBER") return 14;
  if (kind == "SKILL") return 15;
  if (kind == "SPECIAL_CONDITION") return 16;
  return 17;
}

int action_owner_pov(int owner, int me) {
  if (owner == me) return 0;
  if (owner == 1 - me) return 1;
  return -1;
}

int action_target_card_id(const GameState& st, int owner, std::string_view area,
                          int index) {
  if (owner < 0 || owner > 1) return 0;
  const Player& p = st.players[owner];
  if (area == "ACTIVE") return p.activeKnown ? p.active.id : -1;
  if (area == "BENCH")
    return index >= 0 && index < static_cast<int>(p.bench.size())
               ? p.bench[index].id
               : 0;
  if (area == "STADIUM") return st.stadium.empty() ? 0 : st.stadium[0];
  if (area == "PRIZE")
    return index >= 0 && index < static_cast<int>(p.prizes.size())
               ? p.prizes[index]
               : -1;
  return 0;
}

constexpr int ATTACK_DEFINITION_SLOTS_PER_CARD = 4;

int attack_definition_key_for_card(int cardId, int attackId) {
  if (cardId <= 0 || attackId <= 0) return 0;
  const CardInfo* card = find_card(cardId);
  if (!card) return 0;
  for (int slot = 0; slot < card->n_attacks; ++slot) {
    if (card->attacks[slot].id == attackId)
      return cardId * ATTACK_DEFINITION_SLOTS_PER_CARD + slot + 1;
  }
  return 0;
}

int resolved_attack_definition_key(const GameState& st, const Descriptor& d,
                                   int selectType) {
  const int attackId = desc_int(d, 1);
  if (attackId <= 0 || st.yourIndex < 0 || st.yourIndex > 1) return 0;

  // Copied-attack selection descriptors carry their precise source card in
  // the fourth atom.  Prefer it over reconstructing context from the board.
  if (d.size() >= 4) {
    const int explicitSourceCardId = desc_int(d, 3);
    if (const int key =
            attack_definition_key_for_card(explicitSourceCardId, attackId))
      return key;
  }

  const Player& me = st.players[st.yourIndex];
  const Player& opp = st.players[1 - st.yourIndex];

  if (selectType == CG_SELECT_ATTACK) {
    // Bespoke/VM copied-attack choices may keep the source only on the active
    // effect frame (for example Slowking's Seek Inspiration).
    if (!st.effectStack.empty()) {
      if (const int key = attack_definition_key_for_card(
              st.effectStack.back().sourceCardId, attackId))
        return key;
    }
    if (opp.activeKnown) {
      if (const int key =
              attack_definition_key_for_card(opp.active.id, attackId))
        return key;
    }
    if (me.activeKnown) {
      if (const int key =
              attack_definition_key_for_card(me.active.id, attackId))
        return key;
    }
    return 0;
  }

  if (!me.activeKnown) return 0;

  // Zoroark/Clefable replace their printed copy attack with the opponent's
  // Active attacks in the main-action list.
  if (me.active.id == 615 || me.active.id == 958) {
    if (opp.activeKnown) {
      if (const int key =
              attack_definition_key_for_card(opp.active.id, attackId))
        return key;
    }
  }

  // N's Zoroark ex offers attacks from eligible Benched N's Pokemon.  Attack
  // IDs are unique in this card database, so the matching bench occurrence is
  // the resolved source selected by the engine.
  if (me.active.id == 293) {
    for (const InPlay& benched : me.bench) {
      if (const int key =
              attack_definition_key_for_card(benched.id, attackId))
        return key;
    }
  }

  if (const int key = attack_definition_key_for_card(me.active.id, attackId))
    return key;

  // Relicanth's Memory Dive offers printed attacks from the Active's actual
  // pre-evolution chain.
  for (int preEvolutionId : me.active.preEvo) {
    if (const int key =
            attack_definition_key_for_card(preEvolutionId, attackId))
      return key;
  }

  // Lillie's Clefairy ex can receive Core Memory's Geobuster from the Tool.
  if (me.active.id == 1056 && attackId == 1556)
    return attack_definition_key_for_card(1180, attackId);
  return 0;
}

template <bool Compact, typename T>
void write_raw_reference(IdWriter<T> row, int value) {
  if constexpr (Compact) {
    row[ACTION_RAW_REF_LOW_COLUMN] =
        value & ((1 << ACTION_RAW_REF_SHIFT) - 1);
    row[ACTION_RAW_REF_HIGH_COLUMN] = value >> ACTION_RAW_REF_SHIFT;
  } else {
    row[ACTION_RAW_REF_LOW_COLUMN] = value;
  }
}

template <bool Compact, typename T>
void encode_action_id_option(const GameState& st, const Descriptor& d,
                             CgHandIndexResolver& hand, int selectType,
                             int actionIndex, IdWriter<T> row) {
  const int me = st.yourIndex;
  const std::string_view kind =
      d.empty() ? std::string_view("END") : desc_str(d, 0);
  row[0] = 1;
  row[2] = action_kind_code(kind);
  row[17] = actionIndex;
  if (kind == "END") {
    row[1] = CG_OPTION_END;
  } else if (kind == "PLAY") {
    row[1] = CG_OPTION_PLAY;
    row[3] = CG_AREA_HAND;
    row[4] = hand.index_for(d, desc_int(d, 1));
    row[5] = 0;
    row[8] = desc_int(d, 1);
  } else if (kind == "SETUP_ACTIVE") {
    row[1] = CG_OPTION_CARD;
    row[3] = CG_AREA_HAND;
    row[4] = hand.index_for(d, desc_int(d, 1));
    row[5] = 0;
    row[8] = desc_int(d, 1);
  } else if (kind == "ATTACH") {
    std::string_view targetArea = desc_str(d, 2);
    int targetIndex = desc_int(d, 3);
    row[1] = CG_OPTION_ATTACH;
    row[3] = CG_AREA_HAND;
    row[4] = hand.index_for(d, desc_int(d, 1));
    row[5] = 0;
    row[6] = cg_area_code(targetArea);
    row[7] = cg_area_index(targetArea, targetIndex);
    row[8] = desc_int(d, 1);
    row[9] = action_target_card_id(st, me, targetArea, targetIndex);
  } else if (kind == "EVOLVE") {
    std::string_view targetArea = desc_str(d, 2);
    int targetIndex = desc_int(d, 3);
    row[1] = CG_OPTION_EVOLVE;
    row[3] = CG_AREA_HAND;
    row[4] = hand.index_for(d, desc_int(d, 1));
    row[5] = 0;
    row[6] = cg_area_code(targetArea);
    row[7] = cg_area_index(targetArea, targetIndex);
    row[8] = desc_int(d, 1);
    row[9] = action_target_card_id(st, me, targetArea, targetIndex);
  } else if (kind == "ATTACK") {
    row[1] = CG_OPTION_ATTACK;
    row[10] = desc_int(d, 1);
    row[ACTION_ATTACK_DEFINITION_COLUMN] =
        resolved_attack_definition_key(st, d, selectType);
  } else if (kind == "RETREAT") {
    row[1] = CG_OPTION_RETREAT;
    row[3] = CG_AREA_ACTIVE;
    row[4] = 0;
    row[5] = 0;
    row[9] = action_target_card_id(st, me, "ACTIVE", 0);
  } else if (kind == "ABILITY") {
    std::string_view area = desc_str(d, 1);
    int index = desc_int(d, 2);
    row[1] = CG_OPTION_ABILITY;
    row[3] = cg_area_code(area);
    row[4] = cg_area_index(area, index);
    row[5] = 0;
    row[9] = action_target_card_id(st, me, area, index);
  } else if (kind == "DISCARD") {
    std::string_view area = desc_str(d, 1);
    int index = desc_int(d, 2);
    row[1] = CG_OPTION_DISCARD;
    row[3] = cg_area_code(area);
    row[4] = cg_area_index(area, index);
    row[5] = 0;
    row[9] = action_target_card_id(st, me, area, index);
  } else if (kind == "CARD") {
    std::string_view area = desc_str(d, 1);
    int rawIndex = desc_int(d, 2);
    int cardId = desc_int(d, d.empty() ? 0 : d.size() - 1);
    int owner = me;
    int inplayIdx = 0;
    int toolIdx = 0;
    if (cg_decode_tool_ref(rawIndex, me, owner, inplayIdx, toolIdx)) {
      row[1] = CG_OPTION_TOOL_CARD;
      row[3] = cg_area_code(area);
      row[4] = cg_area_index(area, inplayIdx);
      row[5] = action_owner_pov(owner, me);
      row[8] = cardId;
      row[9] = action_target_card_id(st, owner, area, inplayIdx);
      row[15] = toolIdx;
      write_raw_reference<Compact>(row, rawIndex);
    } else {
      owner = cg_infer_card_owner(st, me, area, rawIndex, cardId);
      row[1] = CG_OPTION_CARD;
      row[3] = cg_area_code(area);
      int semanticIndex = cg_area_index(area, rawIndex);
      if (area == "DECK" && owner >= 0 && owner < 2 && rawIndex >= 0 &&
          rawIndex < static_cast<int>(st.players[owner].deck.size())) {
        // Engine deck vectors are bottom-first, while observation deck slots
        // and model positions are top-first.  Keep the engine reference in
        // columns 18/19 and expose only this canonical top-distance here.
        semanticIndex = static_cast<int>(st.players[owner].deck.size()) - 1 -
                        rawIndex;
      }
      row[4] = semanticIndex;
      row[5] = action_owner_pov(owner, me);
      row[8] = cardId;
      row[9] = action_target_card_id(st, owner, area, rawIndex);
      write_raw_reference<Compact>(row, rawIndex);
    }
  } else if (kind == "ENERGY") {
    std::string_view area = desc_str(d, 1);
    int ref = desc_int(d, 2);
    if (ref >= 200000) ref -= 200000;
    int inplayIdx = cg_energy_ref_inplay(ref);
    int energyIdx = cg_energy_ref_index(ref);
    int cardId = desc_int(d, d.empty() ? 0 : d.size() - 1);
    int owner = cg_infer_energy_owner(st, me, inplayIdx, energyIdx, cardId);
    bool attachedCardSelect =
        selectType == CG_SELECT_ATTACHED_CARD ||
        selectType == CG_SELECT_CARD_OR_ATTACHED_CARD;
    row[1] = attachedCardSelect ? CG_OPTION_ENERGY_CARD : CG_OPTION_ENERGY;
    row[3] = cg_area_code(area);
    row[4] = cg_area_index(area, inplayIdx);
    row[5] = action_owner_pov(owner, me);
    row[8] = cardId;
    row[9] = action_target_card_id(st, owner, area, inplayIdx);
    row[13] = energyIdx;
    if (!attachedCardSelect)
      row[14] = cg_energy_count(st, owner, inplayIdx, energyIdx);
    write_raw_reference<Compact>(row, ref);
  } else if (kind == "YES") {
    row[1] = CG_OPTION_YES;
  } else if (kind == "NO") {
    row[1] = CG_OPTION_NO;
  } else if (kind == "COUNT" || kind == "NUMBER") {
    row[1] = CG_OPTION_NUMBER;
    row[11] = desc_int(d, d.empty() ? 0 : d.size() - 1);
  } else if (kind == "SKILL") {
    row[1] = CG_OPTION_SKILL;
    row[8] = desc_int(d, 1);
    row[12] = desc_int(d, 2);
  } else if (kind == "SPECIAL_CONDITION") {
    row[1] = CG_OPTION_SPECIAL_CONDITION;
    row[16] = desc_int(d, 1);
  } else {
    row[1] = CG_OPTION_END;
  }
}

}  // namespace

ActionIdView action_id_view(const std::vector<Descriptor>& descriptors) {
  return ActionIdView{false, static_cast<int>(descriptors.size()),
                      &descriptors};
}

template <typename T>
bool fill_observation_ids_impl(const GameState& st, T* ip_data, T* zn_data,
                               T* ct_data, T* stt_data, T* gl_data) {
  bool ok = true;
  auto ip = id_writer(ip_data, &ok);
  auto zn = id_writer(zn_data, &ok);
  auto ct = id_writer(ct_data, &ok);
  auto stt = id_writer(stt_data, &ok);
  auto gl = id_writer(gl_data, &ok);
  const int me = st.yourIndex;

  ip.clear(2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH);
  zn.clear(2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS);
  ct.clear(2 * 5);
  stt.clear(2 * 5);
  gl.clear(STATE_GLOBAL_WIDTH);

  auto inplay_row = [&](int pov, int slot) {
    return ip + (pov * STATE_INPLAY_SLOTS + slot) * STATE_INPLAY_WIDTH;
  };
  auto zone_row = [&](int pov, int zone) {
    return zn + (pov * STATE_ZONE_COUNT + zone) * STATE_ZONE_SLOTS;
  };
  auto encode_card_value = [](bool present, bool known, int cid) {
    return !present ? 0 : (known ? cid : -1);
  };
  auto turn_phase = [&](int effect_turn) {
    if (effect_turn < st.turn) return 0;
    const int delta = effect_turn - st.turn;
    return delta == 0 ? 1 : (delta == 1 ? 2 : 3);
  };
  auto bounded = [](int value, int maximum, bool& clipped) {
    if (value < 0) {
      clipped = true;
      return 0;
    }
    if (value > maximum) {
      clipped = true;
      return maximum;
    }
    return value;
  };
  auto damage_units = [&](int value, int maximum, bool& clipped) {
    if (value < 0) {
      clipped = true;
      return 0;
    }
    if (value % 10 != 0) clipped = true;
    return bounded(value / 10, maximum, clipped);
  };

  for (int pov = 0; pov < 2; ++pov) {
    const int owner = pov == 0 ? me : 1 - me;
    const Player& p = st.players[owner];
    for (int slot = 0; slot < STATE_INPLAY_SLOTS; ++slot) {
      const InPlay* pk = state_inplay_at_slot(p, slot);
      auto row = inplay_row(pov, slot);
      row[3] = slot == 0 ? AREA_ACTIVE : AREA_BENCH;
      row[4] = slot == 0 ? 0 : slot - 1;
      if (!pk) continue;
      bool known = slot != 0 || p.activeKnown;
      row[0] = 1;
      row[1] = known ? 1 : 0;
      row[2] = known ? pk->id : -1;
      if (known) {
        row[5] = pk->hp;
        row[6] = pk->maxHp;
      }
      int flags = 0;
      flags |= pk->appearThisTurn ? (1 << 0) : 0;
      flags |= pk->abilityUsedThisTurn ? (1 << 1) : 0;
      flags |= pk->movedToActiveThisTurn ? (1 << 2) : 0;
      flags |= pk->healedThisTurn ? (1 << 3) : 0;
      flags |= pk->noEvolveTurn == st.turn ? (1 << 4) : 0;
      flags |= pk->noRetreatTurn == st.turn ? (1 << 5) : 0;
      flags |= pk->noAttackTurn == st.turn ? (1 << 6) : 0;
      flags |= (pk->abilityUsedThisTurn || pk->reconstructedAbilityPrevUsed)
                   ? (1 << 7)
                   : 0;
      row[7] = flags;
      row[8] = pk->serial;
      row[9] = static_cast<int>(pk->energies.size());
      row[10] = static_cast<int>(pk->energyCardIds.size());
      row[11] = static_cast<int>(pk->tools.size());
      row[12] = static_cast<int>(pk->preEvo.size());
      row[13] = pk->lockTurn == st.turn ? pk->lockId : 0;
      row[14] = pk->activeLockId;

      // The first two words contain seven independent 2-bit turn phases each.
      // Values are emitted only while their phase is observable, so stale
      // lifecycle residue cannot leak into the model input.
      const int dmg_reduce_phase = turn_phase(pk->dmgReduceTurn);
      const int attack_cost_phase = turn_phase(pk->attackCostModTurn);
      const int retreat_cost_phase = turn_phase(pk->retreatCostModTurn);
      const int delayed_damage_phase = turn_phase(pk->delayedDamageTurn);
      const int delayed_ko_phase = turn_phase(pk->delayedKoTurn);
      const int prevent_damage_phase = turn_phase(pk->preventDmgTurn);
      const int prevent_effects_phase = turn_phase(pk->preventEffectsTurn);
      const int flip_fail_phase = turn_phase(pk->attackFlipFailTurn);
      const int no_weakness_phase = turn_phase(pk->noWeaknessTurn);
      const int take_more_phase = turn_phase(pk->takeMoreDamageTurn);
      const int next_attack_phase = turn_phase(pk->nextAttackBonusTurn);
      const int reactive_damage_phase =
          turn_phase(pk->damagedByAttackCountersTurn);
      const int reactive_equal_phase =
          turn_phase(pk->damagedByAttackEqualCountersTurn);
      const int energy_reactive_phase =
          turn_phase(pk->energyAttachCountersTurn);
      const int attack_reduce_phase = turn_phase(pk->attackDmgReduceTurn);
      const int attack_bonus_phase = turn_phase(pk->attackBonusTurn);

      row[STATE_INPLAY_EFFECT_PHASES_0_COLUMN] =
          dmg_reduce_phase | (attack_cost_phase << 2) |
          (retreat_cost_phase << 4) | (delayed_damage_phase << 6) |
          (delayed_ko_phase << 8) | (prevent_damage_phase << 10) |
          (prevent_effects_phase << 12);
      row[STATE_INPLAY_EFFECT_PHASES_1_COLUMN] =
          flip_fail_phase | (no_weakness_phase << 2) |
          (take_more_phase << 4) | (next_attack_phase << 6) |
          (reactive_damage_phase << 8) | (reactive_equal_phase << 10) |
          (energy_reactive_phase << 12);

      int history_age = 0;
      if (pk->damagedByAttackTurn >= 0 &&
          pk->damagedByAttackTurn <= st.turn) {
        const int age = st.turn - pk->damagedByAttackTurn;
        history_age = age == 0 ? 1 : (age == 1 ? 2 : 3);
      }
      bool values_clipped = false;
      int effect_meta = attack_reduce_phase | (attack_bonus_phase << 2);
      effect_meta |= pk->delayedKoPromoteBeforePrize && delayed_ko_phase
                         ? (1 << 4)
                         : 0;
      effect_meta |= pk->energyAttachCountersFromHandOnly &&
                             energy_reactive_phase
                         ? (1 << 5)
                         : 0;
      const int reactive_status =
          reactive_damage_phase && pk->damagedByAttackStatus >= 0
              ? bounded(pk->damagedByAttackStatus + 1, 7, values_clipped)
              : 0;
      effect_meta |= reactive_status << 6;
      effect_meta |= history_age << 9;
      effect_meta |= history_age && pk->damagedByAttackSide >= 0 &&
                             pk->damagedByAttackSide != owner
                         ? (1 << 11)
                         : 0;
      effect_meta |= slot == 0 && p.poisoned && p.poisonDamageCounters > 1
                         ? (1 << 12)
                         : 0;
      effect_meta |= next_attack_phase && pk->nextAttackSetBase >= 0
                         ? (1 << 13)
                         : 0;

      const int dmg_reduce = dmg_reduce_phase
                                 ? damage_units(pk->dmgReduce, 63,
                                                values_clipped)
                                 : 0;
      const int attack_damage_reduce =
          attack_reduce_phase
              ? damage_units(pk->attackDmgReduce, 31, values_clipped)
              : 0;
      const int attack_cost =
          attack_cost_phase
              ? bounded(pk->attackCostMod, 3, values_clipped)
              : 0;
      const int retreat_cost =
          retreat_cost_phase
              ? bounded(pk->retreatCostMod, 3, values_clipped)
              : 0;
      row[STATE_INPLAY_EFFECT_VALUES_0_COLUMN] =
          dmg_reduce | (attack_damage_reduce << 6) | (attack_cost << 11) |
          (retreat_cost << 13);

      const int prevent_damage_cond =
          prevent_damage_phase
              ? bounded(pk->preventDmgCond, 7, values_clipped)
              : 0;
      const int prevent_damage_value =
          prevent_damage_phase
              ? damage_units(pk->preventDmgValue, 63, values_clipped)
              : 0;
      const int take_more = take_more_phase
                                ? damage_units(pk->takeMoreDamage, 63,
                                               values_clipped)
                                : 0;
      row[STATE_INPLAY_EFFECT_VALUES_1_COLUMN] =
          prevent_damage_cond | (prevent_damage_value << 3) |
          (take_more << 9);

      const int prevent_effects_cond =
          prevent_effects_phase
              ? bounded(pk->preventEffectsCond, 7, values_clipped)
              : 0;
      const int prevent_effects_value =
          prevent_effects_phase
              ? damage_units(pk->preventEffectsValue, 63, values_clipped)
              : 0;
      const int delayed_counters =
          delayed_damage_phase
              ? bounded(pk->delayedDamageCounters, 63, values_clipped)
              : 0;
      row[STATE_INPLAY_EFFECT_VALUES_2_COLUMN] =
          prevent_effects_cond | (prevent_effects_value << 3) |
          (delayed_counters << 9);

      const int last_damage =
          history_age
              ? damage_units(pk->damagedByAttackAmount, 63, values_clipped)
              : 0;
      const int hp_before =
          history_age
              ? damage_units(pk->damagedByAttackBeforeHp, 63, values_clipped)
              : 0;
      const int reactive_counters =
          reactive_damage_phase
              ? bounded(pk->damagedByAttackCounters, 7, values_clipped)
              : 0;
      row[STATE_INPLAY_EFFECT_VALUES_3_COLUMN] =
          last_damage | (hp_before << 6) | (reactive_counters << 12);

      const int next_attack_id =
          next_attack_phase
              ? bounded(pk->nextAttackBonusId, 2047, values_clipped)
              : 0;
      const int next_attack_additive =
          next_attack_phase
              ? damage_units(pk->nextAttackBonus, 15, values_clipped)
              : 0;
      row[STATE_INPLAY_EFFECT_VALUES_4_COLUMN] =
          next_attack_id | (next_attack_additive << 11);

      const int next_attack_set_base =
          next_attack_phase && pk->nextAttackSetBase >= 0
              ? damage_units(pk->nextAttackSetBase, 62, values_clipped) + 1
              : 0;
      const int direct_attack_bonus =
          attack_bonus_phase
              ? damage_units(pk->attackBonus, 31, values_clipped)
              : 0;
      const int energy_reactive_counters =
          energy_reactive_phase
              ? bounded(pk->energyAttachCounters, 15, values_clipped)
              : 0;
      row[STATE_INPLAY_EFFECT_VALUES_5_COLUMN] =
          next_attack_set_base | (direct_attack_bonus << 6) |
          (energy_reactive_counters << 11);
      if (values_clipped) effect_meta |= 1 << 14;
      row[STATE_INPLAY_EFFECT_META_COLUMN] = effect_meta;

      int e_limit = std::min(
          STATE_MAX_ATTACHED_ENERGY,
          std::max(static_cast<int>(pk->energies.size()),
                   static_cast<int>(pk->energyCardIds.size())));
      for (int e = 0; e < e_limit; ++e) {
        row[16 + e] = e < static_cast<int>(pk->energyCardIds.size())
                          ? pk->energyCardIds[e]
                          : -1;
        row[32 + e] =
            e < static_cast<int>(pk->energies.size()) ? pk->energies[e] : -1;
      }

      int t_limit =
          std::min(STATE_MAX_TOOLS, static_cast<int>(pk->tools.size()));
      for (int t = 0; t < t_limit; ++t) row[48 + t] = pk->tools[t];

      int p_limit =
          std::min(STATE_MAX_PRE_EVOS, static_cast<int>(pk->preEvo.size()));
      for (int i = 0; i < p_limit; ++i) row[52 + i] = pk->preEvo[i];
    }
  }

  for (int pov = 0; pov < 2; ++pov) {
    const Player& p = st.players[pov == 0 ? me : 1 - me];
    const bool owner_pov = pov == 0;
    auto hand = zone_row(pov, 0);
    auto deck = zone_row(pov, 1);
    auto discard = zone_row(pov, 2);
    auto prizes = zone_row(pov, 3);
    for (int i = 0; i < STATE_ZONE_SLOTS; ++i) {
      if (i < p.handCount) {
        bool known = (owner_pov ? p.handKnown : p.handPublicKnown) &&
                     i < static_cast<int>(p.hand.size());
        int cid = known ? p.hand[i] : -1;
        if (!known && i < static_cast<int>(p.handKnownCards.size())) {
          known = true;
          cid = p.handKnownCards[i];
        }
        hand[i] = encode_card_value(true, known, cid);
      }

      if (i < p.deckCount) {
        int deck_idx = static_cast<int>(p.deck.size()) - 1 - i;
        bool has_ordered = deck_idx >= 0;
        bool known = (p.deckKnown && (!p.ownDeckInspected || owner_pov)) ||
                     (deck_idx >= 0 &&
                      deck_idx < static_cast<int>(p.deckKnownMask.size()) &&
                      p.deckKnownMask[deck_idx] &&
                      (!p.ownDeckInspected || owner_pov));
        int cid = has_ordered ? p.deck[deck_idx] : -1;
        if ((!has_ordered || !known) && (!p.ownDeckInspected || owner_pov) &&
            i < static_cast<int>(p.deckKnownCards.size())) {
          known = true;
          cid = p.deckKnownCards[i];
        }
        deck[i] = encode_card_value(true, known, cid);
      }

      if (i < static_cast<int>(p.discard.size())) {
        discard[i] = p.discard[i];
      }
    }

    std::vector<int> inferred_prizes;
    if (owner_pov && p.ownPrizesInferred) {
      inferred_prizes.assign(p.prizes.begin(), p.prizes.end());
      // Slot-known cards (especially face-up Prizes) keep their physical
      // positions and must be removed once from the unordered remainder.
      for (int slot = 0; slot < p.prizeCount; ++slot) {
        const bool slot_known = p.prizesKnown ||
            (slot < static_cast<int>(p.prizesKnownMask.size()) &&
             p.prizesKnownMask[slot]) ||
            (slot < static_cast<int>(p.prizeFaceUp.size()) &&
             p.prizeFaceUp[slot]);
        if (!slot_known || slot >= static_cast<int>(p.prizes.size())) continue;
        auto it = std::find(inferred_prizes.begin(), inferred_prizes.end(),
                            p.prizes[slot]);
        if (it != inferred_prizes.end()) inferred_prizes.erase(it);
      }
      std::sort(inferred_prizes.begin(), inferred_prizes.end());
    }
    int inferred_index = 0;
    for (int i = 0; i < STATE_PRIZE_SLOTS; ++i) {
      if (i >= p.prizeCount) continue;
      bool face_up = i < static_cast<int>(p.prizeFaceUp.size()) &&
                     p.prizeFaceUp[i];
      bool known = p.prizesKnown ||
                   (i < static_cast<int>(p.prizesKnownMask.size()) &&
                    p.prizesKnownMask[i]) || face_up;
      int cid = i < static_cast<int>(p.prizes.size()) ? p.prizes[i] : -1;
      if (!known && owner_pov && p.ownPrizesInferred &&
          inferred_index < static_cast<int>(inferred_prizes.size())) {
        // Exact deduction gives an unordered multiset, never a mapping from
        // identities to physical face-down Prize slots. Emit its canonical
        // sorted representation so compatible slot permutations agree.
        known = true;
        cid = inferred_prizes[inferred_index++];
      }
      if ((cid <= 0 || !known) &&
          i < static_cast<int>(p.prizesKnownCards.size())) {
        known = true;
        cid = p.prizesKnownCards[i];
      }
      prizes[i] = encode_card_value(true, known, cid);
      if (face_up) prizes[STATE_PRIZE_SLOTS + i] = 1;
    }

    ct[pov * 5 + 0] = p.handCount;
    ct[pov * 5 + 1] = p.deckCount;
    ct[pov * 5 + 2] = p.prizeCount;
    ct[pov * 5 + 3] = static_cast<int32_t>(p.discard.size());
    ct[pov * 5 + 4] = static_cast<int32_t>(p.bench.size());
    stt[pov * 5 + 0] = p.poisoned ? 1 : 0;
    stt[pov * 5 + 1] = p.burned ? 1 : 0;
    stt[pov * 5 + 2] = p.asleep ? 1 : 0;
    stt[pov * 5 + 3] = p.paralyzed ? 1 : 0;
    stt[pov * 5 + 4] = p.confused ? 1 : 0;
  }

  gl[0] = st.turn;
  gl[1] = st.turnActionCount;
  gl[2] = st.yourIndex;
  gl[3] = st.firstPlayer;
  gl[4] = st.result;
  gl[5] = st.supporterPlayed ? 1 : 0;
  gl[6] = st.stadiumPlayed ? 1 : 0;
  gl[7] = st.energyAttached ? 1 : 0;
  gl[8] = st.retreated ? 1 : 0;
  gl[9] = st.teamRocketSupporterPlayed ? 1 : 0;
  gl[10] = st.ancientSupporterPlayed ? 1 : 0;
  gl[11] = st.stadium.empty() ? 0 : st.stadium[0];
  gl[12] = st.stadiumOwner == me ? 0 : (st.stadiumOwner == 1 - me ? 1 : -1);
  gl[13] = st.stadiumAbilityUsed ? 1 : 0;
  gl[14] = st.has_pending() ? 1 : 0;
  gl[15] = st.has_pending() ? st.pending.context : -1;
  gl[16] = st.has_pending() ? st.pending.minCount : 0;
  gl[17] = st.has_pending() ? st.pending.maxCount : 0;
  gl[18] = st.has_pending() ? static_cast<int>(st.pending.options.size()) : 0;
  gl[19] = st.players[me].benchMax;
  gl[20] = st.players[1 - me].benchMax;
  gl[21] = st.players[me].handKnown ? 1 : 0;
  gl[22] = st.players[1 - me].handKnown ? 1 : 0;
  gl[23] = st.players[me].deckKnown ? 1 : 0;
  gl[24] = st.players[1 - me].deckKnown &&
                   !st.players[1 - me].ownDeckInspected
               ? 1
               : 0;
  gl[25] = (st.players[me].prizesKnown ||
            st.players[me].ownPrizesInferred) ? 1 : 0;
  gl[26] = st.players[1 - me].prizesKnown ? 1 : 0;

  // All qualifier and value fields share the one saturation bit in the
  // trailing values word. Keep clipping state across every packing lambda so
  // an out-of-domain qualifier can never silently alias a valid code.
  bool global_values_clipped = false;
  auto restriction_word = [&](int side, bool turn_flag) {
    const int item_phase = turn_phase(st.noItemTurn[side]);
    const int supporter_phase = turn_phase(st.noSupporterTurn[side]);
    const int evolve_phase = turn_phase(st.noEvolveTurn[side]);
    const int stadium_phase = turn_phase(st.noStadiumTurn[side]);
    const int low_energy_phase = turn_phase(st.noAttackEnergyLeTurn[side]);
    const int threshold =
        low_energy_phase
            ? bounded(st.noAttackEnergyLeThreshold[side] + 1, 7,
                      global_values_clipped)
            : 0;
    const bool shared_ability_used =
        st.abilityGroupUsedTurn[side][3] == st.turn;
    return item_phase | (supporter_phase << 2) | (evolve_phase << 4) |
           (stadium_phase << 6) | (low_energy_phase << 8) |
           (threshold << 10) | (shared_ability_used ? (1 << 13) : 0) |
           (turn_flag ? (1 << 14) : 0);
  };
  gl[STATE_GLOBAL_RESTRICTIONS_SELF_COLUMN] =
      restriction_word(me, st.lunarUsedThisTurn);
  gl[STATE_GLOBAL_RESTRICTIONS_OPP_COLUMN] =
      restriction_word(1 - me, st.canariPlayed);

  auto global_effect_word = [&](int side) {
    const int discard_phase = turn_phase(st.discardHandEndTurn[side]);
    const int team_phase = turn_phase(st.teamReduceTurn[side]);
    const int active_ex_phase = turn_phase(st.activeExDamageBuffTurn[side]);
    const int prize_phase = turn_phase(st.prizeBonusTurn[side]);
    const int hand_threshold =
        discard_phase
            ? bounded(st.discardHandEndThreshold[side], 7,
                      global_values_clipped)
            : 0;
    const int team_type =
        team_phase && st.teamReduceType[side] >= 0
            ? bounded(st.teamReduceType[side] + 1, 15,
                      global_values_clipped)
            : 0;
    return discard_phase | (team_phase << 2) | (active_ex_phase << 4) |
           (prize_phase << 6) | (hand_threshold << 8) |
           (team_type << 11);
  };
  gl[STATE_GLOBAL_EFFECTS_SELF_COLUMN] = global_effect_word(me);
  gl[STATE_GLOBAL_EFFECTS_OPP_COLUMN] = global_effect_word(1 - me);

  auto team_stacked = [&](int side) {
    if (!turn_phase(st.teamReduceTurn[side])) return 0;
    if (st.teamReduceAmount[side] != 30 && st.teamReduceAmount[side] != 60)
      global_values_clipped = true;
    return st.teamReduceAmount[side] > 30 ? 1 : 0;
  };
  auto active_ex_code = [&](int side) {
    if (!turn_phase(st.activeExDamageBuffTurn[side])) return 0;
    const int units =
        damage_units(st.activeExDamageBuffAmount[side], 6,
                     global_values_clipped);
    if (units < 3) {
      global_values_clipped = true;
      return 0;
    }
    return units - 3;
  };
  const int self_team_stacked = team_stacked(me);
  const int opp_team_stacked = team_stacked(1 - me);
  const int self_active_ex_code = active_ex_code(me);
  const int opp_active_ex_code = active_ex_code(1 - me);
  const int self_prize_amount =
      turn_phase(st.prizeBonusTurn[me])
          ? bounded(st.prizeBonusAmount[me], 3, global_values_clipped)
          : 0;
  const int opp_prize_amount =
      turn_phase(st.prizeBonusTurn[1 - me])
          ? bounded(st.prizeBonusAmount[1 - me], 3, global_values_clipped)
          : 0;
  const int self_prize_kind =
      turn_phase(st.prizeBonusTurn[me])
          ? bounded(st.prizeBonusKind[me], 1, global_values_clipped)
          : 0;
  const int opp_prize_kind =
      turn_phase(st.prizeBonusTurn[1 - me])
          ? bounded(st.prizeBonusKind[1 - me], 1, global_values_clipped)
          : 0;
  const int fighting_present = st.fightingBuff > 0 ? 1 : 0;
  if (st.fightingBuff != 0 && st.fightingBuff != 30)
    global_values_clipped = true;
  gl[STATE_GLOBAL_EFFECT_VALUES_COLUMN] =
      self_team_stacked | (opp_team_stacked << 1) |
      (self_active_ex_code << 2) | (opp_active_ex_code << 4) |
      (self_prize_amount << 6) | (opp_prize_amount << 8) |
      (self_prize_kind << 10) | (opp_prize_kind << 11) |
      (fighting_present << 12) | (st.tarragonPlayed ? (1 << 13) : 0) |
      (global_values_clipped ? (1 << 14) : 0);
  return ok;
}

void fill_observation_ids(const GameState& st, int32_t* in_play,
                          int32_t* zones, int32_t* player_counts,
                          int32_t* player_status, int32_t* global) {
  fill_observation_ids_impl(st, in_play, zones, player_counts, player_status,
                            global);
}

template <bool Compact, typename T>
bool fill_action_ids_impl(const GameState& st, ActionIdView ids, T* meta_data,
                          T* options_data, T* deck_data, uint8_t* mask) {
  bool ok = true;
  auto meta = id_writer(meta_data, &ok);
  auto options = id_writer(options_data, &ok);
  auto deck = id_writer(deck_data, &ok);
  CgActionView view = action_view_from_id_view(st, ids);
  const int selectType = cg_select_type_from_view(view);
  const int n_options =
      std::min(static_cast<int>(view.descriptors.size()), ACTION_MAX_OPTIONS);
  meta.clear(ACTION_META_WIDTH);
  options.clear(ACTION_MAX_OPTIONS * ACTION_OPTION_WIDTH);
  deck.clear(STATE_ZONE_SLOTS);
  if (mask) std::memset(mask, 0, ACTION_MAX_OPTIONS);

  meta[0] = view.ctx < 0 ? CG_CONTEXT_MAIN : view.ctx;
  meta[1] = selectType;
  meta[2] = view.min_count;
  meta[3] = view.max_count;
  meta[4] = n_options;
  meta[5] = 0;
  meta[6] = 0;
  meta[7] = 0;
  meta[8] = st.yourIndex;

  CgHandIndexResolver hand(st.players[st.yourIndex]);
  for (int i = 0; i < n_options; ++i) {
    encode_action_id_option<Compact>(
        st, view.descriptors[i], hand, selectType, i,
        options + i * ACTION_OPTION_WIDTH);
    if (mask) mask[i] = 1;
  }

  int max_deck_idx = -1;
  for (const Descriptor& d : view.descriptors) {
    if (d.size() >= 4 && desc_str(d, 0) == "CARD" &&
        desc_str(d, 1) == "DECK") {
      int idx = desc_int(d, 2);
      if (idx >= 0 && idx < STATE_ZONE_SLOTS) {
        deck[idx] = desc_int(d, 3);
        max_deck_idx = std::max(max_deck_idx, idx);
      }
    }
  }
  meta[5] = max_deck_idx + 1;
  return ok;
}

void fill_action_ids(const GameState& st, ActionIdView ids, int32_t* meta,
                     int32_t* options, int32_t* deck, uint8_t* mask) {
  fill_action_ids_impl<false>(st, ids, meta, options, deck, mask);
}

bool fill_observation_ids16(const GameState& st, int16_t* in_play,
                            int16_t* zones, int16_t* player_counts,
                            int16_t* player_status, int16_t* global) {
  return fill_observation_ids_impl(st, in_play, zones, player_counts,
                                   player_status, global);
}

bool fill_action_ids16(const GameState& st, ActionIdView view, int16_t* meta,
                       int16_t* options, int16_t* deck, uint8_t* mask) {
  return fill_action_ids_impl<true>(st, view, meta, options, deck, mask);
}

ObservationIds make_observation_ids(const GameState& st) {
  ObservationIds out;
  out.in_play.resize(2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH);
  out.zones.resize(2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS);
  out.player_counts.resize(2 * 5);
  out.player_status.resize(2 * 5);
  out.global.resize(STATE_GLOBAL_WIDTH);
  fill_observation_ids(st, out.in_play.data(), out.zones.data(),
                        out.player_counts.data(), out.player_status.data(),
                        out.global.data());
  return out;
}

ActionIds make_action_ids(const GameState& st, ActionIdView ids) {
  ActionIds out;
  out.meta.resize(ACTION_META_WIDTH);
  out.options.resize(ACTION_MAX_OPTIONS * ACTION_OPTION_WIDTH);
  out.deck.resize(STATE_ZONE_SLOTS);
  out.mask.resize(ACTION_MAX_OPTIONS);
  fill_action_ids(st, ids, out.meta.data(), out.options.data(),
                  out.deck.data(), out.mask.data());
  return out;
}

ActionIds make_action_ids(const GameState& st) {
  if (st.has_pending()) {
    return make_action_ids(st, ActionIdView{true, static_cast<int>(
        st.pending.options.size()), nullptr});
  }
  std::vector<Descriptor> descriptors = legal_main(st);
  return make_action_ids(st, action_id_view(descriptors));
}

IdTensorSpec id_tensor_spec() {
  return IdTensorSpec{
      {2, STATE_INPLAY_SLOTS, STATE_INPLAY_WIDTH},
      {2, STATE_ZONE_COUNT, STATE_ZONE_SLOTS},
      {2, 5},
      {2, 5},
      {STATE_GLOBAL_WIDTH},
      {ACTION_META_WIDTH},
      {ACTION_MAX_OPTIONS, ACTION_OPTION_WIDTH},
      {STATE_ZONE_SLOTS},
      {ACTION_MAX_OPTIONS},
  };
}

}  // namespace ptcg
