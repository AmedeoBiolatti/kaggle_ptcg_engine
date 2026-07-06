#include "ptcg/id_tensors.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <string_view>

namespace ptcg {
namespace {

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

void encode_action_id_option(const GameState& st, const Descriptor& d,
                             CgHandIndexResolver& hand, int selectType,
                             int actionIndex, int32_t* row) {
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
    row[4] = hand.index_for(d, desc_int(d, 1));
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
      row[18] = rawIndex;
    } else {
      owner = cg_infer_card_owner(st, me, area, rawIndex, cardId);
      row[1] = CG_OPTION_CARD;
      row[3] = cg_area_code(area);
      row[4] = cg_area_index(area, rawIndex);
      row[5] = action_owner_pov(owner, me);
      row[8] = cardId;
      row[9] = action_target_card_id(st, owner, area, rawIndex);
      row[18] = rawIndex;
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
    row[18] = ref;
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

void fill_observation_ids(const GameState& st, int32_t* ip, int32_t* zn,
                           int32_t* ct, int32_t* stt, int32_t* gl) {
  const int me = st.yourIndex;

  std::fill(ip, ip + 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH, 0);
  std::fill(zn, zn + 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS, 0);
  std::fill(ct, ct + 2 * 5, 0);
  std::fill(stt, stt + 2 * 5, 0);
  std::fill(gl, gl + STATE_GLOBAL_WIDTH, 0);

  auto inplay_row = [&](int pov, int slot) {
    return ip + (pov * STATE_INPLAY_SLOTS + slot) * STATE_INPLAY_WIDTH;
  };
  auto zone_row = [&](int pov, int zone) {
    return zn + (pov * STATE_ZONE_COUNT + zone) * STATE_ZONE_SLOTS;
  };
  auto encode_card_value = [](bool present, bool known, int cid) {
    return !present ? 0 : (known ? cid : -1);
  };

  for (int pov = 0; pov < 2; ++pov) {
    const Player& p = st.players[pov == 0 ? me : 1 - me];
    for (int slot = 0; slot < STATE_INPLAY_SLOTS; ++slot) {
      const InPlay* pk = state_inplay_at_slot(p, slot);
      int32_t* row = inplay_row(pov, slot);
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
      row[13] = pk->lockId;
      row[14] = pk->activeLockId;

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
    int32_t* hand = zone_row(pov, 0);
    int32_t* deck = zone_row(pov, 1);
    int32_t* discard = zone_row(pov, 2);
    int32_t* prizes = zone_row(pov, 3);
    for (int i = 0; i < STATE_ZONE_SLOTS; ++i) {
      if (i < p.handCount) {
        bool known = p.handKnown && i < static_cast<int>(p.hand.size());
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
        bool known = p.deckKnown ||
                     (deck_idx >= 0 &&
                      deck_idx < static_cast<int>(p.deckKnownMask.size()) &&
                      p.deckKnownMask[deck_idx]);
        int cid = has_ordered ? p.deck[deck_idx] : -1;
        if ((!has_ordered || !known) &&
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

    for (int i = 0; i < STATE_PRIZE_SLOTS; ++i) {
      if (i >= p.prizeCount) continue;
      bool face_up = i < static_cast<int>(p.prizeFaceUp.size()) &&
                     p.prizeFaceUp[i];
      bool known = p.prizesKnown ||
                   (i < static_cast<int>(p.prizesKnownMask.size()) &&
                    p.prizesKnownMask[i]) ||
                   face_up;
      int cid = i < static_cast<int>(p.prizes.size()) ? p.prizes[i] : -1;
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
  gl[24] = st.players[1 - me].deckKnown ? 1 : 0;
  gl[25] = st.players[me].prizesKnown ? 1 : 0;
  gl[26] = st.players[1 - me].prizesKnown ? 1 : 0;
}

void fill_action_ids(const GameState& st, ActionIdView ids, int32_t* meta,
                     int32_t* options, int32_t* deck, uint8_t* mask) {
  CgActionView view = action_view_from_id_view(st, ids);
  const int selectType = cg_select_type_from_view(view);
  const int n_options =
      std::min(static_cast<int>(view.descriptors.size()), ACTION_MAX_OPTIONS);
  std::fill(meta, meta + ACTION_META_WIDTH, 0);
  std::fill(options, options + ACTION_MAX_OPTIONS * ACTION_OPTION_WIDTH, 0);
  std::fill(deck, deck + STATE_ZONE_SLOTS, 0);
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
    encode_action_id_option(st, view.descriptors[i], hand, selectType, i,
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
