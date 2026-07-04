#include "ptcg/vector_env.hpp"

#include <algorithm>
#include <cstring>
#include <utility>

#include "ptcg/card_db.hpp"

namespace ptcg {

static inline uint64_t vector_rand(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

static constexpr int F_PER_PLAYER = 29;
static constexpr int F_GLOBAL = 3;
static constexpr int VECTOR_OBS_DIM = 2 * F_PER_PLAYER + F_GLOBAL;

int vector_obs_dim() { return VECTOR_OBS_DIM; }

static int encode_player(const Player& p, float* o) {
  int k = 0;
  bool a = p.activeKnown;
  o[k++] = a ? 1.f : 0.f;
  o[k++] = (a && p.active.maxHp > 0)
               ? static_cast<float>(p.active.hp) / p.active.maxHp
               : 0.f;
  o[k++] = a ? p.active.hp / 300.f : 0.f;
  o[k++] = a ? p.active.maxHp / 300.f : 0.f;
  o[k++] = a ? static_cast<int>(p.active.energies.size()) / 12.f : 0.f;
  int etype[11] = {0};
  if (a) {
    for (int e : p.active.energies) {
      if (e >= 0 && e < 11) ++etype[e];
    }
  }
  for (int t = 0; t < 10; ++t) o[k++] = etype[t] / 6.f;
  o[k++] = a ? static_cast<int>(p.active.tools.size()) / 2.f : 0.f;
  o[k++] = p.poisoned ? 1.f : 0.f;
  o[k++] = p.burned ? 1.f : 0.f;
  o[k++] = p.asleep ? 1.f : 0.f;
  o[k++] = p.paralyzed ? 1.f : 0.f;
  o[k++] = p.confused ? 1.f : 0.f;
  const CardInfo* c = a ? find_card(p.active.id) : nullptr;
  o[k++] = c ? c->retreat / 4.f : 0.f;
  o[k++] = (c && c->ex) ? 1.f : 0.f;
  o[k++] = (c && c->megaEx) ? 1.f : 0.f;
  o[k++] = static_cast<int>(p.bench.size()) / 5.f;
  int bench_hp = 0;
  for (const auto& b : p.bench) bench_hp += b.hp;
  o[k++] = bench_hp / 1500.f;
  o[k++] = p.prizeCount / 6.f;
  o[k++] = p.handCount / 15.f;
  o[k++] = p.deckCount / 60.f;
  return k;
}

void vector_encode_obs(const GameState& st, float* out) {
  int me = st.yourIndex;
  int k = 0;
  k += encode_player(st.players[me], out + k);
  k += encode_player(st.players[1 - me], out + k);
  out[k++] = st.turn / 50.f;
  out[k++] = (me == st.firstPlayer) ? 1.f : 0.f;
  out[k++] = st.stadium.empty() ? 0.f : 1.f;
}

static Action descriptor_to_action(const Descriptor& d) {
  Action a;
  std::string_view kind = atom_sv(d[0]);
  if (kind == "END") {
    a.kind = ACT_END;
  } else if (kind == "ATTACH") {
    a.kind = ACT_ATTACH;
    a.cardId = static_cast<int>(d[1].i);
    a.targetArea = atom_is(d[2], "ACTIVE") ? AREA_ACTIVE : AREA_BENCH;
    a.targetIndex = static_cast<int>(d[3].i);
  } else if (kind == "PLAY") {
    int cid = static_cast<int>(d[1].i);
    const CardInfo* ci = find_card(cid);
    a.kind = (ci && ci->cardType == POKEMON && ci->basic) ? ACT_PLAY_BASIC
                                                          : ACT_PLAY_TRAINER;
    a.cardId = cid;
  } else if (kind == "EVOLVE") {
    a.kind = ACT_EVOLVE;
    a.cardId = static_cast<int>(d[1].i);
    a.targetArea = atom_is(d[2], "ACTIVE") ? AREA_ACTIVE : AREA_BENCH;
    a.targetIndex = static_cast<int>(d[3].i);
  } else if (kind == "ATTACK") {
    a.kind = ACT_ATTACK;
    a.attackId = static_cast<int>(d[1].i);
  } else if (kind == "RETREAT") {
    a.kind = ACT_RETREAT;
  } else if (kind == "ABILITY") {
    a.kind = ACT_ABILITY;
    a.targetArea = atom_is(d[1], "ACTIVE") ? AREA_ACTIVE
                   : atom_is(d[1], "STADIUM") ? AREA_STADIUM
                                              : AREA_BENCH;
    a.targetIndex = static_cast<int>(d[2].i);
  } else if (kind == "DISCARD") {
    a.kind = ACT_DISCARD_INPLAY;
    a.targetArea = atom_is(d[1], "ACTIVE") ? AREA_ACTIVE : AREA_BENCH;
    a.targetIndex = static_cast<int>(d[2].i);
  } else if (kind == "SETUP_ACTIVE") {
    a.kind = ACT_SETUP_ACTIVE;
    a.cardId = static_cast<int>(d[1].i);
  }
  return a;
}

static void fill_mask(int n_actions, uint8_t* mask) {
  std::memset(mask, 0, VECTOR_MAX_ACTIONS);
  int n = std::min(std::max(n_actions, 0), VECTOR_MAX_ACTIONS);
  for (int i = 0; i < n; ++i) mask[i] = 1;
}

int vector_legal_mask(const GameState& st, uint8_t* mask) {
  int n = st.has_pending() ? static_cast<int>(st.pending.options.size())
                           : static_cast<int>(legal_main(st).size());
  fill_mask(n, mask);
  return std::min(std::max(n, 0), VECTOR_MAX_ACTIONS);
}

static const std::vector<int>& single_selection(int action) {
  thread_local std::vector<int> sel;
  sel.resize(1);
  sel[0] = action;
  return sel;
}

// Auto-resolve forced/multi-select sub-decisions to the next explicit
// decision. When `out` is non-null it captures that decision's options so
// callers never regenerate legal_main() for the same state. `out` may alias a
// caller-held option set: it is only written once the game state is final.
static void normalize_for_explicit_decision(GameState& st, uint64_t& rng,
                                            VectorOptions* out = nullptr) {
  while (st.result < 0) {
    if (st.has_pending()) {
      const PendingDecision& pd = st.pending;
      int n = static_cast<int>(pd.options.size());
      if (pd.maxCount == 1 && n >= 2) {
        if (out) {
          out->pending = true;
          out->count = n;
          out->descriptors.clear();
        }
        return;
      }

      std::vector<int> selection;
      int lo = pd.minCount;
      int hi = std::min(pd.maxCount, n);
      if (hi < lo) hi = lo;
      int k = lo + (hi > lo ? static_cast<int>(vector_rand(rng) % (hi - lo + 1))
                            : 0);
      selection.resize(n);
      for (int i = 0; i < n; ++i) selection[i] = i;
      for (int i = 0; i < k; ++i) {
        std::swap(selection[i],
                  selection[i + static_cast<int>(vector_rand(rng) % (n - i))]);
      }
      selection.resize(k);
      resolve(st, selection);
      continue;
    }

    std::vector<Descriptor> options = legal_main(st);
    if (options.size() >= 2 || options.empty()) {
      if (out) {
        out->pending = false;
        out->count = static_cast<int>(options.size());
        out->descriptors = std::move(options);
      }
      return;
    }
    apply(st, descriptor_to_action(options[0]));
  }
  if (out) {  // terminal (possibly with leftover pending)
    out->pending = st.has_pending();
    out->count =
        out->pending ? static_cast<int>(st.pending.options.size()) : 0;
    out->descriptors.clear();
  }
}

void vector_step_action(GameState& st, int action, uint64_t& rng) {
  if (st.has_pending()) {
    if (action >= 0 && action < static_cast<int>(st.pending.options.size())) {
      resolve(st, single_selection(action));
    }
  } else {
    std::vector<Descriptor> options = legal_main(st);
    if (action >= 0 && action < static_cast<int>(options.size())) {
      apply(st, descriptor_to_action(options[action]));
    }
  }
  normalize_for_explicit_decision(st, rng);
}

// vector_step_action against pre-captured options; bounds checks mirror it
// exactly (pending decisions validate against the live pending list).
static void step_action_cached(GameState& st, const VectorOptions& opts,
                               int action, uint64_t& rng, VectorOptions* next) {
  if (st.has_pending()) {
    if (action >= 0 && action < static_cast<int>(st.pending.options.size())) {
      resolve(st, single_selection(action));
    }
  } else if (action >= 0 &&
             action < static_cast<int>(opts.descriptors.size())) {
    apply(st, descriptor_to_action(opts.descriptors[action]));
  }
  normalize_for_explicit_decision(st, rng, next);  // `next` may alias `opts`
}

VectorEnv::VectorEnv(std::vector<int> deck0, std::vector<int> deck1, int n,
                     uint64_t seed)
    : deck0_(std::move(deck0)),
      deck1_(std::move(deck1)),
      rng_(seed ? seed : 0x9e3779b97f4a7c15ULL) {
  games_.resize(n);
  opts_.resize(n);
  reset_all();
}

void VectorEnv::reset(int i) {
  uint64_t s = vector_rand(rng_);
  games_[i] = new_game(deck0_, deck1_, s ? s : 1);
  normalize_for_explicit_decision(games_[i], rng_, &opts_[i]);
}

void VectorEnv::reset_all() {
  for (int i = 0; i < size(); ++i) reset(i);
}

void VectorEnv::observe(float* obs, uint8_t* mask, int32_t* player,
                        int32_t* result) const {
  int D = VECTOR_OBS_DIM;
  for (int i = 0; i < size(); ++i) {
    vector_encode_obs(games_[i], obs + i * D);
    fill_mask(opts_[i].count, mask + i * VECTOR_MAX_ACTIONS);
    player[i] = games_[i].yourIndex;
    result[i] = games_[i].result;
  }
}

void VectorEnv::step(const int* actions, float* obs, float* reward,
                     uint8_t* done, uint8_t* mask, int32_t* player,
                     int32_t* result) {
  int D = VECTOR_OBS_DIM;
  for (int i = 0; i < size(); ++i) {
    int actor = games_[i].yourIndex;
    step_action_cached(games_[i], opts_[i], actions[i], rng_, &opts_[i]);
    int r = games_[i].result;
    if (r >= 0) {
      reward[i] = (r == 2) ? 0.f : (r == actor ? 1.f : -1.f);
      done[i] = 1;
      reset(i);
    } else {
      reward[i] = 0.f;
      done[i] = 0;
    }
    vector_encode_obs(games_[i], obs + i * D);
    fill_mask(opts_[i].count, mask + i * VECTOR_MAX_ACTIONS);
    player[i] = games_[i].yourIndex;
    result[i] = games_[i].result;
  }
}

}  // namespace ptcg
