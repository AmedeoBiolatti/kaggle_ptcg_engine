#include "ptcg/rl.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "ptcg/card_db.hpp"
#include "ptcg/id_tensors.hpp"
#include "ptcg/types.hpp"

namespace ptcg {

namespace {

static_assert(ACTION_MAX_OPTIONS == RL_MAX_ACTIONS,
              "action ID tensor width must match action mask width");

ActionIdView action_id_view_from_options(const RlOptionSet& opts) {
  return ActionIdView{opts.pending, opts.n, &opts.descriptors};
}

}  // namespace

// --- local RNG (xorshift64, matching the engine's) ------------------------
static inline uint64_t rl_rand(uint64_t& s) {
  s ^= s << 13;
  s ^= s >> 7;
  s ^= s << 17;
  return s;
}

GameState rl_determinize_own_hidden(const GameState& state,
                                    const std::vector<int>& decklist,
                                    int perspective, uint64_t seed) {
  if (perspective != 0 && perspective != 1)
    throw std::invalid_argument("perspective must be 0 or 1");
  if (state.has_pending() || !state.effectStack.empty())
    throw std::invalid_argument(
        "own-hidden determinization requires a stable main-decision root");
  if (decklist.empty())
    throw std::invalid_argument("submitted decklist must not be empty");

  const Player& source = state.players[perspective];
  if (source.handCount != static_cast<int>(source.hand.size()))
    throw std::invalid_argument(
        "own hand identities must be exact before determinization");

  std::map<int, int> remaining;
  for (int id : decklist) {
    if (id <= 0)
      throw std::invalid_argument("decklist contains a non-positive card id");
    ++remaining[id];
  }
  auto consume = [&](int id, const char* context) {
    if (id <= 0) return;
    auto it = remaining.find(id);
    if (it == remaining.end() || it->second <= 0)
      throw std::invalid_argument(std::string("deck constraint conflict at ") +
                                  context + " for card " +
                                  std::to_string(id));
    if (--it->second == 0) remaining.erase(it);
  };
  auto consume_inplay = [&](const InPlay& pokemon) {
    consume(pokemon.id, "in-play Pokemon");
    for (int id : pokemon.energyCardIds) consume(id, "attached Energy");
    for (int id : pokemon.tools) consume(id, "attached Tool");
    for (int id : pokemon.preEvo) consume(id, "pre-evolution stack");
  };

  if (source.activePresent) {
    if (!source.activeKnown)
      throw std::invalid_argument("own Active identity is unknown");
    consume_inplay(source.active);
  }
  for (const InPlay& pokemon : source.bench) consume_inplay(pokemon);
  for (int id : source.hand) consume(id, "own hand");
  for (int id : source.discard) consume(id, "own discard");
  if (state.pendingTrainerOwner == perspective)
    consume(state.pendingTrainerDiscard, "resolving Trainer");
  if (state.stadiumOwner == perspective)
    for (int id : state.stadium) consume(id, "owned Stadium");

  int hidden_count = 0;
  for (const auto& [id, count] : remaining) {
    (void)id;
    hidden_count += count;
  }
  if (hidden_count != source.deckCount + source.prizeCount)
    throw std::invalid_argument(
        "decklist remainder does not match Deck plus Prize counts");

  std::vector<int> hidden_union;
  hidden_union.reserve(static_cast<std::size_t>(hidden_count));
  for (const auto& [id, count] : remaining)
    for (int i = 0; i < count; ++i) hidden_union.push_back(id);
  if (!source.deckPrizeKnownCards.empty()) {
    std::vector<int> known(source.deckPrizeKnownCards.begin(),
                           source.deckPrizeKnownCards.end());
    std::sort(known.begin(), known.end());
    if (known != hidden_union)
      throw std::invalid_argument(
          "known Deck/Prize union conflicts with submitted decklist");
  }

  std::vector<int> sampled_deck(static_cast<std::size_t>(source.deckCount), 0);
  std::vector<int> sampled_prizes(
      static_cast<std::size_t>(source.prizeCount), 0);
  auto extract = [&](int id, const char* context) {
    auto it = std::find(hidden_union.begin(), hidden_union.end(), id);
    if (it == hidden_union.end())
      throw std::invalid_argument(std::string("hidden constraint conflict at ") +
                                  context + " for card " +
                                  std::to_string(id));
    hidden_union.erase(it);
  };

  if (source.deckKnown &&
      static_cast<int>(source.deck.size()) != source.deckCount)
    throw std::invalid_argument("known Deck lacks exact ordered identities");
  for (int slot = 0; slot < source.deckCount; ++slot) {
    const bool known = source.deckKnown ||
        (slot < static_cast<int>(source.deckKnownMask.size()) &&
         source.deckKnownMask[slot]);
    if (!known) continue;
    if (slot >= static_cast<int>(source.deck.size()) ||
        source.deck[slot] <= 0)
      throw std::invalid_argument("known Deck slot lacks an identity");
    sampled_deck[slot] = source.deck[slot];
    extract(sampled_deck[slot], "known Deck slot");
  }

  if (source.prizesKnown &&
      static_cast<int>(source.prizes.size()) != source.prizeCount)
    throw std::invalid_argument("known Prizes lack exact identities");
  for (int slot = 0; slot < source.prizeCount; ++slot) {
    const bool face_up = slot < static_cast<int>(source.prizeFaceUp.size()) &&
                         source.prizeFaceUp[slot];
    const bool known = source.prizesKnown || face_up ||
        (slot < static_cast<int>(source.prizesKnownMask.size()) &&
         source.prizesKnownMask[slot]);
    if (!known) continue;
    if (slot >= static_cast<int>(source.prizes.size()) ||
        source.prizes[slot] <= 0)
      throw std::invalid_argument("known Prize slot lacks an identity");
    sampled_prizes[slot] = source.prizes[slot];
    extract(sampled_prizes[slot], "known Prize slot");
  }

  auto place_members = [&](const auto& members, std::vector<int>& zone,
                           const char* context) {
    for (int id : members) {
      auto slot = std::find(zone.begin(), zone.end(), 0);
      if (slot == zone.end())
        throw std::invalid_argument(std::string(context) +
                                    " exceeds available zone slots");
      extract(id, context);
      *slot = id;
    }
  };
  place_members(source.deckKnownCards, sampled_deck,
                "known unordered Deck membership");
  place_members(source.prizesKnownCards, sampled_prizes,
                "known unordered Prize membership");

  std::mt19937_64 sampling_rng(seed ? seed : 1);
  if (source.ownPrizesInferred) {
    if (static_cast<int>(source.prizes.size()) != source.prizeCount)
      throw std::invalid_argument(
          "inferred Prize multiset lacks exact owner identities");
    std::vector<int> inferred(source.prizes.begin(), source.prizes.end());
    for (int id : sampled_prizes) {
      if (id <= 0) continue;
      auto it = std::find(inferred.begin(), inferred.end(), id);
      if (it == inferred.end())
        throw std::invalid_argument(
            "known Prize slot conflicts with inferred multiset");
      inferred.erase(it);
    }
    std::sort(inferred.begin(), inferred.end());
    std::shuffle(inferred.begin(), inferred.end(), sampling_rng);
    place_members(inferred, sampled_prizes, "inferred Prize multiset");
  }

  std::shuffle(hidden_union.begin(), hidden_union.end(), sampling_rng);
  std::size_t cursor = 0;
  for (int& id : sampled_deck)
    if (id == 0) id = hidden_union.at(cursor++);
  for (int& id : sampled_prizes)
    if (id == 0) id = hidden_union.at(cursor++);
  if (cursor != hidden_union.size())
    throw std::logic_error("determinizer left unassigned hidden cards");

  GameState out = state;
  Player& sampled = out.players[perspective];
  sampled.deck.assign(sampled_deck.begin(), sampled_deck.end());
  sampled.prizes.assign(sampled_prizes.begin(), sampled_prizes.end());
  // Knowledge metadata describes the real information state, not the sampled
  // world's temporary completion, and is intentionally copied unchanged.
  out.rng = state.rng;
  return out;
}

// splitmix64 finalizer: derive well-separated per-game RNG streams from
// (master seed, game index). Each game's trajectory is then a pure function
// of (seed, index) — independent of batch order and thread count.
static inline uint64_t rl_mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

static std::vector<uint64_t> make_game_rngs(uint64_t seed, int n) {
  std::vector<uint64_t> rngs(static_cast<size_t>(std::max(n, 0)));
  uint64_t base = seed ? seed : 0x9e3779b97f4a7c15ULL;
  for (int i = 0; i < n; ++i) {
    uint64_t s = rl_mix64(base + static_cast<uint64_t>(i));
    rngs[static_cast<size_t>(i)] = s ? s : 1;
  }
  return rngs;
}

namespace {

// Persistent worker pool shared by the batch envs. One job (an env's batched
// call) runs at a time; workers pull game indices from a shared counter. This
// is safe because per-game work touches only that game's state/options/rng
// and disjoint output slices, plus read-only global tables (card db, card
// summaries, effect programs). Engine step code must not throw.
class EnvPool {
 public:
  static EnvPool& instance() {
    static EnvPool pool;
    return pool;
  }

  // threads: 0 = auto, 1 = serial, >1 = total participants including caller.
  void run(int n, int threads, const std::function<void(int)>& fn) {
    if (n <= 0) return;
    const int min_parallel =
        workers_.size() >= 16 ? kMinParallelHighCore : kMinParallel;
    if (threads == 1 || n < min_parallel || workers_.empty()) {
      for (int i = 0; i < n; ++i) fn(i);
      return;
    }
    const int participants =
        threads > 1 ? std::min(threads, static_cast<int>(workers_.size()) + 1)
                    : static_cast<int>(workers_.size()) + 1;
    std::lock_guard<std::mutex> job_lock(job_mutex_);  // one job at a time
    {
      std::lock_guard<std::mutex> lk(m_);
      fn_ = &fn;
      count_ = n;
      worker_limit_ = participants - 1;
      next_.store(0, std::memory_order_relaxed);
      in_flight_ = 1;  // the calling thread
      ++generation_;
    }
    cv_.notify_all();
    work(&fn, n);  // the calling thread participates
    std::unique_lock<std::mutex> lk(m_);
    if (--in_flight_ > 0)
      done_cv_.wait(lk, [&] { return in_flight_ == 0; });
    // All participants have exited work(): every index was claimed AND
    // finished, and no straggler can touch next_ after this point.
    fn_ = nullptr;
  }

 private:
  // Waking a full machine's worker set costs more than it saves for small
  // batches. Batch 256 comfortably amortizes the synchronization overhead;
  // batch 128 does not on high-core-count CPUs.
  static constexpr int kMinParallel = 128;
  static constexpr int kMinParallelHighCore = 192;

  EnvPool() {
    unsigned hw = std::thread::hardware_concurrency();
    int n_workers = static_cast<int>(hw > 1 ? hw - 1 : 0);
    if (n_workers > 31) n_workers = 31;
    for (int t = 0; t < n_workers; ++t)
      workers_.emplace_back([this, t] { worker_loop(t); });
  }

  ~EnvPool() {
    {
      std::lock_guard<std::mutex> lk(m_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  void worker_loop(int worker_id) {
    uint64_t seen = 0;
    for (;;) {
      const std::function<void(int)>* fn = nullptr;
      int count = 0;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return generation_ != seen || stop_; });
        if (stop_) return;
        seen = generation_;
        fn = fn_;  // may be null if this job already fully completed
        count = count_;
        if (fn && worker_id < worker_limit_)
          ++in_flight_;
        else
          fn = nullptr;
      }
      if (!fn) continue;
      work(fn, count);
      {
        std::lock_guard<std::mutex> lk(m_);
        if (--in_flight_ == 0) done_cv_.notify_all();
      }
    }
  }

  void work(const std::function<void(int)>* fn, int count) {
    for (;;) {
      int i = next_.fetch_add(1, std::memory_order_relaxed);
      if (i >= count) return;
      (*fn)(i);
    }
  }

  std::mutex job_mutex_;
  std::mutex m_;
  std::condition_variable cv_;
  std::condition_variable done_cv_;
  std::vector<std::thread> workers_;
  const std::function<void(int)>* fn_ = nullptr;
  int count_ = 0;
  int in_flight_ = 0;
  int worker_limit_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
  std::atomic<int> next_{0};
};

}  // namespace

// --- observation encoder --------------------------------------------------
static constexpr int F_PER_PLAYER = 29;
static constexpr int F_GLOBAL = 3;
static constexpr int RL_OBS_DIM = 2 * F_PER_PLAYER + F_GLOBAL;  // 61

int rl_obs_dim() { return RL_OBS_DIM; }

static int encode_player(const Player& p, float* o) {
  int k = 0;
  bool a = p.activeKnown;
  o[k++] = a ? 1.f : 0.f;
  o[k++] = (a && p.active.maxHp > 0)
               ? static_cast<float>(p.active.hp) / p.active.maxHp : 0.f;
  o[k++] = a ? p.active.hp / 300.f : 0.f;
  o[k++] = a ? p.active.maxHp / 300.f : 0.f;
  o[k++] = a ? static_cast<int>(p.active.energies.size()) / 12.f : 0.f;
  int etype[11] = {0};
  if (a)
    for (int e : p.active.energies)
      if (e >= 0 && e < 11) ++etype[e];
  for (int t = 0; t < 10; ++t) o[k++] = etype[t] / 6.f;  // 10 type buckets
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
  int benchHp = 0;
  for (const auto& b : p.bench) benchHp += b.hp;
  o[k++] = benchHp / 1500.f;
  o[k++] = p.prizeCount / 6.f;
  o[k++] = p.handCount / 15.f;
  o[k++] = p.deckCount / 60.f;
  return k;  // == F_PER_PLAYER
}

void rl_encode_obs(const GameState& st, float* out) {
  int me = st.yourIndex;
  int k = 0;
  k += encode_player(st.players[me], out + k);
  k += encode_player(st.players[1 - me], out + k);
  out[k++] = st.turn / 50.f;
  out[k++] = (me == st.firstPlayer) ? 1.f : 0.f;
  out[k++] = st.stadium.empty() ? 0.f : 1.f;
  // k == RL_OBS_DIM
}

float rl_heuristic_value(const GameState& st) {
  int me = st.yourIndex;
  const Player& mine = st.players[me];
  const Player& opp = st.players[1 - me];
  float v = static_cast<float>(opp.prizeCount - mine.prizeCount) / 6.0f;
  auto active_hp_term = [](const Player& p) {
    if (p.activeKnown && p.active.maxHp > 0)
      return static_cast<float>(p.active.hp) / static_cast<float>(p.active.maxHp);
    return 0.0f;
  };
  v += 0.1f * active_hp_term(mine);
  v -= 0.1f * active_hp_term(opp);
  return std::clamp(v, -1.0f, 1.0f);
}

// --- options / action routing ---------------------------------------------
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
                   : atom_is(d[1], "STADIUM") ? AREA_STADIUM : AREA_BENCH;
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

RlOptionSet rl_options_from_descriptors(std::vector<Descriptor> descriptors) {
  RlOptionSet out;
  out.pending = false;
  out.n = std::min(static_cast<int>(descriptors.size()), RL_MAX_ACTIONS);
  if (static_cast<int>(descriptors.size()) > out.n)
    descriptors.resize(out.n);
  out.descriptors = std::move(descriptors);
  return out;
}

RlOptionSet rl_options(const GameState& st) {
  RlOptionSet out;
  if (st.has_pending()) {
    out.pending = true;
    out.n = std::min(static_cast<int>(st.pending.options.size()), RL_MAX_ACTIONS);
    return out;
  }
  return rl_options_from_descriptors(legal_main(st));
}

// A random valid selection for an auto-resolved sub-decision (forced / multi).
static void random_subset_into(const PendingDecision& pd, uint64_t& rng,
                               std::vector<int>& out) {
  out.clear();
  int n = static_cast<int>(pd.options.size());
  if (n == 0) return;
  int lo = pd.minCount, hi = std::min(pd.maxCount, n);
  if (hi < lo) hi = lo;
  int k = lo + (hi > lo ? static_cast<int>(rl_rand(rng) % (hi - lo + 1)) : 0);
  out.resize(n);
  for (int i = 0; i < n; ++i) out[i] = i;
  for (int i = 0; i < k; ++i)  // partial Fisher-Yates
    std::swap(out[i], out[i + static_cast<int>(rl_rand(rng) % (n - i))]);
  out.resize(k);
}

static std::vector<int> random_subset(const PendingDecision& pd, uint64_t& rng) {
  std::vector<int> out;
  random_subset_into(pd, rng, out);
  return out;
}

static const std::vector<int>& single_selection(int action) {
  thread_local std::vector<int> sel;
  sel.resize(1);
  sel[0] = action;
  return sel;
}

bool rl_is_agent_decision(const GameState& st) {
  if (st.result >= 0) return false;
  if (st.has_pending())
    return st.pending.maxCount == 1 &&
           static_cast<int>(st.pending.options.size()) >= 2;
  return static_cast<int>(legal_main(st).size()) >= 2;
}

static int fill_legal_mask(int n_actions, uint8_t* mask);

int rl_legal_mask(const GameState& st, uint8_t* mask) {
  int n = st.has_pending() ? static_cast<int>(st.pending.options.size())
                           : static_cast<int>(legal_main(st).size());
  return fill_legal_mask(n, mask);
}

static std::string_view desc_str(const Descriptor& d, size_t i) {
  return (i < d.size() && d[i].is_str) ? atom_sv(d[i]) : std::string_view();
}

static int desc_int(const Descriptor& d, size_t i, int fallback = 0) {
  return (i < d.size() && !d[i].is_str && !d[i].is_none)
             ? static_cast<int>(d[i].i)
             : fallback;
}

static int ppo_area_code(std::string_view area) {
  if (area == "ACTIVE") return 0;
  if (area == "BENCH") return 1;
  if (area == "HAND") return 2;
  if (area == "DISCARD") return 3;
  if (area == "STADIUM") return 4;
  if (area == "DECK") return 5;
  if (area == "PRIZES") return 6;
  return -1;
}

static int ppo_kind_code(std::string_view kind) {
  if (kind == "END") return 0;
  if (kind == "PLAY") return 1;
  if (kind == "ATTACH") return 2;
  if (kind == "EVOLVE") return 3;
  if (kind == "ABILITY") return 4;
  if (kind == "ATTACK") return 5;
  if (kind == "RETREAT") return 6;
  if (kind == "DISCARD") return 7;
  if (kind == "SETUP_ACTIVE") return 8;
  if (kind == "CARD") return 9;
  if (kind == "ENERGY") return 10;
  if (kind == "YES") return 11;
  if (kind == "NO") return 12;
  if (kind == "COUNT") return 13;
  return 14;
}

struct CachedCardSummary {
  std::array<float, 6> summary{};
  std::array<float, PPO_CARD_FEAT_DIM> features{};
};

namespace card_feat {
constexpr int kEnergyTypes = TEAM_ROCKET + 1;

enum Index {
  PRESENT = 0,
  CARD_ID = 1,
  CARD_TYPE_BASE = 2,
  HP = 9,
  RETREAT = 10,
  BASIC = 11,
  STAGE1 = 12,
  STAGE2 = 13,
  EX = 14,
  MEGA_EX = 15,
  TERA = 16,
  ACE_SPEC = 17,
  HAS_ABILITY = 18,
  FREE_RETREAT = 19,
  ATTACK_COUNT = 20,
  MAX_DAMAGE = 21,
  MIN_COST = 22,
  ENERGY_TYPE = 23,
  WEAKNESS = 24,
  RESISTANCE = 25,
  OWNER_SELF = 26,
  OWNER_OPP = 27,
  AREA = 28,
  INDEX = 29,
  HP_RATIO = 30,
  ATTACHED_ENERGY_COUNT = 31,
  ATTACK_COST_TYPE_BASE = 32,
  BEST_DAMAGE_BUDGET_BASE = 44,
  MIN_COLORED_COST = 48,
  MAX_COLORED_COST = 49,
  MEAN_DAMAGE = 50,
  DAMAGE_ATTACK_COUNT = 51,
  ATTACHED_ENERGY_TYPE_BASE = 52,
  READY_ATTACK_COUNT = 64,
  BEST_READY_DAMAGE = 65,
  MIN_MISSING_ATTACK_COST = 66,
  BEST_DAMAGE_MISSING_COST = 67,
  TOOL_COUNT = 68,
  PRE_EVO_COUNT = 69,
  DAMAGE_COUNTERS = 70,
  NO_ATTACK = 71,
  NO_RETREAT = 72,
  PREVENT_DAMAGE = 73,
  TAKE_MORE_DAMAGE = 74,
  ATTACK_DAMAGE_DELTA = 75,
  DAMAGE_REDUCTION = 76,
  ABILITY_USED = 77,
  MOVED_TO_ACTIVE = 78,
  HEALED_THIS_TURN = 79,
};

static_assert(ATTACHED_ENERGY_TYPE_BASE + kEnergyTypes <= READY_ATTACK_COUNT);
static_assert(HEALED_THIS_TURN < PPO_CARD_FEAT_DIM);
}  // namespace card_feat

static int energy_feature_index(int energy_type) {
  return std::clamp(energy_type, 0, card_feat::kEnergyTypes - 1);
}

static float norm_card_id(int cid) {
  return std::clamp(cid / 2000.0f, 0.0f, 1.0f);
}

static float norm_damage(int damage) {
  return std::clamp(damage / 350.0f, 0.0f, 1.0f);
}

static float norm_energy_count(int count, int denom = 12) {
  return std::clamp(static_cast<float>(count) / static_cast<float>(denom),
                    0.0f, 1.0f);
}

static float norm_optional_energy_type(int energy_type) {
  if (energy_type < 0 || energy_type > TEAM_ROCKET) return 0.0f;
  return static_cast<float>(energy_type + 1) /
         static_cast<float>(card_feat::kEnergyTypes);
}

static int colored_attack_cost(const AttackInfo& at) {
  int colored = 0;
  for (int i = 0; i < at.n_cost; ++i)
    if (at.cost[i] != COLORLESS) ++colored;
  return colored;
}

static CachedCardSummary build_cached_card_summary(int cid) {
  CachedCardSummary out;
  if (cid <= 0) return out;

  out.summary[0] = norm_card_id(cid);
  out.features[card_feat::PRESENT] = 1.0f;  // known/present
  out.features[card_feat::CARD_ID] = out.summary[0];

  const CardInfo* c = find_card(cid);
  if (!c) return out;

  if (c->cardType == POKEMON) out.summary[1] = 1.0f;
  if (c->cardType == ITEM || c->cardType == TOOL ||
      c->cardType == SUPPORTER || c->cardType == STADIUM) {
    out.summary[2] = 1.0f;
  }
  if (c->cardType == BASIC_ENERGY || c->cardType == SPECIAL_ENERGY) {
    out.summary[3] = 1.0f;
  }
  if (c->ex) out.summary[4] = 1.0f;
  if (c->megaEx) out.summary[5] = 1.0f;

  out.features[card_feat::CARD_TYPE_BASE + std::min(std::max(c->cardType, 0), 6)] = 1.0f;
  out.features[card_feat::HP] = c->hp / 400.0f;
  out.features[card_feat::RETREAT] = std::clamp(c->retreat / 5.0f, 0.0f, 1.0f);
  out.features[card_feat::BASIC] = c->basic ? 1.0f : 0.0f;
  out.features[card_feat::STAGE1] = c->stage1 ? 1.0f : 0.0f;
  out.features[card_feat::STAGE2] = c->stage2 ? 1.0f : 0.0f;
  out.features[card_feat::EX] = c->ex ? 1.0f : 0.0f;
  out.features[card_feat::MEGA_EX] = c->megaEx ? 1.0f : 0.0f;
  out.features[card_feat::TERA] = c->tera ? 1.0f : 0.0f;
  out.features[card_feat::ACE_SPEC] = c->aceSpec ? 1.0f : 0.0f;
  out.features[card_feat::HAS_ABILITY] = c->hasAbility ? 1.0f : 0.0f;
  out.features[card_feat::FREE_RETREAT] = c->freeRetreat ? 1.0f : 0.0f;
  out.features[card_feat::ATTACK_COUNT] = std::clamp(c->n_attacks / 4.0f, 0.0f, 1.0f);
  int max_damage = 0;
  int min_cost = 99;
  int min_colored = 99;
  int max_colored = 0;
  int total_damage = 0;
  int damage_attacks = 0;
  std::array<int, 4> best_damage_by_budget{};
  for (int i = 0; i < c->n_attacks; ++i) {
    const AttackInfo& at = c->attacks[i];
    max_damage = std::max(max_damage, at.damage);
    min_cost = std::min(min_cost, at.n_cost);
    int colored = colored_attack_cost(at);
    min_colored = std::min(min_colored, colored);
    max_colored = std::max(max_colored, colored);
    total_damage += at.damage;
    if (at.damage > 0) ++damage_attacks;
    int budget_index = std::min(std::max(at.n_cost, 1), 4) - 1;
    best_damage_by_budget[budget_index] =
        std::max(best_damage_by_budget[budget_index], at.damage);
    for (int j = 0; j < at.n_cost; ++j) {
      int energy_idx = energy_feature_index(at.cost[j]);
      int offset = card_feat::ATTACK_COST_TYPE_BASE + energy_idx;
      out.features[offset] = std::min(1.0f, out.features[offset] + 0.125f);
    }
  }
  if (min_cost == 99) min_cost = 0;
  if (min_colored == 99) min_colored = 0;
  int cumulative_best = 0;
  for (int i = 0; i < 4; ++i) {
    cumulative_best = std::max(cumulative_best, best_damage_by_budget[i]);
    out.features[card_feat::BEST_DAMAGE_BUDGET_BASE + i] =
        norm_damage(cumulative_best);
  }
  out.features[card_feat::MAX_DAMAGE] = norm_damage(max_damage);
  out.features[card_feat::MIN_COST] = std::clamp(min_cost / 5.0f, 0.0f, 1.0f);
  out.features[card_feat::MIN_COLORED_COST] =
      std::clamp(min_colored / 5.0f, 0.0f, 1.0f);
  out.features[card_feat::MAX_COLORED_COST] =
      std::clamp(max_colored / 5.0f, 0.0f, 1.0f);
  out.features[card_feat::MEAN_DAMAGE] =
      c->n_attacks > 0 ? norm_damage(total_damage / c->n_attacks) : 0.0f;
  out.features[card_feat::DAMAGE_ATTACK_COUNT] =
      std::clamp(damage_attacks / 4.0f, 0.0f, 1.0f);
  out.features[card_feat::ENERGY_TYPE] = norm_optional_energy_type(c->energyType);
  out.features[card_feat::WEAKNESS] = norm_optional_energy_type(c->weakness);
  out.features[card_feat::RESISTANCE] = norm_optional_energy_type(c->resistance);
  return out;
}

static const CachedCardSummary& cached_card_summary(int cid) {
  // Prebuilt flat table (covers card ids and attack ids fed in via action
  // features). Immutable after the magic-static init, so it is safe to read
  // concurrently once the bindings release the GIL. Ids beyond the table are
  // built into a thread_local scratch (matches the old lazy-map behavior).
  static constexpr int kCacheSize = 8192;
  static const CachedCardSummary empty;
  static const std::vector<CachedCardSummary> cache = [] {
    std::vector<CachedCardSummary> v(kCacheSize);
    for (int i = 1; i < kCacheSize; ++i) v[i] = build_cached_card_summary(i);
    return v;
  }();
  if (cid <= 0) return empty;
  if (cid < kCacheSize) return cache[cid];
  thread_local CachedCardSummary overflow;
  overflow = build_cached_card_summary(cid);
  return overflow;
}

static int ppo_target_card_id(const GameState& st, int me,
                              std::string_view area, int index) {
  const Player& p = st.players[me];
  if (area == "ACTIVE") return p.activeKnown ? p.active.id : 0;
  if (area == "BENCH")
    return (index >= 0 && index < static_cast<int>(p.bench.size()))
               ? p.bench[index].id
               : 0;
  if (area == "STADIUM") return st.stadium.empty() ? 0 : st.stadium[0];
  return 0;
}

static void add_card_summary(float* f, int base, int cid) {
  if (cid <= 0) return;
  const auto& s = cached_card_summary(cid).summary;
  for (int i = 0; i < 6; ++i) f[base + i] = s[i];
}

static int important_card_code(int cid) {
  switch (cid) {
    case 6: return 0;
    case 673: return 1;
    case 674: return 2;
    case 675: return 3;
    case 676: return 4;
    case 677: return 5;
    case 678: return 6;
    case 1102: return 7;
    case 1123: return 8;
    case 1141: return 9;
    case 1142: return 10;
    case 1152: return 11;
    case 1159: return 12;
    case 1182: return 13;
    case 1192: return 14;
    case 1227: return 15;
    case 1252: return 16;
    case 1267: return 17;
    case 344: return 18;
    case 345: return 19;
    case 721: return 20;
    case 722: return 21;
    case 723: return 22;
    default: return -1;
  }
}

static void add_important_card_onehot(float* f, int base, int cid) {
  int k = important_card_code(cid);
  if (k >= 0) f[base + k] = 1.0f;
}

static int pending_min_count(const GameState& st) {
  return st.has_pending() ? st.pending.minCount : 1;
}

static int pending_max_count(const GameState& st) {
  return st.has_pending() ? st.pending.maxCount : 1;
}

static void encode_action_feature(const GameState& st, const Descriptor& d,
                                  int ctx, int action_index, float* f) {
  std::fill(f, f + PPO_ACTION_FEAT_DIM, 0.0f);
  if (d.empty() || !d[0].is_str) return;
  std::string_view kind = atom_sv(d[0]);
  f[std::min(ppo_kind_code(kind), 15)] = 1.0f;
  f[16] = ctx >= 0 ? 1.0f : 0.0f;
  f[17] = ctx >= 0 ? std::clamp(ctx / 64.0f, 0.0f, 1.0f) : 0.0f;

  int source = 0;
  int target = 0;
  int attack = 0;
  std::string_view area;
  int index = 0;
  int player = st.yourIndex;
  const int me = st.yourIndex;
  if (kind == "PLAY" || kind == "SETUP_ACTIVE") {
    source = desc_int(d, 1);
  } else if (kind == "ATTACH" || kind == "EVOLVE") {
    source = desc_int(d, 1);
    area = desc_str(d, 2);
    index = desc_int(d, 3);
    target = ppo_target_card_id(st, me, area, index);
  } else if (kind == "ABILITY" || kind == "DISCARD") {
    area = desc_str(d, 1);
    index = desc_int(d, 2);
    target = ppo_target_card_id(st, me, area, index);
  } else if (kind == "RETREAT") {
    area = "ACTIVE";
    index = 0;
    target = st.players[me].activeKnown ? st.players[me].active.id : 0;
  } else if (kind == "CARD" || kind == "ENERGY") {
    area = desc_str(d, 1);
    index = desc_int(d, 2);
    player = desc_int(d, 3, me);
    if (!d.empty() && !d.back().is_str && !d.back().is_none)
      target = static_cast<int>(d.back().i);
  } else if (kind == "ATTACK" || kind == "COUNT") {
    source = desc_int(d, 1);
    attack = source;
  }

  int ac = ppo_area_code(area);
  if (ac >= 0) f[18 + std::min(ac, 6)] = 1.0f;
  f[25] = std::clamp(index / 7.0f, 0.0f, 1.0f);
  add_card_summary(f, 26, source ? source : target);
  if (PPO_ACTION_FEAT_DIM <= 32) return;

  f[32] = std::clamp(source / 2000.0f, 0.0f, 1.0f);
  f[33] = std::clamp(target / 2000.0f, 0.0f, 1.0f);
  f[34] = std::clamp(attack / 2200.0f, 0.0f, 1.0f);
  f[35] = std::clamp(index / 15.0f, 0.0f, 1.0f);
  f[36] = player == me ? 1.0f : 0.0f;
  f[37] = player == 1 - me ? 1.0f : 0.0f;
  f[38] = std::clamp(pending_min_count(st) / 6.0f, 0.0f, 1.0f);
  f[39] = std::clamp(pending_max_count(st) / 6.0f, 0.0f, 1.0f);
  add_card_summary(f, 40, source);
  add_card_summary(f, 46, target);
  add_important_card_onehot(f, 52, source);
  add_important_card_onehot(f, 75, target);
  f[98] = std::clamp(action_index / 63.0f, 0.0f, 1.0f);
  f[99 + std::min(std::max(action_index, 0), 28)] = 1.0f;
}

void rl_action_features(const GameState& st, float* out) {
  std::fill(out, out + RL_MAX_ACTIONS * PPO_ACTION_FEAT_DIM, 0.0f);
  int ctx = st.has_pending() ? st.pending.context : -1;
  const std::vector<Descriptor> desc =
      st.has_pending() ? st.pending.options : legal_main(st);
  int n = std::min(static_cast<int>(desc.size()), RL_MAX_ACTIONS);
  for (int a = 0; a < n; ++a)
    encode_action_feature(st, desc[a], ctx, a, out + a * PPO_ACTION_FEAT_DIM);
}

static int fill_legal_mask(int n_actions, uint8_t* mask) {
  std::memset(mask, 0, RL_MAX_ACTIONS);
  int n = std::min(std::max(n_actions, 0), RL_MAX_ACTIONS);
  for (int i = 0; i < n; ++i) mask[i] = 1;
  return n;
}

static int fill_legal_mask(const RlOptionSet& opts, uint8_t* mask) {
  return fill_legal_mask(opts.n, mask);
}

static void encode_action_features_from_options(const GameState& st,
                                                const RlOptionSet& opts,
                                                float* out) {
  std::fill(out, out + RL_MAX_ACTIONS * PPO_ACTION_FEAT_DIM, 0.0f);
  const int ctx = opts.pending ? st.pending.context : -1;
  if (opts.pending) {
    int n = std::min({opts.n, static_cast<int>(st.pending.options.size()),
                      RL_MAX_ACTIONS});
    for (int a = 0; a < n; ++a) {
      encode_action_feature(st, st.pending.options[a], ctx, a,
                            out + a * PPO_ACTION_FEAT_DIM);
    }
    return;
  }

  int n = std::min({opts.n, static_cast<int>(opts.descriptors.size()),
                    RL_MAX_ACTIONS});
  for (int a = 0; a < n; ++a) {
    encode_action_feature(st, opts.descriptors[a], ctx, a,
                          out + a * PPO_ACTION_FEAT_DIM);
  }
}

static void add_static_card_features(float* f, int cid) {
  if (cid <= 0) return;
  const auto& cached = cached_card_summary(cid).features;
  std::copy(cached.begin(), cached.end(), f);
}

static void add_location_features(float* f, int owner_pov, int area, int index) {
  f[card_feat::OWNER_SELF] = owner_pov == 0 ? 1.0f : 0.0f;
  f[card_feat::OWNER_OPP] = owner_pov == 1 ? 1.0f : 0.0f;
  f[card_feat::AREA] = std::clamp(area / 7.0f, 0.0f, 1.0f);
  f[card_feat::INDEX] = std::clamp(index / 8.0f, 0.0f, 1.0f);
}

static bool energy_satisfies_type(int available, int required) {
  if (available == required) return true;
  if (available == RAINBOW) return true;
  if ((required == PSYCHIC || required == DARKNESS) &&
      available == TEAM_ROCKET)
    return true;
  return false;
}

static int attack_missing_energy_units(const InPlay& p, const AttackInfo& at) {
  std::array<int, card_feat::kEnergyTypes> pool{};
  for (int e : p.energies)
    pool[energy_feature_index(e)] += 1;

  int colorless = 0;
  int missing = 0;
  for (int i = 0; i < at.n_cost; ++i) {
    int required = at.cost[i];
    if (required == COLORLESS) {
      ++colorless;
      continue;
    }
    int match = -1;
    for (int e = 0; e < card_feat::kEnergyTypes; ++e) {
      if (pool[e] > 0 && energy_satisfies_type(e, required)) {
        match = e;
        break;
      }
    }
    if (match >= 0)
      pool[match] -= 1;
    else
      ++missing;
  }

  int remaining = 0;
  for (int n : pool) remaining += n;
  return missing + std::max(0, colorless - remaining);
}

static void encode_inplay_slot(float* f, const InPlay& p, int owner_pov,
                               int area, int index) {
  add_static_card_features(f, p.id);
  f[card_feat::HP] = p.hp / 400.0f;
  f[card_feat::HP_RATIO] =
      p.maxHp > 0 ? static_cast<float>(p.hp) / p.maxHp : 0.0f;
  f[card_feat::ATTACHED_ENERGY_COUNT] =
      norm_energy_count(std::min(static_cast<int>(p.energies.size()), 12));
  for (int e : p.energies) {
    int offset = card_feat::ATTACHED_ENERGY_TYPE_BASE + energy_feature_index(e);
    f[offset] = std::min(1.0f, f[offset] + (1.0f / 12.0f));
  }
  f[card_feat::TOOL_COUNT] =
      std::clamp(static_cast<float>(p.tools.size()) / 4.0f, 0.0f, 1.0f);
  f[card_feat::PRE_EVO_COUNT] =
      std::clamp(static_cast<float>(p.preEvo.size()) / 4.0f, 0.0f, 1.0f);
  f[card_feat::DAMAGE_COUNTERS] =
      std::clamp(static_cast<float>(std::max(0, p.maxHp - p.hp)) / 400.0f,
                 0.0f, 1.0f);
  f[card_feat::NO_ATTACK] = p.noAttackTurn >= 0 ? 1.0f : 0.0f;
  f[card_feat::NO_RETREAT] = p.noRetreatTurn >= 0 ? 1.0f : 0.0f;
  f[card_feat::PREVENT_DAMAGE] = p.preventDmgTurn >= 0 ? 1.0f : 0.0f;
  f[card_feat::TAKE_MORE_DAMAGE] =
      p.takeMoreDamageTurn >= 0 ? std::clamp(p.takeMoreDamage / 100.0f, 0.0f, 1.0f)
                                : 0.0f;
  int attack_delta = 0;
  if (p.attackBonusTurn >= 0) attack_delta += p.attackBonus;
  if (p.attackDmgReduceTurn >= 0) attack_delta -= p.attackDmgReduce;
  f[card_feat::ATTACK_DAMAGE_DELTA] =
      std::clamp((attack_delta + 100.0f) / 200.0f, 0.0f, 1.0f);
  f[card_feat::DAMAGE_REDUCTION] =
      p.dmgReduceTurn >= 0 ? std::clamp(p.dmgReduce / 100.0f, 0.0f, 1.0f)
                           : 0.0f;
  f[card_feat::ABILITY_USED] = p.abilityUsedThisTurn ? 1.0f : 0.0f;
  f[card_feat::MOVED_TO_ACTIVE] = p.movedToActiveThisTurn ? 1.0f : 0.0f;
  f[card_feat::HEALED_THIS_TURN] = p.healedThisTurn ? 1.0f : 0.0f;

  const CardInfo* c = find_card(p.id);
  if (c && c->n_attacks > 0) {
    int ready_attacks = 0;
    int best_ready_damage = 0;
    int min_missing = 99;
    int missing_for_best_damage = 99;
    int best_damage = -1;
    for (int i = 0; i < c->n_attacks; ++i) {
      const AttackInfo& at = c->attacks[i];
      int missing = attack_missing_energy_units(p, at);
      min_missing = std::min(min_missing, missing);
      if (missing == 0) {
        ++ready_attacks;
        best_ready_damage = std::max(best_ready_damage, at.damage);
      }
      if (at.damage > best_damage) {
        best_damage = at.damage;
        missing_for_best_damage = missing;
      } else if (at.damage == best_damage) {
        missing_for_best_damage = std::min(missing_for_best_damage, missing);
      }
    }
    if (min_missing == 99) min_missing = 0;
    if (missing_for_best_damage == 99) missing_for_best_damage = min_missing;
    f[card_feat::READY_ATTACK_COUNT] =
        std::clamp(ready_attacks / 4.0f, 0.0f, 1.0f);
    f[card_feat::BEST_READY_DAMAGE] = norm_damage(best_ready_damage);
    f[card_feat::MIN_MISSING_ATTACK_COST] =
        std::clamp(min_missing / 5.0f, 0.0f, 1.0f);
    f[card_feat::BEST_DAMAGE_MISSING_COST] =
        std::clamp(missing_for_best_damage / 5.0f, 0.0f, 1.0f);
  }
  add_location_features(f, owner_pov, area, index);
}

static void encode_card_slot(float* f, int cid, int owner_pov, int area,
                             int index) {
  add_static_card_features(f, cid);
  add_location_features(f, owner_pov, area, index);
}

static void encode_player_card_slots(const Player& p, int owner_pov,
                                     float*& out) {
  if (p.activeKnown) encode_inplay_slot(out, p.active, owner_pov, AREA_ACTIVE, 0);
  out += PPO_CARD_FEAT_DIM;
  for (int i = 0; i < 5; ++i) {
    if (i < static_cast<int>(p.bench.size()))
      encode_inplay_slot(out, p.bench[i], owner_pov, AREA_BENCH, i);
    out += PPO_CARD_FEAT_DIM;
  }
  for (int i = 0; i < 8; ++i) {
    if (p.handKnown && i < static_cast<int>(p.hand.size()))
      encode_card_slot(out, p.hand[i], owner_pov, AREA_HAND, i);
    else if (i < static_cast<int>(p.handKnownCards.size()))
      encode_card_slot(out, p.handKnownCards[i], owner_pov, AREA_HAND, i);
    out += PPO_CARD_FEAT_DIM;
  }
  for (int i = 0; i < 8; ++i) {
    if (i < static_cast<int>(p.discard.size()))
      encode_card_slot(out, p.discard[i], owner_pov, AREA_DISCARD, i);
    out += PPO_CARD_FEAT_DIM;
  }
}

static void encode_own_deck_slots(const Player& p, float*& out) {
  for (int i = 0; i < 16; ++i) {
    if (i < static_cast<int>(p.deck.size())) {
      int idx = static_cast<int>(p.deck.size()) - 1 - i;  // top-of-deck first
      encode_card_slot(out, p.deck[idx], 0, AREA_DECK, i);
    } else if (i < static_cast<int>(p.deckKnownCards.size())) {
      encode_card_slot(out, p.deckKnownCards[i], 0, AREA_DECK, i);
    }
    out += PPO_CARD_FEAT_DIM;
  }
}

static void encode_own_prize_slots(const Player& p, float*& out) {
  for (int i = 0; i < 3; ++i) {
    bool known_slot = p.prizesKnown ||
                      (i < static_cast<int>(p.prizesKnownMask.size()) &&
                       p.prizesKnownMask[i]);
    if (known_slot && i < static_cast<int>(p.prizes.size())) {
      encode_card_slot(out, p.prizes[i], 0, AREA_PRIZE, i);
    } else if (i < static_cast<int>(p.prizesKnownCards.size())) {
      encode_card_slot(out, p.prizesKnownCards[i], 0, AREA_PRIZE, i);
    }
    out += PPO_CARD_FEAT_DIM;
  }
}

void rl_card_features(const GameState& st, float* out) {
  std::fill(out, out + PPO_CARD_SLOTS * PPO_CARD_FEAT_DIM, 0.0f);
  float* p = out;
  const int me = st.yourIndex;
  encode_player_card_slots(st.players[me], 0, p);
  encode_player_card_slots(st.players[1 - me], 1, p);
  if (!st.stadium.empty())
    encode_card_slot(p, st.stadium[0], 0, AREA_STADIUM, 0);
  p += PPO_CARD_FEAT_DIM;
  encode_own_deck_slots(st.players[me], p);
  encode_own_prize_slots(st.players[me], p);
}

void rl_deck_features(const std::vector<int>& deck, float* out) {
  std::fill(out, out + PPO_DECK_SLOTS * PPO_CARD_FEAT_DIM, 0.0f);
  const int n = std::min(static_cast<int>(deck.size()), PPO_DECK_SLOTS);
  for (int i = 0; i < n; ++i) {
    encode_card_slot(out + i * PPO_CARD_FEAT_DIM, deck[i], 0, AREA_DECK, i);
  }
}

static void append_belief_slot(float* out, int& slot, int cid, int owner_pov,
                               int area, int index) {
  if (slot >= PPO_BELIEF_SLOTS || cid <= 0) return;
  encode_card_slot(out + slot * PPO_CARD_FEAT_DIM, cid, owner_pov, area, index);
  ++slot;
}

static void append_known_deck_beliefs(float* out, int& slot, const Player& p,
                                      int owner_pov) {
  if (p.deckKnown || !p.deckKnownMask.empty()) {
    for (int i = 0; i < static_cast<int>(p.deck.size()); ++i) {
      bool known = p.deckKnown ||
                   (i < static_cast<int>(p.deckKnownMask.size()) &&
                    p.deckKnownMask[i]);
      if (known) append_belief_slot(out, slot, p.deck[i], owner_pov, AREA_DECK, i);
    }
  }
  for (int i = 0; i < static_cast<int>(p.deckKnownCards.size()); ++i) {
    append_belief_slot(out, slot, p.deckKnownCards[i], owner_pov, AREA_DECK, i);
  }
}

static void append_known_prize_beliefs(float* out, int& slot, const Player& p,
                                       int owner_pov) {
  if (p.prizesKnown || !p.prizesKnownMask.empty()) {
    for (int i = 0; i < static_cast<int>(p.prizes.size()); ++i) {
      bool known = p.prizesKnown ||
                   (i < static_cast<int>(p.prizesKnownMask.size()) &&
                    p.prizesKnownMask[i]);
      if (known)
        append_belief_slot(out, slot, p.prizes[i], owner_pov, AREA_PRIZE, i);
    }
  }
  for (int i = 0; i < static_cast<int>(p.prizesKnownCards.size()); ++i) {
    append_belief_slot(out, slot, p.prizesKnownCards[i], owner_pov, AREA_PRIZE, i);
  }
}

void rl_belief_features(const GameState& st, float* out) {
  std::fill(out, out + PPO_BELIEF_SLOTS * PPO_CARD_FEAT_DIM, 0.0f);
  int slot = 0;
  const int me = st.yourIndex;
  const Player& mine = st.players[me];
  const Player& opp = st.players[1 - me];
  append_known_prize_beliefs(out, slot, mine, 0);
  append_known_deck_beliefs(out, slot, mine, 0);
  for (int i = 0; i < static_cast<int>(opp.handKnownCards.size()); ++i) {
    append_belief_slot(out, slot, opp.handKnownCards[i], 1, AREA_HAND, i);
  }
  append_known_deck_beliefs(out, slot, opp, 1);
  append_known_prize_beliefs(out, slot, opp, 1);
}

static void add_type_count(float* out, int offset, int cid, float scale = 4.0f) {
  const CardInfo* c = find_card(cid);
  if (!c) return;
  int t = std::min(std::max(c->cardType, 0), 6);
  out[offset + t] += 1.0f / scale;
}

static int known_deck_slot_count(const Player& p) {
  if (p.deckKnown) return p.deckCount;
  return static_cast<int>(
      std::count(p.deckKnownMask.begin(), p.deckKnownMask.end(), true));
}

static int known_prize_slot_count(const Player& p) {
  if (p.prizesKnown) return p.prizeCount;
  return static_cast<int>(
      std::count(p.prizesKnownMask.begin(), p.prizesKnownMask.end(), true));
}

static void add_known_deck_type_counts(float* out, int offset, const Player& p) {
  if (p.deckKnown || !p.deckKnownMask.empty()) {
    for (int i = 0; i < static_cast<int>(p.deck.size()); ++i) {
      bool known = p.deckKnown ||
                   (i < static_cast<int>(p.deckKnownMask.size()) &&
                    p.deckKnownMask[i]);
      if (known) add_type_count(out, offset, p.deck[i]);
    }
  }
  for (int cid : p.deckKnownCards) add_type_count(out, offset, cid);
}

static void add_known_prize_type_counts(float* out, int offset, const Player& p) {
  if (p.prizesKnown || !p.prizesKnownMask.empty()) {
    for (int i = 0; i < static_cast<int>(p.prizes.size()); ++i) {
      bool known = p.prizesKnown ||
                   (i < static_cast<int>(p.prizesKnownMask.size()) &&
                    p.prizesKnownMask[i]);
      if (known) add_type_count(out, offset, p.prizes[i]);
    }
  }
  for (int cid : p.prizesKnownCards) add_type_count(out, offset, cid);
}

void rl_belief_summary(const GameState& st, float* out) {
  std::fill(out, out + PPO_BELIEF_SUMMARY_DIM, 0.0f);
  const int me = st.yourIndex;
  const Player& mine = st.players[me];
  const Player& opp = st.players[1 - me];

  const int own_known_deck =
      known_deck_slot_count(mine) + static_cast<int>(mine.deckKnownCards.size());
  const int own_known_prize =
      known_prize_slot_count(mine) + static_cast<int>(mine.prizesKnownCards.size());
  const int opp_known_hand = static_cast<int>(opp.handKnownCards.size());
  const int opp_known_deck =
      known_deck_slot_count(opp) + static_cast<int>(opp.deckKnownCards.size());
  const int opp_known_prize =
      known_prize_slot_count(opp) + static_cast<int>(opp.prizesKnownCards.size());

  out[0] = mine.deckCount / 60.0f;
  out[1] = mine.prizeCount / 6.0f;
  out[2] = opp.handCount / 15.0f;
  out[3] = opp.deckCount / 60.0f;
  out[4] = opp.prizeCount / 6.0f;
  out[5] = std::min(own_known_deck, 60) / 60.0f;
  out[6] = std::min(own_known_prize, 6) / 6.0f;
  out[7] = std::min(opp_known_hand, 15) / 15.0f;
  out[8] = std::min(opp_known_deck, 60) / 60.0f;
  out[9] = std::min(opp_known_prize, 6) / 6.0f;
  out[10] = mine.deckKnown ? 1.0f : 0.0f;
  out[11] = mine.prizesKnown ? 1.0f : 0.0f;
  out[12] = opp.deckKnown ? 1.0f : 0.0f;
  out[13] = opp.prizesKnown ? 1.0f : 0.0f;
  out[14] = opp.handKnown ? 1.0f : 0.0f;

  add_known_prize_type_counts(out, 16, mine);  // 16..22
  add_known_deck_type_counts(out, 23, mine);   // 23..29
  for (int cid : opp.handKnownCards) add_type_count(out, 30, cid);  // 30..36
  add_known_deck_type_counts(out, 37, opp);    // 37..43
  add_known_prize_type_counts(out, 44, opp);   // 44..50
}

static long long elapsed_profile_ns(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end);

// Auto-resolve forced (<=1 option) and multi-select sub-decisions until the next
// single-select multi-option agent decision or game end. When `out` is non-null
// it receives exactly what rl_options() would return for the resulting state,
// so callers need not regenerate legal_main(). `out` may alias a caller-held
// option set: it is only written after the game state is final.
static void advance_to_agent(GameState& st, uint64_t& rng,
                             RlOptionSet* out = nullptr) {
  while (st.result < 0) {
    if (st.has_pending()) {
      const PendingDecision& pd = st.pending;
      int n = static_cast<int>(pd.options.size());
      if (pd.maxCount > 1 || n <= 1) {
        thread_local std::vector<int> selection;
        random_subset_into(pd, rng, selection);
        resolve(st, selection);
        continue;
      }
      if (out) {
        out->pending = true;
        out->n = std::min(n, RL_MAX_ACTIONS);
        out->descriptors.clear();
      }
      return;  // single-select, multi-option
    }
    auto opts = legal_main(st);
    if (opts.size() <= 1) {
      apply(st, descriptor_to_action(opts[0]));
      continue;
    }
    if (out) *out = rl_options_from_descriptors(std::move(opts));
    return;  // multi-option MAIN
  }
  if (out) *out = rl_options(st);  // terminal (possibly with leftover pending)
}

static void advance_to_agent_profiled(GameState& st, uint64_t& rng,
                                      PpoStepProfile& profile,
                                      RlOptionSet* out = nullptr) {
  while (st.result < 0) {
    if (st.has_pending()) {
      const PendingDecision& pd = st.pending;
      int n = static_cast<int>(pd.options.size());
      if (pd.maxCount > 1 || n <= 1) {
        auto t0 = std::chrono::steady_clock::now();
        thread_local std::vector<int> selection;
        random_subset_into(pd, rng, selection);
        resolve(st, selection);
        auto t1 = std::chrono::steady_clock::now();
        profile.auto_pending_ns += elapsed_profile_ns(t0, t1);
        ++profile.auto_pending_decisions;
        continue;
      }
      if (out) {
        out->pending = true;
        out->n = std::min(n, RL_MAX_ACTIONS);
        out->descriptors.clear();
      }
      return;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto opts = legal_main(st);
    auto t1 = std::chrono::steady_clock::now();
    profile.auto_main_options_ns += elapsed_profile_ns(t0, t1);
    if (opts.size() <= 1) {
      auto t2 = std::chrono::steady_clock::now();
      apply(st, descriptor_to_action(opts[0]));
      auto t3 = std::chrono::steady_clock::now();
      profile.auto_main_apply_ns += elapsed_profile_ns(t2, t3);
      ++profile.auto_main_decisions;
      continue;
    }
    if (out) *out = rl_options_from_descriptors(std::move(opts));
    return;
  }
  if (out) *out = rl_options(st);
}

static void apply_choice(GameState& st, int action) {
  if (st.has_pending()) {
    resolve(st, single_selection(action));
  } else {
    auto opts = legal_main(st);
    if (action >= 0 && action < static_cast<int>(opts.size()))
      apply(st, descriptor_to_action(opts[action]));
  }
}

// apply_choice against a pre-built option set (must equal rl_options(st)).
// Mirrors apply_choice exactly: pending selections are resolved without a
// range check, MAIN actions are bounds-checked against the stored descriptors.
static void apply_choice_cached(GameState& st, const RlOptionSet& opts,
                                int action) {
  if (opts.pending) {
    resolve(st, single_selection(action));
  } else if (action >= 0 &&
             action < static_cast<int>(opts.descriptors.size())) {
    apply(st, descriptor_to_action(opts.descriptors[action]));
  }
}

void rl_step(GameState& st, int action, uint64_t& rng) {
  apply_choice(st, action);
  advance_to_agent(st, rng);
}

void rl_step_cached(GameState& st, const RlOptionSet& opts, int action,
                    uint64_t& rng, RlOptionSet* next) {
  if (opts.pending) {
    if (action >= 0 && action < opts.n)
      resolve(st, single_selection(action));
  } else if (action >= 0 && action < opts.n &&
             action < static_cast<int>(opts.descriptors.size())) {
    apply(st, descriptor_to_action(opts.descriptors[action]));
  }
  advance_to_agent(st, rng, next);  // `next` may alias `opts`; opts is done
}

static void rl_step_cached_profiled(GameState& st, const RlOptionSet& opts,
                                    int action, uint64_t& rng,
                                    PpoStepProfile& profile,
                                    RlOptionSet* next = nullptr) {
  if (opts.pending) {
    if (action >= 0 && action < opts.n)
      resolve(st, single_selection(action));
  } else if (action >= 0 && action < opts.n &&
             action < static_cast<int>(opts.descriptors.size())) {
    apply(st, descriptor_to_action(opts.descriptors[action]));
  }
  advance_to_agent_profiled(st, rng, profile, next);
}

namespace {

struct MctsNode {
  GameState state;
  int mover = 0;
  bool terminal = false;
  int n_act = 0;
  double totalN = 0.0;
  RlOptionSet opts;
  std::vector<double> childN;
  std::vector<double> childW;
  std::vector<int> child;

  explicit MctsNode(GameState s) : state(std::move(s)) {
    mover = state.yourIndex;
    terminal = state.result >= 0;
  }
};

static double term_value(const GameState& st, int me) {
  if (st.result == 2) return 0.0;
  return st.result == me ? 1.0 : -1.0;
}

static void expand_mcts_node(MctsNode& node, RlOptionSet opts) {
  node.opts = std::move(opts);
  node.n_act = std::max(node.opts.n, 1);
  node.childN.assign(node.n_act, 0.0);
  node.childW.assign(node.n_act, 0.0);
  node.child.assign(node.n_act, -1);
}

static void expand_mcts_node(MctsNode& node) {
  expand_mcts_node(node, rl_options(node.state));
}

static int select_mcts_action(const MctsNode& node, int me, double c_puct) {
  double c = c_puct * std::sqrt(node.totalN + 1.0);
  double prior_u = c / static_cast<double>(node.n_act);
  bool flip = node.mover != me;
  int best_i = 0;
  double best_u = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < node.n_act; ++i) {
    double n = node.childN[i];
    double q = n > 0.0 ? node.childW[i] / n : 0.0;
    if (flip) q = -q;
    double u = q + prior_u / (1.0 + n);
    if (u > best_u) {
      best_u = u;
      best_i = i;
    }
  }
  return best_i;
}

}  // namespace

MctsResult rl_value_mcts(const GameState& root_state, int n_sims, uint64_t seed,
                         double c_puct) {
  std::vector<MctsNode> tree;
  tree.reserve(std::max(1, n_sims) + 1);
  tree.emplace_back(root_state);
  MctsNode& root = tree[0];
  MctsResult out;
  out.terminal = root.terminal;
  if (root.terminal) return out;

  int me = root.mover;
  expand_mcts_node(root);
  uint64_t rng = seed ? seed : 1;
  for (int sim = 0; sim < n_sims; ++sim) {
    int node_idx = 0;
    std::vector<std::pair<int, int>> path;
    double v_me = 0.0;
    while (true) {
      MctsNode& node = tree[node_idx];
      int a = select_mcts_action(node, me, c_puct);
      path.emplace_back(node_idx, a);
      int child_idx = node.child[a];
      if (child_idx < 0) {
        GameState cs = node.state;
        rng = (rng * 6364136223846793005ULL + 1442695040888963407ULL);
        cs.rng = rng;
        uint64_t step_rng = rng;
        RlOptionSet leaf_opts;
        rl_step_cached(cs, node.opts, a, step_rng, &leaf_opts);
        tree.emplace_back(std::move(cs));
        int leaf_idx = static_cast<int>(tree.size()) - 1;
        tree[node_idx].child[a] = leaf_idx;
        MctsNode& leaf = tree[leaf_idx];
        if (leaf.terminal) {
          v_me = term_value(leaf.state, me);
        } else {
          expand_mcts_node(leaf, std::move(leaf_opts));
          double v = rl_heuristic_value(leaf.state);
          v_me = leaf.mover == me ? v : -v;
        }
        break;
      }
      node_idx = child_idx;
      if (tree[node_idx].terminal) {
        v_me = term_value(tree[node_idx].state, me);
        break;
      }
    }
    for (auto [idx, a] : path) {
      MctsNode& nd = tree[idx];
      nd.childN[a] += 1.0;
      nd.childW[a] += v_me;
      nd.totalN += 1.0;
    }
  }

  out.terminal = root.terminal;
  out.n_act = root.n_act;
  out.childN = root.childN;
  out.childW = root.childW;
  return out;
}

// --- batch environment ----------------------------------------------------
BatchEnv::BatchEnv(std::vector<int> deck0, std::vector<int> deck1, int n,
                   uint64_t seed, int threads)
    : deck0_(std::move(deck0)), deck1_(std::move(deck1)), threads_(threads) {
  rngs_ = make_game_rngs(seed, n);
  games_.resize(n);
  opts_.resize(n);
  reset_all();
}

void BatchEnv::reset(int i) {
  uint64_t s = rl_rand(rngs_[i]);
  games_[i] = new_game(deck0_, deck1_, s ? s : 1);
  advance_to_agent(games_[i], rngs_[i], &opts_[i]);
}

void BatchEnv::reset_all() {
  EnvPool::instance().run(size(), threads_, [&](int i) { reset(i); });
}

void BatchEnv::observe(float* obs, uint8_t* mask) const {
  int D = RL_OBS_DIM;
  EnvPool::instance().run(size(), threads_, [&](int i) {
    rl_encode_obs(games_[i], obs + i * D);
    fill_legal_mask(opts_[i], mask + i * RL_MAX_ACTIONS);
  });
}

void BatchEnv::step(const int* actions, float* obs, float* reward, uint8_t* done,
                    uint8_t* mask) {
  int D = RL_OBS_DIM;
  EnvPool::instance().run(size(), threads_, [&](int i) {
    int actor = games_[i].yourIndex;  // the mover whose reward this is
    apply_choice_cached(games_[i], opts_[i], actions[i]);
    advance_to_agent(games_[i], rngs_[i], &opts_[i]);
    int r = games_[i].result;
    if (r >= 0) {
      reward[i] = (r == 2) ? 0.f : (r == actor ? 1.f : -1.f);
      done[i] = 1;
      reset(i);
    } else {
      reward[i] = 0.f;
      done[i] = 0;
    }
    rl_encode_obs(games_[i], obs + i * D);
    fill_legal_mask(opts_[i], mask + i * RL_MAX_ACTIONS);
  });
}

VectorEnv::VectorEnv(std::vector<int> deck0, std::vector<int> deck1, int n,
                     uint64_t seed, int threads)
    : deck0_(std::move(deck0)), deck1_(std::move(deck1)), threads_(threads) {
  rngs_ = make_game_rngs(seed, n);
  games_.resize(n);
  opts_.resize(n);
  reset_all();
}

void VectorEnv::reset(int i) {
  uint64_t s = rl_rand(rngs_[i]);
  games_[i] = new_game(deck0_, deck1_, s ? s : 1);
  advance_to_agent(games_[i], rngs_[i], &opts_[i]);
}

void VectorEnv::step_one(int i, int action, float& reward, uint8_t& done) {
  const int actor = games_[i].yourIndex;
  apply_choice_cached(games_[i], opts_[i], action);
  advance_to_agent(games_[i], rngs_[i], &opts_[i]);
  const int game_result = games_[i].result;
  if (game_result < 0) {
    reward = 0.f;
    done = 0;
    return;
  }
  reward = game_result == 2 ? 0.f : (game_result == actor ? 1.f : -1.f);
  done = 1;
  reset(i);
}

void VectorEnv::reset_all() {
  EnvPool::instance().run(size(), threads_, [&](int i) { reset(i); });
}

void VectorEnv::observe(float* obs, uint8_t* mask, int32_t* player,
                        int32_t* result) const {
  int D = RL_OBS_DIM;
  EnvPool::instance().run(size(), threads_, [&](int i) {
    rl_encode_obs(games_[i], obs + i * D);
    fill_legal_mask(opts_[i], mask + i * RL_MAX_ACTIONS);
    player[i] = games_[i].yourIndex;
    result[i] = games_[i].result;
  });
}

void VectorEnv::observe_ids(int32_t* in_play, int32_t* zones,
                            int32_t* player_counts, int32_t* player_status,
                            int32_t* global, int32_t* action_meta,
                            int32_t* action_options, int32_t* action_deck,
                            uint8_t* action_mask, int32_t* player,
                            int32_t* result) const {
  EnvPool::instance().run(size(), threads_, [&](int i) {
    const GameState& st = games_[i];
    fill_observation_ids(
        st,
        in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    fill_action_ids(
        st, action_id_view_from_options(opts_[i]),
        action_meta + i * ACTION_META_WIDTH,
        action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
        action_deck + i * STATE_ZONE_SLOTS,
        action_mask + i * RL_MAX_ACTIONS);
    player[i] = st.yourIndex;
    result[i] = st.result;
  });
}

bool VectorEnv::observe_ids16(
    int16_t* in_play, int16_t* zones, int16_t* player_counts,
    int16_t* player_status, int16_t* global, int16_t* action_meta,
    int16_t* action_options, int16_t* action_deck, uint8_t* action_mask,
    int32_t* player, int32_t* result) const {
  std::atomic<bool> ok{true};
  EnvPool::instance().run(size(), threads_, [&](int i) {
    const GameState& st = games_[i];
    bool row_ok = fill_observation_ids16(
        st, in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    row_ok = fill_action_ids16(
                 st, action_id_view_from_options(opts_[i]),
                 action_meta + i * ACTION_META_WIDTH,
                 action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                 action_deck + i * STATE_ZONE_SLOTS,
                 action_mask + i * RL_MAX_ACTIONS) &&
             row_ok;
    if (!row_ok) ok.store(false, std::memory_order_relaxed);
    player[i] = st.yourIndex;
    result[i] = st.result;
  });
  return ok.load(std::memory_order_relaxed);
}

void VectorEnv::step(const int* actions, float* obs, float* reward,
                     uint8_t* done, uint8_t* mask, int32_t* player,
                     int32_t* result) {
  int D = RL_OBS_DIM;
  EnvPool::instance().run(size(), threads_, [&](int i) {
    step_one(i, actions[i], reward[i], done[i]);
    rl_encode_obs(games_[i], obs + i * D);
    fill_legal_mask(opts_[i], mask + i * RL_MAX_ACTIONS);
    player[i] = games_[i].yourIndex;
    result[i] = games_[i].result;
  });
}

void VectorEnv::step_ids(const int* actions, float* reward, uint8_t* done,
                         int32_t* in_play, int32_t* zones,
                         int32_t* player_counts, int32_t* player_status,
                         int32_t* global, int32_t* action_meta,
                         int32_t* action_options, int32_t* action_deck,
                         uint8_t* action_mask, int32_t* player,
                         int32_t* result) {
  EnvPool::instance().run(size(), threads_, [&](int i) {
    step_one(i, actions[i], reward[i], done[i]);

    const GameState& st = games_[i];
    fill_observation_ids(
        st,
        in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    fill_action_ids(
        st, action_id_view_from_options(opts_[i]),
        action_meta + i * ACTION_META_WIDTH,
        action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
        action_deck + i * STATE_ZONE_SLOTS,
        action_mask + i * RL_MAX_ACTIONS);
    player[i] = st.yourIndex;
    result[i] = st.result;
  });
}

bool VectorEnv::step_ids16(
    const int* actions, float* reward, uint8_t* done, int16_t* in_play,
    int16_t* zones, int16_t* player_counts, int16_t* player_status,
    int16_t* global, int16_t* action_meta, int16_t* action_options,
    int16_t* action_deck, uint8_t* action_mask, int32_t* player,
    int32_t* result) {
  std::atomic<bool> ok{true};
  EnvPool::instance().run(size(), threads_, [&](int i) {
    step_one(i, actions[i], reward[i], done[i]);
    const GameState& st = games_[i];
    bool row_ok = fill_observation_ids16(
        st, in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    row_ok = fill_action_ids16(
                 st, action_id_view_from_options(opts_[i]),
                 action_meta + i * ACTION_META_WIDTH,
                 action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                 action_deck + i * STATE_ZONE_SLOTS,
                 action_mask + i * RL_MAX_ACTIONS) &&
             row_ok;
    if (!row_ok) ok.store(false, std::memory_order_relaxed);
    player[i] = st.yourIndex;
    result[i] = st.result;
  });
  return ok.load(std::memory_order_relaxed);
}

namespace {

constexpr int C_MAKUHITA = 673;
constexpr int C_HARIYAMA = 674;
constexpr int C_LUNATONE = 675;
constexpr int C_SOLROCK = 676;
constexpr int C_RIOLU = 677;
constexpr int C_MEGA_LUCARIO_EX = 678;
constexpr int C_KYOGRE = 721;
constexpr int C_SNOVER = 722;
constexpr int C_MEGA_ABOMASNOW_EX = 723;
constexpr int C_SWITCH = 1123;
constexpr int C_PREMIUM_POWER_PRO = 1141;
constexpr int C_HERO_CAPE = 1159;
constexpr int C_BOSS_ORDERS = 1182;
constexpr int C_CARMINE = 1192;
constexpr int C_LILLIE_DETERMINATION = 1227;
constexpr int C_GRAVITY_MOUNTAIN = 1252;
constexpr int C_LUMIOSE_CITY = 1267;
constexpr int C_BASIC_FIGHTING_ENERGY = 6;
constexpr int C_LEGACY_ENERGY = 12;
constexpr int A_MEGA_BRAVE = 983;
constexpr int LOW_DECK_COUNT_940 = 10;

struct Native940Plan {
  int attacker = -1;      // 0 active, 1.. bench+1
  int target = -1;        // 0 active, 1.. bench+1
  int attack_index = -1;  // 0 first attack, 1 second attack
  int remain_hp = -1;
  bool needs_energy = false;
};

struct Native940Ctx {
  std::array<uint8_t, 2048> field{};
  std::array<uint8_t, 2048> hand{};
  std::array<uint8_t, 2048> discard{};
  std::array<bool, 6> can_evolve_board{};
  bool ready_lucario = false;
  bool ready_hariyama = false;
  bool can_switch = false;
  bool can_gust = false;
  bool can_attack = false;
  bool can_mega_brave = false;
  int stadium_id = 0;
  Native940Plan plan;
};

static int clamp_cid(int cid) { return std::clamp(cid, 0, 2047); }

static void increment_940_count(std::array<uint8_t, 2048>& counts, int cid) {
  uint8_t& v = counts[clamp_cid(cid)];
  if (v < std::numeric_limits<uint8_t>::max()) ++v;
}

static int energy_count(const InPlay& p) {
  return static_cast<int>(p.energies.size());
}

static const InPlay* board_at(const Player& p, int board_index) {
  if (board_index == 0) return p.activeKnown ? &p.active : nullptr;
  int b = board_index - 1;
  return (b >= 0 && b < static_cast<int>(p.bench.size())) ? &p.bench[b] : nullptr;
}

static int board_count(const Player& p) {
  return (p.activeKnown ? 1 : 0) + static_cast<int>(p.bench.size());
}

static bool board_has(const Player& p, int cid) {
  if (p.activeKnown && p.active.id == cid) return true;
  for (const InPlay& b : p.bench)
    if (b.id == cid) return true;
  return false;
}

static bool opp_water_940(const Player& opp) {
  return board_has(opp, C_KYOGRE) || board_has(opp, C_SNOVER) ||
         board_has(opp, C_MEGA_ABOMASNOW_EX);
}

static bool opp_crustle_940(const Player& opp) {
  return board_has(opp, 344) || board_has(opp, 345);
}

static int prize_count_940(const InPlay& p) {
  const CardInfo* c = find_card(p.id);
  int count = (c && c->megaEx) ? 3 : (c && c->ex) ? 2 : 1;
  for (int cid : p.energyCardIds)
    if (cid == C_LEGACY_ENERGY) --count;
  return std::max(0, count);
}

static int target_score_940(const InPlay& p) {
  const CardInfo* c = find_card(p.id);
  int score = prize_count_940(p) * 2000 + energy_count(p) * 300 +
              static_cast<int>(p.tools.size()) * 200;
  if (c && c->stage2) score += 500;
  else if (c && c->stage1) score += 250;
  if (p.id == 144 || p.id == 322 || p.id == 323 || p.id == 337) score -= 200;
  if (p.id == C_SNOVER) score += 950;
  else if (p.id == C_MEGA_ABOMASNOW_EX) score += 250;
  if (p.id == C_RIOLU) score += 800;
  else if (p.id == C_MEGA_LUCARIO_EX) score += 100;
  return score + p.hp;
}

static bool base_attack_940(const Native940Ctx& cx, const Player& opp,
                            const InPlay& p, int board_idx, int attack_idx,
                            int& req, int& damage, int& base_score) {
  req = damage = base_score = 0;
  if (p.id == C_MEGA_LUCARIO_EX) {
    if (attack_idx == 0) {
      req = 1;
      damage = 130;
      base_score +=
          60 * std::min(3, static_cast<int>(cx.discard[C_BASIC_FIGHTING_ENERGY]));
    } else {
      req = 2;
      damage = 270;
    }
    if (opp_water_940(opp) && opp.prizeCount <= 3) base_score -= 500;
  } else if (attack_idx == 1) {
    return false;
  } else if (p.id == C_HARIYAMA) {
    req = 3;
    damage = 210;
  } else if (p.id == C_MAKUHITA) {
    if (board_idx < 0 || board_idx >= static_cast<int>(cx.can_evolve_board.size()) ||
        !cx.can_evolve_board[board_idx])
      return false;
    req = 3;
    damage = 210;
    base_score -= 100;
  } else if (p.id == C_SOLROCK && cx.field[C_LUNATONE] >= 1) {
    req = 1;
    damage = 70;
  }
  return damage > 0;
}

static void build_ctx_940(const GameState& st, const std::vector<Descriptor>& desc,
                          Native940Ctx& cx) {
  const Player& me = st.players[st.yourIndex];
  for (int i = 0; i < board_count(me); ++i) {
    const InPlay* p = board_at(me, i);
    if (!p) continue;
    increment_940_count(cx.field, p->id);
    if ((p->id == C_MAKUHITA || p->id == C_HARIYAMA) && energy_count(*p) >= 3)
      cx.ready_hariyama = true;
    if ((p->id == C_RIOLU || p->id == C_MEGA_LUCARIO_EX) &&
        energy_count(*p) >= 2)
      cx.ready_lucario = true;
  }
  if (me.handKnown)
    for (int cid : me.hand) increment_940_count(cx.hand, cid);
  for (int cid : me.discard) increment_940_count(cx.discard, cid);
  cx.stadium_id = st.stadium.empty() ? 0 : st.stadium[0];

  for (const Descriptor& d : desc) {
    std::string_view kind = desc_str(d, 0);
    if (kind == "PLAY") {
      int cid = desc_int(d, 1);
      if (cid == C_SWITCH) cx.can_switch = true;
      else if (cid == C_BOSS_ORDERS) cx.can_gust = true;
    } else if (kind == "EVOLVE") {
      if (desc_int(d, 1) == C_HARIYAMA) cx.can_gust = true;
      int idx = desc_int(d, 3);
      if (desc_str(d, 2) == "BENCH") ++idx;
      if (idx >= 0 && idx < static_cast<int>(cx.can_evolve_board.size()))
        cx.can_evolve_board[idx] = true;
    } else if (kind == "RETREAT") {
      cx.can_switch = true;
    } else if (kind == "ATTACK") {
      cx.can_attack = true;
      if (desc_int(d, 1) == A_MEGA_BRAVE) cx.can_mega_brave = true;
    }
  }
}

static void plan_attack_940(const GameState& st, Native940Ctx& cx) {
  if (st.turn < 2) return;
  const Player& me = st.players[st.yourIndex];
  const Player& opp = st.players[1 - st.yourIndex];
  float best = -1.0f;
  for (int ai = 0; ai < board_count(me); ++ai) {
    const InPlay* attacker = board_at(me, ai);
    if (!attacker) continue;
    if (ai != 0 && !cx.can_switch) break;
    for (int attack_idx = 0; attack_idx < 2; ++attack_idx) {
      int req = 0, damage = 0, base = 0;
      if (!base_attack_940(cx, opp, *attacker, ai, attack_idx, req, damage,
                           base)) {
        continue;
      }
      int ecount = energy_count(*attacker);
      if (attack_idx == 1 && ai == 0 && ecount >= 2 && !cx.can_mega_brave)
        break;
      bool needs_energy = false;
      if (ecount < req) {
        if (cx.hand[C_BASIC_FIGHTING_ENERGY] >= 1 && !st.energyAttached) {
          ++ecount;
          needs_energy = ecount >= req;
        }
        if (!needs_energy) continue;
      }
      for (int ti = 0; ti < board_count(opp); ++ti) {
        const InPlay* target = board_at(opp, ti);
        if (!target) continue;
        if (ti != 0 && !cx.can_gust) break;
        if (opp_crustle_940(opp) && attacker->id == C_MEGA_LUCARIO_EX &&
            target->id == 345) {
          continue;
        }
        int adjusted = damage;
        const CardInfo* tc = find_card(target->id);
        if (tc && tc->weakness == FIGHTING) adjusted *= 2;
        else if (tc && tc->resistance == FIGHTING) adjusted -= 30;
        float score = static_cast<float>(target_score_940(*target));
        int prize = target->hp <= adjusted ? prize_count_940(*target) : 0;
        if (prize == 0 && target->hp > 0)
          score *= static_cast<float>(adjusted) / static_cast<float>(target->hp);
        if (opp.prizeCount <= prize) score = 500000.0f;
        score += static_cast<float>(base + (ai == 0 ? 220 : 0) +
                                    (ti == 0 ? 300 : 0) + ecount);
        if (score > best) {
          best = score;
          cx.plan = {ai, ti, attack_idx, target->hp - adjusted, needs_energy};
        }
      }
    }
  }
}

static int energy_target_score_940(const GameState& st, const Native940Ctx& cx,
                                   const InPlay& p, bool active) {
  int e = energy_count(p);
  int score = 8000 + (active ? 10 : 0);
  bool crustle = opp_crustle_940(st.players[1 - st.yourIndex]);
  if (p.id == C_MAKUHITA || p.id == C_HARIYAMA) {
    if (p.id == C_HARIYAMA) ++score;
    if (crustle) score += e < 3 ? 260 : 30;
    else {
      score += e < 3 ? 100 : 0;
      if (cx.ready_hariyama) score -= 50;
    }
  } else if (p.id == C_LUNATONE) {
    score -= 100;
  } else if (p.id == C_SOLROCK) {
    score += e < 1 ? 20 : -100;
  } else if (p.id == C_RIOLU || p.id == C_MEGA_LUCARIO_EX) {
    if (p.id == C_MEGA_LUCARIO_EX) ++score;
    score += e < 2 ? 100 : 0;
    if (cx.ready_lucario) score -= 50;
  }
  return score;
}

static const InPlay* own_target_from_desc(const GameState& st, const Descriptor& d) {
  const Player& me = st.players[st.yourIndex];
  std::string_view area = desc_str(d, 2);
  int idx = desc_int(d, 3);
  std::string_view kind = desc_str(d, 0);
  if (kind == "ABILITY" || kind == "DISCARD" || kind == "CARD" ||
      kind == "ENERGY") {
    area = desc_str(d, 1);
    idx = desc_int(d, 2);
  }
  if (area == "ACTIVE") return me.activeKnown ? &me.active : nullptr;
  if (area == "BENCH" && idx >= 0 && idx < static_cast<int>(me.bench.size()))
    return &me.bench[idx];
  return nullptr;
}

static float score_card_choice_940(const GameState& st, const Native940Ctx& cx,
                                   const Descriptor& d) {
  int cid = (!d.empty() && !d.back().is_str && !d.back().is_none)
                ? static_cast<int>(d.back().i)
                : 0;
  if (cid <= 0) return 0.0f;
  if (cid == C_SOLROCK) return st.firstPlayer == st.yourIndex ? 2.0f : 4.0f;
  if (cid == C_RIOLU) return 3.0f;
  if (cid == C_MAKUHITA) return 1.0f;
  bool crustle = opp_crustle_940(st.players[1 - st.yourIndex]);
  float score = 200.0f - 100.0f * cx.hand[clamp_cid(cid)];
  if (cid == C_MAKUHITA)
    score += crustle ? (cx.field[cid] < 2 ? 80 : -20)
                     : (cx.field[cid] >= 1 ? -10 : 10);
  else if (cid == C_HARIYAMA)
    score += crustle ? (cx.field[C_MAKUHITA] >= 1 ? 120 : -5)
                     : (cx.field[C_MAKUHITA] >= 1 ? 20 : -20);
  else if (cid == C_LUNATONE)
    score += cx.field[cid] >= 1 ? -250 : 60;
  else if (cid == C_SOLROCK)
    score += cx.field[cid] >= 1 ? -250 : 50;
  else if (cid == C_RIOLU)
    score += (cx.field[C_RIOLU] + cx.field[C_MEGA_LUCARIO_EX] >= 2) ? -150
             : (cx.field[C_RIOLU] + cx.field[C_MEGA_LUCARIO_EX] >= 1) ? -3
                                                                       : 40;
  else if (cid == C_MEGA_LUCARIO_EX)
    score += cx.field[C_RIOLU] >= 1 ? 40 : -15;
  else if (cid == C_BASIC_FIGHTING_ENERGY)
    score += !st.energyAttached ? 30 : -1;
  return score;
}

static float score_option_940(const GameState& st, const Native940Ctx& cx,
                              const Descriptor& d) {
  std::string_view kind = desc_str(d, 0);
  const Player& me = st.players[st.yourIndex];
  const Player& opp = st.players[1 - st.yourIndex];
  if (kind == "COUNT") return static_cast<float>(desc_int(d, 1));
  if (kind == "YES") return 1.0f;
  if (kind == "NO") return 0.0f;
  if (kind == "CARD" || kind == "ENERGY") return score_card_choice_940(st, cx, d);
  if (kind == "PLAY" || kind == "SETUP_ACTIVE") {
    int cid = desc_int(d, 1);
    const CardInfo* ci = find_card(cid);
    if (ci && ci->cardType == POKEMON) {
      if ((cid == C_LUNATONE || cid == C_SOLROCK) && cx.field[cid] >= 1) return -1;
      if (cid == C_RIOLU &&
          cx.field[C_RIOLU] + cx.field[C_MEGA_LUCARIO_EX] >= 2) return -1;
      return 20000;
    }
    if (cid == C_SWITCH) return cx.plan.attacker > 0 ? 6000 : -1;
    if (cid == C_PREMIUM_POWER_PRO) {
      if (st.supporterPlayed && cx.plan.remain_hp <= 0) return -1;
      if (!cx.can_attack)
        return (!st.supporterPlayed && cx.hand[C_CARMINE] > 0 &&
                cx.hand[C_LILLIE_DETERMINATION] == 0 &&
                me.deckCount > LOW_DECK_COUNT_940)
                   ? 3050
                   : -1;
      return 5000;
    }
    if (cid == C_BOSS_ORDERS) return cx.plan.target >= 1 ? 3200 : -1;
    if (cid == C_CARMINE) return me.deckCount <= LOW_DECK_COUNT_940 ? -1 : 3000;
    if (cid == C_LILLIE_DETERMINATION)
      return me.deckCount <= LOW_DECK_COUNT_940 ? -1 : 3100;
    if (cid == C_GRAVITY_MOUNTAIN) {
      bool opp_stage2 = false;
      if (opp.activeKnown) {
        const CardInfo* ac = find_card(opp.active.id);
        opp_stage2 = ac && ac->stage2;
      }
      for (const InPlay& p : opp.bench) {
        const CardInfo* pc = find_card(p.id);
        opp_stage2 = opp_stage2 || (pc && pc->stage2);
      }
      return opp_stage2 ? 3500 : (cx.stadium_id ? 1200 : -1);
    }
    return 10000;
  }
  if (kind == "ATTACH") {
    int cid = desc_int(d, 1);
    const InPlay* p = own_target_from_desc(st, d);
    if (!p) return 0;
    if (cid == C_HERO_CAPE) {
      if (opp_water_940(opp))
        return p->id == C_RIOLU ? 12200
             : p->id == C_MEGA_LUCARIO_EX ? 12800
                                           : 7000;
      return 7000 + (p->id == C_RIOLU ? 100 : p->id == C_MEGA_LUCARIO_EX ? 200 : 0);
    }
    int board_idx = desc_str(d, 2) == "ACTIVE" ? desc_int(d, 3)
                                                : desc_int(d, 3) + 1;
    float score = static_cast<float>(
        energy_target_score_940(st, cx, *p, desc_str(d, 2) == "ACTIVE"));
    if (board_idx == cx.plan.attacker && cx.plan.needs_energy) score += 200;
    return score;
  }
  if (kind == "EVOLVE") {
    const InPlay* p = own_target_from_desc(st, d);
    if (!p) return 0;
    if (p->id == C_MAKUHITA && cx.plan.target == 0 && !opp_crustle_940(opp))
      return -1;
    return 9000 + static_cast<int>(p->energies.size());
  }
  if (kind == "ABILITY") {
    int cid = ppo_target_card_id(st, st.yourIndex, desc_str(d, 1), desc_int(d, 2));
    if (cid == C_LUMIOSE_CITY) return 1;
    if (cid == C_LUNATONE && me.deckCount <= LOW_DECK_COUNT_940) return -1;
    return 30000;
  }
  if (kind == "RETREAT") return cx.plan.attacker >= 1 ? 2000 : -1;
  if (kind == "ATTACK") {
    if (opp_crustle_940(opp) && me.activeKnown && opp.activeKnown &&
        me.active.id == C_MEGA_LUCARIO_EX && opp.active.id == 345 &&
        cx.plan.target < 0) {
      return -1;
    }
    bool mega = desc_int(d, 1) == A_MEGA_BRAVE;
    return mega == (cx.plan.attack_index == 1) ? 1100 : 1000;
  }
  return 0;
}

static int native_940_action_from_options(const GameState& st,
                                          const RlOptionSet& opts) {
  const std::vector<Descriptor>& desc =
      opts.pending ? st.pending.options : opts.descriptors;
  int n = std::min({opts.n, static_cast<int>(desc.size()), RL_MAX_ACTIONS});
  if (n <= 1) return 0;
  Native940Ctx cx;
  build_ctx_940(st, desc, cx);
  plan_attack_940(st, cx);
  int best = 0;
  float best_score = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) {
    float score = score_option_940(st, cx, desc[i]);
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

static int native_940_action_impl(const GameState& st) {
  RlOptionSet opts = rl_options(st);
  return native_940_action_from_options(st, opts);
}

struct MetaScoredAction {
  int action = 0;
  double score = 0.0;
};

static double value_for_player(const GameState& st, int player) {
  if (st.result >= 0) {
    if (st.result == 2) return 0.0;
    return st.result == player ? 1.0 : -1.0;
  }
  double v = rl_heuristic_value(st);
  return st.yourIndex == player ? v : -v;
}

static std::vector<MetaScoredAction> rank_actions_940(const GameState& st) {
  const std::vector<Descriptor> desc =
      st.has_pending() ? st.pending.options : legal_main(st);
  int n = std::min(static_cast<int>(desc.size()), RL_MAX_ACTIONS);
  std::vector<MetaScoredAction> out;
  out.reserve(std::max(n, 0));
  if (n <= 0) return out;
  Native940Ctx cx;
  build_ctx_940(st, desc, cx);
  plan_attack_940(st, cx);
  for (int i = 0; i < n; ++i) {
    out.push_back({i, score_option_940(st, cx, desc[i])});
  }
  return out;
}

static std::vector<MetaScoredAction> rank_actions_heuristic(const GameState& st,
                                                           uint64_t seed) {
  uint8_t mask[RL_MAX_ACTIONS];
  int n = std::max(rl_legal_mask(st, mask), 1);
  const int me = st.yourIndex;
  std::vector<MetaScoredAction> out;
  out.reserve(n);
  for (int a = 0; a < n; ++a) {
    GameState cs = st;
    uint64_t rng = seed + static_cast<uint64_t>(a + 1) * 0x9e3779b97f4a7c15ULL;
    rl_step(cs, a, rng);
    out.push_back({a, value_for_player(cs, me)});
  }
  return out;
}

static std::vector<MetaScoredAction> rank_actions_random(const GameState& st,
                                                        uint64_t seed) {
  uint8_t mask[RL_MAX_ACTIONS];
  int n = std::max(rl_legal_mask(st, mask), 1);
  std::vector<MetaScoredAction> out;
  out.reserve(n);
  uint64_t rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
  for (int a = 0; a < n; ++a) {
    out.push_back({a, static_cast<double>(rl_rand(rng)) /
                          static_cast<double>(std::numeric_limits<uint64_t>::max())});
  }
  return out;
}

static std::vector<MetaScoredAction> rank_actions_inner(const GameState& st,
                                                       int inner_mode,
                                                       uint64_t seed) {
  if (inner_mode == PPO_OPP_940 || inner_mode == PPO_OPP_BEAM_940)
    return rank_actions_940(st);
  if (inner_mode == PPO_OPP_HEURISTIC)
    return rank_actions_heuristic(st, seed);
  return rank_actions_random(st, seed);
}

static void sort_ranked_actions(std::vector<MetaScoredAction>& actions) {
  std::sort(actions.begin(), actions.end(), [](const auto& a, const auto& b) {
    if (a.score != b.score) return a.score > b.score;
    return a.action < b.action;
  });
}

static int best_inner_action(const GameState& st, int inner_mode, uint64_t seed) {
  std::vector<MetaScoredAction> ranked = rank_actions_inner(st, inner_mode, seed);
  if (ranked.empty()) return 0;
  sort_ranked_actions(ranked);
  return ranked.front().action;
}

static void advance_opponents_with_inner(GameState& st, int root_player,
                                         int inner_mode, uint64_t& seed,
                                         int max_steps = 256) {
  int steps = 0;
  while (st.result < 0 && st.yourIndex != root_player && steps < max_steps) {
    int a = best_inner_action(st, inner_mode, seed);
    rl_step(st, a, seed);
    ++steps;
  }
}

struct BeamState {
  GameState state;
  double value = 0.0;
};

}  // namespace

int rl_native_940_action(const GameState& st) {
  return native_940_action_impl(st);
}

static long long elapsed_profile_ns(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end);

int rl_meta_beam_action(const GameState& st, int inner_mode, int beam_width,
                        int depth, uint64_t seed) {
  if (st.result >= 0) return 0;
  const int root_player = st.yourIndex;
  const int width = std::max(1, beam_width);
  const int search_depth = std::max(1, depth);
  uint8_t mask[RL_MAX_ACTIONS];
  const int n_root = std::max(rl_legal_mask(st, mask), 1);
  if (n_root <= 1) return 0;

  int best_root = 0;
  double best_value = -std::numeric_limits<double>::infinity();
  for (int root_action = 0; root_action < n_root; ++root_action) {
    uint64_t root_seed =
        (seed ? seed : 0x9e3779b97f4a7c15ULL) ^
        (static_cast<uint64_t>(root_action + 1) * 0xbf58476d1ce4e5b9ULL);
    GameState first = st;
    rl_step(first, root_action, root_seed);
    advance_opponents_with_inner(first, root_player, inner_mode, root_seed);
    std::vector<BeamState> beam{{std::move(first), 0.0}};

    for (int d = 1; d < search_depth; ++d) {
      std::vector<BeamState> next;
      next.reserve(static_cast<size_t>(width) * beam.size());
      for (const BeamState& b : beam) {
        if (b.state.result >= 0) {
          next.push_back({b.state, value_for_player(b.state, root_player)});
          continue;
        }
        if (b.state.yourIndex != root_player) {
          GameState cs = b.state;
          uint64_t s2 = root_seed + static_cast<uint64_t>(d + 17);
          advance_opponents_with_inner(cs, root_player, inner_mode, s2);
          next.push_back({std::move(cs), value_for_player(cs, root_player)});
          continue;
        }
        std::vector<MetaScoredAction> ranked =
            rank_actions_inner(b.state, inner_mode,
                               root_seed + static_cast<uint64_t>(d + 1));
        sort_ranked_actions(ranked);
        int take = std::min(width, static_cast<int>(ranked.size()));
        for (int i = 0; i < take; ++i) {
          GameState cs = b.state;
          uint64_t s2 = root_seed ^
                        (static_cast<uint64_t>(d + 1) * 0x94d049bb133111ebULL) ^
                        static_cast<uint64_t>(ranked[i].action + 1);
          rl_step(cs, ranked[i].action, s2);
          advance_opponents_with_inner(cs, root_player, inner_mode, s2);
          next.push_back({std::move(cs), value_for_player(cs, root_player)});
        }
      }
      if (next.empty()) break;
      std::sort(next.begin(), next.end(), [](const BeamState& a, const BeamState& b) {
        return a.value > b.value;
      });
      if (static_cast<int>(next.size()) > width) next.resize(width);
      beam = std::move(next);
    }

    double value = -std::numeric_limits<double>::infinity();
    for (const BeamState& b : beam)
      value = std::max(value, value_for_player(b.state, root_player));
    if (value > best_value) {
      best_value = value;
      best_root = root_action;
    }
  }
  return best_root;
}

namespace {

SmallVec<int, 64> public_zone_cards_for_observer(const GameState& st,
                                                 int observer) {
  const Player& p = st.players[1 - std::clamp(observer, 0, 1)];
  SmallVec<int, 64> cards;
  const auto append = [&](int card) {
    if (card > 0) cards.push_back(card);
  };
  if (p.handKnown) {
    for (int card : p.hand) append(card);
  } else {
    for (int card : p.handKnownCards) append(card);
  }
  if (!p.ownDeckInspected) {
    if (p.deckKnown) {
      for (int card : p.deck) append(card);
    } else {
      const int count =
          std::min<int>(p.deck.size(), p.deckKnownMask.size());
      for (int i = 0; i < count; ++i) {
        if (p.deckKnownMask[static_cast<std::size_t>(i)])
          append(p.deck[static_cast<std::size_t>(i)]);
      }
      for (int card : p.deckKnownCards) append(card);
    }
  }
  for (int card : p.discard) append(card);
  if (p.prizesKnown) {
    for (int card : p.prizes) append(card);
  } else {
    const int count = std::min<int>(
        p.prizes.size(),
        std::max(p.prizesKnownMask.size(), p.prizeFaceUp.size()));
    for (int i = 0; i < count; ++i) {
      const bool mask_known =
          i < static_cast<int>(p.prizesKnownMask.size()) &&
          p.prizesKnownMask[static_cast<std::size_t>(i)];
      const bool face_up =
          i < static_cast<int>(p.prizeFaceUp.size()) &&
          p.prizeFaceUp[static_cast<std::size_t>(i)];
      if (mask_known || face_up) append(p.prizes[static_cast<std::size_t>(i)]);
    }
    for (int card : p.prizesKnownCards) append(card);
  }
  return cards;
}

template <typename T>
bool write_public_card_memory(const SmallVec<int, 64>& memory, T* out) {
  std::fill(out, out + PPO_DECK_SLOTS, static_cast<T>(0));
  bool ok = true;
  const int count =
      std::min<int>(PPO_DECK_SLOTS, static_cast<int>(memory.size()));
  for (int i = 0; i < count; ++i) {
    const int card = memory[static_cast<std::size_t>(i)];
    if (card > static_cast<int>(std::numeric_limits<T>::max())) ok = false;
    out[i] = static_cast<T>(card);
  }
  return ok;
}

std::array<int, PPO_PUBLIC_EVENT_WIDTH> empty_public_event() {
  std::array<int, PPO_PUBLIC_EVENT_WIDTH> event{};
  for (int column = 6; column <= 9; ++column) event[column] = -1;
  return event;
}

template <typename T>
bool write_public_events(const PpoPublicEventRing& ring, int observer,
                         int current_turn, T* out) {
  std::memset(out, 0,
              PPO_PUBLIC_EVENT_SLOTS * PPO_PUBLIC_EVENT_WIDTH * sizeof(T));
  bool ok = true;
  const int first_out = PPO_PUBLIC_EVENT_SLOTS - ring.size;
  for (int i = 0; i < ring.size; ++i) {
    const auto& event = ring.events[static_cast<std::size_t>(
        (ring.begin + i) % PPO_PUBLIC_EVENT_SLOTS)];
    T* row = out + (first_out + i) * PPO_PUBLIC_EVENT_WIDTH;
    const auto checked_store = [&](int column, int value) {
      if constexpr (sizeof(T) < sizeof(int)) {
        if (value < static_cast<int>(std::numeric_limits<T>::min()) ||
            value > static_cast<int>(std::numeric_limits<T>::max()))
          ok = false;
      }
      row[column] = static_cast<T>(value);
    };
    row[0] = static_cast<T>(event[0]);
    row[1] = static_cast<T>(event[1] < 0 ? 0 : (event[1] == observer ? 1 : 2));
    row[2] = static_cast<T>(event[2] < 0 ? 0 : (event[2] == observer ? 1 : 2));
    checked_store(3, event[3]);
    checked_store(4, event[4]);
    checked_store(5, event[5]);
    row[6] = static_cast<T>(event[6] < 0 ? 0 : event[6] + 1);
    row[7] = static_cast<T>(event[7] < 0 ? 0 : event[7] + 1);
    row[8] = static_cast<T>(event[8] < 0 ? 0 : event[8] + 1);
    row[9] = static_cast<T>(event[9] < 0 ? 0 : event[9] + 1);
    checked_store(10, event[10]);
    row[11] = static_cast<T>(
        std::clamp(current_turn - event[11], 0, 32767));
  }
  return ok;
}

}  // namespace

PpoBatchEnv::PpoBatchEnv(std::vector<int> deck0, std::vector<int> deck1, int n,
                         uint64_t seed, int max_steps, double prize_weight,
                         int learner_seat, int opponent_mode, int reward_mode,
                         int threads, bool persistent_public_card_memory,
                         bool public_event_history)
    : deck0_(std::move(deck0)), deck1_(std::move(deck1)),
      persistent_public_card_memory_(persistent_public_card_memory),
      public_event_history_enabled_(public_event_history),
      threads_(threads),
      max_steps_(std::max(1, max_steps)),
      prize_weight_(std::clamp(prize_weight, 0.0, 1.0)),
      learner_seat_(learner_seat),
      opponent_mode_(opponent_mode),
      reward_mode_(std::clamp(reward_mode, PPO_REWARD_TERMINAL,
                              PPO_REWARD_TERMINAL_PLUS_DELTA)) {
  rngs_ = make_game_rngs(seed, n);
  games_.resize(n);
  opts_.resize(n);
  episode_len_.assign(n, 0);
  last_prize_score_.assign(n, 0.0f);
  if (persistent_public_card_memory_ || public_event_history_enabled_) {
    public_card_memory_.resize(static_cast<std::size_t>(n));
    public_card_snapshot_.resize(static_cast<std::size_t>(n));
  }
  if (public_event_history_enabled_)
    public_event_history_.resize(static_cast<std::size_t>(n));
  active_assignments_.resize(n);
  completed_assignments_.resize(n);
  lane_episode_counters_.assign(n, 0);
  reset_all();
}

namespace {
uint64_t assignment_mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

void validate_assignment_deck(const std::vector<int>& deck,
                              std::size_t candidate_index, int seat) {
  constexpr std::size_t kDeckCards = 60;
  constexpr int kMaxPrintedNameCopies = 4;
  const std::string context =
      "assignment plan candidate " + std::to_string(candidate_index) +
      " deck" + std::to_string(seat);
  if (deck.size() != kDeckCards) {
    throw std::invalid_argument(
        context + " must contain exactly 60 cards (got " +
        std::to_string(deck.size()) + ")");
  }

  bool has_basic_pokemon = false;
  int ace_spec_count = 0;
  std::unordered_map<std::string, int> printed_name_counts;
  for (std::size_t slot = 0; slot < deck.size(); ++slot) {
    const int card_id = deck[slot];
    if (card_id <= 0) {
      throw std::invalid_argument(
          context + " has non-positive card id " +
          std::to_string(card_id) + " at slot " + std::to_string(slot));
    }
    const CardInfo* card = find_card(card_id);
    if (card == nullptr) {
      throw std::invalid_argument(
          context + " has unknown native catalog card id " +
          std::to_string(card_id) + " at slot " + std::to_string(slot));
    }
    // CardInfo::name is the generated native catalog's canonical printed-name
    // copy group. Only Basic Energy is exempt from the normal four-copy rule.
    if (card->cardType != BASIC_ENERGY) {
      if (card->name == nullptr || card->name[0] == '\0') {
        throw std::invalid_argument(
            context + " card id " + std::to_string(card_id) +
            " lacks native printed-name metadata");
      }
      const int copies = ++printed_name_counts[card->name];
      if (copies > kMaxPrintedNameCopies) {
        throw std::invalid_argument(
            context + " exceeds the four-copy printed-name limit for \"" +
            card->name + "\"");
      }
    }
    if (card->aceSpec && ++ace_spec_count > 1) {
      throw std::invalid_argument(
          context + " contains more than one ACE SPEC card");
    }
    has_basic_pokemon |= card->cardType == POKEMON && card->basic;
  }
  if (!has_basic_pokemon) {
    throw std::invalid_argument(
        context + " must contain at least one Basic Pokemon");
  }
}
}  // namespace

void PpoBatchEnv::publish_assignment_plan(PpoAssignmentPlan plan) {
  if (plan.plan_token == 0 || plan.distribution_token == 0 ||
      plan.candidates.empty()) {
    throw std::invalid_argument("assignment plan requires nonzero IDs and candidates");
  }
  double total = 0.0;
  for (std::size_t candidate_index = 0;
       candidate_index < plan.candidates.size(); ++candidate_index) {
    auto& candidate = plan.candidates[candidate_index];
    if (candidate.entry.provider1_token == 0) candidate.entry.provider1_token = candidate.entry.provider_token;
    if (candidate.entry.policy1_token == 0) candidate.entry.policy1_token = candidate.entry.policy_token;
    if (candidate.entry.version1_token == 0) candidate.entry.version1_token = candidate.entry.version_token;
    if (!std::isfinite(candidate.weight) || candidate.weight <= 0.0 ||
        candidate.entry.entry_token == 0 || candidate.entry.provider_token == 0 ||
        candidate.entry.policy_token == 0 || candidate.entry.version_token == 0 ||
        candidate.entry.provider1_token == 0 || candidate.entry.policy1_token == 0 ||
        candidate.entry.version1_token == 0 ||
        candidate.entry.deck0.empty() || candidate.entry.deck1.empty() ||
        candidate.entry.trainable_owner_mask > 3 ||
        candidate.entry.learner_seat < -1 || candidate.entry.learner_seat > 1) {
      throw std::invalid_argument("invalid assignment plan candidate");
    }
    // Validate immutable deck artifacts before the plan can become visible to
    // resets. Do not construct a game here: malformed setup data must fail on
    // the publishing thread rather than entering setup or a worker reset.
    validate_assignment_deck(candidate.entry.deck0, candidate_index, 0);
    validate_assignment_deck(candidate.entry.deck1, candidate_index, 1);
    total += candidate.weight;
  }
  if (!std::isfinite(total)) {
    throw std::invalid_argument("assignment plan weights overflow");
  }
  plan.total_weight = total;
  plan.cumulative_weights.clear();
  plan.cumulative_weights.reserve(plan.candidates.size());
  double cumulative = 0.0;
  for (const auto& candidate : plan.candidates) {
    cumulative += candidate.weight;
    plan.cumulative_weights.push_back(cumulative);
  }
  auto frozen = std::make_shared<const PpoAssignmentPlan>(std::move(plan));
  assignment_plan_.store(std::move(frozen), std::memory_order_release);
  dynamic_assignments_.store(true, std::memory_order_release);
}

void PpoBatchEnv::reset_unassigned_lanes() {
  if (!dynamic_assignments_.load(std::memory_order_acquire))
    throw std::runtime_error("no assignment plan is active");
  EnvPool::instance().run(size(), threads_, [&](int i) {
    if (active_assignments_[i].metadata.plan_token == 0) reset(i);
  });
}

PpoBatchEnv::InstalledAssignment PpoBatchEnv::sample_assignment(
    int lane, uint64_t episode_counter) const {
  auto plan = assignment_plan_.load(std::memory_order_acquire);
  if (!plan || plan->candidates.empty()) return {};
  const uint64_t lane_id = plan->lane_id_offset + static_cast<uint64_t>(lane);
  // Lane IDs and episode counters occupy distinct hash domains.  Mixing both
  // with the same function and XORing them makes (lane=0, episode=0) collide
  // with (lane=1, episode=1), because equal terms cancel before the final mix.
  const uint64_t lane_component =
      assignment_mix(lane_id ^ 0x243f6a8885a308d3ULL);
  const uint64_t episode_component =
      assignment_mix(episode_counter ^ 0x13198a2e03707344ULL);
  uint64_t random = assignment_mix(plan->plan_token ^ plan->base_seed ^
      lane_component ^ episode_component);
  const PpoAssignmentCandidate* selected = &plan->candidates.front();
  if (plan->candidates.size() > 1) {
    const double target =
        (static_cast<double>(random >> 11) * 0x1.0p-53) * plan->total_weight;
    const auto found = std::upper_bound(plan->cumulative_weights.begin(),
                                        plan->cumulative_weights.end(), target);
    const auto index = std::min<std::size_t>(
        static_cast<std::size_t>(found - plan->cumulative_weights.begin()),
        plan->candidates.size() - 1);
    selected = &plan->candidates[index];
  }
  InstalledAssignment installed;
  installed.entry =
      std::shared_ptr<const PpoAssignmentEntry>(plan, &selected->entry);
  auto& out = installed.metadata;
  out.plan_token = plan->plan_token;
  out.distribution_token = plan->distribution_token;
  out.assignment_token = assignment_mix(
      plan->plan_token ^ selected->entry.entry_token ^ lane_component ^
      episode_component);
  out.entry_token = selected->entry.entry_token;
  out.provider_token = selected->entry.provider_token;
  out.policy_token = selected->entry.policy_token;
  out.version_token = selected->entry.version_token;
  out.provider1_token = selected->entry.provider1_token;
  out.policy1_token = selected->entry.policy1_token;
  out.version1_token = selected->entry.version1_token;
  out.deck0_token = selected->entry.deck0_token;
  out.deck1_token = selected->entry.deck1_token;
  out.episode_counter = episode_counter;
  out.trainable_owner_mask = selected->entry.trainable_owner_mask;
  out.learner_seat = static_cast<int8_t>(selected->entry.learner_seat);
  out.fixed_actor = static_cast<int8_t>(selected->entry.fixed_actor);
  return installed;
}

const PpoAssignmentMetadata& PpoBatchEnv::active_assignment_at(int i) const {
  return active_assignments_.at(static_cast<std::size_t>(i)).metadata;
}
const PpoAssignmentMetadata& PpoBatchEnv::completed_assignment_at(int i) const {
  return completed_assignments_.at(static_cast<std::size_t>(i));
}
uint64_t PpoBatchEnv::lane_episode_counter(int i) const {
  return lane_episode_counters_.at(static_cast<std::size_t>(i));
}
void PpoBatchEnv::restore_lane_episode_counter(int i, uint64_t counter) {
  lane_episode_counters_.at(static_cast<std::size_t>(i)) = counter;
}
const std::vector<int>& PpoBatchEnv::deck_at(int lane, int seat) const {
  if (lane < 0 || lane >= size() || (seat != 0 && seat != 1)) {
    throw std::out_of_range("assignment deck lane/seat out of range");
  }
  const auto& assignment = active_assignments_[static_cast<std::size_t>(lane)];
  if (assignment.metadata.plan_token == 0 || assignment.entry == nullptr)
    return seat == 0 ? deck0_ : deck1_;
  return seat == 0 ? assignment.entry->deck0 : assignment.entry->deck1;
}

void PpoBatchEnv::update_public_card_memory(int i, int observer) {
  if (!persistent_public_card_memory_ && !public_event_history_enabled_) return;
  auto& by_observer = public_card_memory_[static_cast<std::size_t>(i)];
  const int seat = std::clamp(observer, 0, 1);
  auto& remembered = by_observer[static_cast<std::size_t>(seat)];
  if (remembered.size() == PPO_DECK_SLOTS) return;
  auto visible = public_zone_cards_for_observer(games_[i], seat);
  auto& previous = public_card_snapshot_[static_cast<std::size_t>(i)]
                                        [static_cast<std::size_t>(seat)];
  if (visible == previous) return;
  previous = visible;
  std::sort(visible.begin(), visible.end());
  if (std::includes(remembered.begin(), remembered.end(), visible.begin(),
                    visible.end())) {
    return;
  }
  SmallVec<int, 64> added;
  std::set_difference(visible.begin(), visible.end(), remembered.begin(),
                      remembered.end(), std::back_inserter(added));
  const std::size_t capacity = remembered.size() >= PPO_DECK_SLOTS
      ? 0
      : static_cast<std::size_t>(PPO_DECK_SLOTS) - remembered.size();
  while (added.size() > capacity) added.pop_back();
  SmallVec<int, 64> merged = remembered;
  for (int card : added) merged.push_back(card);
  std::sort(merged.begin(), merged.end());
  if (public_event_history_enabled_) {
    for (int card : added) {
      auto event = empty_public_event();
      event[0] = PPO_EVENT_REVEAL_CARD;
      event[1] = 1 - seat;
      event[2] = 1 - seat;
      event[3] = card;
      event[11] = games_[i].turn;
      public_event_history_[static_cast<std::size_t>(i)].push(event);
    }
  }
  remembered = std::move(merged);
}

void PpoBatchEnv::record_public_action(int i, const Action& selected) {
  const GameState& st = games_[i];
  auto event = empty_public_event();
  event[1] = st.yourIndex;
  event[2] = st.yourIndex;
  event[3] = selected.cardId;
  event[5] = selected.attackId;
  event[8] = selected.targetArea;
  event[9] = selected.targetIndex;
  event[11] = st.turn;
  const Player& player = st.players[st.yourIndex];
  const auto target_card = [&]() {
    if (selected.targetArea == AREA_ACTIVE && player.activeKnown)
      return player.active.id;
    if (selected.targetArea == AREA_BENCH && selected.targetIndex >= 0 &&
        selected.targetIndex < static_cast<int>(player.bench.size()))
      return player.bench[static_cast<std::size_t>(selected.targetIndex)].id;
    if (selected.targetArea == AREA_STADIUM && !st.stadium.empty())
      return st.stadium[0];
    return 0;
  };
  switch (selected.kind) {
    case ACT_END:
      event[0] = PPO_EVENT_END_TURN;
      break;
    case ACT_ATTACH:
      event[0] = PPO_EVENT_ATTACH;
      event[4] = target_card();
      event[6] = AREA_HAND;
      event[7] = selected.targetArea;
      break;
    case ACT_PLAY_BASIC:
      event[0] = PPO_EVENT_PLAY_BASIC;
      event[6] = AREA_HAND;
      event[7] = AREA_BENCH;
      break;
    case ACT_PLAY_TRAINER:
      event[0] = PPO_EVENT_PLAY_TRAINER;
      event[6] = AREA_HAND;
      break;
    case ACT_EVOLVE:
      event[0] = PPO_EVENT_EVOLVE;
      event[4] = target_card();
      event[6] = AREA_HAND;
      event[7] = selected.targetArea;
      break;
    case ACT_ATTACK:
      event[0] = PPO_EVENT_ATTACK;
      event[3] = player.activeKnown ? player.active.id : 0;
      break;
    case ACT_RETREAT:
      event[0] = PPO_EVENT_RETREAT;
      event[3] = player.activeKnown ? player.active.id : 0;
      break;
    case ACT_ABILITY:
      event[0] = PPO_EVENT_ABILITY;
      event[3] = target_card();
      break;
    case ACT_DISCARD_INPLAY:
      event[0] = PPO_EVENT_DISCARD_INPLAY;
      event[3] = target_card();
      break;
    case ACT_SETUP_ACTIVE:
      event[0] = PPO_EVENT_SETUP_ACTIVE;
      event[6] = AREA_HAND;
      event[7] = AREA_ACTIVE;
      break;
  }
  if (event[0] != PPO_EVENT_PAD)
    public_event_history_[static_cast<std::size_t>(i)].push(event);
}

void PpoBatchEnv::record_public_transition(
    int i, int before_turn, int before_result,
    const std::array<int, 2>& before_prizes) {
  if (!public_event_history_enabled_) return;
  const GameState& st = games_[i];
  auto& history = public_event_history_[static_cast<std::size_t>(i)];
  for (int player = 0; player < 2; ++player) {
    const int taken = before_prizes[static_cast<std::size_t>(player)] -
                      st.players[player].prizeCount;
    if (taken > 0) {
      auto event = empty_public_event();
      event[0] = PPO_EVENT_PRIZE_TAKEN;
      event[1] = player;
      event[2] = player;
      event[10] = taken;
      event[11] = st.turn;
      history.push(event);
    }
  }
  if (st.turn != before_turn && st.result < 0) {
    auto event = empty_public_event();
    event[0] = PPO_EVENT_TURN_START;
    event[1] = st.yourIndex;
    event[2] = st.yourIndex;
    event[11] = st.turn;
    history.push(event);
  }
  if (before_result < 0 && st.result >= 0) {
    auto event = empty_public_event();
    event[0] = PPO_EVENT_GAME_END;
    event[10] = st.result;
    event[11] = st.turn;
    history.push(event);
  }
}

void PpoBatchEnv::step_recorded(int i, int action) {
  if (!public_event_history_enabled_) {
    rl_step_cached(games_[i], opts_[i], action, rngs_[i], &opts_[i]);
    return;
  }
  const int before_turn = games_[i].turn;
  const int before_result = games_[i].result;
  const std::array<int, 2> before_prizes = {
      games_[i].players[0].prizeCount, games_[i].players[1].prizeCount};
  if (!opts_[i].pending && action >= 0 && action < opts_[i].n &&
      action < static_cast<int>(opts_[i].descriptors.size())) {
    const Action selected = descriptor_to_action(opts_[i].descriptors[action]);
    record_public_action(i, selected);
    apply(games_[i], selected);
    advance_to_agent(games_[i], rngs_[i], &opts_[i]);
  } else {
    rl_step_cached(games_[i], opts_[i], action, rngs_[i], &opts_[i]);
  }
  record_public_transition(i, before_turn, before_result, before_prizes);
}

void PpoBatchEnv::reset(int i) {
  if (public_event_history_enabled_)
    public_event_history_[static_cast<std::size_t>(i)].clear();
  if (persistent_public_card_memory_ || public_event_history_enabled_) {
    auto& memory = public_card_memory_[static_cast<std::size_t>(i)];
    memory[0].clear();
    memory[1].clear();
    auto& snapshot = public_card_snapshot_[static_cast<std::size_t>(i)];
    snapshot[0].clear();
    snapshot[1].clear();
  }
  if (dynamic_assignments_.load(std::memory_order_acquire)) {
    active_assignments_[i] = sample_assignment(i, lane_episode_counters_[i]++);
  }
  uint64_t s = rl_rand(rngs_[i]);
  const auto& assignment = active_assignments_[i];
  const bool assigned = assignment.entry != nullptr;
  const auto& lane_deck0 = assigned ? assignment.entry->deck0 : deck0_;
  const auto& lane_deck1 = assigned ? assignment.entry->deck1 : deck1_;
  games_[i] = new_game(lane_deck0, lane_deck1, s ? s : 1);
  episode_len_[i] = 0;
  advance_to_agent(games_[i], rngs_[i], &opts_[i]);
  advance_to_learner_or_done(i);
  if (persistent_public_card_memory_ || public_event_history_enabled_) {
    update_public_card_memory(i, 0);
    update_public_card_memory(i, 1);
  }
  last_prize_score_[i] =
      static_cast<float>(prize_score(games_[i], reward_player(games_[i], i)));
}

int PpoBatchEnv::random_action(const GameState& st, uint64_t& rng) {
  uint8_t mask[RL_MAX_ACTIONS];
  int n = rl_legal_mask(st, mask);
  return n > 0 ? static_cast<int>(rl_rand(rng) % n) : 0;
}

int PpoBatchEnv::heuristic_action(const GameState& st,
                                  const RlOptionSet& opts, uint64_t rng) {
  int n = std::max(opts.n, 1);
  const int me = st.yourIndex;
  int best = 0;
  float best_v = -std::numeric_limits<float>::infinity();
  for (int a = 0; a < n; ++a) {
    GameState cs = st;
    uint64_t step_rng = rng;  // snapshot: rollouts do not advance the stream
    rl_step_cached(cs, opts, a, step_rng);
    float v = 0.0f;
    if (cs.result >= 0) {
      v = cs.result == 2 ? 0.0f : (cs.result == me ? 1.0f : -1.0f);
    } else {
      v = rl_heuristic_value(cs);
      if (cs.yourIndex != me) v = -v;
    }
    if (v > best_v) {
      best_v = v;
      best = a;
    }
  }
  return best;
}

int PpoBatchEnv::opponent_action(const GameState& st, const RlOptionSet& opts,
                                 uint64_t& rng) {
  if (opponent_mode_ == PPO_OPP_HEURISTIC)
    return heuristic_action(st, opts, rng);
  if (opponent_mode_ == PPO_OPP_940) return rl_native_940_action(st);
  if (opponent_mode_ == PPO_OPP_BEAM_940)
    return rl_meta_beam_action(st, PPO_OPP_940, 4, 6, rng);
  return random_action(st, rng);
}

int PpoBatchEnv::opponent_action_cached(const GameState& st,
                                        const RlOptionSet& opts,
                                        uint64_t& rng, int opponent_mode,
                                        const PpoAssignmentEntry* assignment) {
  if (routed_opponent_action_callback_ != nullptr && assignment != nullptr) {
    const bool seat1 = st.yourIndex == 1;
    return routed_opponent_action_callback_(
        routed_opponent_action_context_,
        seat1 ? assignment->provider1_token : assignment->provider_token,
        seat1 ? assignment->policy1_token : assignment->policy_token,
        seat1 ? assignment->version1_token : assignment->version_token,
        st, opts);
  }
  if (opponent_action_callback_ != nullptr) {
    return opponent_action_callback_(opponent_action_context_, st, opts);
  }
  if (opponent_mode == PPO_OPP_940) return native_940_action_from_options(st, opts);
  if (opponent_mode == PPO_OPP_RANDOM) return opts.n > 0 ? static_cast<int>(rl_rand(rng) % opts.n) : 0;
  if (opponent_mode == PPO_OPP_HEURISTIC) return heuristic_action(st, opts, rng);
  if (opponent_mode == PPO_OPP_BEAM_940)
    return rl_meta_beam_action(st, PPO_OPP_940, 4, 6, rng);
  return opts.n > 0 ? static_cast<int>(rl_rand(rng) % opts.n) : 0;
}

void PpoBatchEnv::advance_to_learner_or_done(int i) {
  const bool assigned = active_assignments_[i].metadata.plan_token != 0;
  const int mode = assigned ? active_assignments_[i].entry->opponent_mode : opponent_mode_;
  const int seat = assigned ? active_assignments_[i].metadata.learner_seat : learner_seat_;
  if (mode == PPO_OPP_SELF || seat < 0) return;
  while (games_[i].result < 0 && games_[i].yourIndex != seat &&
         episode_len_[i] < max_steps_) {
    const PpoAssignmentEntry* entry =
        assigned ? active_assignments_[i].entry.get() : nullptr;
    int a = opponent_action_cached(games_[i], opts_[i], rngs_[i], mode,
                                   entry);
    if (a < 0 || a >= opts_[i].n) a = 0;
    step_recorded(i, a);
    ++episode_len_[i];
  }
}

void PpoBatchEnv::advance_to_learner_or_done_profiled(int i,
                                                      PpoStepProfile& profile) {
  const bool assigned = active_assignments_[i].metadata.plan_token != 0;
  const int mode = assigned ? active_assignments_[i].entry->opponent_mode : opponent_mode_;
  const int seat = assigned ? active_assignments_[i].metadata.learner_seat : learner_seat_;
  if (mode == PPO_OPP_SELF || seat < 0) return;
  while (games_[i].result < 0 && games_[i].yourIndex != seat &&
         episode_len_[i] < max_steps_) {
    auto t0 = std::chrono::steady_clock::now();
    const PpoAssignmentEntry* entry =
        assigned ? active_assignments_[i].entry.get() : nullptr;
    int a = opponent_action_cached(games_[i], opts_[i], rngs_[i], mode,
                                   entry);
    if (a < 0 || a >= opts_[i].n) a = 0;
    auto t1 = std::chrono::steady_clock::now();
    profile.opponent_action_ns += elapsed_profile_ns(t0, t1);

    t0 = std::chrono::steady_clock::now();
    step_recorded(i, a);
    t1 = std::chrono::steady_clock::now();
    profile.opponent_step_ns += elapsed_profile_ns(t0, t1);
    ++episode_len_[i];
  }
}

void PpoBatchEnv::reset_all() {
  EnvPool::instance().run(size(), threads_, [&](int i) { reset(i); });
}

int PpoBatchEnv::reward_player(const GameState& st, int lane) const {
  const bool assigned = active_assignments_[lane].metadata.plan_token != 0;
  const int mode = assigned ? active_assignments_[lane].entry->opponent_mode : opponent_mode_;
  const int seat = assigned ? active_assignments_[lane].metadata.learner_seat : learner_seat_;
  if (mode != PPO_OPP_SELF && seat >= 0) {
    return seat;
  }
  return st.yourIndex;
}

double PpoBatchEnv::prize_score(const GameState& st, int player) const {
  const int p = std::clamp(player, 0, 1);
  const int own = st.players[p].prizeCount;
  const int opp = st.players[1 - p].prizeCount;
  return std::clamp((static_cast<double>(opp) - own) / 6.0, -1.0, 1.0);
}

float PpoBatchEnv::terminal_reward(const GameState& st, int player) const {
  double terminal = 0.0;
  if (st.result == 2) {
    terminal = 0.0;
  } else {
    terminal = st.result == player ? 1.0 : -1.0;
  }
  double prize = prize_score(st, player);
  return static_cast<float>((1.0 - prize_weight_) * terminal +
                            prize_weight_ * prize);
}

float PpoBatchEnv::transition_reward(const GameState& st, int player, int i,
                                     bool terminal, bool truncated) const {
  if (truncated) return 0.0f;
  if (reward_mode_ == PPO_REWARD_TERMINAL) {
    return terminal ? terminal_reward(st, player) : 0.0f;
  }

  const double current_prize = prize_score(st, player);
  const double delta = current_prize - static_cast<double>(last_prize_score_[i]);
  const double dense = prize_weight_ * delta;
  if (reward_mode_ == PPO_REWARD_PRIZE_DELTA) {
    return static_cast<float>(dense);
  }

  double terminal_bonus = 0.0;
  if (terminal && st.result != 2) {
    terminal_bonus = st.result == player ? 1.0 : -1.0;
  }
  return static_cast<float>(terminal_bonus + dense);
}

void PpoBatchEnv::observe(float* obs, uint8_t* mask, int32_t* player) const {
  int D = RL_OBS_DIM;
  EnvPool::instance().run(size(), threads_, [&](int i) {
    rl_encode_obs(games_[i], obs + i * D);
    fill_legal_mask(opts_[i], mask + i * RL_MAX_ACTIONS);
    player[i] = games_[i].yourIndex;
  });
}

void PpoBatchEnv::observe_ids(int32_t* in_play, int32_t* zones,
                              int32_t* player_counts, int32_t* player_status,
                              int32_t* global, int32_t* action_meta,
                              int32_t* action_options, int32_t* action_deck,
                              uint8_t* action_mask, int32_t* player,
                              int32_t* result,
                              int32_t* known_opponent_cards, int32_t* public_events) const {
  EnvPool::instance().run(size(), threads_, [&](int i) {
    const GameState& st = games_[i];
    fill_observation_ids(
        st,
        in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    fill_action_ids(
        st, action_id_view_from_options(opts_[i]),
        action_meta + i * ACTION_META_WIDTH,
        action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
        action_deck + i * STATE_ZONE_SLOTS,
        action_mask + i * RL_MAX_ACTIONS);
    player[i] = st.yourIndex;
    result[i] = st.result;
    if (known_opponent_cards != nullptr && persistent_public_card_memory_) {
      write_public_card_memory(
          public_card_memory_[static_cast<std::size_t>(i)]
                             [static_cast<std::size_t>(st.yourIndex)],
          known_opponent_cards + i * PPO_DECK_SLOTS);
    }
    if (public_events != nullptr && public_event_history_enabled_) {
      write_public_events(public_event_history_[static_cast<std::size_t>(i)],
                          st.yourIndex, st.turn,
                          public_events + i * PPO_PUBLIC_EVENT_SLOTS *
                                              PPO_PUBLIC_EVENT_WIDTH);
    }
  });
}

bool PpoBatchEnv::observe_ids16(
    int16_t* in_play, int16_t* zones, int16_t* player_counts,
    int16_t* player_status, int16_t* global, int16_t* action_meta,
    int16_t* action_options, int16_t* action_deck, uint8_t* action_mask,
    int32_t* player, int32_t* result,
    int16_t* known_opponent_cards, int16_t* public_events) const {
  std::atomic<bool> ok{true};
  EnvPool::instance().run(size(), threads_, [&](int i) {
    const GameState& st = games_[i];
    bool row_ok = fill_observation_ids16(
        st,
        in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    row_ok = fill_action_ids16(
                 st, action_id_view_from_options(opts_[i]),
                 action_meta + i * ACTION_META_WIDTH,
                 action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                 action_deck + i * STATE_ZONE_SLOTS,
                 action_mask + i * RL_MAX_ACTIONS) &&
             row_ok;
    player[i] = st.yourIndex;
    result[i] = st.result;
    if (known_opponent_cards != nullptr && persistent_public_card_memory_) {
      row_ok = write_public_card_memory(
                   public_card_memory_[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(st.yourIndex)],
                   known_opponent_cards + i * PPO_DECK_SLOTS) &&
               row_ok;
    }
    if (public_events != nullptr && public_event_history_enabled_) {
      row_ok = write_public_events(
                   public_event_history_[static_cast<std::size_t>(i)],
                   st.yourIndex, st.turn,
                   public_events + i * PPO_PUBLIC_EVENT_SLOTS *
                                       PPO_PUBLIC_EVENT_WIDTH) &&
               row_ok;
    }
    if (!row_ok) ok.store(false, std::memory_order_relaxed);
  });
  return ok.load(std::memory_order_relaxed);
}

void PpoBatchEnv::action_features(float* out) const {
  EnvPool::instance().run(size(), threads_, [&](int i) {
    encode_action_features_from_options(
        games_[i], opts_[i], out + i * RL_MAX_ACTIONS * PPO_ACTION_FEAT_DIM);
  });
}

void PpoBatchEnv::card_features(float* out) const {
  EnvPool::instance().run(size(), threads_, [&](int i) {
    rl_card_features(games_[i], out + i * PPO_CARD_SLOTS * PPO_CARD_FEAT_DIM);
  });
}

void PpoBatchEnv::deck_features(float* out) const {
  for (int i = 0; i < size(); ++i) {
    const int p = reward_player(games_[i], i);
    const auto& assignment = active_assignments_[i];
    const std::vector<int>& deck = assignment.metadata.plan_token != 0
        ? (p == 0 ? assignment.entry->deck0 : assignment.entry->deck1)
        : (p == 0 ? deck0_ : deck1_);
    rl_deck_features(deck, out + i * PPO_DECK_SLOTS * PPO_CARD_FEAT_DIM);
  }
}

void PpoBatchEnv::belief_features(float* out) const {
  for (int i = 0; i < size(); ++i) {
    rl_belief_features(games_[i],
                       out + i * PPO_BELIEF_SLOTS * PPO_CARD_FEAT_DIM);
  }
}

void PpoBatchEnv::belief_summary(float* out) const {
  for (int i = 0; i < size(); ++i) {
    rl_belief_summary(games_[i], out + i * PPO_BELIEF_SUMMARY_DIM);
  }
}

void PpoBatchEnv::step_rewards(const int* actions, float* reward, uint8_t* done,
                               int32_t* result, int32_t* episode_len) {
  EnvPool::instance().run(size(), threads_, [&](int i) {
    int actor = reward_player(games_[i], i);
    int n = opts_[i].n;
    int action = actions[i];
    if (action < 0 || action >= n) action = 0;
    step_recorded(i, action);
    ++episode_len_[i];
    advance_to_learner_or_done(i);
    if (persistent_public_card_memory_ || public_event_history_enabled_) {
      update_public_card_memory(i, 0);
      update_public_card_memory(i, 1);
    }

    int r = games_[i].result;
    bool truncated = false;
    if (r < 0 && episode_len_[i] >= max_steps_) {
      r = 2;
      truncated = true;
    }

    if (r >= 0) {
      reward[i] = transition_reward(games_[i], actor, i, true, truncated);
      done[i] = 1;
      result[i] = r;
      episode_len[i] = episode_len_[i];
      if (dynamic_assignments_.load(std::memory_order_acquire)) {
        active_assignments_[i].metadata.acting_seat = static_cast<int8_t>(actor);
        completed_assignments_[i] = active_assignments_[i].metadata;
      }
      reset(i);
    } else {
      reward[i] = transition_reward(games_[i], actor, i, false, false);
      done[i] = 0;
      result[i] = -1;
      episode_len[i] = episode_len_[i];
      last_prize_score_[i] =
          static_cast<float>(prize_score(games_[i], reward_player(games_[i], i)));
    }
  });
}

void PpoBatchEnv::step(const int* actions, float* obs, float* reward,
                       uint8_t* done, uint8_t* mask, int32_t* player,
                       int32_t* result, int32_t* episode_len) {
  step_rewards(actions, reward, done, result, episode_len);

  int D = RL_OBS_DIM;
  EnvPool::instance().run(size(), threads_, [&](int i) {
    rl_encode_obs(games_[i], obs + i * D);
    fill_legal_mask(opts_[i], mask + i * RL_MAX_ACTIONS);
    player[i] = games_[i].yourIndex;
  });
}

void PpoBatchEnv::step_ids(const int* actions, float* reward, uint8_t* done,
                           int32_t* result, int32_t* episode_len,
                           int32_t* in_play, int32_t* zones,
                           int32_t* player_counts, int32_t* player_status,
                           int32_t* global, int32_t* action_meta,
                           int32_t* action_options, int32_t* action_deck,
                           uint8_t* action_mask, int32_t* player,
                           int32_t* known_opponent_cards, int32_t* public_events) {
  step_rewards(actions, reward, done, result, episode_len);
  EnvPool::instance().run(size(), threads_, [&](int i) {
    const GameState& st = games_[i];
    fill_observation_ids(
        st,
        in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    fill_action_ids(
        st, action_id_view_from_options(opts_[i]),
        action_meta + i * ACTION_META_WIDTH,
        action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
        action_deck + i * STATE_ZONE_SLOTS,
        action_mask + i * RL_MAX_ACTIONS);
    player[i] = st.yourIndex;
    if (known_opponent_cards != nullptr && persistent_public_card_memory_) {
      write_public_card_memory(
          public_card_memory_[static_cast<std::size_t>(i)]
                             [static_cast<std::size_t>(st.yourIndex)],
          known_opponent_cards + i * PPO_DECK_SLOTS);
    }
    if (public_events != nullptr && public_event_history_enabled_) {
      write_public_events(public_event_history_[static_cast<std::size_t>(i)],
                          st.yourIndex, st.turn,
                          public_events + i * PPO_PUBLIC_EVENT_SLOTS *
                                              PPO_PUBLIC_EVENT_WIDTH);
    }
  });
}

bool PpoBatchEnv::step_ids16(
    const int* actions, float* reward, uint8_t* done, int32_t* result,
    int32_t* episode_len, int16_t* in_play, int16_t* zones,
    int16_t* player_counts, int16_t* player_status, int16_t* global,
    int16_t* action_meta, int16_t* action_options, int16_t* action_deck,
    uint8_t* action_mask, int32_t* player,
    int16_t* known_opponent_cards, int16_t* public_events) {
  step_rewards(actions, reward, done, result, episode_len);
  std::atomic<bool> ok{true};
  EnvPool::instance().run(size(), threads_, [&](int i) {
    const GameState& st = games_[i];
    bool row_ok = fill_observation_ids16(
        st,
        in_play + i * 2 * STATE_INPLAY_SLOTS * STATE_INPLAY_WIDTH,
        zones + i * 2 * STATE_ZONE_COUNT * STATE_ZONE_SLOTS,
        player_counts + i * 2 * 5, player_status + i * 2 * 5,
        global + i * STATE_GLOBAL_WIDTH);
    row_ok = fill_action_ids16(
                 st, action_id_view_from_options(opts_[i]),
                 action_meta + i * ACTION_META_WIDTH,
                 action_options + i * RL_MAX_ACTIONS * ACTION_OPTION_WIDTH,
                 action_deck + i * STATE_ZONE_SLOTS,
                 action_mask + i * RL_MAX_ACTIONS) &&
             row_ok;
    player[i] = st.yourIndex;
    if (known_opponent_cards != nullptr && persistent_public_card_memory_) {
      row_ok = write_public_card_memory(
                   public_card_memory_[static_cast<std::size_t>(i)]
                                      [static_cast<std::size_t>(st.yourIndex)],
                   known_opponent_cards + i * PPO_DECK_SLOTS) &&
               row_ok;
    }
    if (public_events != nullptr && public_event_history_enabled_) {
      row_ok = write_public_events(
                   public_event_history_[static_cast<std::size_t>(i)],
                   st.yourIndex, st.turn,
                   public_events + i * PPO_PUBLIC_EVENT_SLOTS *
                                       PPO_PUBLIC_EVENT_WIDTH) &&
               row_ok;
    }
    if (!row_ok) ok.store(false, std::memory_order_relaxed);
  });
  return ok.load(std::memory_order_relaxed);
}

void PpoBatchEnv::step_card_features(const int* actions, float* obs,
                                     float* reward, uint8_t* done,
                                     uint8_t* mask, int32_t* player,
                                     int32_t* result, int32_t* episode_len,
                                     float* action_features,
                                     float* card_features) {
  int D = RL_OBS_DIM;
  EnvPool::instance().run(size(), threads_, [&](int i) {
    int actor = reward_player(games_[i], i);
    int n = opts_[i].n;
    int action = actions[i];
    if (action < 0 || action >= n) action = 0;
    step_recorded(i, action);
    ++episode_len_[i];
    advance_to_learner_or_done(i);

    if (persistent_public_card_memory_ || public_event_history_enabled_) {
      update_public_card_memory(i, 0);
      update_public_card_memory(i, 1);
    }
    int r = games_[i].result;
    bool truncated = false;
    if (r < 0 && episode_len_[i] >= max_steps_) {
      r = 2;
      truncated = true;
    }

    if (r >= 0) {
      reward[i] = transition_reward(games_[i], actor, i, true, truncated);
      done[i] = 1;
      result[i] = r;
      episode_len[i] = episode_len_[i];
      reset(i);
    } else {
      reward[i] = transition_reward(games_[i], actor, i, false, false);
      done[i] = 0;
      result[i] = -1;
      episode_len[i] = episode_len_[i];
      last_prize_score_[i] =
          static_cast<float>(prize_score(games_[i], reward_player(games_[i], i)));
    }

    rl_encode_obs(games_[i], obs + i * D);
    const RlOptionSet& next_opts = opts_[i];
    fill_legal_mask(next_opts, mask + i * RL_MAX_ACTIONS);
    player[i] = games_[i].yourIndex;
    encode_action_features_from_options(
        games_[i], next_opts,
        action_features + i * RL_MAX_ACTIONS * PPO_ACTION_FEAT_DIM);
    rl_card_features(games_[i], card_features + i * PPO_CARD_SLOTS * PPO_CARD_FEAT_DIM);
  });
}

static long long elapsed_profile_ns(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

PpoStepProfile PpoBatchEnv::profile_step_card_features(const int* actions,
                                                       int repeats) {
  PpoStepProfile profile;
  profile.repeats = std::max(1, repeats);
  profile.envs = size();
  profile.env_steps = static_cast<long long>(profile.repeats) * profile.envs;

  const int n_envs = size();
  std::vector<float> obs(static_cast<size_t>(n_envs) * RL_OBS_DIM);
  std::vector<float> reward(n_envs);
  std::vector<uint8_t> done(n_envs);
  std::vector<uint8_t> mask(static_cast<size_t>(n_envs) * RL_MAX_ACTIONS);
  std::vector<int32_t> player(n_envs);
  std::vector<int32_t> result(n_envs);
  std::vector<int32_t> ep_len(n_envs);
  std::vector<float> action_features(
      static_cast<size_t>(n_envs) * RL_MAX_ACTIONS * PPO_ACTION_FEAT_DIM);
  std::vector<float> card_features(
      static_cast<size_t>(n_envs) * PPO_CARD_SLOTS * PPO_CARD_FEAT_DIM);

  auto t_total = std::chrono::steady_clock::now();
  for (int rep = 0; rep < profile.repeats; ++rep) {
    for (int i = 0; i < n_envs; ++i) {
      int actor = reward_player(games_[i], i);

      auto t0 = std::chrono::steady_clock::now();
      int n = opts_[i].n;
      auto t1 = std::chrono::steady_clock::now();
      profile.pre_options_ns += elapsed_profile_ns(t0, t1);

      int action = actions[i];
      if (action < 0 || action >= n) action = 0;

      int len_before = episode_len_[i];
      t0 = std::chrono::steady_clock::now();
      rl_step_cached_profiled(games_[i], opts_[i], action, rngs_[i], profile,
                              &opts_[i]);
      ++episode_len_[i];
      t1 = std::chrono::steady_clock::now();
      profile.learner_step_ns += elapsed_profile_ns(t0, t1);

      t0 = std::chrono::steady_clock::now();
      advance_to_learner_or_done_profiled(i, profile);
      t1 = std::chrono::steady_clock::now();
      profile.opponent_advance_ns += elapsed_profile_ns(t0, t1);

      int r = games_[i].result;
      bool truncated = false;
      if (r < 0 && episode_len_[i] >= max_steps_) {
        r = 2;
        truncated = true;
      }

      t0 = std::chrono::steady_clock::now();
      if (r >= 0) {
        reward[i] = transition_reward(games_[i], actor, i, true, truncated);
        done[i] = 1;
        result[i] = r;
        ep_len[i] = episode_len_[i];
        profile.opponent_steps += std::max(0, episode_len_[i] - len_before - 1);
        ++profile.terminal_resets;
        reset(i);
      } else {
        reward[i] = transition_reward(games_[i], actor, i, false, false);
        done[i] = 0;
        result[i] = -1;
        ep_len[i] = episode_len_[i];
        profile.opponent_steps += std::max(0, episode_len_[i] - len_before - 1);
        last_prize_score_[i] =
            static_cast<float>(prize_score(games_[i], reward_player(games_[i], i)));
      }
      t1 = std::chrono::steady_clock::now();
      profile.reward_reset_ns += elapsed_profile_ns(t0, t1);

      t0 = std::chrono::steady_clock::now();
      rl_encode_obs(games_[i], obs.data() + i * RL_OBS_DIM);
      player[i] = games_[i].yourIndex;
      t1 = std::chrono::steady_clock::now();
      profile.obs_ns += elapsed_profile_ns(t0, t1);

      t0 = std::chrono::steady_clock::now();
      const RlOptionSet& next_opts = opts_[i];
      t1 = std::chrono::steady_clock::now();
      profile.post_options_ns += elapsed_profile_ns(t0, t1);

      t0 = std::chrono::steady_clock::now();
      fill_legal_mask(next_opts, mask.data() + i * RL_MAX_ACTIONS);
      t1 = std::chrono::steady_clock::now();
      profile.mask_ns += elapsed_profile_ns(t0, t1);

      t0 = std::chrono::steady_clock::now();
      encode_action_features_from_options(
          games_[i], next_opts,
          action_features.data() + i * RL_MAX_ACTIONS * PPO_ACTION_FEAT_DIM);
      t1 = std::chrono::steady_clock::now();
      profile.action_features_ns += elapsed_profile_ns(t0, t1);

      t0 = std::chrono::steady_clock::now();
      rl_card_features(games_[i],
                       card_features.data() + i * PPO_CARD_SLOTS * PPO_CARD_FEAT_DIM);
      t1 = std::chrono::steady_clock::now();
      profile.card_features_ns += elapsed_profile_ns(t0, t1);
    }
  }
  profile.total_ns = elapsed_profile_ns(t_total, std::chrono::steady_clock::now());
  return profile;
}

// --- native random self-play actor ----------------------------------------
SelfplayResult rl_selfplay(const std::vector<int>& deck0, const std::vector<int>& deck1,
                           int n, uint64_t seed, int max_steps) {
  SelfplayResult res;
  uint64_t rng = seed ? seed : 0x9e3779b97f4a7c15ULL;
  auto t0 = std::chrono::steady_clock::now();
  for (int g = 0; g < n; ++g) {
    GameState st = new_game(deck0, deck1, rl_rand(rng) | 1);
    int steps = 0;
    while (st.result < 0 && steps < max_steps) {
      if (st.has_pending()) {
        resolve(st, random_subset(st.pending, rng));
      } else {
        auto opts = legal_main(st);
        int pick = static_cast<int>(rl_rand(rng) % opts.size());
        apply(st, descriptor_to_action(opts[pick]));
      }
      ++steps;
    }
    res.total_steps += steps;
    if (st.result == 0) ++res.p0_wins;
    else if (st.result == 1) ++res.p1_wins;
    else if (st.result == 2) ++res.draws;
    else ++res.unfinished;
  }
  res.seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
  return res;
}

}  // namespace ptcg
