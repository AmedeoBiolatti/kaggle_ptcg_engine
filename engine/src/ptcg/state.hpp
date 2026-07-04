#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "ptcg/small_vec.hpp"
#include "ptcg/types.hpp"

namespace ptcg {

// A Pokemon in play (active or bench).
struct InPlay {
  int id = 0;
  int serial = 0;  // bridge-only CABT card identity for option ordering
  int hp = 0;
  int maxHp = 0;
  bool appearThisTurn = false;
  bool abilityUsedThisTurn = false;  // its once-per-turn activated ability is spent
  bool reconstructedAbilityLocked = false;  // bridge-only legal-option suppression
  bool reconstructedAbilityPrevUsed = false;
  int reconstructedAbilityLockTurn = -1;
  bool movedToActiveThisTurn = false;
  bool healedThisTurn = false;
  int noEvolveTurn = -1;  // this Pokemon cannot evolve from hand this turn
  SmallVec<int, 16> energies;       // resolved EnergyType per energy unit
  SmallVec<int, 16> energyCardIds;  // attached energy CARD ids (discarded on KO)
  SmallVec<int, 16> energyCardOrders;  // bridge-only attach order for option parity
  SmallVec<int, 4> tools;          // attached tool card ids
  SmallVec<int, 4> toolOrders;     // bridge-only CABT serial/order for option parity
  SmallVec<int, 4> preEvo;         // pre-evolution card ids
  int lockId = 0;    // a self-locking attack id ("can't use next turn")
  int lockTurn = 0;  // the turn number on which lockId is unusable
  int activeLockId = 0;  // an attack unusable until this Pokemon leaves Active
  int reconstructedAttackLockTurn = -1;  // bridge-only legal-option suppression
  SmallVec<int, 4> reconstructedAttackLocks;
  int dmgReduce = 0;       // "-N damage from attacks during the opponent's next turn"
  int dmgReduceTurn = -1;  // the turn on which dmgReduce applies (set turn + 1)
  int noAttackTurn = -1;   // turn this Pokemon can't attack ("can't use attacks")
  int noRetreatTurn = -1;  // turn this Pokemon can't retreat
  int attackCostMod = 0;      // extra colorless attack cost for a turn
  int attackCostModTurn = -1; // the turn on which attackCostMod applies
  int retreatCostMod = 0;     // extra retreat cost for a turn
  int retreatCostModTurn = -1;// the turn on which retreatCostMod applies
  int delayedDamageTurn = -1; // end-turn delayed damage-counter effect
  int delayedDamageCounters = 0;
  int delayedKoTurn = -1;     // end-turn delayed KO effect
  bool delayedKoPromoteBeforePrize = false;
  int preventDmgTurn = -1; // turn all attack damage to this Pokemon is prevented
  int preventDmgCond = 0;  // DPC_* condition; 0 = unconditional
  int preventDmgValue = 0; // threshold/value for conditional prevention
  int preventEffectsTurn = -1; // turn attack effects to this Pokemon are prevented
  int preventEffectsCond = 0;
  int preventEffectsValue = 0;
  int attackFlipFailTurn = -1; // turn this Pokemon flips before attacking
  int noWeaknessTurn = -1;     // turn this Pokemon has no Weakness
  int takeMoreDamageTurn = -1; // turn this Pokemon takes extra attack damage
  int takeMoreDamage = 0;
  int nextAttackBonusId = 0;   // named attack buff/base override for a turn
  int nextAttackBonusTurn = -1;
  int nextAttackBonus = 0;
  int nextAttackSetBase = -1;
  int damagedByAttackCountersTurn = -1; // reactive counters on attacker
  int damagedByAttackCounters = 0;
  int damagedByAttackStatus = -1;       // reactive status on attacker
  int damagedByAttackEqualCountersTurn = -1; // counters equal to damage done
  int damagedByAttackTurn = -1;         // last turn this Pokemon took attack damage
  int damagedByAttackSide = -1;         // player whose attack damaged it
  int damagedByAttackAmount = 0;        // last attack damage amount actually done
  int damagedByAttackBeforeHp = -1;     // HP before that attack damage was applied
  int energyAttachCountersTurn = -1;    // counters when Energy is attached here
  int energyAttachCounters = 0;
  int energyAttachCountersFromHandOnly = 0;
  int attackDmgReduce = 0;      // this Pokemon's attacks do N less damage
  int attackDmgReduceTurn = -1; // the turn on which attackDmgReduce applies
  int attackBonus = 0;          // this Pokemon's attacks do N more damage
  int attackBonusTurn = -1;     // the turn on which attackBonus applies
};

struct Player {
  bool activePresent = false;  // a Pokemon occupies the Active Spot
  bool activeKnown = false;    // its identity is visible (not face-down)
  InPlay active;               // valid iff activeKnown
  SmallVec<InPlay, 5> bench;
  int benchMax = 5;
  int deckCount = 0;
  int handCount = 0;
  int prizeCount = 0;
  bool handKnown = false;  // hand identities visible (own player)
  bool deckKnown = false;  // deck identities are known; order may change on shuffle
  bool prizesKnown = false;  // remaining Prize identities are known
  // All card-id zone lists share one type so zone-pointer dispatch stays
  // uniform (deck is the sizing bound; the others are just as cheap inline).
  SmallVec<int, 64> hand;   // card ids (iff handKnown)
  SmallVec<int, 64> discard;
  SmallVec<int, 64> deck;   // ordered deck (free-running; empty in replay mode)
  SmallVec<int, 64> prizes;  // prize cards (free-running; counts-only in replay)
  SmallVec<bool, 64> deckKnownMask;    // per-slot known card identity in deck
  SmallVec<bool, 64> prizesKnownMask;  // per-slot known card identity in prizes
  SmallVec<int, 64> handKnownCards;   // unordered known hidden hand membership
  SmallVec<int, 64> deckKnownCards;   // unordered known deck membership
  SmallVec<int, 64> prizesKnownCards;  // unordered known Prize membership
  SmallVec<bool, 64> prizeFaceUp;  // true iff a Prize is revealed for rest of game
  bool poisoned = false, burned = false, asleep = false, paralyzed = false,
       confused = false;
  int poisonDamageCounters = 1;  // Checkup poison damage, normally 1 counter.
};

// A semantic, engine-agnostic descriptor of one legal option (matches the
// Python oracle's option_descriptor tuples). An Atom is either a string tag or
// an int field. The string is a static literal or interned constant — never
// owned by the Atom — so Atoms are trivially copyable and Descriptor copies
// reduce to one allocation plus memcpy. Compare tags by CONTENT (atom_sv /
// atom_is), never by raw `sym` pointer: literals and interned strings with
// equal text can live at different addresses.
struct Atom {
  bool is_str = false;
  bool is_none = false;  // a face-down / unknown ref (Python None)
  const char* sym = nullptr;  // static literal or intern_atom_string storage
  long long i = 0;
  static Atom S(const char* v) { Atom a; a.is_str = true; a.sym = v; return a; }
  static Atom S(const std::string& v);  // interns v (see below)
  static Atom I(long long v) { Atom a; a.i = v; return a; }
  static Atom N() { Atom a; a.is_none = true; return a; }
};
// Inline capacity 6 covers every descriptor shape the engine emits, so option
// generation is allocation-free in the common case.
using Descriptor = SmallVec<Atom, 6>;

// Stable-address pool for descriptor tags that arrive as std::string (e.g.
// descriptors rebuilt from Python); engine-generated descriptors use string
// literals and never touch the pool.
const char* intern_atom_string(const std::string& v);
inline Atom Atom::S(const std::string& v) { return S(intern_atom_string(v)); }

inline std::string_view atom_sv(const Atom& a) {
  return a.sym ? std::string_view(a.sym) : std::string_view();
}
inline bool atom_is(const Atom& a, std::string_view tag) {
  return a.is_str && atom_sv(a) == tag;
}

// --- decision-flow engine -------------------------------------------------
// A pending sub-decision the engine is waiting on. `context` is a cabt
// SelectContext value (-1 = none). `options` are descriptors in the same
// vocabulary as the Python oracle's option_descriptor, so they can be matched.
struct PendingDecision {
  int context = -1;
  int minCount = 1;
  int maxCount = 1;
  std::vector<Descriptor> options;
};

struct NativeLog {
  int type = 0;
  int playerIndex = -1;
  int cardId = 0;
  int serial = 0;
  int fromArea = -1;
  int toArea = -1;
  int cardIdActive = 0;
  int serialActive = 0;
  int cardIdBench = 0;
  int serialBench = 0;
  int cardIdTarget = 0;
  int serialTarget = 0;
  int attackId = 0;
  int value = 0;
  int result = -1;
  int reason = 0;
  bool hasBasicPokemon = false;
  bool putDamageCounter = false;
  bool isRecover = false;
  bool head = false;
};

// One in-progress (suspended) effect. The effect stack lets a single action
// fan out into a sequence of sub-decisions and resume after each.
struct EffectFrame {
  int effect = 0;  // EffectId
  int phase = 0;   // remaining sub-steps (e.g. prizes left to take)
  int a = 0;       // scratch (e.g. attacker / actor index)
  int program = 0;            // VM: program start offset (effect == FLOW_PROGRAM)
  int pc = 0;                 // VM: op index
  int attackId = 0;           // VM: the attack being resolved (attack frames)
  int attackCardId = 0;       // VM: original attacking Pokemon card id
  int sourceCardId = 0;       // VM: non-attack card whose program is resolving
  bool copiedAttack = false;  // nested copied attack; outer attack ends the turn
  int copiedAttackBaseDamage = -1;
  int loopRemain = -1;        // VM: OP_FOREACH_CHOSEN remaining iterations
  int selfBench = -1;         // VM: "this Pokemon" position (-1 = Active, else bench idx)
  int savedSrc = -2;          // VM: OP_SAVE_SRC stash (-2 unset; -1 Active / bench idx)
  int savedPhase = -1;        // VM: source zone for savedScratch
  int topDeckCount = 0;       // VM: current revealed top-deck window size
  int topDeckSelectedCount = 0;  // VM: selected cards from the current top window
  int topDeckOwner = -1;      // VM: player index owning the current top-deck window
  int topDeckCountedOut = 0;  // VM: revealed cards still stored in deck but out of deckCount
  int topDeckStart = 0;       // VM: deck index of local top-window option 0
  SmallVec<int, 8> scratch;   // VM: chosen refs
  SmallVec<int, 8> savedScratch;  // VM: saved chosen refs across a later CHOOSE
};

enum EffectId {
  EFF_SWITCH = 1,   // Switch item: swap Active with a chosen Benched Pokemon
  EFF_GUST = 2,     // Boss's Orders: pull a chosen opponent Benched to Active
  EFF_PRIZE = 3,    // take prize card(s) after a KO
  EFF_PROMOTE = 4,  // promote a new Active after a KO
  EFF_RETREAT = 5,  // discard energy to pay retreat cost, then switch
  EFF_LUNAR = 6,    // Lunatone Lunar Cycle: discard {F} Energy, draw 3
  EFF_HARIYAMA = 7, // Hariyama on-evolve: optional gust (YES/NO -> gust)
  EFF_AURAJAB = 8,  // Aura Jab: attach up to 3 {F} from discard to Bench
  EFF_SEARCH = 9,   // Dusk Ball / Fighting Gong / Poke Pad: deck -> hand
  FLOW_PROGRAM = 100,  // run the effect VM on `program`
  EFF_CHECKUP = 101,   // between-turns Checkup KO resolution (prizes/tie/promote)
  EFF_ABILITY_PROMOTE = 102,  // effect removed own Active; same player promotes
  EFF_POWERGLASS = 103,  // optional end-turn discard Basic Energy attach
  EFF_AMULET_HOPE = 104, // KO-trigger deck search, then resume KO flow
  EFF_HUNTAIL = 105,     // optional KO discard replacement for Basic Water Energy
  EFF_ABILITY_KO = 106,  // ability self-KO: opponent prizes, same turn resumes
  EFF_KO_RESUME = 107,   // generated KO hook finished; resume KO resolution
  EFF_HEAVY_BATON = 108, // optional KO-trigger Basic Energy move, then resume KO
  EFF_GRAND_TREE = 109,  // Grand Tree Stadium: deck-evolve Basic then optional Stage 2
  EFF_OGRES_MASK = 110,  // Ogre's Mask: swap discard Ogerpon ex with in-play Ogerpon ex
  EFF_PALAFIN_ZERO_TO_HERO = 111, // Palafin: active-to-bench deck identity swap
  EFF_SLOWKING_SEEK_INSPIRATION = 112, // discard top non-rule-box Pokemon, copy attack
  EFF_NINETALES_SHAPESHIFTER = 113, // discard top Supporter, use its effect as attack
  EFF_BOTHER_BOT = 114, // choose opp Prize, optionally swap with random opp hand
  EFF_HAND_TRIMMER = 115, // opponent then self choose hand cards to discard to 5
  EFF_DAMAGE_TRIGGER_ORDER = 116, // choose ordering for simultaneous on-damage hooks
  EFF_FROSLASS_CHECKUP = 117, // order Froslass Freezing Shroud Checkup triggers
  EFF_MAIN_ACTION_KO = 118, // main-action passive KO: prizes/promote, same turn resumes
  EFF_RISKY_RUINS_BENCH_ENTRY = 119, // deferred Risky Ruins bench damage
  EFF_ON_PLAY_BASIC_TRIGGER_ORDER = 120, // order Risky Ruins vs on-play Ability
  EFF_EVOLVE_TRIGGER_ORDER = 121, // order evolve-triggered skills
  EFF_DARKEST_IMPULSE = 122, // Team Rocket's Ampharos evolve damage
  EFF_KO_TRIGGER_ORDER = 123, // order simultaneous KO-triggered skills
  EFF_ON_ATTACH_TRIGGER_ORDER = 124, // order manual attach-triggered skills
};

struct GameState {
  int turn = 0;
  int turnActionCount = 0;
  int yourIndex = 0;
  int firstPlayer = -1;
  int result = -1;
  bool supporterPlayed = false, stadiumPlayed = false, energyAttached = false,
       retreated = false;
  bool teamRocketSupporterPlayed = false;
  bool ancientSupporterPlayed = false;
  bool canariPlayed = false;
  bool tarragonPlayed = false;
  SmallVec<int, 1> stadium;  // card ids (size 0 or 1)
  int stadiumOwner = -1;     // player who played the in-play Stadium (-1 = none)
  bool stadiumAbilityUsed = false;  // acting player used the Stadium's once/turn ability
  int noItemTurn[2] = {-1, -1};     // per player: a turn they can't play Item cards
  int noSupporterTurn[2] = {-1, -1}; // per player: a turn they can't play Supporters
  int noEvolveTurn[2] = {-1, -1};   // per player: a turn they can't evolve from hand
  int noStadiumTurn[2] = {-1, -1};  // per player: a turn they can't play Stadium cards
  int noAttackEnergyLeTurn[2] = {-1, -1};  // per player: low-energy Pokemon can't attack
  int noAttackEnergyLeThreshold[2] = {-1, -1};
  int discardHandEndTurn[2] = {-1, -1};  // per player: turn to check hand discard
  int discardHandEndThreshold[2] = {0, 0};
  int teamReduceTurn[2] = {-1, -1}; // per player: typed team damage reduction turn
  int teamReduceAmount[2] = {0, 0};
  int teamReduceType[2] = {-1, -1};
  int activeExDamageBuffTurn[2] = {-1, -1}; // per player: +damage to Active ex turn
  int activeExDamageBuffAmount[2] = {0, 0};
  int lastKoTurn[2] = {-1, -1};     // per player: last turn they had a Pokemon KO'd
  int lastAttackDamageKoTurn[2] = {-1, -1}; // last KO by opponent's attack damage
  int lastTeamRocketKoTurn[2] = {-1, -1};  // last turn their Team Rocket Pokemon was KO'd
  int lastEthanKoTurn[2] = {-1, -1};        // last turn their Ethan Pokemon was KO'd
  int lastHopAttackDamageKoTurn[2] = {-1, -1};  // Hop Pokemon KO'd by attack damage
  int lastAncientAttackTurn[2] = {-1, -1};  // last turn their Ancient Pokemon attacked
  int lastAncientAttackCard[2] = {0, 0};
  int lastAncientAttackSerial[2] = {0, 0};
  int lastAttackTurn[2] = {-1, -1};          // last completed attack by player
  int lastAttackId[2] = {0, 0};
  int prizeTakenTurn[2] = {-1, -1};          // most recent turn each player took Prizes
  int prizeTakenCount[2] = {0, 0};           // Prizes taken on prizeTakenTurn[player]
  int prizeBonusTurn[2] = {-1, -1};          // temporary "take N more Prize cards"
  int prizeBonusAmount[2] = {0, 0};
  int prizeBonusKind[2] = {0, 0};
  bool legacyEnergyPrizeUsed[2] = {false, false};
  bool pendingMeganiumAura[2] = {false, false};  // Wild Growth through Prize prompt
  int boomerangEnergyReturnCount[2] = {0, 0}; // card 9 returns after its user's attack
  int festivalLeadAttackTurn[2] = {-1, -1}; // first Festival Lead attack used this turn
  int festivalLeadResumeTurn[2] = {-1, -1}; // resume after KO promotion this turn
  Player players[2];
  uint64_t rng = 0;          // RNG state (free-running self-play)
  bool freeRunning = false;  // true if dealt by new_game (owns real deck/prizes)
  // Opt-in native-log emission. When false (default) apply()/resolve() skip
  // the before/after snapshot + delta diffing entirely and emit_log is a
  // no-op, so self-play/search states stay log-free and clone light. The cg
  // bridge entry points (cg_select_step, NativeCgBattle, load_state) enable
  // it on the states they drive; game logic never reads the logs.
  bool collectLogs = false;
  std::vector<int> replayTape; // draw/reveal ids and tagged replay events
  int replayTapePos = 0;
  std::vector<NativeLog> nativeLogs;
  int fightingBuff = 0;      // Premium Power Pro: +damage from {F} attacks this turn
  bool lunarUsedThisTurn = false;  // Lunatone Lunar Cycle once/turn
  int abilityGroupUsedTurn[2][16] = {};  // shared named Ability groups, by player
  int coinHeads = 0;         // heads count from the most recent OP_FLIP
  int coinFlips = 0;         // flips attempted by the most recent OP_FLIP
  int countReg = 0;          // result of the most recent OP_COUNT (count(domain,pred))
  int discardedCount = 0;    // items discarded by the most recent OP_DISCARD_CHOSEN
  int lastEffectCount = 0;   // count produced by the most recent countable VM op
  int lastAttackDamage = 0;  // damage just dealt to opponent Active by OP_DAMAGE
  bool endTurnAfterProgram = false;  // terminal VM op requested a turn handoff
  bool deferredPostAttack = false;   // attack hooks must finish before post_attack
  int deferredPostAttackPlayer = -1;
  int deferredPostAttackId = 0;
  int deferredPostAttackCard = 0;
  std::vector<int> replacementKoPairs;   // [takerSide, prizeValue, ...]
  std::vector<int> replacementKoGroups;  // grouping ids for replacement KO pairs
  int checkupNext = -1;      // >=0 while a between-turns Checkup KO is resolving:
                             // the player whose turn starts once it finishes
  int checkupKoFirst = -1;   // optional KO-owner order override for attack KOs
  bool checkupPromoteBeforePrize = false;  // end-turn delayed KO CABT ordering

  // decision-flow state
  PendingDecision pending;             // context < 0 when idle
  std::vector<EffectFrame> effectStack;
  std::vector<EffectFrame> afterProgramQueue;  // VM hooks deferred until current ends
  int pendingTrainerDiscard = -1;      // trainer card to discard when its effect ends
  int pendingTrainerOwner = -1;
  bool has_pending() const { return pending.context >= 0; }
};

SmallVec<int, 24> provided_energy_units(const InPlay& pk,
                                        const GameState* st = nullptr,
                                        int ownerSide = -1);

// Generate the structural (card-agnostic) MAIN-phase legal options for the
// acting player (st.yourIndex). Card-specific legality is layered on later.
std::vector<Descriptor> legal_main(const GameState& st);

// Deal a fresh free-running game from two 60-card decklists + a seed: shuffle,
// draw 7 (with mulligan), auto-place Active+Bench, set 6 prizes, flip for first
// player, and draw for turn 1. Returns a state at the first player's turn-1 MAIN.
// `collectLogs` opts the state into native-log emission from the deal onward.
GameState new_game(const std::vector<int>& deck0, const std::vector<int>& deck1,
                   uint64_t seed, bool collectLogs = false);

// --- transitions ----------------------------------------------------------

enum ActionKind {
  ACT_END,
  ACT_ATTACH,
  ACT_PLAY_BASIC,
  ACT_PLAY_TRAINER,
  ACT_EVOLVE,
  ACT_ATTACK,
  ACT_RETREAT,
  ACT_ABILITY,
  ACT_DISCARD_INPLAY,
  ACT_SETUP_ACTIVE,
};

struct Action {
  ActionKind kind = ACT_END;
  int cardId = 0;       // hand card (energy / tool / basic / trainer / evolution)
  int targetArea = 0;   // AREA_ACTIVE / AREA_BENCH (for ATTACH / EVOLVE)
  int targetIndex = 0;  // index within that area
  int attackId = 0;     // for ACT_ATTACK
};

// Apply an (already-legal) action, mutating the state. A trainer with a
// sub-decision leaves the engine with a pending decision (see resolve). `tape`
// supplies card identities the engine can't know on its own (drawn/revealed
// cards) plus tagged replay events; in free-running self-play these come from
// the deck/RNG.
void apply(GameState& st, const Action& a, const std::vector<int>& tape = {});

// Resolve the current pending sub-decision with chosen option indices
// (indices into st.pending.options), advancing the in-progress effect. `tape`
// supplies drawn card ids for sub-decisions that draw (e.g. Lunar Cycle).
void resolve(GameState& st, const std::vector<int>& selection,
             const std::vector<int>& tape = {});

}  // namespace ptcg
