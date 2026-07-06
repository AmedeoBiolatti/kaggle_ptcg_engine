#pragma once
#include <vector>

#include "ptcg/effects.gen.hpp"  // Op, effect_ops(), card_program(), attack_program()

namespace ptcg {

// Op kinds (must match the integer encoding in tools/gen_effects.py).
enum OpKind {
  OP_END = 0,
  OP_DAMAGE,
  OP_RECOIL,
  OP_DRAW,
  OP_DISCARD_HAND,
  OP_SHUFFLE_HAND_INTO_DECK,
  OP_SHUFFLE_DECK,
  OP_SET_BUFF,
  OP_SELF_LOCK,
  OP_SET_FLAG,
  OP_CHOOSE,
  OP_YESNO,
  OP_MOVE_CHOSEN,
  OP_SWITCH_ACTIVE,
  OP_ATTACH_CHOSEN,
  OP_FOREACH_CHOSEN,
  OP_IF_YES,
  OP_REQUIRE,
  OP_LOOP_BACK,
  OP_FLIP,          // flip coins -> st.coinHeads
  OP_APPLY_STATUS,  // inflict a special condition on a target's Active
  OP_IF_HEADS,      // skip p0 ops if the last flip got 0 heads
  OP_DISCARD_CHOSEN,  // discard the chosen items -> st.discardedCount
  OP_PLACE_DAMAGE,    // put damage counters (x10 HP) on a target
  OP_DISCARD_STADIUM, // discard the in-play Stadium (to its owner's discard)
  OP_MILL,            // discard the top N cards of a deck -> st.discardedCount
  OP_DISCARD_ALL_TYPE,// discard all energy of a type off the Active -> discardedCount
  OP_HEAL,            // heal damage (raise HP, capped at max) on own Pokemon
  OP_ATTACH_ENERGY,   // attach energy from a zone to "this Pokemon" (energy accel)
  OP_DISCARD_OPP_ENERGY,  // discard p0 energy from the opponent's Active
  OP_SELF_REDUCE,     // "this Pokemon takes -p0 damage during the opponent's next turn"
  OP_DRAW_UNTIL,      // draw until the hand holds p0 cards
  OP_COUNT,           // count(domain p0, filter p1, etype p2) -> st.countReg
  OP_TURN_EFFECT,     // set a turn-scoped flag (no-attack/no-retreat/prevent) on a target
  OP_OPP_NO_ITEMS,    // lock an opponent Trainer subtype during their next turn
  OP_MOVE_ENERGY,     // move p0 {p1}-energy: src p2 (0 self / 1 saved) -> dest p3 (0 chosen / 1 self)
  OP_RETURN_ENERGY,   // move p1 energy off "this Pokemon" to hand/deck (p0 = Dest)
  OP_SAVE_SRC,        // stash scratch[0] (a chosen in-play index) into fr.savedSrc
  OP_TEAM_REDUCE,     // all own Pokemon of type p1 take -p0 damage next opponent turn
  OP_CHOOSE_TOP_DECK, // choose from the top p0 cards of own deck
  OP_TOP_DECK_REST,   // handle unchosen top-deck cards after a top-deck choose
  OP_SAVE_CHOICE,     // stash the current scratch vector and source phase
  OP_MOVE_CHOSEN_ENERGY, // move previously chosen energy to a target
  OP_IF_EFFECT,       // skip p0 ops unless lastEffectCount >= p1
  OP_ATTACH_CHOSEN_ENERGY, // attach selected Energy cards to a target
  OP_REVEAL_HAND,     // reveal a player's hand for later VM choices
  OP_DISCARD_HAND_MATCHING, // discard every matching card from a hand
  OP_HAND_TO_BOTTOM_DRAW,   // put a hand on bottom of deck, then draw if any moved
  OP_APPEND_CHOICE,   // append scratch refs to savedScratch
  OP_MOVE_SAVED_CHOSEN, // move savedScratch refs
  OP_REORDER_TOP_DECK,  // reorder the current top-deck window by scratch order
  OP_CLEAR_STATUS,       // clear own Active's special conditions
  OP_RETURN_CHOSEN_ENERGY, // return chosen Energy from opponent/own in-play
  OP_DISCARD_SELF_ENERGY,  // discard p0 energy of type p1 from "this Pokemon"
  OP_TAKE_PRIZE,           // take p0 prize cards directly
  OP_DISCARD_OPP_TOOL,     // discard a Tool from the chosen opponent Pokemon
  OP_DISCARD_OPP_HAND,     // opponent discards p0 cards; p1 != 0 means random
  OP_SET_END_TURN_DISCARD_HAND, // at turn end discard own hand if handCount >= p0
  OP_BOTH_HAND_TO_BOTTOM_FLIP_DRAW, // Lucian: both hands bottom, flip/draw 6 or 3
  OP_SHUFFLE_SELF_INTO_DECK, // shuffle "this Pokemon" and attached cards into deck
  OP_MOVE_CHOSEN_TO_TOP, // remove chosen deck cards, shuffle, put them on top
  OP_DECK_EVOLVE_CHOSEN, // evolve self/saved target using the chosen deck card
  OP_END_TURN,           // finish this VM program and end the current turn
  OP_ATTACH_BASIC_ENERGY_EACH_TARGET, // attach one Basic Energy to each target
  OP_DISCARD_OPP_TOOLS_SPECIALS, // discard all opponent Tools/Special Energy
  OP_SHUFFLE_OPP_BENCH_EXCEPT_CHOSEN, // keep chosen opp Bench, shuffle the rest
  OP_KO_OPP_ACTIVE,      // set opponent Active HP to 0 without attack damage
  OP_JUMP,               // skip p0 following ops unconditionally
  OP_RETURN_SELF_TO_HAND, // put "this Pokemon" and attached cards into hand
  OP_RETURN_CHOSEN_TO_HAND, // put chosen own Pokemon and attached cards into hand
  OP_SHUFFLE_CHOSEN_INTO_DECK, // shuffle chosen in-play Pokemon into its owner's deck
  OP_OPP_SWITCH_OUT,     // opponent chooses a Benched Pokemon to promote Active
  OP_SET_ACTIVE_EX_DAMAGE_BUFF, // +p0 damage to opponent Active ex this turn
  OP_DECK_EVOLVE_SAVED,  // evolve saved own in-play targets from deck
  OP_DECK_EVOLVE_BENCH,  // evolve each own Benched Pokemon from deck
  OP_CHOOSE_OPP_ATTACHED_OR_STADIUM,
  OP_DISCARD_CHOSEN_OPP_ATTACHED_OR_STADIUM,
  OP_CRISPIN_CHOOSE_ENERGIES,
  OP_CRISPIN_TAKE_HAND_SAVE_ATTACH,
  OP_CRISPIN_ATTACH_SAVED,
  OP_CHOOSE_COUNT,
  OP_DISCARD_SELF_STACK,
  OP_MOVE_DAMAGE_COUNTER,
  OP_DELAYED_DAMAGE_OPP_ACTIVE,
  OP_DISCARD_HAND_TO_N,
  OP_KO_CHOSEN_OPP_INPLAY,
  OP_CHOOSE_COPIED_ATTACK,
  OP_RUN_COPIED_ATTACK,
  OP_OPP_DISCARD_TO_N,
  OP_DISTRIBUTE_DAMAGE,
  OP_APPLY_DISTRIBUTED_DAMAGE,
  OP_CHOOSE_ATTACH_TARGETS,
  OP_CHOOSE_OPP_ACTIVE_ATTACK,
  OP_DISABLE_CHOSEN_ATTACK,
  OP_ATTACH_SAVED_ENERGY_DISTRIBUTED,
  OP_CHOOSE_DISTINCT_BASIC_ENERGY,
  OP_MARK_SELF_ABILITY_USED,
  OP_SWITCH_SELF_BENCH_ACTIVE,
  OP_ATTACH_CHOSEN_TOOL,
  OP_RETURN_SELF_TO_HAND_DISCARD_ATTACHMENTS,
  OP_DISCARD_ENERGY_FROM_CHOSEN_INPLAY,
  OP_SWITCH_ACTIVE_SAVE,
  OP_MOVE_CHOSEN_TO_BENCH_SAVE,
  OP_OPP_NO_ATTACK_ENERGY_LE,
  OP_MOVE_CHOSEN_TO_DECK_BOTTOM_ORDERED,
  OP_DELAYED_KO_OPP_ACTIVE,
  OP_SELF_ATTACK_BONUS,
  OP_RANDOM_OPP_HAND_TO_DECK,
  OP_SHUFFLE_DISCARD_MATCHING,
  OP_SCALE_LAST_EFFECT,
  OP_ATTACK_DAMAGE_CHOSEN_OPP_BENCH_PER_TARGET_COUNTER,
  OP_KO_OPP_INPLAY_HP_LE,
  OP_RANDOM_DISCARD_HAND_TO_N,
  OP_RUN_CHOSEN_OPP_ACTIVE_ATTACK,
  OP_KO_SELF_ACTIVE,
  OP_ATTACK_DAMAGE_EACH_OPP_INPLAY_FILTERED,
  OP_DAMAGE_PER_TAILS_FROM_LAST_FLIP,
  OP_PLACE_OPP_BENCH_COUNTERS_TO_HP,
  OP_ATTACK_DAMAGE_EACH_OPP_INPLAY_COIN,
  OP_DISCARD_SELF_TOOL_ID,
  OP_SET_PRIZE_BONUS,
  OP_DEVOLVE_CHOSEN,
  OP_DEVOLVE_ALL,
  OP_CHOOSE_DEVOLVE_COUNT,
  OP_CHOOSE_DAMAGE_COUNTER_COUNT,
  OP_MOVE_DAMAGE_COUNTERS_DISTRIBUTED,
  OP_HAND_EVOLVE_CHOSEN,
  OP_CHOOSE_STATUS,
  OP_APPLY_CHOSEN_STATUS,
  OP_HEAL_SELF_LAST_DAMAGE,
  OP_MOVE_DAMAGE_COUNTERS_OWN_TO_OPP,
  OP_DISCARD_SELF_ENERGY_CARD,
  OP_SWAP_CHOSEN_HAND_WITH_TOP_DECK,
  OP_WIN_IF_OWN_PRIZES_EQ,
  OP_OPP_CHOOSE_HAND_TO_DECK,
  OP_HAND_TO_BOTTOM_DRAW_SAME,
  OP_PRIZE_BARGAIN,
  OP_DISCARD_BOTTOM_SELF_TO_TOP,
  OP_KO_LOWEST_HP_EXCEPT_SELF,
  OP_DRAW_UNTIL_COUNTREG,
  OP_CHOOSE_OPP_PRIZE,
  OP_REDEEMABLE_TICKET,
  OP_REPEAT_OPP_ACTIVE_ENERGY_DISCARD,
  OP_MARK_SELF_KO,
};

enum TurnTarget { TT_SELF = 0, TT_OPP_ACTIVE, TT_CHOSEN_OWN_INPLAY };
enum TurnKind {
  TK_NO_ATTACK = 0,
  TK_NO_RETREAT,
  TK_PREVENT_DMG,
  TK_ATTACK_DMG_REDUCE,
  TK_ATTACK_COST_MORE,
  TK_RETREAT_COST_MORE,
  TK_PREVENT_EFFECTS,
  TK_ATTACK_FLIP_FAIL,
  TK_TAKE_MORE_DAMAGE,
  TK_NO_WEAKNESS,
  TK_NAMED_ATTACK_BONUS,
  TK_IF_DAMAGED_COUNTERS,
  TK_IF_DAMAGED_STATUS,
  TK_ENERGY_ATTACH_COUNTERS,
  TK_IF_DAMAGED_DAMAGE_COUNTERS,
};
// scope -> when the effect is active: OPP_NEXT = the immediate next turn (turn+1),
// SELF_NEXT = the owner's next turn (turn+2).
enum TurnScope { TSC_OPP_NEXT = 0, TSC_SELF_NEXT };
enum OpponentTrainerLock {
  OTL_ITEMS = 0,
  OTL_SUPPORTERS,
  OTL_EVOLVE,
  OTL_STADIUM,
};
enum DmgPreventCond {
  DPC_ALL = 0,
  DPC_ATTACKER_BASIC,
  DPC_ATTACKER_BASIC_NON_COLORLESS,
  DPC_ATTACKER_EX,
  DPC_ATTACKER_HAS_ABILITY,
  DPC_DAMAGE_LE,
  DPC_DAMAGE_GE,
  DPC_ATTACKER_HAS_SPECIAL_ENERGY,
};

// Generalized count domains for OP_COUNT (count(zone, predicate)). p1 = filter
// (-1 = any, else a legacy F_* or a predicate id); p2 = energy-type (-1 = any).
enum CountDomain {
  CD_OWN_INPLAY = 0, CD_OPP_INPLAY,    // Pokemon in play matching the filter
  CD_OWN_DISCARD, CD_OWN_DECK,         // cards in a zone matching the filter
  CD_OWN_DAMAGED, CD_OPP_DAMAGED,      // in-play Pokemon with hp < maxHp (+ filter)
  CD_OWN_INPLAY_ENERGY,                // sum {p2}-energy over own Pokemon matching filter
  CD_BOTH_ACTIVE_ENERGY,               // {p2}-energy on both Active Pokemon
  CD_OPP_STATUS,                       // special conditions on the opponent's Active
  CD_OPP_HAND,                         // cards in opponent hand matching the filter
  CD_OPP_DISCARD,                      // cards in opponent discard matching filter
  CD_OWN_TOOLS,                        // Pokemon Tools attached to own Pokemon
  CD_BOTH_INPLAY,                      // both players' Pokemon in play
  CD_OWN_BENCH_DAMAGE,                 // damage counters on own Bench Pokemon
  CD_OWN_DISCARD_ANCIENT,              // Ancient cards in own discard
  CD_OWN_INPLAY_ANCIENT,               // Ancient Pokemon in own in-play area
  CD_OWN_DISCARD_BASIC_ENERGY_TYPED,   // Basic Energy cards of p2 type in own discard
  CD_BOTH_TOOLS,                       // Pokemon Tools attached to all Pokemon
  CD_OWN_BENCH,                        // own Benched Pokemon matching the filter
};

// OP_ATTACH_ENERGY source zone (p0). p1 = energy type (-1 = any basic), p2 = count,
// p3 = target.
enum AttachZone { AZ_HAND = 0, AZ_DISCARD, AZ_DECK };
enum AttachTarget {
  AT_SELF = 0,          // "this Pokemon" (the owner; EffectFrame.selfBench)
  AT_CHOSEN_BENCH,      // scratch[0] = own bench index
  AT_CHOSEN_INPLAY,     // scratch[0] = -1 (Active) or own bench index
  AT_EACH_CHOSEN_INPLAY,// distribute saved Energy cards across scratch targets
  AT_OWN_ACTIVE,         // own Active regardless of the ability user's position
  AT_OPP_CHOSEN_BENCH,   // scratch[0] = opponent bench index
  AT_OPP_CHOSEN_INPLAY,  // scratch[0] = -1 (Active) or opponent bench index
  AT_SAVED_INPLAY,       // savedSrc = -1 (Active) or own bench index
};
enum AttachEachTarget { AET_CHOSEN = 0, AET_OWN_INPLAY, AET_OWN_BENCH };
enum TopDeckRest { TR_SHUFFLE = 0, TR_DISCARD, TR_BOTTOM, TR_SHUFFLE_BOTTOM };

// Gate evaluated in legal_main before offering an activated ability.
enum AbilityGate {
  AG_NONE = 0, AG_SELF_ACTIVE, AG_SELF_HAS_ENERGY,
  AG_STADIUM, AG_PLAYED_SUPPORTER, AG_KO_LAST_TURN, AG_TERA,
  AG_SELF_BENCH, AG_SELF_BENCH_ACTIVE_LARRY, AG_HAS_FIRE_MEGA_EX,
  AG_PLAYED_TEAM_ROCKET_SUPPORTER, AG_ACTIVE_HAS_FESTIVAL_LEAD,
  AG_OWN_HAND_GE_1, AG_OWN_HAND_GE_2,
  AG_FIRST_TURN,
  AG_SELF_HAS_GRASS_ENERGY,
  AG_OWN_HAS_GRASS_MEGA_EX,
  AG_SELF_BENCH_OWN_HAS_MEGA_EX,
  AG_PLAYED_CANARI,
  AG_SELF_HAS_DARKNESS_ENERGY,
  AG_OWN_HAND_HAS_ENERGY,
  AG_SELF_HAS_BASIC_LIGHTNING_ENERGY,
};
enum AttackGate {
  ATG_NONE = 0,
  ATG_ALLOW_FIRST_PLAYER_TURN1,
  ATG_NOT_SECOND_PLAYER_TURN1,
  ATG_ONLY_SECOND_PLAYER_TURN1,
  ATG_TEAM_ROCKET_INPLAY_GE_4,
  ATG_OPP_HAS_EX_OR_V_INPLAY,
  ATG_NOT_USED_LAST_TURN,
};
enum CardGate {
  CG_NONE = 0,
  CG_KO_LAST_TURN,
  CG_TEAM_ROCKET_KO_LAST_TURN,
  CG_SECOND_PLAYER_TURN1,
  CG_LAST_CARD_IN_HAND,
  CG_MORE_PRIZES_THAN_OPP,
  CG_TERA_IN_PLAY,
  CG_OTHER_CARD_IN_HAND,
  CG_OWN_HAND_GE_3,
  CG_NOT_ACTOR_FIRST_TURN,
  CG_OPP_PRIZES_EQ_2,
  CG_ALL_N_TEAM,
  CG_OWN_DECK_GT_0,
  CG_OPP_PRIZES_LE_2,
  CG_RARE_CANDY,
  CG_OPP_HAND_GT_0,
};

enum PrizeBonusKind { PB_TERA_ATTACKER = 0, PB_N_ATTACKER };

// OP_HEAL target (p0). p1 = amount; p2 != 0 means heal ALL damage (to full).
enum HealTarget {
  HEAL_SELF = 0, HEAL_EACH_OWN, HEAL_EACH_OWN_BENCH, HEAL_CHOSEN_OWN_BENCH,
  HEAL_CHOSEN_OWN_INPLAY, HEAL_EACH_OWN_BASIC, HEAL_SELF_FILTERED,
};

enum MillSide { MILL_OWN = 0, MILL_OPP, MILL_EACH };
// How OP_MILL counts what it discarded into st.discardedCount (for scaling).
enum MillCount { CM_NONE = 0, CM_ENERGY, CM_BASIC_ENERGY_TYPED, CM_FILTER };

enum AmtKind { AMT_BASE = 0, AMT_CONST, AMT_COND_CONST, AMT_PER_HEADS,
               AMT_BASE_PLUS_HEADS, AMT_BASE_PLUS_PER, AMT_BASE_PLUS_IF,
               AMT_BASE_PLUS_COUNTREG,    // base + p1 * st.countReg
               AMT_BASE_PLUS_IF_SOURCE_GE,
               AMT_COND_BASE_OVERRIDE };
enum DmgFlag { DMG_IGNORE_WR = 1 };
constexpr int DMG_IGNORE_RESISTANCE = 2;
constexpr int DMG_IGNORE_WEAKNESS = 4;
constexpr int DMG_IGNORE_EFFECTS = 8;

// Scaling sources for AMT_BASE_PLUS_PER (damage = base + per * count(source)).
enum CountSource {
  CS_OWN_ENERGY = 0, CS_OPP_ENERGY, CS_OWN_BENCH, CS_OPP_BENCH,
  CS_OWN_DMG, CS_OPP_DMG, CS_OWN_PRIZES_TAKEN, CS_OPP_PRIZES_TAKEN,
  CS_OWN_DISCARD_ENERGY,  // energy cards in own discard (etype filter via p4)
  CS_OWN_INPLAY, CS_OPP_INPLAY,  // Pokemon in play (etype = type filter, -1 = all)
  CS_OWN_HAND,            // cards in own hand
  CS_DISCARDED,           // items discarded by the preceding OP_DISCARD_CHOSEN
  CS_OPP_HAND,            // cards in the opponent's hand
  CS_OWN_INPLAY_ENERGY,   // energy of type etype across all own in-play Pokemon
  CS_OPP_INPLAY_DMG,      // damage counters across all opponent Pokemon
  CS_OPP_INPLAY_ENERGY,   // energy of type etype across all opponent Pokemon
  CS_LAST_EFFECT,         // count produced by the previous countable VM op
  CS_BOTH_BENCH,          // both players' Benched Pokemon
  CS_OPP_PRIZES_TAKEN_LAST_TURN,
  CS_OPP_RETREAT_COST,
};

// OP_PLACE_DAMAGE targets.
enum PlaceTarget {
  PT_OPP_ACTIVE = 0, PT_SELF,
  PT_CHOSEN_OPP_BENCH,  // the opp Benched Pokemon chosen by a preceding CHOOSE
  PT_EACH_OPP_BENCH,    // every opp Benched Pokemon (spread)
  PT_CHOSEN_OPP_INPLAY, // a chosen opp Pokemon (Active idx -1 or Bench idx)
  PT_DISTRIBUTE_OPP_BENCH,  // spread p1 counters across opp Bench ("in any way")
  PT_CHOSEN_OWN_BENCH,
  PT_CHOSEN_OWN_INPLAY,
  PT_EACH_OPP_INPLAY,
  PT_EACH_OWN_BENCH,
  PT_EACH_OWN_INPLAY,
  PT_EACH_BENCH,
  PT_EACH_DAMAGED_BENCH,
  PT_EACH_OPP_INPLAY_DOUBLE_DMG,
  PT_EACH_OPP_ABILITY_INPLAY,
  PT_EACH_ABILITY_INPLAY,
  PT_EACH_ABILITY_INPLAY_EXCEPT_FROSLASS,
  PT_OPP_ACTIVE_TO_HP10,
  PT_CHOSEN_OWN_BENCH_TO_OPP_ACTIVE,
  PT_CHOSEN_OWN_BENCH_TO_CHOSEN_OPP_INPLAY,
  PT_SAVED_OPP_INPLAY,
};
enum PlaceFlag { PDMG_ATTACK = 1 };
constexpr int PDMG_IGNORE_EFFECTS = 2;

enum FlipMode {
  FLIP_FIXED = 0,
  FLIP_UNTIL_TAILS = 1,
  FLIP_COUNT_SOURCE = 2,
  FLIP_COUNTREG = 3,
};
enum Status { ST_POISON = 0, ST_BURN, ST_ASLEEP, ST_PARALYZED, ST_CONFUSED };

enum CountExpr {
  CNT_CONST = 0,
  CNT_PRIZE_6_8 = 1,
  CNT_SOURCE = 2,
  CNT_COUNTREG = 3,
};
enum Zone { Z_OWN_BENCH = 0, Z_OPP_BENCH, Z_HAND, Z_DISCARD, Z_DECK, Z_DECK_BOTTOM7,
            Z_OWN_ACTIVE_ENERGY,  // energy units on own Active (p1 = type+1, 0=any)
            Z_OWN_INPLAY,         // own Active (idx -1) + Bench (idx j)
            Z_OPP_INPLAY,         // opp Active (idx -1) + Bench (idx j)
            Z_OWN_INPLAY_ENERGY,  // energy units on own Active(-1) + Bench(j)
            Z_OPP_ACTIVE_ENERGY,  // energy units on opponent Active
            Z_OPP_INPLAY_ENERGY,  // energy units on opponent Active(-1) + Bench(j)
            Z_HAND_ENERGY,        // energy cards in own hand
            Z_DISCARD_ENERGY,     // energy cards in own discard
            Z_DECK_ENERGY,        // energy cards in own deck
            Z_OWN_BENCH_ENERGY,   // energy units on own Bench only
            Z_OPP_HAND,           // cards in opponent hand
            Z_OPP_HAND_BY_OPP,    // opponent chooses from their hand
            Z_INPLAY_TOOLS,       // Tool cards attached to either player's Pokemon
            Z_SAVED_OWN_INPLAY_ENERGY, // energy units on saved own Pokemon
            Z_SAVED_OPP_INPLAY_ENERGY, // energy units on saved opponent Pokemon
            Z_OPP_BENCH_ENERGY,   // energy units on opponent Bench only
            Z_SELF_ENERGY };      // energy units on "this Pokemon"
enum Filter {
  F_ANY = 0, F_FENERGY, F_POKEMON, F_FENERGY_OR_FBASIC, F_POKEMON_NORULEBOX,
  F_SUPPORTER, F_TRAINER, F_EVOLUTION, F_BASIC_ENERGY, F_BASIC_POKEMON,
  F_POKEMON_OR_BASIC_ENERGY, F_HAS_ENERGY,  // F_HAS_ENERGY: in-play Pokemon w/ energy
  F_FUTURE_POKEMON,
  F_DAMAGED,
  F_DAMAGE_COUNTERS_EQ_6,
  F_HAS_TOOL_OR_SPECIAL_ENERGY,
};
constexpr int EF_SPECIAL = -1;
constexpr int EF_SPECIAL_WITH_TOOL = -2;
constexpr int EF_BASIC = -3;
constexpr int EF_BASIC_TYPE_BASE = -100;
constexpr int F_EVOLVES_FROM_SELF = -20;
constexpr int F_EVOLVES_FROM_SAVED_OWN_INPLAY = -21;
constexpr int F_EVOLVES_FROM_SAVED_OWN_INPLAY_NO_ABILITY = -22;
constexpr int F_SAME_NAME_AS_OWN_INPLAY = -23;
constexpr int F_SAME_NAME_AS_OPP_INPLAY = -24;
constexpr int F_HAS_DECK_EVOLUTION = -25;
constexpr int F_HAS_DECK_EVOLUTION_NO_ABILITY = -26;
constexpr int F_DAMAGED_TEAM_ROCKET = -27;
constexpr int F_NOT_SAVED_SOURCE = -28;
constexpr int F_EVOLVED_INPLAY = -29;
constexpr int F_EVOLVED_PSYCHIC_INPLAY = -30;
constexpr int F_BASIC_WITH_STAGE2_IN_HAND = -31;
constexpr int F_STAGE2_EVOLVES_FROM_SAVED_BASIC = -32;
constexpr int F_REMAINING_HP_LE_30 = -33;
constexpr int F_DAMAGED_PSYCHIC_INPLAY = -34;
constexpr int F_DAMAGED_MEGA_EX_INPLAY = -35;
constexpr int F_NS_COPYABLE_ATTACK = -36;
constexpr int F_EVOLVES_FROM_OWN_INPLAY_NO_ABILITY = -37;
constexpr int F_EVOLVES_FROM_OWN_INPLAY = -38;
constexpr int F_EVOLVES_FROM_OWN_BENCH = -39;
enum DeckEvolveTarget {
  DET_SELF = 0,
  DET_SAVED_OWN_INPLAY,
  DET_AUTO_OWN_INPLAY_NO_ABILITY,
  DET_AUTO_OWN_INPLAY,
  DET_AUTO_OWN_BENCH,
};

// CHOOSE filters >= FILTER_PREDICATE_BASE are predicate ids (filter - base): a
// generated AND of clauses over CardInfo (see matches_predicate / pred_clauses).
constexpr int FILTER_PREDICATE_BASE = 100;
enum PredField {
  PF_CARDTYPE = 0, PF_ENERGYTYPE, PF_HP, PF_BASIC, PF_STAGE1, PF_STAGE2,
  PF_EX, PF_MEGAEX, PF_TERA, PF_ACESPEC, PF_RULEBOX, PF_NAME,
  PF_WEAKNESS, PF_RESISTANCE, PF_HAS_ABILITY,
};
enum PredCmp { PC_EQ = 0, PC_NE, PC_LE, PC_GE, PC_LT, PC_GT, PC_CONTAINS };
enum Side { S_OWN = 0, S_OPP };
enum Dest { D_HAND = 0, D_DISCARD, D_BENCH, D_DECK, D_DECK_BOTTOM };
enum Flag { FLAG_LUNAR_USED = 0 };
enum Cond {
  COND_OPP_HAS_BENCH = 0, COND_LUNATONE_ON_BENCH = 1, COND_OPP_IS_EX = 2,
  COND_STADIUM_IN_PLAY = 3, COND_OPP_IS_EVOLUTION = 4, COND_OPP_ACTIVE_DAMAGED = 5,
  COND_OPP_POISONED = 6, COND_OPP_BURNED = 7, COND_SELF_HAS_TOOL = 8,
  COND_SELF_HAS_SPECIAL_ENERGY = 9, COND_OPP_PRIZES_LE_3 = 10,
  COND_OWN_HAND_GE_10 = 11, COND_ALL_OWN_INPLAY_TEAM_ROCKET = 12,
  COND_SELF_POISONED = 13, COND_OPP_IS_STAGE2 = 14, COND_OWN_HAS_TERA = 15,
  COND_OWN_PRIZES_GT_OPP = 16,
  COND_ILLUMISE_ON_BENCH = 17,
  COND_SELF_EVOLVED_FROM_GIMMIGHOUL_THIS_TURN = 18,
  COND_SELF_IS_TEAM_ROCKET = 19,
  COND_LAST_EFFECT_POS = 20,
  COND_CHOSE_ACTIVE_AND_EFFECT = 21,
  COND_PLAYED_TEAM_ROCKET_SUPPORTER = 22,
  COND_OWN_KO_LAST_TURN = 23,
  COND_OWN_ETHAN_KO_LAST_TURN = 24,
  COND_OPP_ACTIVE_PSYCHIC = 25,
  COND_OPP_ACTIVE_DRAGON = 26,
  COND_OPP_ACTIVE_DARK = 27,
  COND_OPP_ACTIVE_TERA = 28,
  COND_SELF_HAS_FIGHTING_ENERGY = 29,
  COND_OPP_HAS_STATUS = 30,
  COND_SELF_DAMAGED = 31,
  COND_OWN_OPP_TYPE_OVERLAP = 32,
  COND_SELF_MOVED_FROM_BENCH_THIS_TURN = 33,
  COND_OTHER_ANCIENT_ATTACKED_LAST_TURN = 34,
  COND_BELDUM_METANG_ON_BENCH = 35,
  COND_OWN_BENCH_HAS_TERA = 36,
  COND_SELF_EVOLVED_FROM_MISTY_STARYU_THIS_TURN = 37,
  COND_OWN_BENCH_HAS_NIDOKING = 38,
  COND_SELF_HEALED_THIS_TURN = 39,
  COND_OWN_DECK_LE_3 = 40,
  COND_OWN_DISCARD_BASIC_FIGHTING_GE_10 = 41,
  COND_OWN_DISCARD_ROSA_ENCOURAGEMENT = 42,
  COND_USED_SPIKY_ROLLING_LAST_TURN = 43,
  COND_SELF_HAS_TEAM_ROCKET_ENERGY = 44,
  COND_OPP_STADIUM_IN_PLAY = 45,
  COND_PLAYED_ANCIENT_SUPPORTER = 46,
  COND_UXIE_AZELF_ON_BENCH = 47,
  COND_OPP_PRIZES_3_OR_4 = 48,
  COND_SELF_UNDAMAGED = 49,
  COND_ACTIVE_ENERGY_COUNTS_EQUAL = 50,
  COND_OWN_BENCH_GE_5 = 51,
  COND_OPP_HAND_LE_3 = 52,
  COND_HAND_COUNTS_EQUAL = 53,
  COND_OPP_ACTIVE_FIGHTING_RESISTANCE = 54,
  COND_SELF_BURNED_OR_POISONED = 55,
  COND_OWN_BENCH_DAMAGED = 56,
  COND_OWN_PRIZES_EQ_6 = 57,
  COND_SELF_HP_LE_50 = 58,
  COND_OWN_HAND_EQ_7 = 59,
  COND_DURANT_ON_BENCH = 60,
  COND_BENCHED_PANCHAM_DAMAGED = 61,
  COND_OPP_ACTIVE_BASIC = 62,
  COND_OPP_ACTIVE_DAMAGE_COUNTERS_EQ_6 = 63,
  COND_OWN_BENCH_STAGE2_DARK = 64,
  COND_SELF_EVOLVED_FROM_MAGNETON_THIS_TURN = 65,
  COND_SELF_HP_LE_0 = 66,
  COND_OPP_PRIZES_LE_2 = 67,
  COND_LAST_DITCH_UNUSED = 68,
  COND_PLAYED_TARRAGON = 69,
  COND_SELF_HAS_LIGHTNING_ENERGY = 70,
  COND_OWN_HOP_ATTACK_DAMAGE_KO_LAST_TURN = 71,
  COND_SELF_IS_PSYCHIC = 72,
  COND_CAN_DRAW_UNTIL_6 = 73,
  COND_OWN_HAS_BENCH = 74,
  COND_OPP_ACTIVE_HAS_ENERGY_AND_BENCH = 75,
  COND_OWN_DECK_NONEMPTY = 76,
  COND_OWN_ATTACK_DAMAGE_KO_LAST_TURN = 77,
  COND_LAST_ATTACK_DAMAGE_POS = 78,
};

struct GameState;

// Run the program in the top effect-stack frame from its `pc` until it either
// completes (OP_END) or suspends at a decision op (sets `pending` and returns).
void run_program(GameState& st, const std::vector<int>& tape);

}  // namespace ptcg
