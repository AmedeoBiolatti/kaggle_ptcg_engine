from dataclasses import asdict, dataclass, is_dataclass
from enum import IntEnum
import copy
import csv
import ctypes
import json
import os
import re
import sys

from .utils import to_dataclass, json_to_dataclass


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _use_native() -> bool:
    return os.environ.get("PTCG_BACKEND", os.environ.get("CG_BACKEND", "")).lower() == "native"


def _lib():
    from .sim import lib

    return lib


def _native_engine():
    root = _repo_root()
    for build_dir in (
        os.path.join(root, "engine", "build", "Release"),
        os.path.join(root, "engine", "build"),
    ):
        if os.path.isdir(build_dir) and build_dir not in sys.path:
            sys.path.insert(0, build_dir)
    import ptcg_engine as E

    return E

#region Enums

class AreaType(IntEnum):
    DECK = 1,
    HAND = 2,
    DISCARD = 3, # Discard Pile
    ACTIVE = 4, # Active Spot
    BENCH = 5,
    PRIZE = 6,
    STADIUM = 7,
    ENERGY = 8,
    TOOL = 9,
    PRE_EVOLUTION = 10, # The pre-evolved form of the Pokémon in play.
    PLAYER = 11,
    LOOKING = 12, # The card you are looking.

class EnergyType(IntEnum):
    COLORLESS = 0,
    GRASS = 1,
    FIRE = 2,
    WATER = 3,
    LIGHTNING = 4,
    PSYCHIC = 5,
    FIGHTING = 6,
    DARKNESS = 7,
    METAL = 8,
    DRAGON = 9,
    RAINBOW = 10, # Every Types
    TEAM_ROCKET = 11, # PSYCHIC and DARKNESS 

class CardType(IntEnum):
    POKEMON = 0,
    ITEM = 1,
    TOOL = 2, # Pokémon Tool
    SUPPORTER = 3,
    STADIUM = 4,
    BASIC_ENERGY = 5,
    SPECIAL_ENERGY = 6,

class SpecialConditionType(IntEnum):
    POISON = 0,
    BURN = 1,
    SLEEP = 2,
    PARALYZE = 3,
    CONFUSE = 4,

class SelectType(IntEnum):
    MAIN = 0, # OptionType: PLAY, ATTACH, EVOLVE, ABILITY, DISCARD, RETREAT, ATTACK, END
    CARD = 1, # OptionType: CARD
    ATTACHED_CARD = 2, # OptionType: TOOL_CARD, ENERGY_CARD
    CARD_OR_ATTACHED_CARD = 3, # OptionType: CARD, TOOL_CARD, ENERGY_CARD
    ENERGY = 4, # OptionType: ENERGY
    SKILL = 5, # OptionType: SKILL
    ATTACK = 6, # OptionType: ATTACK
    EVOLVE = 7, # OptionType: EVOLVE
    COUNT = 8, # OptionType: NUMBER
    YES_NO = 9, # OptionType: YES, NO
    SPECIAL_CONDITION = 10, # OptionType: SPECIAL_CONDITION
    
class SelectContext(IntEnum):
    MAIN = 0, # Main. Main selection.
    SETUP_ACTIVE_POKEMON = 1, # Card. Select the Pokémon to put into your Active Spot during Set Up.
    SETUP_BENCH_POKEMON = 2, # Card. Select the Pokémon to put onto your Bench during Set Up.
    SWITCH = 3, # Card. Select the Pokémon to swap with the one in your Active Spot.
    TO_ACTIVE = 4, # Card. Select the Pokémon to put into your Active Spot.
    TO_BENCH = 5, # Card. Select the Pokémon to put onto your Bench.
    TO_FIELD = 6, # Card. Select the Pokémon to put into play.
    TO_HAND = 7, # Card. Select the card to add to your hand.
    DISCARD = 8, # Card. Select the card to discard.
    TO_DECK = 9, # Card. Select the card to return to your deck.
    TO_DECK_BOTTOM = 10, # Card. Select the card to return to the bottom of your deck.
    TO_PRIZE = 11, # Card. Select the card to add to your prize.
    NOT_MOVE = 12, # Card. Select the card to remain where it is.
    DAMAGE_COUNTER = 13, # Card. Select the Pokémon to place damage counters on.
    DAMAGE_COUNTER_ANY = 14, # Card. Select the Pokémon to place damage counters on using the effect that lets you place them as you like.
    DAMAGE = 15, # Card. Select the Pokémon to deal damage.
    REMOVE_DAMAGE_COUNTER = 16, # Card. Select the Pokémon to remove damage counters from.
    HEAL = 17, # Card. Select the Pokémon to heal.
    EVOLVES_FROM = 18, # Card. Select the Pokémon to evolve from.
    EVOLVES_TO = 19, # Card. Select the Pokémon to evolve into.
    DEVOLVE = 20, # Card. Select the Pokémon to devolve.
    ATTACH_FROM = 21, # Card. Select the Pokémon to attach the card to.
    ATTACH_TO = 22, # Card. Select the card to attach to the Pokémon.
    DETACH_FROM = 23, # Card. Select the Pokémon to remove the card from.
    LOOK = 24, # Card. Select the card to look at.
    EFFECT_TARGET = 25, # Card. Select the card to apply the effect to.
    DISCARD_ENERGY_CARD = 26, # AttachedCard. Select the Energy card to discard.
    DISCARD_TOOL_CARD = 27, # AttachedCard. Select the Pokémon tool to trash.
    SWITCH_ENERGY_CARD = 28, # AttachedCard. Select the energy card to replace.
    DISCARD_CARD_OR_ATTACHED_CARD = 29, # CardOrAttachedCard. Select the card to discard.
    DISCARD_ENERGY = 30, # Energy. Select the energy to discard.
    TO_HAND_ENERGY = 31, # Energy. Select the energy to return to your hand.
    TO_DECK_ENERGY = 32, # Energy. Select the energy to return to the deck.
    SWITCH_ENERGY = 33, # Energy. Select the energy to switch.
    SKILL_ORDER = 34, # Skill. Select the order of effect activation.
    ATTACK = 35, # Attack. Select the Attack to use.
    DISABLE_ATTACK = 36, # Attack. Select the Attack to disable.
    EVOLVE = 37, # Evolve. Select the Pokémon that is the evolution source and the Pokémon that is the evolution target.
    DRAW_COUNT = 38, # Count. Select how many cards to draw.
    DAMAGE_COUNTER_COUNT = 39, # Count. Select how many damage counters to place.
    REMOVE_DAMAGE_COUNTER_COUNT = 40, # Count. Select how many damage counters to remove.
    IS_FIRST = 41, # YesNo. Would you like to go first?
    MULLIGAN = 42, # YesNo. Would you like to redraw the cards?
    ACTIVATE = 43, # YesNo. Would you like to activate the effect?
    FIRST_EFFECT = 44, # YesNo. Would you like to select the first effect?
    MORE_DEVOLVE = 45, # YesNo. Do you want to devolve it further?
    COIN_HEAD = 46, # YesNo. Do you want to choose heads?
    AFFECT_SPECIAL_CONDITION = 47, # SpecialCondition. Choose the special condition to affect.
    RECOVER_SPECIAL_CONDITION = 48, # SpecialCondition. Choose the special condition to recover.
    # Please note that new elements may be appended to the Enum during the competition.

class OptionType(IntEnum):
    # number (int):Count.
    NUMBER = 0, # Number to select.

    YES = 1, # Select Yes.

    NO = 2, # Select No.

    # area (AreaType):Area where the card is located.
    # index (int):Index within the area.
    # playerIndex (int):The owning player of the card.
    CARD = 3, # Card to select.

    # area (AreaType):Area of the attached Pokémon.
    # index (int):Index within the area of the attached Pokémon.
    # playerIndex (int):The owning player of the Pokémon.
    # toolIndex (int):Index within the tool.
    TOOL_CARD = 4, # Pokémon Tool Card to select.

    # area (AreaType):Area of the attached Pokémon.
    # index (int):Index within the area of the attached Pokémon.
    # playerIndex (int):The owning player of the Pokémon.
    # energyIndex (int):Index within the energy card.
    ENERGY_CARD = 5, # Energy Card to select.

    # area (AreaType):Area of the attached Pokémon.
    # index (int):Index within the area of the attached Pokémon.
    # playerIndex (int):The owning player of the Pokémon.
    # energyIndex (int):Index within the energy card.
    # count (int):How many energy units does it correspond to?
    ENERGY = 6, # Energy to select.

    # index (int):Index within the hand.
    PLAY = 7, # Play a card from your hand.

    # area (AreaType):Area of the card to attach.
    # index (int):Index within the area of the card to attach.
    # inPlayArea (AreaType):Area of the Pokémon on the field.
    # inPlayIndex (int):Index within the area of the Pokémon on the field.
    ATTACH = 8, # Attach a card to a Pokémon.

    # area (AreaType):Area of the evolved card.
    # index (int):Index within the area of the evolved card.
    # inPlayArea (AreaType):Area of the Pokémon on the field.
    # inPlayIndex (int):Index within the area of the Pokémon on the field.
    EVOLVE = 9, # Select an Evolution.

    # area (AreaType):Area where the card is located.
    # index (int):Index within the area.
    ABILITY = 10, # Use an Ability.

    # area (AreaType):Area where the card is located.
    # index (int):Index within the area.
    DISCARD = 11, # Discard a card in play.

    RETREAT = 12, # Retreat Active Pokémon.

    # attackId (int):Attack ID
    ATTACK = 13, # Select an Attack.

    END = 14, # Turn End.

    # cardId (int):Card ID. When the Card ID is 0, it means handling a Special Condition.
    # serial (int):Card serial
    SKILL = 15, # Select the order of card skills.

    # specialConditionType (SpecialConditionType):Special Condition Type
    SPECIAL_CONDITION = 16, # Select the Special Condition.

class LogType(IntEnum):
    # playerIndex (int)
    SHUFFLE = 0, # Shuffle deck.

    # playerIndex (int)
    # hasBasicPokemon (bool):If false, then no Basic Pokémon exist.
    HAS_BASIC_POKEMON = 1,

    # playerIndex (int)
    TURN_START = 2, # Start turn.

    # playerIndex (int)
    TURN_END = 3, # End turn.

    # playerIndex (int)
    # cardId (int):Drawn card ID
    # serial (int):Drawn card serial
    DRAW = 4, # Drew a card from deck.

    # playerIndex (int)
    DRAW_REVERSE = 5, # Your opponent drew a card from their deck.

    # playerIndex (int)
    # cardId (int):Moved card. ID
    # serial (int):Moved card. serial
    # fromArea (AreaType):Area before movement.
    # toArea (AreaType):Area after movement.
    MOVE_CARD = 6, # A card moved.

    # playerIndex (int)
    # fromArea (AreaType):Area before movement.
    # toArea (AreaType):Area after movement.
    MOVE_CARD_REVERSE = 7, # A card moved face-down.

    # playerIndex (int)
    # cardIdActive (int):Moving to the Bench Pokémon ID
    # serialActive (int):Moving to the Bench Pokémon serial
    # cardIdBench (int):Moving to the Active Pokémon ID
    # serialBench (int):Moving to the Active Pokémon serial
    SWITCH = 8, # Pokémon were switched.

    # playerIndex (int)
    # cardIdBefore (int):Pokémon before change. ID
    # serialBefore (int):Pokémon before change. serial
    # cardIdAfter (int):Pokémon after change. ID
    # serialAfter (int):Pokémon after change. serial
    CHANGE = 9, # Change the Pokémon.

    # playerIndex (int)
    # cardId (int):Played card ID
    # serial (int):Played card serial
    PLAY = 10, # Played a card from hand.

    # playerIndex (int)
    # cardId (int):Attached card ID
    # serial (int):Attached card serial
    # cardIdTarget (int):Pokémon card ID
    # serialTarget (int):Pokémon card serial
    ATTACH = 11, # Attached a card to a Pokémon.

    # playerIndex (int)
    # cardId (int):Evolved card ID
    # serial (int):Evolved card serial
    # cardIdTarget (int):Pokémon card ID
    # serialTarget (int):Pokémon card serial
    EVOLVE = 12, # Evolved a Pokémon.

    # playerIndex (int)
    # cardId (int):Devolved card ID
    # serial (int):Devolved card serial
    # cardIdTarget (int):Pokémon card ID
    # serialTarget (int):Pokémon card serial
    DEVOLVE = 13, # Devolved a Pokémon.

    # playerIndex (int)
    # cardId (int):Attached card ID
    # serial (int):Attached card serial
    # cardIdBefore (int):Pokémon that were attached with cards. ID
    # serialBefore (int):Pokémon that were attached with cards. serial
    # cardIdAfter (int):Pokémon that were newly attached with cards. ID
    # serialAfter (int):Pokémon that were newly attached with cards. serial
    MOVE_ATTACHED = 14, # Move the attached card.

    # playerIndex (int)
    # cardId (int):Pokémon that use attack. ID
    # serial (int):Pokémon that use attack. serial
    # attackId (int):Attack ID
    ATTACK = 15, # Pokémon Attack.

    # playerIndex (int)
    # cardId (int):HP changed card ID
    # serial (int):HP changed card serial
    # value (int):Amount of change.
    # putDamageCounter (bool):True if the HP change is due to the effect of placing a damage counter.
    HP_CHANGE = 16, # A Pokémon’s HP changed.

    # playerIndex (int)
    # isRecover (bool):If true, the special condition has been recovered.
    # cardId (int): ID
    # serial (int): serial
    POISONED = 17, # Poisoned.

    # playerIndex (int)
    # isRecover (bool):If true, the special condition has been recovered.
    # cardId (int): ID
    # serial (int): serial
    BURNED = 18, # Burned.

    # playerIndex (int)
    # isRecover (bool):If true, the special condition has been recovered.
    # cardId (int): ID
    # serial (int): serial
    ASLEEP = 19, # Fell asleep.

    # playerIndex (int)
    # isRecover (bool):If true, the special condition has been recovered.
    # cardId (int): ID
    # serial (int): serial
    PARALYZED = 20, # Paralyzed.

    # playerIndex (int)
    # isRecover (bool):If true, the special condition has been recovered.
    # cardId (int): ID
    # serial (int): serial
    CONFUSED = 21, # Confused.

    # playerIndex (int)
    # head (bool):True if coin is head.
    COIN = 22, # Result of the coin flip.

    # result (int):If 0, the player with player index 0 wins; if 1, the player with player index 1 wins; if 2, it's a draw.
    # reason (int):1: 0 Prize cards. 2: Start turn with 0 deck cards. 3: No Pokémon in Active Spot. 4: A card effect.
    RESULT = 23, # Result of the match.
    
    # Please note that new elements may be appended to the Enum during the competition.

#endregion Enums


# Please note that new attributes may be appended to each class during the competition.

#region Observation class

@dataclass
class Card:
    id: int  # CardData ID.
    serial: int  # Serial Number: A unique value assigned to each card in the match.
    playerIndex: int  # Represents which player's card.

@dataclass
class Pokemon:
    id: int  # CardData ID.
    serial: int  # Serial Number: A unique value assigned to each card in the match.
    hp: int  # Current HP.
    maxHp: int  # Current Max HP.
    appearThisTurn: bool  # True if played this turn.
    energies: list[EnergyType]  # Energies Array
    energyCards: list[Card]  # Attached Energy Card Array
    tools: list[Card]  # Attached Pokémon Tool Array
    preEvolution: list[Card]  # Pre-evolution Card Array
 
@dataclass
class PlayerState:
    active: list[Pokemon | None]  # Active Pokémon (None if the card is facedown). The array size is either 0 or 1.
    bench: list[Pokemon]  # Bench Pokémon.
    benchMax: int  # Maximum Bench Count.
    deckCount: int  # Remaining Cards in Deck.
    discard: list[Card]  # Discard pile Card Array.
    prize: list[Card | None]  # Prize cards (None if the card is facedown). The first element is the bottom of the prize, and the last element is the top.
    handCount: int  # Number of Cards in Hand.
    hand: list[Card] | None  # Hand Card Array. None for the opponent.
    poisoned: bool # Active Pokémon is Poisoned.
    burned: bool # Active Pokémon is Burned.
    asleep: bool # Active Pokémon is Asleep.
    paralyzed: bool # Active Pokémon is Paralyzed.
    confused: bool # Active Pokémon is Confused.

@dataclass
class State:
    turn: int  # Turn Count: 1 indicates the first turn for the starting player. 2 indicates the first turn for the second player. 3 indicates the second turn for the starting player. 0 denotes a time before the starting player's first turn.
    turnActionCount: int  # Number of Actions Taken This Turn.
    yourIndex: int  # Which player is making the selection? (Your Player Index.) 0 or 1.
    firstPlayer: int  # Starting Player Index. When the starting player has not been determined, the value is -1.
    supporterPlayed: bool  # True if a supporter has already been used this turn.
    stadiumPlayed: bool  # True if a stadium has already been used this turn.
    energyAttached: bool  # True if the manual Energy attachment for this turn has already been used.
    retreated: bool  # True if retreated this turn.
    result: int # Win player index. -1 if not battle finished.
    stadium: list[Card]  # Stadium Card. The array size is either 0 or 1.
    looking: list[Card | None] | None  # Looking cards (None if the card is facedown). None if not looking cards.
    players: list[PlayerState]  # An array of player states. The number of elements is 2.

@dataclass
class Option:
    type: OptionType  # Use this parameter to determine which option it is.
    number: int | None = None
    area: AreaType | None = None
    index: int | None = None
    playerIndex: int | None = None
    toolIndex: int | None = None
    energyIndex: int | None = None
    count: int | None = None
    inPlayArea: AreaType | None = None
    inPlayIndex: int | None = None
    attackId: int | None = None
    cardId: int | None = None
    serial: int | None = None
    specialConditionType: SpecialConditionType | None = None

@dataclass
class SelectData:
    type: SelectType  # Selection type.
    context: SelectContext  # What is being selected?
    minCount: int  # Minimum number of selections. It can also be 0.
    maxCount: int  # Maximum number of selections. Never exceeds len(option).
    remainDamageCounter: int  # Remaining number of damage counters that can be placed.
    remainEnergyCost: int  # Used when the type is Energy. The remaining required energy count.
    option: list[Option]  # Array of options.
    deck: list[Card] | None  # An array of cards; None unless selecting cards from the deck.
    contextCard: Card | None  # Which card is the selection concerning? This is sent when the context is "Activate"; otherwise, it is null.
    effect: Card | None  # The card that is activating the effect currently being processed.
    
@dataclass
class Log:
    type: LogType  # Use this parameter to determine which log it is.
    playerIndex: int | None = None
    hasBasicPokemon: bool | None = None
    cardId: int | None = None
    serial: int | None = None
    fromArea: AreaType | None = None
    toArea: AreaType | None = None
    cardIdActive: int | None = None
    serialActive: int | None = None
    cardIdBench: int | None = None
    serialBench: int | None = None
    cardIdBefore: int | None = None
    serialBefore: int | None = None
    cardIdAfter: int | None = None
    serialAfter: int | None = None
    cardIdTarget: int | None = None
    serialTarget: int | None = None
    attackId: int | None = None
    value: int | None = None
    putDamageCounter: bool | None = None
    isRecover: bool | None = None
    head: bool | None = None
    result: int | None = None
    reason: int | None = None
    
@dataclass
class Observation:
    select: SelectData | None  # Selection information. At the time of the initial deck selection, it will be None.
    logs: list[Log]  # Events that have occurred since the last selection.
    current: State | None  # Current state. At the time of the initial deck selection, it will be None.
    search_begin_input: str | None = None # Input to the search_begin function.

#endregion Observation class

@dataclass
class SearchState:
    observation: Observation  # New observation. search_begin_input is None.
    searchId: int  #  Search state ID.
    
@dataclass
class ApiResult:
    state: SearchState | None # Search state.
    error: int # Error if not 0.


@dataclass
class _NativeSearchRecord:
    state: object
    observation: Observation
    manual_coin: bool = False
    released: bool = False

# Abilities and effects at the time of card play.
@dataclass
class Skill:
    name: str  # Skill name.
    text: str  # Explanation.

@dataclass
class CardData:
    cardId: int  # Card ID.
    name: str  # Card name.
    cardType: CardType  # Card type
    retreatCost: int  # Energy cost required to retreat.
    hp: int  # Pokémon HP.
    weakness: EnergyType | None  # Pokémon weakness.
    resistance: EnergyType | None  # Pokémon resistance.
    energyType: EnergyType  # Pokémon or Basic Energy type.
    basic: bool # True if Basic Pokémon.
    stage1: bool # True if Stage1 Pokémon.
    stage2: bool # True if Stage2 Pokémon.
    ex: bool # True if Pokémon ex(include Mega Evolution Pokémon ex). When your Pokémon ex is Knocked Out, your opponent takes 2 prize cards(exclude Mega Evolution Pokémon ex).
    megaEx: bool # True if Mega Evolution Pokémon ex. When your Mega Evolution Pokémon ex is Knocked Out, your opponent takes 3 prize cards.
    tera: bool  # True if Tera Pokémon. Tera Pokémon take no damage from attacks as long as they are on the Bench.
    aceSpec: bool  # True if ACE SPEC. You can't have more than 1 ACE SPEC card in your deck.
    evolvesFrom: str | None  # If the Pokémon has evolved, then the name of its pre-evolution. Otherwise, None.
    skills: list[Skill]  # The skills that the card has.
    attacks: list[int]  # IDs of usable attacks.

@dataclass
class Attack:
    attackId: int  # Attack ID.
    name: str  # Attack name.
    text: str  # Explanation.
    damage: int  # Attack damage
    energies: list[EnergyType]  # Energy required to use.
    

#region functions

def all_card_data() -> list[CardData]:
    """Return all cards."""
    if _use_native():
        return _native_all_card_data()
    bs = _lib().AllCard()
    js = bs.decode()
    cards = json.loads(js)
    return [to_dataclass(v, CardData) for v in cards]

def all_attack() -> list[Attack]:
    """Return all attacks."""
    if _use_native():
        return _native_all_attack()
    bs = _lib().AllAttack()
    js = bs.decode()
    cards = json.loads(js)
    return [to_dataclass(v, Attack) for v in cards]

def to_observation_class(obs: dict) -> Observation:
    """dict to Observation class.

    Returns:
        Observation: Observation dataclass instance.
    """
    return to_dataclass(obs, Observation)

def search_begin(agent_observation: Observation,
                 your_deck: list[int],
                 your_prize: list[int],
                 opponent_deck: list[int],
                 opponent_prize: list[int],
                 opponent_hand: list[int],
                 opponent_active: list[int],
                 manual_coin: bool = False
    ) -> SearchState:
    """Begin search.

    Args:
        agent_observation: You must input the observation argument passed to your agent function exactly as is.
        your_deck: Predicted Card ID your Deck. It must have the same number of cards as your deck. If Observation.select.deck != None, ignored this.
        your_prize: Predicted Card ID your Prize cards. It must have the same number of cards as your prize.
        opponent_deck: Predicted Card ID opponent's deck. It must have the same number of cards as opponent's deck. At setup, at least one Basic Pokémon card is required.
        opponent_prize: Predicted Card ID opponent's prize cards. It must have the same number of cards as opponent's prize.
        opponent_hand: Predicted Card ID opponent's hand. It must have the same number of cards as opponent's hand.
        opponent_active: Predicted Card ID opponent's Active Pokémon. Only if there is a face-down Pokémon in your opponent’s Active Spot. This ID must be a Pokémon card ID.
        manual_coin: If True, the coin's heads or tails can be chosen.

    Returns:
        SearchState: Root search state.
    """
    if _use_native():
        return _native_search_begin(
            agent_observation,
            your_deck,
            your_prize,
            opponent_deck,
            opponent_prize,
            opponent_hand,
            opponent_active,
            manual_coin,
        )
    global agent_ptr
    
    if "agent_ptr" not in globals():
        agent_ptr = _lib().AgentStart()
    
    sbi = agent_observation.search_begin_input
    if sbi == None:
        raise ValueError("Not agent observation.")

    state = agent_observation.current
    your_index = state.yourIndex

    if agent_observation.select.deck != None:
        your_deck = []
    elif len(your_deck) < state.players[your_index].deckCount:
        raise ValueError("your_deck does not match the number of cards in your deck.")
    
    if len(your_prize) < len(state.players[your_index].prize):
        raise ValueError("your_prize does not match the number of cards in your prize.")
    elif len(opponent_deck) < state.players[1 - your_index].deckCount:
        raise ValueError("opponent_deck does not match the number of cards in opponent's deck.")
    elif len(opponent_prize) < len(state.players[1 - your_index].prize):
        raise ValueError("opponent_prize does not match the number of cards in opponent's prize.")
    elif len(opponent_hand) < state.players[1 - your_index].handCount:
        raise ValueError("opponent_hand does not match the number of cards in opponent's hand.")
    
    active = state.players[1 - your_index].active
    if len(active) > 0 and active[0] == None:
        if len(opponent_active) == 0:
            raise ValueError("You need to predict the opponent's Active Pokémon.")
    else:
        opponent_active = []
    
    bs = _lib().SearchBegin(agent_ptr,
                         sbi.encode("ascii"),
                         len(sbi),
                         (ctypes.c_int*len(your_deck))(*your_deck),
                         (ctypes.c_int*len(your_prize))(*your_prize),
                         (ctypes.c_int*len(opponent_deck))(*opponent_deck),
                         (ctypes.c_int*len(opponent_prize))(*opponent_prize),
                         (ctypes.c_int*len(opponent_hand))(*opponent_hand),
                         (ctypes.c_int*len(opponent_active))(*opponent_active),
                         int(manual_coin))
    result = json_to_dataclass(bs, ApiResult)
    if result.error != 0:
        if result.error == 1:
            raise ValueError("Invalid Card ID.")
        elif result.error == 2:
            raise ValueError("Active card must be the ID of a Pokémon card.")
        elif result.error == 30:
            raise ValueError("agent_ptr broken.")
        else:
            raise RuntimeError()

    return result.state

def search_step(search_id: int, select: list[int]) -> SearchState:
    """Proceed to the next selection.
    
    Args:
        search_id: Search ID.
        select: Chosen option index.

    Returns:
        SearchSate: State for the next selection.
    """
    if _use_native():
        return _native_search_step(search_id, select)
    bs = _lib().SearchStep(agent_ptr, search_id, (ctypes.c_int*len(select))(*select), len(select))
    result = json_to_dataclass(bs, ApiResult)
    if result.error != 0:
        if result.error == 1:
            raise ValueError("There is no element with the specified search_id.")
        elif result.error == 2:
            raise ValueError("Released item.")
        elif result.error == 3:
            raise ValueError("Cannot be selected because the battle has ended.")
        elif result.error == 4:
            raise ValueError("Must be Observation.select.minCount <= len(select) <= Observation.select.maxCount.")
        elif result.error == 5:
            raise ValueError("Must be 0 <= select elements < len(Observation.select.option).")
        elif result.error == 6:
            raise ValueError("Duplicate select elements.")
        elif result.error == 30:
            raise ValueError("agent_ptr broken.")
        else:
            raise RuntimeError()
    
    return result.state

def search_end() -> None:
    """Terminate the search. Memory used during the search will be reused in the next search."""
    if _use_native():
        global _NATIVE_SEARCH_NEXT_ID
        _NATIVE_SEARCH_STATES.clear()
        _NATIVE_SEARCH_RELEASED.clear()
        _NATIVE_SEARCH_NEXT_ID = 1
        return
    _lib().SearchEnd(agent_ptr)

def search_release(search_id: int) -> None:
    """Delete the state with the specified ID and make the memory available for reuse.
    
    Args:
        search_id: Search ID.
    """
    if _use_native():
        sid = int(search_id)
        rec = _NATIVE_SEARCH_STATES.get(sid)
        if rec is not None:
            rec.released = True
            _NATIVE_SEARCH_RELEASED.add(sid)
        return
    _lib().SearchRelease(agent_ptr, search_id)

#endregion functions


_CARD_DB_CACHE: tuple[list[CardData], list[Attack]] | None = None
_CARD_TEXT_CACHE: dict[int, dict] | None = None
_OFFICIAL_CARD_META_CACHE: tuple[dict[int, CardData], dict[int, Attack]] | None | bool = None
_NATIVE_SEARCH_STATES: dict[int, _NativeSearchRecord] = {}
_NATIVE_SEARCH_RELEASED: set[int] = set()
_NATIVE_SEARCH_NEXT_ID = 1
_NATIVE_SEARCH_SEED = 1


def _card_db_path() -> str:
    root = _repo_root()
    return os.path.join(root, "engine", "src", "ptcg", "card_db.gen.hpp")


def _card_text_path() -> str:
    root = _repo_root()
    return os.path.join(root, "data", "EN_Card_Data.csv")


def _native_metadata_path() -> str:
    root = _repo_root()
    return os.path.join(root, "data", "native_cg_metadata.json")


def _decode_cpp_string(raw: str) -> str:
    return bytes(raw, "utf-8").decode("unicode_escape")


def _native_bool(raw: str) -> bool:
    return raw == "true"


def _optional_energy(value: int) -> EnergyType | None:
    return None if value < 0 else EnergyType(value)


def _clean_card_text(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    return "" if value.lower() == "n/a" else value


def _native_card_text_metadata() -> dict[int, dict]:
    global _CARD_TEXT_CACHE
    if _CARD_TEXT_CACHE is not None:
        return _CARD_TEXT_CACHE

    by_card: dict[int, dict] = {}
    path = _card_text_path()
    if not os.path.exists(path):
        _CARD_TEXT_CACHE = by_card
        return by_card

    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            raw_id = row.get("Card ID", "")
            try:
                card_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            entry = by_card.setdefault(
                card_id,
                {
                    "name": _clean_card_text(row.get("Card Name")),
                    "skills": [],
                    "attack_rows": [],
                },
            )
            if not entry["name"]:
                entry["name"] = _clean_card_text(row.get("Card Name"))

            move_name = _clean_card_text(row.get("Move Name"))
            effect_text = _clean_card_text(row.get("Effect Explanation"))
            cost = _clean_card_text(row.get("Cost"))
            damage = _clean_card_text(row.get("Damage"))
            if not move_name and not effect_text:
                continue
            is_attack = bool(move_name) and (
                bool(cost) or bool(damage) or move_name == "Tera"
            )
            if is_attack:
                entry["attack_rows"].append({"name": move_name, "text": effect_text})
            elif move_name.startswith("[Ability]"):
                entry["skills"].append(
                    Skill(move_name.replace("[Ability] ", ""), effect_text)
                )
            elif move_name:
                entry["skills"].append(Skill(move_name.strip("[]"), effect_text))
            elif effect_text:
                entry["skills"].append(Skill(entry["name"], effect_text))

    _CARD_TEXT_CACHE = by_card
    return by_card


def _official_card_metadata() -> tuple[dict[int, CardData], dict[int, Attack]] | None:
    global _OFFICIAL_CARD_META_CACHE
    if _OFFICIAL_CARD_META_CACHE is False:
        return None
    if _OFFICIAL_CARD_META_CACHE is not None:
        return _OFFICIAL_CARD_META_CACHE
    path = _native_metadata_path()
    try:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            card_rows = raw["cards"]
            attack_rows = raw["attacks"]
        else:
            card_rows = json.loads(_lib().AllCard().decode())
            attack_rows = json.loads(_lib().AllAttack().decode())
        cards = {
            card.cardId: card
            for card in (to_dataclass(v, CardData) for v in card_rows)
        }
        attacks = {
            attack.attackId: attack
            for attack in (to_dataclass(v, Attack) for v in attack_rows)
        }
    except Exception:
        _OFFICIAL_CARD_META_CACHE = False
        return None
    _OFFICIAL_CARD_META_CACHE = (cards, attacks)
    return _OFFICIAL_CARD_META_CACHE


def _parse_native_card_db() -> tuple[list[CardData], list[Attack]]:
    global _CARD_DB_CACHE
    if _CARD_DB_CACHE is not None:
        return _CARD_DB_CACHE

    cards: list[CardData] = []
    attacks_by_id: dict[int, Attack] = {}
    text_meta = _native_card_text_metadata()
    official_meta = _official_card_metadata()
    official_cards = official_meta[0] if official_meta else {}
    official_attacks = official_meta[1] if official_meta else {}
    card_pattern = re.compile(
        r'^\s*\{(\d+),\s*"((?:[^"\\]|\\.)*)",\s*(-?\d+),\s*(-?\d+),'
        r'\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),'
        r'\s*(true|false),\s*(true|false),\s*(true|false),'
        r'\s*(true|false),\s*(true|false),\s*(true|false),\s*(true|false),'
        r'\s*(nullptr|"((?:[^"\\]|\\.)*)"),\s*(true|false),\s*(true|false),\s*(\d+),'
    )
    attack_pattern = re.compile(r'\{(\d+),\s*(-?\d+),\s*(\d+),\s*\{([^}]*)\}\}')
    with open(_card_db_path(), encoding="utf-8") as f:
        for line in f:
            match = card_pattern.match(line)
            if not match:
                continue
            attack_ids: list[int] = []
            card_id = int(match.group(1))
            card_meta = text_meta.get(card_id, {})
            attack_rows = card_meta.get("attack_rows", [])
            for attack_index, attack_match in enumerate(attack_pattern.finditer(line)):
                attack_id = int(attack_match.group(1))
                if attack_id <= 0:
                    continue
                costs = [
                    EnergyType(int(raw.strip()))
                    for raw in attack_match.group(4).split(",")
                    if raw.strip() and int(raw.strip()) >= 0
                ][: int(attack_match.group(3))]
                official_attack = official_attacks.get(attack_id)
                if official_attack is not None:
                    attacks_by_id.setdefault(
                        attack_id,
                        Attack(
                            attack_id,
                            official_attack.name,
                            official_attack.text,
                            int(attack_match.group(2)),
                            costs,
                        ),
                    )
                else:
                    attack_meta = (
                        attack_rows[attack_index]
                        if attack_index < len(attack_rows)
                        else {}
                    )
                    attacks_by_id.setdefault(
                        attack_id,
                        Attack(
                            attack_id,
                            str(attack_meta.get("name", "")),
                            str(attack_meta.get("text", "")),
                            int(attack_match.group(2)),
                            costs,
                        ),
                    )
                attack_ids.append(attack_id)

            evolves_from = (
                None
                if match.group(16) == "nullptr"
                else _decode_cpp_string(match.group(17))
            )
            official_card = official_cards.get(card_id)
            name = (
                official_card.name
                if official_card
                else str(card_meta.get("name") or _decode_cpp_string(match.group(2)))
            )
            skills = (
                list(official_card.skills)
                if official_card
                else list(card_meta.get("skills", []))
            )
            cards.append(CardData(
                cardId=card_id,
                name=name,
                cardType=CardType(int(match.group(3))),
                retreatCost=int(match.group(8)),
                hp=int(match.group(4)),
                weakness=_optional_energy(int(match.group(5))),
                resistance=_optional_energy(int(match.group(6))),
                energyType=EnergyType(int(match.group(7))),
                basic=_native_bool(match.group(9)),
                stage1=_native_bool(match.group(10)),
                stage2=_native_bool(match.group(11)),
                ex=_native_bool(match.group(12)),
                megaEx=_native_bool(match.group(13)),
                tera=_native_bool(match.group(14)),
                aceSpec=_native_bool(match.group(15)),
                evolvesFrom=evolves_from,
                skills=skills,
                attacks=attack_ids,
            ))

    _CARD_DB_CACHE = (cards, [attacks_by_id[k] for k in sorted(attacks_by_id)])
    return _CARD_DB_CACHE


def _native_all_card_data() -> list[CardData]:
    return list(_parse_native_card_db()[0])


def _native_all_attack() -> list[Attack]:
    return list(_parse_native_card_db()[1])


def _plain(value):
    if is_dataclass(value):
        return {key: _plain(val) for key, val in asdict(value).items()}
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, list):
        return [_plain(val) for val in value]
    if isinstance(value, dict):
        return {key: _plain(val) for key, val in value.items()}
    return value


def _native_card_lookup(card_id: int) -> CardData | None:
    for card in _native_all_card_data():
        if card.cardId == int(card_id):
            return card
    return None


def _validate_card_ids(*lists: list[int]) -> None:
    known = {card.cardId for card in _native_all_card_data()}
    for values in lists:
        for card_id in values:
            if int(card_id) not in known:
                raise ValueError("Invalid Card ID.")


def _validate_search_begin_inputs(agent_observation: Observation,
                                  your_deck: list[int],
                                  your_prize: list[int],
                                  opponent_deck: list[int],
                                  opponent_prize: list[int],
                                  opponent_hand: list[int],
                                  opponent_active: list[int]) -> tuple[
                                      list[int],
                                      list[int],
                                      list[int],
                                      list[int],
                                      list[int],
                                      list[int],
                                  ]:
    if agent_observation.search_begin_input is None or agent_observation.current is None:
        raise ValueError("Not agent observation.")

    state = agent_observation.current
    your_index = state.yourIndex
    if (
        agent_observation.select is not None
        and agent_observation.select.deck is not None
        and len(your_deck) < state.players[your_index].deckCount
    ):
        your_deck = []
    elif len(your_deck) < state.players[your_index].deckCount:
        raise ValueError("your_deck does not match the number of cards in your deck.")

    if len(your_prize) < len(state.players[your_index].prize):
        raise ValueError("your_prize does not match the number of cards in your prize.")
    if len(opponent_deck) < state.players[1 - your_index].deckCount:
        raise ValueError("opponent_deck does not match the number of cards in opponent's deck.")
    if len(opponent_prize) < len(state.players[1 - your_index].prize):
        raise ValueError("opponent_prize does not match the number of cards in opponent's prize.")
    if len(opponent_hand) < state.players[1 - your_index].handCount:
        raise ValueError("opponent_hand does not match the number of cards in opponent's hand.")

    active = state.players[1 - your_index].active
    if len(active) > 0 and active[0] is None:
        if len(opponent_active) == 0:
            raise ValueError("You need to predict the opponent's Active Pokémon.")
        card = _native_card_lookup(opponent_active[0])
        if card is None:
            raise ValueError("Invalid Card ID.")
        if int(card.cardType) != int(CardType.POKEMON):
            raise ValueError("Active card must be the ID of a Pokémon card.")
    else:
        opponent_active = []

    _validate_card_ids(
        your_deck,
        your_prize,
        opponent_deck,
        opponent_prize,
        opponent_hand,
        opponent_active,
    )
    return your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active


def _validate_native_select(observation: Observation, select: list[int]) -> None:
    if observation.current is not None and observation.current.result >= 0:
        raise ValueError("Cannot be selected because the battle has ended.")
    if observation.select is None:
        raise ValueError("Cannot be selected because the battle has ended.")
    if not isinstance(select, list) or not all(isinstance(i, int) for i in select):
        raise ValueError("select is not list[int]")
    if not (observation.select.minCount <= len(select) <= observation.select.maxCount):
        raise ValueError("Must be Observation.select.minCount <= len(select) <= Observation.select.maxCount.")
    n_options = len(observation.select.option)
    if any(i < 0 or i >= n_options for i in select):
        raise ValueError("Must be 0 <= select elements < len(Observation.select.option).")
    if len(set(select)) != len(select):
        raise ValueError("Duplicate select elements.")


def _pokemon_stub(card_id: int, player_index: int) -> dict:
    card = _native_card_lookup(card_id)
    hp = int(card.hp) if card else 0
    return {
        "id": int(card_id),
        "serial": 0,
        "playerIndex": player_index,
        "hp": hp,
        "maxHp": hp,
        "appearThisTurn": False,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    }


def _cards_from_ids(ids: list[int], player_index: int) -> list[dict]:
    return [
        {"id": int(cid), "serial": 0, "playerIndex": player_index}
        for cid in ids
    ]


def _inject_hidden_predictions(cur: dict, obs: Observation, your_deck: list[int],
                               your_prize: list[int], opponent_deck: list[int],
                               opponent_prize: list[int], opponent_hand: list[int],
                               opponent_active: list[int]) -> dict:
    your_index = int(cur["yourIndex"])
    mine = cur["players"][your_index]
    opp = cur["players"][1 - your_index]

    if obs.select is not None and obs.select.deck is not None and not your_deck:
        mine["deck"] = []
    else:
        mine["deck"] = list(your_deck[: int(mine["deckCount"])])
        mine["deckKnown"] = len(mine["deck"]) == int(mine["deckCount"])
    mine["prize"] = _cards_from_ids(your_prize[: len(mine["prize"])], your_index)
    mine["prizesKnown"] = len(mine["prize"]) == len(your_prize[: len(mine["prize"])])

    opp["deck"] = list(opponent_deck[: int(opp["deckCount"])])
    opp["deckKnown"] = len(opp["deck"]) == int(opp["deckCount"])
    opp["prize"] = _cards_from_ids(opponent_prize[: len(opp["prize"])], 1 - your_index)
    opp["prizesKnown"] = len(opp["prize"]) == len(opponent_prize[: len(opp["prize"])])
    opp["hand"] = _cards_from_ids(opponent_hand[: int(opp["handCount"])], 1 - your_index)
    if opp.get("active") and opp["active"][0] is None and opponent_active:
        opp["active"][0] = _pokemon_stub(opponent_active[0], 1 - your_index)
    return cur


def _native_observation_from_state(state, logs: list[dict] | None = None) -> Observation:
    obs = _native_engine().cg_observation(state)
    obs["logs"] = list(logs or [])
    obs["search_begin_input"] = None
    return to_observation_class(obs)


def _native_search_state_summary(search_id: int) -> dict:
    sid = int(search_id)
    rec = _NATIVE_SEARCH_STATES.get(sid)
    if rec is None or rec.released or sid in _NATIVE_SEARCH_RELEASED:
        raise ValueError("There is no element with the specified search_id.")
    return _native_engine().native_state_summary(rec.state)


def _store_native_search_state(state, logs: list[dict] | None = None,
                               manual_coin: bool = False) -> SearchState:
    global _NATIVE_SEARCH_NEXT_ID
    search_id = _NATIVE_SEARCH_NEXT_ID
    _NATIVE_SEARCH_NEXT_ID += 1
    obs = _native_observation_from_state(state, logs)
    _NATIVE_SEARCH_STATES[search_id] = _NativeSearchRecord(state, obs, manual_coin)
    return SearchState(obs, search_id)


def _same_ids(actual, expected: list[int]) -> bool:
    return [int(v) for v in actual] == [int(v) for v in expected]


def _same_id_multiset(actual, expected: list[int]) -> bool:
    return sorted(int(v) for v in actual) == sorted(int(v) for v in expected)


def _validate_native_search_root(
    state,
    agent_observation: Observation,
    your_deck: list[int],
    your_prize: list[int],
    opponent_deck: list[int],
    opponent_prize: list[int],
    opponent_hand: list[int],
    opponent_active: list[int],
) -> None:
    if os.environ.get("PTCG_NATIVE_VALIDATE_SEARCH", "").lower() not in {"1", "true", "yes"}:
        return
    E = _native_engine()
    summary = E.native_state_summary(state)
    me = int(agent_observation.current.yourIndex)
    opp = 1 - me
    players = summary["hidden"]["players"]
    mine = players[me]
    enemy = players[opp]
    if not _same_ids(mine["deck"], your_deck[: len(mine["deck"])]):
        raise RuntimeError("native search validation failed: your_deck mismatch")
    if not _same_ids(mine["prizes"], your_prize[: len(mine["prizes"])]):
        raise RuntimeError("native search validation failed: your_prize mismatch")
    if not _same_ids(enemy["deck"], opponent_deck[: len(enemy["deck"])]):
        raise RuntimeError("native search validation failed: opponent_deck mismatch")
    if not _same_ids(enemy["prizes"], opponent_prize[: len(enemy["prizes"])]):
        raise RuntimeError("native search validation failed: opponent_prize mismatch")
    if not _same_id_multiset(enemy["hand"], opponent_hand[: len(enemy["hand"])]):
        raise RuntimeError("native search validation failed: opponent_hand mismatch")
    if opponent_active:
        active = E.canonical(state)["players"][opp]["active"]
        if active is not None and int(active["id"]) != int(opponent_active[0]):
            raise RuntimeError("native search validation failed: opponent_active mismatch")
    if agent_observation.select is not None:
        expected_context = int(agent_observation.select.context)
        actual_context = int(summary["transients"]["pending"]["context"])
        if expected_context == int(SelectContext.MAIN):
            actual_context = int(SelectContext.MAIN)
        if actual_context != expected_context:
            raise RuntimeError("native search validation failed: pending context mismatch")


def _native_search_begin(agent_observation: Observation, your_deck: list[int],
                         your_prize: list[int], opponent_deck: list[int],
                         opponent_prize: list[int], opponent_hand: list[int],
                         opponent_active: list[int],
                         manual_coin: bool = False) -> SearchState:
    (
        your_deck,
        your_prize,
        opponent_deck,
        opponent_prize,
        opponent_hand,
        opponent_active,
    ) = _validate_search_begin_inputs(
        agent_observation,
        your_deck,
        your_prize,
        opponent_deck,
        opponent_prize,
        opponent_hand,
        opponent_active,
    )

    E = _native_engine()
    from ptcg.cg.native_payload import decode_native_search_begin, registered_native_state

    payload = decode_native_search_begin(agent_observation.search_begin_input)
    payload_current = payload.get("current") if payload else None
    base_current = copy.deepcopy(payload_current) if isinstance(payload_current, dict) else _plain(agent_observation.current)
    main_options = payload.get("main_options") if payload else None
    seed = int(payload.get("seed", 0)) if payload else 0

    cur = _inject_hidden_predictions(
        base_current,
        agent_observation,
        your_deck,
        your_prize,
        opponent_deck,
        opponent_prize,
        opponent_hand,
        opponent_active,
    )
    state = E.load_state(cur, seed, main_options)
    source_state = registered_native_state(payload)
    payload_context = int(payload.get("context", -1)) if payload else -1
    if source_state is not None:
        E.copy_search_transients(state, source_state)
    elif payload and payload.get("transients") is not None:
        E.restore_search_transients(state, payload["transients"])
    elif payload_context >= 0:
        raise ValueError(
            "Native search_begin_input is stale and does not contain a portable "
            "transient snapshot."
        )
    elif payload is None and agent_observation.select is not None and int(agent_observation.select.context) != int(SelectContext.MAIN):
        raise ValueError("Native pending observation requires native search_begin_input.")
    _validate_native_search_root(
        state,
        agent_observation,
        your_deck,
        your_prize,
        opponent_deck,
        opponent_prize,
        opponent_hand,
        opponent_active,
    )
    return _store_native_search_state(state, manual_coin=manual_coin)


def _native_search_step(search_id: int, select: list[int]) -> SearchState:
    global _NATIVE_SEARCH_SEED
    sid = int(search_id)
    if sid in _NATIVE_SEARCH_RELEASED:
        raise ValueError("Released item.")
    rec = _NATIVE_SEARCH_STATES.get(sid)
    if rec is None:
        raise ValueError("There is no element with the specified search_id.")
    if rec.released:
        raise ValueError("Released item.")
    _validate_native_select(rec.observation, select)

    import ptcg_engine as E
    from ptcg.cg.native_backend import _action_logs

    state = E.clone(rec.state)
    before = E.cg_observation(state)
    descriptor = None
    pending = False
    try:
        E.clear_native_logs(state)
        pending = E.pending_decision(state) is not None
        if pending:
            E.resolve(state, list(select))
        else:
            _ctx, descriptors = E.action_view(state)
            if len(select) != 1:
                raise ValueError("Must be Observation.select.minCount <= len(select) <= Observation.select.maxCount.")
            action = int(select[0])
            if action < 0 or action >= len(descriptors):
                raise ValueError("Must be 0 <= select elements < len(Observation.select.option).")
            descriptor = tuple(descriptors[action])
            E.apply(state, descriptor)
        _NATIVE_SEARCH_SEED = (
            _NATIVE_SEARCH_SEED * 6364136223846793005 + 1442695040888963407
        ) & ((1 << 64) - 1)
    except Exception as exc:
        raise ValueError("Must be 0 <= select elements < len(Observation.select.option).") from exc
    after = E.cg_observation(state)
    engine_logs = list(E.native_logs(state))
    return _store_native_search_state(
        state,
        engine_logs or _action_logs(before, after, descriptor, pending),
        manual_coin=rec.manual_coin,
    )

