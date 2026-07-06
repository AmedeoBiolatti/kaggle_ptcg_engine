from __future__ import annotations

import ptcg_engine as engine


def mega_lucario_deck() -> list[int]:
    pairs = [
        (673, 2),
        (674, 2),
        (675, 2),
        (676, 3),
        (677, 3),
        (678, 4),
        (1102, 4),
        (1123, 2),
        (1141, 4),
        (1142, 4),
        (1152, 4),
        (1159, 1),
        (1182, 2),
        (1192, 4),
        (1227, 4),
        (1252, 2),
        (6, 13),
    ]
    return [card for card, count in pairs for _ in range(count)]


def main() -> None:
    deck = mega_lucario_deck()
    state = engine.new_game(deck, deck, 1)
    observation = engine.observation_ids(state)
    actions = engine.action_ids(state)

    print("in_play", observation["in_play"].shape)
    print("zones", observation["zones"].shape)
    print("action_options", actions["options"].shape)
    print("legal_actions", int(actions["mask"].sum()))


if __name__ == "__main__":
    main()
