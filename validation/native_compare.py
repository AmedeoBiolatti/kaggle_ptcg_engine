"""CLI validation modes for native-vs-reference cg parity.

Mode 1: offline random games on a deck.
Mode 2: use ``PTCG_BACKEND=shadow`` in normal code to compare every public
``cg.game`` call while returning the configured primary engine's observation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_DIRS = (ROOT / "engine" / "build" / "Release", ROOT / "engine" / "build")
for path in (str(ROOT), *(str(build_dir) for build_dir in BUILD_DIRS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from validation.decks import ALL_DECKS  # noqa: E402
from validation.oracle.random_branch_parity import (  # noqa: E402
    ParityFailure,
    ParityStats,
    run_random_branch_parity,
)


def _parse_deck_file(path: str) -> list[int]:
    text = Path(path).read_text(encoding="utf-8")
    values = [int(raw) for raw in re.findall(r"\d+", text)]
    if len(values) != 60:
        raise ValueError(f"{path} has {len(values)} card ids; expected 60")
    return values


def _deck(value: str) -> list[int]:
    if value in ALL_DECKS:
        return list(ALL_DECKS[value])
    return _parse_deck_file(value)


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _stats_json(stats: ParityStats, args: argparse.Namespace) -> dict[str, object]:
    return {
        "ok": True,
        "mode": args.mode,
        "deck0": args.deck0,
        "deck1": args.deck1 or args.deck0,
        "seed": args.seed,
        "games": stats.games,
        "battle_steps": stats.battle_steps,
        "states_checked": stats.states_checked,
        "branches_checked": stats.branches_checked,
        "branches_replayed_coin": stats.branches_replayed_coin,
        "branches_skipped_coin": stats.branches_skipped_coin,
        "branches_skipped_missing_native": stats.branches_skipped_missing_native,
        "branches_skipped_deck_search": stats.branches_skipped_deck_search,
        "branches_skipped_limit": stats.branches_skipped_limit,
        "action_kinds": _counter_dict(stats.action_kinds),
        "next_contexts": _counter_dict(stats.next_contexts),
    }


def run_random_games(args: argparse.Namespace) -> int:
    deck0 = _deck(args.deck0)
    deck1 = _deck(args.deck1 or args.deck0)

    try:
        stats = run_random_branch_parity(
            deck0,
            deck1,
            games=args.games,
            seed=args.seed,
            max_steps=args.max_steps,
            max_states=args.max_states,
            branch_limit=args.branch_limit,
            strict_duplicates=args.strict_duplicates,
            strict_order=args.strict_order,
            skip_deck_search_branches=args.skip_deck_search_branches,
        )
    except ParityFailure as exc:
        print(f"native parity failure: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"native validation error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(_stats_json(stats, args), sort_keys=True))
    else:
        print(stats.summary())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Native-vs-cg validation tools")
    parser.add_argument("--mode", choices=["random-games"], default="random-games")
    parser.add_argument("--deck", dest="deck0", default="mega_lucario",
                        help="Deck name from validation.decks or path to 60-id deck file")
    parser.add_argument("--deck1", default=None,
                        help="Optional second deck name/path; defaults to --deck")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-states", type=int, default=None,
                        help="Stop after checking this many MAIN states")
    parser.add_argument("--branch-limit", type=int, default=None,
                        help="Limit checked branches per sampled MAIN state")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--strict-duplicates", action="store_true",
                        help="Compare duplicate semantic options as multiplicities")
    parser.add_argument("--strict-order", action="store_true",
                        help="Compare legal/pending option order after semantic normalization")
    parser.add_argument("--skip-deck-search-branches", action="store_true",
                        help="Skip result comparison for deck-search PLAY branches")
    parser.add_argument("--json", action="store_true",
                        help="Print machine-readable summary JSON")
    args = parser.parse_args(argv)
    return run_random_games(args)


if __name__ == "__main__":
    raise SystemExit(main())
