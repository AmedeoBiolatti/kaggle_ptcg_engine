"""Benchmark reference cg, native cg-format, native direct, and native vectorized.

The four measured paths are intentionally separate:

* reference_cg_format: original ``cg.game`` if the closed shared library exists.
* native_cg_format: ``ptcg.cg.game`` with ``PTCG_BACKEND=native``.
* native_direct: direct ``ptcg_engine`` GameState + native action indices.
* native_vectorized: ``ptcg_engine.VectorEnv`` stepping N games per call.

Run from the repository root:

    python benchmarks/bench_engines.py --steps 2000 --batch-size 256

When a reference ``cg`` package is not available through ``PTCG_REFERENCE_ROOT``
or a local ``cg/`` folder,
``reference_cg_format`` is reported as skipped.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


# Only pyds built from this checkout are eligible.
_NATIVE_BUILD_PATHS = [
    ROOT / "engine" / "build" / "Release",
    ROOT / "engine" / "build",
]


def _add_paths() -> None:
    for path in [ROOT, *_NATIVE_BUILD_PATHS]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


_add_paths()

from validation.decks import ALL_DECKS  # noqa: E402


@dataclass
class BenchResult:
    name: str
    steps: int
    seconds: float
    steps_per_second: float
    games_started: int = 0
    skipped: bool = False
    reason: str = ""


def _timed(name: str, fn) -> BenchResult:
    start = time.perf_counter()
    try:
        steps, games_started = fn()
    except Exception as exc:
        return BenchResult(
            name=name,
            steps=0,
            seconds=0.0,
            steps_per_second=0.0,
            skipped=True,
            reason=f"{type(exc).__name__}: {exc}",
        )
    seconds = time.perf_counter() - start
    rate = steps / seconds if seconds > 0 else 0.0
    return BenchResult(name, steps, seconds, rate, games_started)


def _cg_selection(obs: dict[str, Any], rng: random.Random, selector: str) -> list[int] | None:
    current = obs.get("current")
    if current is not None and int(current.get("result", -1)) >= 0:
        return None
    select = obs.get("select")
    if select is None:
        return None
    options = list(range(len(select.get("option") or [])))
    min_count = int(select.get("minCount", 1))
    max_count = int(select.get("maxCount", min_count))
    if min_count > len(options):
        return None
    if selector == "first":
        return options[:min_count]
    count = rng.randint(min_count, min(max_count, len(options)))
    if count <= 0:
        return []
    return rng.sample(options, count)


def _cg_shared_lib_exists(package_root: Path) -> bool:
    cg_dir = package_root / "cg"
    return (cg_dir / "cg.dll").exists() or (cg_dir / "libcg.so").exists()


def _reference_root() -> Path | None:
    explicit = os.environ.get("PTCG_REFERENCE_ROOT")
    if explicit:
        root = Path(explicit)
        if _cg_shared_lib_exists(root):
            return root
    return ROOT if _cg_shared_lib_exists(ROOT) else None


def _clear_cg_modules() -> None:
    for name in list(sys.modules):
        if name == "cg" or name.startswith("cg.") or name == "ptcg.cg" or name.startswith("ptcg.cg."):
            del sys.modules[name]


def _clear_native_modules() -> None:
    sys.modules.pop("ptcg_engine", None)


def _prefer_import_root(path: Path) -> None:
    raw = str(path)
    if raw in sys.path:
        sys.path.remove(raw)
    sys.path.insert(0, raw)


def _prefer_native_paths() -> None:
    for path in [ROOT, *_NATIVE_BUILD_PATHS]:
        raw = str(path)
        while raw in sys.path:
            sys.path.remove(raw)
    _prefer_import_root(ROOT)
    for path in reversed([p for p in _NATIVE_BUILD_PATHS if p.exists()]):
        sys.path.insert(0, str(path))


def _loaded_native_pyd() -> str | None:
    module = sys.modules.get("ptcg_engine")
    return getattr(module, "__file__", None) if module is not None else None


def bench_cg_game(deck0: list[int], deck1: list[int], *, steps: int, seed: int,
                  backend: str, selector: str) -> tuple[int, int]:
    if backend == "reference":
        ref_root = _reference_root()
        if ref_root is None:
            raise RuntimeError(
                "reference cg package not found; set PTCG_REFERENCE_ROOT or provide ./cg/"
            )
        _clear_cg_modules()
        _prefer_import_root(ref_root)
        os.environ.pop("PTCG_BACKEND", None)
        os.environ.pop("CG_BACKEND", None)
        os.environ.pop("PTCG_NATIVE_LAZY_SEARCH", None)
    else:
        _clear_cg_modules()
        _clear_native_modules()
        _prefer_native_paths()
        os.environ["PTCG_BACKEND"] = "native"
        os.environ["PTCG_NATIVE_SEED"] = str(seed)
        os.environ["PTCG_NATIVE_LAZY_SEARCH"] = "1"

    if backend == "reference":
        import cg.game as game
    else:
        import ptcg.cg.game as game

    rng = random.Random(seed)
    done_steps = 0
    games_started = 0
    obs = None
    while done_steps < steps:
        if obs is None:
            obs, start_data = game.battle_start(deck0, deck1)
            games_started += 1
            if obs is None:
                raise RuntimeError(f"battle_start failed: errorType={start_data.errorType}")
        selection = _cg_selection(obs, rng, selector)
        if selection is None:
            game.battle_finish()
            obs = None
            continue
        obs = game.battle_select(selection)
        done_steps += 1
    game.battle_finish()
    return done_steps, games_started


def bench_native_direct(deck0: list[int], deck1: list[int], *, steps: int,
                        seed: int, selector: str) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E

    rng = random.Random(seed)
    native_seed = seed
    state = E.new_game(deck0, deck1, native_seed)
    done_steps = 0
    games_started = 1
    while done_steps < steps:
        if int(state.result) >= 0:
            native_seed += 1
            state = E.new_game(deck0, deck1, native_seed)
            games_started += 1
        _ctx, descriptors = E.action_view(state)
        n = len(descriptors)
        if n <= 0:
            native_seed += 1
            state = E.new_game(deck0, deck1, native_seed)
            games_started += 1
            continue
        action = 0 if selector == "first" else rng.randrange(n)
        native_seed = int(E.step_action(state, action, native_seed))
        done_steps += 1
    return done_steps, games_started


def _random_actions_from_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    actions = np.zeros(mask.shape[0], dtype=np.int32)
    for i, row in enumerate(mask):
        valid = np.flatnonzero(row)
        actions[i] = int(rng.choice(valid)) if len(valid) else 0
    return actions


def bench_native_vectorized(deck0: list[int], deck1: list[int], *, steps: int,
                            seed: int, batch_size: int, selector: str) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E

    rng = np.random.default_rng(seed)
    env = E.VectorEnv(deck0, deck1, batch_size, seed)
    _obs, mask, _player, _result = env.observe()
    done_steps = 0
    while done_steps < steps:
        if selector == "first":
            actions = np.zeros(batch_size, dtype=np.int32)
        else:
            actions = _random_actions_from_mask(np.asarray(mask), rng)
        _obs, _reward, _done, mask, _player, _result = env.step(actions)
        done_steps += batch_size
    return done_steps, batch_size


def run(args: argparse.Namespace) -> list[BenchResult]:
    deck0 = list(ALL_DECKS[args.deck])
    deck1 = list(ALL_DECKS[args.deck1 or args.deck])

    results = [
        _timed(
            "reference_cg_format",
            lambda: bench_cg_game(
                deck0, deck1, steps=args.steps, seed=args.seed,
                backend="reference", selector=args.selector,
            ),
        ),
        _timed(
            "native_cg_format",
            lambda: bench_cg_game(
                deck0, deck1, steps=args.steps, seed=args.seed,
                backend="native", selector=args.selector,
            ),
        ),
        _timed(
            "native_direct",
            lambda: bench_native_direct(
                deck0, deck1, steps=args.steps, seed=args.seed, selector=args.selector
            ),
        ),
        _timed(
            "native_vectorized",
            lambda: bench_native_vectorized(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
            ),
        ),
    ]
    return results


def _print_table(results: list[BenchResult]) -> None:
    print(f"{'name':24} {'steps':>10} {'seconds':>10} {'steps/s':>12} status")
    print("-" * 72)
    for result in results:
        status = f"SKIP {result.reason}" if result.skipped else "ok"
        print(
            f"{result.name:24} {result.steps:10d} {result.seconds:10.4f} "
            f"{result.steps_per_second:12.1f} {status}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark cg/native engine paths")
    parser.add_argument("--deck", choices=sorted(ALL_DECKS), default="mega_lucario")
    parser.add_argument("--deck1", choices=sorted(ALL_DECKS), default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--selector", choices=["first", "random"], default="first",
                        help="Action selector used during the benchmark")
    parser.add_argument("--json", action="store_true", help="Print JSON results")
    args = parser.parse_args(argv)

    results = run(args)
    native_pyd = _loaded_native_pyd()
    if args.json:
        payload = {
            "native_pyd": native_pyd,
            "results": [asdict(result) for result in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_table(results)
        print(f"native pyd: {native_pyd or 'NOT LOADED'}")
    return 0 if all(not result.skipped for result in results[1:]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
