"""Benchmark reference cg, native cg-format, native direct, and native vectorized.

The measured paths are intentionally separate:

* reference_cg_format: original ``cg.game`` if the closed shared library exists.
* native_cg_format: ``ptcg.cg.game`` with ``PTCG_BACKEND=native``.
* native_rl_step_only: direct ``ptcg_engine`` GameState + compact action indices.
* native_rl_*: direct compact stepping while materializing each observation format.
* native_vectorized: ``ptcg_engine.VectorEnv`` stepping N games per call.
* native_nn_adapter: packed symbolic state/action ids plus ``ptcg.nn`` inference.
* native_vectorized_nn_adapter: vectorized action ids plus ``ptcg.nn`` inference.

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
from collections.abc import Callable
from contextlib import nullcontext
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
    warm_seconds: float | None = None
    warm_steps_per_second: float | None = None
    games_started: int = 0
    skipped: bool = False
    reason: str = ""
    metrics: dict[str, Any] | None = None


def _timed(name: str, fn) -> BenchResult:
    start = time.perf_counter()
    try:
        output = fn()
    except Exception as exc:
        if "jax" in name:
            for module_name in list(sys.modules):
                if module_name == "jax" or module_name.startswith("jax."):
                    sys.modules.pop(module_name, None)
        return BenchResult(
            name=name,
            steps=0,
            seconds=0.0,
            steps_per_second=0.0,
            skipped=True,
            reason=f"{type(exc).__name__}: {exc}",
        )
    metrics = None
    if len(output) == 4:
        steps, games_started, warm_seconds, metrics = output
    elif len(output) == 3:
        steps, games_started, warm_seconds = output
    else:
        steps, games_started = output
        warm_seconds = None
    seconds = time.perf_counter() - start
    rate = steps / seconds if seconds > 0 else 0.0
    warm_rate = steps / warm_seconds if warm_seconds and warm_seconds > 0 else None
    if metrics is not None and warm_seconds is not None:
        metrics = dict(metrics)
        metrics.setdefault("setup_s", max(0.0, seconds - warm_seconds))
    return BenchResult(name, steps, seconds, rate, warm_seconds, warm_rate, games_started, metrics=metrics)


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


def _native_has(name: str) -> bool:
    _clear_native_modules()
    _prefer_native_paths()
    try:
        import ptcg_engine as E
    except Exception:
        return False
    return hasattr(E, name)


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


def _select_native_action(mask: Any, rng: random.Random, selector: str) -> int:
    valid = [i for i, ok in enumerate(np.asarray(mask)) if ok]
    if not valid:
        return 0
    return valid[0] if selector == "first" else rng.choice(valid)


def bench_native_rl(
    deck0: list[int],
    deck1: list[int],
    *,
    steps: int,
    seed: int,
    selector: str,
    observe: Callable[[Any, Any], Any] | None = None,
) -> tuple[int, int]:
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
        if observe is not None:
            observe(E, state)
        n, mask = E.rl_legal_mask(state)
        if int(n) <= 0:
            native_seed += 1
            state = E.new_game(deck0, deck1, native_seed)
            games_started += 1
            continue
        action = _select_native_action(mask, rng, selector)
        native_seed = int(E.rl_step(state, action, native_seed))
        done_steps += 1
    return done_steps, games_started


def bench_native_descriptor_direct(deck0: list[int], deck1: list[int], *, steps: int,
                                   seed: int, selector: str) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E

    if not hasattr(E, "step_action"):
        raise RuntimeError("ptcg_engine.step_action is not available in this build")

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


def bench_native_vectorized_action_ids(
    deck0: list[int], deck1: list[int], *, steps: int, seed: int,
    batch_size: int, selector: str
) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E

    rng = np.random.default_rng(seed)
    env = E.VectorEnv(deck0, deck1, batch_size, seed)
    _obs, mask, _player, _result = env.observe()
    done_steps = 0
    while done_steps < steps:
        env.action_ids()
        if selector == "first":
            actions = np.zeros(batch_size, dtype=np.int32)
        else:
            actions = _random_actions_from_mask(np.asarray(mask), rng)
        _obs, _reward, _done, mask, _player, _result = env.step(actions)
        done_steps += batch_size
    return done_steps, batch_size


def bench_native_nn_adapter(
    deck0: list[int],
    deck1: list[int],
    *,
    steps: int,
    seed: int,
    selector: str,
    card_dim: int,
    hidden_dim: int,
) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E
    import torch
    from ptcg.nn import PtcgPolicyValueNet

    torch.set_num_threads(1)
    model = PtcgPolicyValueNet(card_dim=card_dim, hidden_dim=hidden_dim)
    model.eval()

    rng = random.Random(seed)
    native_seed = seed
    state = E.new_game(deck0, deck1, native_seed)
    done_steps = 0
    games_started = 1
    with torch.inference_mode():
        while done_steps < steps:
            if int(state.result) >= 0:
                native_seed += 1
                state = E.new_game(deck0, deck1, native_seed)
                games_started += 1
            state_ids = E.rl_state_ids(state)
            action_ids = E.rl_action_ids(state)
            model(state_ids, action_ids)
            n, mask = E.rl_legal_mask(state)
            if int(n) <= 0:
                native_seed += 1
                state = E.new_game(deck0, deck1, native_seed)
                games_started += 1
                continue
            action = _select_native_action(mask, rng, selector)
            native_seed = int(E.rl_step(state, action, native_seed))
            done_steps += 1
    return done_steps, games_started


def bench_native_nn_forward_static(
    deck0: list[int],
    deck1: list[int],
    *,
    steps: int,
    seed: int,
    card_dim: int,
    hidden_dim: int,
) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E
    import torch
    from ptcg.nn import PtcgPolicyValueNet

    torch.set_num_threads(1)
    state = E.new_game(deck0, deck1, seed)
    state_ids = E.rl_state_ids(state)
    action_ids = E.rl_action_ids(state)
    model = PtcgPolicyValueNet(card_dim=card_dim, hidden_dim=hidden_dim)
    model.eval()

    with torch.inference_mode():
        for _ in range(steps):
            model(state_ids, action_ids)
    return steps, 1


def bench_native_vectorized_legacy_all_features(
    deck0: list[int],
    deck1: list[int],
    *,
    steps: int,
    seed: int,
    batch_size: int,
    selector: str,
) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E

    rng = np.random.default_rng(seed)
    if not hasattr(E, "PpoBatchEnv"):
        raise RuntimeError("ptcg_engine.PpoBatchEnv is not available in this build")
    env = E.PpoBatchEnv(deck0, deck1, batch_size, seed)
    _obs, mask, _player, _action_features, _card_features, _deck_features, _belief_features, _belief_summary = (
        env.observe_with_all_features()
    )
    done_steps = 0
    while done_steps < steps:
        if selector == "first":
            actions = np.zeros(batch_size, dtype=np.int32)
        else:
            actions = _random_actions_from_mask(np.asarray(mask), rng)
        _obs, _reward, _done, mask, _player, _result, _episode_len = env.step(actions)
        env.observe_with_all_features()
        done_steps += batch_size
    return done_steps, batch_size


def bench_native_vectorized_nn_adapter(
    deck0: list[int],
    deck1: list[int],
    *,
    steps: int,
    seed: int,
    batch_size: int,
    selector: str,
    card_dim: int,
    hidden_dim: int,
) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E
    import torch
    from ptcg.nn import PtcgPolicyValueNet

    torch.set_num_threads(1)
    model = PtcgPolicyValueNet(card_dim=card_dim, hidden_dim=hidden_dim)
    model.eval()

    rng = np.random.default_rng(seed)
    env = E.VectorEnv(deck0, deck1, batch_size, seed)
    _obs, mask, _player, _result = env.observe()
    done_steps = 0
    with torch.inference_mode():
        while done_steps < steps:
            state_batch = env.state_ids()
            action_ids = env.action_ids()
            out = model(state_batch, action_ids)
            if selector == "first":
                actions = np.zeros(batch_size, dtype=np.int32)
            else:
                actions = out["logits"].argmax(dim=-1).cpu().numpy().astype(np.int32)
                empty = np.asarray(action_ids["mask"]).sum(axis=1) <= 0
                if np.any(empty):
                    actions[empty] = _random_actions_from_mask(np.asarray(mask)[empty], rng)
            _obs, _reward, _done, mask, _player, _result = env.step(actions)
            done_steps += batch_size
    return done_steps, batch_size


def bench_native_vectorized_nn_adapter_reuse(
    deck0: list[int],
    deck1: list[int],
    *,
    steps: int,
    seed: int,
    batch_size: int,
    selector: str,
    card_dim: int,
    hidden_dim: int,
    compile_model: bool = False,
    device: str = "cpu",
    matmul_precision: str | None = None,
) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E
    import torch
    from ptcg.nn import PtcgPolicyValueNet

    if not hasattr(E.VectorEnv, "observe_ids_into"):
        raise RuntimeError("VectorEnv.observe_ids_into is not available in this build")

    torch.set_num_threads(1)
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch CUDA requested, but torch.cuda.is_available() is false")
    torch_device = torch.device(device)
    model = PtcgPolicyValueNet(card_dim=card_dim, hidden_dim=hidden_dim)
    model.to(torch_device)
    model.eval()

    rng = np.random.default_rng(seed)
    env = E.VectorEnv(deck0, deck1, batch_size, seed)
    state_np = env.state_ids()
    action_np = env.action_ids()
    player = np.empty((batch_size,), dtype=np.int32)
    result = np.empty((batch_size,), dtype=np.int32)
    torch_state = {
        key: torch.from_numpy(value)
        for key, value in state_np.items()
        if hasattr(value, "shape")
    }
    torch_action = {
        key: torch.from_numpy(value)
        for key, value in action_np.items()
        if hasattr(value, "shape")
    }

    def input_args() -> tuple[Any, ...]:
        args = (
            torch_state["in_play"],
            torch_state["zones"],
            torch_state["player_counts"],
            torch_state["player_status"],
            torch_state["global"],
            torch_action["options"],
            torch_action["mask"],
        )
        if torch_device.type == "cpu":
            return args
        return tuple(t.to(torch_device, non_blocking=True) for t in args)

    def sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    first_actions = np.zeros(batch_size, dtype=np.int32)
    infer = model.forward_tensors
    if compile_model:
        infer = torch.compile(infer, mode="reduce-overhead")

    done_steps = 0
    with torch.inference_mode():
        if compile_model:
            env.observe_ids_into(state_np, action_np, player, result)
            infer(*input_args())
            sync()
        sync()
        warm_start = time.perf_counter()
        while done_steps < steps:
            env.observe_ids_into(state_np, action_np, player, result)
            out = infer(*input_args())
            if selector == "first":
                actions = first_actions
                sync()
            else:
                actions = out["logits"].argmax(dim=-1).cpu().numpy().astype(np.int32)
                empty = np.asarray(action_np["mask"]).sum(axis=1) <= 0
                if np.any(empty):
                    actions[empty] = _random_actions_from_mask(action_np["mask"][empty], rng)
            env.step(actions)
            done_steps += batch_size
    sync()
    warm_seconds = time.perf_counter() - warm_start
    return done_steps, batch_size, warm_seconds


def bench_native_vectorized_jax_reuse(
    deck0: list[int],
    deck1: list[int],
    *,
    steps: int,
    seed: int,
    batch_size: int,
    selector: str,
    card_dim: int,
    hidden_dim: int,
    compile_model: bool = True,
    policy_only: bool = False,
    matmul_precision: str | None = None,
) -> tuple[int, int]:
    _clear_native_modules()
    _prefer_native_paths()
    import jax
    import jax.numpy as jnp
    import ptcg_engine as E
    from ptcg.jax_nn import forward_policy_value, forward_tensors, init_params

    if not hasattr(E.VectorEnv, "observe_ids_into"):
        raise RuntimeError("VectorEnv.observe_ids_into is not available in this build")

    env = E.VectorEnv(deck0, deck1, batch_size, seed)
    state_np = env.state_ids()
    action_np = env.action_ids()
    player = np.empty((batch_size,), dtype=np.int32)
    result = np.empty((batch_size,), dtype=np.int32)
    params = init_params(
        jax.random.PRNGKey(seed),
        card_dim=card_dim,
        hidden_dim=hidden_dim,
    )
    forward = forward_policy_value if policy_only else forward_tensors
    precision_ctx = (
        nullcontext()
        if not matmul_precision or matmul_precision == "default"
        else jax.default_matmul_precision(matmul_precision)
    )
    infer = jax.jit(forward) if compile_model else forward
    first_actions = np.zeros(batch_size, dtype=np.int32)
    rng = np.random.default_rng(seed)

    with precision_ctx:
        env.observe_ids_into(state_np, action_np, player, result)
        out = infer(
            params,
            jnp.asarray(state_np["in_play"]),
            jnp.asarray(state_np["zones"]),
            jnp.asarray(state_np["player_counts"]),
            jnp.asarray(state_np["player_status"]),
            jnp.asarray(state_np["global"]),
            jnp.asarray(action_np["options"]),
            jnp.asarray(action_np["mask"]),
        )
        jax.block_until_ready(out)
        done_steps = 0
        warm_start = time.perf_counter()
        while done_steps < steps:
            env.observe_ids_into(state_np, action_np, player, result)
            out = infer(
                params,
                jnp.asarray(state_np["in_play"]),
                jnp.asarray(state_np["zones"]),
                jnp.asarray(state_np["player_counts"]),
                jnp.asarray(state_np["player_status"]),
                jnp.asarray(state_np["global"]),
                jnp.asarray(action_np["options"]),
                jnp.asarray(action_np["mask"]),
            )
            if selector == "first":
                actions = first_actions
            else:
                logits = np.asarray(jax.device_get(out["logits"]))
                actions = logits.argmax(axis=-1).astype(np.int32)
                empty = action_np["mask"].sum(axis=1) <= 0
                if np.any(empty):
                    actions[empty] = _random_actions_from_mask(action_np["mask"][empty], rng)
            jax.block_until_ready(out)
            env.step(actions)
            done_steps += batch_size
    warm_seconds = time.perf_counter() - warm_start
    return done_steps, batch_size, warm_seconds


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    bootstrap: np.ndarray,
    *,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = np.zeros((rewards.shape[1],), dtype=np.float32)
    next_value = bootstrap.astype(np.float32)
    for t in range(rewards.shape[0] - 1, -1, -1):
        next_nonterminal = 1.0 - dones[t].astype(np.float32)
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_adv = delta + gamma * lam * next_nonterminal * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    adv_mean = float(advantages.mean())
    adv_std = float(advantages.std())
    advantages = (advantages - adv_mean) / max(adv_std, 1e-8)
    return advantages.astype(np.float32), returns.astype(np.float32)


def _ppo_indices(total: int, minibatch_size: int, epochs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    usable = (total // minibatch_size) * minibatch_size
    if usable <= 0:
        raise ValueError("ppo_minibatch_size must be <= rollout_steps * batch_size")
    batches = []
    for _ in range(epochs):
        perm = rng.permutation(total)[:usable].astype(np.int32)
        batches.append(perm.reshape(-1, minibatch_size))
    return np.concatenate(batches, axis=0)


def bench_native_ppo_torch(
    deck0: list[int],
    deck1: list[int],
    *,
    seed: int,
    batch_size: int,
    rollout_steps: int,
    minibatch_size: int,
    epochs: int,
    gamma: float,
    lam: float,
    clip: float,
    vf_coef: float,
    ent_coef: float,
    lr: float,
    card_dim: int,
    hidden_dim: int,
    device: str,
    matmul_precision: str | None,
    compile_model: bool,
    mixed_precision: str,
) -> tuple[int, int, float, dict[str, Any]]:
    _clear_native_modules()
    _prefer_native_paths()
    import ptcg_engine as E
    import torch
    import torch.nn.functional as F
    from ptcg.nn import PtcgPolicyValueNet

    if not hasattr(E.VectorEnv, "observe_ids_into"):
        raise RuntimeError("VectorEnv.observe_ids_into is not available in this build")

    torch.set_num_threads(1)
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("torch CUDA requested, but torch.cuda.is_available() is false")
    torch_device = torch.device(device)

    env = E.VectorEnv(deck0, deck1, batch_size, seed)
    state_np = env.state_ids()
    action_np = env.action_ids()
    state_buf = {
        key: np.empty((rollout_steps, *value.shape), dtype=value.dtype)
        for key, value in state_np.items()
        if hasattr(value, "shape")
    }
    action_buf = {
        key: np.empty((rollout_steps, *value.shape), dtype=value.dtype)
        for key, value in action_np.items()
        if hasattr(value, "shape")
    }
    player = np.empty((batch_size,), dtype=np.int32)
    result = np.empty((batch_size,), dtype=np.int32)
    rewards = np.empty((rollout_steps, batch_size), dtype=np.float32)
    dones = np.empty((rollout_steps, batch_size), dtype=np.float32)
    actions_buf = np.empty((rollout_steps, batch_size), dtype=np.int64)
    logprobs = np.empty((rollout_steps, batch_size), dtype=np.float32)
    values = np.empty((rollout_steps, batch_size), dtype=np.float32)

    model = PtcgPolicyValueNet(card_dim=card_dim, hidden_dim=hidden_dim).to(torch_device)
    model.train()
    fused_adam = False
    adam_kwargs: dict[str, Any] = {"lr": lr}
    if torch_device.type == "cuda":
        adam_kwargs["fused"] = True
        fused_adam = True
    try:
        optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
    except TypeError:
        adam_kwargs.pop("fused", None)
        fused_adam = False
    optimizer = torch.optim.Adam(model.parameters(), **adam_kwargs)
    forward = model.forward_tensors
    effective_compile = compile_model
    if compile_model:
        forward = torch.compile(forward, mode="reduce-overhead")
    autocast_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }.get(mixed_precision)
    use_autocast = torch_device.type == "cuda" and autocast_dtype is not None

    def torch_inputs(src_state: dict[str, np.ndarray], src_action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        return (
            torch.as_tensor(src_state["in_play"], device=torch_device),
            torch.as_tensor(src_state["zones"], device=torch_device),
            torch.as_tensor(src_state["player_counts"], device=torch_device),
            torch.as_tensor(src_state["player_status"], device=torch_device),
            torch.as_tensor(src_state["global"], device=torch_device),
            torch.as_tensor(src_action["options"], device=torch_device),
            torch.as_tensor(src_action["mask"], device=torch_device),
        )

    def sync() -> None:
        if torch_device.type == "cuda":
            torch.cuda.synchronize()

    def autocast_context():
        return (
            torch.autocast(device_type=torch_device.type, dtype=autocast_dtype)
            if use_autocast
            else nullcontext()
        )

    if compile_model:
        env.observe_ids_into(state_np, action_np, player, result)
        try:
            with torch.inference_mode(), autocast_context():
                forward(*torch_inputs(state_np, action_np))
            sync()
        except Exception:
            forward = model.forward_tensors
            effective_compile = False

    rollout_start = time.perf_counter()
    with torch.inference_mode():
        for t in range(rollout_steps):
            env.observe_ids_into(state_np, action_np, player, result)
            for key in state_buf:
                state_buf[key][t] = state_np[key]
            for key in action_buf:
                action_buf[key][t] = action_np[key]
            with autocast_context():
                out = forward(*torch_inputs(state_np, action_np))
            dist = torch.distributions.Categorical(logits=out["logits"])
            action_t = dist.sample()
            actions = action_t.cpu().numpy().astype(np.int32)
            _obs, reward, done, _mask, _player, _result = env.step(actions)
            rewards[t] = np.asarray(reward, dtype=np.float32)
            dones[t] = np.asarray(done, dtype=np.float32)
            actions_buf[t] = actions
            logprobs[t] = dist.log_prob(action_t).detach().float().cpu().numpy().astype(np.float32)
            values[t] = out["value"].detach().float().cpu().numpy().astype(np.float32)
    sync()
    rollout_seconds = time.perf_counter() - rollout_start

    env.observe_ids_into(state_np, action_np, player, result)
    with torch.inference_mode(), autocast_context():
        bootstrap = forward(*torch_inputs(state_np, action_np))["value"].detach().float().cpu().numpy()
    advantages, returns = _compute_gae(rewards, dones, values, bootstrap, gamma=gamma, lam=lam)

    total = rollout_steps * batch_size
    indices = _ppo_indices(total, minibatch_size, epochs, seed + 1009)
    flat_state = {key: torch.as_tensor(value.reshape((total, *value.shape[2:])), device=torch_device) for key, value in state_buf.items()}
    flat_action = {key: torch.as_tensor(value.reshape((total, *value.shape[2:])), device=torch_device) for key, value in action_buf.items()}
    flat_actions = torch.as_tensor(actions_buf.reshape(total), dtype=torch.long, device=torch_device)
    flat_old_logprobs = torch.as_tensor(logprobs.reshape(total), dtype=torch.float32, device=torch_device)
    flat_returns = torch.as_tensor(returns.reshape(total), dtype=torch.float32, device=torch_device)
    flat_advantages = torch.as_tensor(advantages.reshape(total), dtype=torch.float32, device=torch_device)

    usable = (total // minibatch_size) * minibatch_size
    if usable <= 0:
        raise ValueError("ppo_minibatch_size must be <= rollout_steps * batch_size")
    minibatches_per_epoch = usable // minibatch_size
    indices = _ppo_indices(total, minibatch_size, epochs, seed + 1009)

    def train_step(
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
        selected_actions,
        old_logprobs,
        returns_,
        advantages_,
    ):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            out = forward(
                in_play,
                zones,
                player_counts,
                player_status,
                global_,
                options,
                mask,
            )
        logits = out["logits"].float()
        value = out["value"].float()
        logp_all = F.log_softmax(logits, dim=-1)
        new_logprobs = logp_all.gather(1, selected_actions[:, None]).squeeze(1)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * logp_all).sum(dim=-1).mean()
        ratio = torch.exp(new_logprobs - old_logprobs)
        policy_loss = -torch.minimum(
            ratio * advantages_,
            torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages_,
        ).mean()
        value_loss = 0.5 * (value - returns_).pow(2).mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        loss.backward()
        optimizer.step()
        return torch.stack((
            policy_loss.detach(),
            value_loss.detach(),
            entropy.detach(),
            loss.detach(),
        ))

    update_step = torch.compile(train_step, mode="reduce-overhead") if effective_compile else train_step

    last_losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
    update_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_perm = torch.as_tensor(
            indices[epoch * minibatches_per_epoch:(epoch + 1) * minibatches_per_epoch].reshape(-1),
            dtype=torch.long,
            device=torch_device,
        )
        epoch_state = {key: value.index_select(0, epoch_perm) for key, value in flat_state.items()}
        epoch_action = {key: value.index_select(0, epoch_perm) for key, value in flat_action.items()}
        epoch_selected_actions = flat_actions.index_select(0, epoch_perm)
        epoch_old_logprobs = flat_old_logprobs.index_select(0, epoch_perm)
        epoch_returns = flat_returns.index_select(0, epoch_perm)
        epoch_advantages = flat_advantages.index_select(0, epoch_perm)
        for start_idx in range(0, usable, minibatch_size):
            stop_idx = start_idx + minibatch_size
            step_args = (
                epoch_state["in_play"][start_idx:stop_idx],
                epoch_state["zones"][start_idx:stop_idx],
                epoch_state["player_counts"][start_idx:stop_idx],
                epoch_state["player_status"][start_idx:stop_idx],
                epoch_state["global"][start_idx:stop_idx],
                epoch_action["options"][start_idx:stop_idx],
                epoch_action["mask"][start_idx:stop_idx],
                epoch_selected_actions[start_idx:stop_idx],
                epoch_old_logprobs[start_idx:stop_idx],
                epoch_returns[start_idx:stop_idx],
                epoch_advantages[start_idx:stop_idx],
            )
            try:
                losses = update_step(*step_args)
            except Exception:
                if not effective_compile:
                    raise
                update_step = train_step
                effective_compile = False
                losses = update_step(*step_args)
            last_losses = {
                "policy_loss": float(losses[0].detach().cpu()),
                "value_loss": float(losses[1].detach().cpu()),
                "entropy": float(losses[2].detach().cpu()),
                "total_loss": float(losses[3].detach().cpu()),
            }
    sync()
    update_seconds = time.perf_counter() - update_start
    warm_seconds = rollout_seconds + update_seconds
    update_samples = int(indices.size)
    metrics = {
        "rollout_s": rollout_seconds,
        "update_s": update_seconds,
        "rollout/s": (total / rollout_seconds) if rollout_seconds > 0 else 0.0,
        "update/s": (update_samples / update_seconds) if update_seconds > 0 else 0.0,
        "total/s": (total / warm_seconds) if warm_seconds > 0 else 0.0,
        "compiled": effective_compile,
        "fused_adam": fused_adam,
        "contiguous_mb": True,
        "mixed": mixed_precision if use_autocast else "off",
        **last_losses,
    }
    return total, batch_size, warm_seconds, metrics


def bench_native_ppo_jax(
    deck0: list[int],
    deck1: list[int],
    *,
    seed: int,
    batch_size: int,
    rollout_steps: int,
    minibatch_size: int,
    epochs: int,
    gamma: float,
    lam: float,
    clip: float,
    vf_coef: float,
    ent_coef: float,
    lr: float,
    card_dim: int,
    hidden_dim: int,
    matmul_precision: str | None,
    mixed_precision: str,
    full_scan_update: bool = False,
    epoch_scan_update: bool = True,
    diagnostic: str = "full",
    custom_embedding_backward: bool = False,
    compact_card_vocab: bool = True,
) -> tuple[int, int, float, dict[str, Any]]:
    _clear_native_modules()
    _prefer_native_paths()
    import jax
    import jax.numpy as jnp
    import ptcg_engine as E
    from ptcg.jax_nn import forward_policy_value, init_compact_params, init_params

    if not hasattr(E.VectorEnv, "observe_ids_into"):
        raise RuntimeError("VectorEnv.observe_ids_into is not available in this build")

    env = E.VectorEnv(deck0, deck1, batch_size, seed)
    state_np = env.state_ids()
    action_np = env.action_ids()
    state_buf = {
        key: np.empty((rollout_steps, *value.shape), dtype=value.dtype)
        for key, value in state_np.items()
        if hasattr(value, "shape")
    }
    action_buf = {
        key: np.empty((rollout_steps, *value.shape), dtype=value.dtype)
        for key, value in action_np.items()
        if hasattr(value, "shape")
    }
    player = np.empty((batch_size,), dtype=np.int32)
    result = np.empty((batch_size,), dtype=np.int32)
    rewards = np.empty((rollout_steps, batch_size), dtype=np.float32)
    dones = np.empty((rollout_steps, batch_size), dtype=np.float32)
    actions_buf = np.empty((rollout_steps, batch_size), dtype=np.int32)
    logprobs = np.empty((rollout_steps, batch_size), dtype=np.float32)
    values = np.empty((rollout_steps, batch_size), dtype=np.float32)

    if compact_card_vocab:
        params, card_id_remap = init_compact_params(
            jax.random.PRNGKey(seed),
            [*deck0, *deck1],
            card_dim=card_dim,
            hidden_dim=hidden_dim,
        )
        card_id_remap = jnp.asarray(card_id_remap, dtype=jnp.int32)
    else:
        params = init_params(jax.random.PRNGKey(seed), card_dim=card_dim, hidden_dim=hidden_dim)
        card_id_remap = None
    mixed_dtype = {
        "bf16": jnp.bfloat16,
        "fp16": jnp.float16,
    }.get(mixed_precision)
    if mixed_dtype is not None:
        params = jax.tree_util.tree_map(
            lambda value: value.astype(mixed_dtype) if jnp.issubdtype(value.dtype, jnp.floating) else value,
            params,
        )
    precision_ctx = (
        nullcontext()
        if not matmul_precision or matmul_precision == "default"
        else jax.default_matmul_precision(matmul_precision)
    )

    def adam_init(p):
        zeros = jax.tree_util.tree_map(
            lambda value: jnp.zeros(value.shape, dtype=jnp.float32),
            p,
        )
        return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}

    def adam_update(p, grads, opt_state):
        t = opt_state["t"] + 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        m = jax.tree_util.tree_map(lambda m_, g: b1 * m_ + (1.0 - b1) * g, opt_state["m"], grads)
        v = jax.tree_util.tree_map(lambda v_, g: b2 * v_ + (1.0 - b2) * (g * g), opt_state["v"], grads)
        step_size = lr * jnp.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)
        p = jax.tree_util.tree_map(
            lambda p_, m_, v_: (p_.astype(jnp.float32) - step_size * m_ / (jnp.sqrt(v_) + eps)).astype(p_.dtype),
            p,
            m,
            v,
        )
        return p, {"m": m, "v": v, "t": t}

    def forward_model(
        p,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
    ):
        return forward_policy_value(
            p,
            in_play,
            zones,
            player_counts,
            player_status,
            global_,
            options,
            mask,
            custom_embedding_backward=custom_embedding_backward,
            card_id_remap=card_id_remap,
        )

    infer = jax.jit(forward_model)

    @jax.jit
    def sample_policy(
        p,
        key,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
    ):
        key, sample_key = jax.random.split(key)
        out = forward_model(
            p,
            in_play,
            zones,
            player_counts,
            player_status,
            global_,
            options,
            mask,
        )
        logits = out["logits"].astype(jnp.float32)
        actions = jax.random.categorical(sample_key, logits, axis=-1).astype(jnp.int32)
        logp_all = jax.nn.log_softmax(logits, axis=-1)
        logprobs = jnp.take_along_axis(logp_all, actions[:, None], axis=1).squeeze(1)
        return key, actions, logprobs, out["value"].astype(jnp.float32)

    def ppo_loss(
        p,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
        actions,
        old_logprobs,
        advantages_,
        returns_,
    ):
        out = forward_model(
            p,
            in_play,
            zones,
            player_counts,
            player_status,
            global_,
            options,
            mask,
        )
        logits = out["logits"].astype(jnp.float32)
        value = out["value"].astype(jnp.float32)
        logp_all = jax.nn.log_softmax(logits, axis=-1)
        probs = jax.nn.softmax(logits, axis=-1)
        new_logp = jnp.take_along_axis(logp_all, actions[:, None], axis=1).squeeze(1)
        entropy = -jnp.sum(probs * logp_all, axis=-1).mean()
        ratio = jnp.exp(new_logp - old_logprobs)
        policy_loss = -jnp.minimum(
            ratio * advantages_,
            jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * advantages_,
        ).mean()
        value_loss = 0.5 * jnp.square(value - returns_).mean()
        total_loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
        metrics = jnp.array([policy_loss, value_loss, entropy, total_loss], dtype=jnp.float32)
        return total_loss, metrics

    frozen_embedding_keys = frozenset((
        "card_embedding",
        "role_gate",
        "attack_embedding",
        "kind_embedding",
        "type_embedding",
        "area_embedding",
    ))

    def freeze_embedding_params(p):
        return {
            key: jax.lax.stop_gradient(value) if key in frozen_embedding_keys else value
            for key, value in p.items()
        }

    @jax.jit
    def ppo_step(
        params,
        opt_state,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
        actions,
        old_logprobs,
        advantages_,
        returns_,
    ):
        (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            params,
            in_play,
            zones,
            player_counts,
            player_status,
            global_,
            options,
            mask,
            actions,
            old_logprobs,
            advantages_,
            returns_,
        )
        params, opt_state = adam_update(params, grads, opt_state)
        return params, opt_state, metrics

    @jax.jit
    def ppo_forward_only(
        params,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
        actions,
        old_logprobs,
        advantages_,
        returns_,
    ):
        _loss, metrics = ppo_loss(
            params,
            in_play,
            zones,
            player_counts,
            player_status,
            global_,
            options,
            mask,
            actions,
            old_logprobs,
            advantages_,
            returns_,
        )
        return metrics

    @jax.jit
    def ppo_backward_no_adam(
        params,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
        actions,
        old_logprobs,
        advantages_,
        returns_,
    ):
        (_loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
            params,
            in_play,
            zones,
            player_counts,
            player_status,
            global_,
            options,
            mask,
            actions,
            old_logprobs,
            advantages_,
            returns_,
        )
        grad_sum = jax.tree_util.tree_reduce(
            lambda acc, value: acc + jnp.sum(value.astype(jnp.float32) * value.astype(jnp.float32)),
            grads,
            initializer=jnp.array(0.0, dtype=jnp.float32),
        )
        return metrics.at[3].set(metrics[3] + grad_sum * jnp.array(1e-20, dtype=jnp.float32))

    @jax.jit
    def ppo_backward_heads_only(
        params,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        options,
        mask,
        actions,
        old_logprobs,
        advantages_,
        returns_,
    ):
        def loss_fn(p):
            return ppo_loss(
                freeze_embedding_params(p),
                in_play,
                zones,
                player_counts,
                player_status,
                global_,
                options,
                mask,
                actions,
                old_logprobs,
                advantages_,
                returns_,
            )

        (_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        grad_sum = jax.tree_util.tree_reduce(
            lambda acc, value: acc + jnp.sum(value.astype(jnp.float32) * value.astype(jnp.float32)),
            grads,
            initializer=jnp.array(0.0, dtype=jnp.float32),
        )
        return metrics.at[3].set(metrics[3] + grad_sum * jnp.array(1e-20, dtype=jnp.float32))

    @jax.jit
    def ppo_adam_only(params, opt_state, grads):
        params, opt_state = adam_update(params, grads, opt_state)
        leaf_sum = jax.tree_util.tree_reduce(
            lambda acc, value: acc + jnp.sum(value.astype(jnp.float32)),
            params,
            initializer=jnp.array(0.0, dtype=jnp.float32),
        )
        metrics = jnp.array([0.0, 0.0, 0.0, leaf_sum * jnp.array(1e-12, dtype=jnp.float32)], dtype=jnp.float32)
        return params, opt_state, metrics

    @jax.jit
    def ppo_epoch_update(params, opt_state, epoch_batch):
        def step(carry, mb):
            p, opt = carry
            (loss, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                p,
                mb["in_play"],
                mb["zones"],
                mb["player_counts"],
                mb["player_status"],
                mb["global"],
                mb["options"],
                mb["mask"],
                mb["actions"],
                mb["old_logprobs"],
                mb["advantages"],
                mb["returns"],
            )
            p, opt = adam_update(p, grads, opt)
            return (p, opt), metrics

        (params, opt_state), metrics = jax.lax.scan(step, (params, opt_state), epoch_batch)
        return params, opt_state, metrics[-1]

    @jax.jit
    def ppo_update(params, opt_state, batch, indices):
        def loss_fn(p, mb):
            return ppo_loss(
                p,
                batch["in_play"][mb],
                batch["zones"][mb],
                batch["player_counts"][mb],
                batch["player_status"][mb],
                batch["global"][mb],
                batch["options"][mb],
                batch["mask"][mb],
                batch["actions"][mb],
                batch["old_logprobs"][mb],
                batch["advantages"][mb],
                batch["returns"][mb],
            )

        def step(carry, mb):
            p, opt = carry
            (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, mb)
            p, opt = adam_update(p, grads, opt)
            return (p, opt), metrics

        (params, opt_state), metrics = jax.lax.scan(step, (params, opt_state), indices)
        return params, opt_state, metrics[-1]

    rollout_key = jax.random.PRNGKey(seed + 17)
    with precision_ctx:
        env.observe_ids_into(state_np, action_np, player, result)
        rollout_key, warm_actions, warm_logprobs, warm_values = sample_policy(
            params,
            rollout_key,
            jnp.asarray(state_np["in_play"]),
            jnp.asarray(state_np["zones"]),
            jnp.asarray(state_np["player_counts"]),
            jnp.asarray(state_np["player_status"]),
            jnp.asarray(state_np["global"]),
            jnp.asarray(action_np["options"]),
            jnp.asarray(action_np["mask"]),
        )
        jax.block_until_ready((rollout_key, warm_actions, warm_logprobs, warm_values))

        rollout_start = time.perf_counter()
        for t in range(rollout_steps):
            env.observe_ids_into(state_np, action_np, player, result)
            for key in state_buf:
                state_buf[key][t] = state_np[key]
            for key in action_buf:
                action_buf[key][t] = action_np[key]
            rollout_key, action_t, logprob_t, value_t = sample_policy(
                params,
                rollout_key,
                jnp.asarray(state_np["in_play"]),
                jnp.asarray(state_np["zones"]),
                jnp.asarray(state_np["player_counts"]),
                jnp.asarray(state_np["player_status"]),
                jnp.asarray(state_np["global"]),
                jnp.asarray(action_np["options"]),
                jnp.asarray(action_np["mask"]),
            )
            actions, logprob_np, value_np = jax.device_get((action_t, logprob_t, value_t))
            actions = np.asarray(actions, dtype=np.int32)
            _obs, reward, done, _mask, _player, _result = env.step(actions)
            rewards[t] = np.asarray(reward, dtype=np.float32)
            dones[t] = np.asarray(done, dtype=np.float32)
            actions_buf[t] = actions
            logprobs[t] = np.asarray(logprob_np, dtype=np.float32)
            values[t] = np.asarray(value_np, dtype=np.float32)
        jax.block_until_ready(rollout_key)
        rollout_seconds = time.perf_counter() - rollout_start

        env.observe_ids_into(state_np, action_np, player, result)
        bootstrap = infer(
            params,
            jnp.asarray(state_np["in_play"]),
            jnp.asarray(state_np["zones"]),
            jnp.asarray(state_np["player_counts"]),
            jnp.asarray(state_np["player_status"]),
            jnp.asarray(state_np["global"]),
            jnp.asarray(action_np["options"]),
            jnp.asarray(action_np["mask"]),
        )["value"]
        advantages, returns = _compute_gae(
            rewards,
            dones,
            values,
            np.asarray(jax.device_get(bootstrap), dtype=np.float32),
            gamma=gamma,
            lam=lam,
        )
        total = rollout_steps * batch_size
        indices = _ppo_indices(total, minibatch_size, epochs, seed + 1009)
        batch = {
            "in_play": jnp.asarray(state_buf["in_play"].reshape((total, *state_buf["in_play"].shape[2:]))),
            "zones": jnp.asarray(state_buf["zones"].reshape((total, *state_buf["zones"].shape[2:]))),
            "player_counts": jnp.asarray(state_buf["player_counts"].reshape((total, *state_buf["player_counts"].shape[2:]))),
            "player_status": jnp.asarray(state_buf["player_status"].reshape((total, *state_buf["player_status"].shape[2:]))),
            "global": jnp.asarray(state_buf["global"].reshape((total, *state_buf["global"].shape[2:]))),
            "options": jnp.asarray(action_buf["options"].reshape((total, *action_buf["options"].shape[2:]))),
            "mask": jnp.asarray(action_buf["mask"].reshape((total, *action_buf["mask"].shape[2:]))),
            "actions": jnp.asarray(actions_buf.reshape(total), dtype=jnp.int32),
            "old_logprobs": jnp.asarray(logprobs.reshape(total), dtype=jnp.float32),
            "advantages": jnp.asarray(advantages.reshape(total), dtype=jnp.float32),
            "returns": jnp.asarray(returns.reshape(total), dtype=jnp.float32),
        }
        opt_state = adam_init(params)
        update_start = time.perf_counter()
        if diagnostic == "adam_only":
            synthetic_grads = jax.tree_util.tree_map(
                lambda value: jnp.ones(value.shape, dtype=jnp.float32),
                params,
            )
            last = jnp.zeros((4,), dtype=jnp.float32)
            for _mb_np in indices:
                params, opt_state, last = ppo_adam_only(params, opt_state, synthetic_grads)
            update_mode = "diagnostic_adam_only"
        elif diagnostic == "forward_only":
            last = jnp.zeros((4,), dtype=jnp.float32)
            for mb_np in indices:
                mb = jnp.asarray(mb_np, dtype=jnp.int32)
                last = ppo_forward_only(
                    params,
                    batch["in_play"][mb],
                    batch["zones"][mb],
                    batch["player_counts"][mb],
                    batch["player_status"][mb],
                    batch["global"][mb],
                    batch["options"][mb],
                    batch["mask"][mb],
                    batch["actions"][mb],
                    batch["old_logprobs"][mb],
                    batch["advantages"][mb],
                    batch["returns"][mb],
                )
            update_mode = "diagnostic_forward_only"
        elif diagnostic == "backward_no_adam":
            last = jnp.zeros((4,), dtype=jnp.float32)
            for mb_np in indices:
                mb = jnp.asarray(mb_np, dtype=jnp.int32)
                last = ppo_backward_no_adam(
                    params,
                    batch["in_play"][mb],
                    batch["zones"][mb],
                    batch["player_counts"][mb],
                    batch["player_status"][mb],
                    batch["global"][mb],
                    batch["options"][mb],
                    batch["mask"][mb],
                    batch["actions"][mb],
                    batch["old_logprobs"][mb],
                    batch["advantages"][mb],
                    batch["returns"][mb],
                )
            update_mode = "diagnostic_backward_no_adam"
        elif diagnostic == "backward_heads_only":
            last = jnp.zeros((4,), dtype=jnp.float32)
            for mb_np in indices:
                mb = jnp.asarray(mb_np, dtype=jnp.int32)
                last = ppo_backward_heads_only(
                    params,
                    batch["in_play"][mb],
                    batch["zones"][mb],
                    batch["player_counts"][mb],
                    batch["player_status"][mb],
                    batch["global"][mb],
                    batch["options"][mb],
                    batch["mask"][mb],
                    batch["actions"][mb],
                    batch["old_logprobs"][mb],
                    batch["advantages"][mb],
                    batch["returns"][mb],
                )
            update_mode = "diagnostic_backward_heads_only"
        elif full_scan_update:
            params, opt_state, last = ppo_update(params, opt_state, batch, jnp.asarray(indices, dtype=jnp.int32))
            update_mode = "fullscan"
        elif epoch_scan_update:
            usable = (total // minibatch_size) * minibatch_size
            minibatches_per_epoch = usable // minibatch_size
            last = jnp.zeros((4,), dtype=jnp.float32)
            for epoch in range(epochs):
                epoch_perm = indices[
                    epoch * minibatches_per_epoch:(epoch + 1) * minibatches_per_epoch
                ].reshape(-1)
                epoch_perm_jax = jnp.asarray(epoch_perm, dtype=jnp.int32)
                epoch_batch = {
                    key: value[epoch_perm_jax].reshape(
                        (minibatches_per_epoch, minibatch_size, *value.shape[1:])
                    )
                    for key, value in batch.items()
                }
                params, opt_state, last = ppo_epoch_update(params, opt_state, epoch_batch)
            update_mode = "epochscan"
        else:
            last = jnp.zeros((4,), dtype=jnp.float32)
            for mb_np in indices:
                mb = jnp.asarray(mb_np, dtype=jnp.int32)
                params, opt_state, last = ppo_step(
                    params,
                    opt_state,
                    batch["in_play"][mb],
                    batch["zones"][mb],
                    batch["player_counts"][mb],
                    batch["player_status"][mb],
                    batch["global"][mb],
                    batch["options"][mb],
                    batch["mask"][mb],
                    batch["actions"][mb],
                    batch["old_logprobs"][mb],
                    batch["advantages"][mb],
                    batch["returns"][mb],
                )
            update_mode = "stepjit"
        jax.block_until_ready((params, opt_state, last))
        update_seconds = time.perf_counter() - update_start

    warm_seconds = rollout_seconds + update_seconds
    last_np = np.asarray(jax.device_get(last), dtype=np.float32)
    update_samples = int(indices.size)
    metrics = {
        "rollout_s": rollout_seconds,
        "update_s": update_seconds,
        "rollout/s": (total / rollout_seconds) if rollout_seconds > 0 else 0.0,
        "update/s": (update_samples / update_seconds) if update_seconds > 0 else 0.0,
        "total/s": (total / warm_seconds) if warm_seconds > 0 else 0.0,
        "compiled": True,
        "update_mode": update_mode,
        "contiguous_mb": epoch_scan_update and not full_scan_update,
        "custom_emb_bwd": custom_embedding_backward,
        "compact_cards": compact_card_vocab,
        "mixed": mixed_precision if mixed_dtype is not None else "off",
        "policy_loss": float(last_np[0]),
        "value_loss": float(last_np[1]),
        "entropy": float(last_np[2]),
        "total_loss": float(last_np[3]),
    }
    return total, batch_size, warm_seconds, metrics


def _feature_observer(name: str) -> Callable[[Any, Any], Any]:
    def observe(E, state):
        fn = getattr(E, name)
        return fn(state)

    return observe


def run(args: argparse.Namespace) -> list[BenchResult]:
    deck0 = list(ALL_DECKS[args.deck])
    deck1 = list(ALL_DECKS[args.deck1 or args.deck])

    results: list[BenchResult] = []
    only = args.only.lower() if args.only else None

    def should_run(name: str) -> bool:
        return only is None or only in name.lower()

    def add(name: str, fn: Callable[[], tuple[Any, ...]]) -> None:
        if only is None and name.startswith("diagnostic_"):
            return
        if should_run(name):
            result = _timed(name, fn)
            results.append(result)
            if getattr(args, "stream", False):
                _print_result(result)

    if args.include_reference:
        add(
            "reference_cg_format",
            lambda: bench_cg_game(
                deck0, deck1, steps=args.steps, seed=args.seed,
                backend="reference", selector=args.selector,
            ),
        )

    for name, fn in [
        (
            "native_cg_format",
            lambda: bench_cg_game(
                deck0, deck1, steps=args.steps, seed=args.seed,
                backend="native", selector=args.selector,
            ),
        ),
        (
            "native_rl_step_only",
            lambda: bench_native_rl(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector,
            ),
        ),
        (
            "native_rl_compact_obs",
            lambda: bench_native_rl(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector, observe=_feature_observer("rl_encode_obs"),
            ),
        ),
        (
            "native_rl_action_features",
            lambda: bench_native_rl(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector,
                observe=_feature_observer("rl_action_features"),
            ),
        ),
        (
            "native_rl_action_ids",
            lambda: bench_native_rl(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector,
                observe=_feature_observer("rl_action_ids"),
            ),
        ),
        (
            "native_rl_card_features",
            lambda: bench_native_rl(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector, observe=_feature_observer("rl_card_features"),
            ),
        ),
        (
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
        (
            "native_vectorized_action_ids",
            lambda: bench_native_vectorized_action_ids(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
            ),
        ),
        (
            "native_vectorized_legacy_all_features",
            lambda: bench_native_vectorized_legacy_all_features(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
            ),
        ),
        (
            "native_nn_adapter",
            lambda: bench_native_nn_adapter(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
            ),
        ),
        (
            "native_nn_forward_static",
            lambda: bench_native_nn_forward_static(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
            ),
        ),
        (
            "native_vectorized_nn_adapter",
            lambda: bench_native_vectorized_nn_adapter(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
            ),
        ),
        (
            "native_vectorized_nn_reuse",
            lambda: bench_native_vectorized_nn_adapter_reuse(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                device=args.nn_device,
                matmul_precision=args.torch_matmul_precision,
            ),
        ),
        (
            "native_vectorized_nn_compile",
            lambda: bench_native_vectorized_nn_adapter_reuse(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                compile_model=True,
                device=args.nn_device,
                matmul_precision=args.torch_matmul_precision,
            ),
        ),
        (
            "native_vectorized_jax_eager_reuse",
            lambda: bench_native_vectorized_jax_reuse(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                compile_model=False,
                matmul_precision=args.jax_matmul_precision,
            ),
        ),
        (
            "native_vectorized_jax_jit_reuse",
            lambda: bench_native_vectorized_jax_reuse(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                compile_model=True,
                matmul_precision=args.jax_matmul_precision,
            ),
        ),
        (
            "native_vectorized_jax_policy_eager_reuse",
            lambda: bench_native_vectorized_jax_reuse(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                compile_model=False,
                policy_only=True,
                matmul_precision=args.jax_matmul_precision,
            ),
        ),
        (
            "native_vectorized_jax_policy_jit_reuse",
            lambda: bench_native_vectorized_jax_reuse(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                compile_model=True,
                policy_only=True,
                matmul_precision=args.jax_matmul_precision,
            ),
        ),
        (
            "native_vectorized_jax_reuse",
            lambda: bench_native_vectorized_jax_reuse(
                deck0,
                deck1,
                steps=args.steps,
                seed=args.seed,
                batch_size=args.batch_size,
                selector=args.selector,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
            ),
        ),
        (
            "native_ppo_torch",
            lambda: bench_native_ppo_torch(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                device=args.nn_device,
                matmul_precision=args.torch_matmul_precision,
                compile_model=args.ppo_torch_compile,
                mixed_precision=args.nn_mixed_precision,
            ),
        ),
        (
            "native_ppo_jax",
            lambda: bench_native_ppo_jax(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
                mixed_precision=args.nn_mixed_precision,
                full_scan_update=args.ppo_jax_update_mode == "fullscan",
                epoch_scan_update=args.ppo_jax_update_mode == "epochscan",
                custom_embedding_backward=args.jax_custom_embedding_backward,
                compact_card_vocab=args.ppo_jax_compact_cards,
            ),
        ),
        (
            "diagnostic_jax_ppo_forward_only",
            lambda: bench_native_ppo_jax(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
                mixed_precision=args.nn_mixed_precision,
                diagnostic="forward_only",
                compact_card_vocab=args.ppo_jax_compact_cards,
            ),
        ),
        (
            "diagnostic_jax_ppo_backward_no_adam",
            lambda: bench_native_ppo_jax(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
                mixed_precision=args.nn_mixed_precision,
                diagnostic="backward_no_adam",
                compact_card_vocab=args.ppo_jax_compact_cards,
            ),
        ),
        (
            "diagnostic_jax_ppo_backward_heads_only",
            lambda: bench_native_ppo_jax(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
                mixed_precision=args.nn_mixed_precision,
                diagnostic="backward_heads_only",
                compact_card_vocab=args.ppo_jax_compact_cards,
            ),
        ),
        (
            "diagnostic_jax_ppo_backward_custom_embedding",
            lambda: bench_native_ppo_jax(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
                mixed_precision=args.nn_mixed_precision,
                diagnostic="backward_no_adam",
                custom_embedding_backward=True,
                compact_card_vocab=args.ppo_jax_compact_cards,
            ),
        ),
        (
            "diagnostic_jax_ppo_adam_only",
            lambda: bench_native_ppo_jax(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
                mixed_precision=args.nn_mixed_precision,
                diagnostic="adam_only",
                compact_card_vocab=args.ppo_jax_compact_cards,
            ),
        ),
    ]:
        add(name, fn)

    if only == "experimental_jax_fullscan_ppo":
        add(
            "experimental_jax_fullscan_ppo",
            lambda: bench_native_ppo_jax(
                deck0,
                deck1,
                seed=args.seed,
                batch_size=args.batch_size,
                rollout_steps=args.ppo_rollout_steps,
                minibatch_size=args.ppo_minibatch_size,
                epochs=args.ppo_epochs,
                gamma=args.ppo_gamma,
                lam=args.ppo_lambda,
                clip=args.ppo_clip,
                vf_coef=args.ppo_vf_coef,
                ent_coef=args.ppo_ent_coef,
                lr=args.ppo_lr,
                card_dim=args.nn_card_dim,
                hidden_dim=args.nn_hidden_dim,
                matmul_precision=args.jax_matmul_precision,
                mixed_precision=args.nn_mixed_precision,
                full_scan_update=True,
                compact_card_vocab=args.ppo_jax_compact_cards,
            ),
        )

    if should_run("native_rl_state_observation") and (
        args.include_unavailable or _native_has("rl_state_observation")
    ):
        add(
            "native_rl_state_observation",
            lambda: bench_native_rl(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector,
                observe=_feature_observer("rl_state_observation"),
            ),
        )
    if should_run("native_rl_state_ids") and (
        args.include_unavailable or _native_has("rl_state_ids")
    ):
        add(
            "native_rl_state_ids",
            lambda: bench_native_rl(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector, observe=_feature_observer("rl_state_ids"),
            ),
        )
    if should_run("native_descriptor_direct") and (
        args.include_descriptor or args.include_unavailable or _native_has("step_action")
    ):
        add(
            "native_descriptor_direct",
            lambda: bench_native_descriptor_direct(
                deck0, deck1, steps=args.steps, seed=args.seed,
                selector=args.selector,
            ),
        )
    return results


def _print_table(results: list[BenchResult]) -> None:
    _print_header()
    for result in results:
        _print_result(result)


def _print_header() -> None:
    print(
        f"{'name':34} {'steps':>10} {'seconds':>10} "
        f"{'overall/s':>12} {'warm/s':>12} status",
        flush=True,
    )
    print("-" * 96, flush=True)


def _print_result(result: BenchResult) -> None:
    status = f"SKIP {result.reason}" if result.skipped else "ok"
    if result.metrics and not result.skipped:
        metric_text = " ".join(
            f"{key}={value:.4g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in result.metrics.items()
        )
        status = f"{status} {metric_text}"
    warm_rate = (
        f"{result.warm_steps_per_second:12.1f}"
        if result.warm_steps_per_second is not None
        else f"{'-':>12}"
    )
    print(
        f"{result.name:34} {result.steps:10d} {result.seconds:10.4f} "
        f"{result.steps_per_second:12.1f} {warm_rate} {status}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark cg/native engine paths")
    parser.add_argument("--deck", choices=sorted(ALL_DECKS), default="mega_lucario")
    parser.add_argument("--deck1", choices=sorted(ALL_DECKS), default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--nn-card-dim", type=int, default=128,
                        help="Card embedding dimension for ptcg.nn benchmark rows")
    parser.add_argument("--nn-hidden-dim", type=int, default=256,
                        help="Hidden dimension for ptcg.nn benchmark rows")
    parser.add_argument("--nn-device", choices=["cpu", "cuda", "auto"], default="cpu",
                        help="Device for torch ptcg.nn benchmark rows")
    parser.add_argument("--torch-matmul-precision",
                        choices=["highest", "high", "medium"], default="high",
                        help="Optional torch.set_float32_matmul_precision value")
    parser.add_argument("--jax-matmul-precision",
                        choices=["default", "high", "highest", "tensorfloat32"],
                        default="high",
                        help="Optional jax.default_matmul_precision value")
    parser.add_argument("--nn-mixed-precision",
                        choices=["bf16", "fp16", "off"], default="bf16",
                        help="Mixed precision mode for PPO neural network benchmarks")
    parser.add_argument("--ppo-torch-compile", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Compile the Torch PPO forward path with torch.compile")
    parser.add_argument("--ppo-jax-update-mode",
                        choices=["epochscan", "stepjit", "fullscan"],
                        default="epochscan",
                        help="JAX PPO update compilation strategy")
    parser.add_argument("--jax-custom-embedding-backward",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Use experimental custom VJP for JAX embedding gathers")
    parser.add_argument("--ppo-jax-compact-cards",
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Use a deck-local compact card vocabulary for JAX PPO")
    parser.add_argument("--ppo-rollout-steps", type=int, default=128,
                        help="Vectorized PPO rollout length T")
    parser.add_argument("--ppo-minibatch-size", type=int, default=8192,
                        help="Flattened PPO minibatch size")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO update epochs over the rollout")
    parser.add_argument("--ppo-gamma", type=float, default=0.99,
                        help="PPO discount factor")
    parser.add_argument("--ppo-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--ppo-clip", type=float, default=0.2,
                        help="PPO clipping epsilon")
    parser.add_argument("--ppo-vf-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument("--ppo-ent-coef", type=float, default=0.01,
                        help="Entropy coefficient")
    parser.add_argument("--ppo-lr", type=float, default=3e-4,
                        help="PPO Adam learning rate")
    parser.add_argument("--selector", choices=["first", "random"], default="first",
                        help="Action selector used during the benchmark")
    parser.add_argument("--only", default=None,
                        help="Run only benchmark rows whose name contains this text")
    parser.add_argument("--json", action="store_true", help="Print JSON results")
    parser.add_argument("--include-reference", action="store_true",
                        help="Also benchmark the original cg package if available")
    parser.add_argument("--include-descriptor", action="store_true",
                        help="Also benchmark the legacy descriptor step_action path")
    parser.add_argument("--include-unavailable", action="store_true",
                        help="Show rows for APIs unavailable in the loaded extension")
    parser.add_argument("--strict-skips", action="store_true",
                        help="Exit nonzero if any non-reference benchmark row is skipped")
    args = parser.parse_args(argv)

    args.stream = not args.json
    if args.stream:
        _print_header()
    results = run(args)
    native_pyd = _loaded_native_pyd()
    if args.json:
        payload = {
            "native_pyd": native_pyd,
            "results": [asdict(result) for result in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        if not results:
            print(f"no benchmark rows matched --only={args.only!r}", flush=True)
        print(f"native pyd: {native_pyd or 'NOT LOADED'}")
    if args.strict_skips:
        return 0 if all(not result.skipped for result in results[1:]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
