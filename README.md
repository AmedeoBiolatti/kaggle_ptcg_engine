# Kaggle PTCG Native Engine

Native Pokemon TCG engine with a Python compatibility layer for the Kaggle
`ptcg.cg.game` and `ptcg.cg.api` interfaces.

The goal is to make the native engine usable as a drop-in backend for local
experiments, validation, and performance benchmarking while keeping the public
API close to the competition package.

## Competition Use And Rights Notice

This repository is provided only for use in connection with the relevant Kaggle
Pokemon Trading Card Game competition. It is not an official product and is not
affiliated with, endorsed by, sponsored by, or approved by Kaggle, the
competition organizers, The Pokemon Company, Nintendo, Creatures Inc., GAME
FREAK inc., or any other Pokemon rights holder.

Pokemon, Pokemon Trading Card Game, card names, card text, game rules,
trademarks, and related intellectual property belong to their respective owners.
See `LICENSE` for the full competition-use notice.

## Repository Layout

- `engine/`: C++ engine core and `pybind11` extension.
- `ptcg/cg/`: Python compatibility modules for `ptcg.cg.game`
  and `ptcg.cg.api`.
- `validation/`: deck fixtures, random deck generation, parity checks, and
  shadow-mode helpers.
- `benchmarks/`: speed comparison scripts for `cg` format, direct native calls,
  and vectorized stepping.
- `notebooks/`: minimal examples for the compatibility API and vectorized env.
- `data/`: card metadata used by the native compatibility layer.
- `tests/`: focused API and compatibility tests.

## Requirements

- Python 3.11 or 3.12
- CMake 3.18+
- A C++20 compiler
- `pybind11`
- `numpy`

On Windows, use a Visual Studio C++ toolchain. On Linux, use a recent GCC or
Clang toolchain.

## Install And Build

For Kaggle notebooks, prefer a prebuilt wheel distributed through a Kaggle
Dataset:

```python
!pip install /kaggle/input/kaggle-ptcg-native-engine/wheels/*.whl
!python -m validation.smoke
```

See `README_KAGGLE.md` for the full notebook install flow.

For local development, install the Python package in editable mode. If you are
offline or pip build isolation cannot reach PyPI, use `--no-build-isolation`
after installing the requirements in your current environment.

```bash
python -m pip install -e ".[test]"
# offline/local env variant:
python -m pip install -e ".[test]" --no-build-isolation
```

Then build the native extension.

```bash
python -m pip install cmake ninja pybind11
```

On Windows with Visual Studio Build Tools installed:

```powershell
.\engine\build.ps1
```

On Linux:

```bash
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build engine/build --parallel
```

The Python modules search `engine/`, `engine/build/Release`, and `engine/build`
for the compiled extension.

To build a wheel locally:

```bash
python -m pip install -U build cmake ninja pybind11
python -m build --wheel
```

Upload the produced `dist/*.whl` to a Kaggle Dataset for notebook use.

## Use As `ptcg.cg.game`

Set `PTCG_BACKEND=native` before importing or calling `ptcg.cg.game`:

```python
import os

os.environ["PTCG_BACKEND"] = "native"

from ptcg.cg.game import battle_start, battle_select, battle_finish
from validation.decks import MEGA_LUCARIO

obs, start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
if obs is None:
    raise RuntimeError(f"battle_start failed: {start.errorType}")

obs = battle_select([0])
battle_finish()
```

## Vectorized Native Env

The native extension also exposes `ptcg_engine.VectorEnv`, a generic vectorized
environment. It does not hide an opponent policy: each row reports the current
player, and the caller supplies the action for that player.

See `notebooks/native_vector_env_minimal.ipynb` for a compact example.

## Validation

Run native-only compatibility tests:

```bash
python -m unittest tests.test_native_cg_compat
```

Run random branch parity against a reference `cg` shared library:

```bash
python -m validation.native_compare --deck mega_lucario --games 10 --seed 1
```

Runtime shadow mode is available with `PTCG_BACKEND=shadow`. It drives setup
through the reference engine, bootstraps native from the first post-setup public
state, then advances both engines in lockstep and raises if public observations
diverge.

The reference `cg.dll` or `libcg.so` is optional and is not included. Set
`PTCG_REFERENCE_LIB` to its full path, or place it in a local `cg/` folder, only
when you want to run reference parity or shadow validation.

## Benchmark

```bash
python benchmarks/bench_engines.py --steps 2000 --batch-size 256
```

The benchmark reports:

- `reference_cg_format`: reference `cg.game`, when a reference shared library is
  available.
- `native_cg_format`: `ptcg.cg.game` with `PTCG_BACKEND=native`.
- `native_direct`: direct `ptcg_engine` state/action stepping.
- `native_vectorized`: batched stepping through `ptcg_engine.VectorEnv`.

If the reference shared library is absent, the reference row is skipped and the
native rows still run.

## Useful Environment Variables

- `PTCG_BACKEND=native`: route `ptcg.cg.game` calls to the native engine.
- `PTCG_BACKEND=shadow`: run reference and native in lockstep.
- `PTCG_NATIVE_SEED=<int>`: seed native free-running games.
- `PTCG_NATIVE_LAZY_SEARCH=1`: use faster in-process `search_begin_input`
  handles that are valid only until the next battle mutation.
- `PTCG_NATIVE_PORTABLE_SEARCH=1`: emit larger portable search payloads.

## Known Limits

This project aims for API and gameplay parity, but independent validation is
still recommended for new cards, effects, and hidden-state search behavior. Use
`validation.native_compare` and shadow mode when changing engine rules.
