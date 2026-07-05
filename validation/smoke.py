"""Small install check for Kaggle notebooks and local environments."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _import_engine():
    root = Path(__file__).resolve().parents[1]
    source_checkout = (root / "engine" / "CMakeLists.txt").exists()
    local_exts: list[Path] = []
    build_dirs = (
        root / "engine",
        root / "engine" / "build" / "Release",
        root / "engine" / "build",
    )
    for build_dir in build_dirs:
        local_exts.extend(build_dir.glob("ptcg_engine*.pyd"))
        local_exts.extend(build_dir.glob("ptcg_engine*.so"))

    if source_checkout and not local_exts:
        raise RuntimeError(
            "ptcg_engine native extension is not built in this checkout. "
            "On Windows run '.\\engine\\build.ps1' from the repository root; "
            "on Linux run 'cmake -S engine -B engine/build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=\"$(python -m pybind11 --cmakedir)\"' "
            "then 'cmake --build engine/build --parallel'."
        )

    for build_dir in build_dirs:
        if build_dir.is_dir():
            sys.path.insert(0, str(build_dir))

    try:
        import ptcg_engine as engine
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ptcg_engine native extension is not built. "
            "On Windows run '.\\engine\\build.ps1' from the repository root; "
            "on Linux run 'cmake -S engine -B engine/build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=\"$(python -m pybind11 --cmakedir)\"' "
            "then 'cmake --build engine/build --parallel'."
        ) from exc
    return engine


def main() -> int:
    os.environ.setdefault("PTCG_BACKEND", "native")

    engine = _import_engine()
    from ptcg.cg.game import battle_finish, battle_select, battle_start
    from validation.decks import MEGA_LUCARIO

    print("ptcg_engine:", engine.version(), getattr(engine, "__file__", "<built-in>"))
    obs, start = battle_start(MEGA_LUCARIO, MEGA_LUCARIO)
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorType={start.errorType}")
    print("start context:", obs["select"]["context"])
    obs = battle_select([0])
    print("next turn/context:", obs["current"]["turn"], obs["select"]["context"])
    battle_finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
