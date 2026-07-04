"""Small install check for Kaggle notebooks and local environments."""
from __future__ import annotations

import os


def main() -> int:
    os.environ.setdefault("PTCG_BACKEND", "native")

    import ptcg_engine as engine
    from cg.game import battle_finish, battle_select, battle_start
    from validation.decks import MEGA_LUCARIO

    print("ptcg_engine:", engine.version())
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
