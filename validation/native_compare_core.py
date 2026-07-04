"""Shared native-vs-cg observation comparison helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from validation.oracle.canonical import canonical_options, canonical_state, diff


@dataclass
class ComparisonResult:
    ok: bool
    label: str
    state_diffs: list[tuple[str, Any, Any]] = field(default_factory=list)
    select_diffs: list[tuple[str, Any, Any]] = field(default_factory=list)
    option_diffs: list[tuple[str, Any, Any]] = field(default_factory=list)
    log_diffs: list[tuple[str, Any, Any]] = field(default_factory=list)

    def message(self, *, limit: int = 12) -> str:
        lines = [f"{self.label}: native/cg divergence"]
        for name, diffs in (
            ("state", self.state_diffs),
            ("select", self.select_diffs),
            ("options", self.option_diffs),
            ("logs", self.log_diffs),
        ):
            if not diffs:
                continue
            lines.append(f"{name} diffs:")
            for path, ref, native in diffs[:limit]:
                lines.append(f"  {path}: cg={ref!r} native={native!r}")
            if len(diffs) > limit:
                lines.append(f"  ... {len(diffs) - limit} more")
        return "\n".join(lines)


class NativeComparisonError(AssertionError):
    """Raised when native and reference cg observations diverge."""

    def __init__(self, result: ComparisonResult, trace: list[list[int]] | None = None):
        self.result = result
        self.trace = list(trace or [])
        trace_line = f"\naction_trace={self.trace}" if self.trace else ""
        super().__init__(result.message() + trace_line)


def _select_shape(obs: dict[str, Any]) -> dict[str, Any] | None:
    select = obs.get("select")
    if select is None:
        return None
    return {
        "type": select.get("type"),
        "context": select.get("context"),
        "minCount": select.get("minCount"),
        "maxCount": select.get("maxCount"),
        "remainDamageCounter": select.get("remainDamageCounter", 0),
        "remainEnergyCost": select.get("remainEnergyCost", 0),
        "deckCount": None if select.get("deck") is None else len(select.get("deck") or []),
        "optionCount": len(select.get("option") or []),
    }


def _logs(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in log.items() if k != "serial"}
        for log in (obs.get("logs") or [])
    ]


def compare_observations(
    label: str,
    cg_obs: dict[str, Any],
    native_obs: dict[str, Any],
    *,
    compare_logs: bool = False,
    ordered_options: bool = True,
) -> ComparisonResult:
    cg_state = canonical_state(cg_obs.get("current"))
    native_state = canonical_state(native_obs.get("current"))
    state_diffs = diff(cg_state, native_state)

    cg_shape = _select_shape(cg_obs)
    native_shape = _select_shape(native_obs)
    select_diffs = diff(cg_shape, native_shape)

    cg_options = canonical_options(cg_obs.get("current"), cg_obs.get("select"))
    native_options = canonical_options(native_obs.get("current"), native_obs.get("select"))
    if not ordered_options:
        cg_options = sorted(cg_options)
        native_options = sorted(native_options)
    option_diffs = diff(cg_options, native_options)

    log_diffs: list[tuple[str, Any, Any]] = []
    if compare_logs:
        log_diffs = diff(_logs(cg_obs), _logs(native_obs))

    return ComparisonResult(
        ok=not (state_diffs or select_diffs or option_diffs or log_diffs),
        label=label,
        state_diffs=state_diffs,
        select_diffs=select_diffs,
        option_diffs=option_diffs,
        log_diffs=log_diffs,
    )


def dump_divergence_payload(
    path: str,
    *,
    label: str,
    deck0: list[int],
    deck1: list[int],
    trace: list[list[int]],
    cg_obs: dict[str, Any],
    native_obs: dict[str, Any],
    native_debug: Any = None,
) -> None:
    payload = {
        "label": label,
        "deck0": deck0,
        "deck1": deck1,
        "trace": trace,
        "cg": cg_obs,
        "native": native_obs,
        "native_debug": native_debug,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
