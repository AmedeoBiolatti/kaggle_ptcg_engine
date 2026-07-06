from __future__ import annotations

from collections.abc import Sequence
from typing import Any

try:
    import jax
    import jax.numpy as jnp
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without jax.
    raise ImportError(
        "ptcg.jax_nn requires JAX. Install jax to use the JAX neural-network "
        "adapter; the engine itself does not depend on JAX."
    ) from exc


DEFAULT_MAX_CARD_ID = 1300
DEFAULT_CARD_OVERFLOW_BUCKETS = 0
INPLAY_SCALAR_WIDTH = 28


def _normal(key: Any, shape: tuple[int, ...], scale: float = 0.02) -> Any:
    return scale * jax.random.normal(key, shape, dtype=jnp.float32)


def init_params(
    key: Any,
    *,
    max_card_id: int = DEFAULT_MAX_CARD_ID,
    card_dim: int = 128,
    hidden_dim: int = 256,
    max_attack_id: int = 4096,
    roles: int = 10,
    card_overflow_buckets: int = DEFAULT_CARD_OVERFLOW_BUCKETS,
) -> dict[str, Any]:
    keys = jax.random.split(key, 22)
    overflow_buckets = max(0, int(card_overflow_buckets))
    card_embedding = _normal(keys[0], (max_card_id + 2 + overflow_buckets, card_dim))
    card_embedding = card_embedding.at[0].set(0.0)
    return {
        "card_embedding": card_embedding,
        "role_gate": 1.0 + _normal(keys[1], (roles, card_dim), scale=0.01),
        "inplay_scalar_w1": _normal(keys[14], (INPLAY_SCALAR_WIDTH, hidden_dim)),
        "inplay_scalar_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "inplay_scalar_w2": _normal(keys[15], (hidden_dim, card_dim)),
        "inplay_scalar_b2": jnp.zeros((card_dim,), dtype=jnp.float32),
        "state_scalar_w1": _normal(keys[2], (52, hidden_dim)),
        "state_scalar_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "state_scalar_w2": _normal(keys[3], (hidden_dim, card_dim)),
        "state_scalar_b2": jnp.zeros((card_dim,), dtype=jnp.float32),
        "state_out_scale": jnp.ones((card_dim,), dtype=jnp.float32),
        "state_out_bias": jnp.zeros((card_dim,), dtype=jnp.float32),
        "state_out_w": _normal(keys[4], (card_dim, card_dim)),
        "state_out_b": jnp.zeros((card_dim,), dtype=jnp.float32),
        "attack_embedding": _normal(keys[5], (max_attack_id + 1, card_dim)),
        "kind_embedding": _normal(keys[6], (32, card_dim)),
        "type_embedding": _normal(keys[7], (32, card_dim)),
        "area_embedding": _normal(keys[8], (16, card_dim)),
        "action_scalar_w": _normal(keys[9], (8, card_dim)),
        "action_scalar_b": jnp.zeros((card_dim,), dtype=jnp.float32),
        "action_out_scale": jnp.ones((card_dim,), dtype=jnp.float32),
        "action_out_bias": jnp.zeros((card_dim,), dtype=jnp.float32),
        "action_out_w": _normal(keys[10], (card_dim, card_dim)),
        "action_out_b": jnp.zeros((card_dim,), dtype=jnp.float32),
        "policy_state_w": _normal(keys[11], (card_dim, card_dim)),
        "policy_state_b": jnp.zeros((card_dim,), dtype=jnp.float32),
        "value_w1": _normal(keys[12], (card_dim, hidden_dim)),
        "value_b1": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        "value_w2": _normal(keys[13], (hidden_dim, 1)),
        "value_b2": jnp.zeros((1,), dtype=jnp.float32),
    }


def build_card_id_remap(card_ids: Sequence[int]) -> Any:
    distinct = sorted({int(card_id) for card_id in card_ids if int(card_id) > 0})
    max_card_id = max(distinct, default=0)
    remap = np.ones((max_card_id + 1,), dtype=np.int32)
    remap[0] = 0
    for compact_id, card_id in enumerate(distinct, start=2):
        remap[card_id] = compact_id
    return remap


def init_compact_params(
    key: Any,
    card_ids: Sequence[int],
    *,
    card_dim: int = 128,
    hidden_dim: int = 256,
    max_attack_id: int = 4096,
    roles: int = 10,
) -> tuple[dict[str, Any], Any]:
    remap = build_card_id_remap(card_ids)
    params = init_params(
        key,
        max_card_id=max(0, int(remap.max()) - 1),
        card_dim=card_dim,
        hidden_dim=hidden_dim,
        max_attack_id=max_attack_id,
        roles=roles,
        card_overflow_buckets=0,
    )
    return params, remap


def _card_token_bounds(params: dict[str, Any], overflow_buckets: int) -> tuple[int, int]:
    table_size = int(params["card_embedding"].shape[0])
    overflow_buckets = max(0, int(overflow_buckets))
    if overflow_buckets > 0 and table_size > overflow_buckets + 2:
        return table_size - overflow_buckets - 2, overflow_buckets
    return table_size - 2, 0


def _card_tokens(
    card_ids: Any,
    max_card_id: int,
    overflow_buckets: int = 0,
    card_id_remap: Any | None = None,
) -> Any:
    ids = card_ids.astype(jnp.int32)
    if card_id_remap is not None:
        remap = jnp.asarray(card_id_remap, dtype=jnp.int32)
        in_range = (ids > 0) & (ids < remap.shape[0])
        mapped = jnp.take(remap, jnp.clip(ids, 0, remap.shape[0] - 1), axis=0)
        return jnp.where(ids == 0, 0, jnp.where(in_range, mapped, 1))
    clipped = jnp.clip(ids, -1, max_card_id)
    direct = jnp.where(
        clipped > 0,
        clipped + 1,
        jnp.where(clipped < 0, jnp.ones_like(clipped), clipped),
    )
    if overflow_buckets <= 0:
        return direct
    hashed = jnp.mod(ids * jnp.int32(1103515245) + jnp.int32(12345), overflow_buckets)
    overflow = max_card_id + 2 + hashed
    return jnp.where(ids > max_card_id, overflow, direct)


@jax.custom_vjp
def _embedding_take_unique_bwd(table: Any, indices: Any) -> Any:
    return jnp.take(table, indices, axis=0)


def _embedding_take_unique_bwd_fwd(table: Any, indices: Any) -> tuple[Any, tuple[Any, Any]]:
    return jnp.take(table, indices, axis=0), (table, indices)


def _embedding_take_unique_bwd_bwd(res: tuple[Any, Any], grad: Any) -> tuple[Any, None]:
    table, indices = res
    flat_indices = jnp.reshape(indices.astype(jnp.int32), (-1,))
    flat_grad = jnp.reshape(grad, (flat_indices.shape[0], table.shape[-1]))
    order = jnp.argsort(flat_indices)
    table_grad = jax.ops.segment_sum(
        flat_grad[order],
        flat_indices[order],
        num_segments=table.shape[0],
        indices_are_sorted=True,
    )
    return table_grad.astype(table.dtype), None


_embedding_take_unique_bwd.defvjp(
    _embedding_take_unique_bwd_fwd,
    _embedding_take_unique_bwd_bwd,
)


def _embedding_take_one_hot(table: Any, indices: Any) -> Any:
    return jax.nn.one_hot(
        indices.astype(jnp.int32),
        table.shape[0],
        dtype=table.dtype,
    ) @ table


def _embedding_take(
    table: Any,
    indices: Any,
    custom_backward: bool = False,
    one_hot: bool = False,
) -> Any:
    if one_hot:
        return _embedding_take_one_hot(table, indices)
    if custom_backward:
        return _embedding_take_unique_bwd(table, indices)
    return jnp.take(table, indices, axis=0)


def _layer_norm(x: Any, scale: Any, bias: Any, eps: float = 1e-5) -> Any:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps) * scale + bias


def _inplay_scalar_features(in_play: Any) -> Any:
    present = jnp.clip(in_play[..., 0], 0, 1).astype(jnp.float32)
    flags = in_play[..., 7].astype(jnp.int32)
    attached_energy_types = in_play[..., 32:48].astype(jnp.int32)
    energy_known = attached_energy_types >= 0
    energy_type_ids = jnp.clip(attached_energy_types, 0, 11)
    energy_hist = jax.nn.one_hot(energy_type_ids, 12, dtype=jnp.float32)
    energy_hist = energy_hist * energy_known[..., None].astype(jnp.float32)
    energy_hist = jnp.clip(jnp.sum(energy_hist, axis=-2), 0.0, 12.0) / 12.0
    scalars = jnp.stack(
        [
            present,
            jnp.clip(in_play[..., 1], 0, 1).astype(jnp.float32),
            jnp.clip(in_play[..., 5].astype(jnp.float32), 0.0, 400.0) / 400.0,
            jnp.clip(in_play[..., 6].astype(jnp.float32), 0.0, 400.0) / 400.0,
            jnp.clip(in_play[..., 9].astype(jnp.float32), 0.0, 12.0) / 12.0,
            jnp.clip(in_play[..., 10].astype(jnp.float32), 0.0, 12.0) / 12.0,
            jnp.clip(in_play[..., 11].astype(jnp.float32), 0.0, 4.0) / 4.0,
            jnp.clip(in_play[..., 12].astype(jnp.float32), 0.0, 4.0) / 4.0,
            (jnp.bitwise_and(flags, 1 << 0) != 0).astype(jnp.float32),
            (jnp.bitwise_and(flags, 1 << 1) != 0).astype(jnp.float32),
            (jnp.bitwise_and(flags, 1 << 2) != 0).astype(jnp.float32),
            (jnp.bitwise_and(flags, 1 << 3) != 0).astype(jnp.float32),
            (jnp.bitwise_and(flags, 1 << 4) != 0).astype(jnp.float32),
            (jnp.bitwise_and(flags, 1 << 5) != 0).astype(jnp.float32),
            (jnp.bitwise_and(flags, 1 << 6) != 0).astype(jnp.float32),
            (jnp.bitwise_and(flags, 1 << 7) != 0).astype(jnp.float32),
        ],
        axis=-1,
    )
    return jnp.concatenate([scalars, energy_hist], axis=-1) * present[..., None]


def forward_tensors(
    params: dict[str, Any],
    in_play: Any,
    zones: Any,
    player_counts: Any,
    player_status: Any,
    global_: Any,
    action_options: Any,
    action_mask: Any,
    card_overflow_buckets: int = DEFAULT_CARD_OVERFLOW_BUCKETS,
    custom_embedding_backward: bool = False,
    card_id_remap: Any | None = None,
) -> dict[str, Any]:
    card_groups = [
        (jnp.reshape(in_play[..., 2], (in_play.shape[0], -1)), 0),
        (jnp.reshape(in_play[..., 16:32], (in_play.shape[0], -1)), 5),
        (jnp.reshape(in_play[..., 48:52], (in_play.shape[0], -1)), 6),
        (jnp.reshape(in_play[..., 52:56], (in_play.shape[0], -1)), 7),
        (jnp.reshape(zones[:, :, 0, :], (zones.shape[0], -1)), 1),
        (jnp.reshape(zones[:, :, 1, :], (zones.shape[0], -1)), 2),
        (jnp.reshape(zones[:, :, 2, :], (zones.shape[0], -1)), 3),
        (jnp.reshape(zones[:, :, 3, :], (zones.shape[0], -1)), 4),
    ]
    max_card_id, overflow_buckets = _card_token_bounds(params, card_overflow_buckets)
    card_state = jnp.zeros(
        (in_play.shape[0], params["card_embedding"].shape[-1]),
        dtype=params["card_embedding"].dtype,
    )
    for cards, role in card_groups:
        tokens = _card_tokens(cards, max_card_id, overflow_buckets, card_id_remap)
        emb_sum = jnp.sum(
            _embedding_take(
                params["card_embedding"],
                tokens,
                custom_embedding_backward,
                one_hot=card_id_remap is not None,
            ),
            axis=1,
        )
        card_state = card_state + emb_sum * params["role_gate"][role]
    inplay_scalar = _inplay_scalar_features(in_play)
    inplay_state = jax.nn.relu(
        inplay_scalar @ params["inplay_scalar_w1"] + params["inplay_scalar_b1"]
    )
    inplay_state = inplay_state @ params["inplay_scalar_w2"] + params["inplay_scalar_b2"]
    inplay_state = jnp.sum(inplay_state, axis=(1, 2))
    scalar_in = jnp.concatenate(
        [
            global_.astype(jnp.float32),
            jnp.reshape(player_counts.astype(jnp.float32), (player_counts.shape[0], -1)) / 64.0,
            jnp.reshape(player_status.astype(jnp.float32), (player_status.shape[0], -1)),
        ],
        axis=-1,
    )
    scalar_state = jax.nn.relu(scalar_in @ params["state_scalar_w1"] + params["state_scalar_b1"])
    scalar_state = scalar_state @ params["state_scalar_w2"] + params["state_scalar_b2"]
    state = card_state + inplay_state + scalar_state
    state = _layer_norm(state, params["state_out_scale"], params["state_out_bias"])
    state = jax.nn.relu(state @ params["state_out_w"] + params["state_out_b"])

    options = action_options
    present = jnp.clip(options[..., 0], 0, 1).astype(jnp.float32)[..., None]
    cg_type = jnp.clip(options[..., 1], 0, 31)
    kind = jnp.clip(options[..., 2], 0, 31)
    area = jnp.clip(options[..., 3], 0, 15)
    attack = jnp.clip(options[..., 10], 0, params["attack_embedding"].shape[0] - 1)
    scalars = jnp.stack(
        [
            options[..., 4].astype(jnp.float32) / 64.0,
            options[..., 5].astype(jnp.float32),
            options[..., 6].astype(jnp.float32) / 16.0,
            options[..., 7].astype(jnp.float32) / 16.0,
            options[..., 11].astype(jnp.float32) / 64.0,
            options[..., 13].astype(jnp.float32) / 16.0,
            options[..., 14].astype(jnp.float32) / 16.0,
            options[..., 15].astype(jnp.float32) / 8.0,
        ],
        axis=-1,
    )
    action_card_0 = _card_tokens(options[..., 8], max_card_id, overflow_buckets, card_id_remap)
    action_card_1 = _card_tokens(options[..., 9], max_card_id, overflow_buckets, card_id_remap)
    action_card_emb = (
        _embedding_take(
            params["card_embedding"],
            action_card_0,
            custom_embedding_backward,
            one_hot=card_id_remap is not None,
        )
        * params["role_gate"][8]
        + _embedding_take(
            params["card_embedding"],
            action_card_1,
            custom_embedding_backward,
            one_hot=card_id_remap is not None,
        )
        * params["role_gate"][9]
    )
    actions = (
        _embedding_take(params["type_embedding"], cg_type, one_hot=True)
        + _embedding_take(params["kind_embedding"], kind, one_hot=True)
        + _embedding_take(params["area_embedding"], area, one_hot=True)
        + _embedding_take(params["attack_embedding"], attack, custom_backward=True)
        + action_card_emb
        + scalars @ params["action_scalar_w"]
        + params["action_scalar_b"]
    )
    actions = _layer_norm(actions, params["action_out_scale"], params["action_out_bias"])
    actions = jax.nn.relu(actions @ params["action_out_w"] + params["action_out_b"]) * present

    query = state @ params["policy_state_w"] + params["policy_state_b"]
    logits = jnp.sum(actions * query[:, None, :], axis=-1) / jnp.sqrt(actions.shape[-1])
    logits = jnp.where(action_mask.astype(bool), logits, jnp.finfo(jnp.float32).min)
    value = jax.nn.relu(state @ params["value_w1"] + params["value_b1"])
    value = jnp.squeeze(value @ params["value_w2"] + params["value_b2"], axis=-1)
    return {
        "logits": logits,
        "value": value,
        "state_embedding": state,
        "action_embedding": actions,
        "mask": action_mask.astype(jnp.float32),
    }


def forward_policy_value(
    params: dict[str, Any],
    in_play: Any,
    zones: Any,
    player_counts: Any,
    player_status: Any,
    global_: Any,
    action_options: Any,
    action_mask: Any,
    card_overflow_buckets: int = DEFAULT_CARD_OVERFLOW_BUCKETS,
    custom_embedding_backward: bool = False,
    card_id_remap: Any | None = None,
) -> dict[str, Any]:
    out = forward_tensors(
        params,
        in_play,
        zones,
        player_counts,
        player_status,
        global_,
        action_options,
        action_mask,
        card_overflow_buckets,
        custom_embedding_backward,
        card_id_remap,
    )
    return {
        "logits": out["logits"],
        "value": out["value"],
    }
