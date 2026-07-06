from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - exercised only without torch.
    raise ImportError(
        "ptcg.nn requires PyTorch. Install torch to use the neural-network "
        "adapters; the engine itself does not depend on torch."
    ) from exc


DEFAULT_MAX_CARD_ID = 1300
DEFAULT_CARD_OVERFLOW_BUCKETS = 0


def _as_long(value: Any, device: torch.device | None = None) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.dtype in (torch.int32, torch.int64):
        return tensor
    return tensor.long()


def _as_float(value: Any, device: torch.device | None = None) -> Tensor:
    tensor = torch.as_tensor(value, device=device)
    if tensor.dtype == torch.float32:
        return tensor
    return tensor.float()


def _ensure_batch(x: Tensor, dims_without_batch: int) -> tuple[Tensor, bool]:
    if x.dim() == dims_without_batch:
        return x.unsqueeze(0), True
    return x, False


def _card_tokens(card_ids: Tensor, max_card_id: int, overflow_buckets: int = 0) -> Tensor:
    ids = card_ids.long()
    clipped = ids.clamp(min=-1, max=max_card_id)
    direct = torch.where(
        clipped > 0,
        clipped + 1,
        torch.where(clipped < 0, torch.ones_like(clipped), clipped),
    )
    if overflow_buckets <= 0:
        return direct
    hashed = torch.remainder(ids * 1103515245 + 12345, int(overflow_buckets))
    overflow = max_card_id + 2 + hashed
    return torch.where(ids > max_card_id, overflow, direct)


@dataclass(frozen=True)
class PackedRoles:
    in_play: int = 0
    hand: int = 1
    deck: int = 2
    discard: int = 3
    prize: int = 4
    energy: int = 5
    tool: int = 6
    pre_evo: int = 7
    source: int = 8
    target: int = 9

    @property
    def count(self) -> int:
        return 10


class RoleCardEmbedding(nn.Module):
    """Card embedding with a learned elementwise gate per semantic role.

    Raw engine ids are mapped as:

    - ``0`` -> empty padding token
    - ``-1`` -> unknown-card token
    - ``1 <= card_id <= max_card_id`` -> card-specific token
    - ``card_id > max_card_id`` -> stable overflow bucket token

    For each role, a trainable gate transforms the base card embedding:
    ``card_embedding(card_id) * role_gate[role]``.
    """

    def __init__(
        self,
        max_card_id: int = DEFAULT_MAX_CARD_ID,
        dim: int = 128,
        roles: int = 10,
        overflow_buckets: int = DEFAULT_CARD_OVERFLOW_BUCKETS,
    ) -> None:
        super().__init__()
        self.max_card_id = int(max_card_id)
        self.dim = int(dim)
        self.overflow_buckets = int(max(0, overflow_buckets))
        self.embedding = nn.Embedding(
            self.max_card_id + 2 + self.overflow_buckets, dim, padding_idx=0
        )
        self.role_gate = nn.Parameter(torch.empty(roles, dim))
        nn.init.normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
            self.role_gate.fill_(1.0)
            self.role_gate.add_(0.01 * torch.randn_like(self.role_gate))

    def forward(self, card_ids: Tensor, role: int | Tensor) -> Tensor:
        tokens = _card_tokens(card_ids, self.max_card_id, self.overflow_buckets)
        emb = self.embedding(tokens)
        if isinstance(role, int):
            return emb * self.role_gate[role]
        return emb * self.role_gate[role.long()]


class PackedObservationEncoder(nn.Module):
    """Encode packed ``rl_state_ids`` tensors into one state embedding."""

    inplay_scalar_width: int = 28

    def __init__(
        self,
        card_embed: RoleCardEmbedding,
        hidden_dim: int = 256,
        roles: PackedRoles = PackedRoles(),
    ) -> None:
        super().__init__()
        self.card_embed = card_embed
        self.roles = roles
        d = card_embed.dim
        self.inplay_scalar = nn.Sequential(
            nn.Linear(self.inplay_scalar_width, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )
        self.scalar = nn.Sequential(
            nn.Linear(32 + 10 + 10, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d),
        )
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU())

    def forward(self, state_ids: dict[str, Any]) -> Tensor:
        device = self.card_embed.embedding.weight.device
        in_play = _as_long(state_ids["in_play"], device)
        zones = _as_long(state_ids["zones"], device)
        counts = _as_float(state_ids["player_counts"], device)
        status = _as_float(state_ids["player_status"], device)
        global_ = _as_float(state_ids["global"], device)
        return self.forward_tensors(in_play, zones, counts, status, global_)

    def forward_tensors(
        self,
        in_play: Tensor,
        zones: Tensor,
        player_counts: Tensor,
        player_status: Tensor,
        global_: Tensor,
    ) -> Tensor:
        in_play, squeeze = _ensure_batch(in_play, 3)
        zones, _ = _ensure_batch(zones, 3)
        counts, _ = _ensure_batch(player_counts, 2)
        status, _ = _ensure_batch(player_status, 2)
        global_, _ = _ensure_batch(global_, 1)

        card_groups = [
            (in_play[..., 2].reshape(in_play.shape[0], -1), self.roles.in_play),
            (in_play[..., 16:32].reshape(in_play.shape[0], -1), self.roles.energy),
            (in_play[..., 48:52].reshape(in_play.shape[0], -1), self.roles.tool),
            (in_play[..., 52:56].reshape(in_play.shape[0], -1), self.roles.pre_evo),
            (zones[:, :, 0, :].reshape(zones.shape[0], -1), self.roles.hand),
            (zones[:, :, 1, :].reshape(zones.shape[0], -1), self.roles.deck),
            (zones[:, :, 2, :].reshape(zones.shape[0], -1), self.roles.discard),
            (zones[:, :, 3, :].reshape(zones.shape[0], -1), self.roles.prize),
        ]
        card_ids = torch.cat([cards for cards, _role in card_groups], dim=1)
        role_ids = torch.cat(
            [torch.full_like(cards, role) for cards, role in card_groups],
            dim=1,
        )
        card_state = self.card_embed(card_ids, role_ids).sum(dim=1)
        inplay_state = self.inplay_scalar(self._inplay_scalar_features(in_play)).sum(dim=(1, 2))
        scalar_in = torch.cat(
            [
                global_,
                counts.reshape(counts.shape[0], -1) / 64.0,
                status.reshape(status.shape[0], -1),
            ],
            dim=-1,
        )
        state = card_state + inplay_state + self.scalar(scalar_in)
        state = self.out(state)
        return state.squeeze(0) if squeeze else state

    def _inplay_scalar_features(self, in_play: Tensor) -> Tensor:
        present = in_play[..., 0].clamp(0, 1).float()
        flags = in_play[..., 7].long()
        attached_energy_types = in_play[..., 32:48].long()
        energy_known = attached_energy_types >= 0
        energy_type_ids = attached_energy_types.clamp(0, 11)
        energy_hist = torch.nn.functional.one_hot(energy_type_ids, num_classes=12).float()
        energy_hist = energy_hist * energy_known.unsqueeze(-1).float()
        energy_hist = energy_hist.sum(dim=-2).clamp(0, 12) / 12.0
        scalars = torch.stack(
            [
                present,
                in_play[..., 1].clamp(0, 1).float(),
                in_play[..., 5].float().clamp(0, 400) / 400.0,
                in_play[..., 6].float().clamp(0, 400) / 400.0,
                in_play[..., 9].float().clamp(0, 12) / 12.0,
                in_play[..., 10].float().clamp(0, 12) / 12.0,
                in_play[..., 11].float().clamp(0, 4) / 4.0,
                in_play[..., 12].float().clamp(0, 4) / 4.0,
                ((flags & (1 << 0)) != 0).float(),
                ((flags & (1 << 1)) != 0).float(),
                ((flags & (1 << 2)) != 0).float(),
                ((flags & (1 << 3)) != 0).float(),
                ((flags & (1 << 4)) != 0).float(),
                ((flags & (1 << 5)) != 0).float(),
                ((flags & (1 << 6)) != 0).float(),
                ((flags & (1 << 7)) != 0).float(),
            ],
            dim=-1,
        )
        return torch.cat([scalars, energy_hist], dim=-1) * present.unsqueeze(-1)


class PackedActionEncoder(nn.Module):
    """Encode packed ``rl_action_ids`` tensors into per-action embeddings."""

    def __init__(
        self,
        card_embed: RoleCardEmbedding,
        hidden_dim: int = 256,
        max_attack_id: int = 4096,
        roles: PackedRoles = PackedRoles(),
    ) -> None:
        super().__init__()
        self.card_embed = card_embed
        self.roles = roles
        self.attack_embedding = nn.Embedding(max_attack_id + 1, card_embed.dim, padding_idx=0)
        self.kind_embedding = nn.Embedding(32, card_embed.dim, padding_idx=0)
        self.type_embedding = nn.Embedding(32, card_embed.dim, padding_idx=0)
        self.area_embedding = nn.Embedding(16, card_embed.dim, padding_idx=0)
        self.scalar = nn.Linear(8, card_embed.dim)
        self.out = nn.Sequential(nn.LayerNorm(card_embed.dim), nn.Linear(card_embed.dim, card_embed.dim), nn.ReLU())

    def forward(self, action_ids: dict[str, Any]) -> tuple[Tensor, Tensor]:
        device = self.card_embed.embedding.weight.device
        options = _as_long(action_ids["options"], device)
        mask = _as_float(action_ids["mask"], device)
        return self.forward_tensors(options, mask)

    def forward_tensors(self, options: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        options, squeeze = _ensure_batch(options, 2)
        mask, _ = _ensure_batch(mask, 1)

        present = options[..., 0].clamp(0, 1).float().unsqueeze(-1)
        cg_type = options[..., 1].clamp(0, 31)
        kind = options[..., 2].clamp(0, 31)
        area = options[..., 3].clamp(0, 15)
        source = options[..., 8]
        target = options[..., 9]
        attack = options[..., 10].clamp(0, self.attack_embedding.num_embeddings - 1)
        scalars = torch.stack(
            [
                options[..., 4].float() / 64.0,
                options[..., 5].float(),
                options[..., 6].float() / 16.0,
                options[..., 7].float() / 16.0,
                options[..., 11].float() / 64.0,
                options[..., 13].float() / 16.0,
                options[..., 14].float() / 16.0,
                options[..., 15].float() / 8.0,
            ],
            dim=-1,
        )
        action_cards = torch.stack([source, target], dim=-1)
        action_roles = torch.empty_like(action_cards)
        action_roles[..., 0] = self.roles.source
        action_roles[..., 1] = self.roles.target
        action_card_emb = self.card_embed(action_cards, action_roles).sum(dim=-2)
        encoded = (
            self.type_embedding(cg_type)
            + self.kind_embedding(kind)
            + self.area_embedding(area)
            + action_card_emb
            + self.attack_embedding(attack)
            + self.scalar(scalars)
        )
        encoded = self.out(encoded) * present
        return (encoded.squeeze(0), mask.squeeze(0)) if squeeze else (encoded, mask)


class PtcgPolicyValueNet(nn.Module):
    """Small policy/value network for PPO-style experiments on packed tensors."""

    def __init__(
        self,
        max_card_id: int = DEFAULT_MAX_CARD_ID,
        card_dim: int = 128,
        hidden_dim: int = 256,
        max_attack_id: int = 4096,
        card_overflow_buckets: int = DEFAULT_CARD_OVERFLOW_BUCKETS,
    ) -> None:
        super().__init__()
        card_embed = RoleCardEmbedding(
            max_card_id=max_card_id,
            dim=card_dim,
            overflow_buckets=card_overflow_buckets,
        )
        self.state_encoder = PackedObservationEncoder(card_embed, hidden_dim=hidden_dim)
        self.action_encoder = PackedActionEncoder(
            card_embed, hidden_dim=hidden_dim, max_attack_id=max_attack_id
        )
        self.policy_state = nn.Linear(card_dim, card_dim)
        self.value = nn.Sequential(nn.Linear(card_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state_ids: dict[str, Any], action_ids: dict[str, Any]) -> dict[str, Tensor]:
        state = self.state_encoder(state_ids)
        actions, mask = self.action_encoder(action_ids)
        return self._policy_value(state, actions, mask)

    def forward_tensors(
        self,
        in_play: Tensor,
        zones: Tensor,
        player_counts: Tensor,
        player_status: Tensor,
        global_: Tensor,
        action_options: Tensor,
        action_mask: Tensor,
    ) -> dict[str, Tensor]:
        state = self.state_encoder.forward_tensors(
            in_play, zones, player_counts, player_status, global_
        )
        actions, mask = self.action_encoder.forward_tensors(action_options, action_mask)
        return self._policy_value(state, actions, mask)

    def _policy_value(self, state: Tensor, actions: Tensor, mask: Tensor) -> dict[str, Tensor]:
        squeeze = state.dim() == 1
        if squeeze:
            state = state.unsqueeze(0)
            actions = actions.unsqueeze(0)
            mask = mask.unsqueeze(0)
        query = self.policy_state(state).unsqueeze(1)
        logits = (actions * query).sum(dim=-1) / (actions.shape[-1] ** 0.5)
        logits = logits.masked_fill(mask <= 0, torch.finfo(logits.dtype).min)
        value = self.value(state).squeeze(-1)
        if squeeze:
            logits = logits.squeeze(0)
            value = value.squeeze(0)
            state = state.squeeze(0)
            actions = actions.squeeze(0)
            mask = mask.squeeze(0)
        return {
            "logits": logits,
            "value": value,
            "state_embedding": state,
            "action_embedding": actions,
            "mask": mask,
        }

    def act(
        self,
        state_ids: dict[str, Any],
        action_ids: dict[str, Any],
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        out = self.forward(state_ids, action_ids)
        dist = torch.distributions.Categorical(logits=out["logits"])
        action = torch.argmax(out["logits"], dim=-1) if deterministic else dist.sample()
        logprob = dist.log_prob(action)
        return action, logprob, out["value"]
