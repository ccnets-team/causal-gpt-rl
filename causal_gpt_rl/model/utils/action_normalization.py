"""Embedded pre-tanh action coordinates shared by inference and ONNX export."""
import torch


ACTION_NORMALIZATION_COORDINATE = "bounds_atanh_then_standardize_v1"
ENVIRONMENT_ACTION_COORDINATE = "environment_action_v1"
PRE_TANH_ACTION_COORDINATE = "standardized_pre_tanh_v1"
PRE_TANH_ROLLOUT_CAPABILITY = "pre_tanh_rollout_context"


def validate_rollout_action_coordinate(coordinate, *, action_normalized):
    if coordinate not in (ENVIRONMENT_ACTION_COORDINATE, PRE_TANH_ACTION_COORDINATE):
        raise ValueError(f"Unsupported rollout action coordinate: {coordinate!r}")
    if coordinate == PRE_TANH_ACTION_COORDINATE and not action_normalized:
        raise ValueError("Pre-tanh rollout context requires enabled action normalization")


_STATE_NAMES = (
    "action_normalization_enabled",
    "action_normalization_mean",
    "action_normalization_std",
)


class ActionNormalizationMixin:
    def actions_to_rollout_context(
        self, actions, *, action_context_coordinate=ENVIRONMENT_ACTION_COORDINATE,
        is_bos=None,
    ):
        """Encode flat external raw actions for a chosen context coordinate.

        Generated direct-z feedback must use the original sample instead.
        Discrete columns are already feedback one-hot/bits, not output logits.
        """
        validate_rollout_action_coordinate(
            action_context_coordinate,
            action_normalized=self.has_embedded_action_normalizer(),
        )
        actions = torch.as_tensor(
            actions, dtype=torch.float32, device=self.action_normalization_mean.device,
        )
        if actions.shape[-1] != self.action_size:
            raise ValueError(f"Expected action feedback width {self.action_size}")
        parts = []
        for spec, section in zip(self.action_specs, self._action_slices):
            head = actions[..., section]
            if action_context_coordinate == PRE_TANH_ACTION_COORDINATE and spec.type == "continuous":
                head = self._normalize_action_head(head, section)
            parts.append(head)
        context = torch.cat(parts, dim=-1)
        if is_bos is not None:
            bos = torch.as_tensor(is_bos, device=context.device, dtype=torch.bool)
            context = torch.where(bos, torch.zeros_like(context), context)
        return context

    def _init_action_normalization(self):
        self.action_size = sum(s.size for s in self.action_specs)
        self.register_buffer(_STATE_NAMES[0], torch.zeros(1, dtype=torch.float32))
        self.register_buffer(_STATE_NAMES[1], torch.zeros(self.action_size, dtype=torch.float32))
        self.register_buffer(_STATE_NAMES[2], torch.ones(self.action_size, dtype=torch.float32))
        indices, lows, highs = [], [], []
        self._action_slices = []
        self._action_normalization_bounds_valid = True
        offset = 0
        for spec in self.action_specs:
            self._action_slices.append(slice(offset, offset + spec.size))
            if spec.type == "continuous":
                indices.extend(range(offset, offset + spec.size))
                low = torch.as_tensor(spec.low if spec.low is not None else float("nan"), dtype=torch.float32).reshape(-1)
                high = torch.as_tensor(spec.high if spec.high is not None else float("nan"), dtype=torch.float32).reshape(-1)
                if low.numel() == 1:
                    low = low.expand(spec.size)
                if high.numel() == 1:
                    high = high.expand(spec.size)
                valid = (
                    low.numel() == spec.size and high.numel() == spec.size
                    and bool(torch.isfinite(low).all() and torch.isfinite(high).all())
                    and bool((high > low).all() and torch.isfinite(high - low).all())
                )
                self._action_normalization_bounds_valid &= valid
                # Safe unused branch for legacy unbounded specs in torch.where.
                lows.extend(low.tolist() if valid else [-1.] * spec.size)
                highs.extend(high.tolist() if valid else [1.] * spec.size)
            else:
                lows.extend([-1.] * spec.size)
                highs.extend([1.] * spec.size)
            offset += spec.size
        self.register_buffer("_normalized_action_indices", torch.tensor(indices, dtype=torch.long), persistent=False)
        self.register_buffer("_normalized_action_low", torch.tensor(lows, dtype=torch.float32), persistent=False)
        self.register_buffer("_normalized_action_high", torch.tensor(highs, dtype=torch.float32), persistent=False)

    def has_embedded_action_normalizer(self) -> bool:
        return bool(self.action_normalization_enabled.item() == 1.0)

    def _validate_action_normalization(self, enabled, mean, std):
        if any(value.dtype != torch.float32 for value in (enabled, mean, std)):
            raise ValueError("Action normalization state must be float32")
        if enabled.shape != (1,) or not bool(torch.isfinite(enabled).all()) or enabled.item() not in (0., 1.):
            raise ValueError("action_normalization_enabled must be finite float32[1], 0 or 1")
        if mean.shape != (self.action_size,) or std.shape != (self.action_size,):
            raise ValueError(f"Expected action normalization size {self.action_size}")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError("Action normalization statistics must be finite")
        if not bool((std > 0).all()):
            raise ValueError("Action normalization std must be positive")
        for spec, section in zip(self.action_specs, self._action_slices):
            if spec.type != "continuous" and not bool((mean[section] == 0).all() and (std[section] == 1).all()):
                raise ValueError("Non-continuous action statistics must be identity (mean=0, std=1)")
        if enabled.item() == 1. and (not self._normalized_action_indices.numel() or not self._action_normalization_bounds_valid):
            raise ValueError("Action normalization requires finite, increasing continuous action bounds")

    @torch.no_grad()
    def set_action_normalization(self, *, mean, std=None, var=None, enabled=True) -> None:
        """Set full feedback-width pre-tanh statistics; variance uses sqrt, no epsilon."""
        if (std is None) == (var is None):
            raise ValueError("Provide exactly one of std or var")
        target = self.action_normalization_mean
        mean = torch.as_tensor(mean, device=target.device, dtype=torch.float32).reshape(-1)
        scale = torch.as_tensor(std if std is not None else var, device=target.device, dtype=torch.float32).reshape(-1)
        if var is not None:
            scale = torch.sqrt(scale)
        flag = torch.tensor([float(enabled)], device=target.device, dtype=torch.float32)
        self._validate_action_normalization(flag, mean, scale)
        self.action_normalization_mean.copy_(mean)
        self.action_normalization_std.copy_(scale)
        self.action_normalization_enabled.copy_(flag)

    def set_action_normalization_from_state_dict(self, state_dict, *, enabled=True) -> None:
        self.set_action_normalization(
            mean=state_dict["mean"], std=state_dict.get("std"),
            var=state_dict.get("var"), enabled=enabled,
        )

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        keys = [prefix + name for name in _STATE_NAMES]
        present = [key in state_dict for key in keys]
        if any(present):
            if not all(present):
                raise ValueError("Incomplete action normalization state: enabled, mean and std are required")
            values = [state_dict[key] for key in keys]
            self._validate_action_normalization(*values)
        else:
            # Loading a legacy checkpoint also resets a previously enabled model.
            for key, name in zip(keys, _STATE_NAMES):
                current = getattr(self, name)
                state_dict[key] = torch.ones_like(current) if name.endswith("_std") else torch.zeros_like(current)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def _normalize_action_head(self, raw, section):
        low = self._normalized_action_low[section]
        high = self._normalized_action_high[section]
        x = (2.0 * (raw.float() - low) / (high - low) - 1.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        u = torch.atanh(x)
        return (u - self.action_normalization_mean[section]) / self.action_normalization_std[section]

    def _denormalize_action_head(self, z, section):
        u = z.float() * self.action_normalization_std[section] + self.action_normalization_mean[section]
        low = self._normalized_action_low[section]
        high = self._normalized_action_high[section]
        return low + (torch.tanh(u) + 1.0) * (high - low) / 2.0

    def _normalized_or_legacy(self, normalized, legacy):
        # Tensor selection keeps the state authoritative without Tensor.item()
        # or mutable adapter flags inside torch.export / ONNX graphs.
        return torch.where(self.action_normalization_enabled == 1.0, normalized.to(legacy.dtype), legacy)

    def _continuous_sample_to_env(self, z, legacy):
        normalized = self._denormalize_action_head(z, self._normalized_action_indices)
        return self._normalized_or_legacy(normalized, legacy)
