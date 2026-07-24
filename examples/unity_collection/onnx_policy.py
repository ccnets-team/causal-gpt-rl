"""Run a baked ML-Agents `.onnx` policy in onnxruntime, batched across agents.

Handles continuous, discrete, and hybrid (continuous + discrete) behaviors, and
both ONNX observation layouts:

  - per-sensor inputs (`obs_0`, `obs_1`, …), fed channel by channel, and
  - a single concatenated `vector_observation`, fed the concatenated channels.

The rule is uniform: concatenate the build's obs channels in spec order into one
flat vector, then split that vector across the ONNX obs inputs by their declared
dims. (Crawler: 126+32 -> two inputs; PushBlock: 105+105 -> one 210
input.) This concatenation is only how the *driver policy* is fed — it is
independent of how the dataset declares its observation space (per-sensor).

Discrete `discrete_actions` comes in two encoding flavors, detected by width vs the
build's branch sizes:

  - width == num_branches       -> already-sampled indices, used as-is.
  - width == sum(branch_sizes)  -> per-branch (masked) log-probs; one index is
                                   sampled per branch.

`action_masks` inputs are fed all-ones (no masking). Recurrent policies
(`recurrent_in`) are not supported and fail loud.

`act()` returns a `[num_agents, act_dim]` array. Per kind:

  - continuous -> continuous values `[num_agents, cont_size]`
  - discrete   -> 0-based indices `[num_agents, num_branches]`
  - hybrid     -> `[continuous | discrete indices]` concatenated,
                  `[num_agents, cont_size + num_branches]`

Agents absent this step get a zero row (the UnityEnv wrapper masks them out).
"""
import numpy as np
import onnxruntime as ort

_MASK_INPUT = "action_masks"
_RECURRENT_INPUT = "recurrent_in"


class OnnxPolicy:
    def __init__(self, onnx_path, num_agents, obs_shapes, action_spec,
                 providers=None, rng=None, temperature=1.0):
        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.num_agents = num_agents
        self.obs_dims = [int(np.prod(s)) for s in obs_shapes]
        self.total_obs = sum(self.obs_dims)
        self._rng = rng if rng is not None else np.random.default_rng()
        # Discrete temperature dial (additive; independent of NoisyPolicy's
        # noise_std/epsilon). Sampling from softmax(logits / T) needs a `logits`
        # output, present only in the patched tier_policies onnx. T == 1 with no
        # per-agent override is an exact passthrough (existing behavior).
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0 (got {temperature}).")
        self.temperature = float(temperature)
        self._temperature_by_agent = None

        # Classify inputs: obs inputs (partition the flat obs), action_masks, and
        # the unsupported recurrent input.
        self.obs_inputs = []   # (name, dim), in order
        self.mask_inputs = []  # (name, dim)
        for inp in self.session.get_inputs():
            dim = int(inp.shape[-1])
            if inp.name == _MASK_INPUT:
                self.mask_inputs.append((inp.name, dim))
            elif inp.name == _RECURRENT_INPUT:
                raise NotImplementedError(
                    f"Recurrent policy ('{_RECURRENT_INPUT}') is not supported."
                )
            else:
                self.obs_inputs.append((inp.name, dim))

        packed = sum(d for _, d in self.obs_inputs)
        if packed != self.total_obs:
            raise ValueError(
                f"ONNX obs inputs total {packed} != build obs total {self.total_obs} "
                f"(inputs={self.obs_inputs}, channels={self.obs_dims})."
            )

        out = {o.name: o for o in self.session.get_outputs()}
        # Patched tier_policies onnx exposes per-branch pre-softmax `logits`,
        # enabling temperature sampling. Stock onnx lacks it (temperature errors loud).
        self.has_logits = "logits" in out
        cont_size = int(getattr(action_spec, "continuous_size", 0) or 0)
        branches = tuple(int(b) for b in (getattr(action_spec, "discrete_branches", ()) or ()))

        if cont_size > 0 and not branches:
            if "continuous_actions" not in out:
                raise ValueError(f"Continuous behavior but no continuous_actions output ({list(out)}).")
            self.kind = "continuous"
            self.out_name = "continuous_actions"
            self.act_dim = cont_size
            self.branches = ()
        elif branches and cont_size == 0:
            if "discrete_actions" not in out:
                raise ValueError(f"Discrete behavior but no discrete_actions output ({list(out)}).")
            self.kind = "discrete"
            self.out_name = "discrete_actions"
            self.branches = branches
            self.act_dim = len(branches)
            self._disc_width = int(out["discrete_actions"].shape[-1])
            if self._disc_width not in (len(branches), sum(branches)):
                raise ValueError(
                    f"discrete_actions width {self._disc_width} matches neither num_branches "
                    f"{len(branches)} nor sum(branches) {sum(branches)}."
                )
        elif cont_size > 0 and branches:
            for req in ("continuous_actions", "discrete_actions"):
                if req not in out:
                    raise ValueError(f"Hybrid behavior but no {req} output ({list(out)}).")
            self.kind = "hybrid"
            self.out_name = None  # hybrid runs both outputs explicitly in act()
            self.cont_size = cont_size
            self.branches = branches
            # act_dim = continuous dims + one 0-based index column per discrete branch
            self.act_dim = cont_size + len(branches)
            self._disc_width = int(out["discrete_actions"].shape[-1])
            if self._disc_width not in (len(branches), sum(branches)):
                raise ValueError(
                    f"discrete_actions width {self._disc_width} matches neither num_branches "
                    f"{len(branches)} nor sum(branches) {sum(branches)}."
                )
        else:
            raise NotImplementedError(
                f"Action behavior (continuous={cont_size}, discrete={branches}) not supported."
            )

    def _build_feeds(self, present, observations):
        flat = np.stack(
            [
                np.concatenate(
                    [np.asarray(observations[c][g], np.float32).reshape(-1)
                     for c in range(len(self.obs_dims))]
                )
                for g in present
            ],
            axis=0,
        )  # [P, total_obs]
        feeds = {}
        off = 0
        for name, dim in self.obs_inputs:
            feeds[name] = flat[:, off:off + dim]
            off += dim
        for name, dim in self.mask_inputs:
            feeds[name] = np.ones((len(present), dim), np.float32)
        return feeds

    def _decode_discrete(self, raw):
        raw = np.asarray(raw, dtype=np.float32)
        n_present = raw.shape[0]
        nb = len(self.branches)
        if self._disc_width == nb:
            # already-sampled indices
            return np.rint(raw.reshape(n_present, nb)).astype(np.int64)
        # per-branch masked log-probs -> sample one index per branch
        idx = np.zeros((n_present, nb), np.int64)
        off = 0
        for b, size in enumerate(self.branches):
            logits = raw[:, off:off + size]
            off += size
            shifted = logits - logits.max(axis=1, keepdims=True)
            prob = np.exp(shifted)
            prob /= prob.sum(axis=1, keepdims=True)
            for r in range(n_present):
                idx[r, b] = self._rng.choice(size, p=prob[r])
        return idx

    def set_temperature_by_agent(self, values):
        """Override scalar temperature with one value per policy batch row.

        Mirrors ``NoisyPolicy.set_epsilon_by_agent`` for the discrete temperature
        dial: a per-agent per-episode collector pushes a fresh temperature vector
        here so each episode samples ``softmax(logits / T)`` at its own T. Passing
        ``None`` restores scalar ``temperature``.
        """
        if values is None:
            self._temperature_by_agent = None
            return
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if np.any(values <= 0.0):
            raise ValueError("per-agent temperature values must be > 0.")
        self._temperature_by_agent = values.copy()

    def _temperature_for(self, agent_index):
        if self._temperature_by_agent is None:
            return self.temperature
        if agent_index >= len(self._temperature_by_agent):
            raise ValueError(
                "per-agent temperature length does not cover policy batch row "
                f"{agent_index}"
            )
        return float(self._temperature_by_agent[agent_index])

    def _temperature_active(self):
        return self._temperature_by_agent is not None or self.temperature != 1.0

    def _decode_temperature(self, logits, present):
        """Sample one index per branch from softmax(logits / T), per-agent T.

        `logits` is the patched onnx `logits` output `[P, sum(branch_sizes)]`,
        rows aligned with `present`. Replaces the frozen T=1 discrete sample with
        a temperature-controlled draw (higher T -> lower skill) for tier synthesis.
        """
        logits = np.asarray(logits, dtype=np.float32)
        n_present = logits.shape[0]
        nb = len(self.branches)
        idx = np.zeros((n_present, nb), np.int64)
        for r in range(n_present):
            temp = self._temperature_for(present[r])
            off = 0
            for b, size in enumerate(self.branches):
                z = logits[r, off:off + size] / temp
                z = z - z.max()
                prob = np.exp(z)
                prob /= prob.sum()
                idx[r, b] = self._rng.choice(size, p=prob)
                off += size
        return idx

    def act(self, observations):
        n = self.num_agents
        present = [g for g in range(n) if observations[0][g] is not None]
        out = np.zeros((n, self.act_dim), dtype=np.float32)
        if not present:
            return out
        feeds = self._build_feeds(present, observations)
        if self.kind == "hybrid":
            # Two action outputs: continuous_actions (already sampled, in-range) and
            # discrete_actions (per-branch log-probs). Concat [continuous | indices].
            cont, disc = self.session.run(["continuous_actions", "discrete_actions"], feeds)
            cont = np.asarray(cont, np.float32).reshape(len(present), -1)
            idx = self._decode_discrete(disc).astype(np.float32)
            vals = np.concatenate([cont, idx], axis=1)
        elif self.kind == "discrete" and self._temperature_active():
            if not self.has_logits:
                raise ValueError(
                    "temperature sampling requires a 'logits' output; use the patched "
                    "tier_policies onnx (stock onnx only emits the sampled index)."
                )
            raw = self.session.run(["logits"], feeds)[0]
            vals = self._decode_temperature(raw, present).astype(np.float32)
        else:
            raw = self.session.run([self.out_name], feeds)[0]
            if self.kind == "continuous":
                vals = np.asarray(raw, np.float32).reshape(len(present), -1)
            else:
                vals = self._decode_discrete(raw).astype(np.float32)
        for j, g in enumerate(present):
            out[g] = vals[j]
        return out
