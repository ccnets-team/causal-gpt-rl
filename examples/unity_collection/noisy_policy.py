"""Degrade a policy's actions with noise to synthesize lower-quality datasets.

There is only ONE Crawler policy available — the stock ML-Agents `.onnx`. The
`simple`/`medium`/`expert` quality ladder that the MuJoCo Minari datasets get for
free (each tier is a separate policy) is synthesized here from that
single expert by injecting a controlled amount of action noise: more noise ->
lower closed-loop return -> a lower tier.

Two independent dials, both reproducible from `rng`:

  - `noise_std` — additive Gaussian noise on continuous actions, then re-clipped
    to the action range. The smooth dial: raise it to lower the return. It is the
    primary knob for a metric (continuous) action space.
  - `epsilon`   — per-agent probability of discarding the policy action entirely
    and substituting a uniform-random one. `epsilon=1.0` IS a random policy, so
    it is how the `random_ref` baseline used in score normalization is measured.
    For discrete branches (where Gaussian noise is meaningless) it resamples each
    branch, so it is the only degradation dial that path has.

The wrapper records the action it actually returns; the collector stores and the
env steps that same (noised, clipped) action, so the dataset stays self-consistent
and every stored continuous action is a valid sample of `Box[-1, 1]`.
"""
import numpy as np

_LO, _HI = -1.0, 1.0  # continuous action range (ML-Agents Box[-1, 1])


class NoisyPolicy:
    """Wrap a policy, returning `act()` outputs perturbed by Gaussian and/or
    epsilon-random noise. `noise_std == epsilon == 0` is an exact passthrough
    (the `expert` tier). Handles continuous, discrete, and hybrid action layouts;
    delegates `kind`/`branches`/`act_dim` to the inner policy so it drops in
    wherever an `OnnxPolicy` is used.
    """

    def __init__(self, policy, noise_std=0.0, epsilon=0.0, rng=None):
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0 (got {noise_std}).")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1] (got {epsilon}).")
        self._policy = policy
        self.noise_std = float(noise_std)
        self.epsilon = float(epsilon)
        self._rng = rng if rng is not None else np.random.default_rng()

    # Passthrough spec so build_minari.py sees the same action space either way.
    @property
    def kind(self):
        return self._policy.kind

    @property
    def branches(self):
        return self._policy.branches

    @property
    def act_dim(self):
        return self._policy.act_dim

    def act(self, observations):
        a = self._policy.act(observations)  # [n, act_dim]; absent rows are zero
        present = [g for g in range(a.shape[0]) if observations[0][g] is not None]
        if not present:
            return a
        rows = np.asarray(present)

        if self.kind == "continuous":
            if self.noise_std > 0.0:
                a[rows] += self._rng.normal(0.0, self.noise_std, size=a[rows].shape)
            if self.epsilon > 0.0:
                swap = rows[self._rng.random(len(rows)) < self.epsilon]
                if swap.size:
                    a[swap] = self._rng.uniform(_LO, _HI, size=(swap.size, a.shape[1]))
            a[rows] = np.clip(a[rows], _LO, _HI)
        elif self.kind == "hybrid":
            # Layout: [continuous | one index column per discrete branch]. Gaussian
            # perturbs the continuous half (re-clipped); epsilon swaps the whole
            # action for a uniform-random one — continuous ~ U[-1, 1] and each
            # branch ~ U{0..size-1} — so epsilon=1 is a random hybrid policy.
            cont = self.act_dim - len(self.branches)
            if self.noise_std > 0.0:
                a[rows, :cont] += self._rng.normal(0.0, self.noise_std, size=(len(rows), cont))
            if self.epsilon > 0.0:
                swap = rows[self._rng.random(len(rows)) < self.epsilon]
                if swap.size:
                    a[swap, :cont] = self._rng.uniform(_LO, _HI, size=(swap.size, cont))
                    for j, size in enumerate(self.branches):
                        a[swap, cont + j] = self._rng.integers(size, size=swap.size)
            a[rows, :cont] = np.clip(a[rows, :cont], _LO, _HI)
        else:  # discrete: Gaussian is meaningless; epsilon resamples branches
            if self.epsilon > 0.0:
                for g in present:
                    for b, size in enumerate(self.branches):
                        if self._rng.random() < self.epsilon:
                            a[g, b] = self._rng.integers(size)
        return a
