import json
import tempfile
import unittest
from pathlib import Path

import torch

from causal_gpt_rl.inference import bundle
from causal_gpt_rl.model.autoregressive_model import AutoregressiveModel
from causal_gpt_rl.model.schema import ModelConfig, SpaceSpec


class TestRuntimeUpdates(unittest.TestCase):
    def test_model_config_defaults(self):
        config = ModelConfig()

        self.assertEqual(config.num_layers, 4)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.num_heads, 8)
        self.assertEqual(config.dropout, 0.05)

    def test_continuous_sampling_std_scale_zero_uses_mean_action(self):
        model = AutoregressiveModel(
            ModelConfig(d_model=32, num_heads=4),
            state_specs=[
                SpaceSpec(
                    type="continuous",
                    size=2,
                    dtype=torch.float32,
                    low=[-1.0, -1.0],
                    high=[1.0, 1.0],
                )
            ],
            action_specs=[
                SpaceSpec(
                    type="continuous",
                    size=2,
                    dtype=torch.float32,
                    low=[-2.0, -2.0],
                    high=[2.0, 2.0],
                    squash="tanh",
                )
            ],
            device=torch.device("cpu"),
        )
        heads = [
            torch.zeros(1, 2),
            torch.full((1, 2), 100.0),
            torch.zeros(1, 1),
        ]

        action = model.sample_action_from_heads(heads, std_scale=0.0)

        self.assertTrue(torch.allclose(action, torch.zeros_like(action)))

    def test_export_bundle_writes_optional_env_id(self):
        model = AutoregressiveModel(
            ModelConfig(d_model=32, num_heads=4),
            state_specs=[
                SpaceSpec(
                    type="continuous",
                    size=2,
                    dtype=torch.float32,
                    low=[-1.0, -1.0],
                    high=[1.0, 1.0],
                )
            ],
            action_specs=[
                SpaceSpec(
                    type="continuous",
                    size=2,
                    dtype=torch.float32,
                    low=[-1.0, -1.0],
                    high=[1.0, 1.0],
                    squash="tanh",
                )
            ],
            device=torch.device("cpu"),
        )

        class Normalizer:
            def state_dict(self):
                return {"mean": torch.zeros(2), "std": torch.ones(2)}

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle.export_bundle(
                tmpdir,
                model=model,
                model_config=ModelConfig(d_model=32, num_heads=4),
                state_specs=model.state_specs,
                action_specs=model.action_specs,
                context_length=8,
                state_normalizer=Normalizer(),
                env_id="HalfCheetah-v5",
            )

            payload = json.loads((Path(tmpdir) / "config.json").read_text())

        self.assertEqual(payload["env_id"], "HalfCheetah-v5")


if __name__ == "__main__":
    unittest.main()
