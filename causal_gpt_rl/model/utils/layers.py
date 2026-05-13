"""Lightweight layer building blocks used by AutoregressiveModel.

Inference-only utilities — no parameter initialization or RNG seeding (those
live in `tools/init_utils.py` since they are training-side concerns).

Author:
    PARK, Jun-Ho, junho@ccnets.org

Copyright (c) 2026 CCNets, Inc. All rights reserved.
"""

from torch import nn


ACTIVATION_FUNCTIONS = {
    "softmax": nn.Softmax(dim=-1),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
}


def get_activation_function(activation_function, feature_size=None):
    """Returns the appropriate activation function layer."""
    activation_function = activation_function.lower()
    if activation_function in ["none", "linear"]:
        return nn.Identity()
    if activation_function == "layer_norm" and feature_size is not None:
        # For input projection, we disable affine parameters to keep LayerNorm
        # purely as a normalization step. This avoids re-scaling or shifting
        # state/action features after external normalization (e.g., RMS or
        # pre-tanh transform), and prevents the model from learning spurious
        # feature-wise coupling at the input stage.
        return nn.LayerNorm(feature_size, elementwise_affine=False)
    if activation_function in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[activation_function]
    raise ValueError(f"Unsupported activation function: {activation_function}")


class TransformLayer(nn.Module):
    """Linear layer with optional pre/post activation."""

    def __init__(self, input_size, output_size, input_act_fn="none", output_act_fn="none"):
        super().__init__()
        layers = []

        input_activation_layer = get_activation_function(input_act_fn, input_size)
        if not isinstance(input_activation_layer, nn.Identity):
            layers.append(input_activation_layer)

        layers.append(nn.Linear(input_size, output_size))

        output_activation_layer = get_activation_function(output_act_fn, output_size)
        if not isinstance(output_activation_layer, nn.Identity):
            layers.append(output_activation_layer)

        self.layers = nn.Sequential(*layers)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, features):
        return self.layers(features)
