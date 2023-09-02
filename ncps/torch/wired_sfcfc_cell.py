# Copyright 2022 Mathias Lechner. All rights reserved

import numpy as np
import torch
from torch import nn
from typing import Tuple, Optional, Union

from . import CfCCellSlowFastOnline
from . import CfCCell
from typing import Optional, Union


class WiredCfCCellSlowFastOnline(nn.Module):
    def __init__(
        self,
        input_size,
        wiring,
        mode="default",
    ):
        super(WiredCfCCellSlowFastOnline, self).__init__()

        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'input_size' or call the 'wiring.build()'."
            )
        self._wiring = wiring

        self._layers = []
        in_features = wiring.input_dim
        for l in range(wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_layer_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_layer_neurons, :]
            input_sparsity = np.concatenate(
                [
                    input_sparsity,
                    np.ones((len(hidden_units), len(hidden_units))),
                ],
                axis=0,
            )

            # Hack: nn.Module registers child params in set_attribute
            rnn_cell = CfCCellSlowFastOnline(
                in_features,
                len(hidden_units) // 2,  # Assuming half for slow hidden states, adjust as needed
                len(hidden_units) // 2,
                mode=mode,
                backbone_activation="lecun_tanh",
                backbone_units=0,
                backbone_layers=0,
                backbone_dropout=0.0,
                sparsity_mask=input_sparsity,
            )
            self.register_module(f"layer_{l}", rnn_cell)
            self._layers.append(rnn_cell)
            in_features = len(hidden_units)

    @property
    def state_size(self) -> Tuple[int, int]:
        slow_units = sum(len(self._wiring.get_neurons_of_layer(i)) // 2 for i in range(self._wiring.num_layers))
        fast_units = slow_units  # Assuming an equal number of fast and slow units
        return slow_units, fast_units

    @property
    def layer_sizes(self):
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    @property
    def synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    @property
    def sensory_synapse_count(self):
        return np.sum(np.abs(self._wiring.adjacency_matrix))

    def forward(self, input, hx: Tuple[torch.Tensor, torch.Tensor], timespans):
        h_state_slow, h_state_fast = hx
        h_state_slow = torch.split(h_state_slow, self.layer_sizes, dim=1)
        h_state_fast = torch.split(h_state_fast, self.layer_sizes, dim=1)

        new_h_state_slow = []
        new_h_state_fast = []

        inputs = input
        for i in range(self.num_layers):
            h, (new_h_slow, new_h_fast) = self._layers[i].forward(inputs, h_state_slow[i], h_state_fast[i], timespans)
            inputs = h
            new_h_state_slow.append(new_h_slow)
            new_h_state_fast.append(new_h_fast)

        new_h_state_slow = torch.cat(new_h_state_slow, dim=1)
        new_h_state_fast = torch.cat(new_h_state_fast, dim=1)

        return h, (new_h_state_slow, new_h_state_fast)