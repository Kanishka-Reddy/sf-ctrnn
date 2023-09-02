import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np


class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class CfCCellSlowFastOnline(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size_slow,
            hidden_size_fast,
            learning_rate=0.1,
            mode="default",
            backbone_activation="lecun_tanh",
            backbone_units=128,
            backbone_layers=1,
            backbone_dropout=0.0,
            sparsity_mask: Optional[np.array] = None
    ):
        super(CfCCellSlowFastOnline, self).__init__()

        self.input_size = input_size
        self.hidden_size_slow = hidden_size_slow
        self.hidden_size_fast = hidden_size_fast
        self.learning_rate = learning_rate
        self.mode = "default"

        # Sparsity Mask
        self.sparsity_mask = (
            None
            if sparsity_mask is None
            else torch.nn.Parameter(
                data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False,
            )
        )

        # Activation function selection
        if backbone_activation == "lecun_tanh":
            self.activation = LeCun()
        elif backbone_activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = LeCun()
#           raise ValueError(f"Unknown activation {backbone_activation}")

        # Backbone layers
        self.backbone = None
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size_slow + hidden_size_fast, backbone_units),
                self.activation
            ]
            for _ in range(1, backbone_layers):
                layer_list.extend([
                    nn.Linear(backbone_units, backbone_units),
                    self.activation
                ])
                if backbone_dropout > 0.0:
                    layer_list.append(nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Slow dynamics
        self.ff1_slow = nn.Linear(input_size + hidden_size_slow, hidden_size_slow)
        self.ff2_slow = nn.Linear(input_size + hidden_size_slow, hidden_size_slow)

        # Fast dynamics
        self.ff1_fast = nn.Linear(input_size + hidden_size_fast + hidden_size_slow, hidden_size_fast)
        self.ff2_fast = nn.Linear(input_size + hidden_size_fast + hidden_size_slow, hidden_size_fast)

        # Time-based interpolation
        self.time_a = nn.Linear(input_size + hidden_size_slow + hidden_size_fast, 1)
        self.time_b = nn.Linear(input_size + hidden_size_slow + hidden_size_fast, 1)

        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx_slow, hx_fast, ts):
        # If backbone layers are used
        if self.backbone is not None:
            x_backbone = torch.cat([input, hx_slow, hx_fast], 1)
            x_backbone = self.backbone(x_backbone)
        else:
            x_backbone = torch.cat([input, hx_slow, hx_fast], 1)

        # Slow dynamics
        x_slow = torch.cat([input, hx_slow], 1)
        ff1_slow = self.ff1_slow(x_slow)
        ff2_slow = self.ff2_slow(x_slow)
        ff1_slow = self.tanh(ff1_slow)
        ff2_slow = self.tanh(ff2_slow)

        # Fast dynamics
        x_fast = torch.cat([input, hx_fast, hx_slow], 1)
        ff1_fast = self.ff1_fast(x_fast)
        ff2_fast = self.ff2_fast(x_fast)
        ff1_fast = self.tanh(ff1_fast)
        ff2_fast = self.tanh(ff2_fast)

        # Time-based interpolation
        x_time = torch.cat([input, hx_slow, hx_fast], 1)
        t_a = self.time_a(x_time)
        t_b = self.time_b(x_time)
        t_interp = self.sigmoid(t_a * ts + t_b)

        # New hidden states
        new_hidden_slow = ff1_slow * (1.0 - t_interp) + t_interp * ff2_slow
        new_hidden_fast = ff1_fast * (1.0 - t_interp) + t_interp * ff2_fast

        # Hebbian learning for fast dynamics
        for param in [self.ff1_fast.weight, self.ff2_fast.weight]:
            hebbian_update = torch.mm(new_hidden_fast.t(), x_fast)
            param.data += self.learning_rate * hebbian_update.data

        output = new_hidden_slow + new_hidden_fast

        return output, (new_hidden_slow, new_hidden_fast)
