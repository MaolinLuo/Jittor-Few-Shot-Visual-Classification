import math
from operator import mul
from functools import reduce

import jittor as jt
from jittor import nn, init


class AdaptFormer(nn.Module):
    def __init__(self, in_dim, bottle_dim, dtype=None):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.down_proj = nn.Linear(in_dim, bottle_dim)
        self.relu = nn.Relu()
        self.up_proj = nn.Linear(bottle_dim, in_dim)
        self.scale = nn.Parameter(jt.ones([1], dtype=dtype))

        init.kaiming_normal_(self.down_proj.weight, a=math.sqrt(5))
        init.zero_(self.up_proj.weight)
        init.zero_(self.down_proj.bias)
        init.zero_(self.up_proj.bias)

    @property
    def dtype(self):
        return self.ln.weight.dtype

    def execute(self, x):
        x = self.ln(x)
        x = self.down_proj(x)
        x = self.relu(x)
        x = self.up_proj(x)
        x = x * self.scale
        return x

