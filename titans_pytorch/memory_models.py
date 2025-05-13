# FILE: memory_models.py

import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList

from einops import rearrange

# functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# norms

class LayerNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')

        return self.ln(x) * (gamma + 1.)

# norm + residual wrapper, as used in original TTT paper
# but could be removed

class ResidualNorm(Module):
    def __init__(
        self,
        dim,
        model: Module
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.model = model

    def forward(self, x):

        out = self.model(x)

        return self.norm(out) + x

# memory mlp proposed in TTT

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 2.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        # Determine dimensions for each layer
        dims_list = []
        if depth == 1:
            dims_list = [dim, dim]
        elif depth > 1:
            dims_list = [dim] + [dim_hidden] * (depth - 1) + [dim]
        else:
            raise ValueError("Depth must be at least 1")

        self.weights = ParameterList()
        for dim_in, dim_out in zip(dims_list[:-1], dims_list[1:]):
            weight_matrix = Parameter(torch.empty(dim_in, dim_out)) # Use empty and init separately
            if dim_in == dim_out: # If it's a square matrix, aim for identity-like
                nn.init.eye_(weight_matrix)
                # Optionally add small noise:
                # weight_matrix.data.add_(torch.randn_like(weight_matrix) * 0.01)
            else: # For non-square (expansion/contraction), use Xavier
                nn.init.xavier_uniform_(weight_matrix)
            self.weights.append(weight_matrix)

    def forward(
        self,
        x
    ):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x) # Apply activation before non-first layers

            x = x @ weight

        return x

# memory mlp, but with gated residual + final projection

class GatedResidualMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        self.final_proj = Parameter(torch.randn(dim, dim))

        # Initialize weights using Xavier Uniform
        for param_list in self.weights:
            nn.init.xavier_uniform_(param_list[0]) # weight1
            nn.init.xavier_uniform_(param_list[1]) # weight2
            nn.init.xavier_uniform_(param_list[2]) # to_gates
        nn.init.xavier_uniform_(self.final_proj)


    def forward(
        self,
        x
    ):

        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates_input = cat((branch_out, res), dim = -1)
            gates = gates_input @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj

# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes

class FactorizedMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        k = 32
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x
    ):

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return x

# an MLP modelled after the popular swiglu ff in modern transformers

class MemorySwiGluMLP(Module):
    def __init__(
        self,
        dim,
        depth = 1, # default to 2 layer MLP from TTT, depth of 2 would be 4 layer MLP, but done as 2 feedforwards with residual
        expansion_factor = 4.
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        weights = []

        for _ in range(depth):
            # Define weights for one SwiGLU block
            w1 = Parameter(torch.randn(dim, dim_inner * 2)) # Combined gate and value projection
            w2 = Parameter(torch.randn(dim_inner, dim))     # Down projection
            nn.init.xavier_uniform_(w1)
            nn.init.xavier_uniform_(w2)
            weights.append(ParameterList([w1, w2]))

        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim) # Final layer norm

    def forward(self, x):

        for w1, w2 in self.weights:
            residual = x

            hidden_states = x @ w1
            # Split into gate and value
            x_val, gates = hidden_states.chunk(2, dim = -1)

            # Apply SiLU activation (equivalent to Swish/SiLU GLU)
            x = F.silu(gates) * x_val

            # Project down
            x = x @ w2

            # Add residual
            x = x + residual

        # Apply final norm
        return self.norm(x)

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            Parameter(torch.randn(dim, dim)), # queries
            Parameter(torch.randn(dim, dim)), # keys
            Parameter(torch.randn(dim, dim)), # values
            Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv

        # Assuming causal attention is desired for memory update context
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )

        # parallel attention + feedforward block
        # as in PaLM + Gpt-J

        h = F.gelu(x @ ffw1) # Use GELU as in original
        ff_out = h @ ffw2

        return attn_out + ff_out
