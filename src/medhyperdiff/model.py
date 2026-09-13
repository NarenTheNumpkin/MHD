"""Small feature transformer with explicit nonlinear incidence message passing."""
import math
from contextlib import nullcontext

import torch
from torch import nn


class HypergraphBlock(nn.Module):
    def __init__(self, width, feature_count, edges, gated=False, dense=False):
        super().__init__()
        incidence = torch.zeros(feature_count, len(edges))
        for j, edge in enumerate(edges):
            incidence[edge, j] = 1
        if dense:
            incidence.fill_(1)  # Same edge slots and learned parameters; remove clinical membership.
        self.register_buffer("incidence", incidence)
        self.residual_gate = nn.Parameter(torch.zeros(())) if gated else None
        self.edge_identity = nn.Parameter(torch.randn(len(edges), width) * .02)
        self.to_edge = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.edge_update = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.to_node = nn.Sequential(nn.Linear(width * 2, width), nn.SiLU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def forward(self, tokens):
        incidence = self.incidence
        edges = torch.einsum("de,bdh->beh", incidence, self.to_edge(tokens))
        edges = edges / incidence.sum(0).clamp(min=1)[None, :, None]
        edges = self.edge_update(edges + self.edge_identity)
        message = torch.einsum("de,beh->bdh", incidence, edges)
        degree = incidence.sum(1)
        message = message / degree.clamp(min=1)[None, :, None]
        update = self.to_node(torch.cat([tokens, message], dim=-1))
        # Isolated nodes must not receive an unrelated hypergraph residual.
        scale = self.residual_gate.tanh() if self.residual_gate is not None else 1.
        return tokens + scale * (degree > 0)[None, :, None] * self.norm(update)


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True, dropout=0.)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 2), nn.SiLU(), nn.Linear(width * 2, width))

    def forward(self, tokens, bias):
        normed = self.norm1(tokens)
        update, _ = self.attention(normed, normed, normed, attn_mask=bias, need_weights=False)
        tokens = tokens + update
        return tokens + self.ff(self.norm2(tokens))


class Denoiser(nn.Module):
    def __init__(self, n_numeric, cardinalities, graph, edges, allowed, width=48, layers=2, heads=4,
                 gated_structure=False, dense_structure=False, paired_initialization=False):
        super().__init__()
        if width % heads:
            raise ValueError("Width must be divisible by attention heads")
        self.n_numeric = n_numeric
        self.cardinalities = cardinalities
        self.width = width
        d = n_numeric + len(cardinalities)
        self.numeric_weight = nn.Parameter(torch.randn(n_numeric, width) * .05)
        self.numeric_bias = nn.Parameter(torch.zeros(n_numeric, width))
        self.category_embeddings = nn.ModuleList(nn.Embedding(k + 1, width) for k in cardinalities)
        self.feature_identity = nn.Parameter(torch.randn(d, width) * .02)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.label_embedding = nn.Embedding(2, width)
        self.register_buffer("graph", torch.as_tensor(graph, dtype=torch.float32))
        self.graph_strength = nn.Parameter(torch.tensor(0. if gated_structure else 1.))
        self.blocks = nn.ModuleList(AttentionBlock(width, heads) for _ in range(layers))
        # Keep common output heads identical across ablations with the same seed.
        # Modules are initialized on CPU before the complete model moves to its device.
        with torch.random.fork_rng(devices=[]) if paired_initialization else nullcontext():
            self.hyper_blocks = nn.ModuleList(HypergraphBlock(width, d, edges, gated_structure,
                dense_structure) for _ in range(layers)) if edges else None
        self.output_norm = nn.LayerNorm(width)
        self.numeric_heads = nn.ModuleList(nn.Linear(width, 1) for _ in range(n_numeric))
        self.category_heads = nn.ModuleList(nn.Linear(width, k) for k in cardinalities)
        for i, mask in enumerate(allowed):
            self.register_buffer(f"allowed_{i}", torch.as_tensor(mask, dtype=torch.bool))

    def forward(self, numeric, categorical, time_fraction, labels):
        tokens = [numeric[:, i, None] * self.numeric_weight[i] + self.numeric_bias[i] for i in range(self.n_numeric)]
        tokens += [emb(categorical[:, i]) for i, emb in enumerate(self.category_embeddings)]
        tokens = torch.stack(tokens, dim=1) + self.feature_identity
        freq = torch.exp(torch.arange(self.width // 2, device=numeric.device) * (-math.log(10000) / max(self.width // 2 - 1, 1)))
        phase = time_fraction[:, None] * 1000 * freq
        t_embed = torch.cat([phase.sin(), phase.cos()], dim=-1)
        tokens = tokens + (self.time_mlp(t_embed) + self.label_embedding(labels))[:, None]
        bias = self.graph_strength * self.graph
        for i, block in enumerate(self.blocks):
            if self.hyper_blocks is not None:
                tokens = self.hyper_blocks[i](tokens)
            tokens = block(tokens, bias)
        tokens = self.output_norm(tokens)
        numeric_out = torch.cat([head(tokens[:, i]) for i, head in enumerate(self.numeric_heads)], dim=1) if self.n_numeric else numeric
        logits = [head(tokens[:, self.n_numeric + i]).masked_fill(~getattr(self, f"allowed_{i}"), -1e9)
                  for i, head in enumerate(self.category_heads)]
        return numeric_out, logits
