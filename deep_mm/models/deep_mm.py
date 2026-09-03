from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from deep_mm.solvers.mmcf import (
    gamma_from_mmcf,
    make_hermitian,
    mmcf_direction,
    mmcf_terms,
    _project_trace,
)


class DeepMM(nn.Module):
    """Graph-based projected MM-CF unfolding for direct fixed channels."""

    model_kind = "deep_mm"
    node_feature_dim = 4
    edge_feature_dim = 8

    def __init__(
        self,
        layers: int = 16,
        hidden_size: int = 48,
        embed_dim: int = 24,
        learn_mode: str = "direct",
        scale_min: float = 0.25,
        scale_max: float = 4.0,
        nt_norm: float = 16.0,
        message_passing_steps: int = 1,
    ) -> None:
        super().__init__()
        if layers <= 0 or hidden_size <= 0 or embed_dim <= 0:
            raise ValueError("layers, hidden_size, and embed_dim must be positive")
        if learn_mode not in ("direct", "offset", "scale", "net_direct"):
            raise ValueError("learn_mode must be one of ('direct', 'offset', 'scale', 'net_direct')")
        if scale_min <= 0 or scale_max < scale_min:
            raise ValueError("scale bounds must be positive and ordered")
        if message_passing_steps <= 0:
            raise ValueError("message_passing_steps must be positive")
        self.layers = int(layers)
        self.hidden_size = int(hidden_size)
        self.embed_dim = int(embed_dim)
        self.learn_mode = str(learn_mode)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.message_passing_steps = int(message_passing_steps)
        self.nt_norm = float(nt_norm)
        if self.learn_mode != "net_direct":
            self.u = nn.Parameter(torch.zeros(self.layers))
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_feature_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, embed_dim), nn.ReLU(),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_feature_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, embed_dim), nn.ReLU(),
        )
        self.message_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(3 * embed_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, embed_dim), nn.ReLU(),
            ) for _ in range(message_passing_steps)
        ])
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * embed_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, embed_dim), nn.ReLU(),
            ) for _ in range(message_passing_steps)
        ])
        self.step_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * embed_dim, hidden_size), nn.ReLU(),
                nn.Linear(hidden_size, 1),
            ) for _ in range(layers)
        ])

    @staticmethod
    def _as_batch(
        H: torch.Tensor, G: torch.Tensor, Q0: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool]:
        if H.dim() not in (2, 3) or G.dim() != H.dim():
            raise ValueError("H and G must both have rank 2 or both have rank 3")
        single = H.dim() == 2
        Hb, Gb = (H.unsqueeze(0) if single else H), (G.unsqueeze(0) if single else G)
        if Hb.shape[0] != Gb.shape[0] or Hb.shape[2] != Gb.shape[2]:
            raise ValueError("H and G batch/transmit dimensions must match")
        Qb = None if Q0 is None else (Q0.unsqueeze(0) if single else Q0)
        if Qb is not None and (Qb.shape[0] != Hb.shape[0] or Qb.shape[-1] != Hb.shape[-1]):
            raise ValueError("Q0 is incompatible with H")
        return Hb, Gb, Qb, single

    def node_features(
        self, Q: torch.Tensor, H: torch.Tensor, G: torch.Tensor,
        Q_bar: torch.Tensor, Pt: float,
    ) -> torch.Tensor:
        single = Q.dim() == 2
        if single:
            Q, H, G, Q_bar = (Q.unsqueeze(0), H.unsqueeze(0), G.unsqueeze(0), Q_bar.unsqueeze(0))
        dtype = torch.float32
        h_norm = torch.linalg.vector_norm(H, dim=1).to(dtype)
        g_norm = torch.linalg.vector_norm(G, dim=1).to(dtype)
        q_bar_diag = torch.diagonal(Q_bar, dim1=-2, dim2=-1).real.to(dtype)
        q_diag = torch.diagonal(Q, dim1=-2, dim2=-1).real.to(dtype)
        features = torch.stack([h_norm, g_norm, q_bar_diag, q_diag], dim=-1)
        return features.squeeze(0) if single else features

    def edge_features(
        self, Q: torch.Tensor, Q_bar: torch.Tensor, H: torch.Tensor,
        G: torch.Tensor, Pt: float,
    ) -> torch.Tensor:
        single = Q.dim() == 2
        if single:
            Q, Q_bar, H, G = (Q.unsqueeze(0), Q_bar.unsqueeze(0), H.unsqueeze(0), G.unsqueeze(0))
        dtype = torch.float32
        H_gram = H.conj().transpose(-2, -1) @ H
        G_gram = G.conj().transpose(-2, -1) @ G
        features = torch.stack([
            H_gram.real.to(dtype), H_gram.imag.to(dtype),
            G_gram.real.to(dtype), G_gram.imag.to(dtype),
            Q_bar.real.to(dtype), Q_bar.imag.to(dtype),
            Q.real.to(dtype), Q.imag.to(dtype),
        ], dim=-1)
        return features.squeeze(0) if single else features

    def graph_node_embeddings(
        self, Q: torch.Tensor, Q_bar: torch.Tensor, H: torch.Tensor,
        G: torch.Tensor, Pt: float,
    ) -> torch.Tensor:
        single = Q.dim() == 2
        nodes = self.node_encoder(self.node_features(Q, H, G, Q_bar, Pt))
        edges = self.edge_encoder(self.edge_features(Q, Q_bar, H, G, Pt))
        if single:
            nodes, edges = nodes.unsqueeze(0), edges.unsqueeze(0)
        batch, nt = nodes.shape[:2]
        for message_mlp, update_mlp in zip(self.message_mlps, self.update_mlps):
            dst = nodes.view(batch, nt, 1, self.embed_dim).expand(batch, nt, nt, self.embed_dim)
            src = nodes.view(batch, 1, nt, self.embed_dim).expand(batch, nt, nt, self.embed_dim)
            messages = message_mlp(torch.cat([dst, src, edges], dim=-1)).mean(dim=2)
            nodes = update_mlp(torch.cat([nodes, messages], dim=-1))
        return nodes.squeeze(0) if single else nodes

    def pooled_graph_embedding(self, *args) -> torch.Tensor:
        single = args[0].dim() == 2
        nodes = self.graph_node_embeddings(*args)
        if single:
            nodes = nodes.unsqueeze(0)
        pooled = torch.cat([nodes.mean(dim=1), nodes.max(dim=1).values], dim=-1)
        return pooled.squeeze(0) if single else pooled

    def scalar_gammas(self, H: torch.Tensor, Pt: float) -> torch.Tensor:
        """Return the legacy scalar MM-CF step sizes before learned scaling."""
        if self.learn_mode == "net_direct":
            raise ValueError("scalar_gammas is not used when learn_mode='net_direct'")
        single = H.dim() == 2
        H_batch = H.unsqueeze(0) if single else H
        base = gamma_from_mmcf(H_batch, Pt).to(dtype=torch.float32)
        positive = torch.nn.functional.softplus(self.u).to(device=H.device) + 1e-12
        if self.learn_mode == "direct":
            values = positive.view(1, -1).expand(H_batch.shape[0], -1)
        elif self.learn_mode == "offset":
            values = base.view(-1, 1) + positive.view(1, -1)
        else:
            values = base.view(-1, 1) * (1.0 + positive.view(1, -1))
        return values.squeeze(0) if single else values

    def forward(
        self, H: torch.Tensor, G: torch.Tensor, Pt: float,
        Q0: Optional[torch.Tensor] = None,
        inference_layers: Optional[int] = None,
        return_records: bool = False,
    ):
        depth = self.layers if inference_layers is None else int(inference_layers)
        if depth < 0 or depth > self.layers:
            raise ValueError("inference_layers must be in [0, layers]")
        Hb, Gb, Qb, single = self._as_batch(H, G, Q0)
        batch, _, nt = Hb.shape
        if Qb is None:
            Q = (float(Pt) / nt) * torch.eye(nt, dtype=Hb.dtype, device=Hb.device).unsqueeze(0).expand(batch, nt, nt).clone()
        else:
            Q = Qb
        Q = make_hermitian(Q)
        scalar_gammas = None if self.learn_mode == "net_direct" else self.scalar_gammas(Hb, Pt)
        gammas, records = [], [Q]
        for layer in range(depth):
            H_bar, G_bar = mmcf_terms(Q, Hb, Gb)
            Q_bar = make_hermitian(H_bar - G_bar)
            pooled = self.pooled_graph_embedding(Q, Q_bar, Hb, Gb, Pt)
            raw_gamma = self.step_heads[layer](pooled).squeeze(-1)
            if self.learn_mode == "net_direct":
                gamma = torch.nn.functional.softplus(raw_gamma) + 1e-8
            else:
                scale = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(raw_gamma)
                gamma = scalar_gammas[:, layer] * scale
            _F, update_direction = mmcf_direction(Q, H_bar, G_bar, gamma)
            Q = make_hermitian(_project_trace(Q + update_direction / (2.0 * gamma.view(-1, 1, 1).to(update_direction.dtype) + 1e-12), Pt))
            gammas.append(gamma)
            records.append(Q)
        gamma_tensor = (
            torch.stack(gammas, dim=1)
            if gammas
            else torch.empty(batch, 0, dtype=torch.float32, device=Hb.device)
        )
        if single:
            Q, gamma_tensor = Q.squeeze(0), gamma_tensor.squeeze(0)
            records = [item.squeeze(0) for item in records]
        if return_records:
            return Q, gamma_tensor, records
        return Q, gamma_tensor


DeepMMNet = DeepMM

