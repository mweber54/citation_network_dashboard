#!/usr/bin/env python
"""
Train THGNN to predict multi-signal topic residuals (u, m, a) from a temporal keyword–paper hypergraph.

Matches your build_hypergraph.py hypergraph.pt schema:
  - hyperedge_index
  - node_features
  - hyperedge_features
  - node_labels
  - year_per_edge
  - edge_id_to_paper (optional but present in your builder)

Signals:
  u_{c,t}: topic usage mass (weighted sum of keyword paper-frequency)
  m_{c,t}: topic citation mass (weighted sum of keyword citation mass)
  a_{c,t}: citation intensity = m / (u + eps)

Baselines:
  rolling curve fit on Z=log(1+signal): logistic / exp / power, choose by MSE.

THGNN:
  HypergraphConv over windowed sub-hypergraph, pool keywords -> topics, predict residuals (u,m,a).

"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG
from torch_geometric.nn import HypergraphConv

# -----------------------------
# Baseline curve families
# -----------------------------
_CURVE_BASIS_CACHE: Dict[Tuple[str, Tuple[float, ...]], Dict[str, np.ndarray]] = {}


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -60.0, 60.0))


def g_exp(tau, alpha, beta, gamma):
    tau = np.asarray(tau, dtype=np.float64)
    return alpha + beta * _safe_exp(gamma * tau)


def g_power(tau, alpha, beta, gamma):
    tau = np.asarray(tau, dtype=np.float64)
    return alpha + beta * np.power(tau + 1.0, gamma)


def g_logistic(tau, K, r, tau0, alpha):
    tau = np.asarray(tau, dtype=np.float64)
    return alpha + K / (1.0 + _safe_exp(-r * (tau - tau0)))


def _get_curve_basis_specs(tau: np.ndarray, family: str) -> Dict[str, np.ndarray]:
    key = (family, tuple(np.asarray(tau, dtype=np.float64).tolist()))
    cached = _CURVE_BASIS_CACHE.get(key)
    if cached is not None:
        return cached

    tau = np.asarray(tau, dtype=np.float64)

    if family == "exp":
        gamma = np.linspace(-2.0, 2.0, 121, dtype=np.float64)
        basis = _safe_exp(gamma[:, None] * tau[None, :])
        spec = {"basis": basis, "gamma": gamma}
    elif family == "power":
        gamma = np.linspace(-2.0, 5.0, 141, dtype=np.float64)
        basis = np.power(tau[None, :] + 1.0, gamma[:, None])
        spec = {"basis": basis, "gamma": gamma}
    elif family == "logistic":
        r = np.linspace(-5.0, 5.0, 61, dtype=np.float64)
        tau0 = np.linspace(-10.0, 10.0, 61, dtype=np.float64)
        r_grid, tau0_grid = np.meshgrid(r, tau0, indexing="ij")
        logits = r_grid[..., None] * (tau[None, None, :] - tau0_grid[..., None])
        basis = 1.0 / (1.0 + _safe_exp(-logits))
        spec = {
            "basis": basis.reshape(-1, tau.size),
            "r": r_grid.reshape(-1),
            "tau0": tau0_grid.reshape(-1),
        }
    else:
        raise ValueError(f"Unknown family: {family}")

    _CURVE_BASIS_CACHE[key] = spec
    return spec


def _fit_affine_basis_grid(y: np.ndarray, basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)

    y_mean = float(np.mean(y))
    b_mean = np.mean(basis, axis=1)
    b_centered = basis - b_mean[:, None]
    y_centered = y - y_mean

    denom = np.sum(b_centered * b_centered, axis=1)
    # Keep this elementwise to avoid BLAS/MKL dispatch on Windows.
    numer = np.sum(b_centered * y_centered[None, :], axis=1)

    scale = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 1e-12)
    scale = np.maximum(scale, 0.0)
    alpha = y_mean - scale * b_mean

    pred = alpha[:, None] + scale[:, None] * basis
    mse = np.mean((pred - y[None, :]) ** 2, axis=1)
    return alpha, scale, mse


def fit_curve_family(tau: np.ndarray, y: np.ndarray, family: str):
    spec = _get_curve_basis_specs(tau, family)
    alpha, scale, mse = _fit_affine_basis_grid(y, spec["basis"])
    best = int(np.argmin(mse))

    if family in {"exp", "power"}:
        params = np.array([alpha[best], scale[best], spec["gamma"][best]], dtype=np.float64)
    elif family == "logistic":
        params = np.array([scale[best], spec["r"][best], spec["tau0"][best], alpha[best]], dtype=np.float64)
    else:
        raise ValueError(f"Unknown family: {family}")

    return params, float(mse[best])

def predict_curve_family(tau: np.ndarray, family: str, params: np.ndarray) -> np.ndarray:
    if family == "exp" and len(params) > 1:
        return g_exp(tau, *params)
    if family == "power" and len(params) > 1:
        return g_power(tau, *params)
    if family == "logistic" and len(params) > 1:
        return g_logistic(tau, *params)
    # mean fallback
    return np.full_like(tau, float(params[0]), dtype=float)

# -----------------------------
# THGNN
# -----------------------------
class THGNN(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int, out_dim: int, layers: int, dropout: float):
        super().__init__()
        self.node_proj = nn.Linear(node_in, hidden)
        self.edge_proj = nn.Linear(edge_in, hidden)
        self.convs = nn.ModuleList([HypergraphConv(hidden, hidden) for _ in range(layers)])
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        x_node: torch.Tensor,             # [N, Fn]
        hyperedge_index: torch.Tensor,    # [2, Iw]
        x_edge: torch.Tensor,             # [Ew, Fe]
        node_labels: torch.Tensor,        # [N] topic id
        C: int
    ) -> torch.Tensor:
        h = self.node_proj(x_node)
        he = self.edge_proj(x_edge)

        for conv in self.convs:
            h = conv(h, hyperedge_index, hyperedge_attr=he)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # pool keyword->topic by mean
        topic_emb = torch.zeros((C, h.size(1)), device=h.device, dtype=h.dtype)
        counts = torch.zeros((C,), device=h.device, dtype=h.dtype)

        t = node_labels.long()
        topic_emb.index_add_(0, t, h)
        counts.index_add_(0, t, torch.ones((h.size(0),), device=h.device, dtype=h.dtype))
        topic_emb = topic_emb / torch.clamp(counts.unsqueeze(1), min=1.0)

        return self.head(topic_emb)  # [C, 3]

# -----------------------------
# Data plumbing
# -----------------------------
def build_window_subhypergraph(
    hyperedge_index: torch.Tensor,   # [2, I]
    year_per_edge: torch.Tensor,     # [M] raw year
    year_min: int,
    year_max: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Filter incidences whose hyperedge year in [year_min, year_max],
    remap hyperedge ids to contiguous [0..Ew-1].

    Returns:
      sub_hyperedge_index [2, Iw]
      edge_map [Ew] original edge ids
    """
    node = hyperedge_index[0].long()
    edge = hyperedge_index[1].long()

    edge_year = year_per_edge[edge]
    mask = (edge_year >= year_min) & (edge_year <= year_max)

    node_w = node[mask]
    edge_w_orig = edge[mask]

    uniq_edges, inv = torch.unique(edge_w_orig, sorted=True, return_inverse=True)
    edge_w_new = inv

    sub = torch.stack([node_w, edge_w_new], dim=0)
    return sub, uniq_edges

def compute_keyword_year_u_m(
    hyperedge_index: torch.Tensor,        # [2, I]
    year_per_edge: torch.Tensor,          # [M] raw year
    paper_citations: torch.Tensor,        # [M] raw citations OR proxy (float)
    years: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keyword-year usage u_{k,t} and citation mass m_{k,t}.
    Assumes each (keyword,paper) incidence is unique (your builder may include duplicates if a phrase repeats).
    """
    node_idx = hyperedge_index[0].long().cpu().numpy()
    edge_idx = hyperedge_index[1].long().cpu().numpy()

    K = int(node_idx.max()) + 1
    year_arr = year_per_edge.long().cpu().numpy()
    cit_arr = paper_citations.float().cpu().numpy()

    year_to_col = {y: i for i, y in enumerate(years)}
    T = len(years)

    u = np.zeros((K, T), dtype=np.float32)
    m = np.zeros((K, T), dtype=np.float32)

    e_year = year_arr[edge_idx]
    e_cit = cit_arr[edge_idx]

    cols = np.array([year_to_col.get(int(y), -1) for y in e_year], dtype=np.int32)
    mask = cols >= 0

    k = node_idx[mask]
    cols = cols[mask]
    e_cit = e_cit[mask]

    np.add.at(u, (k, cols), 1.0)
    np.add.at(m, (k, cols), e_cit)
    return u, m

def compute_topic_aggregates(
    u_k_t: np.ndarray,
    m_k_t: np.ndarray,
    node_labels: np.ndarray,          # [N] topic id
    node_scores: np.ndarray,          # [N] score_norm (0..1) used for within-topic weights
    C: int,
    eps: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, T = u_k_t.shape
    topic = node_labels.astype(np.int64)

    score = np.maximum(node_scores.astype(np.float32), 0.0)
    denom = np.zeros((C,), dtype=np.float32)
    np.add.at(denom, topic, score)
    denom = np.maximum(denom, eps)
    w = score / denom[topic]  # within-topic normalized

    u_c_t = np.zeros((C, T), dtype=np.float32)
    m_c_t = np.zeros((C, T), dtype=np.float32)
    for k in range(N):
        c = topic[k]
        u_c_t[c] += w[k] * u_k_t[k]
        m_c_t[c] += w[k] * m_k_t[k]

    a_c_t = m_c_t / (u_c_t + eps)
    return u_c_t, m_c_t, a_c_t

def rolling_baseline_residuals(
    Z: np.ndarray,               # [C, T]
    years: List[int],
    L: int,
    families: Tuple[str, ...],
) -> np.ndarray:
    C, T = Z.shape
    resid = np.full((C, T), np.nan, dtype=np.float32)
    tau = np.arange(L, dtype=np.float64)

    for c in range(C):
        for ti in range(L, T):
            y_win = Z[c, ti - L: ti].astype(np.float64)
            best_mse = float("inf")
            best_fam = families[0]
            best_params = None

            for fam in families:
                params, mse = fit_curve_family(tau, y_win, fam)
                if mse < best_mse:
                    best_mse = mse
                    best_fam = fam
                    best_params = params

            yhat_next = float(predict_curve_family(np.array([L], dtype=np.float64), best_fam, best_params)[0])
            resid[c, ti] = float(Z[c, ti] - yhat_next)

    return resid

# -----------------------------
# load raw citations from papers.csv
# -----------------------------
def load_raw_citations_from_papers_csv(
    papers_csv: str,
    edge_id_to_paper: Dict[int, str],
    num_edges: int
) -> torch.Tensor:
    import pandas as pd
    df = pd.read_csv(papers_csv, encoding="utf-8")
    if "paper_id" not in df.columns or "citation_count" not in df.columns:
        raise ValueError("papers_csv must include columns: paper_id, citation_count")

    df["paper_id"] = df["paper_id"].astype(str)
    df["citation_count"] = pd.to_numeric(df["citation_count"], errors="coerce").fillna(0).astype(float)
    cit_map = dict(zip(df["paper_id"], df["citation_count"]))

    cits = np.zeros((num_edges,), dtype=np.float32)
    missing = 0
    for eid in range(num_edges):
        pid = str(edge_id_to_paper.get(eid, ""))
        if pid in cit_map:
            cits[eid] = float(cit_map[pid])
        else:
            missing += 1
            cits[eid] = 0.0

    if missing > 0:
        print(f"[warn] {missing}/{num_edges} edges missing citation_count in papers.csv; set to 0.")

    return torch.tensor(cits, dtype=torch.float32)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypergraph_pt", default="hypergraph.pt")
    ap.add_argument("--papers_csv", default=None, help="Optional: papers.csv to recover raw citation_count.")
    ap.add_argument("--train_year_min", type=int, default=2005)
    ap.add_argument("--train_year_max", type=int, default=2025)
    ap.add_argument("--gnn_window", type=int, default=5)
    ap.add_argument("--baseline_window", type=int, default=5)
    ap.add_argument("--families", nargs="+", default=["logistic", "exp", "power"])
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hg = torch.load(args.hypergraph_pt, map_location="cpu")

    required = ["hyperedge_index", "node_features", "hyperedge_features", "node_labels", "year_per_edge"]
    for k in required:
        if k not in hg:
            raise KeyError(f"hypergraph.pt missing key '{k}'. Keys: {sorted(list(hg.keys()))}")

    hyperedge_index = hg["hyperedge_index"].long()
    node_features = hg["node_features"].float()
    hyperedge_features = hg["hyperedge_features"].float()
    node_labels = hg["node_labels"].long()
    year_per_edge = hg["year_per_edge"].long()

    N = node_features.size(0)
    M = year_per_edge.numel()
    C = int(node_labels.max().item()) + 1

    # node score_norm lives in node_features[:,1] by your builder
    node_score = node_features[:, 1].detach().cpu().numpy()

    # ---- citations per edge: raw if papers_csv provided; else proxy from hyperedge_features[:,1] ----
    if args.papers_csv is not None:
        if "edge_id_to_paper" not in hg:
            raise KeyError("papers_csv provided but hypergraph.pt missing 'edge_id_to_paper'.")
        edge_id_to_paper = hg["edge_id_to_paper"]
        # edge_id_to_paper keys are ints, values are paper_id (string-ish)
        paper_citations = load_raw_citations_from_papers_csv(args.papers_csv, edge_id_to_paper, M)
        print("[info] Using RAW citation_count from papers.csv.")
    else:
        # proxy: normalized log citations (0..1), loses scale but still ranks influence
        paper_citations = hyperedge_features[:, 1].clone()
        print("[warn] papers_csv not provided: using hyperedge_features[:,1] (log_cit_norm) as citation proxy.")

    # years index
    years = sorted(set(year_per_edge.detach().cpu().numpy().astype(int).tolist()))
    year_to_idx = {y: i for i, y in enumerate(years)}
    T = len(years)
    print(f"Loaded: N={N:,} keywords | M={M:,} papers | C={C} topics | years {years[0]}..{years[-1]} (T={T})")

    # ---- compute u_k_t and m_k_t ----
    u_k_t, m_k_t = compute_keyword_year_u_m(
        hyperedge_index=hyperedge_index,
        year_per_edge=year_per_edge,
        paper_citations=paper_citations,
        years=years
    )

    # ---- topic aggregates u_c_t, m_c_t, a_c_t ----
    u_c_t, m_c_t, a_c_t = compute_topic_aggregates(
        u_k_t=u_k_t,
        m_k_t=m_k_t,
        node_labels=node_labels.detach().cpu().numpy(),
        node_scores=node_score,
        C=C,
        eps=1e-6
    )

    # ---- stabilized Z ----
    Z_u = np.log1p(np.maximum(u_c_t, 0.0))
    Z_m = np.log1p(np.maximum(m_c_t, 0.0))
    Z_a = np.log1p(np.maximum(a_c_t, 0.0))

    # ---- rolling residual targets ----
    L = args.baseline_window
    families = tuple(args.families)

    T_u = rolling_baseline_residuals(Z_u, years, L=L, families=families)
    T_m = rolling_baseline_residuals(Z_m, years, L=L, families=families)
    T_a = rolling_baseline_residuals(Z_a, years, L=L, families=families)

    # ---- pick training years (must have residuals => index >= L) ----
    train_years = [y for y in years if args.train_year_min <= y <= args.train_year_max and year_to_idx[y] >= L]
    if not train_years:
        raise RuntimeError("No training years available after applying baseline window and year range.")

    Y_list = []
    kept_years = []
    for y in train_years:
        ti = year_to_idx[y]
        yct = np.stack([T_u[:, ti], T_m[:, ti], T_a[:, ti]], axis=1).astype(np.float32)  # [C,3]
        if np.any(np.isnan(yct)):
            continue
        Y_list.append(yct)
        kept_years.append(y)
    train_years = kept_years
    Y = np.stack(Y_list, axis=0)  # [Nyears, C, 3]
    Ny = Y.shape[0]
    print(f"Residual targets built: {Ny} years x {C} topics x 3 channels")

    # channel weights = inverse variance (normalize)
    var = np.var(Y.reshape(-1, 3), axis=0) + 1e-8
    omega = 1.0 / var
    omega = omega / omega.sum()
    omega_t = torch.tensor(omega, device=args.device).view(1, 1, 3)
    print(f"Channel weights omega = {omega.tolist()} (u,m,a)")

    # ---- precompute windowed subgraphs per year ----
    windows: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for y in train_years:
        y0, y1 = y - args.gnn_window, y - 1
        sub_idx, edge_map = build_window_subhypergraph(hyperedge_index, year_per_edge, y0, y1)
        windows[y] = (sub_idx, edge_map)

    # ---- model ----
    device = args.device
    model = THGNN(
        node_in=node_features.size(1),
        edge_in=hyperedge_features.size(1),
        hidden=args.hidden,
        out_dim=3,
        layers=args.layers,
        dropout=args.dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    node_features = node_features.to(device)
    hyperedge_features = hyperedge_features.to(device)
    node_labels = node_labels.to(device)
    Y_t = torch.tensor(Y, device=device)

    # simple time split: last 20% years = val
    split = max(int(0.8 * Ny), 1)
    train_idx = list(range(0, split))
    val_idx = list(range(split, Ny))
    print(f"Train/val split: train={len(train_idx)} years, val={len(val_idx)} years")

    def run_epoch(idxs: List[int], train: bool) -> float:
        model.train(train)
        total = 0.0
        count = 0
        for j in idxs:
            y = train_years[j]
            sub_idx, edge_map = windows[y]
            sub_idx = sub_idx.to(device)
            edge_map = edge_map.to(device)

            x_edge = hyperedge_features[edge_map]  # [Ew, Fe]
            pred = model(node_features, sub_idx, x_edge, node_labels, C=C)  # [C,3]
            target = Y_t[j]  # [C,3]

            loss = (((pred - target) ** 2) * omega_t[0]).mean()

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            total += float(loss.item())
            count += 1
        return total / max(count, 1)

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(train_idx, train=True)
        va = run_epoch(val_idx, train=False) if val_idx else float("nan")

        if val_idx and va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} | train {tr:.6f} | val {va:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- inference for 2026 ----
    target_year = 2026
    y0, y1 = target_year - args.gnn_window, target_year - 1
    sub_idx_2026, edge_map_2026 = build_window_subhypergraph(hyperedge_index, year_per_edge, y0, y1)
    sub_idx_2026 = sub_idx_2026.to(device)
    edge_map_2026 = edge_map_2026.to(device)
    x_edge_2026 = hyperedge_features[edge_map_2026]

    model.eval()
    with torch.no_grad():
        pred_resid = model(node_features, sub_idx_2026, x_edge_2026, node_labels, C=C)  # [C,3]
        fused = (pred_resid * omega_t[0]).sum(dim=1)  # [C]

    fused_np = fused.detach().cpu().numpy()
    top = np.argsort(-fused_np)[:10]

    topic_names = hg.get("topic_names", None)

    print("\nTop predicted trending topics for 2026 (fused residual):")
    for r, c in enumerate(top, 1):
        name = topic_names.get(int(c), f"topic_{int(c)}") if isinstance(topic_names, dict) else f"topic_{int(c)}"
        u_, m_, a_ = pred_resid[int(c)].detach().cpu().tolist()
        print(f"{r:2d}. {int(c):2d} | {name[:70]} | score={fused_np[int(c)]:.6f} | (u,m,a)=({u_:.4f},{m_:.4f},{a_:.4f})")

    out_path = "thgnn_residual_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "omega": omega,
            "config": vars(args),
        },
        out_path
    )
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()