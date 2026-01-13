#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit XAI Dashboard for InfluencerRank (MaskOpt-style gates)

What this app provides (per selected checkpoint + user + month):
  - Top-K feature attribution bar
  - Top-K edge attribution bar (neighbor names)
  - Deletion/Insertion curves (toggle features/edges)
  - Ego graph (edge width = edge importance; month slider)
  - Evidence view (hashtag/mention frequency + example posts)
  - Time series view (true engagement, attention/sensitivity, prediction)

How to run
----------
1) Put this file next to your training script (default) OR point the "Script path" to it:
   - influencer_rank_full_fixed_xai_paper_no_imagecsv_v19_lossA_logrank_sampler_v2_log_scale.py

2) Ensure the required data files exist (default names, same as your script):
   - dataset_A_active_all.csv
   - hashtags_2017.csv
   - mentions_2017.csv
   - (optional) image_features_v2_full_fixed.csv

3) Run:
   streamlit run streamlit_xai_app.py

Notes
-----
- This app computes MaskOpt-like gates on-demand and caches results on disk under ./xai_cache/.
- If XAI is expensive, reduce epochs in the sidebar (e.g., 80-150) and/or restrict to a subgraph (num_hops=2).
"""

from __future__ import annotations

import os
import json
import math
import time
import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import streamlit as st

import torch

# optional deps used for specific views
import matplotlib.pyplot as plt
import networkx as nx


# -------------------------
# Utilities
# -------------------------
def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _month_labels(end_date: str, num_months: int) -> List[str]:
    end = pd.to_datetime(end_date)
    # months in ascending order (oldest -> newest)
    months = pd.period_range(end=end.to_period("M"), periods=int(num_months), freq="M")
    return [str(m) for m in months]


def _period_bounds(label: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    p = pd.Period(label, freq="M")
    start = p.to_timestamp(how="start")
    end = p.to_timestamp(how="end")
    return start, end


def _binary_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # p in [0,1]
    p = torch.clamp(p, eps, 1 - eps)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


@dataclass
class XAIResult:
    feat_gate: np.ndarray
    edge_gate: Optional[np.ndarray]
    df_feat: pd.DataFrame
    df_edge: pd.DataFrame
    meta: Dict[str, Any]
    edge_group_names: List[str]


# -------------------------
# Dynamic import of your training script
# -------------------------
@st.cache_resource(show_spinner=False)
def load_ir_module(script_path: str):
    script_path = str(Path(script_path).expanduser().resolve())
    if not Path(script_path).exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location("ir_mod", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


# -------------------------
# Cached loaders (model + graphs + raw data)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_from_ckpt(ir, ckpt_path: str, device_str: str):
    device = torch.device(device_str)
    model, feature_dim, params = ir.load_model_from_ckpt(ckpt_path, device=device)
    return model, feature_dim, params, device


@st.cache_resource(show_spinner=False)
def load_graphs(ir, data_dir: str, end_date: str, num_months: int,
               metric_numerator: str, metric_denominator: str, use_image_features: bool):
    # Point the script's file constants to absolute paths
    data_dir_p = Path(data_dir).expanduser().resolve()
    if not data_dir_p.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir_p}")

    ir.PREPROCESSED_FILE = str(data_dir_p / "dataset_A_active_all.csv")
    ir.HASHTAGS_FILE = str(data_dir_p / "hashtags_2017.csv")
    ir.MENTIONS_FILE = str(data_dir_p / "mentions_2017.csv")
    ir.IMAGE_DATA_FILE = str(data_dir_p / "image_features_v2_full_fixed.csv")
    ir.INFLUENCERS_FILE = str(data_dir_p / "influencers.txt")

    # NOTE: the v19 script expects end_date as pandas Timestamp because it does:
    #   start_date = end_date - pd.DateOffset(...)
    # Passing a raw string causes: "unsupported operand type(s) for -: 'str' and 'DateOffset'".
    end_date_ts = pd.to_datetime(end_date)

    prep = ir.prepare_graph_data(
        end_date=end_date_ts,
        num_months=int(num_months),
        metric_numerator=metric_numerator,
        metric_denominator=metric_denominator,
        use_image_features=bool(use_image_features),
    )
    if prep[0] is None:
        raise RuntimeError("prepare_graph_data returned None. Check your input files / columns.")
    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prep

    idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}
    feature_names = list(static_cols) + list(dynamic_cols)
    # (sanity) feature dim must match
    feature_dim = int(monthly_graphs[0].x.shape[1])
    if len(feature_names) != feature_dim:
        # fallback: create generic names (keeps app functional)
        feature_names = [f"feat_{i}" for i in range(feature_dim)]

    return monthly_graphs, influencer_indices, node_to_idx, idx_to_node, feature_names, follower_feat_idx, static_cols, dynamic_cols


@st.cache_resource(show_spinner=False)
def load_raw_tables(data_dir: str):
    p = Path(data_dir).expanduser().resolve()
    posts_path = p / "dataset_A_active_all.csv"
    h_path = p / "hashtags_2017.csv"
    m_path = p / "mentions_2017.csv"

    df_posts = pd.read_csv(posts_path)
    df_hash = pd.read_csv(h_path)
    df_mentions = pd.read_csv(m_path)

    # normalize columns a bit (best-effort)
    for d in (df_posts,):
        if "datetime" not in d.columns and "date" in d.columns:
            d["datetime"] = d["date"]
        if "username" not in d.columns and "source" in d.columns:
            d["username"] = d["source"]
        if "caption" not in d.columns and "text" in d.columns:
            d["caption"] = d["text"]
        if "post_id" not in d.columns and "id" in d.columns:
            d["post_id"] = d["id"]
        if "datetime" in d.columns:
            d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")

    for d, src_col, tgt_col, new_tgt in [
        (df_hash, "source", "target", "hashtag"),
        (df_mentions, "source", "target", "mention"),
    ]:
        if "username" not in d.columns and src_col in d.columns:
            d["username"] = d[src_col]
        if new_tgt not in d.columns and tgt_col in d.columns:
            d[new_tgt] = d[tgt_col]
        if "datetime" not in d.columns and "timestamp" in d.columns:
            d["datetime"] = pd.to_datetime(d["timestamp"], unit="s", errors="coerce")
        if "datetime" in d.columns:
            d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce")

    return df_posts, df_hash, df_mentions


# -------------------------
# MaskOpt-like gate optimization (compact, returns gates)
# -------------------------
def compute_maskopt_gates(
    ir,
    model,
    monthly_graphs,
    target_node_idx: int,
    explain_pos: int,
    feature_names: List[str],
    idx_to_node: Dict[int, str],
    device: torch.device,
    *,
    use_subgraph: bool = True,
    num_hops: int = 2,
    undirected: bool = True,
    feat_mask_scope: str = "target",
    edge_mask_scope: str = "incident",
    epochs: int = 150,
    lr: float = 0.05,
    fid_weight: float = 100.0,
    coeffs: Optional[Dict[str, float]] = None,
    budget_feat: Optional[float] = 10.0,
    budget_edge: Optional[float] = 20.0,
    budget_weight: float = 1.0,
    topk_feat: int = 20,
    topk_edge: int = 30,
    impact_reference: str = "unmasked",  # "unmasked" | "masked" | "both"
    seed: int = 0,
) -> XAIResult:
    """
    A compact implementation inspired by your script's MaskOpt E2E.
    Produces continuous gates in [0,1] and returns Top-K bars + score impact.
    """
    if coeffs is None:
        coeffs = {"edge_size": 0.08, "edge_ent": 0.15, "node_feat_size": 0.02, "node_feat_ent": 0.15}

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = model.to(device)
    model.eval()
    input_graphs = [g.to(device) for g in monthly_graphs]

    wrapper = ir.E2EMaskOptWrapper(
        model=model,
        input_graphs=input_graphs,
        target_node_idx=int(target_node_idx),
        explain_pos=int(explain_pos),
        device=device,
        use_subgraph=bool(use_subgraph),
        num_hops=int(num_hops),
        undirected=bool(undirected),
        feat_mask_scope=str(feat_mask_scope),
        edge_mask_scope=str(edge_mask_scope),
        edge_grouping="neighbor",
        idx_to_node=idx_to_node,
    )

    Fdim = int(wrapper.feature_dim)
    Edim = int(wrapper.num_edge_params) if getattr(wrapper, "num_edge_params", 0) else 0

    # logits -> sigmoid gates
    feat_logit = torch.zeros(Fdim, device=device, requires_grad=True)
    edge_logit = None
    if Edim > 0:
        edge_logit = torch.zeros(Edim, device=device, requires_grad=True)

    optim_params = [feat_logit] + ([edge_logit] if edge_logit is not None else [])
    opt = torch.optim.Adam(optim_params, lr=float(lr))

    ones_feat = torch.ones(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None
    zeros_feat = torch.zeros(Fdim, device=device)
    zeros_edge = torch.zeros(Edim, device=device) if Edim > 0 else None

    with torch.no_grad():
        pred_unmasked = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())

    best = {"loss": float("inf"), "feat": None, "edge": None, "pred": None}

    for _ in range(int(epochs)):
        opt.zero_grad(set_to_none=True)

        feat_gate = torch.sigmoid(feat_logit)
        edge_gate = torch.sigmoid(edge_logit) if edge_logit is not None else None

        pred = wrapper.predict_with_gates(feat_gate, edge_gate)
        loss_fid = (pred - pred_unmasked) ** 2

        # regularizers
        loss_size_feat = feat_gate.mean()
        loss_ent_feat = _binary_entropy(feat_gate).mean()

        loss_size_edge = torch.tensor(0.0, device=device)
        loss_ent_edge = torch.tensor(0.0, device=device)
        if edge_gate is not None and edge_gate.numel() > 0:
            loss_size_edge = edge_gate.mean()
            loss_ent_edge = _binary_entropy(edge_gate).mean()

        loss_budget = torch.tensor(0.0, device=device)
        if budget_feat is not None:
            loss_budget = loss_budget + torch.relu(feat_gate.sum() - float(budget_feat))
        if (budget_edge is not None) and (edge_gate is not None) and edge_gate.numel() > 0:
            loss_budget = loss_budget + torch.relu(edge_gate.sum() - float(budget_edge))

        total = (
            float(fid_weight) * loss_fid +
            float(coeffs.get("node_feat_size", 0.0)) * loss_size_feat +
            float(coeffs.get("node_feat_ent", 0.0)) * loss_ent_feat +
            float(coeffs.get("edge_size", 0.0)) * loss_size_edge +
            float(coeffs.get("edge_ent", 0.0)) * loss_ent_edge +
            float(budget_weight) * loss_budget
        )

        total.backward()
        opt.step()

        with torch.no_grad():
            lval = float(total.item())
            if lval < best["loss"]:
                best["loss"] = lval
                best["feat"] = feat_gate.detach().clone()
                best["edge"] = edge_gate.detach().clone() if edge_gate is not None else None
                best["pred"] = float(pred.item())

    feat_gate = best["feat"]
    edge_gate = best["edge"]

    # Build Top-K tables + score impact
    with torch.no_grad():
        pred_masked = float(wrapper.predict_with_gates(feat_gate, edge_gate).item())

    def _direction(diff: float, base: float, eps_abs: float = 1e-9, eps_rel: float = 1e-6) -> str:
        thr = max(eps_abs, abs(base) * eps_rel)
        if diff > thr:
            return "pos"
        if diff < -thr:
            return "neg"
        return "flat"

    # ---- Features ----
    feat_np = feat_gate.detach().cpu().numpy().astype(np.float32)
    top_feat_idx = np.argsort(feat_np)[::-1][: int(topk_feat)]
    feat_rows = []
    for j in top_feat_idx:
        imp = float(feat_np[j])
        if imp <= 0:
            continue
        name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        row = {"Type": "Feature", "Name": name, "Importance": imp}

        if impact_reference in ("unmasked", "both"):
            ab = ones_feat.clone()
            ab[j] = 0.0
            p_abl = float(wrapper.predict_with_gates(ab, ones_edge).item())
            diff = pred_unmasked - p_abl
            row["Score_Impact(unmasked)"] = float(diff)
            row["Direction(unmasked)"] = _direction(diff, pred_unmasked)

        if impact_reference in ("masked", "both"):
            ab = feat_gate.clone()
            ab[j] = 0.0
            p_abl = float(wrapper.predict_with_gates(ab, edge_gate).item())
            diff = pred_masked - p_abl
            row["Score_Impact(masked)"] = float(diff)
            row["Direction(masked)"] = _direction(diff, pred_masked)

        feat_rows.append(row)

    df_feat = pd.DataFrame(feat_rows)
    if not df_feat.empty:
        df_feat = df_feat.sort_values("Importance", ascending=False).reset_index(drop=True)

    # ---- Edges (neighbor groups) ----
    edge_group_names = list(getattr(wrapper, "edge_group_names", []))
    df_edge = pd.DataFrame()
    edge_np = None
    if edge_gate is not None and edge_gate.numel() > 0:
        edge_np = edge_gate.detach().cpu().numpy().astype(np.float32)
        top_edge_idx = np.argsort(edge_np)[::-1][: int(topk_edge)]
        edge_rows = []
        for gidx in top_edge_idx:
            imp = float(edge_np[gidx])
            if imp <= 0:
                continue
            nm = edge_group_names[gidx] if gidx < len(edge_group_names) else f"edge_group_{gidx}"
            row = {"Type": "Edge", "Name": nm, "Importance": imp}

            if impact_reference in ("unmasked", "both"):
                ab = ones_edge.clone()
                ab[gidx] = 0.0
                p_abl = float(wrapper.predict_with_gates(ones_feat, ab).item())
                diff = pred_unmasked - p_abl
                row["Score_Impact(unmasked)"] = float(diff)
                row["Direction(unmasked)"] = _direction(diff, pred_unmasked)

            if impact_reference in ("masked", "both"):
                ab = edge_gate.clone()
                ab[gidx] = 0.0
                p_abl = float(wrapper.predict_with_gates(feat_gate, ab).item())
                diff = pred_masked - p_abl
                row["Score_Impact(masked)"] = float(diff)
                row["Direction(masked)"] = _direction(diff, pred_masked)

            edge_rows.append(row)

        df_edge = pd.DataFrame(edge_rows)
        if not df_edge.empty:
            df_edge = df_edge.sort_values("Importance", ascending=False).reset_index(drop=True)

    meta = {
        "target_node": int(target_node_idx),
        "explain_pos": int(explain_pos),
        "pred_unmasked": float(pred_unmasked),
        "pred_masked": float(pred_masked),
        "best_loss": float(best["loss"]),
        "best_pred": float(best["pred"]) if best["pred"] is not None else None,
        "Fdim": int(Fdim),
        "Edim": int(Edim),
        "epochs": int(epochs),
        "lr": float(lr),
        "fid_weight": float(fid_weight),
        "coeffs": dict(coeffs),
        "budget_feat": None if budget_feat is None else float(budget_feat),
        "budget_edge": None if budget_edge is None else float(budget_edge),
        "budget_weight": float(budget_weight),
        "feat_mask_scope": str(feat_mask_scope),
        "edge_mask_scope": str(edge_mask_scope),
        "use_subgraph": bool(use_subgraph),
        "num_hops": int(num_hops),
    }

    return XAIResult(
        feat_gate=feat_np,
        edge_gate=edge_np,
        df_feat=df_feat,
        df_edge=df_edge,
        meta=meta,
        edge_group_names=edge_group_names,
    )


def _cache_key(ckpt_path: str, user: str, explain_pos: int, cfg: Dict[str, Any]) -> str:
    s = json.dumps({"ckpt": str(Path(ckpt_path).resolve()), "user": user, "pos": int(explain_pos), **cfg}, sort_keys=True)
    return _sha1(s)


def load_or_compute_xai(
    ir,
    model,
    monthly_graphs,
    target_node_idx: int,
    explain_pos: int,
    feature_names: List[str],
    idx_to_node: Dict[int, str],
    device: torch.device,
    cache_dir: Path,
    *,
    ckpt_path: str,
    user_name: str,
    xai_cfg: Dict[str, Any],
) -> XAIResult:
    key = _cache_key(ckpt_path, user_name, explain_pos, xai_cfg)
    npz_path = cache_dir / f"xai_{key}.npz"
    json_path = cache_dir / f"xai_{key}.json"
    feat_csv = cache_dir / f"xai_feat_{key}.csv"
    edge_csv = cache_dir / f"xai_edge_{key}.csv"

    if npz_path.exists() and json_path.exists() and feat_csv.exists() and edge_csv.exists():
        blob = np.load(npz_path, allow_pickle=True)
        feat_gate = blob["feat_gate"]
        edge_gate = blob["edge_gate"] if "edge_gate" in blob.files else None
        edge_gate = None if (edge_gate is not None and edge_gate.dtype == object) else edge_gate
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        df_feat = pd.read_csv(feat_csv)
        df_edge = pd.read_csv(edge_csv)
        edge_group_names = meta.get("edge_group_names", [])
        return XAIResult(
            feat_gate=feat_gate,
            edge_gate=edge_gate,
            df_feat=df_feat,
            df_edge=df_edge,
            meta=meta,
            edge_group_names=edge_group_names,
        )

    res = compute_maskopt_gates(
        ir=ir,
        model=model,
        monthly_graphs=monthly_graphs,
        target_node_idx=target_node_idx,
        explain_pos=explain_pos,
        feature_names=feature_names,
        idx_to_node=idx_to_node,
        device=device,
        **xai_cfg,
    )

    # persist
    np.savez_compressed(npz_path, feat_gate=res.feat_gate, edge_gate=res.edge_gate)
    res.df_feat.to_csv(feat_csv, index=False)
    res.df_edge.to_csv(edge_csv, index=False)

    meta = dict(res.meta)
    meta["edge_group_names"] = res.edge_group_names
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return res


# -------------------------
# Views
# -------------------------
def plot_bar(df: pd.DataFrame, title: str, topk: int = 20):
    if df is None or df.empty:
        st.info("No data.")
        return
    d = df.head(int(topk)).copy()
    # matplotlib horizontal bar
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(d))))
    ax.barh(list(reversed(d["Name"].astype(str).tolist())), list(reversed(d["Importance"].astype(float).tolist())))
    ax.set_title(title)
    ax.set_xlabel("Importance (gate value)")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def deletion_insertion_curves(
    ir,
    model,
    monthly_graphs,
    target_node_idx: int,
    explain_pos: int,
    feature_names: List[str],
    idx_to_node: Dict[int, str],
    device: torch.device,
    *,
    mode: str,   # "feature" | "edge"
    order_names: List[str],
    order_scores: List[float],
    k_max: int = 20,
    use_subgraph: bool = True,
    num_hops: int = 2,
):
    """
    Compute Deletion/Insertion curves using the given ranked list.
    Deletion: start from unmasked (all ones), progressively set top-j gates to 0.
    Insertion: start from all masked (all zeros), progressively set top-j gates to 1.
    """
    input_graphs = [g.to(device) for g in monthly_graphs]
    wrapper = ir.E2EMaskOptWrapper(
        model=model.to(device),
        input_graphs=input_graphs,
        target_node_idx=int(target_node_idx),
        explain_pos=int(explain_pos),
        device=device,
        use_subgraph=bool(use_subgraph),
        num_hops=int(num_hops),
        undirected=True,
        feat_mask_scope="target",
        edge_mask_scope="incident",
        edge_grouping="neighbor",
        idx_to_node=idx_to_node,
    )

    Fdim = int(wrapper.feature_dim)
    Edim = int(wrapper.num_edge_params) if getattr(wrapper, "num_edge_params", 0) else 0

    ones_feat = torch.ones(Fdim, device=device)
    zeros_feat = torch.zeros(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None
    zeros_edge = torch.zeros(Edim, device=device) if Edim > 0 else None

    # mapping from name -> index
    if mode == "feature":
        name_to_idx = {feature_names[i]: i for i in range(min(len(feature_names), Fdim))}
    else:
        name_to_idx = {str(nm): i for i, nm in enumerate(getattr(wrapper, "edge_group_names", []))}

    order_idx = [name_to_idx.get(str(n), None) for n in order_names]
    order_idx = [i for i in order_idx if i is not None]

    k_max = min(int(k_max), len(order_idx))
    xs = list(range(0, k_max + 1))
    del_scores = []
    ins_scores = []

    with torch.no_grad():
        # Deletion
        fg = ones_feat.clone()
        eg = ones_edge.clone() if ones_edge is not None else None
        del_scores.append(float(wrapper.predict_with_gates(fg, eg).item()))
        for j in range(1, k_max + 1):
            idx = order_idx[j - 1]
            if mode == "feature":
                fg[idx] = 0.0
            else:
                if eg is not None:
                    eg[idx] = 0.0
            del_scores.append(float(wrapper.predict_with_gates(fg, eg).item()))

        # Insertion
        if mode == "feature":
            fg = zeros_feat.clone()
            # keep edges unmasked when evaluating feature insertion
            eg = ones_edge.clone() if ones_edge is not None else None
        else:
            # edge insertion: keep features unmasked
            fg = ones_feat.clone()
            eg = zeros_edge.clone() if zeros_edge is not None else None
        ins_scores.append(float(wrapper.predict_with_gates(fg, eg).item()))
        for j in range(1, k_max + 1):
            idx = order_idx[j - 1]
            if mode == "feature":
                fg[idx] = 1.0
            else:
                if eg is not None:
                    eg[idx] = 1.0
            ins_scores.append(float(wrapper.predict_with_gates(fg, eg).item()))

    # plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, del_scores, marker="o", label="Deletion (remove top-k)")
    ax.plot(xs, ins_scores, marker="o", label="Insertion (add top-k)")
    ax.set_title(f"Deletion/Insertion curves ({mode})")
    ax.set_xlabel("k")
    ax.set_ylabel("Predicted score")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    # quick AUC summary (normalized by k range)
    auc_del = float(np.trapz(del_scores, xs) / max(1.0, k_max))
    auc_ins = float(np.trapz(ins_scores, xs) / max(1.0, k_max))
    st.caption(f"AUC (approx): Deletion={auc_del:.4f}, Insertion={auc_ins:.4f}")

# -------------------------
# Score impact over time (pos comparison)
# -------------------------
def _rank_df(df: pd.DataFrame, *, rank_by: str, impact_ref: str) -> pd.DataFrame:
    """Return df sorted by the chosen criterion."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    if rank_by == "Gate (Importance)":
        return d.sort_values("Importance", ascending=False).reset_index(drop=True)

    # Score_Impact based ranking
    col = f"Score_Impact({impact_ref})"
    if col not in d.columns:
        # fallback
        return d.sort_values("Importance", ascending=False).reset_index(drop=True)

    if rank_by == "Score impact (signed)":
        key = d[col].astype(float)
    else:
        key = d[col].astype(float).abs()

    d["_rank_key"] = key
    d = d.sort_values("_rank_key", ascending=False).drop(columns=["_rank_key"]).reset_index(drop=True)
    return d


def score_impact_at_k(
    ir,
    model,
    monthly_graphs,
    target_node_idx: int,
    explain_pos: int,
    feature_names: List[str],
    idx_to_node: Dict[int, str],
    device: torch.device,
    *,
    mode: str,  # "feature" | "edge"
    order_names: List[str],
    k: int,
    use_subgraph: bool = True,
    num_hops: int = 2,
) -> Dict[str, float]:
    """
    Score-based impact for a given month position.
    - baseline: unmasked score (all ones gates)
    - deletion_score_k: score after masking top-k items (other gate type stays unmasked)
    - drop_k = baseline - deletion_score_k
    - insertion_start: score with only the chosen gate type masked (other stays unmasked)
    - insertion_score_k: score after inserting top-k items from the masked start
    - gain_k = insertion_score_k - insertion_start
    """
    input_graphs = [g.to(device) for g in monthly_graphs]
    wrapper = ir.E2EMaskOptWrapper(
        model=model.to(device),
        input_graphs=input_graphs,
        target_node_idx=int(target_node_idx),
        explain_pos=int(explain_pos),
        device=device,
        use_subgraph=bool(use_subgraph),
        num_hops=int(num_hops),
        undirected=True,
        feat_mask_scope="target",
        edge_mask_scope="incident",
        edge_grouping="neighbor",
        idx_to_node=idx_to_node,
    )

    Fdim = int(wrapper.feature_dim)
    Edim = int(wrapper.num_edge_params) if getattr(wrapper, "num_edge_params", 0) else 0

    ones_feat = torch.ones(Fdim, device=device)
    zeros_feat = torch.zeros(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None
    zeros_edge = torch.zeros(Edim, device=device) if Edim > 0 else None

    # name -> index mapping
    if mode == "feature":
        name_to_idx = {feature_names[i]: i for i in range(min(len(feature_names), Fdim))}
    else:
        name_to_idx = {str(nm): i for i, nm in enumerate(getattr(wrapper, "edge_group_names", []))}

    order_idx = [name_to_idx.get(str(n), None) for n in (order_names or [])]
    order_idx = [i for i in order_idx if i is not None]
    k = min(int(k), len(order_idx))

    with torch.no_grad():
        # baseline (everything unmasked)
        baseline = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())

        # deletion @k (mask top-k of selected type; keep the other type unmasked)
        fg = ones_feat.clone()
        eg = ones_edge.clone() if ones_edge is not None else None
        for j in range(k):
            idx = order_idx[j]
            if mode == "feature":
                fg[idx] = 0.0
            else:
                if eg is not None:
                    eg[idx] = 0.0
        deletion_score_k = float(wrapper.predict_with_gates(fg, eg).item())
        drop_k = float(baseline - deletion_score_k)

        # insertion start (mask only the selected type; keep the other unmasked)
        if mode == "feature":
            fg = zeros_feat.clone()
            eg = ones_edge.clone() if ones_edge is not None else None
        else:
            fg = ones_feat.clone()
            eg = zeros_edge.clone() if zeros_edge is not None else None

        insertion_start = float(wrapper.predict_with_gates(fg, eg).item())

        for j in range(k):
            idx = order_idx[j]
            if mode == "feature":
                fg[idx] = 1.0
            else:
                if eg is not None:
                    eg[idx] = 1.0

        insertion_score_k = float(wrapper.predict_with_gates(fg, eg).item())
        gain_k = float(insertion_score_k - insertion_start)

    return {
        "baseline": baseline,
        "deletion_score_k": deletion_score_k,
        "drop_k": drop_k,
        "insertion_start": insertion_start,
        "insertion_score_k": insertion_score_k,
        "gain_k": gain_k,
    }


def score_impact_over_time_view(
    ir,
    model,
    monthly_graphs,
    target_node_idx: int,
    feature_names: List[str],
    idx_to_node: Dict[int, str],
    device: torch.device,
    *,
    ckpt_path: str,
    user_name: str,
    month_labels: List[str],
    explain_pos: int,
    cache_dir: Path,
    xai_cfg: Dict[str, Any],
):
    st.caption("Compare month positions using score-based impact (not only gate magnitude). This reuses the per-month XAI cache.")

    T = len(month_labels)
    if T == 0:
        st.info("No months loaded.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.radio("Target", ["feature", "edge"], horizontal=True)
    with c2:
        rank_by = st.selectbox("Rank items by", ["Gate (Importance)", "Score impact (abs)", "Score impact (signed)"], index=1)
    with c3:
        impact_ref = st.selectbox("Score impact reference", ["unmasked", "masked"], index=0)
    with c4:
        k = st.number_input("Top-k for score impact", min_value=1, max_value=500, value=30, step=1)

    # default: past 11 months relative to the currently selected explain_pos
    include_current = st.checkbox("Include current month (selected pos)", value=False)
    default_end = int(explain_pos) if include_current else max(0, int(explain_pos) - 1)
    default_start = max(0, default_end - 10)  # 11 months window
    if default_end < default_start:
        default_start, default_end = 0, max(0, T - 1)

    start_pos, end_pos = st.slider("pos range", 0, max(0, T - 1), (int(default_start), int(min(default_end, T - 1))))
    positions = list(range(int(start_pos), int(end_pos) + 1))

    st.write({
        "positions": positions,
        "months": [month_labels[p] for p in positions],
    })

    run = st.button("Compute score impact for selected range", type="primary")

    if not run:
        st.info("Press the button to compute. XAI for each month is cached under ./xai_cache/.")
        return

    prog = st.progress(0.0)
    rows = []
    # store per-pos ranked tables for item-level view
    per_pos_tables: Dict[int, pd.DataFrame] = {}

    for i, pos in enumerate(positions):
        prog.progress((i + 1) / max(1, len(positions)))

        res = load_or_compute_xai(
            ir=ir,
            model=model,
            monthly_graphs=monthly_graphs,
            target_node_idx=target_node_idx,
            explain_pos=int(pos),
            feature_names=feature_names,
            idx_to_node=idx_to_node,
            device=device,
            cache_dir=cache_dir,
            ckpt_path=ckpt_path,
            user_name=user_name,
            xai_cfg=xai_cfg,
        )

        df = res.df_feat if mode == "feature" else res.df_edge
        df_rank = _rank_df(df, rank_by=rank_by, impact_ref=impact_ref)
        per_pos_tables[int(pos)] = df_rank

        order_names = df_rank["Name"].astype(str).tolist() if not df_rank.empty else []
        met = score_impact_at_k(
            ir=ir,
            model=model,
            monthly_graphs=monthly_graphs,
            target_node_idx=target_node_idx,
            explain_pos=int(pos),
            feature_names=feature_names,
            idx_to_node=idx_to_node,
            device=device,
            mode=mode,
            order_names=order_names,
            k=int(k),
            use_subgraph=bool(xai_cfg.get("use_subgraph", True)),
            num_hops=int(xai_cfg.get("num_hops", 2)),
        )

        rows.append({
            "pos": int(pos),
            "month": month_labels[int(pos)] if int(pos) < len(month_labels) else str(pos),
            **met,
        })

    df_imp = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
    st.subheader("Top-k score impact over time")
    st.dataframe(df_imp, use_container_width=True)

    # plots
    xs = df_imp["pos"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(xs, df_imp["drop_k"].astype(float).values, marker="o")
    ax.set_title(f"Deletion impact (drop@{int(k)}) across pos — {mode}, ranked by {rank_by}")
    ax.set_xlabel("pos (oldest -> newest)")
    ax.set_ylabel("baseline - deletion_score_k")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.plot(xs, df_imp["baseline"].astype(float).values, marker="o", label="baseline")
    ax2.plot(xs, df_imp["deletion_score_k"].astype(float).values, marker="o", label="after deletion@k")
    ax2.set_title("Predicted score levels across pos")
    ax2.set_xlabel("pos (oldest -> newest)")
    ax2.set_ylabel("score")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

    # -------- Item-level tracking across pos (score impact per feature/edge) --------
    st.subheader("Item-level score impact across pos")
    union_top = st.number_input("Candidate pool: union of top-N per pos", min_value=5, max_value=500, value=50, step=5)

    # build candidate list from per_pos_tables
    cand = []
    for pos, dfr in per_pos_tables.items():
        if dfr is None or dfr.empty:
            continue
        cand.extend(dfr.head(int(union_top))["Name"].astype(str).tolist())
    cand = sorted(list(dict.fromkeys(cand)))

    if not cand:
        st.info("No candidates (edge groups may be empty for this user/month). Try feature mode or adjust XAI settings.")
        return

    item = st.selectbox("Pick an item to track", cand, index=0)
    col_imp = f"Score_Impact({impact_ref})"

    track_rows = []
    for pos in positions:
        # reload cached df for exact pos (cheap) so lookup is correct
        res_pos = load_or_compute_xai(
            ir=ir,
            model=model,
            monthly_graphs=monthly_graphs,
            target_node_idx=target_node_idx,
            explain_pos=int(pos),
            feature_names=feature_names,
            idx_to_node=idx_to_node,
            device=device,
            cache_dir=cache_dir,
            ckpt_path=ckpt_path,
            user_name=user_name,
            xai_cfg=xai_cfg,
        )
        dfo = res_pos.df_feat if mode == "feature" else res_pos.df_edge
        hit = dfo[dfo["Name"].astype(str) == str(item)].head(1)

        imp = float(hit["Importance"].iloc[0]) if (not hit.empty and "Importance" in hit.columns) else float("nan")
        sc = float(hit[col_imp].iloc[0]) if (not hit.empty and col_imp in hit.columns) else float("nan")
        track_rows.append({
            "pos": int(pos),
            "month": month_labels[int(pos)] if int(pos) < len(month_labels) else str(pos),
            "Importance(gate)": imp,
            col_imp: sc,
        })

    df_track = pd.DataFrame(track_rows).sort_values("pos").reset_index(drop=True)
    st.dataframe(df_track, use_container_width=True)

    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    ax3.plot(df_track["pos"].values, df_track[col_imp].values, marker="o")
    ax3.set_title(f"{item} — {col_imp} across pos")
    ax3.set_xlabel("pos")
    ax3.set_ylabel(col_imp)
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, clear_figure=True)

    fig4, ax4 = plt.subplots(figsize=(9, 3.5))
    ax4.plot(df_track["pos"].values, df_track["Importance(gate)"].values, marker="o")
    ax4.set_title(f"{item} — gate importance across pos")
    ax4.set_xlabel("pos")
    ax4.set_ylabel("Importance")
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4, clear_figure=True)


def plot_ego_graph(target_name: str, edge_df: pd.DataFrame, topk: int = 20):
    if edge_df is None or edge_df.empty:
        st.info("No edge attribution available.")
        return

    d = edge_df.head(int(topk)).copy()
    G = nx.Graph()
    G.add_node(target_name)

    # Add neighbors
    for _, r in d.iterrows():
        nbr = str(r["Name"])
        w = float(r["Importance"])
        G.add_node(nbr)
        G.add_edge(target_name, nbr, weight=w)

    # layout
    pos = nx.spring_layout(G, seed=0, k=0.8)

    weights = np.array([G[u][v]["weight"] for u, v in G.edges()], dtype=float)
    # scale widths for visibility
    if len(weights) > 0:
        w_min, w_max = float(weights.min()), float(weights.max())
        widths = 1.0 + 6.0 * ((weights - w_min) / (w_max - w_min + 1e-9))
    else:
        widths = []

    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700)
    nx.draw_networkx_edges(G, pos, ax=ax, width=widths)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    ax.set_title("Ego graph (edge width ~ importance)")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)


def evidence_view(
    df_posts: pd.DataFrame,
    df_hash: pd.DataFrame,
    df_mentions: pd.DataFrame,
    username: str,
    token: str,
    month_label: str,
    topn_posts: int = 10,
):
    start, end = _period_bounds(month_label)

    st.subheader("Frequency")
    # hashtag counts
    if "hashtag" in df_hash.columns:
        d1 = df_hash[(df_hash["username"].astype(str) == str(username)) & (df_hash["hashtag"].astype(str) == str(token))].copy()
        if "datetime" in d1.columns:
            d1 = d1[(d1["datetime"] >= start) & (d1["datetime"] <= end)]
        st.write({"hashtag_rows_in_month": int(len(d1))})
    # mention counts
    if "mention" in df_mentions.columns:
        d2 = df_mentions[(df_mentions["username"].astype(str) == str(username)) & (df_mentions["mention"].astype(str) == str(token))].copy()
        if "datetime" in d2.columns:
            d2 = d2[(d2["datetime"] >= start) & (d2["datetime"] <= end)]
        st.write({"mention_rows_in_month": int(len(d2))})

    st.subheader("Example posts (best-effort)")
    if "caption" not in df_posts.columns:
        st.info("posts CSV has no 'caption' column (or could not be normalized).")
        return

    d = df_posts[df_posts["username"].astype(str) == str(username)].copy()
    if "datetime" in d.columns:
        d = d[(d["datetime"] >= start) & (d["datetime"] <= end)]
    # heuristic: token present in caption
    pattern = str(token)
    d = d[d["caption"].astype(str).str.contains(pattern, case=False, na=False)]
    d = d.sort_values("datetime", ascending=False) if "datetime" in d.columns else d

    show_cols = []
    for c in ["datetime", "post_id", "caption", "likes", "comments"]:
        if c in d.columns:
            show_cols.append(c)
    st.dataframe(d[show_cols].head(int(topn_posts)), use_container_width=True)


def time_series_view(
    ir,
    model,
    monthly_graphs,
    target_node_idx: int,
    device: torch.device,
    month_labels: List[str],
):
    st.subheader("Engagement × Importance × Predicted score")

    model = model.to(device)
    model.eval()

    # true engagement by month
    y = [float(monthly_graphs[t].y[target_node_idx].view(()).item()) for t in range(len(monthly_graphs))]

    # attention/sensitivity provided by your script utility
    sens_df, selected_pos, pred_full, alpha = ir.compute_time_step_sensitivity(
        model=model,
        input_graphs=[g.to(device) for g in monthly_graphs],
        target_node_idx=int(target_node_idx),
        device=device,
        topk=min(5, len(month_labels)),
        score_mode="alpha_x_delta",
        min_delta=1e-6,
    )
    alpha = np.asarray(alpha, dtype=float).reshape(-1)

    # align lengths
    T = len(month_labels)
    y = np.asarray(y[:T], dtype=float)
    alpha = np.asarray(alpha[:T], dtype=float)

    df = pd.DataFrame({
        "month": month_labels,
        "true_engagement": y,
        "alpha_attention": alpha,
        "alpha_x_engagement": alpha * y,
    })
    st.caption(f"Model predicted score (scalar): {float(pred_full):.6f}")
    st.dataframe(df, use_container_width=True)

    # plots
    fig1, ax1 = plt.subplots(figsize=(9, 3.5))
    ax1.plot(range(T), df["true_engagement"].values, marker="o")
    ax1.set_title("True engagement over months")
    ax1.set_xlabel("month index (oldest->newest)")
    ax1.set_ylabel("engagement")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1, clear_figure=True)

    fig2, ax2 = plt.subplots(figsize=(9, 3.5))
    ax2.plot(range(T), df["alpha_attention"].values, marker="o")
    ax2.set_title("Attention weight (alpha) over months")
    ax2.set_xlabel("month index (oldest->newest)")
    ax2.set_ylabel("alpha")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2, clear_figure=True)

    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    ax3.plot(range(T), df["alpha_x_engagement"].values, marker="o")
    ax3.set_title("alpha × engagement (simple alignment signal)")
    ax3.set_xlabel("month index (oldest->newest)")
    ax3.set_ylabel("alpha*engagement")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3, clear_figure=True)

    st.subheader("Sensitivity table (alpha×delta)")
    st.dataframe(sens_df, use_container_width=True)
    st.caption(f"Top positions by sensitivity: {selected_pos}")


# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="InfluencerRank XAI Dashboard", layout="wide")

st.title("InfluencerRank XAI Dashboard (Streamlit)")

with st.sidebar:
    st.header("Setup")

    script_path = st.text_input(
        "Script path (your v19 file)",
        value=str(Path(__file__).with_name("influencer_rank_full_fixed_xai_paper_no_imagecsv_v19_lossA_logrank_sampler_v2_log_scale.py")),
    )

    data_dir = st.text_input("Data directory", value=str(Path.cwd()))
    ckpt_dir = st.text_input("Checkpoints directory", value=str(Path(data_dir) / "checkpoints"))

    end_date = st.text_input("End date (YYYY-MM-DD)", value="2017-12-31")
    num_months = st.number_input("Num months", min_value=2, max_value=36, value=12, step=1)

    metric_numerator = st.selectbox("Metric numerator", ["likes", "likes_and_comments", "comments"], index=1)
    metric_denominator = st.selectbox("Metric denominator", ["posts", "followers"], index=1)
    use_image_features = st.checkbox("Use image features if available", value=False)

    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
    device_str = str(_auto_device()) if device_choice == "auto" else device_choice

    st.divider()
    st.header("MaskOpt (compute gates)")
    epochs = st.slider("epochs", 30, 600, 150, 10)
    lr = st.number_input("lr", min_value=0.001, max_value=0.5, value=0.05, step=0.005, format="%.3f")
    fid_weight = st.number_input("fidelity weight", min_value=1.0, max_value=10000.0, value=100.0, step=10.0)
    budget_feat = st.number_input("budget_feat (sum gates)", min_value=0.0, max_value=2000.0, value=10.0, step=1.0)
    budget_edge = st.number_input("budget_edge (sum gates)", min_value=0.0, max_value=2000.0, value=20.0, step=1.0)
    budget_weight = st.number_input("budget_weight", min_value=0.0, max_value=1000.0, value=1.0, step=0.5)

    use_subgraph = st.checkbox("Use subgraph (faster)", value=True)
    num_hops = st.slider("num_hops", 1, 6, 2, 1)

    feat_mask_scope = st.selectbox("feat_mask_scope", ["target", "all"], index=0)
    edge_mask_scope = st.selectbox("edge_mask_scope", ["incident", "all"], index=0)
    impact_reference = st.selectbox("Score impact reference", ["unmasked", "masked", "both"], index=0)

    topk_feat = st.slider("Top-K features", 5, 100, 20, 5)
    topk_edge = st.slider("Top-K edges", 5, 200, 30, 5)

# Load module + data
try:
    ir = load_ir_module(script_path)
except Exception as e:
    st.error(f"Failed to import your script: {e}")
    st.stop()

try:
    monthly_graphs, influencer_indices, node_to_idx, idx_to_node, feature_names, follower_feat_idx, static_cols, dynamic_cols = load_graphs(
        ir, data_dir, end_date, int(num_months), metric_numerator, metric_denominator, bool(use_image_features)
    )
except Exception as e:
    st.error(f"Failed to build graphs: {e}")
    st.stop()

month_labels = _month_labels(end_date, int(num_months))

# checkpoint selection
ckpt_dir_p = Path(ckpt_dir).expanduser()
pths = sorted(list(ckpt_dir_p.rglob("*.pth"))) if ckpt_dir_p.exists() else []
ckpt_path = st.selectbox(
    "Checkpoint (.pth)",
    options=[str(p) for p in pths] if pths else [],
    index=0 if pths else None,
    placeholder="No checkpoint found. Set the correct directory.",
)
if not ckpt_path:
    st.warning("Select a checkpoint to continue.")
    st.stop()

try:
    model, feature_dim, ckpt_params, device = load_model_from_ckpt(ir, ckpt_path, device_str=device_str)
except Exception as e:
    st.error(f"Failed to load model from checkpoint: {e}")
    st.stop()

# user selection
inf_usernames = [idx_to_node[int(i)] for i in influencer_indices if int(i) in idx_to_node]
inf_usernames_sorted = sorted(set(inf_usernames))

colA, colB, colC = st.columns([2, 1, 1])
with colA:
    user_name = st.selectbox("XAI target user (influencer)", options=inf_usernames_sorted, index=0)
with colB:
    month_label = st.selectbox("Month label", options=month_labels, index=len(month_labels) - 1)
with colC:
    explain_pos = month_labels.index(month_label)

target_node_idx = int(node_to_idx[user_name])

# cache dir
cache_dir = _safe_mkdir(Path(data_dir) / "xai_cache")

# XAI config (used for both computation and cache key)
xai_cfg = dict(
    use_subgraph=bool(use_subgraph),
    num_hops=int(num_hops),
    undirected=True,
    feat_mask_scope=str(feat_mask_scope),
    edge_mask_scope=str(edge_mask_scope),
    epochs=int(epochs),
    lr=float(lr),
    fid_weight=float(fid_weight),
    budget_feat=float(budget_feat) if budget_feat > 0 else None,
    budget_edge=float(budget_edge) if budget_edge > 0 else None,
    budget_weight=float(budget_weight),
    topk_feat=int(topk_feat),
    topk_edge=int(topk_edge),
    impact_reference=str(impact_reference),
    seed=0,
)

# tabs
tab_overview, tab_attr, tab_curves, tab_impact, tab_ego, tab_evidence, tab_ts = st.tabs(
    ["Overview", "Attribution bars", "Deletion/Insertion", "Score impact (pos)", "Ego graph", "Evidence", "Time series"]
)

with tab_overview:
    st.subheader("Selected configuration")
    st.write({
        "checkpoint": str(Path(ckpt_path).name),
        "device": str(device),
        "user": user_name,
        "target_node_idx": int(target_node_idx),
        "month": month_label,
        "explain_pos": int(explain_pos),
        "feature_dim": int(feature_dim),
        "metric": f"{metric_numerator}/{metric_denominator}",
        "subgraph": bool(use_subgraph),
        "num_hops": int(num_hops),
    })
    st.caption("Tip: run XAI only when needed (button below). Results are cached in ./xai_cache/.")

    run_btn = st.button("Run / Load XAI for this month", type="primary")

with tab_attr:
    st.subheader("Top-K attributions")
    st.write("Run XAI first (Overview tab) to populate these views.")

with tab_curves:
    st.subheader("Deletion / Insertion sanity check")
    st.write("Run XAI first (Overview tab) to populate these views.")

with tab_impact:
    st.subheader("Score impact over time (pos comparison)")
    st.write("Compute score-based impacts across multiple months. Uses cached per-month XAI.")

with tab_ego:
    st.subheader("Ego graph")
    st.write("Run XAI first (Overview tab) to populate these views.")

with tab_evidence:
    st.subheader("Evidence view")
    st.write("Run XAI first (Overview tab) to populate these views.")

with tab_ts:
    st.subheader("Time series")
    st.write("This tab does NOT require MaskOpt gates; it uses attention/sensitivity from the model.")

# always show time series
with tab_ts:
    time_series_view(ir, model, monthly_graphs, target_node_idx, device, month_labels)

# score impact (pos comparison)
with tab_impact:
    score_impact_over_time_view(
        ir=ir,
        model=model,
        monthly_graphs=monthly_graphs,
        target_node_idx=target_node_idx,
        feature_names=feature_names,
        idx_to_node=idx_to_node,
        device=device,
        ckpt_path=ckpt_path,
        user_name=user_name,
        month_labels=month_labels,
        explain_pos=explain_pos,
        cache_dir=cache_dir,
        xai_cfg=xai_cfg,
    )

# compute/load XAI on demand
if run_btn:
    # xai_cfg is defined above (and used as part of cache key)


    with st.spinner("Computing (or loading cached) XAI..."):
        res = load_or_compute_xai(
            ir=ir,
            model=model,
            monthly_graphs=monthly_graphs,
            target_node_idx=target_node_idx,
            explain_pos=explain_pos,
            feature_names=feature_names,
            idx_to_node=idx_to_node,
            device=device,
            cache_dir=cache_dir,
            ckpt_path=ckpt_path,
            user_name=user_name,
            xai_cfg=xai_cfg,
        )

    st.success("XAI ready.")

    # --- Attribution tab ---
    with tab_attr:
        left, right = st.columns(2)
        with left:
            plot_bar(res.df_feat, "Top-K Feature attribution", topk=topk_feat)
            if not res.df_feat.empty:
                st.dataframe(res.df_feat.head(int(topk_feat)), use_container_width=True)
        with right:
            plot_bar(res.df_edge, "Top-K Edge attribution (neighbor groups)", topk=topk_edge)
            if not res.df_edge.empty:
                st.dataframe(res.df_edge.head(int(topk_edge)), use_container_width=True)

        st.subheader("Meta")
        st.json(res.meta)

    # --- Curves tab ---
    with tab_curves:
        mode = st.radio("Mask type", ["feature", "edge"], horizontal=True)
        k_max = st.slider("k_max", 5, 200, min(30, max(5, (topk_edge if mode=="edge" else topk_feat))), 5)

        if mode == "feature":
            order_names = res.df_feat["Name"].astype(str).tolist()
            order_scores = res.df_feat["Importance"].astype(float).tolist()
        else:
            order_names = res.df_edge["Name"].astype(str).tolist()
            order_scores = res.df_edge["Importance"].astype(float).tolist()

        if len(order_names) == 0:
            st.info("No attribution available for the selected mode.")
        else:
            deletion_insertion_curves(
                ir=ir,
                model=model,
                monthly_graphs=monthly_graphs,
                target_node_idx=target_node_idx,
                explain_pos=explain_pos,
                feature_names=feature_names,
                idx_to_node=idx_to_node,
                device=device,
                mode=mode,
                order_names=order_names,
                order_scores=order_scores,
                k_max=int(k_max),
                use_subgraph=bool(use_subgraph),
                num_hops=int(num_hops),
            )

    # --- Ego graph tab ---
    with tab_ego:
        plot_ego_graph(user_name, res.df_edge, topk=min(30, int(topk_edge)))

    # --- Evidence tab ---
    with tab_evidence:
        try:
            df_posts, df_hash, df_mentions = load_raw_tables(data_dir)
        except Exception as e:
            st.error(f"Failed to load raw tables: {e}")
            st.stop()

        st.caption("Pick a token from Top-K edges (neighbor name) or type your own (hashtag / mention / username).")
        default_token = ""
        if not res.df_edge.empty:
            default_token = str(res.df_edge.iloc[0]["Name"])
        token = st.text_input("token", value=default_token)

        evidence_view(df_posts, df_hash, df_mentions, user_name, token, month_label, topn_posts=10)
