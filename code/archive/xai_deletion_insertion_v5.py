#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deletion/Insertion curves for MaskOpt explanations (feature/edge gates).

What this script does
---------------------
For each (target_node, explain_pos) pair:
  1) Obtain an importance ranking for Features / Edge-groups (neighbors)
     - either by running MaskOpt once (maskopt_e2e_explain), OR
     - by reading existing XAI CSVs (df_feat/df_edge) previously exported.
  2) Evaluate faithfulness by counterfactual ablation:
     - Deletion: start from "all kept" (gate=1), then set top-k elements to 0.
     - Insertion: start from "all removed" (gate=0), then set top-k elements to 1.
  3) Compare against baselines (random order, degree order for edges).
  4) Save curves + AUC summary (and plots).

Assumptions
-----------
- You have the modular package "influencer_rank" (from influencer_rank_modular.zip) on PYTHONPATH.
- You run from the directory that contains the dataset CSVs as configured in influencer_rank/config.py,
  OR pass --data_dir to change working directory.

Example (MaskOpt -> curves)
---------------------------
python xai_deletion_insertion.py \
  --ckpt path/to/model.ckpt \
  --end_date 2017-12-01 --num_months 11 \
  --target_nodes 1238805,916490 \
  --positions 4,1,5 \
  --k_max 100 --k_step 5 \
  --random_repeats 20 \
  --outdir out_di

Example (use existing XAI CSVs -> curves)
----------------------------------------
python xai_deletion_insertion.py \
  --ckpt path/to/model.ckpt \
  --end_date 2017-12-01 --num_months 11 \
  --xai_feat_csv path/to/xai_features_node_1238805_pos_4.csv \
  --xai_edge_csv path/to/xai_edges_node_1238805_pos_4.csv \
  --target_nodes 1238805 --positions 4 \
  --k_max 100 --k_step 5 \
  --outdir out_di

Notes on interpretation
-----------------------
- For deletion curves, "faster drop" (larger normalized drop early) indicates more faithful explanation.
- For insertion curves, "faster recovery" indicates more faithful explanation.
- We report AUC over k-fraction in [0,1]. For deletion we use normalized drop; for insertion normalized recovery.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

import matplotlib.pyplot as plt

# --- imports from your modular package ---
from influencer_rank.data import prepare_graph_data
from influencer_rank.model import HardResidualInfluencerModel
from influencer_rank.xai_maskopt import E2EMaskOptWrapper, maskopt_e2e_explain

def _normalize_graphs_data(graphs_data):
    """Accept both dict-style and tuple-style outputs from prepare_graph_data.

    Supported tuple formats:
      - (monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols)
      - (monthly_graphs, influencer_indices, node_to_idx, feature_dim, follower_feat_idx, static_cols, dynamic_cols)
    """
    if isinstance(graphs_data, dict):
        monthly_graphs = graphs_data.get("monthly_graphs")
        node_to_idx = graphs_data.get("node_to_idx")
        feature_names = graphs_data.get("feature_names")
        influencer_indices = graphs_data.get("influencer_indices")
        follower_feat_idx = graphs_data.get("follower_feat_idx")
        if monthly_graphs is None or node_to_idx is None:
            raise ValueError("graphs_data dict missing required keys: monthly_graphs/node_to_idx")
        if feature_names is None:
            static_cols = graphs_data.get("static_cols") or graphs_data.get("static_feature_cols")
            dynamic_cols = graphs_data.get("dynamic_cols") or graphs_data.get("dynamic_feature_cols")
            if static_cols is not None and dynamic_cols is not None:
                feature_names = list(static_cols) + list(dynamic_cols)
        return monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, feature_names

    if isinstance(graphs_data, (tuple, list)):
        if len(graphs_data) == 6:
            monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = graphs_data
        elif len(graphs_data) == 7:
            monthly_graphs, influencer_indices, node_to_idx, _feature_dim, follower_feat_idx, static_cols, dynamic_cols = graphs_data
        else:
            raise ValueError(f"Unsupported graphs_data tuple length: {len(graphs_data)}")
        feature_names = list(static_cols) + list(dynamic_cols)
        return monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, feature_names

    raise TypeError(f"Unsupported graphs_data type: {type(graphs_data)}")



# ----------------------------
# Utilities
# ----------------------------

def _as_int_list(csv: str) -> List[int]:
    if not csv:
        return []
    out = []
    for s in csv.split(","):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    return out


def _infer_model_hparams_from_state_dict(state: Dict[str, "np.ndarray"]) -> Dict[str, int]:
    """
    Infer (projection_dim, gcn_dim, rnn_dim, num_gcn_layers) from state_dict shapes.
    This avoids having to hardcode hyperparams used during training.
    """
    # projection_dim: Linear(feature_dim -> projection_dim) weight shape [proj, feat]
    proj_w = None
    for k in ["projection_layer.0.weight", "projection.0.weight", "proj.0.weight"]:
        if k in state:
            proj_w = state[k]
            break
    if proj_w is None:
        # fallback: find any weight that looks like [proj, feat] on first linear
        for k, v in state.items():
            if k.endswith("projection_layer.0.weight") or k.endswith("projection.0.weight"):
                proj_w = v
                break
    if proj_w is None:
        raise ValueError("Could not infer projection_dim from ckpt. "
                         "Expected key like 'projection_layer.0.weight' in state_dict.")
    projection_dim = int(proj_w.shape[0])

    # gcn_dim: first GCN conv weight, typical key includes "gcn_encoder.convs.0"
    gcn_w = None
    for k, v in state.items():
        if re.search(r"gcn_encoder\.convs\.0\..*weight", k):
            gcn_w = v
            break
        if re.search(r"gcn_encoder\.convs\.0\..*lin\.weight", k):
            gcn_w = v
            break
    if gcn_w is None:
        # try older key patterns
        for k, v in state.items():
            if "convs.0" in k and k.endswith("weight") and v.ndim == 2:
                gcn_w = v
                break
    if gcn_w is None:
        raise ValueError("Could not infer gcn_dim from ckpt state_dict (no conv0 weight found).")
    gcn_dim = int(gcn_w.shape[0])

    # num_gcn_layers: count distinct conv indices in keys
    conv_ids = set()
    for k in state.keys():
        m = re.search(r"gcn_encoder\.convs\.(\d+)\.", k)
        if m:
            conv_ids.add(int(m.group(1)))
    num_gcn_layers = max(conv_ids) + 1 if conv_ids else 2

    # rnn_dim: LSTM weight_ih_l0 shape [4*rnn_dim, input_size]
    lstm_w = None
    for k in ["attentive_rnn.lstm.weight_ih_l0", "lstm.weight_ih_l0", "rnn.lstm.weight_ih_l0"]:
        if k in state:
            lstm_w = state[k]
            break
    if lstm_w is None:
        # search any "weight_ih_l0"
        for k, v in state.items():
            if k.endswith("weight_ih_l0") and v.ndim == 2:
                lstm_w = v
                break
    if lstm_w is None:
        raise ValueError("Could not infer rnn_dim from ckpt state_dict (no weight_ih_l0 found).")
    rnn_dim = int(lstm_w.shape[0] // 4)

    return dict(
        projection_dim=projection_dim,
        gcn_dim=gcn_dim,
        rnn_dim=rnn_dim,
        num_gcn_layers=num_gcn_layers,
    )


def _load_ckpt_state_dict(ckpt_path: str, map_location: str = "cpu") -> Dict[str, "np.ndarray"]:
    """Load a checkpoint and return a plain PyTorch state_dict.

    Supported formats:
      - { "model_state_dict": state_dict, ... }  (common training script format)
      - { "state_dict": state_dict, "params": ..., "feature_dim": ... }  (your save_model_checkpoint format)
      - state_dict itself (mapping param_name -> tensor)
    """
    import torch

    obj = torch.load(ckpt_path, map_location=map_location)

    # 1) Wrapper dict formats
    if isinstance(obj, dict):
        if "model_state_dict" in obj and isinstance(obj["model_state_dict"], dict):
            return obj["model_state_dict"]
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]

        # 2) Some scripts store weights under 'net' or similar
        for key in ("net", "model", "weights"):
            if key in obj and isinstance(obj[key], dict) and any(k.endswith((".weight", ".bias")) for k in obj[key].keys()):
                return obj[key]

        # 3) Sometimes the state dict is the whole dict already
        if any(k.endswith((".weight", ".bias")) for k in obj.keys()):
            return obj

    raise ValueError(
        f"Unexpected ckpt format: {type(obj)} keys={list(obj.keys())[:10] if isinstance(obj, dict) else 'n/a'}"
    )



def _build_idx_to_node_from_node_to_idx(node_to_idx: Dict[str, int]) -> Dict[int, str]:
    # In your pipeline, node_to_idx keys are already human-readable labels
    return {idx: name for name, idx in node_to_idx.items()}


def _ensure_pd():
    if pd is None:
        raise RuntimeError("pandas is required for this script. Please `pip install pandas`.")


def _trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    # x must be increasing in [0,1]
    return float(np.trapz(y, x))


# ----------------------------
# Ordering builders
# ----------------------------

@dataclass
class Orderings:
    # Each element is ('feat', feat_idx) or ('edge', edge_group_idx)
    order_top: List[Tuple[str, int]]
    order_random_list: List[List[Tuple[str, int]]]  # list of random permutations (repeats)
    order_degree: Optional[List[Tuple[str, int]]]   # only for edges (may be None)


def _build_orders_from_importance(
    df_feat, df_edge,
    feature_name_to_idx: Dict[str, int],
    edge_name_to_idx: Dict[str, int],
    random_repeats: int,
    seed: int,
    degree_order_edge_idxs: Optional[List[int]] = None,
    combine_feat_edge: bool = True,
) -> Orderings:
    """
    Build top-k order from Importance, plus random baselines and (optional) degree order for edges.
    """
    rng = np.random.default_rng(seed)

    # Prepare ranked lists with a stable mapping to indices
    items: List[Tuple[str, int, float]] = []

    if df_feat is not None and len(df_feat) > 0:
        for _, r in df_feat.iterrows():
            name = str(r["Name"])
            imp = float(r["Importance"])
            if name in feature_name_to_idx:
                items.append(("feat", feature_name_to_idx[name], imp))

    if df_edge is not None and len(df_edge) > 0:
        for _, r in df_edge.iterrows():
            name = str(r["Name"])
            imp = float(r["Importance"])
            if name in edge_name_to_idx:
                items.append(("edge", edge_name_to_idx[name], imp))

    # if not combining, keep separate ordering handled outside
    if not combine_feat_edge:
        # still build order_top from union (caller can filter)
        pass

    # Sort by importance desc
    items.sort(key=lambda x: x[2], reverse=True)
    order_top = [(k, idx) for (k, idx, _) in items]

    # Random permutations over the same set of elements
    order_random_list: List[List[Tuple[str, int]]] = []
    base = order_top.copy()
    for _ in range(max(1, random_repeats)):
        perm = base.copy()
        rng.shuffle(perm)
        order_random_list.append(perm)

    # Degree order (edges only) if provided: we output an order over union, with edges by degree first, then others
    order_degree = None
    if degree_order_edge_idxs is not None:
        degree_edges = [("edge", eidx) for eidx in degree_order_edge_idxs if ("edge", eidx) in set(order_top)]
        # fill remaining items (including feats) in original top order
        seen = set(degree_edges)
        rest = [it for it in order_top if it not in seen]
        order_degree = degree_edges + rest

    return Orderings(order_top=order_top, order_random_list=order_random_list, order_degree=order_degree)


# ----------------------------
# Curve evaluation
# ----------------------------

@dataclass
class CurveResult:
    ks: np.ndarray            # integer k values (0..)
    frac: np.ndarray          # k / Kmax in [0,1]
    pred_full: float
    pred_empty: float
    deletion_drop_norm: np.ndarray   # normalized drop vs k
    insertion_rec_norm: np.ndarray   # normalized recovery vs k
    auc_del: float
    auc_ins: float


def _eval_one_order(
    wrapper: E2EMaskOptWrapper,
    order: List[Tuple[str, int]],
    k_max: int,
    k_step: int,
    device: str,
    eps: float = 1e-12,
) -> CurveResult:
    """
    Evaluate deletion/insertion for a single ordering.
    """
    import torch

    feat_dim = int(wrapper.feat_dim)
    edge_dim = int(wrapper.edge_params)

    K = min(k_max, len(order))
    ks = list(range(0, K + 1, max(1, k_step)))
    if ks[-1] != K:
        ks.append(K)

    # gates
    ones_feat = torch.ones(feat_dim, device=device)
    ones_edge = torch.ones(edge_dim, device=device)
    zeros_feat = torch.zeros(feat_dim, device=device)
    zeros_edge = torch.zeros(edge_dim, device=device)

    with torch.no_grad():
        pred_full = float(wrapper.predict_with_gates(ones_feat, ones_edge))
        pred_empty = float(wrapper.predict_with_gates(zeros_feat, zeros_edge))

    denom = abs(pred_full - pred_empty) + eps

    del_drop = []
    ins_rec = []

    # Deletion: start from ones, progressively set top-k to 0
    feat_gate = ones_feat.clone()
    edge_gate = ones_edge.clone()
    k_cursor = 0
    for k in ks:
        while k_cursor < k:
            kind, idx = order[k_cursor]
            if kind == "feat":
                feat_gate[idx] = 0.0
            else:
                edge_gate[idx] = 0.0
            k_cursor += 1
        with torch.no_grad():
            pred_k = float(wrapper.predict_with_gates(feat_gate, edge_gate))
        del_drop.append((pred_full - pred_k) / denom)

    # Insertion: start from zeros, progressively set top-k to 1
    feat_gate = zeros_feat.clone()
    edge_gate = zeros_edge.clone()
    k_cursor = 0
    for k in ks:
        while k_cursor < k:
            kind, idx = order[k_cursor]
            if kind == "feat":
                feat_gate[idx] = 1.0
            else:
                edge_gate[idx] = 1.0
            k_cursor += 1
        with torch.no_grad():
            pred_k = float(wrapper.predict_with_gates(feat_gate, edge_gate))
        ins_rec.append((pred_k - pred_empty) / denom)

    ks_arr = np.array(ks, dtype=int)
    frac = ks_arr / float(K if K > 0 else 1)

    del_drop = np.array(del_drop, dtype=float)
    ins_rec = np.array(ins_rec, dtype=float)

    auc_del = _trapz_auc(frac, del_drop)  # larger = drops earlier (better)
    auc_ins = _trapz_auc(frac, ins_rec)   # larger = recovers earlier (better)

    return CurveResult(
        ks=ks_arr, frac=frac,
        pred_full=pred_full, pred_empty=pred_empty,
        deletion_drop_norm=del_drop, insertion_rec_norm=ins_rec,
        auc_del=auc_del, auc_ins=auc_ins,
    )


@dataclass
class AggregateCurve:
    ks: np.ndarray
    frac: np.ndarray
    mean_del: np.ndarray
    std_del: np.ndarray
    mean_ins: np.ndarray
    std_ins: np.ndarray
    auc_del_mean: float
    auc_del_std: float
    auc_ins_mean: float
    auc_ins_std: float


def _aggregate_curve(results: List[CurveResult]) -> AggregateCurve:
    assert len(results) > 0
    ks = results[0].ks
    frac = results[0].frac
    del_mat = np.stack([r.deletion_drop_norm for r in results], axis=0)
    ins_mat = np.stack([r.insertion_rec_norm for r in results], axis=0)
    auc_del = np.array([r.auc_del for r in results], dtype=float)
    auc_ins = np.array([r.auc_ins for r in results], dtype=float)

    return AggregateCurve(
        ks=ks, frac=frac,
        mean_del=del_mat.mean(axis=0),
        std_del=del_mat.std(axis=0),
        mean_ins=ins_mat.mean(axis=0),
        std_ins=ins_mat.std(axis=0),
        auc_del_mean=float(auc_del.mean()),
        auc_del_std=float(auc_del.std()),
        auc_ins_mean=float(auc_ins.mean()),
        auc_ins_std=float(auc_ins.std()),
    )


def _plot_aggregate(
    agg_by_label: Dict[str, AggregateCurve],
    title: str,
    out_png: str,
):
    # Deletion plot
    plt.figure()
    for label, agg in agg_by_label.items():
        plt.plot(agg.frac, agg.mean_del, label=f"{label} (AUC={agg.auc_del_mean:.3f}±{agg.auc_del_std:.3f})")
        plt.fill_between(agg.frac, agg.mean_del - agg.std_del, agg.mean_del + agg.std_del, alpha=0.2)
    plt.xlabel("k / Kmax")
    plt.ylabel("Deletion: normalized drop")
    plt.title(title + " — Deletion")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_deletion.png"), dpi=200)
    plt.close()

    # Insertion plot
    plt.figure()
    for label, agg in agg_by_label.items():
        plt.plot(agg.frac, agg.mean_ins, label=f"{label} (AUC={agg.auc_ins_mean:.3f}±{agg.auc_ins_std:.3f})")
        plt.fill_between(agg.frac, agg.mean_ins - agg.std_ins, agg.mean_ins + agg.std_ins, alpha=0.2)
    plt.xlabel("k / Kmax")
    plt.ylabel("Insertion: normalized recovery")
    plt.title(title + " — Insertion")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_insertion.png"), dpi=200)
    plt.close()


# ----------------------------
# Main
# ----------------------------

def main():
    _ensure_pd()

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="", help="If set, chdir to this directory before loading CSVs.")
    ap.add_argument("--ckpt", type=str, default="", help="Path to local ckpt (required unless --mlflow_run_id).")
    ap.add_argument("--mlflow_run_id", type=str, default="", help="If set, download ckpt from MLflow run.")
    ap.add_argument("--mlflow_ckpt_relpath", type=str, default="model.ckpt", help="Artifact path inside run.")
    ap.add_argument("--device", type=str, default="cuda:0", help="e.g., cuda:0 or cpu")

    ap.add_argument("--end_date", type=str, default="2017-12-01")
    ap.add_argument("--num_months", type=int, default=11)
    ap.add_argument("--metric_num", type=str, default="comment")
    ap.add_argument("--metric_den", type=str, default="follower")
    ap.add_argument("--use_image_features", action="store_true")

    ap.add_argument("--target_nodes", type=str, default="", help="Comma-separated global node ids")
    ap.add_argument("--positions", type=str, default="", help="Comma-separated positions to explain (0-based)")

    # Importance source
    ap.add_argument("--importance_source", type=str, default="maskopt",
                    choices=["maskopt", "csv"],
                    help="maskopt: run MaskOpt to obtain df_feat/df_edge; csv: read existing xai CSVs.")
    ap.add_argument("--xai_feat_csv", type=str, default="", help="(csv mode) Path to features CSV for this case.")
    ap.add_argument("--xai_edge_csv", type=str, default="", help="(csv mode) Path to edges CSV for this case.")

    # Mask scope options (must match your explanation settings)
    ap.add_argument("--use_subgraph", action="store_true", help="Use k-hop subgraph around target node.")
    ap.add_argument("--num_hops", type=int, default=1)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--feat_mask_scope", type=str, default="all", choices=["all", "subgraph", "target"])
    ap.add_argument("--edge_mask_scope", type=str, default="incident", choices=["all", "subgraph", "incident"])
    ap.add_argument("--edge_grouping", type=str, default="neighbor", choices=["none", "neighbor"])

    # Curve params
    ap.add_argument("--k_max", type=int, default=100)
    ap.add_argument("--k_step", type=int, default=5)
    ap.add_argument("--random_repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)

    # Output
    ap.add_argument("--outdir", type=str, default="out_deletion_insertion")

    args = ap.parse_args()

    # Parse end_date (string) -> pandas.Timestamp
    try:
        end_date = pd.to_datetime(args.end_date)
    except Exception as e:
        raise SystemExit(f"--end_date must be parseable date string (e.g., 2017-12-31). got={args.end_date!r} ({e})")


    if args.data_dir:
        os.chdir(args.data_dir)

    os.makedirs(args.outdir, exist_ok=True)

    # --- graphs ---
    graphs_data = prepare_graph_data(
        end_date=end_date,
        num_months=args.num_months,
        metric_numerator=args.metric_num,
        metric_denominator=args.metric_den,
        use_image_features=args.use_image_features,
    )
    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, feature_names = _normalize_graphs_data(graphs_data)
    if feature_names is None:
        raise SystemExit("Could not infer feature_names from graphs_data. Ensure prepare_graph_data returns static/dynamic columns, or extend _normalize_graphs_data().")
    idx_to_node = _build_idx_to_node_from_node_to_idx(node_to_idx)

    # --- ckpt ---
    ckpt_path = args.ckpt
    if args.mlflow_run_id:
        ckpt_path = maybe_download_ckpt_from_mlflow(args.mlflow_run_id, args.mlflow_ckpt_relpath)
    if not ckpt_path:
        raise SystemExit("Need --ckpt or --mlflow_run_id")

    # --- model build (infer dims from ckpt state dict) ---
    import torch
    state = _load_ckpt_state_dict(ckpt_path, map_location="cpu")
    h = _infer_model_hparams_from_state_dict(state)

    feat_dim = int(monthly_graphs[0].x.shape[1])

    # Ensure feature_names length matches feat_dim (avoid index errors when labeling)
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(feat_dim)]
    elif len(feature_names) < feat_dim:
        feature_names = list(feature_names) + [f"feat_{i}" for i in range(len(feature_names), feat_dim)]
    elif len(feature_names) > feat_dim:
        feature_names = list(feature_names)[:feat_dim]

    model = HardResidualInfluencerModel(
        feature_dim=feat_dim,
        gcn_dim=h["gcn_dim"],
        rnn_dim=h["rnn_dim"],
        num_gcn_layers=h["num_gcn_layers"],
        dropout_prob=0.0,          # dropout has no parameters; any value is fine at eval
        projection_dim=h["projection_dim"],
    )

    device = args.device
    state_norm = _normalize_state_dict_keys(state, model)
    missing, unexpected = model.load_state_dict(state_norm, strict=False)
    if missing:
        print(f"[ckpt] Missing keys (showing up to 20): {missing[:20]}")
    if unexpected:
        print(f"[ckpt] Unexpected keys (showing up to 20): {unexpected[:20]}")
    model = model.to(device)
    model.eval()

    # Parse targets / positions
    target_nodes = _as_int_list(args.target_nodes)
    positions = _as_int_list(args.positions)
    if not target_nodes:
        raise SystemExit("Please provide --target_nodes (comma-separated)")
    if not positions:
        raise SystemExit("Please provide --positions (comma-separated)")

    # Degree baselines need per-month degree info
    # We'll compute degrees once per month from monthly_graphs[pos].edge_index
    def degree_for_month(pos: int) -> np.ndarray:
        ei = monthly_graphs[pos].edge_index
        if isinstance(ei, np.ndarray):
            src = ei[0]
            dst = ei[1]
            n = int(monthly_graphs[pos].num_nodes)
            deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
            return deg
        else:
            # torch tensor
            src = ei[0].detach().cpu().numpy()
            dst = ei[1].detach().cpu().numpy()
            n = int(monthly_graphs[pos].num_nodes)
            deg = np.bincount(src, minlength=n) + np.bincount(dst, minlength=n)
            return deg

    # Run per explain_pos (aggregate across nodes)
    summary_rows = []

    for pos in positions:
        # collect per-ordering results across nodes
        curves_top = []
        curves_rand_mean = []  # we will average random repeats per node, then aggregate across nodes
        curves_deg = []

        for node_id in target_nodes:
            # Build a wrapper for this case (used for prediction under modified gates)
            wrapper = E2EMaskOptWrapper(
                model=model,
                input_graphs=monthly_graphs,
                target_node=node_id,
                explain_pos=pos,
                idx_to_node=idx_to_node,
                local2global=None,
                use_subgraph=args.use_subgraph,
                num_hops=args.num_hops,
                undirected=args.undirected,
                feat_mask_scope=args.feat_mask_scope,
                edge_mask_scope=args.edge_mask_scope,
                edge_grouping=args.edge_grouping,
                device=device,
            )

            # --- get importance tables ---
            if args.importance_source == "maskopt":
                df_feat, df_edge, meta = maskopt_e2e_explain(
                    model=model,
                    input_graphs=monthly_graphs,
                    target_node=node_id,
                    explain_pos=pos,
                    feature_names=feature_names,
                    idx_to_node=idx_to_node,
                    use_subgraph=args.use_subgraph,
                    num_hops=args.num_hops,
                    undirected=args.undirected,
                    feat_mask_scope=args.feat_mask_scope,
                    edge_mask_scope=args.edge_mask_scope,
                    edge_grouping=args.edge_grouping,
                    device=device,
                )
                # Save raw importance tables (so you can reuse with --importance_source csv)
                df_feat.to_csv(os.path.join(args.outdir, f"xai_features_node_{node_id}_pos_{pos}.csv"), index=False)
                df_edge.to_csv(os.path.join(args.outdir, f"xai_edges_node_{node_id}_pos_{pos}.csv"), index=False)
            else:
                if not args.xai_feat_csv or not args.xai_edge_csv:
                    raise SystemExit("--importance_source csv requires --xai_feat_csv and --xai_edge_csv")
                df_feat = pd.read_csv(args.xai_feat_csv)
                df_edge = pd.read_csv(args.xai_edge_csv)

            # Map names to indices
            feature_name_to_idx = {n: i for i, n in enumerate(feature_names)}
            edge_name_to_idx = {n: i for i, n in enumerate(getattr(wrapper, "edge_group_names", []))}

            # Degree ordering for edge groups (by global node degree in month=pos)
            deg = degree_for_month(pos)
            degree_edge_idxs = None
            if hasattr(wrapper, "edge_group_meta") and wrapper.edge_group_meta is not None:
                # edge_group_meta contains neighbor_global
                # We'll rank edge groups by deg[neighbor_global] desc
                tmp = []
                for i, meta in enumerate(wrapper.edge_group_meta):
                    ng = meta.get("neighbor_global", None)
                    if ng is None:
                        continue
                    tmp.append((i, float(deg[int(ng)])))
                tmp.sort(key=lambda x: x[1], reverse=True)
                degree_edge_idxs = [i for (i, _) in tmp]

            orders = _build_orders_from_importance(
                df_feat=df_feat,
                df_edge=df_edge,
                feature_name_to_idx=feature_name_to_idx,
                edge_name_to_idx=edge_name_to_idx,
                random_repeats=args.random_repeats,
                seed=args.seed + (pos * 100000 + node_id) % 100000,
                degree_order_edge_idxs=degree_edge_idxs,
                combine_feat_edge=True,
            )

            # --- Evaluate TOP order ---
            curves_top.append(_eval_one_order(wrapper, orders.order_top, args.k_max, args.k_step, device=device))

            # --- Evaluate RANDOM baseline (average repeats for this node) ---
            rand_curves = [_eval_one_order(wrapper, od, args.k_max, args.k_step, device=device)
                           for od in orders.order_random_list]
            # average over repeats (same ks)
            del_mat = np.stack([r.deletion_drop_norm for r in rand_curves], axis=0)
            ins_mat = np.stack([r.insertion_rec_norm for r in rand_curves], axis=0)
            # store as a single CurveResult-like object (use mean curves; auc mean)
            rand_mean = CurveResult(
                ks=rand_curves[0].ks,
                frac=rand_curves[0].frac,
                pred_full=rand_curves[0].pred_full,
                pred_empty=rand_curves[0].pred_empty,
                deletion_drop_norm=del_mat.mean(axis=0),
                insertion_rec_norm=ins_mat.mean(axis=0),
                auc_del=float(np.mean([r.auc_del for r in rand_curves])),
                auc_ins=float(np.mean([r.auc_ins for r in rand_curves])),
            )
            curves_rand_mean.append(rand_mean)

            # --- Evaluate DEGREE baseline (if available) ---
            if orders.order_degree is not None:
                curves_deg.append(_eval_one_order(wrapper, orders.order_degree, args.k_max, args.k_step, device=device))

        # aggregate across nodes
        agg_top = _aggregate_curve(curves_top)
        agg_rand = _aggregate_curve(curves_rand_mean)

        agg_by_label = {
            "Top(Importance)": agg_top,
            "Random": agg_rand,
        }
        if len(curves_deg) == len(target_nodes) and len(curves_deg) > 0:
            agg_deg = _aggregate_curve(curves_deg)
            agg_by_label["Degree(Edge)"] = agg_deg

        # plot
        title = f"Deletion/Insertion pos={pos} (nodes={len(target_nodes)})"
        out_png = os.path.join(args.outdir, f"di_curves_pos_{pos}.png")
        _plot_aggregate(agg_by_label, title=title, out_png=out_png)

        # write summary
        for label, agg in agg_by_label.items():
            summary_rows.append(dict(
                pos=pos,
                label=label,
                k_max=args.k_max,
                k_step=args.k_step,
                auc_del_mean=agg.auc_del_mean,
                auc_del_std=agg.auc_del_std,
                auc_ins_mean=agg.auc_ins_mean,
                auc_ins_std=agg.auc_ins_std,
            ))

        # write curve CSV (mean ± std) for each label
        for label, agg in agg_by_label.items():
            out_csv = os.path.join(args.outdir, f"di_curve_pos_{pos}_{label.replace('(','').replace(')','').replace(' ','_')}.csv")
            df_out = pd.DataFrame({
                "k": agg.ks,
                "k_frac": agg.frac,
                "del_mean": agg.mean_del,
                "del_std": agg.std_del,
                "ins_mean": agg.mean_ins,
                "ins_std": agg.std_ins,
            })
            df_out.to_csv(out_csv, index=False)

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(args.outdir, "di_auc_summary.csv"), index=False)
    print("[write]", os.path.join(args.outdir, "di_auc_summary.csv"))
    print("[done] outdir =", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())