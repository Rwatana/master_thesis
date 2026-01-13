#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAI evaluation utilities for MaskOpt (deletion/insertion, stability, alignment, sensitivity).

This script is intentionally modular so you can import functions from v11.py without
re-training the model. Typical usage is to load a model + graphs, then call
`run_deletion_insertion_eval` or `run_stability_analysis`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

from v11 import E2EMaskOptWrapper, maskopt_e2e_explain


@dataclass
class CurveResult:
    steps: List[int]
    preds: List[float]
    aopc: float
    auc: float


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    return float(np.trapz(y, x))


def _aopc(values: np.ndarray, reference: float, mode: str) -> float:
    if values.size == 0:
        return 0.0
    if mode == "deletion":
        return float(np.mean(reference - values))
    if mode == "insertion":
        return float(np.mean(values - reference))
    raise ValueError(f"Unknown mode: {mode}")


def _build_order_from_gates(
    feat_gate: Optional[np.ndarray],
    edge_gate: Optional[np.ndarray],
    kind: str = "combined",
) -> List[Tuple[str, int, float]]:
    order = []
    if kind in ("combined", "feature") and feat_gate is not None:
        order.extend([("feature", int(i), float(v)) for i, v in enumerate(feat_gate)])
    if kind in ("combined", "edge") and edge_gate is not None:
        order.extend([("edge", int(i), float(v)) for i, v in enumerate(edge_gate)])
    order.sort(key=lambda t: t[2], reverse=True)
    return order


def _random_order(order: List[Tuple[str, int, float]], rng: random.Random) -> List[Tuple[str, int, float]]:
    order = list(order)
    rng.shuffle(order)
    return order


def compute_deletion_insertion_curves(
    wrapper: E2EMaskOptWrapper,
    feat_gate: Optional[np.ndarray],
    edge_gate: Optional[np.ndarray],
    order: List[Tuple[str, int, float]],
    mode: str = "deletion",
) -> CurveResult:
    if mode not in ("deletion", "insertion"):
        raise ValueError(f"Unknown mode: {mode}")

    feat_gate_t = torch.ones(wrapper.feature_dim, device=wrapper.device)
    if mode == "insertion":
        feat_gate_t = torch.zeros(wrapper.feature_dim, device=wrapper.device)

    if wrapper.edge_mask_scope == "incident":
        edge_dim = int(wrapper.incident_edge_idx.numel())
    else:
        edge_dim = int(wrapper.ei_exp.size(1))
    edge_gate_t = torch.ones(edge_dim, device=wrapper.device) if edge_dim > 0 else None
    if mode == "insertion" and edge_gate_t is not None:
        edge_gate_t = torch.zeros(edge_dim, device=wrapper.device)

    preds = []
    steps = []

    with torch.no_grad():
        base_pred = float(wrapper.predict_with_gates(feat_gate_t, edge_gate_t).item())
    preds.append(base_pred)
    steps.append(0)

    for step_idx, (kind, idx, _imp) in enumerate(order, start=1):
        if kind == "feature":
            feat_gate_t[idx] = 0.0 if mode == "deletion" else 1.0
        elif kind == "edge" and edge_gate_t is not None:
            edge_gate_t[idx] = 0.0 if mode == "deletion" else 1.0

        with torch.no_grad():
            pred = float(wrapper.predict_with_gates(feat_gate_t, edge_gate_t).item())
        preds.append(pred)
        steps.append(step_idx)

    preds_np = np.asarray(preds, dtype=float)
    steps_np = np.asarray(steps, dtype=float)
    reference = preds_np[0] if mode == "insertion" else preds_np[0]
    if mode == "deletion":
        reference = preds_np[0]
    else:
        reference = preds_np[0]
    return CurveResult(
        steps=steps,
        preds=preds,
        aopc=_aopc(preds_np, reference, mode),
        auc=_auc(steps_np, preds_np),
    )


def run_deletion_insertion_eval(
    model,
    input_graphs,
    target_node_idx,
    explain_pos,
    feature_names,
    rng_seed: int = 0,
    order_kind: str = "combined",
    topk: Optional[int] = None,
):
    df_feat, df_edge, meta, gates = maskopt_e2e_explain(
        model,
        input_graphs,
        target_node_idx=target_node_idx,
        explain_pos=explain_pos,
        feature_names=feature_names,
        node_to_idx=None,
        use_subgraph=True,
        num_hops=1,
        edge_mask_scope="incident",
        edge_grouping="neighbor",
        impact_reference="masked",
        use_contrastive=False,
        return_gates=True,
    )

    wrapper = E2EMaskOptWrapper(
        model=model,
        input_graphs=input_graphs,
        target_node_idx=target_node_idx,
        explain_pos=explain_pos,
        device=next(model.parameters()).device,
        use_subgraph=True,
        num_hops=1,
        undirected=True,
        feat_mask_scope="target",
        edge_mask_scope="incident",
    )

    order = _build_order_from_gates(gates["feat_gate"], gates["edge_gate"], kind=order_kind)
    if topk is not None:
        order = order[: int(topk)]

    rng = random.Random(rng_seed)
    rand_order = _random_order(order, rng)

    deletion = compute_deletion_insertion_curves(wrapper, gates["feat_gate"], gates["edge_gate"], order, mode="deletion")
    insertion = compute_deletion_insertion_curves(wrapper, gates["feat_gate"], gates["edge_gate"], order, mode="insertion")
    deletion_rand = compute_deletion_insertion_curves(wrapper, gates["feat_gate"], gates["edge_gate"], rand_order, mode="deletion")
    insertion_rand = compute_deletion_insertion_curves(wrapper, gates["feat_gate"], gates["edge_gate"], rand_order, mode="insertion")

    return {
        "meta": meta,
        "deletion": deletion,
        "insertion": insertion,
        "deletion_random": deletion_rand,
        "insertion_random": insertion_rand,
        "df_feat": df_feat,
        "df_edge": df_edge,
    }


def _topk_names(df: pd.DataFrame, k: int) -> List[str]:
    if df is None or df.empty:
        return []
    return df.sort_values("Importance", ascending=False).head(int(k))["Name"].astype(str).tolist()


def _rank_vector(df: pd.DataFrame, universe: List[str]) -> np.ndarray:
    ranks = {name: i for i, name in enumerate(_topk_names(df, len(df)), start=1)}
    worst = len(universe) + 1
    return np.asarray([ranks.get(name, worst) for name in universe], dtype=float)


def compute_stability(
    dfs: List[pd.DataFrame],
    topk: int = 20,
) -> Dict[str, float]:
    if len(dfs) < 2:
        return {"jaccard_mean": 0.0, "spearman_mean": 0.0}

    topk_sets = [set(_topk_names(df, topk)) for df in dfs]
    jaccards = []
    spearmans = []

    for i in range(len(dfs)):
        for j in range(i + 1, len(dfs)):
            a = topk_sets[i]
            b = topk_sets[j]
            if not a and not b:
                jaccards.append(1.0)
            else:
                jaccards.append(len(a & b) / max(1, len(a | b)))

            universe = sorted(list(a | b))
            if len(universe) < 2:
                spearmans.append(0.0)
            else:
                r1 = _rank_vector(dfs[i], universe)
                r2 = _rank_vector(dfs[j], universe)
                spearmans.append(float(spearmanr(r1, r2).correlation))

    return {
        "jaccard_mean": float(np.mean(jaccards)) if jaccards else 0.0,
        "spearman_mean": float(np.mean(spearmans)) if spearmans else 0.0,
    }


def compute_importance_impact_alignment(df: pd.DataFrame, impact_key: str = "Score_Impact(masked)") -> Dict[str, float]:
    if df is None or df.empty:
        return {"spearman": 0.0, "topk_overlap": 0.0}

    if impact_key not in df.columns:
        for candidate in ["Score_Impact(unmasked)", "Score_Impact(masked)"]:
            if candidate in df.columns:
                impact_key = candidate
                break

    if impact_key not in df.columns:
        return {"spearman": 0.0, "topk_overlap": 0.0}

    importance = df["Importance"].astype(float).values
    impact = df[impact_key].astype(float).abs().values
    corr = spearmanr(importance, impact).correlation

    k = min(20, len(df))
    top_imp = set(df.sort_values("Importance", ascending=False).head(k)["Name"].astype(str))
    top_impact = set(df.assign(abs_impact=impact).sort_values("abs_impact", ascending=False).head(k)["Name"].astype(str))

    overlap = len(top_imp & top_impact) / max(1, len(top_imp | top_impact))
    return {"spearman": float(corr), "topk_overlap": float(overlap)}


def run_maskopt_stability(
    model,
    input_graphs,
    target_node_idx,
    explain_pos,
    feature_names,
    num_runs: int = 5,
    topk: int = 20,
    base_seed: int = 0,
) -> Dict[str, float]:
    feat_runs = []
    edge_runs = []

    for k in range(num_runs):
        seed = base_seed + k
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        df_feat, df_edge, _meta = maskopt_e2e_explain(
            model,
            input_graphs,
            target_node_idx=target_node_idx,
            explain_pos=explain_pos,
            feature_names=feature_names,
            node_to_idx=None,
            use_subgraph=True,
            num_hops=1,
            edge_mask_scope="incident",
            edge_grouping="neighbor",
            impact_reference="masked",
            use_contrastive=False,
        )
        feat_runs.append(df_feat)
        edge_runs.append(df_edge)

    feat_stats = compute_stability(feat_runs, topk=topk)
    edge_stats = compute_stability(edge_runs, topk=topk)
    return {
        "feature_jaccard": feat_stats["jaccard_mean"],
        "feature_spearman": feat_stats["spearman_mean"],
        "edge_jaccard": edge_stats["jaccard_mean"],
        "edge_spearman": edge_stats["spearman_mean"],
    }


def run_score_impact_sensitivity(
    model,
    input_graphs,
    target_node_idx,
    explain_pos,
    feature_names,
    baselines: Iterable[str],
    rhos: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for baseline in baselines:
        for rho in rhos:
            df_feat, df_edge, meta = maskopt_e2e_explain(
                model,
                input_graphs,
                target_node_idx=target_node_idx,
                explain_pos=explain_pos,
                feature_names=feature_names,
                node_to_idx=None,
                use_subgraph=True,
                num_hops=1,
                edge_mask_scope="incident",
                edge_grouping="neighbor",
                impact_reference="masked",
                ablation_mode="baseline_mix",
                impact_baseline=baseline,
                impact_rho=float(rho),
                use_contrastive=False,
            )
            feat_align = compute_importance_impact_alignment(df_feat)
            edge_align = compute_importance_impact_alignment(df_edge)
            rows.append({
                "impact_baseline": baseline,
                "impact_rho": float(rho),
                "feat_spearman": feat_align["spearman"],
                "feat_topk_overlap": feat_align["topk_overlap"],
                "edge_spearman": edge_align["spearman"],
                "edge_topk_overlap": edge_align["topk_overlap"],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("This module provides XAI evaluation utilities. Import and call functions from your experiment script.")
