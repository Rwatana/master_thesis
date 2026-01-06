#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess MLflow artifacts (NO-IMAGE VERSION)

Purpose
-------
- Re-aggregate existing run artifacts WITHOUT re-training
- Produce "paper-ready" numeric outputs only (CSV/JSON/TXT)
- No PNG generation in this script

What it produces
----------------
In --outdir:
  - summary_runs.csv                        : run-level metrics + chosen months + zero ratios
  - long_xai_features.csv                   : all maskopt feature rows (with run_id/tag/pos/node)
  - long_xai_edges.csv                      : all maskopt edge rows (with run_id/tag/pos/node)
  - gate_hist_features.csv                  : histogram bins per run/tag
  - gate_hist_edges.csv                     : histogram bins per run/tag
  - heatmap_table_month_x_feature.csv       : month (pos) × feature aggregated importance (mean/sum)
  - topk_feature_trajectory.csv             : influencer/topk evolution table (for later plotting)
  - zero_hell_breakdown.csv                 : %zeros by ref kind for each explained month

You can later make figures in a separate plotting step, but this keeps "一旦画像なし".
"""

import argparse
import os
import re
import json
from pathlib import Path
import numpy as np
import pandas as pd

def _find_files(root: Path, pattern: str):
    return list(root.rglob(pattern))

def _parse_run_id_from_path(p: Path):
    # mlruns/<exp_id>/<run_id>/artifacts/...
    parts = p.parts
    for i in range(len(parts)-1):
        if re.fullmatch(r"[0-9a-f]{32}", parts[i]):
            return parts[i]
    # fallback: scan
    m = re.search(r"([0-9a-f]{32})", str(p))
    return m.group(1) if m else "unknown_run"

def _parse_node_pos_from_name(name: str):
    # maskopt_feat_pos_3_node_916490_pos_3.csv  (tag includes pos, then explicit pos again)
    node = None
    pos = None
    m = re.search(r"node_(\d+)", name)
    if m: node = int(m.group(1))
    m = re.search(r"pos_(\d+)\.csv$", name)
    if m: pos = int(m.group(1))
    return node, pos

def _hist(series: np.ndarray, bins=30):
    series = np.asarray(series, dtype=float)
    series = series[np.isfinite(series)]
    if series.size == 0:
        return None, None
    h, edges = np.histogram(series, bins=bins, range=(0.0, 1.0))
    return h.astype(int), edges.astype(float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlruns", type=str, required=True, help="Path to mlruns or mlruns_artifacts root")
    ap.add_argument("--outdir", type=str, default="xai_paper_tables", help="Output directory")
    ap.add_argument("--bins", type=int, default=30, help="Histogram bins for gates")
    args = ap.parse_args()

    root = Path(args.mlruns).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect artifacts
    feat_files = _find_files(root, "maskopt_feat_*.csv")
    edge_files = _find_files(root, "maskopt_edge_*.csv")
    meta_files = _find_files(root, "maskopt_meta_*.json")
    gate_files = _find_files(root, "maskopt_gates_*.npz")
    diag_files = _find_files(root, "xai_zero_diag_*.csv")
    pred_tables = _find_files(root, "pred_table_*.csv")

    # ---- Long tables: feat/edge ----
    feat_rows = []
    for fp in feat_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        run_id = _parse_run_id_from_path(fp)
        node, pos = _parse_node_pos_from_name(fp.name)
        df.insert(0, "run_id", run_id)
        df.insert(1, "target_node", node if node is not None else np.nan)
        df.insert(2, "pos", pos if pos is not None else np.nan)
        df.insert(3, "tag", fp.name.split("_node_")[0].replace("maskopt_feat_", ""))
        feat_rows.append(df)
    df_feat_long = pd.concat(feat_rows, ignore_index=True) if feat_rows else pd.DataFrame()
    df_feat_long.to_csv(outdir / "long_xai_features.csv", index=False)

    edge_rows = []
    for fp in edge_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        run_id = _parse_run_id_from_path(fp)
        node, pos = _parse_node_pos_from_name(fp.name)
        df.insert(0, "run_id", run_id)
        df.insert(1, "target_node", node if node is not None else np.nan)
        df.insert(2, "pos", pos if pos is not None else np.nan)
        df.insert(3, "tag", fp.name.split("_node_")[0].replace("maskopt_edge_", ""))
        edge_rows.append(df)
    df_edge_long = pd.concat(edge_rows, ignore_index=True) if edge_rows else pd.DataFrame()
    df_edge_long.to_csv(outdir / "long_xai_edges.csv", index=False)

    # ---- Gate histograms ----
    hist_feat_rows = []
    hist_edge_rows = []
    for fp in gate_files:
        run_id = _parse_run_id_from_path(fp)
        node, pos = _parse_node_pos_from_name(fp.name.replace(".npz", ".csv"))
        tag = fp.name.split("_node_")[0].replace("maskopt_gates_", "")
        try:
            z = np.load(fp, allow_pickle=True)
        except Exception:
            continue
        fg = z.get("feat_gate", None)
        eg = z.get("edge_gate", None)

        if fg is not None and fg.size > 0:
            h, edges = _hist(fg, bins=args.bins)
            if h is not None:
                for i in range(len(h)):
                    hist_feat_rows.append({
                        "run_id": run_id, "target_node": node, "pos": pos, "tag": tag,
                        "bin_left": float(edges[i]), "bin_right": float(edges[i+1]),
                        "count": int(h[i]),
                    })

        if eg is not None and eg.size > 0:
            h, edges = _hist(eg, bins=args.bins)
            if h is not None:
                for i in range(len(h)):
                    hist_edge_rows.append({
                        "run_id": run_id, "target_node": node, "pos": pos, "tag": tag,
                        "bin_left": float(edges[i]), "bin_right": float(edges[i+1]),
                        "count": int(h[i]),
                    })

    pd.DataFrame(hist_feat_rows).to_csv(outdir / "gate_hist_features.csv", index=False)
    pd.DataFrame(hist_edge_rows).to_csv(outdir / "gate_hist_edges.csv", index=False)

    # ---- Month×Feature “heatmap table” (numeric) ----
    # aggregate Importance by (run_id, target_node, pos, Name)
    if not df_feat_long.empty:
        df_hm = (df_feat_long
                 .groupby(["run_id", "target_node", "pos", "Name"], as_index=False)
                 .agg(importance_mean=("Importance", "mean"),
                      importance_sum=("Importance", "sum")))
        df_hm.to_csv(outdir / "heatmap_table_month_x_feature.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "heatmap_table_month_x_feature.csv", index=False)

    # ---- Influencerごとの “重要特徴トップkの推移” 用テーブル ----
    # Here we create a simple "topK per pos" table for each (run_id,target_node,pos).
    if not df_feat_long.empty:
        topk = 10
        traj_rows = []
        for (run_id, node), g in df_feat_long.groupby(["run_id", "target_node"]):
            gg = g.dropna(subset=["pos"]).copy()
            for pos, gp in gg.groupby("pos"):
                gp = gp.sort_values("Importance", ascending=False).head(topk)
                for rank, (_, row) in enumerate(gp.iterrows(), start=1):
                    traj_rows.append({
                        "run_id": run_id,
                        "target_node": node,
                        "pos": int(pos),
                        "rank": rank,
                        "feature": row.get("Namename", row.get("Name")),
                        "importance": float(row.get("Importance")),
                        # include score impact cols if present
                        "score_impact_masked": float(row.get("Score_Impact(masked)", np.nan)) if "Score_Impact(masked)" in row else np.nan,
                        "score_impact_unmasked": float(row.get("Score_Impact(unmasked)", np.nan)) if "Score_Impact(unmasked)" in row else np.nan,
                    })
        pd.DataFrame(traj_rows).to_csv(outdir / "topk_feature_trajectory.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "topk_feature_trajectory.csv", index=False)

    # ---- Zero-hell breakdown ----
    diag_rows = []
    for fp in diag_files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        run_id = _parse_run_id_from_path(fp)
        node, pos = _parse_node_pos_from_name(fp.name)
        df.insert(0, "run_id", run_id)
        df.insert(1, "target_node", node if node is not None else np.nan)
        df.insert(2, "pos", pos if pos is not None else np.nan)
        diag_rows.append(df)
    df_diag_long = pd.concat(diag_rows, ignore_index=True) if diag_rows else pd.DataFrame()
    df_diag_long.to_csv(outdir / "zero_hell_breakdown.csv", index=False)

    # ---- Run summary ----
    # We cannot reliably read mlflow metrics without mlflow client config here;
    # instead we summarize what artifacts exist.
    sum_rows = []
    by_run = {}
    for fp in meta_files:
        run_id = _parse_run_id_from_path(fp)
        try:
            meta = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        by_run.setdefault(run_id, []).append(meta)

    for run_id, metas in by_run.items():
        metas = sorted(metas, key=lambda m: int(m.get("explain_pos", 9999)))
        positions = [int(m.get("explain_pos")) for m in metas if "explain_pos" in m]
        tgt_nodes = sorted({int(m.get("target_node")) for m in metas if "target_node" in m})
        impact_refs = sorted({str(m.get("impact_reference")) for m in metas if "impact_reference" in m})

        sum_rows.append({
            "run_id": run_id,
            "target_nodes": ",".join(map(str, tgt_nodes)),
            "explained_positions": ",".join(map(str, positions)),
            "impact_references": ",".join(impact_refs),
            "n_xai_meta": len(metas),
        })

    df_sum = pd.DataFrame(sum_rows).sort_values("run_id") if sum_rows else pd.DataFrame()
    df_sum.to_csv(outdir / "summary_runs.csv", index=False)

    print("[write]", outdir / "long_xai_features.csv")
    print("[write]", outdir / "long_xai_edges.csv")
    print("[write]", outdir / "gate_hist_features.csv")
    print("[write]", outdir / "gate_hist_edges.csv")
    print("[write]", outdir / "heatmap_table_month_x_feature.csv")
    print("[write]", outdir / "topk_feature_trajectory.csv")
    print("[write]", outdir / "zero_hell_breakdown.csv")
    print("[write]", outdir / "summary_runs.csv")
    print("\nDone.")

if __name__ == "__main__":
    raise SystemExit(main())
