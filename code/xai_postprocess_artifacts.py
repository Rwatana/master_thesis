#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocess MLflow artifacts into "paper-ready" figures (no torch needed).

Inputs:
  - artifact_dir: MLflow run artifacts root directory
    e.g. mlruns/<exp_id>/<run_id>/artifacts

Outputs:
  - scatter plots (importance vs score impact)
  - month x feature heatmaps
  - top-k trend plots
  - gate histograms (from NPZ) if available
  - simple zero-diagnosis text

Run:
  python xai_postprocess_artifacts.py --artifact_dir /path/to/.../artifacts --outdir paper_figs --impact_col "Score_Impact(masked)"
"""

import argparse
import os
import glob
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _load_csvs(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    files = sorted(set(files))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__srcfile"] = os.path.basename(f)
            dfs.append(df)
        except Exception:
            pass
    return files, (pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame())

def _maybe_pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def scatter_importance_vs_impact(df, out_png, impact_col, title):
    if df.empty:
        return
    if ("Importance" not in df.columns) or (impact_col not in df.columns):
        return

    x = df["Importance"].astype(float).to_numpy()
    y = df[impact_col].astype(float).to_numpy()

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size == 0:
        return

    plt.figure()
    plt.scatter(x, y, s=10)
    plt.xlabel("Importance (gate value)")
    plt.ylabel(impact_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def heatmap_month_feature(df_feat, out_png, value_col, top_features=30, title=""):
    if df_feat.empty:
        return
    if ("Name" not in df_feat.columns) or (value_col not in df_feat.columns):
        return

    month_col = None
    for c in ["explain_pos", "month_label", "pos", "tag", "label"]:
        if c in df_feat.columns:
            month_col = c
            break
    if month_col is None:
        return

    d = df_feat.copy()
    d[value_col] = d[value_col].astype(float)

    agg = d.groupby("Name")[value_col].apply(lambda s: float(np.nanmean(np.abs(s)))).sort_values(ascending=False)
    keep = agg.head(int(top_features)).index.tolist()
    d = d[d["Name"].isin(keep)]

    pv = d.pivot_table(index=month_col, columns="Name", values=value_col, aggfunc="mean").fillna(0.0)

    plt.figure(figsize=(max(10, pv.shape[1]*0.35), max(6, pv.shape[0]*0.6)))
    plt.imshow(pv.values, aspect="auto")
    plt.yticks(range(pv.shape[0]), [str(i) for i in pv.index])
    plt.xticks(range(pv.shape[1]), pv.columns, rotation=90)
    plt.colorbar(label=value_col)
    plt.title(title or f"Heatmap: {value_col}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def influencer_topk_trend(df_feat, out_png, value_col="Importance", k=10, title=""):
    if df_feat.empty:
        return
    if ("Name" not in df_feat.columns) or (value_col not in df_feat.columns):
        return

    month_col = "explain_pos" if "explain_pos" in df_feat.columns else ("pos" if "pos" in df_feat.columns else None)
    if month_col is None:
        return

    target_col = "target_node" if "target_node" in df_feat.columns else None
    if target_col is None:
        groups = [("single", df_feat)]
    else:
        groups = [(str(t), g) for t, g in df_feat.groupby(target_col)]

    for tname, g in groups:
        agg = g.groupby("Name")[value_col].apply(lambda s: float(np.nanmean(np.abs(s)))).sort_values(ascending=False)
        keep = agg.head(int(k)).index.tolist()
        gg = g[g["Name"].isin(keep)].copy()
        pv = gg.pivot_table(index=month_col, columns="Name", values=value_col, aggfunc="mean").fillna(0.0).sort_index()

        plt.figure(figsize=(12, 6))
        for col in pv.columns:
            plt.plot(pv.index.astype(int), pv[col].to_numpy(), label=col)
        plt.xlabel(month_col)
        plt.ylabel(value_col)
        plt.title(title or f"Top-{k} feature trends | target={tname}")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        op = out_png.replace(".png", f"_target_{tname}.png") if target_col else out_png
        plt.savefig(op, dpi=200)
        plt.close()

def gate_hist_from_npz(npz_files, out_png_prefix):
    for f in npz_files:
        try:
            z = np.load(f, allow_pickle=True)
            feat = z["feat_gate"]
            edge = z["edge_gate"] if "edge_gate" in z.files else None

            if feat is not None and feat is not np.array(None):
                feat = np.asarray(feat, dtype=np.float32)
                plt.figure()
                plt.hist(feat, bins=50)
                plt.xlabel("feat_gate value")
                plt.ylabel("count")
                plt.title(os.path.basename(f) + " | feat_gate")
                plt.tight_layout()
                plt.savefig(out_png_prefix + "_" + os.path.basename(f).replace(".npz","") + "_feat.png", dpi=200)
                plt.close()

            if edge is not None and edge is not np.array(None):
                edge = np.asarray(edge, dtype=np.float32)
                plt.figure()
                plt.hist(edge, bins=50)
                plt.xlabel("edge_gate value")
                plt.ylabel("count")
                plt.title(os.path.basename(f) + " | edge_gate")
                plt.tight_layout()
                plt.savefig(out_png_prefix + "_" + os.path.basename(f).replace(".npz","") + "_edge.png", dpi=200)
                plt.close()
        except Exception:
            continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact_dir", required=True, help="MLflow run artifacts root dir (contains xai/ plots/ etc.)")
    ap.add_argument("--outdir", default="xai_paper_figs")
    ap.add_argument("--impact_col", default=None, help="e.g., 'Score_Impact(masked)'")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    feat_files, df_feat = _load_csvs([
        os.path.join(args.artifact_dir, "**", "maskopt_feat_*.csv"),
        os.path.join(args.artifact_dir, "**", "maskopt_feature_all_*.csv"),
    ])
    edge_files, df_edge = _load_csvs([
        os.path.join(args.artifact_dir, "**", "maskopt_edge_*.csv"),
        os.path.join(args.artifact_dir, "**", "maskopt_edge_all_*.csv"),
    ])

    impact_col = args.impact_col
    if impact_col is None:
        impact_col = _maybe_pick_col(df_feat, ["Score_Impact(masked)", "Score_Impact(unmasked)", "Score_Impact", "score_impact"])
    if impact_col is None:
        impact_col = "Score_Impact(masked)"

    scatter_importance_vs_impact(
        df_feat, os.path.join(args.outdir, "scatter_feat_importance_vs_impact.png"),
        impact_col=impact_col,
        title=f"Feature: Importance vs {impact_col}",
    )

    impact_col_e = _maybe_pick_col(df_edge, [impact_col, "Score_Impact(masked)", "Score_Impact(unmasked)", "Score_Impact"])
    if impact_col_e is None:
        impact_col_e = "Score_Impact(masked)"
    scatter_importance_vs_impact(
        df_edge, os.path.join(args.outdir, "scatter_edge_importance_vs_impact.png"),
        impact_col=impact_col_e,
        title=f"Edge: Importance vs {impact_col_e}",
    )

    heatmap_month_feature(
        df_feat, os.path.join(args.outdir, "heatmap_month_feature_importance.png"),
        value_col="Importance",
        top_features=30,
        title="Month x Feature (Importance)",
    )

    if impact_col in df_feat.columns:
        heatmap_month_feature(
            df_feat, os.path.join(args.outdir, "heatmap_month_feature_impact.png"),
            value_col=impact_col,
            top_features=30,
            title=f"Month x Feature ({impact_col})",
        )

    influencer_topk_trend(
        df_feat, os.path.join(args.outdir, "trend_topk_features.png"),
        value_col="Importance",
        k=10,
        title="Top-K feature trends",
    )

    npz_files = sorted(glob.glob(os.path.join(args.artifact_dir, "**", "maskopt_gates_*.npz"), recursive=True))
    if npz_files:
        gate_hist_from_npz(npz_files, os.path.join(args.outdir, "hist"))

    if impact_col in df_feat.columns:
        y = df_feat[impact_col].astype(float).to_numpy()
        y = y[np.isfinite(y)]
        if y.size:
            zero_rate = float(np.mean(np.abs(y) == 0.0))
            with open(os.path.join(args.outdir, "zero_diagnosis.txt"), "w", encoding="utf-8") as f:
                f.write(f"impact_col={impact_col}\n")
                f.write(f"n={y.size}\n")
                f.write(f"zero_rate(abs==0)={zero_rate:.6f}\n")
                f.write(f"abs_min={float(np.min(np.abs(y))):.8e}\n")
                f.write(f"abs_median={float(np.median(np.abs(y))):.8e}\n")
                f.write(f"abs_max={float(np.max(np.abs(y))):.8e}\n")

    print(f"[OK] wrote figures to: {args.outdir}")
    print(f"[info] loaded feat_csv={len(feat_files)} edge_csv={len(edge_files)} gates_npz={len(npz_files)}")

if __name__ == "__main__":
    main()
