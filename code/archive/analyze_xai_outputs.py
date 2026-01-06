#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob
import pandas as pd
import numpy as np

OUTDIR = "xai_export_out"
SUMMARY = os.path.join(OUTDIR, "summary_dev_mps.csv")  # merged があるならそっちに変えてOK

def load_all(pattern):
    files = glob.glob(os.path.join(OUTDIR, pattern))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df["__file"] = os.path.basename(f)
            dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main():
    # --- summary ---
    s = pd.read_csv(SUMMARY)
    print("[summary] rows =", len(s))
    # 品質でフィルタ（必要なら閾値調整）
    s_ok = s.copy()
    s_ok = s_ok[s_ok["maskopt_best_fid"].fillna(1e9) < 1e-6]
    print("[summary] ok rows (fid<1e-6) =", len(s_ok))

    # --- features / neighbors ---
    feat = load_all("feat_tgt_*.csv")
    nbr  = load_all("neighbor_tgt_*.csv")
    print("[feat] rows =", len(feat), "files =", feat["__file"].nunique() if len(feat) else 0)
    print("[nbr ] rows =", len(nbr),  "files =", nbr["__file"].nunique() if len(nbr) else 0)

    # summaryで “成立した説明だけ” に絞る（TargetGlobalIdx+ExplainPos でJOIN）
    if len(s_ok) and len(feat):
        feat = feat.merge(
            s_ok[["TargetGlobalIdx","ExplainPos","maskopt_best_fid"]],
            on=["TargetGlobalIdx","ExplainPos"], how="inner"
        )
    if len(s_ok) and len(nbr):
        nbr = nbr.merge(
            s_ok[["TargetGlobalIdx","ExplainPos","maskopt_best_fid"]],
            on=["TargetGlobalIdx","ExplainPos"], how="inner"
        )

    # --- 集計：posごとの頻出特徴 ---
    if len(feat):
        g = feat.groupby(["ExplainPos","Name"]).agg(
            n=("Abs_Impact","size"),
            mean_abs=("Abs_Impact","mean"),
            median_abs=("Abs_Impact","median"),
            mean_imp=("Importance","mean"),
            mean_signed=("Score_Impact","mean"),
        ).reset_index()
        g = g.sort_values(["ExplainPos","mean_abs","n"], ascending=[True, False, False])
        out1 = os.path.join(OUTDIR, "agg_features_by_pos.csv")
        g.to_csv(out1, index=False)
        print("[write]", out1)

    # --- 集計：posごとの頻出近傍ノード ---
    if len(nbr):
        g2 = nbr.groupby(["ExplainPos","NeighborName"]).agg(
            n=("Abs_Impact","size"),
            mean_abs=("Abs_Impact","mean"),
            median_abs=("Abs_Impact","median"),
            mean_imp=("Importance","mean"),
            mean_signed=("Score_Impact","mean"),
            mean_group_edges=("GroupSizeEdges","mean"),
        ).reset_index()
        g2 = g2.sort_values(["ExplainPos","mean_abs","n"], ascending=[True, False, False])
        out2 = os.path.join(OUTDIR, "agg_neighbors_by_pos.csv")
        g2.to_csv(out2, index=False)
        print("[write]", out2)

    # --- ざっくり上位表示 ---
    if len(feat):
        print("\nTop features overall:")
        top = feat.groupby("Name")["Abs_Impact"].mean().sort_values(ascending=False).head(20)
        print(top)

    if len(nbr):
        print("\nTop neighbors overall:")
        top2 = nbr.groupby("NeighborName")["Abs_Impact"].mean().sort_values(ascending=False).head(20)
        print(top2)

if __name__ == "__main__":
    main()
