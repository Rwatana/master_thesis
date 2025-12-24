#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
paperize_xai_outputs.py

目的:
  1) xai_export_out 配下にある MaskOpt 出力（summary / feat_*.csv / neighbor_*.csv）を集計
  2) NeighborName が "nbr_12345" などのプレースホルダの場合、NeighborGlobalIdx を使って「実名」に復元
  3) 論文に貼れる図 (PNG/PDF) と、表 (CSV/LaTeX) を自動生成

前提:
  - export_maskopt_all_importance_faster_v3.py の出力を想定
  - Neighbor 実名復元を行う場合は、元データファイル(投稿/hashtags/mentions)が必要

実行例:
  # すでに実名が CSV に入っている場合（復元不要）
  python paperize_xai_outputs.py --xai-dir xai_export_out --no-recover-names

  # "nbr_XXXX" を実名に戻したい場合（重い処理）
  python paperize_xai_outputs.py \
    --xai-dir xai_export_out \
    --posts-file dataset_A_active_all.csv \
    --hashtags-file hashtags_2017.csv \
    --mentions-file mentions_2017.csv

出力:
  xai_export_out/paper/
    - agg_features_by_pos_named.csv
    - agg_neighbors_by_pos_named.csv
    - tables/*.tex
    - figs/*.png , figs/*.pdf
    - figure_captions_ja.md
"""

import argparse
import os
import glob
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------
# Utilities
# ------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _is_placeholder_name(s: str) -> bool:
    if s is None:
        return True
    s = str(s)
    return bool(re.fullmatch(r"(nbr|node|idx)_[0-9]+", s))

def _short_label(s: str, maxlen: int = 28) -> str:
    s = "" if s is None else str(s)
    if len(s) <= maxlen:
        return s
    return s[: max(0, maxlen - 1)] + "…"

def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """(pearson, spearman). Avoid numpy warnings when std=0 or len<2."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return (float("nan"), float("nan"))
    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return (float("nan"), float("nan"))
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    if np.nanstd(rx) == 0.0 or np.nanstd(ry) == 0.0:
        spearman = float("nan")
    else:
        spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman

def _save_fig(fig: plt.Figure, out_png: Path, out_pdf: Path, dpi: int = 300) -> None:
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_pdf)  # PDF は dpi 指定不要
    plt.close(fig)


# ------------------------
# Load XAI outputs
# ------------------------
def load_xai_outputs(xai_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_summary, df_feat_all, df_nbr_all
    """
    # summary candidates
    sum_paths = sorted(glob.glob(str(xai_dir / "summary*.csv")))
    if not sum_paths:
        raise FileNotFoundError(f"summary*.csv not found under: {xai_dir}")

    # prefer merged if exists
    sum_path = None
    for p in sum_paths:
        if p.endswith("summary_merged.csv"):
            sum_path = p
            break
    if sum_path is None:
        sum_path = sum_paths[0]

    df_summary = pd.read_csv(sum_path)
    # normalize column names
    if "maskopt_best_fid" not in df_summary.columns and "maskopt_best_fid" in df_summary.columns:
        pass

    feat_paths = sorted(glob.glob(str(xai_dir / "feat_*.csv")))
    nbr_paths  = sorted(glob.glob(str(xai_dir / "neighbor_*.csv")))

    if not feat_paths:
        raise FileNotFoundError(f"feat_*.csv not found under: {xai_dir}")
    if not nbr_paths:
        raise FileNotFoundError(f"neighbor_*.csv not found under: {xai_dir}")

    df_feat_all = pd.concat((pd.read_csv(p) for p in feat_paths), ignore_index=True)
    df_nbr_all  = pd.concat((pd.read_csv(p) for p in nbr_paths), ignore_index=True)

    return df_summary, df_feat_all, df_nbr_all


# ------------------------
# Recover real names for nodes
# ------------------------
def build_node_to_idx(
    posts_file: Path,
    hashtags_file: Path,
    mentions_file: Path,
    dec_year: int = 2017,
    dec_month: int = 12,
    chunksize: int = 1_000_000,
) -> Dict[str, int]:
    """
    prepare_graph_data と同じ node_to_idx を再現して、node_to_idx(dict: name -> idx) を返します。
    ※とても重い処理です（ファイルが巨大）。
    """
    # 1) valid users in Dec YYYY-MM
    start = pd.Timestamp(f"{dec_year}-{dec_month:02d}-01")
    end = (start + pd.offsets.MonthEnd(1)).replace(hour=23, minute=59, second=59)
    valid_users = set()

    usecols = ["username", "datetime"]
    for chunk in pd.read_csv(posts_file, usecols=usecols, parse_dates=["datetime"], chunksize=chunksize, low_memory=True):
        chunk["username"] = chunk["username"].astype(str).str.strip()
        m = (chunk["datetime"] >= start) & (chunk["datetime"] <= end)
        if m.any():
            valid_users.update(chunk.loc[m, "username"].dropna().astype(str).str.strip().unique().tolist())
    valid_users = set(u for u in valid_users if u != "")

    if len(valid_users) == 0:
        raise RuntimeError("valid_users in Dec is empty. Check posts_file datetime range/column names.")

    # influencer_set in your pipeline is essentially valid_users_dec
    influencer_set = set(valid_users)

    # 2) hashtags used by valid users
    all_hashtags = set()
    if hashtags_file.exists():
        # tolerate different column names
        # expected: source,target,timestamp
        for chunk in pd.read_csv(hashtags_file, chunksize=chunksize, low_memory=True):
            # try to normalize
            if "source" in chunk.columns and "target" in chunk.columns:
                src = chunk["source"].astype(str).str.strip()
                tgt = chunk["target"].astype(str).str.strip()
            elif "username" in chunk.columns and "hashtag" in chunk.columns:
                src = chunk["username"].astype(str).str.strip()
                tgt = chunk["hashtag"].astype(str).str.strip()
            else:
                continue
            m = src.isin(valid_users)
            if m.any():
                all_hashtags.update(tgt[m].dropna().astype(str).str.strip().unique().tolist())

    # 3) mentions used by valid users
    all_mentions = set()
    if mentions_file.exists():
        for chunk in pd.read_csv(mentions_file, chunksize=chunksize, low_memory=True):
            if "source" in chunk.columns and "target" in chunk.columns:
                src = chunk["source"].astype(str).str.strip()
                tgt = chunk["target"].astype(str).str.strip()
            elif "username" in chunk.columns and "mention" in chunk.columns:
                src = chunk["username"].astype(str).str.strip()
                tgt = chunk["mention"].astype(str).str.strip()
            else:
                continue
            m = src.isin(valid_users)
            if m.any():
                all_mentions.update(tgt[m].dropna().astype(str).str.strip().unique().tolist())

    # 4) objects: your recent pipeline often has 0; ignore here (safe)
    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    return node_to_idx


def invert_node_to_idx_for_needed(node_to_idx: Dict[str, int], needed: Iterable[int]) -> Dict[int, str]:
    needed_set = set(int(x) for x in needed if x is not None and str(x) != "nan")
    inv: Dict[int, str] = {}
    if not needed_set:
        return inv
    # iterate once through dict and pick only needed indices (fast enough)
    for name, idx in node_to_idx.items():
        if idx in needed_set:
            inv[idx] = name
    return inv


def apply_name_recovery(
    df_feat_all: pd.DataFrame,
    df_nbr_all: pd.DataFrame,
    df_summary: pd.DataFrame,
    idx_to_name: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    NeighborName / TargetName が placeholder の場合に idx_to_name で実名に置換する
    """
    df_feat = df_feat_all.copy()
    df_nbr  = df_nbr_all.copy()
    df_sum  = df_summary.copy()

    # targets
    if "TargetGlobalIdx" in df_feat.columns and "TargetName" in df_feat.columns:
        m = df_feat["TargetName"].map(_is_placeholder_name)
        if m.any():
            df_feat.loc[m, "TargetName"] = df_feat.loc[m, "TargetGlobalIdx"].map(lambda i: idx_to_name.get(int(i), df_feat.loc[m, "TargetName"])).values

    if "TargetGlobalIdx" in df_nbr.columns and "TargetName" in df_nbr.columns:
        m = df_nbr["TargetName"].map(_is_placeholder_name)
        if m.any():
            df_nbr.loc[m, "TargetName"] = df_nbr.loc[m, "TargetGlobalIdx"].map(lambda i: idx_to_name.get(int(i), df_nbr.loc[m, "TargetName"])).values

    if "TargetGlobalIdx" in df_sum.columns and "TargetName" in df_sum.columns:
        m = df_sum["TargetName"].map(_is_placeholder_name)
        if m.any():
            df_sum.loc[m, "TargetName"] = df_sum.loc[m, "TargetGlobalIdx"].map(lambda i: idx_to_name.get(int(i), df_sum.loc[m, "TargetName"])).values

    # neighbors
    if "NeighborGlobalIdx" in df_nbr.columns:
        m = df_nbr["NeighborName"].map(_is_placeholder_name)
        if m.any():
            df_nbr.loc[m, "NeighborName"] = df_nbr.loc[m, "NeighborGlobalIdx"].map(lambda i: idx_to_name.get(int(i), df_nbr.loc[m, "NeighborName"])).values

    return df_feat, df_nbr, df_sum


# ------------------------
# Aggregation
# ------------------------
def aggregate_features_by_pos(df_feat: pd.DataFrame) -> pd.DataFrame:
    g = df_feat.groupby(["ExplainPos", "Name"], as_index=False).agg(
        mean_abs_impact=("Abs_Impact", "mean"),
        sum_abs_impact=("Abs_Impact", "sum"),
        mean_importance=("Importance", "mean"),
        n=("Abs_Impact", "size"),
    )
    g.sort_values(["ExplainPos", "mean_abs_impact"], ascending=[True, False], inplace=True)
    return g

def aggregate_neighbors_by_pos(df_nbr: pd.DataFrame) -> pd.DataFrame:
    name_col = "NeighborName" if "NeighborName" in df_nbr.columns else "Name"
    g = df_nbr.groupby(["ExplainPos", name_col], as_index=False).agg(
        mean_abs_impact=("Abs_Impact", "mean"),
        sum_abs_impact=("Abs_Impact", "sum"),
        mean_importance=("Importance", "mean"),
        n=("Abs_Impact", "size"),
    )
    g.rename(columns={name_col: "NeighborName"}, inplace=True)
    g.sort_values(["ExplainPos", "mean_abs_impact"], ascending=[True, False], inplace=True)
    return g


# ------------------------
# Figures (matplotlib only)
# ------------------------
def fig_fidelity_hist(df_summary: pd.DataFrame, outdir: Path, fid_thr: float = 1e-6, dpi: int = 300) -> None:
    if "maskopt_best_fid" not in df_summary.columns:
        return
    fid = pd.to_numeric(df_summary["maskopt_best_fid"], errors="coerce").dropna().to_numpy()
    if fid.size == 0:
        return
    fig = plt.figure(figsize=(6.0, 3.8))
    ax = fig.add_subplot(111)
    ax.hist(np.log10(fid + 1e-30), bins=40)
    ax.axvline(np.log10(fid_thr), linestyle="--")
    ax.set_xlabel("log10(maskopt_best_fid)")
    ax.set_ylabel("count")
    ax.set_title("Fidelity (smaller is better)")
    _save_fig(fig, outdir / "fig01_fidelity_hist.png", outdir / "fig01_fidelity_hist.pdf", dpi=dpi)

def fig_top_features_bar(agg_feat: pd.DataFrame, outdir: Path, pos: int, topk: int = 20, dpi: int = 300) -> None:
    dfp = agg_feat[agg_feat["ExplainPos"] == pos].head(topk).copy()
    if dfp.empty:
        return
    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    y = np.arange(len(dfp))
    ax.barh(y, dfp["mean_abs_impact"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(s, 40) for s in dfp["Name"].tolist()])
    ax.invert_yaxis()
    ax.set_xlabel("mean |score impact|")
    ax.set_title(f"Top features (ExplainPos={pos})")
    _save_fig(fig, outdir / f"fig02_top_features_pos{pos}.png", outdir / f"fig02_top_features_pos{pos}.pdf", dpi=dpi)

def fig_top_neighbors_bar(agg_nbr: pd.DataFrame, outdir: Path, pos: int, topk: int = 20, dpi: int = 300) -> None:
    dfp = agg_nbr[agg_nbr["ExplainPos"] == pos].head(topk).copy()
    if dfp.empty:
        return
    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    y = np.arange(len(dfp))
    ax.barh(y, dfp["mean_abs_impact"].to_numpy())
    ax.set_yticks(y)
    ax.set_yticklabels([_short_label(s, 40) for s in dfp["NeighborName"].tolist()])
    ax.invert_yaxis()
    ax.set_xlabel("mean |score impact|")
    ax.set_title(f"Top neighbor nodes (ExplainPos={pos})")
    _save_fig(fig, outdir / f"fig03_top_neighbors_pos{pos}.png", outdir / f"fig03_top_neighbors_pos{pos}.pdf", dpi=dpi)

def fig_feat_heatmap(agg_feat: pd.DataFrame, outdir: Path, positions: List[int], topk: int = 25, dpi: int = 300) -> None:
    # pick topk overall by sum_abs_impact
    overall = agg_feat.groupby("Name", as_index=False)["sum_abs_impact"].sum().sort_values("sum_abs_impact", ascending=False).head(topk)
    names = overall["Name"].tolist()
    mat = []
    for nm in names:
        row = []
        for pos in positions:
            sub = agg_feat[(agg_feat["ExplainPos"] == pos) & (agg_feat["Name"] == nm)]
            row.append(float(sub["mean_abs_impact"].iloc[0]) if len(sub) else 0.0)
        mat.append(row)
    mat = np.array(mat, dtype=float)

    fig = plt.figure(figsize=(7.2, max(3.8, 0.22 * len(names))))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels([_short_label(s, 40) for s in names])
    ax.set_xticks(np.arange(len(positions)))
    ax.set_xticklabels([str(p) for p in positions])
    ax.set_xlabel("ExplainPos")
    ax.set_title("Feature impact heatmap (mean |score impact|)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, outdir / "fig04_feature_heatmap.png", outdir / "fig04_feature_heatmap.pdf", dpi=dpi)

def fig_importance_vs_impact_scatter(df_feat_all: pd.DataFrame, outdir: Path, dpi: int = 300, max_points: int = 20000) -> None:
    df = df_feat_all.copy()
    # keep finite
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["Importance", "Abs_Impact"])
    if df.empty:
        return
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)

    x = df["Importance"].to_numpy(dtype=float)
    y = df["Abs_Impact"].to_numpy(dtype=float)
    pear, spear = _safe_corr(x, y)

    fig = plt.figure(figsize=(5.6, 4.2))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=6, alpha=0.35)
    ax.set_xlabel("gate value (Importance)")
    ax.set_ylabel("|score impact|")
    ax.set_title("Importance vs score impact")
    ax.text(0.02, 0.98, f"pearson={pear:.3f}\\nspearman={spear:.3f}", transform=ax.transAxes, va="top")
    _save_fig(fig, outdir / "fig05_scatter_importance_vs_impact.png", outdir / "fig05_scatter_importance_vs_impact.pdf", dpi=dpi)

def fig_feature_vs_neighbor_total(df_feat_all: pd.DataFrame, df_nbr_all: pd.DataFrame, outdir: Path, positions: List[int], dpi: int = 300) -> None:
    rows = []
    for pos in positions:
        fsum = float(df_feat_all[df_feat_all["ExplainPos"] == pos]["Abs_Impact"].sum())
        nsum = float(df_nbr_all[df_nbr_all["ExplainPos"] == pos]["Abs_Impact"].sum())
        rows.append((pos, fsum, nsum))
    df = pd.DataFrame(rows, columns=["ExplainPos", "FeatureTotalAbsImpact", "NeighborTotalAbsImpact"])

    fig = plt.figure(figsize=(6.2, 3.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(df))
    ax.bar(x - 0.15, df["FeatureTotalAbsImpact"].to_numpy(), width=0.3, label="features")
    ax.bar(x + 0.15, df["NeighborTotalAbsImpact"].to_numpy(), width=0.3, label="neighbors")
    ax.set_xticks(x)
    ax.set_xticklabels(df["ExplainPos"].astype(str).tolist())
    ax.set_xlabel("ExplainPos")
    ax.set_ylabel("sum |score impact|")
    ax.set_title("Total impact: features vs neighbors")
    ax.legend()
    _save_fig(fig, outdir / "fig06_total_feature_vs_neighbor.png", outdir / "fig06_total_feature_vs_neighbor.pdf", dpi=dpi)

def fig_static_dynamic_share(agg_feat: pd.DataFrame, outdir: Path, positions: List[int], dpi: int = 300) -> None:
    # heuristic: static = profile/category/type related
    def is_static(name: str) -> bool:
        name = str(name)
        if name in ("followers", "followees", "posts_history"):
            return True
        if name.startswith("cat_") or name.startswith("type_"):
            return True
        return False

    rows = []
    for pos in positions:
        dfp = agg_feat[agg_feat["ExplainPos"] == pos]
        static_sum = float(dfp[dfp["Name"].map(is_static)]["sum_abs_impact"].sum())
        dyn_sum = float(dfp[~dfp["Name"].map(is_static)]["sum_abs_impact"].sum())
        total = static_sum + dyn_sum + 1e-30
        rows.append((pos, static_sum / total, dyn_sum / total))
    df = pd.DataFrame(rows, columns=["ExplainPos", "StaticShare", "DynamicShare"])

    fig = plt.figure(figsize=(6.2, 3.8))
    ax = fig.add_subplot(111)
    x = np.arange(len(df))
    ax.bar(x - 0.15, df["StaticShare"].to_numpy(), width=0.3, label="static")
    ax.bar(x + 0.15, df["DynamicShare"].to_numpy(), width=0.3, label="dynamic")
    ax.set_xticks(x)
    ax.set_xticklabels(df["ExplainPos"].astype(str).tolist())
    ax.set_xlabel("ExplainPos")
    ax.set_ylabel("share of total |impact|")
    ax.set_title("Static vs dynamic feature share")
    ax.legend()
    _save_fig(fig, outdir / "fig07_static_dynamic_share.png", outdir / "fig07_static_dynamic_share.pdf", dpi=dpi)


# ------------------------
# LaTeX tables
# ------------------------
def latex_escape(s: str) -> str:
    s = str(s)
    # minimal escaping
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("_", "\\_")
    s = s.replace("%", "\\%")
    s = s.replace("&", "\\&")
    s = s.replace("#", "\\#")
    s = s.replace("{", "\\{").replace("}", "\\}")
    return s

def write_topk_table_tex(df: pd.DataFrame, cols: List[str], out_tex: Path, caption: str, label: str, topk: int = 20) -> None:
    d = df.head(topk).copy()
    # escape text columns
    for c in cols:
        if d[c].dtype == object:
            d[c] = d[c].map(latex_escape)
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{latex_escape(label)}}}")
    fmt = "l" + "r" * (len(cols) - 1)
    lines.append(f"\\begin{{tabular}}{{{fmt}}}")
    lines.append("\\hline")
    header = " & ".join([latex_escape(c) for c in cols]) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    for _, r in d.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                row.append(f"{v:.3e}")
            else:
                row.append(str(v))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    out_tex.write_text("\n".join(lines), encoding="utf-8")


# ------------------------
# Captions template
# ------------------------
def write_caption_template(out_md: Path, positions: List[int]) -> None:
    md = []
    md.append("# 図キャプション案（日本語）")
    md.append("")
    md.append("以下は、そのまま論文に貼れるようにしたキャプションの叩き台です。")
    md.append("")
    md.append("## Fig.01 Fidelity")
    md.append("- MaskOpt 最適化後の fidelity (maskopt_best_fid) の分布。閾値（例: 1e-6）より小さい実行が多いほど、マスク付き予測が元予測を良く保持している。")
    md.append("")
    md.append("## Fig.02 Top features (pos別)")
    md.append("- ExplainPos ごとに、平均 |score impact| が大きい特徴量を上位 K 個可視化。モデル予測に寄与する主要因（プロファイル系 / 投稿間隔 / 投稿量など）を示す。")
    md.append("")
    md.append("## Fig.03 Top neighbor nodes (pos別)")
    md.append("- ExplainPos ごとに、近傍ノード（エッジグループ=neighbor）を上位 K 個可視化。構造寄与の代表例（特定ハッシュタグ/メンション/ユーザとの関係など）を示す。")
    md.append("")
    md.append("## Fig.04 Feature heatmap")
    md.append(f"- 上位特徴量の寄与を月位置 {positions} に対してヒートマップ化。寄与の時間変化（最近効き始めた/継続的に重要）を視覚化する。")
    md.append("")
    md.append("## Fig.05 Scatter (importance vs impact)")
    md.append("- ゲート値（importance）と |score impact| の関係を散布図で示す。両者の相関が高いほど、importance が予測スコア変化を良く反映していると解釈できる（ただし非線形・飽和などで相関が低くなる場合もある）。")
    md.append("")
    md.append("## Fig.06 Total feature vs neighbor impact")
    md.append("- ExplainPos ごとに、特徴量側の総 |impact| と近傍ノード側の総 |impact| を比較。予測根拠が『特徴量主導』か『構造主導』かを議論する材料。")
    md.append("")
    md.append("## Fig.07 Static vs dynamic share")
    md.append("- 静的特徴（followers/followees/posts_history + cat_/type_）と動的特徴（投稿・コメント・間隔など）で、総寄与の比率を比較。『過去プロフィールが強い』などの考察に対応。")
    out_md.write_text("\n".join(md), encoding="utf-8")


# ------------------------
# Main
# ------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xai-dir", type=str, required=True, help="xai_export_out directory")
    p.add_argument("--outdir", type=str, default=None, help="output dir (default: <xai-dir>/paper)")
    p.add_argument("--dpi", type=int, default=300)

    # name recovery
    p.add_argument("--no-recover-names", action="store_true", help="skip name recovery even if placeholders exist")
    p.add_argument("--posts-file", type=str, default="dataset_A_active_all.csv")
    p.add_argument("--hashtags-file", type=str, default="hashtags_2017.csv")
    p.add_argument("--mentions-file", type=str, default="mentions_2017.csv")
    p.add_argument("--dec-year", type=int, default=2017)
    p.add_argument("--dec-month", type=int, default=12)

    # figures
    p.add_argument("--positions", type=str, default="0,1,2", help="comma-separated ExplainPos values")
    p.add_argument("--topk-features", type=int, default=20)
    p.add_argument("--topk-neighbors", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    xai_dir = Path(args.xai_dir)
    out_base = Path(args.outdir) if args.outdir else (xai_dir / "paper")
    figs_dir = out_base / "figs"
    tables_dir = out_base / "tables"
    _ensure_dir(out_base)
    _ensure_dir(figs_dir)
    _ensure_dir(tables_dir)

    positions = [int(x) for x in args.positions.split(",") if x.strip() != ""]

    df_sum, df_feat, df_nbr = load_xai_outputs(xai_dir)

    # name recovery if necessary
    need_recover = False
    if (not args.no_recover_names) and ("NeighborName" in df_nbr.columns) and df_nbr["NeighborName"].map(_is_placeholder_name).any():
        need_recover = True
    if (not args.no_recover_names) and ("TargetName" in df_sum.columns) and df_sum["TargetName"].map(_is_placeholder_name).any():
        need_recover = True

    idx_to_name = {}
    if need_recover:
        print("[recover] placeholders detected. rebuilding node_to_idx to recover real names...")
        node_to_idx = build_node_to_idx(
            posts_file=Path(args.posts_file),
            hashtags_file=Path(args.hashtags_file),
            mentions_file=Path(args.mentions_file),
            dec_year=args.dec_year,
            dec_month=args.dec_month,
        )
        # collect needed indices
        needed = set()
        for c in ["TargetGlobalIdx"]:
            if c in df_feat.columns:
                needed.update(pd.to_numeric(df_feat[c], errors="coerce").dropna().astype(int).tolist())
            if c in df_nbr.columns:
                needed.update(pd.to_numeric(df_nbr[c], errors="coerce").dropna().astype(int).tolist())
            if c in df_sum.columns:
                needed.update(pd.to_numeric(df_sum[c], errors="coerce").dropna().astype(int).tolist())
        if "NeighborGlobalIdx" in df_nbr.columns:
            needed.update(pd.to_numeric(df_nbr["NeighborGlobalIdx"], errors="coerce").dropna().astype(int).tolist())

        idx_to_name = invert_node_to_idx_for_needed(node_to_idx, needed)
        # save idmap
        idmap_path = out_base / "node_idmap_used.csv"
        pd.DataFrame({"GlobalIdx": list(idx_to_name.keys()), "Name": list(idx_to_name.values())}).sort_values("GlobalIdx").to_csv(idmap_path, index=False)
        print(f"[recover] wrote: {idmap_path}")

        df_feat, df_nbr, df_sum = apply_name_recovery(df_feat, df_nbr, df_sum, idx_to_name)
    else:
        print("[recover] skipped (no placeholders or --no-recover-names)")

    # aggregation
    agg_feat = aggregate_features_by_pos(df_feat)
    agg_nbr  = aggregate_neighbors_by_pos(df_nbr)

    agg_feat_path = out_base / "agg_features_by_pos_named.csv"
    agg_nbr_path  = out_base / "agg_neighbors_by_pos_named.csv"
    agg_feat.to_csv(agg_feat_path, index=False)
    agg_nbr.to_csv(agg_nbr_path, index=False)
    print(f"[write] {agg_feat_path}")
    print(f"[write] {agg_nbr_path}")

    # LaTeX tables (overall top)
    overall_feat = agg_feat.groupby("Name", as_index=False)["sum_abs_impact"].sum().sort_values("sum_abs_impact", ascending=False)
    overall_nbr  = agg_nbr.groupby("NeighborName", as_index=False)["sum_abs_impact"].sum().sort_values("sum_abs_impact", ascending=False)

    write_topk_table_tex(
        overall_feat.rename(columns={"sum_abs_impact": "sum_abs_impact"})[["Name", "sum_abs_impact"]],
        cols=["Name", "sum_abs_impact"],
        out_tex=tables_dir / "tab_top_features_overall.tex",
        caption="上位特徴量（全 ExplainPos 合算）",
        label="tab:top_features_overall",
        topk=args.topk_features,
    )
    write_topk_table_tex(
        overall_nbr.rename(columns={"sum_abs_impact": "sum_abs_impact"})[["NeighborName", "sum_abs_impact"]],
        cols=["NeighborName", "sum_abs_impact"],
        out_tex=tables_dir / "tab_top_neighbors_overall.tex",
        caption="上位近傍ノード（全 ExplainPos 合算）",
        label="tab:top_neighbors_overall",
        topk=args.topk_neighbors,
    )

    # figures
    fig_fidelity_hist(df_sum, figs_dir, fid_thr=1e-6, dpi=args.dpi)
    for pos in positions:
        fig_top_features_bar(agg_feat, figs_dir, pos=pos, topk=args.topk_features, dpi=args.dpi)
        fig_top_neighbors_bar(agg_nbr, figs_dir, pos=pos, topk=args.topk_neighbors, dpi=args.dpi)

    fig_feat_heatmap(agg_feat, figs_dir, positions=positions, topk=max(15, args.topk_features), dpi=args.dpi)
    fig_importance_vs_impact_scatter(df_feat, figs_dir, dpi=args.dpi)
    fig_feature_vs_neighbor_total(df_feat, df_nbr, figs_dir, positions=positions, dpi=args.dpi)
    fig_static_dynamic_share(agg_feat, figs_dir, positions=positions, dpi=args.dpi)

    # captions
    write_caption_template(out_base / "figure_captions_ja.md", positions=positions)

    # also write normalized summary
    df_sum.to_csv(out_base / "summary_named.csv", index=False)
    print(f"[write] {out_base / 'summary_named.csv'}")

    print("[DONE] paper artifacts generated under:", out_base)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
