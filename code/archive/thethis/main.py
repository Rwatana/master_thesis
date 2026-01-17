#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import textwrap
import pandas as pd
import matplotlib.pyplot as plt


def _pick_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _wrap(s: str, width: int = 30) -> str:
    s = "" if s is None else str(s)
    return "\n".join(textwrap.wrap(s, width=width)) if len(s) > width else s


def load_and_topk(csv_path: str, topk: int):
    df = pd.read_csv(csv_path)

    col_imp = _pick_col(df, ["Importance"])
    col_name = _pick_col(df, ["Name"])
    col_type = _pick_col(df, ["Type"])
    col_si = _pick_col(df, ["Score_Impact(masked)"])
    col_dir = _pick_col(df, ["Direction(masked)"])

    if col_imp is None:
        raise ValueError(f"{csv_path}: Importance column not found. cols={list(df.columns)}")

    df[col_imp] = pd.to_numeric(df[col_imp], errors="coerce")
    if col_si is not None:
        df[col_si] = pd.to_numeric(df[col_si], errors="coerce")

    # 0-1外が混ざってたら警告（正規化はしない）
    oor = df[df[col_imp].notna() & ((df[col_imp] < 0) | (df[col_imp] > 1))]
    if len(oor) > 0:
        print(f"[warn] {csv_path}: {len(oor)} rows out of [0,1] in Importance (no normalization).")

    # sort: Importance desc, tie-break by Score_Impact(masked) desc (if exists)
    sort_cols = [col_imp]
    asc = [False]
    if col_si is not None:
        sort_cols.append(col_si)
        asc.append(False)

    df = df.sort_values(by=sort_cols, ascending=asc).head(topk).copy()

    # labels
    def make_label(row):
        t = row[col_type] if col_type is not None else ""
        n = row[col_name] if col_name is not None else ""
        lab = f"{t}:{n}" if t != "" else f"{n}"
        return lab

    df["__label__"] = df.apply(make_label, axis=1)
    df["__label__"] = df["__label__"].map(lambda x: _wrap(x, width=34))

    return df, col_imp, col_si, col_dir


def plot_barh(df, col_imp, title: str, out_png: str, col_si=None, col_dir=None):
    # 上に大きいものが来るように（barhは下が0番になりがちなので逆転）
    df_plot = df.iloc[::-1].reset_index(drop=True)

    y = df_plot["__label__"]
    x = df_plot[col_imp]

    # サイズはラベル量に応じて伸ばす
    height = max(6.0, 0.35 * len(df_plot) + 1.5)
    width = 11.0
    plt.figure(figsize=(width, height))

    plt.barh(y, x)
    plt.xlim(0, 1)  # Importanceは0-1スケール前提
    plt.xlabel("Importance (0-1 scale, raw)")
    plt.title(title)

    # # 値の注釈（必要ならScore_ImpactやDirectionも）
    # for i, (imp_val) in enumerate(x):
    #     note = f"{imp_val:.4f}"
    #     if col_si is not None and col_si in df_plot.columns:
    #         si = df_plot.loc[i, col_si]
    #         if pd.notna(si):
    #             note += f" | SI={si:.4g}"
    #     if col_dir is not None and col_dir in df_plot.columns:
    #         dr = df_plot.loc[i, col_dir]
    #         if pd.notna(dr) and str(dr) != "":
    #             note += f" | {dr}"
    #     plt.text(min(1.0, imp_val + 0.01), i, note, va="center", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[write] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", type=int, required=True, help="node id in filename")
    ap.add_argument("--pos", type=int, default=2, help="explain_pos (default=2)")
    ap.add_argument("--topk", type=int, default=20, help="TopK (default=20)")
    ap.add_argument("--outdir", default=".", help="output directory for png")
    args = ap.parse_args()

    node = args.node
    pos = args.pos
    topk = args.topk
    outdir = args.outdir

    feat_csv = f"maskopt_feat_pos_{pos}_node_{node}_pos_{pos}.csv"
    edge_csv = f"maskopt_edge_pos_{pos}_node_{node}_pos_{pos}.csv"

    # FEATURE
    df_f, col_imp_f, col_si_f, col_dir_f = load_and_topk(feat_csv, topk)
    out_feat_png = os.path.join(outdir, f"top{topk}-feat-pos{pos}.png")
    plot_barh(
        df_f, col_imp_f,
        title=f"Top{topk} Feature Importance (pos={pos}, node={node})",
        out_png=out_feat_png,
        col_si=col_si_f, col_dir=col_dir_f
    )

    # EDGE
    df_e, col_imp_e, col_si_e, col_dir_e = load_and_topk(edge_csv, topk)
    out_edge_png = os.path.join(outdir, f"top{topk}-edge-pos{pos}.png")
    plot_barh(
        df_e, col_imp_e,
        title=f"Top{topk} Edge Importance (pos={pos}, node={node})",
        out_png=out_edge_png,
        col_si=col_si_e, col_dir=col_dir_e
    )


if __name__ == "__main__":
    main()
