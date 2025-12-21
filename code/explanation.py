#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hard-coded XAI report (no log input, no server re-run).
- Uses a hand-picked "argument-friendly" subset:
  Run_000_20251216_1500 with pos_0/pos_1/pos_2
- Outputs:
  - console summary
  - CSV tables
  - PNG plots (histograms / bar charts / scatter)

Run:
  python hardcoded_xai_report.py
  python hardcoded_xai_report.py --outdir out_report --topn 15
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

# optional
try:
    import pandas as pd
except Exception:
    pd = None

import matplotlib.pyplot as plt
import japanize_matplotlib

@dataclass
class Item:
    kind: str          # "Feature" or "Edge"
    name: str
    importance: float
    score_impact: float
    direction: str


@dataclass
class Explain:
    pos: str
    target_node: int
    orig: float
    masked_pred: float
    items: List[Item]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_hardcoded_data() -> Dict:
    # Picked from your terminal output in this chat:
    # Run_000_20251216_1500, target_node=916490, pos_0/1/2
    run = {
        "run_name": "Run_000_20251216_1500",
        "metrics": {
            "MAE": 0.076062,
            "RMSE": 0.148764,
            "Pearson": 0.2244,
            "Spearman": 0.3350,
        },
        "explains": [
            Explain(
                pos="pos_0",
                target_node=916490,
                orig=0.04147957265377045,
                masked_pred=0.04147312417626381,
                items=[
                    # Features
                    Item("Feature", "post_interval_sec_max", 2.67030858e-02, -2.52525534e-01, "Negative (-)"),
                    Item("Feature", "post_interval_sec_mean", 1.09368684e-02,  1.16746873e-04, "Positive (+)"),
                    Item("Feature", "type_image_object",      1.08814761e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "comment_avg_neg_mean",   1.08596645e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "caption_length_max",     1.08502908e-02, -1.91479921e-06, "Negative (-)"),
                    Item("Feature", "rate_post_cat_4",        1.08374013e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "comment_avg_pos_max",    1.08017800e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "comment_avg_pos_median", 1.07594309e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "color_temp_proxy_median",1.07320026e-02,  2.51084566e-06, "Positive (+)"),
                    Item("Feature", "rate_post_cat_6",        1.07255857e-02,  0.00000000e+00, "Zero (0)"),
                    # Edges (named)
                    Item("Edge", "#化粧品",        9.85115860e-03, 2.39163637e-05, "Positive (+)"),
                    Item("Edge", "noahhinsdale",   9.83904582e-03, 2.35885382e-05, "Positive (+)"),
                    Item("Edge", "pier",           9.81948804e-03, 2.43932009e-05, "Positive (+)"),
                    Item("Edge", "candacecbure",   9.80682019e-03, 2.22064555e-05, "Positive (+)"),
                    Item("Edge", "soni_nicole",    9.77464113e-03, 2.39275396e-05, "Positive (+)"),
                    Item("Edge", "sanfrancisco",   9.75545589e-03, 2.23033130e-05, "Positive (+)"),
                    Item("Edge", "#ロケ地巡り",     9.74511448e-03, 2.37934291e-05, "Positive (+)"),
                    Item("Edge", "#mountfuji",     9.73501988e-03, 2.23405659e-05, "Positive (+)"),
                    Item("Edge", "#Okinawasong",   9.73488670e-03, 2.41138041e-05, "Positive (+)"),
                    Item("Edge", "#フラーハウス",   9.73309390e-03, 2.20797956e-05, "Positive (+)"),
                ],
            ),
            Explain(
                pos="pos_1",
                target_node=916490,
                orig=0.04147957265377045,
                masked_pred=0.0414731502532959,
                items=[
                    Item("Feature", "post_interval_sec_max", 2.68425643e-02, -2.52646685e-01, "Negative (-)"),
                    Item("Feature", "rate_post_cat_5",       1.10566225e-02, -1.11758709e-08, "Zero (0)"),
                    Item("Feature", "post_interval_sec_mean",1.09843938e-02,  1.16568059e-04, "Positive (+)"),
                    Item("Feature", "caption_sent_pos_median",1.09807989e-02, 0.00000000e+00, "Zero (0)"),
                    Item("Feature", "emoji_count_max",       1.09801050e-02,  4.84287739e-08, "Positive (+)"),
                    Item("Feature", "caption_length_min",    1.09404074e-02, -1.31130219e-06, "Negative (-)"),
                    Item("Feature", "post_interval_sec_median",1.09191341e-02,4.97810543e-05, "Positive (+)"),
                    Item("Feature", "type_image_object",     1.09089678e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "cat_Unknown",           1.09031713e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "comment_avg_neg_min",   1.08965188e-02,  0.00000000e+00, "Zero (0)"),
                    # Edges
                    Item("Edge", "#Beautifultree",           9.84533224e-03, 2.37599015e-05, "Positive (+)"),
                    Item("Edge", "lakeside",                 9.78355762e-03, 2.37040222e-05, "Positive (+)"),
                    Item("Edge", "#famoustown",              9.75902099e-03, 2.18935311e-05, "Positive (+)"),
                    Item("Edge", "#ファインディングネバーランド", 9.75300092e-03, 2.30632722e-05, "Positive (+)"),
                    Item("Edge", "acoustic guitar",          9.70930234e-03, 2.35065818e-05, "Positive (+)"),
                    Item("Edge", "lancomeofficial",          9.69079696e-03, 2.31601298e-05, "Positive (+)"),
                    Item("Edge", "#bluerose",                9.68185905e-03, 2.15917826e-05, "Positive (+)"),
                    Item("Edge", "juanpablodipace",          9.67721082e-03, 2.35661864e-05, "Positive (+)"),
                    Item("Edge", "#uke",                     9.67607740e-03, 2.24411488e-05, "Positive (+)"),
                    Item("Edge", "#movielocation",           9.66915675e-03, 3.93018126e-06, "Positive (+)"),
                ],
            ),
            Explain(
                pos="pos_2",
                target_node=916490,
                orig=0.04147957265377045,
                masked_pred=0.04147344082593918,
                items=[
                    Item("Feature", "post_interval_sec_max", 2.67377757e-02, -2.52122201e-01, "Negative (-)"),
                    Item("Feature", "post_interval_sec_mean",1.09735876e-02,  1.16650015e-04, "Positive (+)"),
                    Item("Feature", "caption_length_mean",   1.08302832e-02, -6.10947609e-07, "Negative (-)"),
                    Item("Feature", "cat_fashion",           1.08029982e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "caption_sent_neu_mean", 1.07221408e-02,  0.00000000e+00, "Zero (0)"),
                    Item("Feature", "comment_avg_neg_min",   1.07144248e-02,  1.86264515e-08, "Zero (0)"),
                    Item("Feature", "brightness_mean",       1.06893238e-02, -4.28408384e-07, "Negative (-)"),
                    Item("Feature", "comment_avg_compound_mean",1.06561054e-02,2.60770321e-08, "Zero (0)"),
                    Item("Feature", "caption_sent_compound_min",1.06461002e-02,0.00000000e+00, "Zero (0)"),
                    Item("Feature", "mention_count_min",      1.06101343e-02,0.00000000e+00, "Zero (0)"),
                    # Edges
                    Item("Edge", "#マシューモリソン",       9.90704540e-03, 2.28695571e-05, "Positive (+)"),
                    Item("Edge", "#manhattanbridge",          9.82618518e-03, 2.41659582e-05, "Positive (+)"),
                    Item("Edge", "#rossgeller",               9.79774538e-03, 2.37524509e-05, "Positive (+)"),
                    Item("Edge", "#エアーブラシ",             9.76422988e-03, 2.35177577e-05, "Positive (+)"),
                    Item("Edge", "#davidschwimmer",           9.73720755e-03, 2.41287053e-05, "Positive (+)"),
                    Item("Edge", "candacecbure",              9.72873345e-03, 2.37040222e-05, "Positive (+)"),
                    Item("Edge", "#bluesky",                  9.70115326e-03, 2.30930746e-05, "Positive (+)"),
                    Item("Edge", "#Japan",                    9.69911925e-03, 2.28621066e-05, "Positive (+)"),
                    Item("Edge", "#Yokohama",                 9.68752615e-03, 2.41287053e-05, "Positive (+)"),
                ],
            ),
        ],
    }
    return run


def to_table(run: Dict) -> Optional["pd.DataFrame"]:
    if pd is None:
        return None
    rows = []
    for ex in run["explains"]:
        for it in ex.items:
            rows.append({
                "run": run["run_name"],
                "pos": ex.pos,
                "target_node": ex.target_node,
                "orig": ex.orig,
                "masked_pred": ex.masked_pred,
                "delta": ex.masked_pred - ex.orig,
                "kind": it.kind,
                "name": it.name,
                "importance": it.importance,
                "score_impact": it.score_impact,
                "direction": it.direction,
            })
    return pd.DataFrame(rows)


def print_console_summary(run: Dict, df) -> None:
    print("=" * 100)
    print(f"[RUN] {run['run_name']}")
    m = run["metrics"]
    print(f"[METRICS] MAE={m['MAE']:.6g} | RMSE={m['RMSE']:.6g} | Pearson={m['Pearson']:.6g} | Spearman={m['Spearman']:.6g}")
    print("")

    for ex in run["explains"]:
        rel_err = abs(ex.masked_pred - ex.orig) / max(abs(ex.orig), 1e-12)
        feats = [it for it in ex.items if it.kind == "Feature"]
        edges = [it for it in ex.items if it.kind == "Edge"]
        feat_nz = sum(1 for it in feats if abs(it.score_impact) > 0.0)
        edge_nz = sum(1 for it in edges if abs(it.score_impact) > 0.0)

        print("-" * 100)
        print(f"[EXPLAIN] node={ex.target_node} {ex.pos}")
        print(f"  orig={ex.orig:.12g}  masked_pred={ex.masked_pred:.12g}  delta={(ex.masked_pred - ex.orig):.3g}  rel_err={rel_err:.3g}")
        print(f"  rows: n_feat={len(feats)} (nonzero impact={feat_nz}) | n_edge={len(edges)} (nonzero impact={edge_nz})")

        # show top-5 by importance for quick glance
        feats_sorted = sorted(feats, key=lambda x: x.importance, reverse=True)[:5]
        edges_sorted = sorted(edges, key=lambda x: x.importance, reverse=True)[:5]
        print("  top features:")
        for it in feats_sorted:
            print(f"    - {it.name:28s} imp={it.importance:.6g}  impact={it.score_impact:.6g}  {it.direction}")
        print("  top edges:")
        for it in edges_sorted:
            print(f"    - {it.name:28s} imp={it.importance:.6g}  impact={it.score_impact:.6g}  {it.direction}")

    if df is not None:
        print("\n[Table preview] (first 10 rows)")
        print(df.head(10).to_string(index=False))


def save_csv(run: Dict, df, outdir: str) -> None:
    ensure_dir(outdir)
    if df is None:
        return
    df.to_csv(os.path.join(outdir, "selected_rows.csv"), index=False)

    # aggregated tables
    feat = df[df["kind"] == "Feature"].copy()
    edge = df[df["kind"] == "Edge"].copy()

    if len(feat) > 0:
        feat_agg = (feat.groupby("name")
                        .agg(mean_importance=("importance", "mean"),
                             mean_abs_impact=("score_impact", lambda s: float((s.abs()).mean())),
                             nonzero_frac=("score_impact", lambda s: float((s.abs() > 0).mean())))
                        .sort_values("mean_importance", ascending=False)
                        .reset_index())
        feat_agg.to_csv(os.path.join(outdir, "feature_agg.csv"), index=False)

    if len(edge) > 0:
        edge_agg = (edge.groupby("name")
                        .agg(mean_importance=("importance", "mean"),
                             mean_abs_impact=("score_impact", lambda s: float((s.abs()).mean())),
                             nonzero_frac=("score_impact", lambda s: float((s.abs() > 0).mean())))
                        .sort_values("mean_importance", ascending=False)
                        .reset_index())
        edge_agg.to_csv(os.path.join(outdir, "edge_agg.csv"), index=False)


def plot_all(run: Dict, df, outdir: str, topn: int) -> None:
    ensure_dir(outdir)

    # If pandas unavailable, build minimal arrays
    if df is None:
        # minimal extraction
        importances = []
        impacts_nz = []
        imp_vs = []
        for ex in run["explains"]:
            for it in ex.items:
                importances.append(it.importance)
                if abs(it.score_impact) > 0:
                    impacts_nz.append(it.score_impact)
                    imp_vs.append((it.importance, it.score_impact))
    else:
        importances = df["importance"].tolist()
        impacts_nz = df.loc[df["score_impact"].abs() > 0, "score_impact"].tolist()
        imp_vs = list(zip(df.loc[df["score_impact"].abs() > 0, "importance"].tolist(),
                          df.loc[df["score_impact"].abs() > 0, "score_impact"].tolist()))

    # 1) Importance histogram
    plt.figure()
    plt.hist(importances, bins=25)
    plt.title("Importance distribution (hard-coded selected explains)")
    plt.xlabel("Importance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "importance_hist.png"), dpi=200)
    plt.close()

    # 2) Score_Impact histogram (non-zero only)
    if len(impacts_nz) > 0:
        plt.figure()
        plt.hist(impacts_nz, bins=25)
        plt.title("Score_Impact distribution (non-zero only)")
        plt.xlabel("Score_Impact")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "score_impact_hist_nonzero.png"), dpi=200)
        plt.close()

    # 3) Scatter: importance vs score impact
    if len(imp_vs) > 0:
        xs = [a for a, _ in imp_vs]
        ys = [b for _, b in imp_vs]
        plt.figure()
        plt.scatter(xs, ys, s=12)
        plt.title("Importance vs Score_Impact (non-zero only)")
        plt.xlabel("Importance")
        plt.ylabel("Score_Impact")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "importance_vs_scoreimpact_scatter.png"), dpi=200)
        plt.close()

    # 4) Delta per pos
    poses = [ex.pos for ex in run["explains"]]
    deltas = [ex.masked_pred - ex.orig for ex in run["explains"]]
    plt.figure()
    plt.bar(poses, deltas)
    plt.title("Prediction delta (masked_pred - orig) per pos")
    plt.xlabel("pos")
    plt.ylabel("delta")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "delta_per_pos_bar.png"), dpi=200)
    plt.close()

    # 5) Top features/edges barh by mean importance  ※x軸スケール共有
    if df is not None:
        feat = df[df["kind"] == "Feature"].copy()
        edge = df[df["kind"] == "Edge"].copy()

        # ---- 共有x軸上限を計算（Feature/Edgeの平均Importanceの最大値）----
        common_xmax = 0.0
        if len(feat) > 0:
            common_xmax = max(common_xmax, float(feat.groupby("name")["importance"].mean().max()))
        if len(edge) > 0:
            common_xmax = max(common_xmax, float(edge.groupby("name")["importance"].mean().max()))

        # ちょい余白（ゼロ割/ゼロ幅対策）
        if common_xmax <= 0:
            common_xmax = 1.0
        xlim_max = common_xmax * 1.05

        if len(feat) > 0:
            top_feat = (feat.groupby("name")["importance"].mean()
                            .sort_values(ascending=False)
                            .head(topn))
            plt.figure(figsize=(10, max(4, 0.35 * len(top_feat))))
            ax = top_feat.sort_values().plot(kind="barh")
            ax.set_xlim(0, xlim_max)  # ★共通スケール
            plt.title("Top Features by mean Importance (shared x-scale)")
            plt.xlabel("Mean Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "top_features_barh.png"), dpi=200)
            plt.close()

        if len(edge) > 0:
            top_edge = (edge.groupby("name")["importance"].mean()
                            .sort_values(ascending=False)
                            .head(topn))
            plt.figure(figsize=(10, max(4, 0.35 * len(top_edge))))
            ax = top_edge.sort_values().plot(kind="barh")
            ax.set_xlim(0, xlim_max)  # ★共通スケール
            plt.title("Top Edges by mean Importance (shared x-scale)")
            plt.xlabel("Mean Importance")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "top_edges_barh.png"), dpi=200)
            plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out_report_hardcoded", help="output directory")
    ap.add_argument("--topn", type=int, default=15, help="top-N for bar plots")
    args = ap.parse_args()

    run = build_hardcoded_data()
    df = to_table(run)

    print_console_summary(run, df)

    ensure_dir(args.outdir)
    if df is not None:
        save_csv(run, df, args.outdir)

    plot_dir = os.path.join(args.outdir, "plots")
    plot_all(run, df, plot_dir, topn=args.topn)

    print("\n[SAVED]")
    if df is not None:
        print(f"  - {os.path.join(args.outdir, 'selected_rows.csv')}")
        print(f"  - {os.path.join(args.outdir, 'feature_agg.csv')}")
        print(f"  - {os.path.join(args.outdir, 'edge_agg.csv')}")
    print(f"  - {plot_dir}/ (png files)")


if __name__ == "__main__":
    main()
