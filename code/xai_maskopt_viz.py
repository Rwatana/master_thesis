#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize MaskOpt loss curves and mask timing per epoch.

Expected inputs are saved by maskopt_e2e_explain when save_history_dir is set.
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_optional_csv(path):
    if path is None or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _plot_loss_curves(df_loss, out_path, title_prefix="", logy=False):
    cols = [
        "loss_total",
        "loss_fid",
        "loss_contrast",
        "loss_budget",
        "loss_feat_size",
        "loss_feat_ent",
        "loss_edge_size",
        "loss_edge_ent",
    ]
    plt.figure(figsize=(10, 5))
    for c in cols:
        if c in df_loss.columns:
            plt.plot(df_loss["epoch"], df_loss[c], label=c, linewidth=1.4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if logy:
        plt.yscale("log")
    title = "MaskOpt Loss Curves"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_mask_heatmap(mask, labels, out_path, title, max_items=60):
    if mask is None or mask.size == 0:
        return
    epochs, items = mask.shape
    if items == 0 or epochs == 0:
        return

    masked_rate = mask.mean(axis=0)
    order = np.argsort(-masked_rate)
    keep = order[: min(int(max_items), items)]
    mask_sel = mask[:, keep]
    labels_sel = [labels[i] if labels is not None and i < len(labels) else f"idx_{i}" for i in keep]

    plt.figure(figsize=(10, 6))
    plt.imshow(mask_sel.T, aspect="auto", interpolation="nearest", cmap="Greys")
    plt.xlabel("Epoch")
    plt.ylabel("Feature/Edge")
    plt.title(title)
    if len(labels_sel) <= 30:
        plt.yticks(ticks=np.arange(len(labels_sel)), labels=labels_sel, fontsize=7)
    else:
        plt.yticks([])
    plt.colorbar(label="Masked (1) / Unmasked (0)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MaskOpt visualization utility")
    parser.add_argument("--input_dir", type=str, default="xai_viz")
    parser.add_argument("--prefix", type=str, required=True, help="Prefix used by maskopt_e2e_explain")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--logy", action="store_true", help="Use log scale for loss plot")
    parser.add_argument("--max_features", type=int, default=60)
    parser.add_argument("--max_edges", type=int, default=60)
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    loss_csv = os.path.join(input_dir, f"maskopt_loss_history_{args.prefix}.csv")
    mask_npz = os.path.join(input_dir, f"maskopt_mask_history_{args.prefix}.npz")
    feat_csv = os.path.join(input_dir, f"maskopt_feature_names_{args.prefix}.csv")
    edge_csv = os.path.join(input_dir, f"maskopt_edge_names_{args.prefix}.csv")

    if not os.path.exists(loss_csv):
        raise FileNotFoundError(f"loss history not found: {loss_csv}")
    if not os.path.exists(mask_npz):
        raise FileNotFoundError(f"mask history not found: {mask_npz}")

    df_loss = pd.read_csv(loss_csv)
    mask_data = np.load(mask_npz)
    feat_mask = mask_data.get("feat_mask")
    edge_mask = mask_data.get("edge_mask")

    feat_df = _load_optional_csv(feat_csv)
    edge_df = _load_optional_csv(edge_csv)
    feat_labels = feat_df["name"].tolist() if feat_df is not None and "name" in feat_df.columns else None
    edge_labels = edge_df["label"].tolist() if edge_df is not None and "label" in edge_df.columns else None

    loss_png = os.path.join(out_dir, f"maskopt_loss_curves_{args.prefix}.png")
    _plot_loss_curves(df_loss, loss_png, title_prefix=args.prefix, logy=args.logy)

    feat_png = os.path.join(out_dir, f"maskopt_feat_mask_{args.prefix}.png")
    _plot_mask_heatmap(feat_mask, feat_labels, feat_png, "Feature Mask Timeline", max_items=args.max_features)

    edge_png = os.path.join(out_dir, f"maskopt_edge_mask_{args.prefix}.png")
    _plot_mask_heatmap(edge_mask, edge_labels, edge_png, "Edge Mask Timeline", max_items=args.max_edges)

    print(f"Saved: {loss_png}")
    print(f"Saved: {feat_png}")
    print(f"Saved: {edge_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
