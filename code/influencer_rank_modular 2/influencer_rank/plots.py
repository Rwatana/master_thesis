# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

# seaborn is optional but used if available
try:
    import seaborn as sns
except Exception:
    sns = None

def plot_attention_weights(attention_matrix, run_name):
    """Plot attention weights and persist numeric alpha values."""
    attention_matrix = np.asarray(attention_matrix)
    if attention_matrix.ndim == 3 and attention_matrix.shape[-1] == 1:
        attention_matrix = attention_matrix[..., 0]

    mean_att = np.mean(attention_matrix, axis=0)
    time_steps = np.arange(len(mean_att))

    plt.figure(figsize=(10, 6))
    # keep deterministic colors without style libs; user can change later
    bars = plt.bar(time_steps, mean_att, edgecolor='black', alpha=0.7)

    plt.xlabel('Time Steps (Months)')
    plt.ylabel('Average Attention Weight')
    plt.title(f'Average Attention Weights across Time\nRun: {run_name}')

    labels = [f"T-{len(mean_att)-1-i}" for i in range(len(mean_att))]
    labels[-1] = "Current (T)"
    plt.xticks(time_steps, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    filename_bar = f"attention_weights_bar_{run_name}.png"
    plt.savefig(filename_bar, bbox_inches='tight')
    plt.close()

    filename_heat = None
    if sns is not None:
        plt.figure(figsize=(12, 8))
        subset_matrix = attention_matrix[:50, :]
        sns.heatmap(subset_matrix, cmap="Blues", annot=False, cbar_kws={'label': 'Attention Weight'})
        plt.xlabel('Time Steps (Oldest -> Newest)')
        plt.ylabel('Sample Users (Top 50)')
        plt.title('Attention Weights Heatmap (Individual)')
        plt.xticks(time_steps + 0.5, labels)

        filename_heat = f"attention_weights_heatmap_{run_name}.png"
        plt.savefig(filename_heat, bbox_inches='tight')
        plt.close()

    df_mean = pd.DataFrame({
        "pos": np.arange(len(mean_att), dtype=int),
        "label": labels,
        "alpha_mean": mean_att.astype(float),
    })
    filename_csv = f"attention_weights_mean_{run_name}.csv"
    df_mean.to_csv(filename_csv, index=False, float_format="%.8e")

    filename_raw = f"attention_weights_raw_{run_name}.npz"
    np.savez_compressed(filename_raw, attention=attention_matrix)

    return filename_bar, filename_heat, filename_csv, filename_raw

def generate_enhanced_scatter_plot(x_data, y_data, x_label, y_label, run_id, filename_suffix,
                                  color_data=None, color_label=None, title_suffix=""):
    plt.figure(figsize=(11, 9))
    if sns is not None:
        sns.set_style("whitegrid")

    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if color_data is not None:
        mask = mask & np.isfinite(color_data)

    x_masked = np.asarray(x_data)[mask]
    y_masked = np.asarray(y_data)[mask]

    if len(x_masked) == 0:
        plt.close()
        return None

    if color_data is not None:
        c_masked = np.asarray(color_data)[mask]
        scatter = plt.scatter(x_masked, y_masked, c=c_masked, alpha=0.6, s=30)
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_label)
    else:
        plt.scatter(x_masked, y_masked, alpha=0.5, s=30, label='Data Points')

    min_val = float(min(x_masked.min(), y_masked.min()))
    max_val = float(max(x_masked.max(), y_masked.max()))
    margin = (max_val - min_val) * 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin
    plt.plot([plot_min, plot_max], [plot_min, plot_max], '--', linewidth=2, label='Ideal (y=x)')
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    if len(x_masked) > 1:
        p_corr, _ = pearsonr(x_masked, y_masked)
        s_corr, _ = spearmanr(x_masked, y_masked)
        corr_text = f"\nPearson: {p_corr:.4f} | Spearman: {s_corr:.4f}"
    else:
        corr_text = "\n(Not enough data for correlation)"

    plt.title(f"{y_label} vs {x_label} {title_suffix}{corr_text}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if color_data is None:
        plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    filename = f"scatter_{filename_suffix}_{run_id}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename

def mlflow_log_pred_scatter(y_true, y_pred, tag="eval", step=None, artifact_path="plots",
                           fname="pred_vs_true_scatter.png", title=None):
    import numpy as np
    import matplotlib.pyplot as plt

    try:
        import mlflow
        mlflow_available = True
    except Exception:
        mlflow_available = False

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    assert y_true.shape == y_pred.shape, f"shape mismatch: {y_true.shape} vs {y_pred.shape}"

    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    mn = float(np.nanmin([y_true.min(), y_pred.min()]))
    mx = float(np.nanmax([y_true.max(), y_pred.max()]))
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title or f"Pred vs True ({tag})")
    plt.tight_layout()

    out_path = fname
    plt.savefig(out_path, dpi=200)
    plt.close()

    if mlflow_available and mlflow.active_run() is not None:
        ap = artifact_path
        if step is not None:
            ap = os.path.join(ap, f"step_{int(step)}")
        mlflow.log_artifact(out_path, artifact_path=ap)

    try:
        os.remove(out_path)
    except Exception:
        pass

def mlflow_log_maskopt_plots(df_feat, df_edge, meta=None, tag="pos_0", topk_feat=15, topk_edge=15,
                             artifact_path="xai", fname_prefix="maskopt"):
    import matplotlib.pyplot as plt
    try:
        import mlflow
        mlflow_available = True
    except Exception:
        mlflow_available = False

    if df_feat is not None and len(df_feat) > 0:
        d = df_feat.sort_values("Importance", ascending=False).head(int(topk_feat))
        plt.figure()
        plt.barh(list(reversed(d["Name"].tolist())), list(reversed(d["Importance"].astype(float).tolist())))
        plt.xlabel("Importance (gate)")
        plt.title(f"Top Features ({tag})")
        plt.tight_layout()

        fpath = f"{fname_prefix}_feat_{tag}.png"
        plt.savefig(fpath, dpi=200)
        plt.close()

        if mlflow_available and mlflow.active_run() is not None:
            mlflow.log_artifact(fpath, artifact_path=artifact_path)
        try:
            os.remove(fpath)
        except Exception:
            pass

    if df_edge is not None and len(df_edge) > 0:
        d = df_edge.sort_values("Importance", ascending=False).head(int(topk_edge))
        plt.figure()
        plt.barh(list(reversed(d["Name"].tolist())), list(reversed(d["Importance"].astype(float).tolist())))
        plt.xlabel("Importance (gate)")
        plt.title(f"Top Edges ({tag})")
        plt.tight_layout()

        epath = f"{fname_prefix}_edge_{tag}.png"
        plt.savefig(epath, dpi=200)
        plt.close()

        if mlflow_available and mlflow.active_run() is not None:
            mlflow.log_artifact(epath, artifact_path=artifact_path)
        try:
            os.remove(epath)
        except Exception:
            pass

    if meta is not None:
        import json
        mpath = f"{fname_prefix}_meta_{tag}.json"
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if mlflow_available and mlflow.active_run() is not None:
            mlflow.log_artifact(mpath, artifact_path=artifact_path)
        try:
            os.remove(mpath)
        except Exception:
            pass
