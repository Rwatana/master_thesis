# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import itertools
import random
import datetime
import numpy as np
import pandas as pd

import torch

from .config import DEFAULT_EXPERIMENT_NAME, DEFAULT_ARTIFACT_DIR
from .device import get_device
from .mlflow_utils import setup_mlflow_experiment
from .data import prepare_graph_data
from .experiment import run_experiment

def main(args) -> int:
    # seeds
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    device = get_device(int(args.device))
    print("[Device] Using:", device)

    experiment_name, experiment_id = setup_mlflow_experiment(
        experiment_base_name=args.experiment_name or DEFAULT_EXPERIMENT_NAME,
        tracking_uri=args.tracking_uri,
        local_artifact_dir=args.artifact_dir or DEFAULT_ARTIFACT_DIR,
    )

    # mode
    mode = args.mode
    if args.xai_only:
        mode = "infer"
    if mode is None:
        mode = "infer" if (args.ckpt is not None or args.mlflow_run_id is not None) else "train"

    target_date = pd.to_datetime("2017-12-31")
    prep = prepare_graph_data(
        end_date=target_date,
        num_months=12,
        metric_numerator="likes_and_comments",
        metric_denominator="followers",
        use_image_features=False,
    )
    if prep[0] is None:
        print("Data preparation failed.")
        return 1

    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prep
    feature_dim = int(monthly_graphs[0].x.shape[1])
    print(f"Final feature dimension: {feature_dim}")
    print(f"Follower feature index: {follower_feat_idx}")

    graphs_data = (
        monthly_graphs,
        influencer_indices,
        node_to_idx,
        feature_dim,
        follower_feat_idx,
        static_cols,
        dynamic_cols
    )

    base_params = {
        "name_prefix": "Run",
        "note": "Default",
        "LR": 0.003,
        "POINTWISE_LOSS_WEIGHT": 0.5,
        "DROPOUT_PROB": 0.2,
        "GCN_DIM": 128,
        "RNN_DIM": 128,
        "NUM_GCN_LAYERS": 2,
        "PROJECTION_DIM": 128,
        "EPOCHS": 10,
        "LIST_SIZE": 50,
        "BATCH_SIZE": 50 * 64,
        "USE_SAMPLER": True,
        "MODE": mode,
        "CKPT_PATH": args.ckpt,
        "CKPT_MLFLOW_RUN_ID": args.mlflow_run_id,
        "CKPT_MLFLOW_ARTIFACT": args.mlflow_ckpt_artifact,
        # XAI control (reproducibility)
        "explain_use_sensitivity": True,
        "xai_topk_pos": 3,
        "sensitivity_score_mode": "alpha_x_delta",
        "sensitivity_min_delta": 1e-4,
        "xai_attn_min_w": 0.0,
    }

    # Put sweep overrides here later if needed
    overrides_list = [{}]

    def _suffix(overrides):
        if not overrides:
            return "base"
        parts = []
        for k, v in overrides.items():
            parts.append(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}")
        return ",".join(parts)

    summary_rows = []
    for i, ov in enumerate(overrides_list):
        p = dict(base_params)
        p.update(ov)
        p["name_prefix"] = f"{base_params['name_prefix']}_{i:03d}"
        p["note"] = f"{base_params.get('note','')} | sweep={i+1}/{len(overrides_list)} | {_suffix(ov)}"

        run_id, metrics = run_experiment(p, graphs_data, device=device, experiment_id=experiment_id)
        row = {"run_index": i, "run_id": run_id, "note": p.get("note", "")}
        for k in [
            "LR","DROPOUT_PROB","POINTWISE_LOSS_WEIGHT",
            "GCN_DIM","RNN_DIM","NUM_GCN_LAYERS","PROJECTION_DIM",
            "EPOCHS","LIST_SIZE","BATCH_SIZE","USE_SAMPLER",
        ]:
            row[k] = p.get(k)
        if isinstance(metrics, dict):
            row.update(metrics)
        summary_rows.append(row)

    if len(summary_rows) > 1:
        import mlflow
        df_sum = pd.DataFrame(summary_rows)
        sum_csv = f"sweep_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_sum.to_csv(sum_csv, index=False)
        print(f"\n📌 Sweep summary saved: {sum_csv}")
        try:
            with mlflow.start_run(run_name=f"SweepSummary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_artifact(sum_csv)
        except Exception as e:
            print(f"⚠️ Failed to log sweep summary to MLflow: {e}")
        finally:
            try:
                os.remove(sum_csv)
            except Exception:
                pass

    print("\n🎉 Done. Run 'mlflow ui' to view results.")
    return 0
