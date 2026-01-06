# -*- coding: utf-8 -*-
"""CLI parsing.

IMPORTANT: this module must NOT import torch, torch_geometric, etc.
So it can be safely imported before setting CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations
import argparse
import os
from dataclasses import dataclass

@dataclass
class Args:
    # device / visibility
    device: int = 0
    visible: str | None = None

    # run mode
    mode: str | None = None          # "train" | "infer"
    xai_only: bool = False

    # checkpoint (infer)
    ckpt: str | None = None
    mlflow_run_id: str | None = None
    mlflow_ckpt_artifact: str = "model/model_state.pt"

    # mlflow
    tracking_uri: str | None = None
    experiment_name: str | None = None
    artifact_dir: str | None = None

def parse_args_pre_torch(argv=None) -> Args:
    ap = argparse.ArgumentParser()
    # device visibility (must be applied before torch import)
    ap.add_argument("--device", type=int, default=0,
                    help="PyTorch-visible GPU index. If CUDA_VISIBLE_DEVICES is set, this is remapped.")
    ap.add_argument("--visible", type=str, default=None,
                    help="Set CUDA_VISIBLE_DEVICES (e.g., '0', '1', '0,1'). Must be set before importing torch.")

    # run mode
    ap.add_argument("--mode", choices=["train", "infer"], default=None,
                    help="train: train+infer+XAI / infer: load checkpoint and run infer+XAI only")
    ap.add_argument("--xai_only", action="store_true",
                    help="Alias for infer mode.")

    # checkpoint
    ap.add_argument("--ckpt", type=str, default=os.environ.get("INFLUENCER_MODEL_CKPT"),
                    help="Path to checkpoint (.pt/.pth). Used in infer mode.")
    ap.add_argument("--mlflow_run_id", type=str, default=None,
                    help="If set, download checkpoint artifact from this MLflow run_id and use it.")
    ap.add_argument("--mlflow_ckpt_artifact", type=str, default="model/model_state.pt",
                    help="Artifact path under the run_id to download.")

    # mlflow
    ap.add_argument("--tracking_uri", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"),
                    help="MLflow tracking URI. If omitted, uses env MLFLOW_TRACKING_URI or default.")
    ap.add_argument("--experiment_name", type=str, default=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
                    help="MLflow experiment name (env MLFLOW_EXPERIMENT_NAME respected).")
    ap.add_argument("--artifact_dir", type=str, default=os.environ.get("MLFLOW_ARTIFACT_DIR"),
                    help="Local artifact root directory for file-based MLflow tracking.")

    ns, _ = ap.parse_known_args(argv)

    # Apply CUDA_VISIBLE_DEVICES early
    if ns.visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ns.visible

    return Args(
        device=int(ns.device),
        visible=ns.visible,
        mode=ns.mode,
        xai_only=bool(ns.xai_only),
        ckpt=ns.ckpt,
        mlflow_run_id=ns.mlflow_run_id,
        mlflow_ckpt_artifact=ns.mlflow_ckpt_artifact,
        tracking_uri=ns.tracking_uri,
        experiment_name=ns.experiment_name,
        artifact_dir=ns.artifact_dir,
    )
