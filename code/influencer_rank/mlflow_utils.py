# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from pathlib import Path

import torch

def _is_http_uri(uri: str) -> bool:
    return isinstance(uri, str) and (uri.startswith("http://") or uri.startswith("https://"))

def setup_mlflow_experiment(
    experiment_base_name: str = "InfluencerRankSweep",
    tracking_uri: str | None = None,
    local_artifact_dir: str = "mlruns_artifacts",
):
    """Local-first MLflow setup.

    If an experiment was created while using an MLflow server with `--serve-artifacts`,
    its artifact_location can become `mlflow-artifacts:/...`. When switching back to
    file-based tracking, artifact logging can fail. This helper creates a file-based
    experiment when needed.
    """
    import datetime
    import mlflow

    # Respect env unless explicitly provided
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    active_tracking_uri = mlflow.get_tracking_uri()
    is_remote_tracking = _is_http_uri(active_tracking_uri)

    base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_base_name)
    exp_name = base_name
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    artifact_dir = (Path.cwd() / local_artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def _get_exp(name: str):
        try:
            return mlflow.get_experiment_by_name(name)
        except Exception:
            return None

    exp = _get_exp(exp_name)

    # If experiment exists but uses mlflow-artifacts while we're local-file tracking, make a fresh one.
    if (not is_remote_tracking) and (exp is not None) and str(exp.artifact_location).startswith("mlflow-artifacts:"):
        exp_name = f"{base_name}_file_{ts}"
        exp = None

    # Create if missing
    if exp is None:
        try:
            if is_remote_tracking:
                exp_id = mlflow.create_experiment(exp_name)
            else:
                exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_dir.as_uri())
        except Exception:
            exp2 = _get_exp(exp_name)
            if exp2 is None:
                raise
            exp_id = exp2.experiment_id
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(exp_name)

    os.environ["MLFLOW_TRACKING_URI"] = mlflow.get_tracking_uri()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = exp_name

    print(f"[MLflow] tracking_uri={mlflow.get_tracking_uri()}")
    print(f"[MLflow] experiment={exp_name} (id={exp_id})")
    if not is_remote_tracking:
        print(f"[MLflow] artifact_root={artifact_dir.as_uri()}")

    return exp_name, exp_id

def save_model_checkpoint(model, params: dict, feature_dim: int, out_path: str):
    """Save a lightweight checkpoint + config json for later infer/XAI-only runs."""
    import json
    ckpt = {
        "state_dict": model.state_dict(),
        "params": {
            "GCN_DIM": int(params.get("GCN_DIM", 128)),
            "RNN_DIM": int(params.get("RNN_DIM", 128)),
            "NUM_GCN_LAYERS": int(params.get("NUM_GCN_LAYERS", 2)),
            "DROPOUT_PROB": float(params.get("DROPOUT_PROB", 0.2)),
            "PROJECTION_DIM": int(params.get("PROJECTION_DIM", 128)),
        },
        "feature_dim": int(feature_dim),
    }
    torch.save(ckpt, out_path)
    cfg_path = os.path.splitext(out_path)[0] + ".json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({"feature_dim": int(feature_dim), **ckpt["params"]}, f, ensure_ascii=False, indent=2)
    return out_path, cfg_path

def maybe_download_ckpt_from_mlflow(run_id: str, artifact_path: str, out_dir: str = "mlflow_ckpt_cache") -> str:
    """Download a checkpoint artifact from MLflow, trying multiple common paths."""
    from pathlib import Path
    import mlflow
    from mlflow.exceptions import MlflowException

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    def _try(ap: str):
        return mlflow.artifacts.download_artifacts(
            run_id=str(run_id),
            artifact_path=ap,
            dst_path=str(out_dir_p / str(run_id))
        )

    candidates = []
    if artifact_path:
        candidates.append(artifact_path)
        if "/" not in artifact_path:
            candidates.append("model/" + artifact_path)
        if artifact_path.endswith(".pt"):
            candidates.append(artifact_path[:-3] + ".pth")
            if "/" not in artifact_path:
                candidates.append("model/" + artifact_path[:-3] + ".pth")
        if artifact_path.endswith(".pth"):
            candidates.append(artifact_path[:-4] + ".pt")
            if "/" not in artifact_path:
                candidates.append("model/" + artifact_path[:-4] + ".pt")

    candidates += [
        "model/model_state.pt",
        "model/model_state.pth",
        "model_state.pt",
        "model_state.pth",
    ]

    seen, uniq = set(), []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c); seen.add(c)

    last_err = None
    for ap in uniq:
        try:
            return _try(ap)
        except Exception as e:
            last_err = e

    # Last resort: list artifacts and pick a .pt/.pth
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()

        def _walk(prefix=""):
            for info in client.list_artifacts(str(run_id), prefix):
                if info.is_dir:
                    yield from _walk(info.path)
                else:
                    yield info.path

        files = list(_walk(""))
        prefer = [f for f in files if f.startswith("model/") and (f.endswith(".pt") or f.endswith(".pth"))]
        others = [f for f in files if (f.endswith(".pt") or f.endswith(".pth"))]
        for ap in prefer + others:
            try:
                return _try(ap)
            except Exception as e:
                last_err = e
    except Exception:
        pass

    raise MlflowException(
        f"Failed to download checkpoint artifact for run_id={run_id}. "
        f"Tried: {uniq}. Last error: {last_err}"
    )

def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """Load model + feature_dim + hyperparams from checkpoint saved by save_model_checkpoint."""
    from .model import HardResidualInfluencerModel

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and "feature_dim" in ckpt:
        params = ckpt.get("params", {})
        feature_dim = int(ckpt.get("feature_dim"))
        model = HardResidualInfluencerModel(
            feature_dim=feature_dim,
            gcn_dim=int(params.get("GCN_DIM", 128)),
            rnn_dim=int(params.get("RNN_DIM", 128)),
            num_gcn_layers=int(params.get("NUM_GCN_LAYERS", 2)),
            dropout_prob=float(params.get("DROPOUT_PROB", 0.2)),
            projection_dim=int(params.get("PROJECTION_DIM", 128)),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        return model, feature_dim, params
    raise ValueError("Unsupported checkpoint format. Re-save with save_model_checkpoint().")
