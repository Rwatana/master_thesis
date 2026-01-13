#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InfluencerRank reproduction + training/inference + XAI (MaskOpt E2E) — FULL SCRIPT (paper-ready)

This file is based on your current "FULL SCRIPT (fixed)" and adds:
  - Edge grouping "neighbor" with real-name labels (username/hashtag/object) via idx_to_node mapping
  - MLflow artifacts for:
      * full gates (feat_gate / edge_gate) as NPZ
      * edge group map CSV (group_id -> neighbor_name/global_id)
  - Fix: use selected target node + selected explain_pos (no hard-coded node/pos)
  - Fix: remove duplicated / undefined variables in trailing logging calls
  - Log xai_positions + xai_target_node for strict reproducibility

Notes:
  - This script expects the same CSV / txt files as your current pipeline.
  - MLflow logs plots + explanation CSVs as artifacts.
"""

import os
import shutil
import sys
import argparse
import time
import datetime
import random
import gc
import math
import warnings
import itertools

import pandas as pd
import numpy as np
from tqdm import tqdm

# --------- CLI (must set CUDA_VISIBLE_DEVICES before torch import) ---------
def _parse_pre_torch_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", type=int, default=0,
                   help="PyTorch-visible GPU index. If CUDA_VISIBLE_DEVICES is set, this is remapped.")
    p.add_argument("--visible", type=str, default=None,
                   help="Set CUDA_VISIBLE_DEVICES (e.g., '0', '1', '0,1'). Must be set before importing torch.")
    args, _ = p.parse_known_args()
    if args.visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.visible
    return args

PRE_ARGS = _parse_pre_torch_args()
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, coalesce, k_hop_subgraph, degree

from torch.nn import LSTM, Linear, ReLU, Dropout
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.utils.data import Sampler

from sklearn.metrics import ndcg_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.pytorch
import copy

try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass


# --------- MLflow setup (robust for local file store vs. server) ---------
def _is_http_uri(uri: str) -> bool:
    return isinstance(uri, str) and (uri.startswith("http://") or uri.startswith("https://"))

def setup_mlflow_experiment(
    experiment_base_name: str = "InfluencerRankSweep",
    tracking_uri: str | None = None,
    local_artifact_dir: str = "mlruns_artifacts",
):
    """
    Local-first MLflow setup (A案: ローカル完結).

    Why:
      - If an experiment was created while using an MLflow *server* with `--serve-artifacts`,
        its artifact_location can become `mlflow-artifacts:/...`.
      - When later switching to file-based tracking URI, MLflow cannot resolve `mlflow-artifacts:`
        without an HTTP tracking server, and `mlflow.log_artifact` will crash.

    This helper:
      1) Sets tracking URI (env MLFLOW_TRACKING_URI is respected unless overridden)
      2) Ensures experiment uses *file:* artifact_location when tracking is file:*
         (creates a new experiment with a suffix if needed)
      3) Returns (experiment_name, experiment_id) and sets env vars.
    """
    import mlflow
    import datetime
    from pathlib import Path

    # Respect env unless explicitly provided
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    active_tracking_uri = mlflow.get_tracking_uri()
    is_remote_tracking = active_tracking_uri.startswith("http://") or active_tracking_uri.startswith("https://")

    base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_base_name)
    exp_name = base_name

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cwd = Path.cwd()
    artifact_dir = (cwd / local_artifact_dir).resolve()
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

    # print(f"[MLflow] tracking_uri={mlflow.get_tracking_uri()}")
    # print(f"[MLflow] experiment={exp_name} (id={exp_id})")
    if not is_remote_tracking:
        print(f"[MLflow] artifact_root={artifact_dir.as_uri()}")

    return exp_name, exp_id

def summarize_gate_tensor(gate: torch.Tensor, name: str, thr_list=(0.01, 0.05, 0.1, 0.9, 0.95, 0.99)):
    """
    gate: shape [F] or [E_group] (sigmoid後 0..1)
    Returns dict of numeric summaries (python floats/ints only).
    """
    if gate is None or (not torch.is_tensor(gate)) or gate.numel() == 0:
        return {
            "name": name,
            "numel": 0,
            "mean": None, "std": None, "min": None, "max": None,
            "q00": None, "q01": None, "q05": None, "q10": None, "q25": None,
            "q50": None, "q75": None, "q90": None, "q95": None, "q99": None, "q100": None,
            **{f"count_le_{t}": None for t in thr_list},
            **{f"count_ge_{t}": None for t in thr_list},
        }

    g = gate.detach().float().flatten()
    g_cpu = g.cpu().numpy()

    q = np.quantile(g_cpu, [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0])
    out = {
        "name": name,
        "numel": int(g.numel()),
        "mean": float(g.mean().item()),
        "std": float(g.std(unbiased=False).item()),
        "min": float(g.min().item()),
        "max": float(g.max().item()),
        "q00": float(q[0]),  "q01": float(q[1]),  "q05": float(q[2]),  "q10": float(q[3]),  "q25": float(q[4]),
        "q50": float(q[5]),  "q75": float(q[6]),  "q90": float(q[7]),  "q95": float(q[8]),  "q99": float(q[9]),  "q100": float(q[10]),
    }

    for t in thr_list:
        t = float(t)
        out[f"count_le_{t}"] = int((g <= t).sum().item())
        out[f"count_ge_{t}"] = int((g >= t).sum().item())
        out[f"ratio_le_{t}"] = float(out[f"count_le_{t}"] / max(1, out["numel"]))
        out[f"ratio_ge_{t}"] = float(out[f"count_ge_{t}"] / max(1, out["numel"]))

    return out


def save_model_checkpoint(model, params, feature_dim, out_path):
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
    """Download a checkpoint artifact from MLflow, trying multiple common paths.

    Returns local file path on success; raises MlflowException on failure.
    """
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

    # Build candidate artifact paths (ordered)
    candidates = []
    if artifact_path:
        candidates.append(artifact_path)
        # if user passes "model_state.pt", try "model/model_state.pt"
        if "/" not in artifact_path:
            candidates.append("model/" + artifact_path)
        # swap extensions
        if artifact_path.endswith(".pt"):
            candidates.append(artifact_path[:-3] + ".pth")
            if "/" not in artifact_path:
                candidates.append("model/" + artifact_path[:-3] + ".pth")
        if artifact_path.endswith(".pth"):
            candidates.append(artifact_path[:-4] + ".pt")
            if "/" not in artifact_path:
                candidates.append("model/" + artifact_path[:-4] + ".pt")

    # common defaults
    candidates += [
        "model/model_state.pt",
        "model/model_state.pth",
        "model_state.pt",
        "model_state.pth",
    ]

    # de-dup while keeping order
    seen = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            uniq.append(c)
            seen.add(c)

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
        # prefer model/ dir
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
        f"Tried: {uniq}. "
        f"Last error: {last_err}"
    )

def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    """Load model + feature_dim + hyperparams from checkpoint saved by save_model_checkpoint."""
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


def get_device(requested_idx: int):
    # CUDA
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        # print(f"[Device] torch sees {n} CUDA device(s). CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        #TODO Update
        # for i in range(n):
            # print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")

        if requested_idx < 0:
            return torch.device("cpu")
        if requested_idx >= n:
            # print(f"[Device] WARNING: requested cuda:{requested_idx} but only 0..{n-1} available. Fallback cuda:0")
            requested_idx = 0

        torch.cuda.set_device(requested_idx)
        return torch.device(f"cuda:{requested_idx}")

    # MPS (mac) fallback (best-effort)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # print("[Device] CUDA not available -> try MPS")
        try:
            _ = torch.tensor([1.0], device="mps")
            return torch.device("mps")
        except Exception as e:
            print(f"[Device] MPS not usable ({e}) -> CPU")
    return torch.device("cpu")


device = get_device(PRE_ARGS.device)
# print("[Device] Using:", device)

# --------- Files ---------
PREPROCESSED_FILE = 'dataset_A_active_all.csv'
IMAGE_DATA_FILE   = 'image_features_v2_full_fixed.csv'
HASHTAGS_FILE     = 'hashtags_2017.csv'
MENTIONS_FILE     = 'mentions_2017.csv'
INFLUENCERS_FILE  = 'influencers.txt'

# --------- Helper utils ---------
def _select_positions_by_attention(attn_w, num_positions, topk=3, min_w=0.0):
    if topk is None or topk <= 0 or topk >= num_positions:
        return list(range(num_positions))
    if attn_w is None:
        return list(range(min(topk, num_positions)))
    attn_w = attn_w.detach().float().flatten()
    T = min(num_positions, attn_w.numel())
    attn_w = attn_w[:T]
    if min_w > 0:
        keep = torch.nonzero(attn_w >= min_w, as_tuple=False).flatten().tolist()
        if len(keep) == 0:
            keep = torch.topk(attn_w, k=min(topk, T)).indices.tolist()
        w_keep = attn_w[keep]
        order = torch.argsort(w_keep, descending=True).tolist()
        return [keep[i] for i in order[:min(topk, len(keep))]]
    return torch.topk(attn_w, k=min(topk, T)).indices.tolist()

def _feature_baseline(x, node_mask=None, method="mean"):
    if node_mask is not None:
        xx = x[node_mask]
        if xx.numel() == 0:
            xx = x
    else:
        xx = x
    if method == "mean":
        return xx.mean(dim=0)
    if method == "median":
        return xx.median(dim=0).values
    raise ValueError(f"Unknown baseline method: {method}")

def _binary_entropy(p, eps=1e-12):
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

# ===================== Data Loading / Graph Building =====================
def load_influencer_profiles():
    """Read influencers.txt (tab-separated)."""
    # print(f"Loading influencer profiles from {INFLUENCERS_FILE}...")
    try:
        df_inf = pd.read_csv(INFLUENCERS_FILE, sep='\t', dtype=str)
        rename_map = {
            'Username': 'username',
            'Category': 'category',
            '#Followers': 'followers',
            '#Followees': 'followees',
            '#Posts': 'posts_history'
        }
        df_inf.rename(columns=rename_map, inplace=True)
        required_cols = ['username', 'category', 'followers', 'followees', 'posts_history']
        for c in required_cols:
            if c not in df_inf.columns:
                df_inf[c] = '0'
        df_inf = df_inf[required_cols].copy()
        df_inf['username'] = df_inf['username'].astype(str).str.strip()
        for c in ['followers', 'followees', 'posts_history']:
            df_inf[c] = pd.to_numeric(df_inf[c], errors='coerce').fillna(0)
        # print(f"Loaded {len(df_inf)} influencer profiles.")
        return df_inf
    except Exception as e:
        # print(f"Error loading influencers.txt: {e}")
        return pd.DataFrame(columns=['username', 'followers', 'followees', 'posts_history', 'category'])

def resolve_target_node_idx(target_node, influencer_name, node_to_idx, influencer_indices):
    # 1) explicit node id wins
    if target_node is not None:
        return int(target_node), None

    # 2) resolve by influencer_name
    if influencer_name is not None:
        name_raw = str(influencer_name).strip()
        name = name_raw[1:] if name_raw.startswith("@") else name_raw

        # exact match
        if name in node_to_idx:
            idx = int(node_to_idx[name])
            return idx, name

        # case-insensitive match
        lower_map = {}
        for k in node_to_idx.keys():
            if isinstance(k, str):
                lk = k.lower()
                if lk not in lower_map:
                    lower_map[lk] = k
        if name.lower() in lower_map:
            k = lower_map[name.lower()]
            idx = int(node_to_idx[k])
            return idx, k

        # fuzzy suggestions (only among influencer usernames)
        try:
            import difflib
            inf_set = set(int(i) for i in influencer_indices)
            influencer_names = [k for k, v in node_to_idx.items() if int(v) in inf_set]
            suggestions = difflib.get_close_matches(name, influencer_names, n=10, cutoff=0.6)
        except Exception:
            suggestions = []

        raise ValueError(
            f"--influencer_name '{name_raw}' not found in node_to_idx. "
            f"Example candidates: {suggestions}"
        )

    # 3) auto hub selection
    return None, None


def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts', use_image_features=False):
    """
    Build graph sequence for each month.
    Returns:
      monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols
    """
    # print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    # print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")

    # --- 1. Load Post Data ---
    try:
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False, dtype={'post_id': str})
        # print(f"Loaded {len(df_posts)} posts from {PREPROCESSED_FILE}")
        df_posts['username'] = df_posts['username'].astype(str).str.strip()

        target_month_start = pd.Timestamp('2017-12-01')
        target_month_end   = pd.Timestamp('2017-12-31 23:59:59')
        dec_posts = df_posts[(df_posts['datetime'] >= target_month_start) & (df_posts['datetime'] <= target_month_end)]
        valid_users_dec = set(dec_posts['username'].unique())
        # print(f"Users who posted in Dec 2017: {len(valid_users_dec):,}")
        if len(valid_users_dec) == 0:
            # print("Warning: No users found who posted in Dec 2017. Check date range.")
            return None, None, None, None, None, None

        original_count = len(df_posts)
        df_posts = df_posts[df_posts['username'].isin(valid_users_dec)].copy()
        # print(f"Filtered posts dataset: {original_count:,} -> {len(df_posts):,} rows")

        col_map = {
            'like_count': 'likes',
            'comment_count': 'comments',
            'hashtag_count': 'tag_count',
            'user_followers': 'followers_dynamic',
            'user_following': 'followees_dynamic',
            'user_media_count': 'posts_count_dynamic',
            'caption_len': 'caption_length',
            'sentiment_pos': 'caption_sent_pos',
            'sentiment_neg': 'caption_sent_neg',
            'sentiment_neu': 'caption_sent_neu',
            'sentiment_compound': 'caption_sent_compound',
            'comment_sentiment_pos': 'comment_avg_pos',
            'comment_sentiment_neg': 'comment_avg_neg',
            'comment_sentiment_neu': 'comment_avg_neu',
            'comment_sentiment_compound': 'comment_avg_compound'
        }
        df_posts.rename(columns=col_map, inplace=True)

        for col in ['comments', 'feedback_rate', 'likes']:
            if col not in df_posts.columns:
                df_posts[col] = 0
            else:
                df_posts[col] = df_posts[col].fillna(0)

    except FileNotFoundError:
        # print(f"Error: '{PREPROCESSED_FILE}' not found.")
        return None, None, None, None, None, None

    # --- 2. (Optional) Image Data ---
    # NOTE: You asked to run WITHOUT image_features CSV input.
    # If use_image_features=False (default), we skip reading IMAGE_DATA_FILE entirely
    # and simply set image-derived numeric features to 0, and omit image_object edges/nodes.
    df_objects_slim = pd.DataFrame(columns=["post_id", "username", "image_object"])
    if use_image_features:
        try:
            df_image_data = pd.read_csv(IMAGE_DATA_FILE, low_memory=False, dtype={'post_id': str})
            # object edges
            if {"post_id", "username", "detected_object"}.issubset(df_image_data.columns):
                df_objects_slim = df_image_data[["post_id", "username", "detected_object"]].copy()
                df_objects_slim.rename(columns={"detected_object": "image_object"}, inplace=True)
                df_objects_slim["username"] = df_objects_slim["username"].astype(str).str.strip()
                df_objects_slim = df_objects_slim[df_objects_slim["username"].isin(valid_users_dec)]
            else:
                df_objects_slim = pd.DataFrame(columns=["post_id", "username", "image_object"])

            # numeric image features
            if "color_temp" in df_image_data.columns and "color_temp_proxy" not in df_image_data.columns:
                df_image_data.rename(columns={"color_temp": "color_temp_proxy"}, inplace=True)

            image_feature_cols = ["post_id", "brightness", "colorfulness", "color_temp_proxy"]
            for col in image_feature_cols:
                if col not in df_image_data.columns:
                    df_image_data[col] = 0.0
            df_image_features = df_image_data[image_feature_cols].copy()

            # merge numeric features
            df_posts = pd.merge(df_posts, df_image_features, on="post_id", how="left")
            for col in ["brightness", "colorfulness", "color_temp_proxy"]:
                df_posts[col] = df_posts[col].fillna(0.0)

        except FileNotFoundError:
            # print(f"Warning: '{IMAGE_DATA_FILE}' not found. Continue without image features.")
            use_image_features = False
        except Exception as e:
            # print(f"Warning: Failed to load '{IMAGE_DATA_FILE}' ({e}). Continue without image features.")
            use_image_features = False

    if not use_image_features:
        # Ensure numeric image feature columns exist with zeros (so downstream aggregation works)
        for col in ["brightness", "colorfulness", "color_temp_proxy"]:
            if col not in df_posts.columns:
                df_posts[col] = 0.0

# --- 3. Merge Posts and Image Features ---
    # (merged above if use_image_features=True; otherwise filled with zeros)
    # --- 4. Prepare Graph Edges ---
    df_object_edges = pd.merge(df_objects_slim, df_posts[['post_id', 'datetime']], on='post_id')

    # Hashtags
    try:
        df_hashtags = pd.read_csv(HASHTAGS_FILE)
        df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
        df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce')
        df_hashtags['username'] = df_hashtags['username'].astype(str).str.strip()
        df_hashtags = df_hashtags[df_hashtags['username'].isin(valid_users_dec)]
    except Exception:
        df_hashtags = pd.DataFrame(columns=['username', 'hashtag', 'datetime'])

    # Mentions
    try:
        df_mentions = pd.read_csv(MENTIONS_FILE)
        df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)
        df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce')
        df_mentions['username'] = df_mentions['username'].astype(str).str.strip()
        df_mentions = df_mentions[df_mentions['username'].isin(valid_users_dec)]
    except Exception:
        df_mentions = pd.DataFrame(columns=['username', 'mention', 'datetime'])

    # --- 5. Prepare Influencer Profiles ---
    # print("Merging profile features from influencers.txt...")
    df_influencers_external = load_influencer_profiles()
    df_influencers_external = df_influencers_external[df_influencers_external['username'].isin(valid_users_dec)]
    # print(f"Filtered profiles from influencers.txt: {len(df_influencers_external):,} users (posted in Dec 2017)")

    df_active_base = pd.DataFrame({'username': list(valid_users_dec)})
    df_influencers = pd.merge(df_active_base, df_influencers_external, on='username', how='left')
    df_influencers['followers'] = df_influencers['followers'].fillna(0)
    df_influencers['followees'] = df_influencers['followees'].fillna(0)
    df_influencers['posts_history'] = df_influencers['posts_history'].fillna(0)
    df_influencers['category'] = df_influencers['category'].fillna('Unknown')

    current_date = time.strftime("%Y%m%d")
    output_user_file = f'active_influencers_v8_{current_date}.csv'
    # print(f"Saving {len(df_influencers)} active influencers to '{output_user_file}'...")
    df_influencers.to_csv(output_user_file, index=False)

    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time

    # --- 6. Prepare Nodes ---
    influencer_set = set(df_influencers['username'].astype(str))
    # sort the influencer set to ensure consistent ordering
    influencer_list = sorted(set(df_influencers["username"].astype(str)))



    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_image_objects = set(df_object_edges['image_object'].astype(str))

    # --- Delete NaN nodes ---
    all_hashtags = set(df_hashtags['hashtag'].dropna().astype(str))
    all_mentions = set(df_mentions['mention'].dropna().astype(str))
    all_image_objects = set(df_object_edges['image_object'].dropna().astype(str))

    # print(f"Node counts: Influencers={len(influencer_set)}, Hashtags={len(all_hashtags)}, Mentions={len(all_mentions)}, ImageObjects={len(all_image_objects)}")

    all_nodes = sorted(list(set(influencer_list) | all_hashtags | all_mentions | all_image_objects))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_list if inf in node_to_idx]

    # --- 7. Static Features ---
    node_df = pd.DataFrame({'username': all_nodes})
    profile_features = pd.merge(
        node_df,
        df_influencers[['username', 'followers', 'followees', 'posts_history', 'category']],
        on='username',
        how='left'
    )
    for col in ['followers', 'followees', 'posts_history']:
        profile_features[col] = pd.to_numeric(profile_features[col], errors='coerce').fillna(0)
    for col in ['followers', 'followees', 'posts_history']:
        profile_features[col] = np.log1p(profile_features[col])

    category_dummies = pd.get_dummies(profile_features['category'], prefix='cat', dummy_na=True)
    profile_features = pd.concat([profile_features, category_dummies], axis=1).drop(columns=['category'])

    node_df['type'] = 'unknown'
    node_df.loc[node_df['username'].isin(influencer_set), 'type'] = 'influencer'
    node_df.loc[node_df['username'].isin(all_hashtags), 'type'] = 'hashtag'
    node_df.loc[node_df['username'].isin(all_mentions), 'type'] = 'mention'
    node_df.loc[node_df['username'].isin(all_image_objects), 'type'] = 'image_object'

    node_type_dummies = pd.get_dummies(node_df['type'], prefix='type')
    static_features = pd.concat([profile_features, node_type_dummies], axis=1)
    static_feature_cols = list(static_features.drop('username', axis=1).columns)

    try:
        follower_feat_idx = static_feature_cols.index('followers')
        # print(f"DEBUG: 'followers' feature is at index {follower_feat_idx} in static features.")
    except ValueError:
        # print("Warning: 'followers' not found in static_feature_cols.")
        follower_feat_idx = 0

    # --- 8. Dynamic Features ---
    STATS_AGG = ['mean', 'median', 'min', 'max']
    required_cols = [
        'brightness', 'colorfulness', 'color_temp_proxy',
        'tag_count', 'mention_count', 'emoji_count', 'caption_length',
        'caption_sent_pos', 'caption_sent_neg', 'caption_sent_neu', 'caption_sent_compound',
        'feedback_rate', 'comment_avg_pos', 'comment_avg_neg', 'comment_avg_neu', 'comment_avg_compound'
    ]
    for col in required_cols:
        if col not in df_posts.columns:
            df_posts[col] = 0.0

    df_posts.sort_values(by=['username', 'datetime'], inplace=True)
    df_posts['post_interval_sec'] = df_posts.groupby('username')['datetime'].diff().dt.total_seconds().fillna(0)
    # make interval to log scale
    df_posts['post_interval_sec'] = np.log1p(df_posts['post_interval_sec'])

    if 'post_category' not in df_posts.columns:
        post_categories = [f'post_cat_{i}' for i in range(10)]
        df_posts['post_category'] = np.random.choice(post_categories, size=len(df_posts))
    if 'is_ad' not in df_posts.columns:
        df_posts['is_ad'] = 0

    agg_config = {
        'brightness': STATS_AGG,
        'colorfulness': STATS_AGG,
        'color_temp_proxy': STATS_AGG,
        'tag_count': STATS_AGG,
        'mention_count': STATS_AGG,
        'emoji_count': STATS_AGG,
        'caption_length': STATS_AGG,
        'caption_sent_pos': STATS_AGG,
        'caption_sent_neg': STATS_AGG,
        'caption_sent_neu': STATS_AGG,
        'caption_sent_compound': STATS_AGG,
        'post_interval_sec': STATS_AGG,
        'comment_avg_pos': STATS_AGG,
        'comment_avg_neg': STATS_AGG,
        'comment_avg_neu': STATS_AGG,
        'comment_avg_compound': STATS_AGG,
        'feedback_rate': 'mean',
        'is_ad': 'mean',
        'datetime': 'size'
    }

    dynamic_agg = df_posts.groupby(['username', 'month']).agg(agg_config)
    dynamic_agg.columns = ['_'.join(col).strip() for col in dynamic_agg.columns.values]
    dynamic_agg = dynamic_agg.reset_index()
    dynamic_agg.rename(columns={'datetime_size': 'monthly_post_count',
                                'feedback_rate_mean': 'feedback_rate',
                                'is_ad_mean': 'ad_rate'}, inplace=True)

    post_category_rate = df_posts.groupby(['username', 'month'])['post_category'].value_counts(normalize=True).unstack(fill_value=0)
    post_category_rate.columns = [f'rate_{col}' for col in post_category_rate.columns]
    post_category_rate = post_category_rate.reset_index()

    dynamic_features = pd.merge(dynamic_agg, post_category_rate, on=['username', 'month'], how='left')
    dynamic_feature_cols = list(dynamic_features.drop(['username', 'month'], axis=1).columns)

    # --- 9. Construct Graphs ---
    monthly_graphs = []
    start_date = end_date - pd.DateOffset(months=num_months-1)

    feature_columns = static_feature_cols + dynamic_feature_cols
    feature_dim = len(feature_columns)
    # print(f"Total feature dimension: {feature_dim}")

    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='M'), desc="Building monthly graphs"):
        snapshot_month = snapshot_date.to_period('M').start_time

        current_hashtags = df_hashtags[(df_hashtags['datetime'] >= snapshot_month) & (df_hashtags['datetime'] <= snapshot_date)]
        current_mentions = df_mentions[(df_mentions['datetime'] >= snapshot_month) & (df_mentions['datetime'] <= snapshot_date)]
        current_image_objects = df_object_edges[(df_object_edges['datetime'] >= snapshot_month) & (df_object_edges['datetime'] <= snapshot_date)]

        edges_io = [(node_to_idx[str(u)], node_to_idx[str(o)])
                    for u, o in zip(current_image_objects['username'], current_image_objects['image_object'])
                    if str(u) in node_to_idx and str(o) in node_to_idx]
        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)])
                    for u, h in zip(current_hashtags['username'], current_hashtags['hashtag'])
                    if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)])
                    for u, m in zip(current_mentions['username'], current_mentions['mention'])
                    if str(u) in node_to_idx and str(m) in node_to_idx]

        all_edges = list(set(edges_ht + edges_mt + edges_io))
        all_edges += [(idx, idx) for idx in influencer_indices]
        all_edges = list(set(all_edges))
        if not all_edges:
            all_edges = [(idx, idx) for idx in influencer_indices]

        num_nodes = len(all_nodes)
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

        current_dynamic = dynamic_features[dynamic_features['month'] == snapshot_month]
        snapshot_features = pd.merge(static_features, current_dynamic, on='username', how='left')
        snapshot_features = snapshot_features[feature_columns].fillna(0)
        x = torch.tensor(snapshot_features.astype(float).values, dtype=torch.float)

        # Target calculation per month
        monthly_posts_period = df_posts[df_posts['datetime'].dt.to_period('M') == snapshot_date.to_period('M')]
        monthly_agg = monthly_posts_period.groupby('username').agg(
            total_likes=('likes', 'sum'),
            total_comments=('comments', 'sum'),
            post_count=('datetime', 'size')
        ).reset_index()

        if metric_numerator == 'likes_and_comments':
            monthly_agg['numerator'] = monthly_agg['total_likes'] + monthly_agg['total_comments']
        else:
            monthly_agg['numerator'] = monthly_agg['total_likes']

        if metric_denominator == 'followers':
            numer_vals = pd.to_numeric(monthly_agg['numerator'], errors='coerce').fillna(0).values.astype(float)
            count_vals = pd.to_numeric(monthly_agg['post_count'], errors='coerce').fillna(0).values.astype(float)
            monthly_agg['avg_engagement_per_post'] = np.divide(
                numer_vals, count_vals,
                out=np.zeros_like(numer_vals),
                where=count_vals != 0
            )

            merged_data = pd.merge(monthly_agg, df_influencers[['username', 'followers']], on='username', how='left')
            numer = merged_data['avg_engagement_per_post'].values.astype(float)
            denom = pd.to_numeric(merged_data['followers'], errors='coerce').fillna(0).values.astype(float)
            merged_data['engagement'] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
        else:
            merged_data = monthly_agg
            numer = merged_data['numerator'].values.astype(float)
            denom = merged_data['post_count'].values.astype(float)
            merged_data['engagement'] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)

        engagement_data = pd.merge(
            pd.DataFrame({'username': all_nodes}),
            merged_data[['username', 'engagement']],
            on='username',
            how='left'
        ).fillna(0)

        y = torch.tensor(engagement_data['engagement'].values, dtype=torch.float).view(-1, 1)
        monthly_graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols

# ===================== Model =====================
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_channels, hidden_channels)] +
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

    def forward(self, x, edge_index, edge_weight=None):
        outs = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight).relu()
            outs.append(x)
        return torch.cat(outs, dim=1)

class AttentiveRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention_layer = Linear(hidden_dim, 1)

    def forward(self, sequence_of_embeddings):
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_scores = self.attention_layer(rnn_out).tanh()
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        return context_vector, attention_weights

class HardResidualInfluencerModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5, projection_dim=128):
        super().__init__()
        self.projection_layer = nn.Sequential(Linear(feature_dim, projection_dim), ReLU())
        self.gcn_encoder = GCNEncoder(projection_dim, gcn_dim, num_gcn_layers)
        combined_dim = (gcn_dim * num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(combined_dim, rnn_dim)
        self.predictor = nn.Sequential(
            Linear(rnn_dim, 64),
            ReLU(),
            Dropout(dropout_prob),
            Linear(64, 1)
        )

    def forward(self, gcn_embeddings, raw_features, baseline_scores=None):
        final_rep, attention_weights = self.attentive_rnn(gcn_embeddings)
        raw_output = self.predictor(final_rep).squeeze()
        predicted_scores = F.softplus(raw_output)
        return predicted_scores, attention_weights

# ===================== Loss =====================
class TailMixedListBatchSampler(Sampler):
    """
    1 list (=list_size) に tail を最低 n_hi 個入れる。
    lists_per_batch 個の list を連結して 1 batch (=batch_size) を返す。
    ※ batch 全体はシャッフルしない（view(-1, list_size) の境界を壊さないため）
    """
    def __init__(self, y, list_size, batch_size,
                 q_hi=0.90, q_lo=0.10, n_hi=1, n_lo=0,
                 seed=0, replacement=True):
        self.y = np.asarray(y, dtype=float)
        self.list_size = int(list_size)
        self.batch_size = int(batch_size)
        assert self.batch_size % self.list_size == 0, "batch_size must be multiple of list_size"
        self.lists_per_batch = self.batch_size // self.list_size

        self.n_hi = int(n_hi)
        self.n_lo = int(n_lo)
        assert self.n_hi + self.n_lo <= self.list_size

        thr_hi = np.quantile(self.y, q_hi)
        thr_lo = np.quantile(self.y, q_lo)

        self.hi_idx = np.where(self.y >= thr_hi)[0]
        self.lo_idx = np.where(self.y <= thr_lo)[0]
        self.mid_idx = np.where((self.y < thr_hi) & (self.y > thr_lo))[0]

        if len(self.hi_idx) == 0:
            raise RuntimeError("hi bucket empty. Adjust q_hi.")
        if self.n_lo > 0 and len(self.lo_idx) == 0:
            raise RuntimeError("lo bucket empty. Adjust q_lo or set n_lo=0.")

        self.rng = np.random.default_rng(seed)
        self.replacement = replacement

        # drop_last 相当：端数は切る
        self.num_batches = len(self.y) // self.batch_size

    def __len__(self):
        return self.num_batches

    def _sample(self, pool, k):
        if k <= 0:
            return []
        return self.rng.choice(pool, size=k, replace=self.replacement).tolist()

    def _make_one_list(self):
        one = []
        one += self._sample(self.hi_idx, self.n_hi)
        if self.n_lo > 0:
            one += self._sample(self.lo_idx, self.n_lo)

        rest = self.list_size - len(one)
        pool = self.mid_idx if len(self.mid_idx) > 0 else np.arange(len(self.y))
        one += self._sample(pool, rest)

        self.rng.shuffle(one)  # list 内はシャッフルOK
        return one

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            for _j in range(self.lists_per_batch):
                batch += self._make_one_list()
            yield batch

class ListMLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_scores, true_scores):
        batch_size, list_size = pred_scores.size()
        sorted_indices = torch.argsort(true_scores, dim=1, descending=True)
        batch_idx = torch.arange(batch_size).unsqueeze(1).to(pred_scores.device)
        sorted_preds = pred_scores[batch_idx, sorted_indices]
        max_val, _ = sorted_preds.max(dim=1, keepdim=True)
        sorted_preds_exp = torch.exp(sorted_preds - max_val)
        cum_sum = torch.flip(torch.cumsum(torch.flip(sorted_preds_exp, dims=[1]), dim=1), dims=[1])
        log_cum_sum = torch.log(cum_sum + 1e-10)
        log_likelihood = (sorted_preds - max_val) - log_cum_sum
        loss = -torch.sum(log_likelihood, dim=1)
        return loss.mean()

# ===================== Plot helpers =====================
def plot_attention_weights(attention_matrix, run_name):
    """Plot attention weights and also persist numeric alpha values.

    Returns:
        (bar_png, heat_png, mean_csv, raw_npz)
    """
    attention_matrix = np.asarray(attention_matrix)
    if attention_matrix.ndim == 3 and attention_matrix.shape[-1] == 1:
        attention_matrix = attention_matrix[..., 0]

    mean_att = np.mean(attention_matrix, axis=0)
    time_steps = np.arange(len(mean_att))

    plt.figure(figsize=(10, 6))
    colors = ['skyblue'] * (len(mean_att)-1) + ['salmon']
    bars = plt.bar(time_steps, mean_att, color=colors, edgecolor='black', alpha=0.7)

    plt.xlabel('Time Steps (Months)', fontsize=12)
    plt.ylabel('Average Attention Weight', fontsize=12)
    plt.title(f'Average Attention Weights across Time\nRun: {run_name}', fontsize=14)

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

    plt.figure(figsize=(12, 8))
    subset_matrix = attention_matrix[:50, :]
    sns.heatmap(subset_matrix, cmap="Blues", annot=False, cbar_kws={'label': 'Attention Weight'})
    plt.xlabel('Time Steps (Oldest -> Newest)', fontsize=12)
    plt.ylabel('Sample Users (Top 50)', fontsize=12)
    plt.title('Attention Weights Heatmap (Individual)', fontsize=14)
    plt.xticks(time_steps + 0.5, labels)

    filename_heat = f"attention_weights_heatmap_{run_name}.png"
    plt.savefig(filename_heat, bbox_inches='tight')
    plt.close()

    labels = [f"T-{len(mean_att)-1-i}" for i in range(len(mean_att))]
    labels[-1] = "Current (T)"
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
    sns.set_style("whitegrid")

    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if color_data is not None:
        mask = mask & np.isfinite(color_data)

    x_masked = x_data[mask]
    y_masked = y_data[mask]

    if len(x_masked) == 0:
        plt.close()
        return None

    if color_data is not None:
        c_masked = color_data[mask]
        scatter = plt.scatter(x_masked, y_masked, c=c_masked, cmap='viridis', alpha=0.6, s=30)
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_label, fontsize=12)
    else:
        plt.scatter(x_masked, y_masked, alpha=0.5, color='blue', s=30, label='Data Points')

    min_val = min(x_masked.min(), y_masked.min())
    max_val = max(x_masked.max(), y_masked.max())
    margin = (max_val - min_val) * 0.05
    plot_min = min_val - margin
    plot_max = max_val + margin
    plt.plot([plot_min, plot_max], [plot_min, plot_max], 'r--', linewidth=2, label='Ideal (y=x)')
    plt.xlim(plot_min, plot_max)
    plt.ylim(plot_min, plot_max)

    if len(x_masked) > 1:
        p_corr, _ = pearsonr(x_masked, y_masked)
        s_corr, _ = spearmanr(x_masked, y_masked)
        corr_text = f"\nPearson: {p_corr:.4f} | Spearman: {s_corr:.4f}"
    else:
        corr_text = "\n(Not enough data for correlation)"

    plt.title(f"{y_label} vs {x_label} {title_suffix}{corr_text}", fontsize=14)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    if color_data is None:
        plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    filename = f"scatter_{filename_suffix}_{run_id}.png"
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    return filename

# ===================== Dataset helper =====================
def get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-1):
    num_graphs = len(monthly_graphs)
    positive_target_idx = (target_idx + num_graphs) % num_graphs
    if positive_target_idx == 0:
        raise ValueError("Cannot use target_idx that points to the first month (no history).")

    target_graph = monthly_graphs[positive_target_idx]
    baseline_graph = monthly_graphs[positive_target_idx - 1]

    target_y = target_graph.y[influencer_indices].squeeze()
    baseline_y = baseline_graph.y[influencer_indices].squeeze()

    return TensorDataset(
        torch.tensor(influencer_indices, dtype=torch.long),
        target_y,
        baseline_y
    )

# ===================== MaskOpt E2E Explainer =====================
def _gcn_forward_concat(gcn_encoder, x, edge_index, edge_weight=None):
    layer_outputs = []
    h = x
    for conv in gcn_encoder.convs:
        try:
            if edge_weight is None or edge_index.numel() == 0:
                h = conv(h, edge_index)
            else:
                h = conv(h, edge_index, edge_weight=edge_weight)
        except TypeError:
            h = conv(h, edge_index)
        h = F.relu(h)
        layer_outputs.append(h)
    return torch.cat(layer_outputs, dim=1)

class E2EMaskOptWrapper(nn.Module):
    def __init__(
        self,
        model,
        input_graphs,
        target_node_idx,
        explain_pos,
        device,
        use_subgraph=True,
        num_hops=2,
        undirected=True,
        feat_mask_scope="target",
        edge_mask_scope="incident",
        # NEW:
        edge_grouping="none",     # "none" | "neighbor"
        idx_to_node=None,         # {global_idx: "username/hashtag/object"}
    ):
        super().__init__()
        self.model = model
        self.input_graphs = input_graphs
        self.T = len(input_graphs)
        self.target_global = int(target_node_idx)
        self.explain_pos = int(explain_pos)
        self.device = device

        self.use_subgraph = use_subgraph
        self.num_hops = int(num_hops)
        self.undirected = bool(undirected)
        self.feat_mask_scope = feat_mask_scope
        self.edge_mask_scope = edge_mask_scope

        self.edge_grouping = edge_grouping
        self.idx_to_node = idx_to_node or {}

        self.edge_group_names = None
        self.edge_group_members = None   # list[list[int]] (edge positions in ei_exp)
        self.edge_group_meta = None

        self.cached_proj = [None] * self.T
        self.cached_gcn  = [None] * self.T
        self._prepare_cache()
        self._prepare_explain_graph()

    def _prepare_cache(self):
        self.model.eval()
        with torch.no_grad():
            for t, g in enumerate(self.input_graphs):
                if t == self.explain_pos:
                    continue
                g = g.to(self.device)
                p = self.model.projection_layer(g.x)
                out = _gcn_forward_concat(self.model.gcn_encoder, p, g.edge_index, edge_weight=None)
                self.cached_proj[t] = p[self.target_global].detach()
                self.cached_gcn[t]  = out[self.target_global].detach()

    def _prepare_explain_graph(self):
        g = self.input_graphs[self.explain_pos].to(self.device)
        x_full = g.x
        ei_full = g.edge_index

        # --- choose explain graph (full or k-hop) ---
        if (not self.use_subgraph) or (ei_full.numel() == 0):
            self.x_exp = x_full
            self.ei_exp = ei_full
            self.target_local = self.target_global
            self.local2global = None
        else:
            subset, ei_sub, mapping, _ = k_hop_subgraph(
                self.target_global,
                self.num_hops,
                ei_full,
                relabel_nodes=True,
                num_nodes=x_full.size(0)
            )
            x_sub = x_full[subset]
            if self.undirected and ei_sub.numel() > 0:
                ei_sub = to_undirected(ei_sub, num_nodes=x_sub.size(0))
                ei_sub = coalesce(ei_sub, num_nodes=x_sub.size(0))
            self.x_exp = x_sub
            self.ei_exp = ei_sub
            self.target_local = int(mapping.item())
            self.local2global = subset

        # --- ALWAYS build incident_edge_idx when edge_mask_scope="incident" ---
        if self.edge_mask_scope == "incident":
            if self.ei_exp.numel() == 0:
                self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)
            else:
                src, dst = self.ei_exp
                incident = (src == self.target_local) | (dst == self.target_local)

                # ✅ ターゲット self-loop を確実に除外
                is_target_self_loop = (src == self.target_local) & (dst == self.target_local)
                incident_wo_self = incident & (~is_target_self_loop)

                self.incident_edge_idx = torch.where(incident_wo_self)[0]
        else:
            # not used in non-incident scope, but keep for safety
            self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)

        # --- neighbor grouping ---
        if (
            self.edge_mask_scope == "incident"
            and self.edge_grouping == "neighbor"
            and self.incident_edge_idx.numel() > 0
        ):
            src, dst = self.ei_exp
            groups = {}  # neighbor_global -> list[edge_pos]

            for epos in self.incident_edge_idx.detach().cpu().tolist():
                s = int(src[epos].item())
                d = int(dst[epos].item())
                nbr_local = d if s == self.target_local else s

                # 念のため self を除外
                if nbr_local == self.target_local:
                    continue

                if self.local2global is not None:
                    nbr_global = int(self.local2global[nbr_local].item())
                else:
                    nbr_global = int(nbr_local)

                groups.setdefault(nbr_global, []).append(epos)

            keys = sorted(groups.keys())
            self.edge_group_members = [groups[k] for k in keys]
            self.edge_group_names = [str(self.idx_to_node.get(k, f"node_{k}")) for k in keys]
            self.edge_group_meta = []
            for k in keys:
                self.edge_group_meta.append({
                    "neighbor_global": int(k),
                    "neighbor_name": str(self.idx_to_node.get(k, f"node_{k}")),
                    "num_edges_in_group": int(len(groups[k])),
                })
            self.num_edge_params = int(len(self.edge_group_members))
        else:
            if self.edge_mask_scope == "incident":
                self.num_edge_params = int(self.incident_edge_idx.numel())
            else:
                self.num_edge_params = int(self.ei_exp.size(1))

        self.feature_dim = int(self.x_exp.size(1))


    def num_mask_params(self):
        return self.feature_dim, self.num_edge_params

    def _apply_feature_gate(self, x, feat_gate):
        if self.feat_mask_scope in ("all", "subgraph"):
            return x * feat_gate.view(1, -1)

        n = x.size(0)
        sel = F.one_hot(
            torch.tensor(self.target_local, device=x.device),
            num_classes=n
        ).to(x.dtype).unsqueeze(1)  # [N,1]

        return x + sel * x * (feat_gate.view(1, -1) - 1.0)

    def _make_edge_weight(self, edge_gate):
        E = int(self.ei_exp.size(1))
        w = torch.ones(E, device=self.device)

        if E == 0 or edge_gate is None or edge_gate.numel() == 0:
            return w

        # NEW: grouped by neighbor
        if (self.edge_mask_scope == "incident") and (self.edge_grouping == "neighbor") and (self.edge_group_members is not None):
            w = w.clone()
            for g, epos_list in enumerate(self.edge_group_members):
                if not epos_list:
                    continue
                idx = torch.tensor(epos_list, device=self.device, dtype=torch.long)
                w[idx] = edge_gate[g]
            return w

        # original incident behavior
        if self.edge_mask_scope == "incident":
            w = w.clone()
            if self.incident_edge_idx.numel() > 0:
                w[self.incident_edge_idx] = edge_gate
            return w

        # "subgraph" scope: edge_gate is already [E]
        return edge_gate

    def predict_with_gates(self, feat_gate, edge_gate, edge_weight_override=None, x_override=None):
        """
        Returns scalar prediction for the target node.

        edge_weight_override: Tensor[E_sub] (1=keep, 0=drop)
        x_override: Tensor[n_sub, F] for explain_pos only (used in feature ablation)
        """
        seq_gcn = []
        seq_raw = []

        for t in range(self.T):
            if t != self.explain_pos:
                seq_gcn.append(self.cached_gcn[t])
                seq_raw.append(self.cached_proj[t])
                continue

            x = x_override if x_override is not None else self.x_exp
            ei = self.ei_exp

            x_masked = self._apply_feature_gate(x, feat_gate)
            ew = self._make_edge_weight(edge_gate)
            if edge_weight_override is not None:
                ew = ew * edge_weight_override

            p = self.model.projection_layer(x_masked)
            out = _gcn_forward_concat(self.model.gcn_encoder, p, ei, edge_weight=ew)
            seq_gcn.append(out[self.target_local])
            seq_raw.append(p[self.target_local])

        seq_gcn = torch.stack(seq_gcn, dim=0).unsqueeze(0)
        seq_raw = torch.stack(seq_raw, dim=0).unsqueeze(0)

        pred, _ = self.model(seq_gcn, seq_raw, baseline_scores=None)
        return pred.view(())

    @torch.no_grad()
    def original_pred(self):
        feat_gate = torch.ones(self.feature_dim, device=self.device)
        edge_gate = torch.ones(int(self.num_edge_params), device=self.device) if self.num_edge_params > 0 else None
        if edge_gate is not None and edge_gate.numel() == 0:
            edge_gate = None
        return float(self.predict_with_gates(feat_gate, edge_gate).item())

class _DisableCudnn:
    def __enter__(self):
        self.prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
    def __exit__(self, exc_type, exc, tb):
        torch.backends.cudnn.enabled = self.prev
        return False




def maskopt_e2e_explain(
    model,
    input_graphs,
    target_node_idx,
    explain_pos,
    feature_names,
    node_to_idx=None,
    device=None,
    use_subgraph=True,
    num_hops=2,
    undirected=True,
    feat_mask_scope="target",
    edge_mask_scope="incident",
    epochs=300,
    lr=0.05,
    coeffs=None,
    print_every=50,
    topk_feat=20,
    topk_edge=30,
    min_show=1e-6,
    disable_cudnn_rnn=True,
    mlflow_log=False,
    fid_weight=100.0,
    use_contrastive=False,
    contrastive_margin=0.002,
    contrastive_weight=1.0,
    tag="pos_0",
    edge_grouping="neighbor",              # "none" | "neighbor"
    impact_reference="masked",         # "masked" | "unmasked" | "both"
    ablation_mode="gate_zero",
    budget_feat=None,
    budget_edge=None,
    budget_weight=0.0,
    eps_abs_feat=1e-9,
    eps_rel_feat=1e-6,
    eps_abs_edge=1e-9,
    eps_rel_edge=1e-6,
    log_diagnostics=True,
    mask_zero_thr=1e-24,
    # ---- NEW diagnostics controls ----
    diag_thr_list=None,              # e.g., [0.01, 0.05, 0.1]
    diag_quantiles=(0.1, 0.5, 0.9),   # p10/p50/p90
    diag_log_legacy_thr=True,         # keep old feat_zero_count etc (mask_zero_thr)
    # --- NEW: per-epoch topK logging (importance = gate value) ---
    log_topk_each_epoch=True,
    topk_each_epoch_feat=10,
    topk_each_epoch_edge=10,
    topk_each_epoch_every=1,      # 1 = every epoch, 10 = every 10 epochs
    topk_each_epoch_min_gate=0.0, # filter very small gate
    topk_each_epoch_print=5,      # # print top-N to stdout each time (<= topk_each_epoch_*)

):
    assert len(input_graphs) >= 2, "input_graphs length must be >= 2"

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = len(input_graphs)
    if explain_pos < 0:
        explain_pos = (explain_pos + T) % T

    if coeffs is None:
        coeffs = {
            "edge_size": 0.05,
            "edge_ent": 0.10,
            "node_feat_size": 0.02,
            "node_feat_ent": 0.10
        }

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # invert mapping for real names
    idx_to_node = None
    if node_to_idx is not None and isinstance(node_to_idx, dict):
        idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

    # MLflow logger handle (safe)
    _ml = None
    try:
        if bool(mlflow_log) and (mlflow.active_run() is not None):
            _ml = mlflow
    except Exception:
        _ml = None


    wrapper = E2EMaskOptWrapper(
    model=model,
    input_graphs=input_graphs,
    target_node_idx=target_node_idx,
    explain_pos=explain_pos,
    device=device,
    use_subgraph=use_subgraph,
    num_hops=num_hops,
    undirected=undirected,
    feat_mask_scope=feat_mask_scope,
    edge_mask_scope=edge_mask_scope,
    edge_grouping=edge_grouping,
    idx_to_node=idx_to_node,
    )


    Fdim, Edim = wrapper.num_mask_params()

    # --- NEW: names for per-epoch topK (Edim is known now) ---
    feat_name_list = list(feature_names) if feature_names is not None else [f"feat_{i}" for i in range(int(Fdim))]
    if getattr(wrapper, "edge_group_names", None) is not None and len(wrapper.edge_group_names) == int(Edim):
        edge_name_list = list(wrapper.edge_group_names)
        edge_kind_label = "Neighbor" if str(edge_grouping) == "neighbor" else "Edge"
    else:
        edge_name_list = [f"edge_group_{i}" for i in range(int(Edim))]
        edge_kind_label = "Edge"

    topk_rows_each_epoch = []  # long-form table: epoch, kind, rank, name, gate


    feat_logits = nn.Parameter(0.1 * torch.randn(Fdim, device=device))
    edge_logits = nn.Parameter(0.1 * torch.randn(Edim, device=device)) if Edim > 0 else None
    mask_params = [feat_logits] + ([edge_logits] if edge_logits is not None else [])
    opt = torch.optim.Adam(mask_params, lr=lr)

    orig = float(wrapper.original_pred())
    orig_t = torch.tensor(orig, device=device)

    # print(f"🧠 [MaskOpt] target_node={int(target_node_idx)} explain_pos={explain_pos}/{T-1} orig={orig:.6f}")
    # print(f"   use_subgraph={use_subgraph}, num_hops={num_hops}, undirected={undirected}, feat_dim={Fdim}, edge_params={Edim}")
    # print(f"   edge_grouping={edge_grouping}")

    if disable_cudnn_rnn:
        cudnn_ctx = _DisableCudnn()
    else:
        class _Null:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        cudnn_ctx = _Null()

    best = {"loss": float("inf"), "feat": None, "edge": None, "pred": None}
    hist_rows = []  # per-epoch diagnostics

    def _budget_loss(gate, budget, denom):
        # gate may be None when Edim==0
        if (budget is None) or (gate is None) or (not torch.is_tensor(gate)) or (gate.numel() == 0):
            return feat_logits.new_zeros(())  # safe scalar on correct device/dtype
        return ((gate.sum() - float(budget)) / float(max(1, denom))) ** 2


    for ep in range(1, epochs + 1):
        opt.zero_grad()

        feat_gate = torch.sigmoid(feat_logits)
        edge_gate = torch.sigmoid(edge_logits) if (Edim > 0 and edge_logits is not None) else None

        with cudnn_ctx:
            pred = wrapper.predict_with_gates(feat_gate, edge_gate)

        loss_fid = (pred - orig_t) ** 2

        if use_contrastive:
            feat_gate_drop = (1.0 - feat_gate).clamp(0.0, 1.0)
            edge_gate_drop = (1.0 - edge_gate).clamp(0.0, 1.0) if edge_gate is not None else None
            with cudnn_ctx:
                pred_drop = wrapper.predict_with_gates(feat_gate_drop, edge_gate_drop)
            delta = (pred_drop - orig_t).abs()
            margin_t = torch.as_tensor(float(contrastive_margin), device=device, dtype=delta.dtype)
            loss_contrast = F.relu(margin_t - delta)
        else:
            delta = pred.new_zeros(())
            loss_contrast = pred.new_zeros(())

        loss_feat_size = feat_gate.mean()
        loss_feat_ent  = _binary_entropy(feat_gate).mean()

        if edge_gate is not None and edge_gate.numel() > 0:
            loss_edge_size = edge_gate.mean()
            loss_edge_ent  = _binary_entropy(edge_gate).mean()
        else:
            loss_edge_size = pred.new_zeros(())
            loss_edge_ent  = pred.new_zeros(())

        loss_budget = pred.new_zeros(())
        if float(budget_weight) > 0.0:
            loss_budget = (
                _budget_loss(feat_gate, budget_feat, Fdim)
                + _budget_loss(edge_gate, budget_edge, max(1, Edim))
            )

        loss = (
            float(fid_weight) * loss_fid
            + float(contrastive_weight) * loss_contrast
            + float(budget_weight) * loss_budget
            + coeffs["node_feat_size"] * loss_feat_size
            + coeffs["node_feat_ent"]  * loss_feat_ent
            + coeffs["edge_size"]      * loss_edge_size
            + coeffs["edge_ent"]       * loss_edge_ent
        )


        # ---- diagnostics record (every epoch) ----
        if log_diagnostics:
            with torch.no_grad():
                fg = feat_gate.detach()
                eg = edge_gate.detach() if edge_gate is not None else None

                # --- helpers ---
                def _fmt_thr(thr: float) -> str:
                    # 0.01 -> "0p01"
                    s = f"{thr:g}"
                    return s.replace(".", "p").replace("-", "m")

                def _safe_quantiles(t: torch.Tensor, probs):
                    if t is None or t.numel() == 0:
                        return [float("nan")] * len(probs)
                    probs_t = torch.tensor(list(probs), device=t.device, dtype=torch.float32)
                    try:
                        q = torch.quantile(t.float(), probs_t)
                        return [float(v.item()) for v in q]
                    except Exception:
                        # fallback (older torch)
                        arr = t.detach().float().cpu().numpy()
                        import numpy as _np
                        return [float(_np.quantile(arr, p)) for p in probs]

                def _cnt_le_ge(t: torch.Tensor, thr: float):
                    if t is None or t.numel() == 0:
                        return 0, 0
                    le = int((t <= thr).sum().item())
                    ge = int((t >= (1.0 - thr)).sum().item())
                    return le, ge

                # --- thresholds to log ---
                thr_list = diag_thr_list
                if thr_list is None:
                    thr_list = [0.01, 0.05, 0.1]
                # sanitize / unique / sorted
                thr_list = sorted({float(x) for x in thr_list if x is not None and float(x) > 0.0 and float(x) < 0.5})

                # --- legacy single-thr counts (optional) ---
                legacy_feat_le = legacy_feat_ge = legacy_edge_le = legacy_edge_ge = 0
                if diag_log_legacy_thr:
                    legacy_feat_le, legacy_feat_ge = _cnt_le_ge(fg, float(mask_zero_thr))
                    if eg is not None and eg.numel() > 0:
                        legacy_edge_le, legacy_edge_ge = _cnt_le_ge(eg, float(mask_zero_thr))

                # --- weighted components (already computed above) ---
                w_fid = float(fid_weight) * float(loss_fid.item())
                w_con = float(contrastive_weight) * float(loss_contrast.item())
                w_bud = float(budget_weight) * float(loss_budget.item())
                w_fsz = float(coeffs["node_feat_size"]) * float(loss_feat_size.item())
                w_fen = float(coeffs["node_feat_ent"])  * float(loss_feat_ent.item())
                w_esz = float(coeffs["edge_size"])      * float(loss_edge_size.item())
                w_een = float(coeffs["edge_ent"])       * float(loss_edge_ent.item())

                # --- quantiles + sums ---
                q_probs = tuple(float(p) for p in diag_quantiles)
                fg_q = _safe_quantiles(fg, q_probs)
                eg_q = _safe_quantiles(eg, q_probs) if (eg is not None and eg.numel() > 0) else [float("nan")] * len(q_probs)

                feat_sum = float(fg.sum().item()) if fg.numel() > 0 else 0.0
                edge_sum = float(eg.sum().item()) if (eg is not None and eg.numel() > 0) else 0.0

                row = {
                    "epoch": int(ep),
                    "pred": float(pred.item()),
                    "loss_total": float(loss.item()),
                    "loss_fid": float(loss_fid.item()),
                    "loss_contrast": float(loss_contrast.item()),
                    "loss_budget": float(loss_budget.item()),
                    "loss_feat_size": float(loss_feat_size.item()),
                    "loss_feat_ent": float(loss_feat_ent.item()),
                    "loss_edge_size": float(loss_edge_size.item()),
                    "loss_edge_ent": float(loss_edge_ent.item()),
                    "w_fid": w_fid,
                    "w_contrast": w_con,
                    "w_budget": w_bud,
                    "w_feat_size": w_fsz,
                    "w_feat_ent": w_fen,
                    "w_edge_size": w_esz,
                    "w_edge_ent": w_een,

                    "feat_mean": float(fg.mean().item()) if fg.numel() > 0 else 0.0,
                    "feat_max": float(fg.max().item()) if fg.numel() > 0 else 0.0,
                    "edge_mean": float(eg.mean().item()) if (eg is not None and eg.numel() > 0) else 0.0,
                    "edge_max": float(eg.max().item()) if (eg is not None and eg.numel() > 0) else 0.0,

                    # NEW: sums
                    "feat_sum": feat_sum,
                    "edge_sum": edge_sum,
                }

                # NEW: quantiles (p10/p50/p90 etc)
                for p, v in zip(q_probs, fg_q):
                    row[f"feat_p{int(round(p*100)):02d}"] = float(v)
                for p, v in zip(q_probs, eg_q):
                    row[f"edge_p{int(round(p*100)):02d}"] = float(v)

                # NEW: multi-threshold counts
                for thr in thr_list:
                    key = _fmt_thr(thr)  # "0p01"
                    f_le, f_ge = _cnt_le_ge(fg, thr)
                    e_le, e_ge = _cnt_le_ge(eg, thr) if (eg is not None and eg.numel() > 0) else (0, 0)
                    row[f"feat_le_thr_{key}"] = int(f_le)   # feat <= thr
                    row[f"feat_ge_thr_{key}"] = int(f_ge)   # feat >= 1-thr
                    row[f"edge_le_thr_{key}"] = int(e_le)
                    row[f"edge_ge_thr_{key}"] = int(e_ge)

                # Legacy columns (keep your old plots compatible)
                if diag_log_legacy_thr:
                    row["feat_zero_count"] = int(legacy_feat_le)
                    row["feat_one_count"]  = int(legacy_feat_ge)
                    row["edge_zero_count"] = int(legacy_edge_le)
                    row["edge_one_count"]  = int(legacy_edge_ge)

                hist_rows.append(row)


        # ---- NEW: per-epoch TopK (importance = gate value) ----
        if bool(log_topk_each_epoch) and (int(topk_each_epoch_every) > 0) and (ep % int(topk_each_epoch_every) == 0):
            with torch.no_grad():
                # Feature TopK
                if feat_gate is not None and feat_gate.numel() > 0 and int(topk_each_epoch_feat) > 0:
                    kf = min(int(topk_each_epoch_feat), int(feat_gate.numel()))
                    vals_f, idx_f = torch.topk(feat_gate.detach(), k=kf, largest=True, sorted=True)
                    for r, (j, v) in enumerate(zip(idx_f.tolist(), vals_f.tolist()), start=1):
                        if float(v) < float(topk_each_epoch_min_gate):
                            continue
                        nm = feat_name_list[j] if j < len(feat_name_list) else f"feat_{j}"
                        topk_rows_each_epoch.append({
                            "epoch": int(ep),
                            "kind": "Feature",
                            "rank": int(r),
                            "name": str(nm),
                            "gate": float(v),
                        })

                # Edge/Neighbor TopK
                if edge_gate is not None and edge_gate.numel() > 0 and int(topk_each_epoch_edge) > 0:
                    ke = min(int(topk_each_epoch_edge), int(edge_gate.numel()))
                    vals_e, idx_e = torch.topk(edge_gate.detach(), k=ke, largest=True, sorted=True)
                    for r, (j, v) in enumerate(zip(idx_e.tolist(), vals_e.tolist()), start=1):
                        if float(v) < float(topk_each_epoch_min_gate):
                            continue
                        nm = edge_name_list[j] if j < len(edge_name_list) else f"edge_group_{j}"
                        topk_rows_each_epoch.append({
                            "epoch": int(ep),
                            "kind": str(edge_kind_label),  # "Neighbor" when grouping=neighbor
                            "rank": int(r),
                            "name": str(nm),
                            "gate": float(v),
                        })

                # Optional: # print to stdout (top few only)
                pN = int(topk_each_epoch_print)
                if pN > 0:
                    # # print top features
                    if feat_gate is not None and feat_gate.numel() > 0:
                        kf2 = min(pN, int(topk_each_epoch_feat), int(feat_gate.numel()))
                        vals_f2, idx_f2 = torch.topk(feat_gate.detach(), k=kf2, largest=True, sorted=True)
                        items_f = []
                        for j, v in zip(idx_f2.tolist(), vals_f2.tolist()):
                            nm = feat_name_list[j] if j < len(feat_name_list) else f"feat_{j}"
                            items_f.append(f"{nm}:{float(v):.4f}")
                        # print(f"    [TopK ep={ep}] Feature  : " + ", ".join(items_f))

                    # # print top neighbors/edges
                    if edge_gate is not None and edge_gate.numel() > 0:
                        ke2 = min(pN, int(topk_each_epoch_edge), int(edge_gate.numel()))
                        vals_e2, idx_e2 = torch.topk(edge_gate.detach(), k=ke2, largest=True, sorted=True)
                        items_e = []
                        for j, v in zip(idx_e2.tolist(), vals_e2.tolist()):
                            nm = edge_name_list[j] if j < len(edge_name_list) else f"edge_group_{j}"
                            items_e.append(f"{nm}:{float(v):.4f}")
                        # print(f"    [TopK ep={ep}] {edge_kind_label:<9}: " + ", ".join(items_e))

            
        # IMPORTANT: backward/step must run regardless of topK logging on/off
        loss.backward()
        opt.step()

        lval = float(loss.item())
        if lval < best["loss"]:
            best["loss"] = lval
            best["feat"] = feat_gate.detach().clone()
            best["edge"] = edge_gate.detach().clone() if edge_gate is not None else None
            best["pred"] = float(pred.detach().item())

        if (ep == 1) or (ep % print_every == 0) or (ep == epochs):
            feat_max = float(feat_gate.max().item()) if feat_gate.numel() > 0 else 0.0
            edge_max = float(edge_gate.max().item()) if (edge_gate is not None and edge_gate.numel() > 0) else 0.0
            # print(
            #     f"  [MaskOpt] ep={ep:4d} loss={lval:.6e} fid={float(loss_fid.item()):.3e} "
            #     f"drop_d={float(delta.item()):.3e} pred={float(pred.item()):.6f} feat_max={feat_max:.4f} edge_max={edge_max:.4f}"
            # )

    feat_gate = best["feat"].clamp(0.0, 1.0) if best["feat"] is not None else None
    edge_gate = best["edge"].clamp(0.0, 1.0) if best["edge"] is not None else None

    def _debug_single_edge_drop(wrapper, feat_gate, edge_gate, drop_i=0):
        if edge_gate is None or edge_gate.numel() == 0:
            print("[impact-debug] no edge_gate")
            return

        ones_feat = torch.ones_like(feat_gate)
        ones_edge = torch.ones_like(edge_gate)

        with torch.no_grad():
            base = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())

            e = ones_edge.clone()
            e[drop_i] = 0.0
            p = float(wrapper.predict_with_gates(ones_feat, e).item())

        print("[impact-debug] drop_i", int(drop_i))
        print("[impact-debug] base", base, "pred_drop", p, "diff(base-pred_drop)", (base - p))

    # ===== Experiment 2: gate distribution / saturation stats =====
    try:
        feat_stats = summarize_gate_tensor(feat_gate, name="feat_gate")
        edge_stats = summarize_gate_tensor(edge_gate, name="edge_gate")

        print("[gate-stats] feat:", feat_stats)
        print("[gate-stats] edge:", edge_stats)

        # 保存（CSV）
        stats_rows = [feat_stats, edge_stats]
        stats_df = pd.DataFrame(stats_rows)
        stats_csv = f"maskopt_gate_stats_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
        stats_df.to_csv(stats_csv, index=False)

        # MLflow artifact に残す（既存の _ml がある前提）
        if _ml is not None:
            _ml.log_artifact(stats_csv, artifact_path="xai")
            # 主要メトリクスだけ metrics にも（見やすさ優先）
            if feat_stats["mean"] is not None:
                _ml.log_metric("maskopt_feat_gate_mean", feat_stats["mean"])
                _ml.log_metric("maskopt_feat_gate_ratio_ge_0.9", feat_stats.get("ratio_ge_0.9", 0.0))
            if edge_stats["mean"] is not None:
                _ml.log_metric("maskopt_edge_gate_mean", edge_stats["mean"])
                _ml.log_metric("maskopt_edge_gate_ratio_ge_0.9", edge_stats.get("ratio_ge_0.9", 0.0))

        os.remove(stats_csv)
    except Exception as e:
        print(f"⚠️ gate-stats logging failed: {e}")
    
    def log_gate_stats_exp2(
        feat_gate: torch.Tensor | None,
        edge_gate: torch.Tensor | None,
        tag: str,
        target_node_idx: int,
        explain_pos: int,
        artifact_path: str = "xai/maskopt_diag",
        thr_list=(0.01, 0.05, 0.1, 0.9, 0.95, 0.99),
    ):
        feat_stats = summarize_gate_tensor(feat_gate, name="feat_gate", thr_list=thr_list)
        edge_stats = summarize_gate_tensor(edge_gate, name="edge_gate", thr_list=thr_list)

        stats_df = pd.DataFrame([feat_stats, edge_stats])
        stats_csv = f"maskopt_gate_stats_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
        stats_df.to_csv(stats_csv, index=False)

        # MLflow（run があるときだけ）
        if mlflow.active_run() is not None:
            mlflow.log_artifact(stats_csv, artifact_path=artifact_path)

            step = int(explain_pos)

            # 代表値だけメトリクス化（比較しやすい）
            if feat_stats["mean"] is not None:
                mlflow.log_metric("maskopt_feat_gate_mean", float(feat_stats["mean"]), step=step)
                if "ratio_ge_0.9" in feat_stats:
                    mlflow.log_metric("maskopt_feat_gate_ratio_ge_0.9", float(feat_stats["ratio_ge_0.9"]), step=step)
                if "ratio_le_0.1" in feat_stats:
                    mlflow.log_metric("maskopt_feat_gate_ratio_le_0.1", float(feat_stats["ratio_le_0.1"]), step=step)

            if edge_stats["mean"] is not None:
                mlflow.log_metric("maskopt_edge_gate_mean", float(edge_stats["mean"]), step=step)
                if "ratio_ge_0.9" in edge_stats:
                    mlflow.log_metric("maskopt_edge_gate_ratio_ge_0.9", float(edge_stats["ratio_ge_0.9"]), step=step)
                if "ratio_le_0.1" in edge_stats:
                    mlflow.log_metric("maskopt_edge_gate_ratio_le_0.1", float(edge_stats["ratio_le_0.1"]), step=step)

        try:
            os.remove(stats_csv)
        except Exception:
            pass
    # feat_gate / edge_gate が best に確定した直後
    if mlflow_log:
        log_gate_stats_exp2(
            feat_gate=feat_gate,
            edge_gate=edge_gate,
            tag=str(tag),
            target_node_idx=int(target_node_idx),
            explain_pos=int(explain_pos),
        )




    # ---- write diagnostics artifacts (loss curves / mask zeros/ones / gate hist) ----
    if log_diagnostics and len(hist_rows) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df_hist = pd.DataFrame(hist_rows)
            loss_csv = f"maskopt_loss_hist_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
            df_hist.to_csv(loss_csv, index=False, float_format="%.8e")

            # total loss
            plt.figure()
            plt.plot(df_hist["epoch"], df_hist["loss_total"])
            plt.xlabel("epoch")
            plt.ylabel("loss_total")
            plt.title(f"MaskOpt Loss Total ({tag})")
            plt.tight_layout()
            loss_png = f"maskopt_loss_total_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.png"
            plt.savefig(loss_png, dpi=200)
            plt.close()

            # weighted components (sum to loss_total)
            plt.figure()
            for col in ["w_fid","w_contrast","w_budget","w_feat_size","w_feat_ent","w_edge_size","w_edge_ent"]:
                if col in df_hist.columns:
                    plt.plot(df_hist["epoch"], df_hist[col], label=col)
            plt.xlabel("epoch")
            plt.ylabel("weighted term")
            plt.title(f"MaskOpt Weighted Terms ({tag})")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            terms_png = f"maskopt_loss_terms_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.png"
            plt.savefig(terms_png, dpi=200)
            plt.close()

            # mask near-zero / near-one counts
            # --- plot: gate saturation counts (multi-threshold) ---
            thr_list = diag_thr_list
            if thr_list is None:
                thr_list = [0.01, 0.05, 0.1]
            thr_list = sorted({float(x) for x in thr_list if x is not None and float(x) > 0.0 and float(x) < 0.5})

            def _fmt_thr(thr: float) -> str:
                # 0.01 -> "0p01" (column key suffix)
                s = f"{thr:g}"
                return s.replace(".", "p").replace("-", "m")

            plt.figure()
            for thr in thr_list:
                key = _fmt_thr(thr)
                c1 = f"feat_le_thr_{key}"  # feat <= thr
                c2 = f"edge_le_thr_{key}"  # edge <= thr
                c3 = f"feat_ge_thr_{key}"  # feat >= 1-thr
                c4 = f"edge_ge_thr_{key}"  # edge >= 1-thr

                if c1 in df_hist.columns:
                    plt.plot(df_hist["epoch"], df_hist[c1], label=f"feat<= {thr:g}")
                if c2 in df_hist.columns:
                    plt.plot(df_hist["epoch"], df_hist[c2], label=f"edge<= {thr:g}")
                if c3 in df_hist.columns:
                    plt.plot(df_hist["epoch"], df_hist[c3], label=f"feat>= {1.0-thr:g}")
                if c4 in df_hist.columns:
                    plt.plot(df_hist["epoch"], df_hist[c4], label=f"edge>= {1.0-thr:g}")

            plt.xlabel("epoch")
            plt.ylabel("count")
            plt.title(f"MaskOpt Gate Saturation Count ({tag}) thr={thr_list}")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            zero_png = f"maskopt_mask_zeros_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.png"
            plt.savefig(zero_png, dpi=200)
            plt.close()


            # final gate histograms
            feat_hist_png = None
            edge_hist_png = None
            if feat_gate is not None and feat_gate.numel() > 0:
                plt.figure()
                plt.hist(feat_gate.detach().cpu().numpy().astype(float), bins=40)
                plt.xlabel("feat_gate")
                plt.ylabel("count")
                plt.title(f"Final Feature Gate Histogram ({tag})")
                plt.tight_layout()
                feat_hist_png = f"maskopt_feat_gate_hist_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.png"
                plt.savefig(feat_hist_png, dpi=200)
                plt.close()
            if edge_gate is not None and edge_gate.numel() > 0:
                plt.figure()
                plt.hist(edge_gate.detach().cpu().numpy().astype(float), bins=40)
                plt.xlabel("edge_gate")
                plt.ylabel("count")
                plt.title(f"Final Edge Gate Histogram ({tag})")
                plt.tight_layout()
                edge_hist_png = f"maskopt_edge_gate_hist_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.png"
                plt.savefig(edge_hist_png, dpi=200)
                plt.close()

            if mlflow_log:
                for f in [loss_csv, loss_png, terms_png, zero_png, feat_hist_png, edge_hist_png]:
                    if f and os.path.exists(f):
                        mlflow.log_artifact(f, artifact_path="xai/maskopt_diag")
                        try:
                            os.remove(f)
                        except Exception:
                            pass
        except Exception as e:
            print(f"⚠️ MaskOpt diagnostics failed: {e}")
    

    # ---- NEW: persist per-epoch TopK table ----
    if bool(log_topk_each_epoch) and len(topk_rows_each_epoch) > 0:
        try:
            df_topk = pd.DataFrame(topk_rows_each_epoch)
            topk_csv = f"maskopt_topk_each_epoch_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
            df_topk.to_csv(topk_csv, index=False, float_format="%.8e")

            if mlflow_log:
                mlflow.log_artifact(topk_csv, artifact_path="xai/maskopt_diag")

            try:
                os.remove(topk_csv)
            except Exception:
                pass
        except Exception as e:
            print(f"⚠️ per-epoch TopK export failed: {e}")



    ones_feat = torch.ones(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None

    def _thr(diff, pred_base, eps_abs, eps_rel):
        return max(float(eps_abs), float(eps_rel) * abs(float(pred_base)))

    def _direction(diff, pred_base, eps_abs, eps_rel):
        th = _thr(diff, pred_base, eps_abs, eps_rel)
        if abs(diff) <= th:
            return "Zero (0)"
        return "Positive (+)" if diff > 0 else "Negative (-)"

    with torch.no_grad():
        with cudnn_ctx:
            pred_unmasked = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())
            pred_masked   = float(wrapper.predict_with_gates(feat_gate, edge_gate).item()) if (feat_gate is not None) else pred_unmasked

            rel = abs(pred_masked - pred_unmasked) / (abs(pred_unmasked) + 1e-12)
            print(f"[gate-check] rel_delta = {rel:.6e}")

            print("[gate-check] pred_unmasked =", pred_unmasked)
            print("[gate-check] pred_masked   =", pred_masked)
            print("[gate-check] delta(masked-unmasked) =", (pred_masked - pred_unmasked))

            # (B) 強制ゼロ
            zeros_feat = torch.zeros_like(ones_feat)
            zeros_edge = torch.zeros_like(ones_edge) if (ones_edge is not None) else None

            pred_feat0 = float(wrapper.predict_with_gates(zeros_feat, ones_edge).item())
            pred_edge0 = float(wrapper.predict_with_gates(ones_feat, zeros_edge).item()) if zeros_edge is not None else pred_unmasked

            print("[gate-check] delta_feat0 =", (pred_feat0 - pred_unmasked))
            print("[gate-check] delta_edge0 =", (pred_edge0 - pred_unmasked))

            # (C) 1本ドロップ（edge があるときだけ）
            if zeros_edge is not None and ones_edge.numel() > 0:
                drop_i = 0  # まずは 0 番でOK。後で top edge の index に差し替える
                e1 = ones_edge.clone()
                e1[drop_i] = 0.0
                pred_drop1 = float(wrapper.predict_with_gates(ones_feat, e1).item())
                print("[gate-check] delta_drop1(edge0) =", (pred_drop1 - pred_unmasked))

    _debug_single_edge_drop(wrapper, feat_gate, edge_gate, drop_i=0)


    # ---- Feature importance + impact (gate ablation) ----
    feat_rows = []
    df_feat = pd.DataFrame()

    if feat_gate is not None and feat_gate.numel() > 0:
        feat_np = feat_gate.detach().cpu().numpy()
        top_feat_idx = np.argsort(feat_np)[::-1][:topk_feat]

        for j in top_feat_idx:
            imp = float(feat_np[j])
            if imp < min_show:
                continue

            refs = []
            if impact_reference in ("unmasked", "both"):
                base_f = ones_feat.clone()
                base_e = ones_edge.clone() if ones_edge is not None else None
                ab_f = base_f.clone()
                ab_f[j] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(ab_f, base_e).item())
                diff = pred_unmasked - pred_abl
                refs.append(("unmasked", diff, _direction(diff, pred_unmasked, eps_abs_feat, eps_rel_feat)))

            if impact_reference in ("masked", "both"):
                base_f = feat_gate.clone()
                base_e = edge_gate.clone() if edge_gate is not None else None
                ab_f = base_f.clone()
                ab_f[j] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(ab_f, base_e).item())
                diff = pred_masked - pred_abl
                refs.append(("masked", diff, _direction(diff, pred_masked, eps_abs_feat, eps_rel_feat)))

            name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
            row = {"Type": "Feature", "feature_idx": int(j), "Name": name, "Importance": imp}

            for key, diff, direc in refs:
                row[f"Score_Impact({key})"] = float(diff)
                row[f"Direction({key})"] = direc
            feat_rows.append(row)

    df_feat = pd.DataFrame(feat_rows)
    if not df_feat.empty:
        df_feat.insert(0, "target_node", int(target_node_idx))
        df_feat.insert(1, "explain_pos", int(explain_pos))
        df_feat.insert(2, "tag", str(tag))
        df_feat = df_feat.sort_values("Importance", ascending=False).reset_index(drop=True)
    # ---- Edge importance + impact (GROUP gate ablation) ----
    edge_rows = []
    df_edge = pd.DataFrame()

    if edge_gate is not None and edge_gate.numel() > 0:
        edge_np = edge_gate.detach().cpu().numpy()
        top_edge_idx = np.argsort(edge_np)[::-1][:topk_edge]

        group_names = None
        if getattr(wrapper, "edge_group_names", None) is not None and len(wrapper.edge_group_names) == Edim:
            group_names = list(wrapper.edge_group_names)
        if group_names is None:
            group_names = [f"edge_group_{i}" for i in range(Edim)]

        for gidx in top_edge_idx:
            imp = float(edge_np[gidx])
            if imp < min_show:
                continue

            refs = []
            if impact_reference in ("unmasked", "both"):
                base_f = ones_feat.clone()
                base_e = ones_edge.clone() if ones_edge is not None else None
                ab_e = base_e.clone() if base_e is not None else None
                if ab_e is not None:
                    ab_e[gidx] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(base_f, ab_e).item())
                diff = pred_unmasked - pred_abl
                refs.append(("unmasked", diff, _direction(diff, pred_unmasked, eps_abs_edge, eps_rel_edge)))

            if impact_reference in ("masked", "both"):
                base_f = feat_gate.clone()
                base_e = edge_gate.clone()
                ab_e = base_e.clone()
                ab_e[gidx] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(base_f, ab_e).item())
                diff = pred_masked - pred_abl
                refs.append(("masked", diff, _direction(diff, pred_masked, eps_abs_edge, eps_rel_edge)))

            nm = group_names[gidx] if gidx < len(group_names) else f"edge_group_{gidx}"
            row = {"Type": "Edge", "Name": nm, "Importance": imp}
            for key, diff, direc in refs:
                row[f"Score_Impact({key})"] = float(diff)
                row[f"Direction({key})"] = direc
            edge_rows.append(row)

        df_edge = pd.DataFrame(edge_rows)
        if not df_edge.empty:
            df_edge.insert(0, "target_node", int(target_node_idx))
            df_edge.insert(1, "explain_pos", int(explain_pos))
            df_edge.insert(2, "tag", str(tag))
            df_edge = df_edge.sort_values("Importance", ascending=False).reset_index(drop=True)

    meta = {
        "orig_pred": float(orig),
        "best_pred": float(best["pred"]),
        "best_loss": float(best["loss"]),
        "target_node": int(target_node_idx),
        "explain_pos": int(explain_pos),
        "T": int(T),
        "feat_dim": int(Fdim),
        "edge_params": int(Edim),
        "edge_grouping": str(edge_grouping),
        "pred_unmasked": float(pred_unmasked),
        "pred_masked": float(pred_masked),
        "impact_reference": str(impact_reference),
        "budget_feat": None if budget_feat is None else float(budget_feat),
        "budget_edge": None if budget_edge is None else float(budget_edge),
        "budget_weight": float(budget_weight),
        "coeffs": dict(coeffs),
        "fid_weight": float(fid_weight),
        "eps_abs_feat": float(eps_abs_feat),
        "eps_rel_feat": float(eps_rel_feat),
        "eps_abs_edge": float(eps_abs_edge),
        "eps_rel_edge": float(eps_rel_edge),
    }

    # ---- MLflow artifacts ----
    if mlflow_log:
        try:
            # CSV
            feat_csv = f"maskopt_feat_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
            edge_csv = f"maskopt_edge_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
            df_feat.to_csv(feat_csv, index=False, float_format="%.8e")
            df_edge.to_csv(edge_csv, index=False, float_format="%.8e")
            _ml.log_artifact(feat_csv, artifact_path="xai")
            _ml.log_artifact(edge_csv, artifact_path="xai")
            os.remove(feat_csv)
            os.remove(edge_csv)

            # full gates
            gates_npz = f"maskopt_gates_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.npz"
            np.savez_compressed(
                gates_npz,
                feat_gate=feat_gate.detach().cpu().numpy().astype(np.float32) if feat_gate is not None else None,
                edge_gate=edge_gate.detach().cpu().numpy().astype(np.float32) if edge_gate is not None else None,
                meta=meta,
            )
            _ml.log_artifact(gates_npz, artifact_path="xai")
            os.remove(gates_npz)

            # edge group map (for real-name restore)
            if (edge_grouping == "neighbor") and (getattr(wrapper, "edge_group_meta", None) is not None):
                m = pd.DataFrame(wrapper.edge_group_meta)
                m.insert(0, "group_id", np.arange(len(m), dtype=int))
                m.insert(1, "target_node", int(target_node_idx))
                m.insert(2, "explain_pos", int(explain_pos))
                map_csv = f"maskopt_edge_group_map_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
                m.to_csv(map_csv, index=False)
                _ml.log_artifact(map_csv, artifact_path="xai")
                os.remove(map_csv)

            # meta json
            import json
            mpath = f"maskopt_meta_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.json"
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            _ml.log_artifact(mpath, artifact_path="xai")
            os.remove(mpath)
        except Exception as e:
            print(f"⚠️ MLflow log failed: {e}")

    return df_feat, df_edge, meta


import gc

import numpy as np
import torch

def _get_l2g(g):
    for attr in ["global_node_ids", "n_id", "node_ids", "orig_n_id", "local2global"]:
        if hasattr(g, attr):
            a = getattr(g, attr)
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            return np.asarray(a)
    return None

def count_nonself_out_edges(g, target_gid: int):
    l2g = _get_l2g(g)
    if l2g is None:
        u = int(target_gid)
        has_u = True
    else:
        hit = np.where(l2g == int(target_gid))[0]
        if len(hit) == 0:
            return 0, None, False
        u = int(hit[0])
        has_u = True

    ei = g.edge_index
    src = ei[0].detach().cpu().numpy()
    dst = ei[1].detach().cpu().numpy()
    nonself = (src == u) & (dst != u)
    return int(nonself.sum()), u, has_u


def run_maskopt_for_all_months(
    model,
    monthly_graphs,
    target_node_global_idx,
    feature_names,
    node_to_idx,
    device,
    run_name=None,
    num_hops=6,
    use_subgraph=True,
    # 追加: “MaskOptでedgeが空になった時のフォールバック”
    fallback_rerun_without_subgraph=True,
):
    """
    Driver: run MaskOpt for every month pos (T months).

    追加したこと:
    - 各 explain_pos で「グラフ上に target の非self out edge があるか」を事前に数える
    - df_edge が空になった月だけ、(option) use_subgraph=False で再実行して原因切り分け
    """
    model.eval()

    # あなたの実装が「最後の月をラベルにして入力は monthly_graphs[:-1]」ならそのままでOK
    input_graphs = monthly_graphs[:-1]
    T = len(input_graphs)
    month_labels = [f"pos_{i}" for i in range(T)]

    all_feat, all_edge = [], []
    diag_rows = []

    for explain_pos in range(T):
        label = month_labels[explain_pos]
        tag = label

        g = input_graphs[explain_pos]
        n_out, u_local, has_u = count_nonself_out_edges(g, target_node_global_idx)
        print(f"[diag] pos={explain_pos} has_target={has_u} nonself_out={n_out} "
            f"num_nodes={g.num_nodes} num_edges={g.edge_index.size(1)}")
        print(f"\n[xai] pos={explain_pos}/{T-1} {label}  has_target={has_u}  nonself_out_edges={n_out}  use_subgraph={use_subgraph}")

        # 診断ログ（あとでCSVにできる）
        diag_rows.append({
            "explain_pos": explain_pos,
            "label": label,
            "has_target": bool(has_u),
            "nonself_out_edges": int(n_out),
            "use_subgraph": bool(use_subgraph),
        })

        # グラフに target がそもそもいないなら、MaskOpt以前の問題
        if not has_u:
            print(f"[WARN] target_node {target_node_global_idx} is NOT in node-set at {label}. "
                  f"-> graph construction / active-node filtering issue.")
            continue

        # グラフ上で非self out が 0 なら、「incidentが落ちてる」か「edge_indexがincidentを持ってない」問題
        if n_out == 0:
            print(f"[WARN] graph has ZERO non-self out edges at {label}. "
                  f"-> either this month really has no incident edges for this user in graph, "
                  f"or incident edges are not included in edge_index for subgraph/sequence.")
            # それでも MaskOpt を回してみたいなら続行してOK（多分 df_edge 空）
            # continue

        # ---- 1st try ----
        try:
            df_feat, df_edge, meta = maskopt_e2e_explain(
                model=model,
                input_graphs=input_graphs,
                target_node_idx=target_node_global_idx,
                explain_pos=explain_pos,
                feature_names=feature_names,
                node_to_idx=node_to_idx,
                device=device,
                use_subgraph=use_subgraph,
                num_hops=num_hops,
                edge_mask_scope="incident",
                edge_grouping="neighbor",
                fid_weight=2000.0,
                coeffs={"edge_size":0.08,"edge_ent":0.15,"node_feat_size":0.02,"node_feat_ent":0.15},
                budget_feat=10, budget_edge=20, budget_weight=1.0,
                impact_reference="masked",
                use_contrastive=False,
                mlflow_log=True,
                tag=tag,
                log_diagnostics=True,
                diag_thr_list=[0.01, 0.05, 0.1],
                diag_quantiles=(0.1, 0.5, 0.9),
            )

            # 保存
            if df_feat is not None and not df_feat.empty:
                df_feat.insert(3, "month_label", label)
                all_feat.append(df_feat)

            if df_edge is not None and not df_edge.empty:
                df_edge.insert(3, "month_label", label)
                all_edge.append(df_edge)
            else:
                # ここが「存在するはずなのに edge が空」問題の本丸
                print(f"[WARN] df_edge empty at {label} (incident). "
                      f"graph nonself_out_edges={n_out}.")
                # ---- fallback: rerun without subgraph ----
                if fallback_rerun_without_subgraph and use_subgraph:
                    print(f"[fallback] rerun maskopt with use_subgraph=False at {label}")
                    df_feat2, df_edge2, meta2 = maskopt_e2e_explain(
                        model=model,
                        input_graphs=input_graphs,
                        target_node_idx=target_node_global_idx,
                        explain_pos=explain_pos,
                        feature_names=feature_names,
                        node_to_idx=node_to_idx,
                        device=device,
                        use_subgraph=False,            # ★ここだけ変える
                        num_hops=num_hops,
                        edge_mask_scope="incident",
                        edge_grouping="neighbor",
                        fid_weight=2000.0,
                        coeffs={"edge_size":0.08,"edge_ent":0.15,"node_feat_size":0.02,"node_feat_ent":0.15},
                        budget_feat=10, budget_edge=20, budget_weight=1.0,
                        impact_reference="masked",
                        use_contrastive=False,
                        mlflow_log=True,
                        tag=f"{tag}_nosub",
                        log_diagnostics=True,
                        diag_thr_list=[0.01, 0.05, 0.1],
                        diag_quantiles=(0.1, 0.5, 0.9),
                    )
                    if df_edge2 is not None and not df_edge2.empty:
                        df_edge2.insert(3, "month_label", label)
                        df_edge2.insert(4, "fallback", "nosubgraph")
                        all_edge.append(df_edge2)
                    else:
                        print(f"[fallback] still empty df_edge at {label} even without subgraph "
                              f"-> incident edge candidate extraction / grouping bug inside maskopt_e2e_explain.")

        except Exception as e:
            print(f"💥 MaskOpt Error at explain_pos={explain_pos} ({label}): {e}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    df_feat_all = pd.concat(all_feat, ignore_index=True) if all_feat else pd.DataFrame()
    df_edge_all = pd.concat(all_edge, ignore_index=True) if all_edge else pd.DataFrame()

    # 診断ログを出す（これが超役に立つ）
    df_diag = pd.DataFrame(diag_rows)
    if run_name:
        df_diag.to_csv(f"maskopt_diag_{run_name}.csv", index=False)
        if not df_feat_all.empty:
            df_feat_all.to_csv(f"maskopt_feature_all_{run_name}.csv", index=False)
        if not df_edge_all.empty:
            df_edge_all.to_csv(f"maskopt_edge_all_{run_name}.csv", index=False)

    return df_feat_all, df_edge_all, df_diag

# ===================== Additional helpers (sensitivity + MLflow plotting) =====================

def compute_time_step_sensitivity(
    model,
    input_graphs,
    target_node_idx,
    device,
    topk=3,
    score_mode="alpha_x_delta",
    min_delta=1e-6,
):
    """Compute which time steps matter for the final prediction (post-GCN sensitivity)."""
    model = model.to(device)
    model.eval()

    T = len(input_graphs)
    target_node_idx = int(target_node_idx)

    with torch.no_grad():
        seq_gcn = []
        seq_raw = []
        for g in input_graphs:
            g = g.to(device)
            p = model.projection_layer(g.x)
            h = _gcn_forward_concat(model.gcn_encoder, p, g.edge_index, edge_weight=None)
            seq_gcn.append(h[target_node_idx])
            seq_raw.append(p[target_node_idx])

        seq_gcn = torch.stack(seq_gcn, dim=0).unsqueeze(0)  # [1,T,D]
        seq_raw = torch.stack(seq_raw, dim=0).unsqueeze(0)  # [1,T,P]

        pred_full_t, alpha_t = model(seq_gcn, seq_raw, baseline_scores=None)
        pred_full = float(pred_full_t.view(()).item())

        if alpha_t is None:
            alpha = np.ones(T, dtype=np.float32) / float(T)
        else:
            alpha = alpha_t.detach().view(-1).cpu().numpy().astype(np.float32)
            if alpha.size != T:
                alpha = np.ones(T, dtype=np.float32) / float(T)

        rows = []
        for t in range(T):
            sg = seq_gcn.clone()
            sr = seq_raw.clone()
            sg[:, t, :] = 0.0
            sr[:, t, :] = 0.0
            pred_drop_t, _ = model(sg, sr, baseline_scores=None)
            pred_drop = float(pred_drop_t.view(()).item())
            delta = abs(pred_full - pred_drop)
            if delta < float(min_delta):
                delta = 0.0

            if score_mode == "delta":
                score = delta
            elif score_mode == "alpha":
                score = float(alpha[t])
            else:
                score = float(alpha[t]) * delta

            rows.append({
                "pos": int(t),
                "alpha": float(alpha[t]),
                "pred_full": float(pred_full),
                "pred_drop": float(pred_drop),
                "delta_total": float(delta),
                "score": float(score),
            })

    sens_df = pd.DataFrame(rows).sort_values(["score","delta_total","alpha"], ascending=False).reset_index(drop=True)
    selected_positions = sens_df.head(int(topk))["pos"].astype(int).tolist()
    return sens_df, selected_positions, pred_full, alpha


def mlflow_log_pred_scatter(
    y_true,
    y_pred,
    tag="eval",
    step=None,
    artifact_path="plots",
    fname="pred_vs_true_scatter.png",
    title=None,
):
    """Logs a scatter plot (y_true vs y_pred) to MLflow as an artifact."""
    import os
    import numpy as np

    import matplotlib
    matplotlib.use("Agg")
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


def mlflow_log_maskopt_plots(
    df_feat,
    df_edge,
    meta=None,
    tag="pos_0",
    topk_feat=50,
    topk_edge=50,
    artifact_path="xai",
    fname_prefix="maskopt",
):
    """Logs bar plots for top feature/edge importance to MLflow."""
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import mlflow
        mlflow_available = True
    except Exception:
        mlflow_available = False

    print("[debug] df_feat rows:", None if df_feat is None else len(df_feat))
    print("[debug] df_edge rows:", None if df_edge is None else len(df_edge))
    print("[debug] topk_feat/topk_edge:", topk_feat, topk_edge)
    print("[debug] feat columns:", None if df_feat is None else list(df_feat.columns))
    print("[debug] edge columns:", None if df_edge is None else list(df_edge.columns))


    if df_feat is not None and len(df_feat) > 0:
        d = df_feat.sort_values("Importance", ascending=False).head(int(topk_feat))
        # print the number of features being logged
        print(f"🔧 Logging top {len(d)} features for MaskOpt explanation ({tag})")
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
        # print the number of edges being plotted
        print(f"🔧 Logging top {len(d)} edges for MaskOpt explanation ({tag})")
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


from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
from scipy.stats import kendalltau

def _safe_flat(a):
    import numpy as np
    a = np.asarray(a)
    return a.reshape(-1).astype(float)

def _smape(y_true, y_pred, eps=1e-12):
    import numpy as np
    y_true = _safe_flat(y_true)
    y_pred = _safe_flat(y_pred)
    denom = np.maximum(eps, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return float(np.mean(np.abs(y_pred - y_true) / denom))

def _topk_set(arr, k):
    import numpy as np
    arr = _safe_flat(arr)
    k = int(min(max(1, k), arr.size))
    return set(np.argpartition(-arr, k-1)[:k].tolist())

def _ndcg_at_k(y_true, y_pred, k):
    import numpy as np
    from sklearn.metrics import ndcg_score
    y_true = _safe_flat(y_true)
    y_pred = _safe_flat(y_pred)
    k = int(min(max(1, k), y_true.size))
    # sklearn ndcg_score wants shape [n_samples, n_labels]
    try:
        return float(ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k))
    except TypeError:
        # older sklearn without k=
        return float(ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1)))

def compute_eval_metrics(
    y_true,
    y_pred,
    baseline=None,
    ks=(10, 50, 100),
    prefix="test",
    eps=1e-12,
):
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr, spearmanr

    yt = _safe_flat(y_true)
    yp = _safe_flat(y_pred)

    out = {}

    # --- sanity / distribution checks ---
    out[f"{prefix}_n"] = int(yt.size)
    out[f"{prefix}_true_zero_rate"] = float(np.mean(np.abs(yt) <= 0.0))
    out[f"{prefix}_pred_zero_rate"] = float(np.mean(np.abs(yp) <= 0.0))

    # --- regression ---
    out[f"{prefix}_mae"] = float(mean_absolute_error(yt, yp))
    out[f"{prefix}_rmse"] = float(np.sqrt(mean_squared_error(yt, yp)))
    out[f"{prefix}_r2"] = float(r2_score(yt, yp))

    # log-space (align with your log1p(b_target*100) training)
    yt_log = np.log1p(np.maximum(0.0, yt) * 100.0)
    yp_log = np.log1p(np.maximum(0.0, yp) * 100.0)
    out[f"{prefix}_mae_log1p100"] = float(mean_absolute_error(yt_log, yp_log))
    out[f"{prefix}_rmse_log1p100"] = float(np.sqrt(mean_squared_error(yt_log, yp_log)))

    # correlations (guard)
    if yt.size >= 2 and np.std(yt) > 0 and np.std(yp) > 0:
        out[f"{prefix}_pearson"] = float(pearsonr(yt, yp)[0])
        out[f"{prefix}_spearman"] = float(spearmanr(yt, yp)[0])
        out[f"{prefix}_kendalltau"] = float(kendalltau(yt, yp).correlation)
    else:
        out[f"{prefix}_pearson"] = float("nan")
        out[f"{prefix}_spearman"] = float("nan")
        out[f"{prefix}_kendalltau"] = float("nan")

    out[f"{prefix}_smape"] = _smape(yt, yp, eps=eps)

    # --- ranking metrics ---
    for k in ks:
        k = int(min(max(1, k), yt.size))
        out[f"{prefix}_ndcg_at_{k}"] = _ndcg_at_k(yt, yp, k)

        true_top = _topk_set(yt, k)
        pred_top = _topk_set(yp, k)
        inter = len(true_top & pred_top)
        union = len(true_top | pred_top)

        out[f"{prefix}_precision_at_{k}"] = float(inter / max(1, k))      # (= recall_at_k when both sets size k)
        out[f"{prefix}_jaccard_at_{k}"] = float(inter / max(1, union))

        # regret_at_k: “真の上位K平均” − “予測上位K平均”
        true_top_mean = float(np.mean(yt[list(true_top)])) if len(true_top) else 0.0
        pred_top_mean = float(np.mean(yt[list(pred_top)])) if len(pred_top) else 0.0
        out[f"{prefix}_regret_mean_at_{k}"] = float(true_top_mean - pred_top_mean)

        # optional: TopK classification AUC (true_top as positives)
        labels = np.zeros_like(yt, dtype=int)
        for idx in true_top:
            labels[idx] = 1
        # AUC requires both classes present
        if labels.min() != labels.max():
            try:
                out[f"{prefix}_rocauc_top_at_{k}"] = float(roc_auc_score(labels, yp))
                out[f"{prefix}_prauc_top_at_{k}"] = float(average_precision_score(labels, yp))
            except Exception:
                out[f"{prefix}_rocauc_top_at_{k}"] = float("nan")
                out[f"{prefix}_prauc_top_at_{k}"] = float("nan")
        else:
            out[f"{prefix}_rocauc_top_at_{k}"] = float("nan")
            out[f"{prefix}_prauc_top_at_{k}"] = float("nan")

    # --- baseline comparison ---
    if baseline is not None:
        yb = _safe_flat(baseline)
        if yb.size == yt.size:
            base = compute_eval_metrics(yt, yb, baseline=None, ks=ks, prefix=f"{prefix}_baseline", eps=eps)
            out.update(base)
            out[f"{prefix}_mae_improve_vs_baseline"] = float(out[f"{prefix}_baseline_mae"] - out[f"{prefix}_mae"])
            for k in ks:
                k = int(min(max(1, k), yt.size))
                out[f"{prefix}_ndcg_at_{k}_improve_vs_baseline"] = float(
                    out.get(f"{prefix}_ndcg_at_{k}", float("nan")) - out.get(f"{prefix}_baseline_ndcg_at_{k}", float("nan"))
                )

    return out

import numpy as np

def engagement_to_rel_paper(e: np.ndarray) -> np.ndarray:
    """
    InfluencerRank paper (Table 2) の 6段階 relevance level へ変換
    """
    e = np.asarray(e, dtype=float)
    rel = np.zeros_like(e, dtype=int)

    rel[(e >= 0.01) & (e < 0.03)] = 1
    rel[(e >= 0.03) & (e < 0.05)] = 2
    rel[(e >= 0.05) & (e < 0.07)] = 3
    rel[(e >= 0.07) & (e < 0.10)] = 4
    rel[(e >= 0.10)] = 5
    return rel

def dcg_at_k(rels_sorted: np.ndarray, k: int, gain: str = "exp2") -> float:
    """
    rels_sorted: 予測順位でソート済みの relevance 配列（上位から）
    gain:
      - "exp2": (2^rel - 1) / log2(rank+1)  ※典型的NDCG
      - "linear": rel / log2(rank+1)
    """
    k = min(k, rels_sorted.size)
    if k <= 0:
        return 0.0
    rels_sorted = rels_sorted[:k]

    ranks = np.arange(1, k + 1, dtype=float)  # 1-based
    discounts = 1.0 / np.log2(ranks + 1.0)

    if gain == "exp2":
        gains = (2.0 ** rels_sorted) - 1.0
    elif gain == "linear":
        gains = rels_sorted.astype(float)
    else:
        raise ValueError(f"Unknown gain: {gain}")

    return float(np.sum(gains * discounts))

def ndcg_at_k(y_true_rel: np.ndarray, y_pred_score: np.ndarray, k: int, gain: str = "exp2") -> float:
    """
    y_true_rel: relevance（連続でも離散でもOK）
    y_pred_score: 予測スコア（大きいほど上位）
    """
    y_true_rel = np.asarray(y_true_rel)
    y_pred_score = np.asarray(y_pred_score)

    order = np.argsort(-y_pred_score)             # pred降順
    rel_pred_sorted = y_true_rel[order]

    dcg = dcg_at_k(rel_pred_sorted, k=k, gain=gain)

    ideal_order = np.argsort(-y_true_rel)         # idealはrel降順
    rel_ideal_sorted = y_true_rel[ideal_order]
    idcg = dcg_at_k(rel_ideal_sorted, k=k, gain=gain)

    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)

def save_pred_csv(node_ids, y_true, y_pred, idx_to_node, out_csv):
    """
    node_ids: (N,) 予測対象ノードID（ユーザIDのglobal node id）
    y_true  : (N,) 正解スコア
    y_pred  : (N,) 予測スコア
    idx_to_node: dict[int, str] 例: {981081: "00_rocketgirl", ...}
    """
    node_ids = np.asarray(node_ids).reshape(-1)
    y_true   = np.asarray(y_true).reshape(-1)
    y_pred   = np.asarray(y_pred).reshape(-1)

    if not (len(node_ids) == len(y_true) == len(y_pred)):
        raise ValueError(f"length mismatch: ids={len(node_ids)} true={len(y_true)} pred={len(y_pred)}")

    usernames = [idx_to_node.get(int(nid), str(int(nid))) for nid in node_ids]

    df = pd.DataFrame({
        "node_id": node_ids.astype(np.int64),
        "username": usernames,
        "true_score": y_true.astype(np.float64),
        "pred_score": y_pred.astype(np.float64),
    })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return df

def compute_two_ndcgs(df, k_list=(1, 10, 50, 100, 200)):
    """
    df columns: true_score, pred_score
    returns: dict of metrics
    """
    true_e = df["true_score"].to_numpy(dtype=float)
    pred = df["pred_score"].to_numpy(dtype=float)

    # (A) continuous NDCG: relevance = engagement rate (linear gain 推奨)
    # 連続値に exp2 を使うと極端に差が増幅されるので、まずは linear が扱いやすい
    out = {}
    for k in k_list:
        out[f"ndcg_cont_at_{k}"] = ndcg_at_k(true_e, pred, k=k, gain="linear")

    # (B) paper NDCG: relevance level 0..5 (exp2 gain が典型)
    rel_paper = engagement_to_rel_paper(true_e)
    for k in k_list:
        out[f"ndcg_paper_at_{k}"] = ndcg_at_k(rel_paper, pred, k=k, gain="exp2")

    return out

def _make_pos_replaced_seq(
    seq: torch.Tensor,   # [B, T, D]
    raw: torch.Tensor,   # [B, T, P]
    pos: int,
    mode: str,
    global_pos_mean_seq=None,  # [T,D]
    global_pos_mean_raw=None,  # [T,P]
    rng: torch.Generator | None = None,
):
    """
    Return (seq2, raw2) where time-step 'pos' is replaced by baseline.
    """
    B, T, D = seq.shape
    _, _, P = raw.shape
    assert 0 <= pos < T

    seq2 = seq.clone()
    raw2 = raw.clone()

    if mode == "zero":
        seq2[:, pos, :] = 0.0
        raw2[:, pos, :] = 0.0

    elif mode == "user_mean":
        # mean over other positions per user
        if T == 1:
            seq2[:, pos, :] = 0.0
            raw2[:, pos, :] = 0.0
        else:
            mask = torch.ones(T, device=seq.device, dtype=torch.bool)
            mask[pos] = False
            seq2[:, pos, :] = seq[:, mask, :].mean(dim=1)
            raw2[:, pos, :] = raw[:, mask, :].mean(dim=1)

    elif mode == "global_pos_mean":
        assert global_pos_mean_seq is not None and global_pos_mean_raw is not None
        seq2[:, pos, :] = global_pos_mean_seq[pos].view(1, -1)
        raw2[:, pos, :] = global_pos_mean_raw[pos].view(1, -1)

    elif mode == "shuffle_pos":
        # permute users within the batch at the same pos
        if rng is None:
            perm = torch.randperm(B, device=seq.device)
        else:
            perm = torch.randperm(B, device=seq.device, generator=rng)
        seq2[:, pos, :] = seq2[perm, pos, :]
        raw2[:, pos, :] = raw2[perm, pos, :]

    else:
        raise ValueError(f"unknown baseline mode: {mode}")

    return seq2, raw2


@torch.no_grad()
def compute_pos_importance_from_embeddings(
    model,
    f_seq: torch.Tensor,   # [N, T, D]  (GCN output for influencers)
    f_raw: torch.Tensor,   # [N, T, P]  (projection output for influencers)
    baseline_scores: torch.Tensor,  # [N]
    device,
    baseline_modes=("zero", "user_mean", "global_pos_mean", "shuffle_pos"),
    batch_size=1024,
    seed=0,
):
    """
    Returns:
      df_pos_long: columns = [baseline_mode, pos, pred_orig_mean, delta_median, delta_mean, delta_abs_mean]
      deltas: dict[mode] -> np.ndarray [N,T] (optional heavy)
    """
    model.eval()
    N, T, D = f_seq.shape
    _, _, P = f_raw.shape

    # precompute global pos mean (over users) for global_pos_mean mode
    global_pos_mean_seq = f_seq.mean(dim=0).to(device)  # [T,D]
    global_pos_mean_raw = f_raw.mean(dim=0).to(device)  # [T,P]

    # orig preds
    preds_orig = []
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        b_seq = f_seq[i:j].to(device)
        b_raw = f_raw[i:j].to(device)
        b_base = baseline_scores[i:j].to(device)
        p, _ = model(b_seq, b_raw, b_base)
        preds_orig.append(p.view(-1).detach().cpu())
    preds_orig = torch.cat(preds_orig, dim=0)  # [N]
    pred_orig_mean = float(preds_orig.mean().item())

    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))

    rows = []
    # (optional) keep all deltas if you want heatmaps later
    deltas_np = {}

    for mode in baseline_modes:
        # allocate [N,T] deltas on CPU to avoid GPU RAM blow
        deltas_mode = torch.empty((N, T), dtype=torch.float32)

        for pos in range(T):
            preds_mod = []
            for i in range(0, N, batch_size):
                j = min(i + batch_size, N)

                b_seq = f_seq[i:j].to(device)
                b_raw = f_raw[i:j].to(device)
                b_base = baseline_scores[i:j].to(device)

                b_seq2, b_raw2 = _make_pos_replaced_seq(
                    b_seq, b_raw, pos=pos, mode=mode,
                    global_pos_mean_seq=global_pos_mean_seq,
                    global_pos_mean_raw=global_pos_mean_raw,
                    rng=rng,
                )
                p2, _ = model(b_seq2, b_raw2, b_base)
                preds_mod.append(p2.view(-1).detach().cpu())

            preds_mod = torch.cat(preds_mod, dim=0)  # [N]
            delta = (preds_orig - preds_mod).to(torch.float32)  # signed
            deltas_mode[:, pos] = delta

        # aggregate per pos
        delta_median = torch.median(deltas_mode, dim=0).values.numpy()
        delta_mean = deltas_mode.mean(dim=0).numpy()
        delta_abs_mean = deltas_mode.abs().mean(dim=0).numpy()

        for pos in range(T):
            rows.append({
                "baseline_mode": str(mode),
                "pos": int(pos),
                "pred_orig_mean": float(pred_orig_mean),
                "delta_median": float(delta_median[pos]),
                "delta_mean": float(delta_mean[pos]),
                "delta_abs_mean": float(delta_abs_mean[pos]),
            })

        deltas_np[str(mode)] = deltas_mode.numpy()

    df_pos_long = pd.DataFrame(rows).sort_values(["baseline_mode","pos"]).reset_index(drop=True)
    return df_pos_long, deltas_np

import copy


def _apply_node_feature_mask_on_graph(g, target_node: int, masked_feats, mode="zero"):
    g2 = copy.copy(g)
    x = g.x.clone()
    if len(masked_feats) > 0:
        if mode == "zero":
            x[target_node, masked_feats] = 0.0
        else:
            raise ValueError(mode)
    g2.x = x
    return g2



def apply_edge_mask(graph, target_node: int, masked_neighbors):
    edge_index = graph.edge_index
    src, dst = edge_index

    if isinstance(masked_neighbors, list):
        masked_neighbors = torch.tensor(masked_neighbors, dtype=dst.dtype, device=dst.device)
    else:
        masked_neighbors = masked_neighbors.to(device=dst.device, dtype=dst.dtype)

    if masked_neighbors.numel() == 0:
        return graph  # no-op

    keep = ~(
        ((src == target_node) & torch.isin(dst, masked_neighbors)) |
        ((dst == target_node) & torch.isin(src, masked_neighbors))
    )

    graph_m = copy.copy(graph)
    graph_m.edge_index = edge_index[:, keep]
    return graph_m


def deletion_insertion_edge(
    model,
    input_graphs,    # list of T graphs
    target_node,
    baseline_score,
    edge_order,
    mode="deletion",
    random_trials=20,
    device="cpu"
):
    T = len(input_graphs)
    E = len(edge_order)
    k_list = list(range(0, E + 1))

    def predict(graphs):
        with torch.no_grad():
            seq_emb, raw_emb = [], []
            for g in graphs:
                g = g.to(device)
                p_x = model.projection_layer(g.x)
                gcn_out = model.gcn_encoder(p_x, g.edge_index)
                raw_emb.append(p_x[target_node].unsqueeze(0))
                seq_emb.append(gcn_out[target_node].unsqueeze(0))
            f_seq = torch.stack(seq_emb, dim=1)
            f_raw = torch.stack(raw_emb, dim=1)
            p, _ = model(f_seq, f_raw, baseline_score.to(device))
        return float(p.item())

    scores_main = []

    for k in k_list:
        if mode == "deletion":
            masked = edge_order[:k]
        else:
            masked = edge_order[k:]

        graphs_m = [
            apply_edge_mask(g, target_node, masked)
            for g in input_graphs
        ]
        scores_main.append(predict(graphs_m))

    # random
    rand_scores = np.zeros((random_trials, len(k_list)))

    for t in range(random_trials):
        rand_order = np.random.permutation(edge_order).tolist()
        for i, k in enumerate(k_list):
            masked = rand_order[:k] if mode == "deletion" else rand_order[k:]
            graphs_m = [
                apply_edge_mask(g, target_node, masked)
                for g in input_graphs
            ]
            rand_scores[t, i] = predict(graphs_m)

    return k_list, np.array(scores_main), rand_scores.mean(0), rand_scores.std(0)


def predict_seq_only_from_graphs(model, graphs, target_node, baseline_score, device="cpu"):
    with torch.no_grad():
        seq_emb = []
        for g in graphs:
            g = g.to(device)
            p_x = model.projection_layer(g.x)
            gcn_out = model.gcn_encoder(p_x, g.edge_index)
            seq_emb.append(gcn_out[target_node].unsqueeze(0))
        f_seq = torch.stack(seq_emb, dim=1)              # [1,T,D]
        f_raw = torch.zeros((1, f_seq.shape[1], model.projection_dim),
                            device=device, dtype=f_seq.dtype)  # ここはあなたのP次元に合わせる
        b = baseline_score if torch.is_tensor(baseline_score) else torch.tensor([float(baseline_score)])
        p, _ = model(f_seq, f_raw, b.to(device))
    return float(p.view(-1)[0].item())


def diagnose_path_dependency(model, f_seq_user, f_raw_user, base_user, device="cpu"):
    model.eval()
    with torch.no_grad():
        p_full, _ = model(f_seq_user.to(device), f_raw_user.to(device), base_user.to(device))
        p_full = float(p_full.view(-1)[0].item())

        p_seq_only, _ = model(f_seq_user.to(device),
                              torch.zeros_like(f_raw_user).to(device),
                              base_user.to(device))
        p_seq_only = float(p_seq_only.view(-1)[0].item())

        p_raw_only, _ = model(torch.zeros_like(f_seq_user).to(device),
                              f_raw_user.to(device),
                              base_user.to(device))
        p_raw_only = float(p_raw_only.view(-1)[0].item())

    return {"full": p_full, "seq_only": p_seq_only, "raw_only": p_raw_only}


def run_experiment(params, graphs_data, target_node_idx, experiment_id=None):
    """
    FULL run_experiment (no omission) with:
      ✅ consistent nonzero filtering (fixes boolean index mismatch)
      ✅ attention plot for selected user (nz-aligned)
      ✅ pos-importance (baseline replacement) global + user-wise export + selected-user plot
      ✅ deletion/insertion curves:
            - FEATURE (node features)  : recompute embeddings from masked graphs (so it actually affects seq+raw)
            - EDGE (incident neighbors): recompute embeddings from masked graphs
         (both: importance-order vs random mean±std)
      ✅ path-dependency diagnostics (full / seq-only / raw-only)
      ✅ MaskOpt remains as your main explainer; DI uses its ranking when available

    Assumptions (already exist in your codebase):
      - mlflow, device
      - HardResidualInfluencerModel
      - ListMLELoss
      - get_dataset_with_baseline
      - TailMixedListBatchSampler (optional)
      - DataLoader
      - save_model_checkpoint, load_model_from_ckpt, maybe_download_ckpt_from_mlflow
      - compute_eval_metrics, compute_two_ndcgs
      - generate_enhanced_scatter_plot
      - plot_attention_weights
      - mlflow_log_pred_scatter
      - _select_positions_by_attention
      - compute_time_step_sensitivity
      - maskopt_e2e_explain
      - mlflow_log_maskopt_plots
      - compute_pos_importance_from_embeddings   (returns df_pos_long, deltas_np[mode]->[N,T])
    """

    import os
    import gc
    import json
    import copy
    import shutil
    import datetime
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    from torch_geometric.utils import degree
    from torch.utils.data import DataLoader

    # -----------------------------
    # local helpers (keep inside)
    # -----------------------------
    def _to_1d_float(x):
        if isinstance(x, (float, int)):
            return float(x)
        if torch.is_tensor(x):
            return float(x.detach().view(-1)[0].cpu().item())
        return float(x)

    def diagnose_path_dependency(model, f_seq1, f_raw1, base1, device):
        """
        f_seq1: [1,T,D], f_raw1:[1,T,P], base1:[1] tensor
        returns dict of preds: full / seq_only / raw_only
        """
        with torch.no_grad():
            p_full, _ = model(f_seq1.to(device), f_raw1.to(device), base1.to(device))

            # raw-only: zero seq
            p_raw, _ = model(torch.zeros_like(f_seq1).to(device), f_raw1.to(device), base1.to(device))

            # seq-only: zero raw
            p_seq, _ = model(f_seq1.to(device), torch.zeros_like(f_raw1).to(device), base1.to(device))

        return {
            "full": _to_1d_float(p_full),
            "raw_only": _to_1d_float(p_raw),
            "seq_only": _to_1d_float(p_seq),
        }

    def _select_users_for_ice(all_indices_nz, predicted_scores_nz, n_users=60, include_node_id=None):
        """
        予測スコアで層化サンプル（低〜高を均等に拾う）+ 指定ユーザを必ず含める
        returns: rows (nz-row indices)
        """
        n = int(min(max(5, n_users), len(all_indices_nz)))
        order = np.argsort(predicted_scores_nz)  # low -> high
        pick = np.linspace(0, len(order) - 1, n, dtype=int)
        rows = order[pick].astype(int).tolist()

        if include_node_id is not None:
            include_node_id = int(include_node_id)
            hit = np.where(all_indices_nz == include_node_id)[0]
            if len(hit) > 0:
                r = int(hit[0])
                if r not in rows:
                    rows[0] = r  # 先頭を差し替えて必ず入れる
        return rows

    def _predict_users_from_graphs(model, graphs, user_ids_np, baseline_scores_np, device):
        """
        graphs: list[Data] length T
        user_ids_np: [B] global node ids (graph node index)
        baseline_scores_np: [B]
        returns preds_np: [B]
        """
        model.eval()
        user_ids = torch.tensor(user_ids_np, dtype=torch.long, device=device)
        base_t = torch.tensor(baseline_scores_np, dtype=torch.float32, device=device)

        seq_emb, raw_emb = [], []
        with torch.no_grad():
            for g in graphs:
                g = g.to(device)
                p_x = model.projection_layer(g.x)               # [N,P]
                gcn_out = model.gcn_encoder(p_x, g.edge_index)  # [N,D]
                raw_emb.append(p_x.index_select(0, user_ids))   # [B,P]
                seq_emb.append(gcn_out.index_select(0, user_ids))  # [B,D]

            f_seq = torch.stack(seq_emb, dim=1)  # [B,T,D]
            f_raw = torch.stack(raw_emb, dim=1)  # [B,T,P]
            preds, _ = model(f_seq, f_raw, baseline_scores=base_t)
            preds = preds.view(-1).detach().cpu().numpy().astype(float)
        return preds

    def _make_ice_grid(values_np, n_grid=25, qlo=0.05, qhi=0.95):
        """
        分布の外れ値を避けて [qlo, qhi] の範囲を等間隔グリッド化
        """
        v = np.asarray(values_np, dtype=float)
        if v.size == 0:
            return np.linspace(0.0, 1.0, int(n_grid))
        lo = float(np.quantile(v, qlo))
        hi = float(np.quantile(v, qhi))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(v))
            hi = float(np.max(v)) if float(np.max(v)) > lo else (lo + 1.0)
        return np.linspace(lo, hi, int(n_grid), dtype=float)

    def run_ice_pdp_for_feature(
        model,
        input_graphs,
        pos,
        feature_idx,
        user_ids_np,
        usernames,
        baseline_scores_np,
        device,
        run_name,
        tag,
        feature_name=None,
        n_grid=25,
        qlo=0.05,
        qhi=0.95,
        artifact_path="xai/ice",
    ):
        """
        あるpos（月）のある特徴をグリッドで動かして ICE/PDP を作る。
        - 他の特徴/他のposは固定
        - posのgraphだけ x を差し替えて埋め込み再計算 → 予測
        出力: (grid, preds_mat[B,G]) を返し、png/csv を MLflow に保存
        """
        import copy

        pos = int(pos)
        feat = int(feature_idx)
        B = int(len(user_ids_np))

        # グリッド範囲は「そのposにおけるユーザ集合の実値分布」から取る
        x_pos = input_graphs[pos].x.detach().cpu().numpy()
        base_vals = x_pos[user_ids_np, feat]
        grid = _make_ice_grid(base_vals, n_grid=n_grid, qlo=qlo, qhi=qhi)  # [G]

        preds_mat = np.zeros((B, len(grid)), dtype=np.float32)

        # 元の予測（sanity）
        preds_orig = _predict_users_from_graphs(model, input_graphs, user_ids_np, baseline_scores_np, device=device)

        for j, v in enumerate(grid):
            graphs_m = list(input_graphs)
            g0 = input_graphs[pos]
            g2 = copy.copy(g0)
            x2 = g0.x.clone()
            # 同じ v を全ユーザにセット（ICEでは「各ユーザ曲線を v に対して描く」のでこれでOK）
            x2[user_ids_np, feat] = float(v)
            g2.x = x2
            graphs_m[pos] = g2

            preds = _predict_users_from_graphs(model, graphs_m, user_ids_np, baseline_scores_np, device=device)
            preds_mat[:, j] = preds.astype(np.float32)

        # PDP（平均）
        pdp = preds_mat.mean(axis=0)

        # --- plot ---
        plt.figure()
        # ICE: 薄線（全ユーザ）
        for i in range(B):
            plt.plot(grid, preds_mat[i], alpha=0.15, linewidth=1.0)
        # PDP: 太線（平均）
        plt.plot(grid, pdp, linewidth=2.5, label="PDP (mean)")
        plt.xlabel(feature_name or f"feature_{feat}")
        plt.ylabel("pred score")
        title = f"ICE/PDP pos={pos} feat={feature_name or feat} ({tag})"
        plt.title(title)
        plt.legend()
        plt.tight_layout()

        fig_path = f"ice_pdp_pos{pos}_feat{feat}_{tag}_{run_name}.png"
        plt.savefig(fig_path, dpi=220)
        plt.close()
        mlflow.log_artifact(fig_path, artifact_path=artifact_path)
        os.remove(fig_path)

        # --- csv (long) ---
        rows = []
        for j, v in enumerate(grid):
            rows.append(pd.DataFrame({
                "tag": tag,
                "pos": pos,
                "feature_idx": feat,
                "feature_name": feature_name or f"feature_{feat}",
                "grid_value": float(v),
                "node_id": user_ids_np.astype(np.int64),
                "username": usernames,
                "pred_score": preds_mat[:, j].astype(float),
                "pred_orig": preds_orig.astype(float),
            }))
        df_long = pd.concat(rows, ignore_index=True)
        csv_path = f"ice_pdp_pos{pos}_feat{feat}_{tag}_{run_name}.csv"
        df_long.to_csv(csv_path, index=False, float_format="%.8e", encoding="utf-8-sig")
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)
        os.remove(csv_path)

        return grid, preds_mat


    def apply_feature_mask_on_x_target(graph, target_node, masked_feature_idx, mode="zero"):
        g2 = copy.copy(graph)
        x2 = graph.x.clone()
        if mode == "zero":
            x2[int(target_node), masked_feature_idx] = 0.0
        else:
            raise ValueError(mode)
        g2.x = x2
        return g2

    # --------------------------------
    # unpack graphs_data
    # --------------------------------
    run_id = None
    final_test_metrics = None

    monthly_graphs, influencer_indices, node_to_idx, feature_dim, follower_feat_idx, static_cols, dynamic_cols = graphs_data
    idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{params.get('name_prefix', 'Run')}_{current_time_str}"

    # global knobs
    baseline_modes = params.get("POS_BASELINE_MODES", None)
    if baseline_modes is None:
        baseline_modes = ["zero", "user_mean", "global_pos_mean", "shuffle_pos"]

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(params)

        if "note" in params:
            print(f"Note: {params['note']}")

        # -----------------------------
        # build model
        # -----------------------------
        model = HardResidualInfluencerModel(
            feature_dim=feature_dim,
            gcn_dim=params["GCN_DIM"],
            rnn_dim=params["RNN_DIM"],
            num_gcn_layers=params["NUM_GCN_LAYERS"],
            dropout_prob=params["DROPOUT_PROB"],
            projection_dim=params["PROJECTION_DIM"],
        ).to(device)

        mode_run = str(params.get("MODE", "train")).lower()
        ckpt_path = params.get("CKPT_PATH")
        ckpt_run_id = params.get("CKPT_MLFLOW_RUN_ID")
        ckpt_art = params.get("CKPT_MLFLOW_ARTIFACT", "model/model_state.pt")

        if mode_run == "infer":
            if ckpt_run_id:
                ckpt_path = maybe_download_ckpt_from_mlflow(str(ckpt_run_id), str(ckpt_art))
            if not ckpt_path:
                raise ValueError("infer mode requires --ckpt or --mlflow_run_id")
            loaded_model, loaded_feature_dim, _ = load_model_from_ckpt(str(ckpt_path), device=device)
            if int(loaded_feature_dim) != int(feature_dim):
                raise ValueError(f"feature_dim mismatch: ckpt={loaded_feature_dim} vs current={feature_dim}")
            model.load_state_dict(loaded_model.state_dict(), strict=True)
            model.eval()
            mlflow.log_param("infer_only", 1)
            mlflow.log_param("ckpt_path", str(ckpt_path))

        optimizer = torch.optim.Adam(model.parameters(), lr=float(params["LR"]))
        criterion_list = ListMLELoss().to(device)
        criterion_mse = nn.MSELoss().to(device)

        # -----------------------------
        # dataset for training
        # -----------------------------
        train_dataset = get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-2)

        train_input_graphs = monthly_graphs[:-2]
        gpu_graphs = [g.to(device) for g in train_input_graphs]

        # global->local map (only influencers)
        inf_global = torch.tensor(influencer_indices, dtype=torch.long, device=device)
        num_nodes_all = int(gpu_graphs[0].num_nodes)
        global2local = torch.full((num_nodes_all,), -1, dtype=torch.long, device=device)
        global2local[inf_global] = torch.arange(inf_global.numel(), device=device, dtype=torch.long)
        if int((global2local[inf_global] < 0).sum().item()) != 0:
            raise RuntimeError("global2local mapping failed for some influencer indices.")

        # -----------------------------
        # training
        # -----------------------------
        if mode_run != "infer":
            list_size = int(params["LIST_SIZE"])
            if len(train_dataset) < list_size:
                raise RuntimeError(f"train_dataset too small ({len(train_dataset)}) for LIST_SIZE={list_size}")

            safe_bs = min(int(params["BATCH_SIZE"]), len(train_dataset))
            safe_bs = (safe_bs // list_size) * list_size
            safe_bs = max(list_size, safe_bs)

            use_sampler = bool(params.get("USE_SAMPLER", False))
            if use_sampler:
                y_all = train_dataset.tensors[1].detach().cpu().numpy()
                batch_sampler = TailMixedListBatchSampler(
                    y=y_all,
                    list_size=list_size,
                    batch_size=safe_bs,
                    q_hi=float(params.get("RIGHT_Q", 0.90)),
                    q_lo=float(params.get("LEFT_Q", 0.10)),
                    n_hi=int(params.get("N_HIGH", 1)),
                    n_lo=int(params.get("N_LOW", 0)),
                    seed=int(params.get("SEED", 0)),
                    replacement=True,
                )
                dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler)
            else:
                dataloader = DataLoader(train_dataset, batch_size=safe_bs, shuffle=True, drop_last=True)

            for epoch in range(int(params["EPOCHS"])):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                loss_sum = None
                num_batches = 0

                # precompute influencer-only sequences for this epoch
                seq_emb, raw_emb = [], []
                for g in gpu_graphs:
                    p_x = model.projection_layer(g.x)              # [N,P]
                    gcn_out = model.gcn_encoder(p_x, g.edge_index) # [N,D]
                    raw_emb.append(p_x.index_select(0, inf_global))       # [Ninf,P]
                    seq_emb.append(gcn_out.index_select(0, inf_global))   # [Ninf,D]
                full_seq = torch.stack(seq_emb, dim=1)  # [Ninf,T,D]
                full_raw = torch.stack(raw_emb, dim=1)  # [Ninf,T,P]

                # tail weights
                RIGHT_Q = float(params.get("RIGHT_Q", 0.90))
                RIGHT_W = float(params.get("RIGHT_W", 5.0))
                LEFT_Q  = float(params.get("LEFT_Q",  0.10))
                LEFT_W  = float(params.get("LEFT_W",  1.0))

                y_all_np = train_dataset.tensors[1].detach().cpu().numpy().astype(float)
                thr_hi_global = float(np.quantile(y_all_np, RIGHT_Q))
                thr_lo_global = float(np.quantile(y_all_np, LEFT_Q))
                thr_hi_t = torch.tensor(thr_hi_global, device=device, dtype=torch.float32)
                thr_lo_t = torch.tensor(thr_lo_global, device=device, dtype=torch.float32)

                for batch in dataloader:
                    b_idx_global, b_target, b_baseline = batch
                    b_idx_global = b_idx_global.to(device)
                    b_target = b_target.to(device)
                    b_baseline = b_baseline.to(device)

                    b_local = global2local[b_idx_global]
                    if int((b_local < 0).sum().item()) != 0:
                        raise RuntimeError("Found indices not in influencer set (global2local == -1).")

                    b_seq = full_seq.index_select(0, b_local)
                    b_raw = full_raw.index_select(0, b_local)

                    preds, _ = model(b_seq, b_raw, baseline_scores=b_baseline)
                    preds = preds.view(-1)

                    log_target = torch.log1p(b_target * 100.0)
                    log_pred   = torch.log1p(preds * 100.0)

                    w = torch.ones_like(b_target)
                    w = torch.where(b_target >= thr_hi_t, b_target.new_full((), RIGHT_W), w)
                    w = torch.where(b_target <= thr_lo_t, b_target.new_full((), LEFT_W),  w)

                    loss_rank = criterion_list(
                        log_pred.view(-1, list_size),
                        log_target.view(-1, list_size),
                    )

                    loss_point_log = (w * F.smooth_l1_loss(log_pred, log_target, reduction="none")).mean()
                    loss_point_lin = (w * F.smooth_l1_loss(preds, b_target, reduction="none")).mean()

                    w_rank = float(params.get("W_RANK", 0.3))
                    w_lin  = float(params.get("W_LIN",  1.0))
                    w_log  = float(params.get("W_LOG",  0.2))

                    loss = (w_rank * loss_rank) + (w_log * loss_point_log) + (w_lin * loss_point_lin)

                    total_loss += float(loss.item())
                    num_batches += 1
                    loss_sum = loss if loss_sum is None else (loss_sum + loss)

                if loss_sum is not None:
                    (loss_sum / float(max(1, num_batches))).backward()
                optimizer.step()

                del full_seq, full_raw, seq_emb, raw_emb
                if (epoch + 1) % 10 == 0:
                    mlflow.log_metric("train_loss", total_loss / max(1, num_batches), step=epoch + 1)

        else:
            print("[InferOnly] training skipped")

        # -----------------------------
        # save ckpt (train mode only)
        # -----------------------------
        if mode_run != "infer":
            try:
                os.makedirs(os.path.join("checkpoints", run_name), exist_ok=True)

                ckpt_local = os.path.join("checkpoints", run_name, "model_state.pt")
                ckpt_local, cfg_local = save_model_checkpoint(model, params, feature_dim=feature_dim, out_path=ckpt_local)

                ckpt_pth = os.path.splitext(ckpt_local)[0] + ".pth"
                try:
                    shutil.copy2(ckpt_local, ckpt_pth)
                except Exception:
                    torch.save(torch.load(ckpt_local, map_location="cpu"), ckpt_pth)

                try:
                    mlflow.log_artifact(ckpt_local, artifact_path="model")
                    mlflow.log_artifact(ckpt_pth, artifact_path="model")
                    mlflow.log_artifact(cfg_local, artifact_path="model")

                    info_txt = os.path.join("checkpoints", run_name, "checkpoint_info.txt")
                    with open(info_txt, "w", encoding="utf-8") as f:
                        f.write(f"run_name={run_name}\n")
                        f.write(f"ckpt_local={ckpt_local}\n")
                        f.write(f"ckpt_pth={ckpt_pth}\n")
                        f.write("mlflow_artifacts:\n")
                        f.write("  - model/model_state.pt\n")
                        f.write("  - model/model_state.pth\n")
                        f.write("  - model/model_config.json\n")
                    mlflow.log_artifact(info_txt, artifact_path="model")
                except Exception as e:
                    print(f"⚠️ [Checkpoint] MLflow log failed: {e}")

                print(f"[Checkpoint] local saved: {ckpt_local} (+ {ckpt_pth}) and {cfg_local}")
            except Exception as e:
                print(f"⚠️ [Checkpoint] save/log failed: {e}")

        # -----------------------------
        # inference
        # -----------------------------
        print("\nStarting Inference...")
        model.eval()

        test_dataset = get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-1)
        all_indices = test_dataset.tensors[0]
        all_targets = test_dataset.tensors[1]
        all_baselines = test_dataset.tensors[2]

        inf_input_graphs = monthly_graphs[:-1]  # Jan..Nov
        with torch.no_grad():
            seq_emb_l, raw_emb_l = [], []
            for g in inf_input_graphs:
                g = g.to(device)
                p_x = model.projection_layer(g.x)
                gcn_out = model.gcn_encoder(p_x, g.edge_index)
                raw_emb_l.append(p_x.cpu())
                seq_emb_l.append(gcn_out.cpu())

            # embeddings aligned to influencer_indices order
            f_seq = torch.stack(seq_emb_l)[:, influencer_indices].permute(1, 0, 2).contiguous()
            f_raw = torch.stack(raw_emb_l)[:, influencer_indices].permute(1, 0, 2).contiguous()

            preds_all = []
            attn_all = []
            infer_bs = int(params.get("INFER_BS", 1024))

            for i in range(0, len(all_indices), infer_bs):
                end = min(i + infer_bs, len(all_indices))
                b_seq = f_seq[i:end].to(device)
                b_raw = f_raw[i:end].to(device)
                b_base = all_baselines[i:end].to(device)

                p, attn = model(b_seq, b_raw, b_base)
                preds_all.append(p.detach().cpu())
                attn_all.append(attn.detach().cpu())

            predicted_scores = torch.cat(preds_all).squeeze().cpu().numpy()
            attention_matrix = torch.cat(attn_all).squeeze().cpu().numpy()

            true_scores = all_targets.cpu().numpy()
            baseline_scores = all_baselines.cpu().numpy()
            all_indices_np = all_indices.detach().cpu().numpy().astype(np.int64)

        # -----------------------------
        # consistent nonzero filter
        # -----------------------------
        nonzero_indices = np.where(true_scores > 0.0)[0]

        true_scores_nz      = true_scores[nonzero_indices]
        predicted_scores_nz = predicted_scores[nonzero_indices]
        baseline_scores_nz  = baseline_scores[nonzero_indices]
        all_indices_nz      = all_indices_np[nonzero_indices]
        attention_matrix_nz = attention_matrix[nonzero_indices]

        f_seq_nz = f_seq[nonzero_indices].contiguous()
        f_raw_nz = f_raw[nonzero_indices].contiguous()
        base_nz_t = torch.tensor(baseline_scores_nz, dtype=torch.float32)

        # -----------------------------
        # prediction CSV
        # -----------------------------
        usernames_nz = [idx_to_node.get(int(nid), str(int(nid))) for nid in all_indices_nz]
        df_pred_csv = pd.DataFrame({
            "node_id": all_indices_nz,
            "username": usernames_nz,
            "true_score": true_scores_nz,
            "pred_score": predicted_scores_nz,
            "baseline_score": baseline_scores_nz,
        })
        pred_csv_path = os.path.join("predictions", f"pred_dec2017_{run_name}.csv")
        os.makedirs(os.path.dirname(pred_csv_path), exist_ok=True)
        df_pred_csv.to_csv(pred_csv_path, index=False, encoding="utf-8-sig")
        mlflow.log_artifact(pred_csv_path, artifact_path="predictions")
        print("[write]", pred_csv_path, "rows=", len(df_pred_csv))

        # -----------------------------
        # eval metrics
        # -----------------------------
        extra = compute_eval_metrics(
            y_true=true_scores_nz,
            y_pred=predicted_scores_nz,
            baseline=baseline_scores_nz,
            ks=(10, 50, 100),
            prefix="test_dec2017",
        )
        mlflow.log_metrics(extra)
        final_test_metrics = dict(extra)

        df_pred = pd.DataFrame({"true_score": true_scores_nz, "pred_score": predicted_scores_nz})
        ndcgs = compute_two_ndcgs(df_pred, k_list=(1, 10, 50, 100, 200))
        for k, v in ndcgs.items():
            mlflow.log_metric(k, v)

        mae = mean_absolute_error(true_scores_nz, predicted_scores_nz)
        rmse = np.sqrt(mean_squared_error(true_scores_nz, predicted_scores_nz))
        p_corr, _ = pearsonr(true_scores_nz, predicted_scores_nz)
        s_corr, _ = spearmanr(true_scores_nz, predicted_scores_nz)

        mlflow.log_metrics({
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson_corr": float(p_corr),
            "spearman_corr": float(s_corr),
        })
        final_test_metrics.update({
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson_corr": float(p_corr),
            "spearman_corr": float(s_corr),
        })

        # -----------------------------
        # general plots
        # -----------------------------
        plot_files = []
        plot_files.append(generate_enhanced_scatter_plot(
            true_scores_nz, predicted_scores_nz,
            "True Engagement Score", "Predicted Score",
            run_name, "score_basic"
        ))

        # attention summaries (all nodes)
        att_bar_file, att_heat_file, att_csv_file, att_raw_file = plot_attention_weights(attention_matrix_nz, run_name)
        for f in [att_bar_file, att_heat_file, att_csv_file, att_raw_file]:
            if f is None:
                continue
            mlflow.log_artifact(f)
            os.remove(f)

        # follower colored plots
        last_input_graph = inf_input_graphs[-1]
        follower_counts = last_input_graph.x[all_indices, follower_feat_idx].detach().cpu().numpy()
        follower_counts_nz = follower_counts[nonzero_indices]
        log_follower_counts = follower_counts_nz

        eps = 1e-9
        true_growth = (true_scores_nz - baseline_scores_nz) / (baseline_scores_nz + eps)
        pred_growth = (predicted_scores_nz - baseline_scores_nz) / (baseline_scores_nz + eps)

        plot_files.append(generate_enhanced_scatter_plot(
            true_scores_nz, predicted_scores_nz,
            "True Engagement Score", "Predicted Score",
            run_name, "score_by_followers",
            color_data=log_follower_counts,
            color_label="log1p(Followers)",
            title_suffix="(Colored by Followers)"
        ))

        plot_files.append(generate_enhanced_scatter_plot(
            true_scores_nz, predicted_scores_nz,
            "True Engagement Score", "Predicted Score",
            run_name, "score_by_growth",
            color_data=true_growth,
            color_label="True Growth Rate",
            title_suffix="(Colored by Growth Rate)"
        ))

        plot_files.append(generate_enhanced_scatter_plot(
            true_growth, pred_growth,
            "True Growth Rate", "Predicted Growth Rate",
            run_name, "growth_by_followers",
            color_data=log_follower_counts,
            color_label="log1p(Followers)",
            title_suffix="(Colored by Followers)"
        ))

        for f in plot_files:
            if f is not None and os.path.exists(f):
                mlflow.log_artifact(f)
                os.remove(f)

        # -----------------------------
        # select target node (manual or hub)
        # -----------------------------
        feature_names = static_cols + dynamic_cols
        target_graph = monthly_graphs[-2]  # Nov
        edge_index = target_graph.edge_index.to(device)
        degs = degree(edge_index[1], num_nodes=target_graph.num_nodes)

        if target_node_idx is not None:
            target_node_global_idx = int(target_node_idx)
        else:
            # choose hub among influencers
            best = -1
            target_node_global_idx = int(influencer_indices[0])
            for idx in influencer_indices:
                di = int(degs[int(idx)].item())
                if di > best:
                    best = di
                    target_node_global_idx = int(idx)

        mlflow.log_param("xai_target_node", int(target_node_global_idx))
        tname = idx_to_node.get(int(target_node_global_idx), None)
        if tname is not None:
            mlflow.log_param("xai_target_name", tname)

        # nz-row for the selected target
        pos = np.where(all_indices_nz == int(target_node_global_idx))[0]
        if len(pos) == 0:
            print("⚠️ target user not in nonzero-filtered set (true_score==0 etc.)")
            row_nz = None
        else:
            row_nz = int(pos[0])

        # baseline/pred for selected user
        if row_nz is not None:
            base_user = float(baseline_scores_nz[row_nz])
            pred_user = float(predicted_scores_nz[row_nz])
        else:
            base_user = float(np.mean(baseline_scores_nz))
            pred_user = float(np.mean(predicted_scores_nz))

        mlflow.log_metric("xai_target_pred", float(pred_user))
        mlflow.log_metric("xai_target_base", float(base_user))

        # -----------------------------
        # selected-user attention plot (nz-aligned)
        # -----------------------------
        if row_nz is not None:
            try:
                attn_user = np.squeeze(attention_matrix_nz[row_nz])
                plt.figure()
                plt.plot(range(len(attn_user)), attn_user, marker="o")
                plt.title(f"Attention weights user={idx_to_node.get(int(target_node_global_idx), str(int(target_node_global_idx)))}")
                plt.xlabel("pos (0=oldest -> T-1=newest)")
                plt.ylabel("attention weight")
                plt.tight_layout()
                fig_path = f"attention_user_{int(target_node_global_idx)}_{run_name}.png"
                plt.savefig(fig_path, dpi=220)
                plt.close()
                mlflow.log_artifact(fig_path, artifact_path="xai/attention_user")
                os.remove(fig_path)
            except Exception as e:
                print("⚠️ attention_user plot failed:", e)

        # -----------------------------
        # path dependency diagnostics (full / raw-only / seq-only)
        # -----------------------------
        if row_nz is not None:
            try:
                f_seq1 = f_seq_nz[row_nz:row_nz+1]
                f_raw1 = f_raw_nz[row_nz:row_nz+1]
                base1 = torch.tensor([base_user], dtype=torch.float32)

                diag = diagnose_path_dependency(model, f_seq1, f_raw1, base1, device=device)
                mlflow.log_metrics({
                    "path_pred_full": float(diag["full"]),
                    "path_pred_raw_only": float(diag["raw_only"]),
                    "path_pred_seq_only": float(diag["seq_only"]),
                })
            except Exception as e:
                print("⚠️ path-diagnose failed:", e)

        # -----------------------------
        # pos-importance baseline replacement (global + userwise)
        # -----------------------------
        df_pos_long, deltas_np = compute_pos_importance_from_embeddings(
            model=model,
            f_seq=f_seq_nz,
            f_raw=f_raw_nz,
            baseline_scores=base_nz_t,
            device=device,
            baseline_modes=baseline_modes,
            batch_size=int(params.get("POS_IMP_BS", 1024)),
            seed=int(params.get("SEED", 0)),
        )

        pos_csv = f"pos_importance_baselines_{run_name}.csv"
        df_pos_long.to_csv(pos_csv, index=False, float_format="%.8e")
        mlflow.log_artifact(pos_csv, artifact_path="xai/pos_importance")
        os.remove(pos_csv)

        # median plot per baseline
        try:
            for mode in baseline_modes:
                ddf = df_pos_long[df_pos_long["baseline_mode"] == mode].sort_values("pos")
                plt.figure()
                plt.plot(ddf["pos"].values, ddf["delta_median"].values, marker="o")
                plt.xlabel("pos (0=oldest -> T-1=newest)")
                plt.ylabel("median(pred_orig - pred_replaced)")
                plt.title(f"Pos importance (median) / baseline={mode}")
                plt.tight_layout()
                fig_path = f"pos_importance_median_{mode}_{run_name}.png"
                plt.savefig(fig_path, dpi=220)
                plt.close()
                mlflow.log_artifact(fig_path, artifact_path="xai/pos_importance")
                os.remove(fig_path)
        except Exception as e:
            print("⚠️ pos-importance plot failed:", e)

        # user-wise export (NPZ + CSV long)
        try:
            T_steps = int(f_seq_nz.shape[1])
            npz_path = f"userwise_pos_delta_{run_name}.npz"
            np.savez_compressed(
                npz_path,
                node_id=all_indices_nz.astype(np.int64),
                username=np.array(usernames_nz, dtype=object),
                **{f"delta_{mode}": deltas_np[mode].astype(np.float32) for mode in baseline_modes}
            )
            mlflow.log_artifact(npz_path, artifact_path="xai/pos_importance_userwise")
            os.remove(npz_path)

            rows = []
            for mode in baseline_modes:
                dmat = deltas_np[mode]  # [N,T]
                for pos_i in range(T_steps):
                    rows.append(pd.DataFrame({
                        "node_id": all_indices_nz,
                        "username": usernames_nz,
                        "pos": pos_i,
                        "baseline_mode": mode,
                        "delta": dmat[:, pos_i],
                    }))
            df_user_pos = pd.concat(rows, ignore_index=True)

            user_csv = f"userwise_pos_delta_{run_name}.csv"
            df_user_pos.to_csv(user_csv, index=False, float_format="%.8e", encoding="utf-8-sig")
            mlflow.log_artifact(user_csv, artifact_path="xai/pos_importance_userwise")
            os.remove(user_csv)
        except Exception as e:
            print("⚠️ user-wise pos delta export failed:", e)

        # selected-user plot (pos delta vs pos) for each baseline mode
        if row_nz is not None:
            try:
                plt.figure()
                for mode in baseline_modes:
                    series = deltas_np[mode][row_nz]
                    plt.plot(np.arange(len(series)), series, marker="o", label=mode)
                plt.xlabel("pos (0=oldest -> T-1=newest)")
                plt.ylabel("delta = pred_orig - pred_replaced")
                plt.title(f"User-wise pos delta (node={idx_to_node.get(int(target_node_global_idx), str(int(target_node_global_idx)))})")
                plt.legend()
                plt.tight_layout()
                fig_path = f"user_pos_delta_{int(target_node_global_idx)}_{run_name}.png"
                plt.savefig(fig_path, dpi=220)
                plt.close()
                mlflow.log_artifact(fig_path, artifact_path="xai/pos_importance_userwise")
                os.remove(fig_path)
            except Exception as e:
                print("⚠️ selected-user pos delta plot failed:", e)

        # diagnostics: feature variance over time (selected user)
        if row_nz is not None:
            try:
                input_graphs = monthly_graphs[:-1]
                mat = np.stack([g.x[int(target_node_global_idx)].detach().cpu().numpy() for g in input_graphs], axis=0)
                df_diag = pd.DataFrame({
                    "feature": feature_names,
                    "std_over_T": mat.std(axis=0),
                    "nonzero_T": (mat != 0).sum(axis=0),
                    "min": mat.min(axis=0),
                    "max": mat.max(axis=0),
                }).sort_values(["std_over_T", "nonzero_T"], ascending=False)

                diag_csv = f"diag_feature_over_time_node_{int(target_node_global_idx)}_{run_name}.csv"
                df_diag.to_csv(diag_csv, index=False, float_format="%.8e")
                mlflow.log_artifact(diag_csv, artifact_path="xai/diagnostics")
                os.remove(diag_csv)
            except Exception as e:
                print("⚠️ diag feature-over-time failed:", e)

        # -----------------------------
        # MaskOpt + (optional) DI curves using MaskOpt ranking
        # -----------------------------
        sens_df = None
        sens_selected = None

        try:
            input_graphs = monthly_graphs[:-1]  # Jan..Nov
            T = len(input_graphs)

            # pick attention weights row for this node (nz-aligned)
            attn_w = None
            if row_nz is not None:
                attn_w = torch.tensor(np.squeeze(attention_matrix_nz[row_nz]), dtype=torch.float32)

            if params.get("explain_use_sensitivity", True):
                try:
                    sens_df, sens_selected, _pred_full, _alpha = compute_time_step_sensitivity(
                        model=model,
                        input_graphs=input_graphs,
                        target_node_idx=int(target_node_global_idx),
                        device=device,
                        topk=int(params.get("xai_topk_pos", 3)),
                        score_mode=str(params.get("sensitivity_score_mode", "alpha_x_delta")),
                        min_delta=float(params.get("sensitivity_min_delta", 1e-4)),
                    )
                except Exception:
                    sens_df, sens_selected = None, None

            if attn_w is None:
                positions_attn = list(range(min(3, T)))
            else:
                positions_attn = _select_positions_by_attention(
                    attn_w, T,
                    topk=int(params.get("xai_topk_pos", 3)),
                    min_w=float(params.get("xai_attn_min_w", 0.0)),
                )

            positions_to_explain = positions_attn
            if sens_selected is not None and len(sens_selected) > 0:
                positions_to_explain = sens_selected[: int(params.get("xai_topk_pos", 3))]

            mlflow.log_param("xai_positions", ",".join(map(str, positions_to_explain)))

            # export attention alpha + sensitivity table
            if attn_w is not None:
                labels = [f"T-{T-1-i}" for i in range(T)]
                df_alpha = pd.DataFrame({
                    "pos": list(range(T)),
                    "label": labels,
                    "alpha": attn_w.detach().cpu().numpy().astype(float),
                    "selected_for_explain": [int(i in positions_to_explain) for i in range(T)],
                })
                if sens_df is not None and (not sens_df.empty):
                    df_alpha = df_alpha.merge(sens_df, on="pos", how="left")

                alpha_csv = f"xai_attention_alpha_node_{int(target_node_global_idx)}_{run_name}.csv"
                df_alpha.to_csv(alpha_csv, index=False, float_format="%.8e")
                mlflow.log_artifact(alpha_csv, artifact_path="xai")
                os.remove(alpha_csv)

            # run MaskOpt for selected positions
            for explain_pos in positions_to_explain:
                tag = f"pos_{int(explain_pos)}"
                df_feat, df_edge, meta = maskopt_e2e_explain(
                    model=model,
                    input_graphs=input_graphs,
                    target_node_idx=int(target_node_global_idx),
                    explain_pos=int(explain_pos),
                    feature_names=feature_names,
                    node_to_idx=node_to_idx,
                    device=device,
                    use_subgraph=True,
                    num_hops=1,
                    edge_mask_scope="incident",
                    edge_grouping="neighbor",
                    fid_weight=float(params.get("maskopt_fid_weight", 2000.0)),
                    coeffs=params.get("maskopt_coeffs", {"edge_size":0.08,"edge_ent":0.15,"node_feat_size":0.02,"node_feat_ent":0.15}),
                    budget_feat=int(params.get("maskopt_budget_feat", 10)),
                    budget_edge=int(params.get("maskopt_budget_edge", 20)),
                    budget_weight=float(params.get("maskopt_budget_weight", 1.0)),
                    impact_reference=str(params.get("maskopt_impact_reference", "masked")),
                    use_contrastive=bool(params.get("maskopt_use_contrastive", False)),
                    mlflow_log=True,
                    tag=tag,
                )

                # quick paper-friendly plots
                mlflow_log_maskopt_plots(
                    df_feat=df_feat,
                    df_edge=df_edge,
                    meta=meta,
                    tag=tag,
                    topk_feat=int(params.get("maskopt_plot_topk_feat", 50)),
                    topk_edge=int(params.get("maskopt_plot_topk_edge", 50)),
                    artifact_path="xai",
                    fname_prefix=f"node_{int(target_node_global_idx)}",
                )
        

                        # =============================
                
                # ICE / PDP: important features
                # =============================
                try:
                    # --- user set for ICE ---
                    n_users_ice = int(params.get("ICE_NUM_USERS", 60))
                    ice_rows = _select_users_for_ice(
                        all_indices_nz=all_indices_nz,
                        predicted_scores_nz=predicted_scores_nz,
                        n_users=n_users_ice,
                        include_node_id=int(target_node_global_idx),
                    )
                    user_ids_ice = all_indices_nz[ice_rows].astype(np.int64)
                    usernames_ice = [usernames_nz[r] for r in ice_rows]
                    base_ice = baseline_scores_nz[ice_rows].astype(np.float32)

                    # --- choose features: top-k from df_feat ---
                    topk_feat = int(params.get("ICE_TOPK_FEATURES", 3))
                    n_grid = int(params.get("ICE_GRID_POINTS", 25))
                    qlo = float(params.get("ICE_QLO", 0.05))
                    qhi = float(params.get("ICE_QHI", 0.95))

                    feat_idx_col = None
                    for c in ["feature_idx", "feat_idx", "feature_index", "idx"]:
                        if df_feat is not None and c in df_feat.columns:
                            feat_idx_col = c
                            break
                    imp_col = None
                    for c in ["importance", "score_impact", "impact", "weight", "mask", "Importance"]:
                        if df_feat is not None and c in df_feat.columns:
                            imp_col = c
                            break

                    chosen_feat_idxs = []
                    if df_feat is not None and (not df_feat.empty) and feat_idx_col is not None:
                        if imp_col is not None:
                            chosen_feat_idxs = (
                                df_feat.sort_values(imp_col, ascending=False)[feat_idx_col]
                                .astype(int).tolist()
                            )
                        else:
                            chosen_feat_idxs = df_feat[feat_idx_col].astype(int).tolist()

                    chosen_feat_idxs = [int(i) for i in chosen_feat_idxs if 0 <= int(i) < len(feature_names)]
                    chosen_feat_idxs = list(dict.fromkeys(chosen_feat_idxs))[:topk_feat]

                    # pos は explain_pos を使う（重要とされた月での反応を見る）
                    pos_for_ice = int(explain_pos)

                    for fidx in chosen_feat_idxs:
                        run_ice_pdp_for_feature(
                            model=model,
                            input_graphs=input_graphs,
                            pos=pos_for_ice,
                            feature_idx=int(fidx),
                            user_ids_np=user_ids_ice,
                            usernames=usernames_ice,
                            baseline_scores_np=base_ice,
                            device=device,
                            run_name=run_name,
                            tag=tag,
                            feature_name=str(feature_names[int(fidx)]),
                            n_grid=n_grid,
                            qlo=qlo,
                            qhi=qhi,
                            artifact_path="xai/ice",
                        )
                except Exception as e:
                    print("⚠️ ICE/PDP failed:", e)

        
        
        except Exception as e:
            print(f"💥 Explanation Error: {e}")

        
        # -----------------------------
        # always log eval scatter (your helper)
        # -----------------------------
        try:
            mlflow_log_pred_scatter(
                y_true=true_scores_nz,
                y_pred=predicted_scores_nz,
                tag="test_dec2017",
                step=params.get("EPOCHS", None),
                artifact_path="plots",
            )
        except Exception as e:
            print("⚠️ mlflow_log_pred_scatter failed:", e)

        # -----------------------------
        # cleanup
        # -----------------------------
        del model, optimizer, criterion_list, criterion_mse
        if "gpu_graphs" in locals():
            del gpu_graphs
        if "f_seq" in locals():
            del f_seq
        if "f_raw" in locals():
            del f_raw
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return run_id, final_test_metrics



# ===================== Main =====================
def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- CLI (train vs infer/XAI-only) ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "infer"], default=None,
                    help="train: train+infer+XAI / infer: load checkpoint and run infer+XAI only")
    ap.add_argument("--ckpt", type=str, default=os.environ.get("INFLUENCER_MODEL_CKPT"),
                    help="Path to checkpoint (.pt). Used in infer mode.")
    ap.add_argument("--mlflow_run_id", type=str, default=None,
                    help="If set, download checkpoint artifact from this MLflow run_id and use it.")
    ap.add_argument("--mlflow_ckpt_artifact", type=str, default="model/model_state.pt",
                    help="Artifact path under the run_id to download.")
    ap.add_argument("--xai_only", action="store_true",
                    help="Alias for infer mode.")
    ap.add_argument("--target_node", type=int, default=None,
                help="global node id to explain (if omitted, auto-select hub)")    
    ap.add_argument(
        "--influencer_name", type=str, default=None,
        help="Username to explain (e.g., london_theplug). You can also pass '@name'. "
            "If set, overrides auto hub selection. If both --target_node and this are set, --target_node wins."
    )



    args, _unknown = ap.parse_known_args()

    mode = args.mode
    if args.xai_only:
        mode = "infer"
    if mode is None:
        mode = "infer" if (args.ckpt is not None or args.mlflow_run_id is not None) else "train"

    experiment_name, experiment_id = setup_mlflow_experiment(
        experiment_base_name=os.environ.get('MLFLOW_EXPERIMENT_NAME', 'InfluencerRankSweep'),
        tracking_uri=os.environ.get('MLFLOW_TRACKING_URI'),
        local_artifact_dir=os.environ.get('MLFLOW_ARTIFACT_DIR', 'mlruns_artifacts'),
    )
    random.seed(seed)

    target_date = pd.to_datetime('2017-12-31')
    prep = prepare_graph_data(
        end_date=target_date,
        num_months=12,
        metric_numerator='likes_and_comments',
        metric_denominator='followers'
    )
    if prep[0] is None:
        # print("Data preparation failed.")
        return 1

    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prep
    feature_dim = monthly_graphs[0].x.shape[1]
    # print(f"Final feature dimension: {feature_dim}")
    # print(f"Follower feature index: {follower_feat_idx}")
    print(f"[Target] influencer_name arg = {args.influencer_name!r}")
    if args.influencer_name not in node_to_idx:
        raise ValueError(f"Unknown influencer_name: {args.influencer_name!r}. "
                        f"Example keys: {list(node_to_idx.keys())[:10]}")
    target_gid = int(node_to_idx[args.influencer_name]) if args.influencer_name is not None else None
    print(f"[Target] global_id = {target_gid}")


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
        'name_prefix': 'Run',
        'note': 'Default',
        'LR': 0.001,
        'POINTWISE_LOSS_WEIGHT': 0.01,
        'DROPOUT_PROB': 0.05,
        'GCN_DIM': 64,
        'RNN_DIM': 64,
        'NUM_GCN_LAYERS': 2,
        'PROJECTION_DIM': 128,
        'EPOCHS': 150,
        'LIST_SIZE': 20,
        'BATCH_SIZE': 20 * 64,
        'USE_SAMPLER': True,
        'MODE': mode,
        'CKPT_PATH': args.ckpt,
        'CKPT_MLFLOW_RUN_ID': args.mlflow_run_id,
        'CKPT_MLFLOW_ARTIFACT': args.mlflow_ckpt_artifact,
        # XAI control (reproducibility)
        "explain_use_sensitivity": True,
        "xai_topk_pos": 3,
        "sensitivity_score_mode": "alpha_x_delta",
        "sensitivity_min_delta": 1e-4,
        "xai_attn_min_w": 0.0,
        # sampler control
        "RIGHT_Q": 0.85,
        "RIGHT_W": 12.0,
        "LEFT_Q": 0.1,
        "LEFT_W": 3.0,
    }
    base_params.update({
        "ICE_NUM_USERS": 60,        # ICE の薄線本数（多いほど重い）
        "ICE_TOPK_FEATURES": 3,     # MaskOpt上位K特徴で ICE
        "ICE_GRID_POINTS": 25,      # x を動かす刻み
        "ICE_QLO": 0.05,            # グリッド範囲の下側（外れ値カット）
        "ICE_QHI": 0.95,            # 上側
    })


    params_list = [
        # {'LR': 0.001, 'DROPOUT_PROB': 0.05, 'POINTWISE_LOSS_WEIGHT': 0, 'GCN_DIM': 64, 'RNN_DIM': 64, 'EPOCHS': 150, 'LIST_SIZE': 20, 'BATCH_SIZE': 1280, 'USE_SAMPLER': False}
    ]

    sweep_grid = {
        # 'LR': [0.001, 0.003, 0.005, 0.0003],
        # 'DROPOUT_PROB': [0.05, 0.1, 0.2],
        # 'POINTWISE_LOSS_WEIGHT': [0.01, 0.1],
        # 'GCN_DIM': [64, 128],
        # 'RNN_DIM': [64, 128],
        # 'EPOCHS': [50, 100, 150, 200],
        'LIST_SIZE': [20],
        # 'BATCH_SIZE': [20 * 64, 50 * 64],
        # 'USE_SAMPLER': [True, False],
        # "RIGHT_Q": [0.85],
        # "RIGHT_W": [1, 12.0],
        # "LEFT_Q": [0.10, 0.15],
        # "LEFT_W": [1, 3.0],
        # "N_HIGH": [1, 3, 5],
        # "N_LOW": [0, 1],
    }

    def _expand_grid(grid_dict):
        keys = list(grid_dict.keys())
        if not keys:
            return [{}]
        values = [grid_dict[k] for k in keys]
        out = []
        for combo in itertools.product(*values):
            out.append({k: v for k, v in zip(keys, combo)})
        return out

    def _merge(base, overrides):
        p = dict(base)
        p.update(overrides)
        return p

    def _suffix(overrides):
        if not overrides:
            return "base"
        parts = []
        for k, v in overrides.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:g}")
            else:
                parts.append(f"{k}={v}")
        return ",".join(parts)

    if len(params_list) > 0:
        overrides_list = list(params_list)
    else:
        overrides_list = _expand_grid(sweep_grid)

    params_list = [_merge(base_params, ov) for ov in overrides_list]

    summary_rows = []
    for i, p in enumerate(params_list):
        p = dict(p)
        overrides_for_name = dict(overrides_list[i]) if i < len(overrides_list) else {}

        p['name_prefix'] = f"{base_params['name_prefix']}_{i:03d}"
        p['note'] = f"{base_params.get('note','')} | sweep={i+1}/{len(params_list)} | {_suffix(overrides_for_name)}"


        target_node_idx_resolved, target_name_resolved = resolve_target_node_idx(
            target_node=args.target_node,
            influencer_name=args.influencer_name,
            node_to_idx=node_to_idx,
            influencer_indices=influencer_indices,
        )

        if target_node_idx_resolved is not None:
            print(f"[Target] resolved target_node_idx={target_node_idx_resolved} "
                f"(name={target_name_resolved if target_name_resolved else 'N/A'})")


        run_id, metrics = run_experiment(
            p, graphs_data,
            target_node_idx=target_node_idx_resolved,
            experiment_id=experiment_id
        )


        row = {
            "run_index": i,
            "run_id": run_id,
            "note": p.get("note", ""),
        }
        for k in [
            "LR", "DROPOUT_PROB", "POINTWISE_LOSS_WEIGHT",
            "GCN_DIM", "RNN_DIM", "NUM_GCN_LAYERS", "PROJECTION_DIM",
            "EPOCHS", "LIST_SIZE", "BATCH_SIZE", "USE_SAMPLER",
        ]:
            row[k] = p.get(k)

        if isinstance(metrics, dict):
            for mk, mv in metrics.items():
                row[mk] = mv

        summary_rows.append(row)

    if len(summary_rows) > 1:
        df_sum = pd.DataFrame(summary_rows)
        sum_csv = f"sweep_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_sum.to_csv(sum_csv, index=False)
        # print(f"\n📌 Sweep summary saved: {sum_csv}")

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

    # print("\n🎉 Done. Run 'mlflow ui' to view results.")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())