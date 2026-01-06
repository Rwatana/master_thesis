#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InfluencerRank reproduction + training/inference + XAI (MaskOpt E2E) — FULL SCRIPT
(no external image CSV + Mac-friendly device selection)

Changes vs your previous script:
  - Removes external image CSV input (no IMAGE_DATA_FILE). Image features are taken from PREPROCESSED_FILE if present;
    otherwise they are created as zeros. Object edges are built only if posts already contain an object column.
  - Device selection supports: --device auto|mps|cpu|<cuda_index>. (Mac: use mps/cpu)
  - Removes seaborn dependency; heatmap uses matplotlib.
  - Fixes broken MaskOpt call sites (no hard-coded node/pos, no undefined 'graphs' variable).
  - Guards MLflow plot logging so it won't crash if explanation fails.

Usage examples:
  # Mac (stable)
  python influencer_rank_full_noimage_mac.py --device cpu

  # Mac (try MPS; if PyG/ops fail, fall back to cpu)
  python influencer_rank_full_noimage_mac.py --device mps

  # CUDA box
  python influencer_rank_full_noimage_mac.py --device 0 --visible 0
"""

import os
import sys
import argparse
import time
import datetime
import random
import gc
import itertools
import warnings

import pandas as pd
import numpy as np
from tqdm import tqdm

# --------- CLI (CUDA_VISIBLE_DEVICES must be set before torch import, but on Mac it's harmless) ---------
def _parse_pre_torch_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto|cpu|mps|<cuda_index>. Example: --device mps, --device cpu, --device 0",
    )
    p.add_argument(
        "--visible",
        type=str,
        default=None,
        help="(CUDA only) set CUDA_VISIBLE_DEVICES before importing torch.",
    )
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# headless safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow

from mlflow.tracking import MlflowClient

def _sanitize_mlflow_name(name: str) -> str:
    """Make a string safe for mlflow metric/param names."""
    return re.sub(r"[^0-9A-Za-z_\-\. :/]+", "_", name)

def _mlflow_list_artifacts_recursive(client: MlflowClient, run_id: str, base_path: str = ""):
    out = []
    stack = [base_path]
    while stack:
        p = stack.pop()
        for f in client.list_artifacts(run_id, p):
            if f.is_dir:
                stack.append(f.path)
            else:
                out.append(f.path)
    return out


# -----------------------------
# Name helpers / caching
# -----------------------------
_IDX_TO_NAME_CACHE = {}

def build_idx_to_name(node_to_idx: dict) -> list:
    """Build (and cache) a list mapping node_idx -> node_name."""
    key = id(node_to_idx)
    cached = _IDX_TO_NAME_CACHE.get(key)
    if cached is not None:
        return cached
    if not node_to_idx:
        _IDX_TO_NAME_CACHE[key] = []
        return _IDX_TO_NAME_CACHE[key]
    max_idx = int(max(node_to_idx.values()))
    idx_to_name = ["<UNK>"] * (max_idx + 1)
    for name, idx in node_to_idx.items():
        try:
            ii = int(idx)
        except Exception:
            continue
        if 0 <= ii < len(idx_to_name):
            idx_to_name[ii] = str(name)
    _IDX_TO_NAME_CACHE[key] = idx_to_name
    return idx_to_name

def node_name(node_to_idx: dict, node_idx: int) -> str:
    """Safe node name lookup from (name->idx) mapping."""
    try:
        ii = int(node_idx)
    except Exception:
        return str(node_idx)
    idx_to_name = build_idx_to_name(node_to_idx)
    if 0 <= ii < len(idx_to_name):
        return idx_to_name[ii]
    return f"<idx:{ii}>"

def sanitize_mlflow_name(name: str) -> str:
    """MLflow 'name' for artifacts/metrics allows limited chars; replace others."""
    if name is None:
        return ""
    # allowed: alphanumerics, underscores, dashes, periods, spaces, colon, slashes
    return re.sub(r"[^0-9A-Za-z_\-\. :/]+", "_", str(name))


def _apply_cli_overrides_to_env(argv):
    """Parse common CLI options and apply them to env vars so the rest of the script can use os.getenv()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mlflow_tracking_uri", default=None)
    parser.add_argument("--mlflow_experiment_name", default=None)
    parser.add_argument("--mlflow_artifact_dir", default=None)
    parser.add_argument("--mlflow_model_run_id", default=None)
    parser.add_argument("--model_pth_path", default=None)
    parser.add_argument("--xai_only", action="store_true")
    parser.add_argument("--inference_only", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--plot_scale_0_1", action="store_true")
    args, _ = parser.parse_known_args(argv)

    if args.device:
        os.environ["DEVICE"] = str(args.device)
    if args.mlflow_tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = str(args.mlflow_tracking_uri)
    if args.mlflow_experiment_name:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = str(args.mlflow_experiment_name)
    if args.mlflow_artifact_dir:
        os.environ["MLFLOW_ARTIFACT_DIR"] = str(args.mlflow_artifact_dir)
    if args.mlflow_model_run_id:
        os.environ["MLFLOW_MODEL_RUN_ID"] = str(args.mlflow_model_run_id)
    if args.model_pth_path:
        os.environ["MODEL_PTH_PATH"] = str(args.model_pth_path)

    if args.xai_only:
        os.environ["XAI_ONLY"] = "1"
    if args.inference_only:
        os.environ["INFERENCE_ONLY"] = "1"
    if args.skip_training:
        os.environ["SKIP_TRAINING"] = "1"
    if args.plot_scale_0_1:
        os.environ["PLOT_SCALE_0_1"] = "1"

    return args

def mlflow_download_model_artifact(
    tracking_uri: str,
    experiment_name: str,
    artifact_basename: str = "model_state_dict.pth",
    preferred_subdir: str = "model",
    run_id: str | None = None,
    dst_dir: str = "downloaded_models",
):
    """Return (run_id, local_path) if found, else (None, None)."""
    client = MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None, None

    def find_in_run(rid: str):
        paths = _mlflow_list_artifacts_recursive(client, rid, "")
        preferred = f"{preferred_subdir}/{artifact_basename}".lstrip("/")
        if preferred in paths:
            return preferred
        for p in paths:
            if os.path.basename(p) == artifact_basename:
                return p
        for cand in [f"{preferred_subdir}/model.pth", "model.pth", f"{preferred_subdir}/model_state.pth"]:
            if cand in paths:
                return cand
        return None

    if run_id is not None:
        found = find_in_run(run_id)
        if found is None:
            return None, None
        local = client.download_artifacts(run_id, found, dst_dir)
        return run_id, local

    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=500)
    for r in runs:
        rid = r.info.run_id
        found = find_in_run(rid)
        if found is None:
            continue
        local = client.download_artifacts(rid, found, dst_dir)
        return rid, local

    return None, None

import mlflow.pytorch

# pandas option (older pandas may not have it)
try:
    pd.set_option("future.no_silent_downcasting", True)
except Exception:
    pass

# --------- MLflow setup (robust for local file store vs. server) ---------
def setup_mlflow_experiment(
    experiment_base_name="InfluencerRankSweep",
    tracking_uri=None,
    local_artifact_dir="mlruns_artifacts",
):
    """
    Local-first MLflow setup.

    If an experiment was created under an MLflow server with `--serve-artifacts`,
    its artifact_location can be `mlflow-artifacts:/...`. When you later switch to a
    file-based tracking URI, MLflow can't resolve that scheme.

    This helper:
      1) Sets tracking URI (env MLFLOW_TRACKING_URI respected unless overridden)
      2) Ensures local tracking uses file:// artifact root (creates a new experiment suffix if needed)
      3) Returns (experiment_name, experiment_id)
    """
    import datetime as _dt
    from pathlib import Path

    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    active_tracking_uri = mlflow.get_tracking_uri()
    is_remote = active_tracking_uri.startswith("http://") or active_tracking_uri.startswith("https://")

    base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_base_name)
    exp_name = base_name

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    cwd = Path.cwd()
    artifact_dir = (cwd / local_artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def _get_exp(name):
        try:
            return mlflow.get_experiment_by_name(name)
        except Exception:
            return None

    exp = _get_exp(exp_name)

    if (not is_remote) and (exp is not None) and str(exp.artifact_location).startswith("mlflow-artifacts:"):
        exp_name = f"{base_name}_file_{ts}"
        exp = None

    if exp is None:
        try:
            if is_remote:
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
    if not is_remote:
        print(f"[MLflow] artifact_root={artifact_dir.as_uri()}")

    return exp_name, exp_id


# --------- Device (Mac-friendly) ---------
def get_device(requested: str):
    req = str(requested).lower().strip()

    if req in ("cpu",):
        print("[Device] Using CPU")
        return torch.device("cpu")

    if req in ("mps",):
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            print("[Device] Using MPS")
            return torch.device("mps")
        print("[Device] MPS requested but not available -> CPU")
        return torch.device("cpu")

    if req in ("auto", ""):
        if torch.cuda.is_available():
            print("[Device] Using CUDA (auto)")
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            print("[Device] Using MPS (auto)")
            return torch.device("mps")
        print("[Device] Using CPU (auto)")
        return torch.device("cpu")

    if req.isdigit():
        idx = int(req)
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if idx >= n:
                print(f"[Device] requested cuda:{idx} but only 0..{n-1} -> cuda:0")
                idx = 0
            torch.cuda.set_device(idx)
            print(f"[Device] Using CUDA cuda:{idx} ({torch.cuda.get_device_name(idx)})")
            return torch.device(f"cuda:{idx}")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            print("[Device] CUDA index given but CUDA unavailable -> MPS")
            return torch.device("mps")
        print("[Device] CUDA index given but CUDA unavailable -> CPU")
        return torch.device("cpu")

    print(f"[Device] Unknown device='{requested}' -> auto")
    return get_device("auto")


device = get_device(PRE_ARGS.device)
print("[Device] Using:", device)

# --------- Files ---------
PREPROCESSED_FILE = "dataset_A_active_all.csv"
HASHTAGS_FILE = "hashtags_2017.csv"
MENTIONS_FILE = "mentions_2017.csv"
INFLUENCERS_FILE = "influencers.txt"

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
        return [keep[i] for i in order[: min(topk, len(keep))]]
    return torch.topk(attn_w, k=min(topk, T)).indices.tolist()


def _binary_entropy(p, eps=1e-12):
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


# ===================== Data Loading / Graph Building =====================
def load_influencer_profiles():
    """Read influencers.txt (tab-separated)."""
    print(f"Loading influencer profiles from {INFLUENCERS_FILE}...")
    try:
        df_inf = pd.read_csv(INFLUENCERS_FILE, sep="\t", dtype=str)
        rename_map = {
            "Username": "username",
            "Category": "category",
            "#Followers": "followers",
            "#Followees": "followees",
            "#Posts": "posts_history",
        }
        df_inf.rename(columns=rename_map, inplace=True)
        required_cols = ["username", "category", "followers", "followees", "posts_history"]
        for c in required_cols:
            if c not in df_inf.columns:
                df_inf[c] = "0"
        df_inf = df_inf[required_cols].copy()
        df_inf["username"] = df_inf["username"].astype(str).str.strip()
        for c in ["followers", "followees", "posts_history"]:
            df_inf[c] = pd.to_numeric(df_inf[c], errors="coerce").fillna(0)
        print(f"Loaded {len(df_inf)} influencer profiles.")
        return df_inf
    except Exception as e:
        print(f"Error loading influencers.txt: {e}")
        return pd.DataFrame(columns=["username", "followers", "followees", "posts_history", "category"])


def prepare_graph_data(end_date, num_months=12, metric_numerator="likes", metric_denominator="posts"):
    """
    Build graph sequence for each month.
    Returns:
      monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")

    # --- 1. Load Post Data ---
    try:
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=["datetime"], low_memory=False, dtype={"post_id": str})
        print(f"Loaded {len(df_posts)} posts from {PREPROCESSED_FILE}")
        df_posts["username"] = df_posts["username"].astype(str).str.strip()

        target_month_start = pd.Timestamp("2017-12-01")
        target_month_end = pd.Timestamp("2017-12-31 23:59:59")
        dec_posts = df_posts[(df_posts["datetime"] >= target_month_start) & (df_posts["datetime"] <= target_month_end)]
        valid_users_dec = set(dec_posts["username"].unique())
        print(f"Users who posted in Dec 2017: {len(valid_users_dec):,}")
        if len(valid_users_dec) == 0:
            print("Warning: No users found who posted in Dec 2017. Check date range.")
            return None, None, None, None, None, None

        original_count = len(df_posts)
        df_posts = df_posts[df_posts["username"].isin(valid_users_dec)].copy()
        print(f"Filtered posts dataset: {original_count:,} -> {len(df_posts):,} rows")

        col_map = {
            "like_count": "likes",
            "comment_count": "comments",
            "hashtag_count": "tag_count",
            "user_followers": "followers_dynamic",
            "user_following": "followees_dynamic",
            "user_media_count": "posts_count_dynamic",
            "caption_len": "caption_length",
            "sentiment_pos": "caption_sent_pos",
            "sentiment_neg": "caption_sent_neg",
            "sentiment_neu": "caption_sent_neu",
            "sentiment_compound": "caption_sent_compound",
            "comment_sentiment_pos": "comment_avg_pos",
            "comment_sentiment_neg": "comment_avg_neg",
            "comment_sentiment_neu": "comment_avg_neu",
            "comment_sentiment_compound": "comment_avg_compound",
        }
        df_posts.rename(columns=col_map, inplace=True)

        for col in ["comments", "feedback_rate", "likes"]:
            if col not in df_posts.columns:
                df_posts[col] = 0
            else:
                df_posts[col] = df_posts[col].fillna(0)

    except FileNotFoundError:
        print(f"Error: '{PREPROCESSED_FILE}' not found.")
        return None, None, None, None, None, None

    # --- (NEW) No external image CSV. Use columns from df_posts if present, else zeros. ---
    if "color_temp" in df_posts.columns and "color_temp_proxy" not in df_posts.columns:
        df_posts.rename(columns={"color_temp": "color_temp_proxy"}, inplace=True)

    for col in ["brightness", "colorfulness", "color_temp_proxy"]:
        if col not in df_posts.columns:
            df_posts[col] = 0.0
        else:
            df_posts[col] = pd.to_numeric(df_posts[col], errors="coerce").fillna(0.0)

    # Optional: object edges without image CSV
    obj_col = None
    if "image_object" in df_posts.columns:
        obj_col = "image_object"
    elif "detected_object" in df_posts.columns:
        obj_col = "detected_object"

    if obj_col is not None:
        df_object_edges = df_posts[["post_id", "username", "datetime", obj_col]].copy()
        df_object_edges.rename(columns={obj_col: "image_object"}, inplace=True)
        df_object_edges["username"] = df_object_edges["username"].astype(str).str.strip()
        df_object_edges["image_object"] = df_object_edges["image_object"].astype(str).str.strip()
        df_object_edges = df_object_edges[df_object_edges["username"].isin(valid_users_dec)]
        df_object_edges = df_object_edges[df_object_edges["image_object"].notna() & (df_object_edges["image_object"] != "")]
    else:
        df_object_edges = pd.DataFrame(columns=["post_id", "username", "datetime", "image_object"])

    # --- 4. Prepare Graph Edges ---
    # Hashtags
    try:
        df_hashtags = pd.read_csv(HASHTAGS_FILE)
        df_hashtags.rename(columns={"source": "username", "target": "hashtag"}, inplace=True)
        df_hashtags["datetime"] = pd.to_datetime(df_hashtags["timestamp"], unit="s", errors="coerce")
        df_hashtags["username"] = df_hashtags["username"].astype(str).str.strip()
        df_hashtags = df_hashtags[df_hashtags["username"].isin(valid_users_dec)]
    except Exception:
        df_hashtags = pd.DataFrame(columns=["username", "hashtag", "datetime"])

    # Mentions
    try:
        df_mentions = pd.read_csv(MENTIONS_FILE)
        df_mentions.rename(columns={"source": "username", "target": "mention"}, inplace=True)
        df_mentions["datetime"] = pd.to_datetime(df_mentions["timestamp"], unit="s", errors="coerce")
        df_mentions["username"] = df_mentions["username"].astype(str).str.strip()
        df_mentions = df_mentions[df_mentions["username"].isin(valid_users_dec)]
    except Exception:
        df_mentions = pd.DataFrame(columns=["username", "mention", "datetime"])

    # --- 5. Prepare Influencer Profiles ---
    print("Merging profile features from influencers.txt...")
    df_influencers_external = load_influencer_profiles()
    df_influencers_external = df_influencers_external[df_influencers_external["username"].isin(valid_users_dec)]
    print(f"Filtered profiles from influencers.txt: {len(df_influencers_external):,} users (posted in Dec 2017)")

    df_active_base = pd.DataFrame({"username": list(valid_users_dec)})
    df_influencers = pd.merge(df_active_base, df_influencers_external, on="username", how="left")
    df_influencers["followers"] = df_influencers["followers"].fillna(0)
    df_influencers["followees"] = df_influencers["followees"].fillna(0)
    df_influencers["posts_history"] = df_influencers["posts_history"].fillna(0)
    df_influencers["category"] = df_influencers["category"].fillna("Unknown")

    current_date = time.strftime("%Y%m%d")
    output_user_file = f"active_influencers_v8_{current_date}.csv"
    print(f"Saving {len(df_influencers)} active influencers to '{output_user_file}'...")
    df_influencers.to_csv(output_user_file, index=False)

    df_posts["month"] = df_posts["datetime"].dt.to_period("M").dt.start_time

    # --- 6. Prepare Nodes ---
    influencer_set = set(df_influencers["username"].astype(str))
    all_hashtags = set(df_hashtags["hashtag"].astype(str))
    all_mentions = set(df_mentions["mention"].astype(str))
    all_image_objects = set(df_object_edges["image_object"].astype(str)) if len(df_object_edges) else set()

    print(
        f"Node counts: Influencers={len(influencer_set)}, Hashtags={len(all_hashtags)}, "
        f"Mentions={len(all_mentions)}, ImageObjects={len(all_image_objects)}"
    )

    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions | all_image_objects))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 7. Static Features ---
    node_df = pd.DataFrame({"username": all_nodes})
    profile_features = pd.merge(
        node_df,
        df_influencers[["username", "followers", "followees", "posts_history", "category"]],
        on="username",
        how="left",
    )
    for col in ["followers", "followees", "posts_history"]:
        profile_features[col] = pd.to_numeric(profile_features[col], errors="coerce").fillna(0)
        profile_features[col] = np.log1p(profile_features[col])

    category_dummies = pd.get_dummies(profile_features["category"], prefix="cat", dummy_na=True)
    profile_features = pd.concat([profile_features, category_dummies], axis=1).drop(columns=["category"])

    node_df["type"] = "unknown"
    node_df.loc[node_df["username"].isin(influencer_set), "type"] = "influencer"
    node_df.loc[node_df["username"].isin(all_hashtags), "type"] = "hashtag"
    node_df.loc[node_df["username"].isin(all_mentions), "type"] = "mention"
    if len(all_image_objects) > 0:
        node_df.loc[node_df["username"].isin(all_image_objects), "type"] = "image_object"

    node_type_dummies = pd.get_dummies(node_df["type"], prefix="type")
    static_features = pd.concat([profile_features, node_type_dummies], axis=1)
    static_feature_cols = list(static_features.drop("username", axis=1).columns)

    try:
        follower_feat_idx = static_feature_cols.index("followers")
        print(f"DEBUG: 'followers' feature is at index {follower_feat_idx} in static features.")
    except ValueError:
        print("Warning: 'followers' not found in static_feature_cols.")
        follower_feat_idx = 0

    # --- 8. Dynamic Features ---
    STATS_AGG = ["mean", "median", "min", "max"]
    required_cols = [
        "brightness",
        "colorfulness",
        "color_temp_proxy",
        "tag_count",
        "mention_count",
        "emoji_count",
        "caption_length",
        "caption_sent_pos",
        "caption_sent_neg",
        "caption_sent_neu",
        "caption_sent_compound",
        "feedback_rate",
        "comment_avg_pos",
        "comment_avg_neg",
        "comment_avg_neu",
        "comment_avg_compound",
    ]
    for col in required_cols:
        if col not in df_posts.columns:
            df_posts[col] = 0.0

    df_posts.sort_values(by=["username", "datetime"], inplace=True)
    df_posts["post_interval_sec"] = df_posts.groupby("username")["datetime"].diff().dt.total_seconds().fillna(0)

    if "post_category" not in df_posts.columns:
        post_categories = [f"post_cat_{i}" for i in range(10)]
        df_posts["post_category"] = np.random.choice(post_categories, size=len(df_posts))
    if "is_ad" not in df_posts.columns:
        df_posts["is_ad"] = 0

    agg_config = {
        "brightness": STATS_AGG,
        "colorfulness": STATS_AGG,
        "color_temp_proxy": STATS_AGG,
        "tag_count": STATS_AGG,
        "mention_count": STATS_AGG,
        "emoji_count": STATS_AGG,
        "caption_length": STATS_AGG,
        "caption_sent_pos": STATS_AGG,
        "caption_sent_neg": STATS_AGG,
        "caption_sent_neu": STATS_AGG,
        "caption_sent_compound": STATS_AGG,
        "post_interval_sec": STATS_AGG,
        "comment_avg_pos": STATS_AGG,
        "comment_avg_neg": STATS_AGG,
        "comment_avg_neu": STATS_AGG,
        "comment_avg_compound": STATS_AGG,
        "feedback_rate": "mean",
        "is_ad": "mean",
        "datetime": "size",
    }

    dynamic_agg = df_posts.groupby(["username", "month"]).agg(agg_config)
    dynamic_agg.columns = ["_".join(col).strip() for col in dynamic_agg.columns.values]
    dynamic_agg = dynamic_agg.reset_index()
    dynamic_agg.rename(
        columns={
            "datetime_size": "monthly_post_count",
            "feedback_rate_mean": "feedback_rate",
            "is_ad_mean": "ad_rate",
        },
        inplace=True,
    )

    post_category_rate = (
        df_posts.groupby(["username", "month"])["post_category"].value_counts(normalize=True).unstack(fill_value=0)
    )
    post_category_rate.columns = [f"rate_{col}" for col in post_category_rate.columns]
    post_category_rate = post_category_rate.reset_index()

    dynamic_features = pd.merge(dynamic_agg, post_category_rate, on=["username", "month"], how="left")
    dynamic_feature_cols = list(dynamic_features.drop(["username", "month"], axis=1).columns)

    # --- 9. Construct Graphs ---
    monthly_graphs = []
    start_date = end_date - pd.DateOffset(months=num_months - 1)

    feature_columns = static_feature_cols + dynamic_feature_cols
    feature_dim = len(feature_columns)
    print(f"Total feature dimension: {feature_dim}")

    # Use 'M' for broad pandas compatibility (month end)
    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq="M"), desc="Building monthly graphs"):
        snapshot_month = snapshot_date.to_period("M").start_time

        current_hashtags = df_hashtags[df_hashtags["datetime"] <= snapshot_date]
        current_mentions = df_mentions[df_mentions["datetime"] <= snapshot_date]
        current_image_objects = df_object_edges[df_object_edges["datetime"] <= snapshot_date] if len(df_object_edges) else df_object_edges

        edges_io = [
            (node_to_idx[str(u)], node_to_idx[str(o)])
            for u, o in zip(current_image_objects.get("username", []), current_image_objects.get("image_object", []))
            if str(u) in node_to_idx and str(o) in node_to_idx
        ]
        edges_ht = [
            (node_to_idx[str(u)], node_to_idx[str(h)])
            for u, h in zip(current_hashtags.get("username", []), current_hashtags.get("hashtag", []))
            if str(u) in node_to_idx and str(h) in node_to_idx
        ]
        edges_mt = [
            (node_to_idx[str(u)], node_to_idx[str(m)])
            for u, m in zip(current_mentions.get("username", []), current_mentions.get("mention", []))
            if str(u) in node_to_idx and str(m) in node_to_idx
        ]

        all_edges = list(set(edges_ht + edges_mt + edges_io))
        all_edges += [(idx, idx) for idx in influencer_indices]
        all_edges = list(set(all_edges))
        if not all_edges:
            all_edges = [(idx, idx) for idx in influencer_indices]

        num_nodes = len(all_nodes)
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

        current_dynamic = dynamic_features[dynamic_features["month"] == snapshot_month]
        snapshot_features = pd.merge(static_features, current_dynamic, on="username", how="left")
        snapshot_features = snapshot_features[feature_columns].fillna(0)
        x = torch.tensor(snapshot_features.astype(float).values, dtype=torch.float)

        # Target calculation per month
        monthly_posts_period = df_posts[df_posts["datetime"].dt.to_period("M") == snapshot_date.to_period("M")]
        monthly_agg = (
            monthly_posts_period.groupby("username")
            .agg(total_likes=("likes", "sum"), total_comments=("comments", "sum"), post_count=("datetime", "size"))
            .reset_index()
        )

        if metric_numerator == "likes_and_comments":
            monthly_agg["numerator"] = monthly_agg["total_likes"] + monthly_agg["total_comments"]
        else:
            monthly_agg["numerator"] = monthly_agg["total_likes"]

        if metric_denominator == "followers":
            numer_vals = pd.to_numeric(monthly_agg["numerator"], errors="coerce").fillna(0).values.astype(float)
            count_vals = pd.to_numeric(monthly_agg["post_count"], errors="coerce").fillna(0).values.astype(float)
            monthly_agg["avg_engagement_per_post"] = np.divide(
                numer_vals, count_vals, out=np.zeros_like(numer_vals), where=count_vals != 0
            )

            merged_data = pd.merge(monthly_agg, df_influencers[["username", "followers"]], on="username", how="left")
            numer = merged_data["avg_engagement_per_post"].values.astype(float)
            denom = pd.to_numeric(merged_data["followers"], errors="coerce").fillna(0).values.astype(float)
            merged_data["engagement"] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
        else:
            merged_data = monthly_agg
            numer = merged_data["numerator"].values.astype(float)
            denom = merged_data["post_count"].values.astype(float)
            merged_data["engagement"] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)

        engagement_data = (
            pd.merge(pd.DataFrame({"username": all_nodes}), merged_data[["username", "engagement"]], on="username", how="left")
            .fillna(0)
        )

        y = torch.tensor(engagement_data["engagement"].values, dtype=torch.float).view(-1, 1)
        monthly_graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols


# ===================== Model =====================
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_channels, hidden_channels)]
            + [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

    def forward(self, x, edge_index, edge_weight=None):
        outs = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight=edge_weight).relu() if edge_weight is not None else conv(x, edge_index).relu()
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
        combined_dim = gcn_dim * num_gcn_layers
        self.attentive_rnn = AttentiveRNN(combined_dim, rnn_dim)
        self.predictor = nn.Sequential(Linear(rnn_dim, 64), ReLU(), Dropout(dropout_prob), Linear(64, 1))

    def forward(self, gcn_embeddings, raw_features, baseline_scores=None):
        final_rep, attention_weights = self.attentive_rnn(gcn_embeddings)
        raw_output = self.predictor(final_rep).squeeze()
        predicted_scores = F.softplus(raw_output)
        return predicted_scores, attention_weights


# ===================== Loss =====================

# ---------------------------------------------------------------------------
# Compatibility alias
# ---------------------------------------------------------------------------
# The training / inference code expects a class named `InfluencerRankModel`.
# In this file, the implementation lives in `HardResidualInfluencerModel`.
# Provide an alias to avoid NameError and keep the rest of the pipeline intact.
InfluencerRankModel = HardResidualInfluencerModel


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
    colors = ["skyblue"] * max(0, (len(mean_att) - 1)) + (["salmon"] if len(mean_att) > 0 else [])
    bars = plt.bar(time_steps, mean_att, color=colors, edgecolor="black", alpha=0.7)

    plt.xlabel("Time Steps (Months)", fontsize=12)
    plt.ylabel("Average Attention Weight", fontsize=12)
    plt.title(f"Average Attention Weights across Time\nRun: {run_name}", fontsize=14)

    labels = [f"T-{len(mean_att)-1-i}" for i in range(len(mean_att))]
    if len(labels) > 0:
        labels[-1] = "Current (T)"
    plt.xticks(time_steps, labels)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom", fontsize=9)

    filename_bar = f"attention_weights_bar_{run_name}.png"
    plt.savefig(filename_bar, bbox_inches="tight")
    plt.close()

    # heatmap via matplotlib
    plt.figure(figsize=(12, 8))
    subset_matrix = attention_matrix[:50, :]
    plt.imshow(subset_matrix, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Time Steps (Oldest -> Newest)", fontsize=12)
    plt.ylabel("Sample Users (Top 50)", fontsize=12)
    plt.title("Attention Weights Heatmap (Individual)", fontsize=14)
    plt.xticks(time_steps, labels, rotation=0)

    filename_heat = f"attention_weights_heatmap_{run_name}.png"
    plt.savefig(filename_heat, bbox_inches="tight")
    plt.close()

    # numeric artifacts
    labels2 = [f"T-{len(mean_att)-1-i}" for i in range(len(mean_att))]
    if len(labels2) > 0:
        labels2[-1] = "Current (T)"
    df_mean = pd.DataFrame({"pos": np.arange(len(mean_att), dtype=int), "label": labels2, "alpha_mean": mean_att.astype(float)})
    filename_csv = f"attention_weights_mean_{run_name}.csv"
    df_mean.to_csv(filename_csv, index=False, float_format="%.8e")

    filename_raw = f"attention_weights_raw_{run_name}.npz"
    np.savez_compressed(filename_raw, attention=attention_matrix)

    return filename_bar, filename_heat, filename_csv, filename_raw


def generate_enhanced_scatter_plot(
    x_data,
    y_data,
    x_label,
    y_label,
    run_id,
    filename_suffix,
    color_data=None,
    color_label=None,
    title_suffix="",
):
    plt.figure(figsize=(11, 9))

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
        scatter = plt.scatter(x_masked, y_masked, c=c_masked, cmap="viridis", alpha=0.6, s=30)
        cbar = plt.colorbar(scatter)
        cbar.set_label(color_label or "color", fontsize=12)
    else:
        plt.scatter(x_masked, y_masked, alpha=0.5, s=30, label="Data Points")

    min_val = float(min(x_masked.min(), y_masked.min()))
    max_val = float(max(x_masked.max(), y_masked.max()))
    margin = (max_val - min_val) * 0.05 if (max_val > min_val) else 1.0
    plot_min = min_val - margin
    plot_max = max_val + margin

    plt.plot([plot_min, plot_max], [plot_min, plot_max], "r--", linewidth=2, label="Ideal (y=x)")
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
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    filename = f"scatter_{filename_suffix}_{run_id}.png"
    plt.savefig(filename, bbox_inches="tight")
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

    # NOTE: Use *local* indices (0..N_inf-1) to keep tensors small on MPS/Metal
    local_idx = torch.arange(len(influencer_indices), dtype=torch.long)
    return TensorDataset(local_idx, target_y, baseline_y)


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
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.input_graphs = input_graphs
        # Backward-compatible alias: some call sites expect `.graphs`.
        # Keep both names to avoid AttributeError.
        self.graphs = input_graphs
        self.T = len(input_graphs)
        self.target_global = int(target_local)
        self.target_node = self.target_global  # alias for helper methods
        self.explain_pos = int(explain_pos)
        self.device = device

        self.use_subgraph = use_subgraph
        self.num_hops = int(num_hops)
        self.undirected = bool(undirected)
        self.feat_mask_scope = feat_mask_scope
        self.edge_mask_scope = edge_mask_scope

        self.cached_proj = [None] * self.T
        self.cached_gcn = [None] * self.T
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
                self.cached_gcn[t] = out[self.target_global].detach()

    def _prepare_explain_graph(self):

        # Use only the target's 1-hop ego network (target + its neighbors)
        # and only incident edges (target, neighbor) to keep explanation fast and interpretable.
        data_full = self.graphs[self.explain_pos]  # already moved to device in __init__
        x_full = data_full.x
        edge_index_full = data_full.edge_index

        target_global = int(self.target_node)

        if edge_index_full.numel() == 0:
            subset = torch.tensor([target_global], dtype=torch.long, device=self.device)
            x_sub = x_full[subset]
            ei_sub = torch.empty((2, 0), dtype=torch.long, device=self.device)
            target_local = 0
        else:
            src, dst = edge_index_full
            inc_mask = (src == target_global) | (dst == target_global)

            if inc_mask.sum().item() == 0:
                subset = torch.tensor([target_global], dtype=torch.long, device=self.device)
                x_sub = x_full[subset]
                ei_sub = torch.empty((2, 0), dtype=torch.long, device=self.device)
                target_local = 0
            else:
                # neighbors in the full graph
                nbrs = torch.unique(torch.cat([dst[src == target_global], src[dst == target_global]], dim=0))
                subset = torch.unique(torch.cat([torch.tensor([target_global], device=self.device), nbrs], dim=0))
                subset, _ = torch.sort(subset)  # ensure sorted for searchsorted

                # keep only incident edges, then remap full->local using searchsorted (no huge idx_map)
                ei = edge_index_full[:, inc_mask]
                ei_sub = torch.searchsorted(subset, ei)
                valid = (subset[ei_sub[0]] == ei[0]) & (subset[ei_sub[1]] == ei[1])
                ei_sub = ei_sub[:, valid]

                x_sub = x_full[subset]
                target_local = int((subset == target_global).nonzero(as_tuple=False).view(-1)[0].item())

        # coalesce if available
        if ei_sub.numel() > 0:
            try:
                ei_sub = coalesce(ei_sub, num_nodes=x_sub.size(0))
            except Exception:
                pass

        self.x_exp = x_sub
        self.ei_exp = ei_sub
        self.target_local = int(target_local)
        self.local2global = subset

        # incident edges in the (local) explain graph
        if self.ei_exp.numel() == 0:
            self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            s, d = self.ei_exp
            incident = (s == self.target_local) | (d == self.target_local)
            self.incident_edge_idx = torch.where(incident)[0]

        if self.edge_mask_scope == "incident":
            self.num_edge_params = int(self.incident_edge_idx.numel())
        else:
            self.num_edge_params = int(self.ei_exp.size(1))

        self.feature_dim = int(self.x_exp.size(1))

        # Baseline for feature replacement (mean over nodes in the explained graph)
        # Used so that gating a feature to 0 replaces it with a typical value, not always literal 0.
        self.x_feat_baseline = self.x_exp.mean(dim=0, keepdim=True).detach()

    def num_mask_params(self):
        return self.feature_dim, self.num_edge_params

    def _apply_feature_gate(self, x, feat_gate):
        """Apply feature gates with mean-value replacement baseline.

        We implement: x' = x * g + ar{x} * (1-g), where ar{x} is the per-feature mean
        over nodes in the explained (sub)graph. This makes score impacts less likely to be
        numerically ~0 when features are centered around 0 or already sparse.
        """
        base = self.x_feat_baseline.to(x.device)
        g = feat_gate.view(1, -1)

        if self.feat_mask_scope in ("all", "subgraph"):
            return x * g + base * (1.0 - g)

        # target-only: only alter the target node's features
        # IMPORTANT (esp. on MPS): avoid in-place update that reads from the same view.
        # Using x2[t:t+1,:] on the RHS and writing back to the same slice can trigger
        # "one of the variables needed for gradient computation has been modified by an inplace operation".
        x2 = x.clone()
        t = int(self.target_local)
        xt = x[t:t+1, :]  # read from the original x (not x2) to avoid view/version conflicts
        new_xt = xt * g + base * (1.0 - g)
        x2[t:t+1, :] = new_xt
        return x2

    def _make_edge_weight(self, edge_gate):
        E = int(self.ei_exp.size(1))
        w = torch.ones(E, device=self.device)
        if E == 0 or edge_gate is None or edge_gate.numel() == 0:
            return w

        if self.edge_mask_scope == "incident":
            w = w.clone()
            if self.incident_edge_idx.numel() > 0:
                w[self.incident_edge_idx] = edge_gate
            return w

        return edge_gate

    def predict_with_gates(self, feat_gate, edge_gate, edge_weight_override=None, x_override=None):
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
        if self.edge_mask_scope == "incident":
            edge_gate = torch.ones(int(self.incident_edge_idx.numel()), device=self.device)
        else:
            edge_gate = torch.ones(int(self.ei_exp.size(1)), device=self.device)
        if edge_gate.numel() == 0:
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
    # extra knobs
    impact_reference="unmask",  impact_all=True, # "masked" | "unmasked" | "both"
    budget_feat=None,
    budget_edge=None,
    budget_weight=0.0,
    eps_abs_feat=1e-9,
    eps_rel_feat=1e-6,
    eps_abs_edge=1e-9,
    eps_rel_edge=1e-6,
):
    assert len(input_graphs) >= 2, "input_graphs length must be >= 2"

    if device is None:
        device = torch.device("mps" if (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()) else "cpu")

    T = len(input_graphs)
    if explain_pos < 0:
        explain_pos = (explain_pos + T) % T

    if coeffs is None:
        coeffs = {"edge_size": 0.05, "edge_ent": 0.10, "node_feat_size": 0.02, "node_feat_ent": 0.10}

    # --- helpers for readable names (safe: no graph ops) ---
    idx_to_name = None
    if node_to_idx is not None:
        try:
            idx_to_name = [None] * len(node_to_idx)
            for _n, _i in node_to_idx.items():
                _ii = int(_i)
                if 0 <= _ii < len(idx_to_name):
                    idx_to_name[_ii] = str(_n)
        except Exception:
            idx_to_name = None

    def _node_name(node_idx: int) -> str:
        try:
            i = int(node_idx)
        except Exception:
            return str(node_idx)
        if idx_to_name is not None and 0 <= i < len(idx_to_name):
            n = idx_to_name[i]
            return n if n is not None else str(i)
        return str(i)

    def _edge_label_and_endpoints(edge_k, ei, target_node_idx, node_to_idx, local2global=None):
        # ei: [2, E] with LOCAL indices.
        s_local = int(ei[0, edge_k].item())
        d_local = int(ei[1, edge_k].item())

        # Map to GLOBAL indices for human-readable names, if available.
        if local2global is not None:
            s = int(local2global[s_local].item())
            d = int(local2global[d_local].item())
            t = int(local2global[int(target_local)].item())
        else:
            s, d, t = s_local, d_local, int(target_local)

        sname = node_name(node_to_idx, s)
        dname = node_name(node_to_idx, d)

        # Force a "target -- neighbor" label whenever possible
        if s == t and d != t:
            return f"{sname}--{dname}", sname, dname, dname
        if d == t and s != t:
            return f"{dname}--{sname}", dname, sname, sname
        return f"{sname}--{dname}", sname, dname, dname
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

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
    )

    target_local = int(wrapper.target_local)

    Fdim, Edim = wrapper.num_mask_params()

    feat_logits = nn.Parameter(0.1 * torch.randn(Fdim, device=device))
    edge_logits = nn.Parameter(0.1 * torch.randn(Edim, device=device)) if Edim > 0 else None
    mask_params = [feat_logits] + ([edge_logits] if edge_logits is not None else [])
    opt = torch.optim.Adam(mask_params, lr=lr)

    orig = float(wrapper.original_pred())
    orig_t = torch.tensor(orig, device=device)

    print(f"🧠 [MaskOpt] target_node={int(target_local)} explain_pos={explain_pos}/{T-1} orig={orig:.6f}")
    print(f"   use_subgraph={use_subgraph}, num_hops={num_hops}, undirected={undirected}, feat_dim={Fdim}, edge_params={Edim}")

    if disable_cudnn_rnn:
        cudnn_ctx = _DisableCudnn()
    else:
        class _Null:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        cudnn_ctx = _Null()

    best = {"loss": float("inf"), "feat": None, "edge": None, "pred": None}

    def _budget_loss(gate, budget, denom):
        if (budget is None) or (gate is None) or (gate.numel() == 0):
            return gate.new_zeros(())
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
        loss_feat_ent = _binary_entropy(feat_gate).mean()

        if edge_gate is not None and edge_gate.numel() > 0:
            loss_edge_size = edge_gate.mean()
            loss_edge_ent = _binary_entropy(edge_gate).mean()
        else:
            loss_edge_size = pred.new_zeros(())
            loss_edge_ent = pred.new_zeros(())

        loss_budget = pred.new_zeros(())
        if float(budget_weight) > 0.0:
            loss_budget = _budget_loss(feat_gate, budget_feat, Fdim) + _budget_loss(edge_gate, budget_edge, max(1, Edim))

        loss = (
            float(fid_weight) * loss_fid
            + float(contrastive_weight) * loss_contrast
            + float(budget_weight) * loss_budget
            + coeffs["node_feat_size"] * loss_feat_size
            + coeffs["node_feat_ent"] * loss_feat_ent
            + coeffs["edge_size"] * loss_edge_size
            + coeffs["edge_ent"] * loss_edge_ent
        )

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
            print(
                f"  [MaskOpt] ep={ep:4d} loss={lval:.6e} fid={float(loss_fid.item()):.3e} "
                f"drop_d={float(delta.item()):.3e} pred={float(pred.item()):.6f} feat_max={feat_max:.4f} edge_max={edge_max:.4f}"
            )

    feat_gate = best["feat"].clamp(0.0, 1.0) if best["feat"] is not None else None
    edge_gate = best["edge"].clamp(0.0, 1.0) if best["edge"] is not None else None

    ones_feat = torch.ones(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None

    # Fallback: if optimization didn't produce gates, use unmasked (all-ones) gates
    if feat_gate is None:
        feat_gate = ones_feat
    if Edim > 0 and edge_gate is None:
        edge_gate = ones_edge

    # Numpy copies for ranking/iteration (importance scores)
    feat_np = feat_gate.detach().float().cpu().numpy()
    edge_np = edge_gate.detach().float().cpu().numpy() if (edge_gate is not None) else np.zeros(Edim, dtype=np.float32)


    def _thr(pred_base, eps_abs, eps_rel):
        return max(float(eps_abs), float(eps_rel) * abs(float(pred_base)))

    def _direction(diff, pred_base, eps_abs, eps_rel):
        th = _thr(pred_base, eps_abs, eps_rel)
        if abs(diff) <= th:
            return "Zero (0)"
        return "Positive (+)" if diff > 0 else "Negative (-)"

    with torch.no_grad():
        with cudnn_ctx:
            pred_unmasked = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())
            pred_masked = float(wrapper.predict_with_gates(feat_gate, edge_gate).item()) if (feat_gate is not None) else pred_unmasked

        # Feature/Edge attribution tables
    # - Score impact reference: "unmask" (default) or "masked"
    # - For each item, we record:
    #   * zero ablation: gate -> 0
    #   * mean replacement: gate -> mean(mask_gate)
    eps = 1e-12

    impact_reference_norm = str(impact_reference).lower() if impact_reference is not None else "unmask"
    if impact_reference_norm in ("unmask", "unmasked", "orig", "original"):
        keys_to_compute = ["unmasked"]
    elif impact_reference_norm in ("mask", "masked"):
        keys_to_compute = ["masked"]
    elif impact_reference_norm in ("both", "all"):
        keys_to_compute = ["unmasked", "masked"]
    else:
        keys_to_compute = ["unmasked"]

    # Baselines
    ref_map = {
        "unmasked": {"feat": ones_feat, "edge": ones_edge, "score": float(pred_unmasked)},
        "masked": {"feat": feat_gate, "edge": edge_gate, "score": float(pred_masked)},
    }
    orig_score = float(pred_unmasked)

    # Replacement values (平均置換)
    feat_mean_gate = float(feat_gate.mean().item()) if hasattr(feat_gate, "mean") else 0.5
    edge_mean_gate = float(edge_gate.mean().item()) if hasattr(edge_gate, "mean") else 0.5

    # Which indices to compute (artifacts want ALL by default)
    if impact_all:
        feat_iter = range(len(feat_np))
        edge_iter = range(len(edge_np))
    else:
        feat_iter = np.argsort(feat_np)[::-1][: int(topk_feat)]
        edge_iter = np.argsort(edge_np)[::-1][: int(topk_edge)]

    feature_rows = []
    for j in feat_iter:
        imp = float(feat_np[j])
        row = {
            "Type": "Feature",
            "Name": str(feature_names[j]),
            "FeatIdx": int(j),
            "Importance": imp,
            "OrigScore": orig_score,
        }

        for key in keys_to_compute:
            base_f = ref_map[key]["feat"]
            base_e = ref_map[key]["edge"]
            ref_score = float(ref_map[key]["score"])

            # --- zero ablation (gate -> 0) ---
            ab_f0 = base_f.clone()
            ab_f0[j] = 0.0
            pred0 = float(wrapper.predict_with_gates(feat_gate=ab_f0, edge_gate=base_e))
            diff0 = ref_score - pred0

            row[f"RefScore({key})"] = ref_score
            row[f"PredAblatedZero({key})"] = pred0
            row[f"Score_Impact({key})"] = diff0
            row[f"ImpactPct({key})"] = (diff0 / (ref_score + eps)) * 100.0
            row[f"ImpactPctOfOrig({key})"] = (diff0 / (orig_score + eps)) * 100.0
            row[f"Direction({key})"] = ("Positive (+)" if diff0 > 0 else ("Negative (-)" if diff0 < 0 else "Zero (0)"))

            # --- mean replacement (gate -> mean(mask_gate)) ---
            ab_fm = base_f.clone()
            ab_fm[j] = feat_mean_gate
            predm = float(wrapper.predict_with_gates(feat_gate=ab_fm, edge_gate=base_e))
            diffm = ref_score - predm

            row[f"PredAblatedMean({key})"] = predm
            row[f"Score_Impact_Mean({key})"] = diffm
            row[f"ImpactPct_Mean({key})"] = (diffm / (ref_score + eps)) * 100.0
            row[f"ImpactPctOfOrig_Mean({key})"] = (diffm / (orig_score + eps)) * 100.0
            row[f"Direction_Mean({key})"] = ("Positive (+)" if diffm > 0 else ("Negative (-)" if diffm < 0 else "Zero (0)"))

        feature_rows.append(row)

    edge_rows = []
    # target node label is derived from target_node_idx when needed
    for k in edge_iter:
        # Keep only edges incident to the target node (avoid edge-edge explanations)
        try:
            src_local = int(wrapper.ei_exp[0, k])
            dst_local = int(wrapper.ei_exp[1, k])
        except Exception:
            continue
        if (src_local != int(target_local)) and (dst_local != int(target_local)):
            continue
        imp = float(edge_np[k])
        label, src_name, dst_name, neighbor = _edge_label_and_endpoints(int(k), wrapper.ei_exp, target_node_idx, node_to_idx, local2global=wrapper.local2global)
        row = {
            "Type": "Edge",
            "Name": label,
            "EdgeIdx": int(k),
            "Src": src_name,
            "Dst": dst_name,
            "Neighbor": neighbor,
            "Importance": imp,
            "OrigScore": orig_score,
        }

        for key in keys_to_compute:
            base_f = ref_map[key]["feat"]
            base_e = ref_map[key]["edge"]
            ref_score = float(ref_map[key]["score"])

            # --- zero ablation (gate -> 0) ---
            ab_e0 = base_e.clone()
            ab_e0[k] = 0.0
            pred0 = float(wrapper.predict_with_gates(feat_gate=base_f, edge_gate=ab_e0))
            diff0 = ref_score - pred0

            row[f"RefScore({key})"] = ref_score
            row[f"PredAblatedZero({key})"] = pred0
            row[f"Score_Impact({key})"] = diff0
            row[f"ImpactPct({key})"] = (diff0 / (ref_score + eps)) * 100.0
            row[f"ImpactPctOfOrig({key})"] = (diff0 / (orig_score + eps)) * 100.0
            row[f"Direction({key})"] = ("Positive (+)" if diff0 > 0 else ("Negative (-)" if diff0 < 0 else "Zero (0)"))

            # --- mean replacement (gate -> mean(mask_gate)) ---
            ab_em = base_e.clone()
            ab_em[k] = edge_mean_gate
            predm = float(wrapper.predict_with_gates(feat_gate=base_f, edge_gate=ab_em))
            diffm = ref_score - predm

            row[f"PredAblatedMean({key})"] = predm
            row[f"Score_Impact_Mean({key})"] = diffm
            row[f"ImpactPct_Mean({key})"] = (diffm / (ref_score + eps)) * 100.0
            row[f"ImpactPctOfOrig_Mean({key})"] = (diffm / (orig_score + eps)) * 100.0
            row[f"Direction_Mean({key})"] = ("Positive (+)" if diffm > 0 else ("Negative (-)" if diffm < 0 else "Zero (0)"))

        edge_rows.append(row)

    df_feat = pd.DataFrame(feature_rows)
    df_edge = pd.DataFrame(edge_rows)

    # ---- Console summaries (ranges + a few rows) ----
    def _print_df_summary(df, name: str):
        if df is None or len(df) == 0:
            print(f"[{name}] (empty)")
            return
        imp = df["Importance"].astype(float)
        print(f"[{name}] rows={len(df)}  Importance(min/mean/max)={imp.min():.4g}/{imp.mean():.4g}/{imp.max():.4g}")
        for c in [col for col in df.columns if col.startswith("Score_Impact(")]:
            v = df[c].astype(float)
            print(f"  {c}: min/mean/max={v.min():.4g}/{v.mean():.4g}/{v.max():.4g}  abs_max={v.abs().max():.4g}")
        # show top-10 by importance and by |impact| (default: impact_reference)
        impact_key = f"Score_Impact({impact_reference})"
        if impact_key not in df.columns:
            # fallback: any available Score_Impact(...)
            cand = [c for c in df.columns if c.startswith("Score_Impact(") and c.endswith(")")]
            if cand:
                impact_key = cand[0]
        if impact_key in df.columns:
            df_by_impact = df.reindex(df[impact_key].abs().sort_values(ascending=False).index).head(10)
            print("Top-10 by |impact|:", impact_key)
            print(df_by_impact[["Name", "Importance", impact_key, "Direction"]].to_string(index=False))
    _print_df_summary(df_feat, "Features")
    _print_df_summary(df_edge, "IncidentEdges")

    meta = {
        "orig_pred": float(orig),
        "best_pred": float(best["pred"]) if best["pred"] is not None else None,
        "best_loss": float(best["loss"]),
        "target_node": int(target_local),
        "explain_pos": int(explain_pos),
        "T": int(T),
        "feat_dim": int(Fdim),
        "edge_params": int(Edim),
        "pred_unmasked": float(pred_unmasked),
        "pred_masked": float(pred_masked),
        "impact_reference": str(impact_reference),
        "budget_feat": None if budget_feat is None else float(budget_feat),
        "budget_edge": None if budget_edge is None else float(budget_edge),
        "budget_weight": float(budget_weight),
        "coeffs": dict(coeffs),
        "fid_weight": float(fid_weight),
    }
    return df_feat, df_edge, meta


def compute_time_step_sensitivity(
    model,
    input_graphs,
    target_node_idx,
    device,
    topk=3,
    score_mode="alpha_x_delta",
    min_delta=1e-6,
):
    """Pick important time steps by dropping each step and measuring delta."""
    model = model.to(device)
    model.eval()

    T = len(input_graphs)
    target_node_idx = int(target_local)

    with torch.no_grad():
        seq_gcn = []
        seq_raw = []
        for g in input_graphs:
            g = g.to(device)
            p = model.projection_layer(g.x)
            h = _gcn_forward_concat(model.gcn_encoder, p, g.edge_index, edge_weight=None)
            seq_gcn.append(h[target_node_idx])
            seq_raw.append(p[target_node_idx])

        seq_gcn = torch.stack(seq_gcn, dim=0).unsqueeze(0)
        seq_raw = torch.stack(seq_raw, dim=0).unsqueeze(0)

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

            rows.append(
                {"pos": int(t), "alpha": float(alpha[t]), "pred_full": pred_full, "pred_drop": pred_drop, "delta_total": float(delta), "score": float(score)}
            )

    sens_df = pd.DataFrame(rows).sort_values(["score", "delta_total", "alpha"], ascending=False).reset_index(drop=True)
    selected_positions = sens_df.head(int(topk))["pos"].astype(int).tolist()
    return sens_df, selected_positions, pred_full, alpha


def mlflow_log_pred_scatter(y_true, y_pred, tag="eval", step=None, artifact_path="plots", fname="pred_vs_true_scatter.png", title=None):
    import numpy as np
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")

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

    if mlflow.active_run() is not None:
        ap = artifact_path
        if step is not None:
            ap = os.path.join(ap, f"step_{int(step)}")
        mlflow.log_artifact(out_path, artifact_path=ap)

    try:
        os.remove(out_path)
    except Exception:
        pass


def mlflow_log_maskopt_plots(df_feat, df_edge, meta=None, tag="pos_0", topk_feat=15, topk_edge=15, artifact_path="xai", fname_prefix="maskopt"):
    # Feature bar plot
    if df_feat is not None and len(df_feat) > 0:
        d = df_feat.sort_values("Importance", ascending=False).head(int(topk_feat))
        plt.figure(figsize=(10, 6))
        plt.barh(list(reversed(d["Name"].tolist())), list(reversed(d["Importance"].astype(float).tolist())))
        plt.xlabel("Importance (mask)")
        plt.title(f"Top Features ({tag})")
        plt.tight_layout()
        fpath = f"{fname_prefix}_feat_{tag}.png"
        plt.savefig(fpath, dpi=200)
        plt.close()
        if mlflow.active_run() is not None:
            mlflow.log_artifact(fpath, artifact_path=artifact_path)
        try:
            os.remove(fpath)
        except Exception:
            pass

    # Edge bar plot
    if df_edge is not None and len(df_edge) > 0:
        d = df_edge.sort_values("Importance", ascending=False).head(int(topk_edge))
        plt.figure(figsize=(10, 6))
        plt.barh(list(reversed(d["Name"].tolist())), list(reversed(d["Importance"].astype(float).tolist())))
        plt.xlabel("Importance (mask)")
        plt.title(f"Top Edges ({tag})")
        plt.tight_layout()
        epath = f"{fname_prefix}_edge_{tag}.png"
        plt.savefig(epath, dpi=200)
        plt.close()
        if mlflow.active_run() is not None:
            mlflow.log_artifact(epath, artifact_path=artifact_path)
        try:
            os.remove(epath)
        except Exception:
            pass

    # Meta json
    if meta is not None:
        import json
        mpath = f"{fname_prefix}_meta_{tag}.json"
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if mlflow.active_run() is not None:
            mlflow.log_artifact(mpath, artifact_path=artifact_path)
        try:
            os.remove(mpath)
        except Exception:
            pass

        # Save full attribution tables (all rows) as CSV artifacts
        try:
            if df_feat is not None and len(df_feat) > 0:
                feat_csv = f"{fname_prefix}_features_all_{tag}.csv"
                df_feat.to_csv(feat_csv, index=False)
                mlflow.log_artifact(feat_csv, artifact_path=artifact_path)
                try:
                    os.remove(feat_csv)
                except Exception:
                    pass
            if df_edge is not None and len(df_edge) > 0:
                edge_csv = f"{fname_prefix}_edges_all_{tag}.csv"
                df_edge.to_csv(edge_csv, index=False)
                mlflow.log_artifact(edge_csv, artifact_path=artifact_path)
                try:
                    os.remove(edge_csv)
                except Exception:
                    pass
        except Exception as _e_csv:
            print(f"[MLflow] (warn) full CSV logging failed: {_e_csv}")


# ---- Extra normalized scatter (0..1) for comparing node-features vs incident-edges ----
# ===================== Training / Evaluation / Explanation =====================

    # --- Rank scatter: Importance rank vs |Score impact| rank (for analysis) ---
    try:
        ref_raw = meta.get("impact_reference", "unmask") if isinstance(meta, dict) else "unmask"
        ref_norm = str(ref_raw).strip().lower() if ref_raw is not None else "unmask"
        if ref_norm in ("unmask", "unmasked", "orig", "original"):
            ref_key = "unmasked"
        elif ref_norm in ("mask", "masked"):
            ref_key = "masked"
        else:
            ref_key = "unmasked"
        ref_col = f"Score_Impact({ref_key})"


        def _log_rank_scatter(df, impact_col, fname, title):
            if df is None or len(df) == 0:
                return
            if impact_col not in df.columns or "Importance" not in df.columns:
                return

            tmp = df[["Importance", impact_col]].dropna().copy()
            if len(tmp) == 0:
                return

            tmp["Rank_Importance"] = (-tmp["Importance"].abs()).rank(method="dense")
            tmp["Rank_ImpactAbs"] = (-tmp[impact_col].abs()).rank(method="dense")

            spearman = tmp["Importance"].corr(tmp[impact_col].abs(), method="spearman")
            if spearman == spearman:
                metric_name = _sanitize_mlflow_name(f"maskopt_spearman_importance_vs_absimpact_{impact_col}")
                mlflow.log_metric(metric_name, float(spearman))

            fig = plt.figure()
            plt.scatter(tmp["Rank_Importance"], tmp["Rank_ImpactAbs"], s=8)
            plt.xlabel("Rank (Importance) [1=highest]")
            plt.ylabel(f"Rank (|Impact|): {impact_col} [1=highest]")
            plt.title(f"{title}\nSpearman(Importance, |Impact|) = {spearman:.3f}" if spearman == spearman else title)
            plt.grid(True, alpha=0.3)
            fig_path = os.path.join(xai_out_dir, fname)
            fig.savefig(fig_path, dpi=160, bbox_inches="tight")
            mlflow.log_artifact(fig_path)
            plt.close(fig)
        def _log_absimpact01_scatter(df_feat, df_edge, impact_col, fname, title):
            if df_feat is None or df_edge is None:
                return
            if impact_col not in df_feat.columns or impact_col not in df_edge.columns:
                return
            if "Importance" not in df_feat.columns or "Importance" not in df_edge.columns:
                return

            max_abs = max(float(df_feat[impact_col].abs().max()), float(df_edge[impact_col].abs().max()))
            if not (max_abs > 0):
                return

            feat_x = df_feat["Importance"].astype(float).clip(0, 1).values
            feat_y = (df_feat[impact_col].abs().astype(float) / max_abs).clip(0, 1).values
            edge_x = df_edge["Importance"].astype(float).clip(0, 1).values
            edge_y = (df_edge[impact_col].abs().astype(float) / max_abs).clip(0, 1).values

            # Combined plot (node-features + incident-edges) on the SAME [0,1] scale
            fig = plt.figure()
            plt.scatter(feat_x, feat_y, s=10, alpha=0.7, label="Node features")
            plt.scatter(edge_x, edge_y, s=10, alpha=0.7, label="Incident edges")
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.xlabel("Importance (gate) [0..1]")
            plt.ylabel(f"|Impact| / max(|Impact|) [0..1]  (col={impact_col})")
            plt.title(title + f"\nmax(|Impact|)={max_abs:.6g}")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best")

            fig_path = os.path.join(xai_out_dir, fname)
            fig.savefig(fig_path, dpi=160, bbox_inches="tight")
            mlflow.log_artifact(fig_path)
            plt.close(fig)
        _log_rank_scatter(df_feat, f"Score_Impact({ref_col})", f"{tag}_feat_rank_scatter_zero.png",
                          f"Features: rank(Importance) vs rank(|Score impact|) [{ref_col}/zero]")
        _log_rank_scatter(df_edge, f"Score_Impact({ref_col})", f"{tag}_edge_rank_scatter_zero.png",
                          f"Edges: rank(Importance) vs rank(|Score impact|) [{ref_col}/zero]")

        _log_rank_scatter(df_feat, f"Score_Impact_Mean({ref_col})", f"{tag}_feat_rank_scatter_mean.png",
                          f"Features: rank(Importance) vs rank(|Score impact|) [{ref_col}/mean]")
        _log_rank_scatter(df_edge, f"Score_Impact_Mean({ref_col})", f"{tag}_edge_rank_scatter_mean.png",
                          f"Edges: rank(Importance) vs rank(|Score impact|) [{ref_col}/mean]")
        if params.get("PLOT_SCALE_0_1", False):
            _log_absimpact01_scatter(
                df_feat, df_edge,
                f"Score_Impact({ref_col})",
                f"{tag}_absimpact01_zero.png",
                f"[MaskOpt] Importance vs |Impact| (normalized 0..1) — ZERO ablation ({ref_col})"
            )
            _log_absimpact01_scatter(
                df_feat, df_edge,
                f"Score_Impact_Mean({ref_col})",
                f"{tag}_absimpact01_mean.png",
                f"[MaskOpt] Importance vs |Impact| (normalized 0..1) — MEAN ablation ({ref_col})"
            )
    except Exception as _e:
        print(f"[MLflow] rank-scatter logging skipped: {_e}")


def run_experiment(params, graphs_data, experiment_id=None):
    """Train/infer InfluencerRank and optionally run MaskOpt XAI.

    graphs_data tuple:
      (monthly_graphs, influencer_indices, node_to_idx, feature_dim, follower_feat_idx, static_cols, dynamic_cols)
    """
    import numpy as np
    import pandas as pd
    import torch
    import mlflow
    from torch.utils.data import DataLoader

    monthly_graphs, influencer_indices, node_to_idx, feature_dim, follower_feat_idx, static_cols, dynamic_cols = graphs_data

    device = params.get("DEVICE", "auto")
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Basic info
    T_total = len(monthly_graphs)
    if T_total < 2:
        raise ValueError("monthly_graphs must contain at least 2 months")
    history_graphs = monthly_graphs[:-1]
    label_graph = monthly_graphs[-1]
    T = len(history_graphs)
    feature_names = list(static_cols) + list(dynamic_cols)

    # MLflow context is managed by main(); but ensure tracking uri is set.
    if params.get("MLFLOW_TRACKING_URI"):
        try:
            mlflow.set_tracking_uri(params["MLFLOW_TRACKING_URI"])
        except Exception:
            pass

    # Create model
    model = InfluencerRankModel(
        feature_dim=feature_dim,
        gcn_hidden_dim=params.get("GCN_HIDDEN_DIM", 64),
        gcn_out_dim=params.get("GCN_OUT_DIM", 64),
        lstm_hidden_dim=params.get("LSTM_HIDDEN_DIM", 64),
        num_gcn_layers=params.get("NUM_GCN_LAYERS", 2),
        num_attention_heads=params.get("NUM_ATTENTION_HEADS", 2),
        dropout=params.get("DROPOUT", 0.1),
    ).to(device)

    # ----- Load model if provided -----
    loaded_from = None
    model_state_local = None

    model_pth = params.get("MODEL_PTH_PATH")
    if model_pth and os.path.exists(model_pth):
        model_state_local = model_pth
        loaded_from = f"pth:{model_pth}"
    else:
        # Try MLflow run-id artifact download
        rid = params.get("MLFLOW_MODEL_RUN_ID")
        if rid:
            # try common artifact paths
            candidate_artifacts = [
                params.get("MLFLOW_MODEL_ARTIFACT", "model/model_state_dict.pth"),
                "model/model_state_dict.pth",
                "model_state_dict.pth",
            ]
            for ap in candidate_artifacts:
                try:
                    lp = mlflow_download_model_artifact(rid, ap, dst_dir=params.get("OUTDIR", "."))
                except Exception:
                    lp = None
                if lp and os.path.exists(lp):
                    model_state_local = lp
                    loaded_from = f"mlflow:{rid}:{ap}"
                    break

    if model_state_local:
        sd = torch.load(model_state_local, map_location="cpu")
        model.load_state_dict(sd)
        model.eval()
        print(f"[Model] ✅ Loaded state_dict from {loaded_from}")
    else:
        if params.get("INFERENCE_ONLY", False):
            print("[Model] ⚠️ Inference-only requested but no model could be loaded; training a new model.")
        # ----- Training -----
        epochs = int(params.get("EPOCHS", 50))
        lr = float(params.get("LR", 1e-3))
        batch_size = int(params.get("BATCH_SIZE", 128))
        list_size = int(params.get("LIST_SIZE", 256))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        crit_rank = ListMLELoss()
        crit_reg = torch.nn.MSELoss()

        # Move history graphs to device once
        gpu_hist = [g.to(device) for g in history_graphs]
        gpu_label = label_graph.to(device)

        train_ds = get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-1)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        last_epoch = 0
        for epoch in range(1, epochs + 1):
            model.train()

            # Precompute influencer embeddings for all months (history)
            emb_list = []
            raw_list = []
            for g in gpu_hist:
                proj_x = model.projection_layer(g.x)
                z = _gcn_forward_concat(model.gcn_encoder, proj_x, g.edge_index)
                emb_list.append(z[influencer_indices])
                raw_list.append(g.x[influencer_indices])
            gcn_embeddings = torch.stack(emb_list, dim=1)   # [Ninf, T, D]
            raw_features = torch.stack(raw_list, dim=1)     # [Ninf, T, F]

            target_y = gpu_label.y[influencer_indices].squeeze()
            baseline_y = monthly_graphs[-2].y[influencer_indices].to(device).squeeze()

            total_loss = 0.0
            n_batches = 0
            for local_idx, y_true_b, y_base_b in train_loader:
                local_idx = local_idx.to(device)
                y_true_b = y_true_b.to(device)
                y_base_b = y_base_b.to(device)

                # Sample a listwise subset from this batch (optional)
                if list_size > 0 and local_idx.numel() > list_size:
                    perm = torch.randperm(local_idx.numel(), device=device)[:list_size]
                    local_idx = local_idx[perm]
                    y_true_b = y_true_b[perm]
                    y_base_b = y_base_b[perm]

                pred = model(
                    gcn_embeddings[local_idx],
                    raw_features[local_idx],
                    baseline_scores=y_base_b,
                ).squeeze()

                loss_rank = crit_rank(pred, y_true_b)
                loss_reg = crit_reg(pred, y_true_b)
                loss = loss_rank + float(params.get("LAMBDA_REG", 0.1)) * loss_reg

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.detach().cpu())
                n_batches += 1

            avg_loss = total_loss / max(1, n_batches)
            if epoch % int(params.get("LOG_EVERY", 10)) == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch}/{epochs} Loss: {avg_loss:.4f}")
                try:
                    mlflow.log_metric("train_loss", avg_loss, step=epoch)
                except Exception:
                    pass

            last_epoch = epoch

        # Save model state to MLflow artifact + local cache
        try:
            local_out = os.path.join(params.get("OUTDIR", "."), "model_state_dict.pth")
            torch.save(model.state_dict(), local_out)
            mlflow.log_artifact(local_out, artifact_path="model")
            run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
            print(f"[MLflow] ✅ Saved trained model artifact: run_id={run_id} -> model/model_state_dict.pth")
            print(f"[MLflow]    local_path={local_out}")
        except Exception as e:
            print(f"[MLflow] ⚠️ Failed to log model artifact: {e}")

    # ----- Inference (always) -----
    model.eval()
    gpu_hist = [g.to(device) for g in history_graphs]
    gpu_label = label_graph.to(device)

    with torch.no_grad():
        emb_list = []
        raw_list = []
        for g in gpu_hist:
            proj_x = model.projection_layer(g.x)
            z = _gcn_forward_concat(model.gcn_encoder, proj_x, g.edge_index)
            emb_list.append(z[influencer_indices])
            raw_list.append(g.x[influencer_indices])
        gcn_embeddings = torch.stack(emb_list, dim=1)
        raw_features = torch.stack(raw_list, dim=1)
        y_true = gpu_label.y[influencer_indices].squeeze()
        y_base = monthly_graphs[-2].y[influencer_indices].to(device).squeeze()
        y_pred = model(gcn_embeddings, raw_features, baseline_scores=y_base).squeeze()

    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # Metrics
    mae = float(np.mean(np.abs(y_pred_np - y_true_np)))
    rmse = float(np.sqrt(np.mean((y_pred_np - y_true_np) ** 2)))
    try:
        pearson = float(np.corrcoef(y_pred_np, y_true_np)[0, 1])
    except Exception:
        pearson = float("nan")
    try:
        from scipy.stats import spearmanr
        spearman = float(spearmanr(y_pred_np, y_true_np).correlation)
    except Exception:
        spearman = float("nan")

    print("\n--- 📊 Evaluation Metrics ---")

    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, Pearson: {pearson:.4f}, Spearman: {spearman:.4f}")
    try:
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("Pearson", pearson)
        mlflow.log_metric("Spearman", spearman)
    except Exception:
        pass

    # Prediction scatter
    try:
        fig = plt.figure()
        plt.scatter(y_true_np, y_pred_np, s=8)
        plt.xlabel("y_true")
        plt.ylabel("y_pred")
        plt.title("Prediction scatter (Influencers)")
        out_png = os.path.join(params.get("OUTDIR", "."), "pred_scatter.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        mlflow.log_artifact(out_png, artifact_path="plots")
    except Exception as e:
        print(f"[Plots] pred scatter logging failed: {e}")

    # ----- Choose hub influencer (many incident edges) -----
    # Use last history graph for degree
    last_hist = history_graphs[-1]
    ei = last_hist.edge_index
    if ei.numel() == 0:
        target_node_global_idx = int(influencer_indices[0])
        max_degree = 0
    else:
        src = ei[0].cpu().numpy()
        dst = ei[1].cpu().numpy()
        deg = {}
        inf_set = set(int(i) for i in influencer_indices)
        for u, v in zip(src, dst):
            if int(u) in inf_set:
                deg[int(u)] = deg.get(int(u), 0) + 1
            if int(v) in inf_set:
                deg[int(v)] = deg.get(int(v), 0) + 1
        if not deg:
            target_node_global_idx = int(influencer_indices[0])
            max_degree = 0
        else:
            target_node_global_idx, max_degree = max(deg.items(), key=lambda kv: kv[1])

    # Pretty name
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    target_name = idx_to_node.get(int(target_node_global_idx), str(target_node_global_idx))
    print(f"\n🎯 Selected Hub Influencer: {_node_name(node_to_idx, target_node_global_idx)} (global_idx={target_node_global_idx}) with {int(max_degree)} incident edges.")

    # ----- XAI (MaskOpt) -----
    if params.get("XAI_ONLY", False) or params.get("RUN_XAI", True):
        positions = params.get("POSITIONS_TO_EXPLAIN", [0, 5, 3])
        positions = [int(x) for x in positions]
        positions = [x for x in positions if 0 <= x < T]
        print(f"[Explain] positions_to_explain={positions} / T={T}")

        # Run explain per position
        for pos in positions:
            try:
                df_feat, df_edge = maskopt_e2e_explain(
                    model=model,
                    graphs=history_graphs,
                    target_node_global_idx=int(target_node_global_idx),
                    explain_pos=int(pos),
                    node_to_idx=node_to_idx,
                    feature_names=feature_names,
                    topk_feat=int(params.get("TOPK_FEAT", 30)),
                    topk_edge=int(params.get("TOPK_EDGE", 200)),
                    outdir=params.get("OUTDIR", "."),
                    ref_col=params.get("SCORE_IMPACT_REF", "unmasked"),
                )

                # Print value ranges
                def _summ(df, label):
                    if df is None or len(df) == 0:
                        print(f"[{label}] empty")
                        return
                    imp = df["Importance"].to_numpy()
                    # prefer Mean column if exists
                    col = None
                    for c in ["Score_Impact_Mean(unmasked)", "Score_Impact(unmasked)", "Score_Impact_Mean(orig)", "Score_Impact(orig)"]:
                        if c in df.columns:
                            col = c
                            break
                    if col is None:
                        return
                    si = df[col].to_numpy()
                    print(f"[{label}] importance: min={imp.min():.6g} max={imp.max():.6g} mean={imp.mean():.6g} | abs(score_impact): min={np.abs(si).min():.6g} max={np.abs(si).max():.6g} mean={np.abs(si).mean():.6g}")

                _summ(df_feat, "Feature")
                _summ(df_edge, "Edge")

                mlflow_log_maskopt_plots(
                    df_feat=df_feat,
                    df_edge=df_edge,
                    tag=f"pos_{pos}",
                    outdir=params.get("OUTDIR", "."),
                    ref_key=params.get("SCORE_IMPACT_REF", "unmasked"),
                    plot_scale_0_1=bool(params.get("PLOT_SCALE_0_1", False)),
                )
            except Exception as e:
                print(f"💥 Explanation Error: {e}")

    return (mlflow.active_run().info.run_id if mlflow.active_run() else None), {
        "MAE": mae,
        "RMSE": rmse,
        "Pearson": pearson,
        "Spearman": spearman,
        "selected_target_global_idx": int(target_node_global_idx),
        "selected_target_name": str(target_name),
    }

def main():
    # Apply CLI overrides (preferred) -> environment variables for backward compatibility
    _apply_cli_overrides_to_env(sys.argv[1:])

    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    experiment_name, experiment_id = setup_mlflow_experiment(
        experiment_base_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "InfluencerRankSweep"),
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI"),
        local_artifact_dir=os.environ.get("MLFLOW_ARTIFACT_DIR", "mlruns_artifacts"),
    )

    target_date = pd.to_datetime("2017-12-31")
    prep = prepare_graph_data(end_date=target_date, num_months=12, metric_numerator="likes_and_comments", metric_denominator="followers")
    if prep[0] is None:
        print("Data preparation failed.")
        return 1

    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prep
    feature_dim = monthly_graphs[0].x.shape[1]
    print(f"Final feature dimension: {feature_dim}")
    print(f"Follower feature index: {follower_feat_idx}")

    graphs_data = (monthly_graphs, influencer_indices, node_to_idx, feature_dim, follower_feat_idx, static_cols, dynamic_cols)

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
        "EPOCHS": 150,
        # inference / XAI controls (from CLI/env)
        "DEVICE": os.getenv("DEVICE", "auto"),
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        "MLFLOW_EXPERIMENT_NAME": os.getenv("MLFLOW_EXPERIMENT_NAME", "InfluencerRankSweep"),
        "MLFLOW_MODEL_RUN_ID": os.getenv("MLFLOW_MODEL_RUN_ID", "").strip(),
        "MLFLOW_MODEL_ARTIFACT": os.getenv("MLFLOW_MODEL_ARTIFACT", "model_state_dict.pth"),
        "MODEL_PTH_PATH": os.getenv("MODEL_PTH_PATH", "").strip(),
        "XAI_ONLY": os.getenv("XAI_ONLY", "0") == "1",
        "INFERENCE_ONLY": os.getenv("INFERENCE_ONLY", "0") == "1",
        "SKIP_TRAINING": os.getenv("SKIP_TRAINING", "0") == "1",
        "PLOT_SCALE_0_1": os.getenv("PLOT_SCALE_0_1", "0") == "1",
        "LIST_SIZE": 50,
        "BATCH_SIZE": 50 * 64,
        "USE_SAMPLER": True,
        # XAI selection knobs (optional)
        "explain_use_sensitivity": True,
        "xai_topk_pos": 3,
        "sensitivity_score_mode": "alpha_x_delta",
        "sensitivity_min_delta": 1e-4,
        "xai_attn_min_w": 0.0,
    }

    # Put your overrides here if you want a sweep.
    params_list = [
        # {"LR": 0.003, "DROPOUT_PROB": 0.2, "POINTWISE_LOSS_WEIGHT": 0.5},
    ]

    def _expand_grid(grid_dict):
        keys = list(grid_dict.keys())
        if not keys:
            return [{}]
        values = [grid_dict[k] for k in keys]
        out = []
        for combo in itertools.product(*values):
            out.append({k: v for k, v in zip(keys, combo)})
        return out

    sweep_grid = {}  # or define grid

    def _merge(base, overrides):
        p = dict(base)
        p.update(overrides)
        return p

    def _suffix(overrides):
        if not overrides:
            return "base"
        parts = []
        for k, v in overrides.items():
            parts.append(f"{k}={v}")
        return ",".join(parts)

    overrides_list = list(params_list) if len(params_list) > 0 else _expand_grid(sweep_grid)
    params_list_full = [_merge(base_params, ov) for ov in overrides_list]

    summary_rows = []
    for i, p in enumerate(params_list_full):
        p = dict(p)
        overrides_for_name = dict(overrides_list[i]) if i < len(overrides_list) else {}
        p["name_prefix"] = f"{base_params['name_prefix']}_{i:03d}"
        p["note"] = f"{base_params.get('note','')} | sweep={i+1}/{len(params_list_full)} | {_suffix(overrides_for_name)}"

        run_id, metrics = run_experiment(p, graphs_data, experiment_id=experiment_id)
        row = {"run_index": i, "run_id": run_id, "note": p.get("note", "")}
        for k in ["LR", "DROPOUT_PROB", "POINTWISE_LOSS_WEIGHT", "GCN_DIM", "RNN_DIM", "NUM_GCN_LAYERS", "PROJECTION_DIM", "EPOCHS", "LIST_SIZE", "BATCH_SIZE", "USE_SAMPLER"]:
            row[k] = p.get(k)
        if isinstance(metrics, dict):
            row.update(metrics)
        summary_rows.append(row)

    if len(summary_rows) > 1:
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


if __name__ == "__main__":
    raise SystemExit(main())