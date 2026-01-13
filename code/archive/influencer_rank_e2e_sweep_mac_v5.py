#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InfluencerRank reproduction + training/inference + XAI (MaskOpt E2E)
(no external image CSV + Mac-friendly device selection + MLflow sweep)

✅ Goals (your requirements)
- No undefined variables / parameters
- Every run is saved in MLflow (params, metrics, plots, model state_dict)
- Epoch is fixed to 50 by default (can override but default is 50)
- Runs without crashing: MPS is tried first, but automatically falls back to CPU if PyG ops on MPS fail
- Customizable execution:
    --mode prepare | train | infer | xai | all
    --sweep_mode small|medium|large or explicit lists
    --xai_on_best to run XAI only for the best run after sweep (recommended)

Usage examples
--------------
# 1) Mac: sweep (train+eval) on MPS if possible, else CPU fallback
python influencer_rank_e2e_sweep_mac.py --device mps --mode train --sweep_mode small

# 2) Mac: same, but also run XAI only for the best run
python influencer_rank_e2e_sweep_mac.py --device mps --mode train --sweep_mode small --xai_on_best 1

# 3) Train+Infer+XAI end-to-end for ONE config (no sweep)
python influencer_rank_e2e_sweep_mac.py --device mps --mode all --max_runs 1 --sweep_mode small

# 4) Infer only from an MLflow run's logged model
python influencer_rank_e2e_sweep_mac.py --device mps --mode infer --load_run_id <RUN_ID>

# 5) XAI only from an MLflow run's logged model
python influencer_rank_e2e_sweep_mac.py --device mps --mode xai --load_run_id <RUN_ID> --xai_top_pred_k 3 --xai_top_pos 3

Notes
-----
- This script expects:
    dataset_A_active_all.csv, hashtags_2017.csv, mentions_2017.csv, influencers.txt
- "No external image CSV": brightness/colorfulness/color_temp_proxy are taken from PREPROCESSED_FILE if present; otherwise zeros.
- Object edges are created only if an object column exists in PREPROCESSED_FILE.
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import datetime
import random
import gc
import itertools
import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Pre-torch args (CUDA_VISIBLE_DEVICES must be set before torch import; harmless on Mac)
# -----------------------------------------------------------------------------
def _parse_pre_torch_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--visible", type=str, default=None, help="(CUDA only) set CUDA_VISIBLE_DEVICES before importing torch.")
    args, _ = p.parse_known_args()
    if args.visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible)
    return args

_PRE_ARGS = _parse_pre_torch_args()
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, coalesce, k_hop_subgraph

from torch.nn import LSTM, Linear, ReLU, Dropout
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

# headless safe
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow


# -----------------------------------------------------------------------------
# Small compatibility helpers
# -----------------------------------------------------------------------------
def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def _mps_pyg_backward_works() -> bool:
    """Quick smoke test: GCNConv forward+backward on MPS.
    Many PyG/torch-scatter ops may not support MPS; if this fails, we fallback to CPU.
    """
    if not _mps_available():
        return False
    try:
        dev = torch.device("mps")
        x = torch.randn(6, 8, device=dev, requires_grad=True)
        edge_index = torch.tensor([[0,1,2,3,4,5, 0,2],
                                   [1,0,3,2,5,4, 2,0]], device=dev, dtype=torch.long)
        conv = GCNConv(8, 16).to(dev)
        y = conv(x, edge_index)
        loss = y.pow(2).mean()
        loss.backward()
        return True
    except Exception:
        return False


def get_device(requested: str) -> torch.device:
    """auto|cpu|mps|<cuda_index> (or 'cuda')"""
    req = str(requested).lower().strip()

    if req in ("cpu",):
        print("[Device] Using CPU")
        return torch.device("cpu")

    if req in ("mps",):
        if _mps_pyg_backward_works():
            print("[Device] Using MPS (PyG OK)")
            return torch.device("mps")
        print("[Device] MPS requested but PyG ops failed -> CPU")
        return torch.device("cpu")

    if req in ("cuda",):
        req = "0"

    if req in ("auto", ""):
        if torch.cuda.is_available():
            print("[Device] Using CUDA (auto)")
            return torch.device("cuda:0")
        if _mps_pyg_backward_works():
            print("[Device] Using MPS (auto, PyG OK)")
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
        if _mps_pyg_backward_works():
            print("[Device] CUDA index given but CUDA unavailable -> MPS (PyG OK)")
            return torch.device("mps")
        print("[Device] CUDA index given but CUDA unavailable -> CPU")
        return torch.device("cpu")

    print(f"[Device] Unknown device='{requested}' -> auto")
    return get_device("auto")


def maybe_empty_cache(device: torch.device):
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# MLflow setup
# -----------------------------------------------------------------------------
def setup_mlflow_experiment(
    experiment_base_name: str,
    tracking_uri: Optional[str],
    local_artifact_dir: str = "mlruns_artifacts",
) -> Tuple[str, str]:
    """
    Local-first MLflow setup.

    - tracking_uri can be sqlite:///mlflow.db (recommended), file:/..., or http(s)://...
    - If existing experiment has artifact_location=mlflow-artifacts:/..., and we are local-file mode,
      create a new experiment with suffix to avoid scheme mismatch.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    active_uri = mlflow.get_tracking_uri()
    is_remote = active_uri.startswith("http://") or active_uri.startswith("https://")

    base_name = experiment_base_name
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

    if (not is_remote) and (exp is not None) and str(exp.artifact_location).startswith("mlflow-artifacts:"):
        exp_name = f"{base_name}_file_{ts}"
        exp = None

    if exp is None:
        if is_remote:
            exp_id = mlflow.create_experiment(exp_name)
        else:
            exp_id = mlflow.create_experiment(exp_name, artifact_location=artifact_dir.as_uri())
    else:
        exp_id = exp.experiment_id

    mlflow.set_experiment(exp_name)

    print(f"[MLflow] tracking_uri={mlflow.get_tracking_uri()}")
    print(f"[MLflow] experiment={exp_name} (id={exp_id})")
    if not is_remote:
        print(f"[MLflow] artifact_root={artifact_dir.as_uri()}")

    return exp_name, exp_id


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------
PREPROCESSED_FILE = "dataset_A_active_all.csv"
HASHTAGS_FILE = "hashtags_2017.csv"
MENTIONS_FILE = "mentions_2017.csv"
INFLUENCERS_FILE = "influencers.txt"


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class GraphsData:
    monthly_graphs: List[Data]
    influencer_indices: List[int]      # global node indices for influencers
    node_to_idx: Dict[str, int]
    idx_to_node: List[str]
    feature_dim: int
    follower_feat_idx: int
    feature_names: List[str]
    static_feature_cols: List[str]
    dynamic_feature_cols: List[str]


# -----------------------------------------------------------------------------
# Data loading / graph building
# -----------------------------------------------------------------------------
def load_influencer_profiles() -> pd.DataFrame:
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


def prepare_graph_data(
    end_date: pd.Timestamp,
    num_months: int = 12,
    metric_numerator: str = "likes",
    metric_denominator: str = "posts",
) -> Optional[GraphsData]:
    """
    Build graph sequence for each month.
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")

    # --- 1. Load Post Data ---
    try:
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=["datetime"], low_memory=False, dtype={"post_id": str})
        print(f"Loaded {len(df_posts)} posts from {PREPROCESSED_FILE}")
        df_posts["username"] = df_posts["username"].astype(str).str.strip()

        # define "active influencers" as those who posted in Dec 2017
        target_month_start = pd.Timestamp("2017-12-01")
        target_month_end = pd.Timestamp("2017-12-31 23:59:59")
        dec_posts = df_posts[(df_posts["datetime"] >= target_month_start) & (df_posts["datetime"] <= target_month_end)]
        valid_users_dec = set(dec_posts["username"].unique())
        print(f"Users who posted in Dec 2017: {len(valid_users_dec):,}")
        if len(valid_users_dec) == 0:
            print("❌ No users found who posted in Dec 2017. Check your datetime parsing/range.")
            return None

        original_count = len(df_posts)
        df_posts = df_posts[df_posts["username"].isin(valid_users_dec)].copy()
        print(f"Filtered posts dataset: {original_count:,} -> {len(df_posts):,} rows")

        # rename common columns
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
                df_posts[col] = pd.to_numeric(df_posts[col], errors="coerce").fillna(0)

    except FileNotFoundError:
        print(f"❌ Error: '{PREPROCESSED_FILE}' not found.")
        return None

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

    # --- 2. Prepare Graph Edges (hashtags / mentions) ---
    try:
        df_hashtags = pd.read_csv(HASHTAGS_FILE)
        df_hashtags.rename(columns={"source": "username", "target": "hashtag"}, inplace=True)
        df_hashtags["datetime"] = pd.to_datetime(df_hashtags["timestamp"], unit="s", errors="coerce")
        df_hashtags["username"] = df_hashtags["username"].astype(str).str.strip()
        df_hashtags = df_hashtags[df_hashtags["username"].isin(valid_users_dec)]
    except Exception:
        df_hashtags = pd.DataFrame(columns=["username", "hashtag", "datetime"])

    try:
        df_mentions = pd.read_csv(MENTIONS_FILE)
        df_mentions.rename(columns={"source": "username", "target": "mention"}, inplace=True)
        df_mentions["datetime"] = pd.to_datetime(df_mentions["timestamp"], unit="s", errors="coerce")
        df_mentions["username"] = df_mentions["username"].astype(str).str.strip()
        df_mentions = df_mentions[df_mentions["username"].isin(valid_users_dec)]
    except Exception:
        df_mentions = pd.DataFrame(columns=["username", "mention", "datetime"])

    # --- 3. Influencer Profiles ---
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
    output_user_file = f"active_influencers_{current_date}.csv"
    print(f"Saving {len(df_influencers)} active influencers to '{output_user_file}'...")
    df_influencers.to_csv(output_user_file, index=False)

    df_posts["month"] = df_posts["datetime"].dt.to_period("M").dt.start_time

    # --- 4. Nodes ---
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
    idx_to_node = list(all_nodes)

    influencer_indices = sorted([node_to_idx[inf] for inf in influencer_set if inf in node_to_idx])

    # --- 5. Static features ---
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

    follower_feat_idx = 0
    try:
        follower_feat_idx = static_feature_cols.index("followers")
        print(f"DEBUG: 'followers' feature is at index {follower_feat_idx} in static features.")
    except ValueError:
        print("Warning: 'followers' not found in static_feature_cols. follower_feat_idx=0 used.")

    # --- 6. Dynamic features ---
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

    # Fill missing post_category/ad
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

    feature_columns = static_feature_cols + dynamic_feature_cols
    feature_dim = len(feature_columns)
    feature_names = list(feature_columns)
    print(f"Total feature dimension: {feature_dim}")

    # --- 7. Construct graphs per month ---
    monthly_graphs: List[Data] = []
    start_date = end_date - pd.DateOffset(months=num_months - 1)

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
        # self-loops for influencers
        all_edges += [(idx, idx) for idx in influencer_indices]
        all_edges = list(set(all_edges))
        if not all_edges:
            all_edges = [(idx, idx) for idx in influencer_indices]

        num_nodes = len(all_nodes)
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index = coalesce(edge_index, num_nodes=num_nodes)

        # node features
        current_dynamic = dynamic_features[dynamic_features["month"] == snapshot_month]
        snapshot_features = pd.merge(static_features, current_dynamic, on="username", how="left")
        snapshot_features = snapshot_features[feature_columns].fillna(0)
        x = torch.tensor(snapshot_features.astype(float).values, dtype=torch.float)

        # Target y per month (engagement)
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

    return GraphsData(
        monthly_graphs=monthly_graphs,
        influencer_indices=influencer_indices,
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        feature_dim=feature_dim,
        follower_feat_idx=follower_feat_idx,
        feature_names=feature_names,
        static_feature_cols=static_feature_cols,
        dynamic_feature_cols=dynamic_feature_cols,
    )


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = int(num_layers)
        self.convs = nn.ModuleList(
            [GCNConv(in_channels, hidden_channels)]
            + [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

    def forward(self, x, edge_index, edge_weight=None):
        outs = []
        h = x
        for conv in self.convs:
            if edge_weight is None:
                h = conv(h, edge_index)
            else:
                try:
                    h = conv(h, edge_index, edge_weight=edge_weight)
                except TypeError:
                    h = conv(h, edge_index)
            h = F.relu(h)
            outs.append(h)
        return torch.cat(outs, dim=1)


class AttentiveRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention_layer = Linear(hidden_dim, 1)

    def forward(self, sequence_of_embeddings):
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_scores = torch.tanh(self.attention_layer(rnn_out))
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        return context_vector, attention_weights


class HardResidualInfluencerModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        gcn_dim: int,
        rnn_dim: int,
        num_gcn_layers: int = 2,
        dropout_prob: float = 0.5,
        projection_dim: int = 64,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.projection_dim = int(projection_dim)
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


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
class ListMLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_scores, true_scores):
        batch_size, list_size = pred_scores.size()
        sorted_indices = torch.argsort(true_scores, dim=1, descending=True)
        batch_idx = torch.arange(batch_size, device=pred_scores.device).unsqueeze(1)
        sorted_preds = pred_scores[batch_idx, sorted_indices]
        max_val, _ = sorted_preds.max(dim=1, keepdim=True)
        sorted_preds_exp = torch.exp(sorted_preds - max_val)
        cum_sum = torch.flip(torch.cumsum(torch.flip(sorted_preds_exp, dims=[1]), dim=1), dims=[1])
        log_cum_sum = torch.log(cum_sum + 1e-10)
        log_likelihood = (sorted_preds - max_val) - log_cum_sum
        loss = -torch.sum(log_likelihood, dim=1)
        return loss.mean()


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_attention_weights(attention_matrix: np.ndarray, run_name: str) -> Tuple[str, str, str, str]:
    """
    Plot attention weights + export numeric alpha.
    Returns: (bar_png, heat_png, mean_csv, raw_npz)
    """
    attention_matrix = np.asarray(attention_matrix)
    if attention_matrix.ndim == 3 and attention_matrix.shape[-1] == 1:
        attention_matrix = attention_matrix[..., 0]
    if attention_matrix.ndim == 1:
        attention_matrix = attention_matrix.reshape(1, -1)

    mean_att = np.mean(attention_matrix, axis=0)
    time_steps = np.arange(len(mean_att))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(time_steps, mean_att, edgecolor="black", alpha=0.7)
    plt.xlabel("Time Steps (Months)")
    plt.ylabel("Average Attention Weight")
    plt.title(f"Average Attention Weights across Time\nRun: {run_name}")
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

    plt.figure(figsize=(12, 8))
    subset = attention_matrix[:50, :]
    plt.imshow(subset, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Attention Weight")
    plt.xlabel("Time Steps (Oldest -> Newest)")
    plt.ylabel("Sample Users (Top 50)")
    plt.title("Attention Weights Heatmap (Individual)")
    plt.xticks(time_steps, labels, rotation=0)
    filename_heat = f"attention_weights_heatmap_{run_name}.png"
    plt.savefig(filename_heat, bbox_inches="tight")
    plt.close()

    labels2 = [f"T-{len(mean_att)-1-i}" for i in range(len(mean_att))]
    if len(labels2) > 0:
        labels2[-1] = "Current (T)"
    df_mean = pd.DataFrame({"pos": np.arange(len(mean_att), dtype=int), "label": labels2, "alpha_mean": mean_att.astype(float)})
    filename_csv = f"attention_weights_mean_{run_name}.csv"
    df_mean.to_csv(filename_csv, index=False, float_format="%.8e")

    filename_raw = f"attention_weights_raw_{run_name}.npz"
    np.savez_compressed(filename_raw, attention=attention_matrix)
    return filename_bar, filename_heat, filename_csv, filename_raw


def generate_scatter_plot(
    x_data,
    y_data,
    x_label,
    y_label,
    filename,
    title=None,
    color_data=None,
    color_label=None,
):
    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if color_data is not None:
        c = np.asarray(color_data, dtype=float)
        mask = mask & np.isfinite(c)
    else:
        c = None

    x = x[mask]
    y = y[mask]
    if c is not None:
        c = c[mask]

    if x.size == 0:
        return None

    plt.figure(figsize=(10, 8))
    if c is None:
        plt.scatter(x, y, s=10)
    else:
        sc = plt.scatter(x, y, s=10, c=c, cmap="viridis", alpha=0.7)
        cb = plt.colorbar(sc)
        cb.set_label(color_label or "color")

    mn = float(np.nanmin([x.min(), y.min()]))
    mx = float(np.nanmax([x.max(), y.max()]))
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=2)

    if x.size > 1:
        p_corr, _ = pearsonr(x, y)
        s_corr, _ = spearmanr(x, y)
        corr_text = f"Pearson={p_corr:.4f}, Spearman={s_corr:.4f}"
    else:
        corr_text = "corr=N/A"

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title or f"{y_label} vs {x_label} ({corr_text})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    return filename


# -----------------------------------------------------------------------------
# Dataset helper
# -----------------------------------------------------------------------------
def get_dataset_with_baseline(monthly_graphs: List[Data], influencer_indices: List[int], target_idx: int = -1):
    num_graphs = len(monthly_graphs)
    positive_target_idx = (target_idx + num_graphs) % num_graphs
    if positive_target_idx == 0:
        raise ValueError("Cannot use target_idx that points to the first month (no history).")

    target_graph = monthly_graphs[positive_target_idx]
    baseline_graph = monthly_graphs[positive_target_idx - 1]

    target_y = target_graph.y[influencer_indices].squeeze()
    baseline_y = baseline_graph.y[influencer_indices].squeeze()

    local_idx = torch.arange(len(influencer_indices), dtype=torch.long)
    return TensorDataset(local_idx, target_y, baseline_y)


# -----------------------------------------------------------------------------
# MaskOpt E2E Explainer (same idea as your previous code, but edge names are real)
# -----------------------------------------------------------------------------
def _binary_entropy(p, eps=1e-12):
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))


def _gcn_forward_concat(gcn_encoder: GCNEncoder, x, edge_index, edge_weight=None):
    layer_outputs = []
    h = x
    for conv in gcn_encoder.convs:
        if edge_weight is None:
            h = conv(h, edge_index)
        else:
            try:
                h = conv(h, edge_index, edge_weight=edge_weight)
            except TypeError:
                h = conv(h, edge_index)
        h = F.relu(h)
        layer_outputs.append(h)
    return torch.cat(layer_outputs, dim=1)


class E2EMaskOptWrapper(nn.Module):
    def __init__(
        self,
        model: HardResidualInfluencerModel,
        input_graphs: List[Data],
        target_node_idx: int,
        explain_pos: int,
        device: torch.device,
        idx_to_node: List[str],
        use_subgraph: bool = True,
        num_hops: int = 2,
        undirected: bool = True,
        feat_mask_scope: str = "target",
        edge_mask_scope: str = "incident",
    ):
        super().__init__()
        self.model = model
        self.input_graphs = input_graphs
        self.T = len(input_graphs)
        self.target_global = int(target_node_idx)
        self.explain_pos = int(explain_pos)
        self.device = device
        self.idx_to_node = idx_to_node

        self.use_subgraph = bool(use_subgraph)
        self.num_hops = int(num_hops)
        self.undirected = bool(undirected)
        self.feat_mask_scope = str(feat_mask_scope)
        self.edge_mask_scope = str(edge_mask_scope)

        self.cached_proj = [None] * self.T
        self.cached_gcn = [None] * self.T

        self.edge_param_names: List[str] = []

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
        g = self.input_graphs[self.explain_pos].to(self.device)
        x_full = g.x
        ei_full = g.edge_index

        if (not self.use_subgraph) or (ei_full.numel() == 0):
            self.x_exp = x_full
            self.ei_exp = ei_full
            self.target_local = self.target_global
            self.local2global = None
        else:
            subset, ei_sub, mapping, _ = k_hop_subgraph(
                self.target_global, self.num_hops, ei_full, relabel_nodes=True, num_nodes=x_full.size(0)
            )
            x_sub = x_full[subset]
            if self.undirected and ei_sub.numel() > 0:
                ei_sub = to_undirected(ei_sub, num_nodes=x_sub.size(0))
                ei_sub = coalesce(ei_sub, num_nodes=x_sub.size(0))
            self.x_exp = x_sub
            self.ei_exp = ei_sub
            self.target_local = int(mapping.item())
            self.local2global = subset

        if self.ei_exp.numel() == 0:
            self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            src, dst = self.ei_exp
            incident = (src == self.target_local) | (dst == self.target_local)
            self.incident_edge_idx = torch.where(incident)[0]

        if self.edge_mask_scope == "incident":
            self.num_edge_params = int(self.incident_edge_idx.numel())
            # build names aligned to edge_gate parameter order
            names = []
            for k in range(self.num_edge_params):
                eidx = int(self.incident_edge_idx[k].item())
                sL = int(self.ei_exp[0, eidx].item())
                dL = int(self.ei_exp[1, eidx].item())
                if self.local2global is not None:
                    sG = int(self.local2global[sL].item())
                    dG = int(self.local2global[dL].item())
                else:
                    sG, dG = sL, dL
                sN = self.idx_to_node[sG] if 0 <= sG < len(self.idx_to_node) else f"node_{sG}"
                dN = self.idx_to_node[dG] if 0 <= dG < len(self.idx_to_node) else f"node_{dG}"
                names.append(f"{sN} -> {dN}")
            self.edge_param_names = names
        else:
            self.num_edge_params = int(self.ei_exp.size(1))
            names = []
            for eidx in range(self.num_edge_params):
                sL = int(self.ei_exp[0, eidx].item())
                dL = int(self.ei_exp[1, eidx].item())
                if self.local2global is not None:
                    sG = int(self.local2global[sL].item())
                    dG = int(self.local2global[dL].item())
                else:
                    sG, dG = sL, dL
                sN = self.idx_to_node[sG] if 0 <= sG < len(self.idx_to_node) else f"node_{sG}"
                dN = self.idx_to_node[dG] if 0 <= dG < len(self.idx_to_node) else f"node_{dG}"
                names.append(f"{sN} -> {dN}")
            self.edge_param_names = names

        self.feature_dim = int(self.x_exp.size(1))

    def num_mask_params(self) -> Tuple[int, int]:
        return self.feature_dim, self.num_edge_params

    def _apply_feature_gate(self, x, feat_gate):
        if self.feat_mask_scope in ("all", "subgraph"):
            return x * feat_gate.view(1, -1)

        n = x.size(0)
        sel = F.one_hot(torch.tensor(self.target_local, device=x.device), num_classes=n).to(x.dtype).unsqueeze(1)
        return x + sel * x * (feat_gate.view(1, -1) - 1.0)

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
    def original_pred(self) -> float:
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
    model: HardResidualInfluencerModel,
    input_graphs: List[Data],
    idx_to_node: List[str],
    target_node_idx: int,
    explain_pos: int,
    feature_names: List[str],
    device: torch.device,
    use_subgraph: bool = True,
    num_hops: int = 2,
    undirected: bool = True,
    feat_mask_scope: str = "target",
    edge_mask_scope: str = "incident",
    epochs: int = 300,
    lr: float = 0.05,
    coeffs: Optional[dict] = None,
    print_every: int = 50,
    topk_feat: int = 20,
    topk_edge: int = 30,
    min_show: float = 1e-6,
    disable_cudnn_rnn: bool = True,
    mlflow_log: bool = True,
    fid_weight: float = 100.0,
    tag: str = "pos_0",
    impact_reference: str = "masked",  # "masked" | "unmasked" | "both"
    eps_abs_feat: float = 1e-9,
    eps_rel_feat: float = 1e-6,
    eps_abs_edge: float = 1e-9,
    eps_rel_edge: float = 1e-6,
):
    assert len(input_graphs) >= 2, "input_graphs length must be >= 2"
    T = len(input_graphs)
    if explain_pos < 0:
        explain_pos = (explain_pos + T) % T

    if coeffs is None:
        coeffs = {"edge_size": 0.05, "edge_ent": 0.10, "node_feat_size": 0.02, "node_feat_ent": 0.10}

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
        idx_to_node=idx_to_node,
        use_subgraph=use_subgraph,
        num_hops=num_hops,
        undirected=undirected,
        feat_mask_scope=feat_mask_scope,
        edge_mask_scope=edge_mask_scope,
    )

    Fdim, Edim = wrapper.num_mask_params()

    feat_logits = nn.Parameter(0.1 * torch.randn(Fdim, device=device))
    edge_logits = nn.Parameter(0.1 * torch.randn(Edim, device=device)) if Edim > 0 else None
    mask_params = [feat_logits] + ([edge_logits] if edge_logits is not None else [])
    opt = torch.optim.Adam(mask_params, lr=lr)

    orig = float(wrapper.original_pred())
    orig_t = torch.tensor(orig, device=device)

    print(f"🧠 [MaskOpt] target_node={int(target_node_idx)} explain_pos={explain_pos}/{T-1} orig={orig:.6f}")
    print(f"   use_subgraph={use_subgraph}, num_hops={num_hops}, undirected={undirected}, feat_dim={Fdim}, edge_params={Edim}")

    cudnn_ctx = _DisableCudnn() if disable_cudnn_rnn else None

    best = {"loss": float("inf"), "feat": None, "edge": None, "pred": None}

    def _thr(pred_base, eps_abs, eps_rel):
        return max(float(eps_abs), float(eps_rel) * abs(float(pred_base)))

    def _direction(diff, pred_base, eps_abs, eps_rel):
        th = _thr(pred_base, eps_abs, eps_rel)
        if abs(diff) <= th:
            return "Zero (0)"
        return "Positive (+)" if diff > 0 else "Negative (-)"

    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)

        feat_gate = torch.sigmoid(feat_logits)
        edge_gate = torch.sigmoid(edge_logits) if (Edim > 0 and edge_logits is not None) else None

        try:
            if cudnn_ctx is not None:
                with cudnn_ctx:
                    pred = wrapper.predict_with_gates(feat_gate, edge_gate)
            else:
                pred = wrapper.predict_with_gates(feat_gate, edge_gate)
        except Exception as e:
            raise RuntimeError(f"MaskOpt forward failed (pos={explain_pos}): {e}")

        loss_fid = (pred - orig_t) ** 2
        loss_feat_size = feat_gate.mean()
        loss_feat_ent = _binary_entropy(feat_gate).mean()

        if edge_gate is not None and edge_gate.numel() > 0:
            loss_edge_size = edge_gate.mean()
            loss_edge_ent = _binary_entropy(edge_gate).mean()
        else:
            loss_edge_size = pred.new_zeros(())
            loss_edge_ent = pred.new_zeros(())

        loss = (
            float(fid_weight) * loss_fid
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
                f"pred={float(pred.item()):.6f} feat_max={feat_max:.4f} edge_max={edge_max:.4f}"
            )

    feat_gate = best["feat"].clamp(0.0, 1.0) if best["feat"] is not None else None
    edge_gate = best["edge"].clamp(0.0, 1.0) if best["edge"] is not None else None

    ones_feat = torch.ones(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None

    with torch.no_grad():
        if cudnn_ctx is not None:
            with cudnn_ctx:
                pred_unmasked = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())
                pred_masked = float(wrapper.predict_with_gates(feat_gate, edge_gate).item()) if (feat_gate is not None) else pred_unmasked
        else:
            pred_unmasked = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())
            pred_masked = float(wrapper.predict_with_gates(feat_gate, edge_gate).item()) if (feat_gate is not None) else pred_unmasked

    # Feature rows
    feat_rows = []
    df_feat = pd.DataFrame()
    if feat_gate is not None and feat_gate.numel() > 0:
        feat_np = feat_gate.detach().cpu().numpy()
        top_feat_idx = np.argsort(feat_np)[::-1][: int(topk_feat)]
        for j in top_feat_idx:
            imp = float(feat_np[j])
            if imp < float(min_show):
                continue

            refs = []
            if impact_reference in ("unmasked", "both"):
                base_f = ones_feat.clone()
                base_e = ones_edge.clone() if ones_edge is not None else None
                ab_f = base_f.clone()
                ab_f[j] = 0.0
                with torch.no_grad():
                    if cudnn_ctx is not None:
                        with cudnn_ctx:
                            pred_abl = float(wrapper.predict_with_gates(ab_f, base_e).item())
                    else:
                        pred_abl = float(wrapper.predict_with_gates(ab_f, base_e).item())
                diff = pred_unmasked - pred_abl
                refs.append(("unmasked", diff, _direction(diff, pred_unmasked, eps_abs_feat, eps_rel_feat)))

            if impact_reference in ("masked", "both"):
                base_f = feat_gate.clone()
                base_e = edge_gate.clone() if edge_gate is not None else None
                ab_f = base_f.clone()
                ab_f[j] = 0.0
                with torch.no_grad():
                    if cudnn_ctx is not None:
                        with cudnn_ctx:
                            pred_abl = float(wrapper.predict_with_gates(ab_f, base_e).item())
                    else:
                        pred_abl = float(wrapper.predict_with_gates(ab_f, base_e).item())
                diff = pred_masked - pred_abl
                refs.append(("masked", diff, _direction(diff, pred_masked, eps_abs_feat, eps_rel_feat)))

            name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
            row = {"Type": "Feature", "Name": name, "Importance": imp}
            for key, diff, direc in refs:
                row[f"Score_Impact({key})"] = float(diff)
                row[f"Direction({key})"] = direc
            feat_rows.append(row)

        df_feat = pd.DataFrame(feat_rows)
        if not df_feat.empty:
            df_feat = df_feat.sort_values("Importance", ascending=False).reset_index(drop=True)

    # Edge rows
    edge_rows = []
    df_edge = pd.DataFrame()
    if edge_gate is not None and edge_gate.numel() > 0:
        edge_np = edge_gate.detach().cpu().numpy()
        top_edge_idx = np.argsort(edge_np)[::-1][: int(topk_edge)]
        for gidx in top_edge_idx:
            imp = float(edge_np[gidx])
            if imp < float(min_show):
                continue

            refs = []
            if impact_reference in ("unmasked", "both"):
                base_f = ones_feat.clone()
                base_e = ones_edge.clone() if ones_edge is not None else None
                ab_e = base_e.clone() if base_e is not None else None
                if ab_e is not None:
                    ab_e[gidx] = 0.0
                with torch.no_grad():
                    if cudnn_ctx is not None:
                        with cudnn_ctx:
                            pred_abl = float(wrapper.predict_with_gates(base_f, ab_e).item())
                    else:
                        pred_abl = float(wrapper.predict_with_gates(base_f, ab_e).item())
                diff = pred_unmasked - pred_abl
                refs.append(("unmasked", diff, _direction(diff, pred_unmasked, eps_abs_edge, eps_rel_edge)))

            if impact_reference in ("masked", "both"):
                base_f = feat_gate.clone()
                base_e = edge_gate.clone()
                ab_e = base_e.clone()
                ab_e[gidx] = 0.0
                with torch.no_grad():
                    if cudnn_ctx is not None:
                        with cudnn_ctx:
                            pred_abl = float(wrapper.predict_with_gates(base_f, ab_e).item())
                    else:
                        pred_abl = float(wrapper.predict_with_gates(base_f, ab_e).item())
                diff = pred_masked - pred_abl
                refs.append(("masked", diff, _direction(diff, pred_masked, eps_abs_edge, eps_rel_edge)))

            nm = wrapper.edge_param_names[gidx] if gidx < len(wrapper.edge_param_names) else f"edge_{gidx}"
            row = {"Type": "Edge", "Name": nm, "Importance": imp}
            for key, diff, direc in refs:
                row[f"Score_Impact({key})"] = float(diff)
                row[f"Direction({key})"] = direc
            edge_rows.append(row)

        df_edge = pd.DataFrame(edge_rows)
        if not df_edge.empty:
            df_edge = df_edge.sort_values("Importance", ascending=False).reset_index(drop=True)

    # MLflow logging
    if mlflow_log and mlflow.active_run() is not None:
        try:
            feat_csv = f"maskopt_feat_{tag}_node_{int(target_node_idx)}_pos_{explain_pos}.csv"
            edge_csv = f"maskopt_edge_{tag}_node_{int(target_node_idx)}_pos_{explain_pos}.csv"
            df_feat.to_csv(feat_csv, index=False, float_format="%.8e")
            df_edge.to_csv(edge_csv, index=False, float_format="%.8e")
            mlflow.log_artifact(feat_csv, artifact_path="xai/tables")
            mlflow.log_artifact(edge_csv, artifact_path="xai/tables")
            os.remove(feat_csv)
            os.remove(edge_csv)
        except Exception as e:
            print(f"⚠️ MLflow log failed (tables): {e}")

    meta = {
        "orig_pred": float(orig),
        "best_pred": float(best["pred"]) if best["pred"] is not None else None,
        "best_loss": float(best["loss"]),
        "target_node": int(target_node_idx),
        "explain_pos": int(explain_pos),
        "T": int(T),
        "feat_dim": int(Fdim),
        "edge_params": int(Edim),
        "pred_unmasked": float(pred_unmasked),
        "pred_masked": float(pred_masked),
        "impact_reference": str(impact_reference),
        "coeffs": dict(coeffs),
        "fid_weight": float(fid_weight),
    }

    if mlflow_log and mlflow.active_run() is not None:
        try:
            mpath = f"maskopt_meta_{tag}_node_{int(target_node_idx)}_pos_{explain_pos}.json"
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(mpath, artifact_path="xai/meta")
            os.remove(mpath)
        except Exception as e:
            print(f"⚠️ MLflow log failed (meta): {e}")

    return df_feat, df_edge, meta


def compute_time_step_sensitivity(
    model: HardResidualInfluencerModel,
    input_graphs: List[Data],
    target_node_idx: int,
    device: torch.device,
    topk: int = 3,
    min_delta: float = 1e-6,
):
    """Drop each time step and measure delta, also export alpha."""
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
            score = float(alpha[t]) * float(delta)
            rows.append({"pos": int(t), "alpha": float(alpha[t]), "pred_full": pred_full, "pred_drop": pred_drop, "delta_total": float(delta), "score": float(score)})

    sens_df = pd.DataFrame(rows).sort_values(["score", "delta_total", "alpha"], ascending=False).reset_index(drop=True)
    selected_positions = sens_df.head(int(topk))["pos"].astype(int).tolist()
    return sens_df, selected_positions, pred_full, alpha


def mlflow_log_maskopt_plots(df_feat, df_edge, tag="pos_0", topk_feat=15, topk_edge=15, artifact_path="xai/plots", fname_prefix="maskopt"):
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


# -----------------------------------------------------------------------------
# Core train / infer
# -----------------------------------------------------------------------------
@dataclass
class RunResult:
    run_id: str
    run_name: str
    metrics: Dict[str, float]
    params: Dict[str, object]


def train_and_eval_one_run(
    graphs: GraphsData,
    params: Dict[str, object],
    device: torch.device,
    exp_id: str,
    do_train: bool = True,
    do_infer: bool = True,
    do_xai: bool = False,
    xai_top_pred_k: int = 3,
    xai_top_pos: int = 3,
    seed: int = 42,
    cache_seq_on_cpu: bool = True,
) -> RunResult:
    """One MLflow run: (train optional) + eval plots + (xai optional)."""
    set_seeds(seed)

    # consistent run name
    run_name = str(params.get("RUN_NAME", f"Run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"))

    with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        mlflow.log_param("DEVICE_USED", str(device))
        mlflow.log_param("cache_seq_on_cpu", int(bool(cache_seq_on_cpu)))
        mlflow.log_param("seed", int(seed))

        model = HardResidualInfluencerModel(
            feature_dim=graphs.feature_dim,
            gcn_dim=int(params["GCN_DIM"]),
            rnn_dim=int(params["RNN_DIM"]),
            num_gcn_layers=int(params["NUM_GCN_LAYERS"]),
            dropout_prob=float(params["DROPOUT_PROB"]),
            projection_dim=int(params["PROJECTION_DIM"]),
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(params["LR"]),
            weight_decay=float(params.get("WEIGHT_DECAY", 0.0)),
        )
        criterion_list = ListMLELoss().to(device)
        criterion_mse = nn.MSELoss().to(device)

        # ---------------- Train ----------------
        if do_train:
            train_dataset = get_dataset_with_baseline(graphs.monthly_graphs, graphs.influencer_indices, target_idx=-2)

            sampler = None
            if bool(params.get("USE_SAMPLER", False)):
                targets_for_weight = train_dataset.tensors[1].cpu().numpy()
                weights = [5.0 if float(t) > 0.01 else 1.0 for t in targets_for_weight]
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

            list_size = int(params["LIST_SIZE"])
            batch_size = min(int(params["BATCH_SIZE"]), len(train_dataset))
            batch_size = (batch_size // list_size) * list_size
            if batch_size <= 0:
                batch_size = list_size
            if batch_size > len(train_dataset):
                batch_size = (len(train_dataset) // list_size) * list_size
                batch_size = max(batch_size, list_size)

            dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                drop_last=True,
            )

            train_input_graphs = graphs.monthly_graphs[:-2]

            print(f"\n🚀 [Train] run_id={run_id} name={run_name}")
            print(f"[Train] epochs={int(params['EPOCHS'])}, list_size={list_size}, batch_size={batch_size}, N_inf={len(graphs.influencer_indices)}")

            model.train()

            for epoch in range(int(params["EPOCHS"])):
                t0 = time.time()
                model.train()
                optimizer.zero_grad(set_to_none=True)

                # Build temporal sequences for influencers
                inf_idx = torch.as_tensor(graphs.influencer_indices, dtype=torch.long, device=device)

                seq_emb_list = []
                raw_emb_list = []

                for g in train_input_graphs:
                    g = g.to(device)
                    p_x = model.projection_layer(g.x)
                    gcn_out = model.gcn_encoder(p_x, g.edge_index)

                    # Keep only influencers
                    raw_inf = p_x.index_select(0, inf_idx).detach()
                    gcn_inf = gcn_out.index_select(0, inf_idx).detach()

                    if cache_seq_on_cpu and device.type != "cuda":
                        raw_inf = raw_inf.cpu()
                        gcn_inf = gcn_inf.cpu()

                    raw_emb_list.append(raw_inf)
                    seq_emb_list.append(gcn_inf)

                    del p_x, gcn_out, g
                    maybe_empty_cache(device)

                # stack (prefer CPU cache on Mac)
                # seq_emb_list: list[T] of [N_inf, D] -> [N_inf, T, D]
                full_seq = torch.stack(seq_emb_list, dim=0).permute(1, 0, 2)
                # raw_emb_list: list[T] of [N_inf, P] -> [T, N_inf, P]
                full_raw_Tfirst = torch.stack(raw_emb_list, dim=0)

                total_loss = 0.0
                num_batches = 0
                loss_sum = None

                for (b_idx, b_target, b_baseline) in dataloader:
                    b_target = b_target.to(device)
                    b_baseline = b_baseline.to(device)

                    # gather and move to device if cached on cpu
                    b_seq = full_seq[b_idx]
                    b_raw = full_raw_Tfirst[:, b_idx, :].permute(1, 0, 2)
                    if cache_seq_on_cpu and device.type != "cuda":
                        b_seq = b_seq.to(device)
                        b_raw = b_raw.to(device)

                    preds, _ = model(b_seq, b_raw, baseline_scores=b_baseline)
                    preds = preds.view(-1)

                    log_target = torch.log1p(b_target * 100.0)
                    log_pred = torch.log1p(preds * 100.0)

                    loss_rank = criterion_list(
                        preds.view(-1, list_size),
                        log_target.view(-1, list_size),
                    )
                    loss_mse = criterion_mse(log_pred, log_target)
                    loss = loss_rank + loss_mse * float(params.get("POINTWISE_LOSS_WEIGHT", 0.3))

                    total_loss += float(loss.item())
                    num_batches += 1
                    loss_sum = loss if loss_sum is None else (loss_sum + loss)

                if loss_sum is not None:
                    (loss_sum / float(max(1, num_batches))).backward()

                # optional grad clip
                grad_clip = float(params.get("GRAD_CLIP", 0.0) or 0.0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

                # cleanup
                del full_seq, full_raw_Tfirst, seq_emb_list, raw_emb_list, inf_idx
                maybe_empty_cache(device)
                gc.collect()

                if (epoch + 1) % 10 == 0 or (epoch + 1) == int(params["EPOCHS"]):
                    avg_loss = total_loss / max(1, num_batches)
                    mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
                    mlflow.log_metric("epoch_time_sec", time.time() - t0, step=epoch + 1)
                    print(f"[Train] epoch {epoch+1:3d}/{int(params['EPOCHS'])} loss={avg_loss:.6f} time={time.time()-t0:.1f}s")

        # save model state_dict as artifact
        model_path_local = None
        try:
            out_dir = Path("trained_models")
            out_dir.mkdir(parents=True, exist_ok=True)
            model_path_local = out_dir / f"model_state_dict_{run_id}.pth"
            torch.save(model.state_dict(), model_path_local.as_posix())
            mlflow.log_artifact(model_path_local.as_posix(), artifact_path="model")
            mlflow.log_param("model_state_dict_path", model_path_local.as_posix())
            print(f"[MLflow] ✅ logged model state_dict: model/{model_path_local.name}")
        except Exception as e:
            print(f"[MLflow] model artifact logging skipped: {e}")

        # ---------------- Infer / Eval ----------------
        metrics = {}
        predicted_scores = None
        attention_matrix = None
        true_scores = None
        baseline_scores = None

        if do_infer:
            print("\n🔎 [Infer] computing Dec prediction + metrics + scatter plots...")
            model.eval()

            test_dataset = get_dataset_with_baseline(graphs.monthly_graphs, graphs.influencer_indices, target_idx=-1)
            all_indices = test_dataset.tensors[0]
            all_targets = test_dataset.tensors[1]
            all_baselines = test_dataset.tensors[2]

            inf_input_graphs = graphs.monthly_graphs[:-1]
            inf_idx = torch.as_tensor(graphs.influencer_indices, dtype=torch.long)

            # Build influencer sequences (cache on cpu, then batch to device)
            with torch.no_grad():
                seq_emb_l = []
                raw_emb_l = []
                for g in inf_input_graphs:
                    g = g.to(device)
                    p_x = model.projection_layer(g.x)
                    gcn_out = model.gcn_encoder(p_x, g.edge_index)
                    raw_inf = p_x.index_select(0, inf_idx.to(device)).detach()
                    gcn_inf = gcn_out.index_select(0, inf_idx.to(device)).detach()
                    # cache on cpu for mac
                    if cache_seq_on_cpu and device.type != "cuda":
                        raw_inf = raw_inf.cpu()
                        gcn_inf = gcn_inf.cpu()
                    raw_emb_l.append(raw_inf)
                    seq_emb_l.append(gcn_inf)
                    del p_x, gcn_out, g
                    maybe_empty_cache(device)

                f_seq = torch.stack(seq_emb_l, dim=0).permute(1, 0, 2)               # [N_inf, T, D]
                f_raw = torch.stack(raw_emb_l, dim=0).permute(1, 0, 2)               # [N_inf, T, P]

                preds_all = []
                attn_all = []
                infer_batch_size = int(params.get("INFER_BATCH", 1024))
                for i0 in range(0, len(all_indices), infer_batch_size):
                    i1 = min(i0 + infer_batch_size, len(all_indices))
                    b_seq = f_seq[i0:i1]
                    b_raw = f_raw[i0:i1]
                    if cache_seq_on_cpu and device.type != "cuda":
                        b_seq = b_seq.to(device)
                        b_raw = b_raw.to(device)
                    b_base = all_baselines[i0:i1].to(device)

                    p, attn = model(b_seq, b_raw, b_base)
                    preds_all.append(p.detach().cpu())
                    attn_all.append(attn.detach().cpu() if attn is not None else torch.zeros((p.shape[0], f_seq.shape[1], 1)))

                predicted_scores = torch.cat(preds_all).squeeze().numpy()
                attention_matrix = torch.cat(attn_all).squeeze().numpy()
                if attention_matrix.ndim == 1:
                    attention_matrix = attention_matrix.reshape(1, -1)

                true_scores = all_targets.cpu().numpy()
                baseline_scores = all_baselines.cpu().numpy()

            # metrics
            mae = float(mean_absolute_error(true_scores, predicted_scores))
            rmse = float(np.sqrt(mean_squared_error(true_scores, predicted_scores)))
            p_corr = float(pearsonr(true_scores, predicted_scores)[0]) if len(true_scores) > 1 else 0.0
            s_corr = float(spearmanr(true_scores, predicted_scores)[0]) if len(true_scores) > 1 else 0.0

            metrics = {"mae": mae, "rmse": rmse, "pearson_corr": p_corr, "spearman_corr": s_corr}
            mlflow.log_metrics(metrics)
            print(f"[Eval] MAE={mae:.6f}, RMSE={rmse:.6f}, Pearson={p_corr:.4f}, Spearman={s_corr:.4f}")

            # plots + artifacts
            tmpd = Path(tempfile.mkdtemp(prefix="irank_"))
            try:
                # scatter
                sc1 = tmpd / "pred_vs_true.png"
                generate_scatter_plot(true_scores, predicted_scores, "True", "Pred", sc1.as_posix(), title="Pred vs True")
                mlflow.log_artifact(sc1.as_posix(), artifact_path="plots")

                # attention plots
                att_bar, att_heat, att_csv, att_npz = plot_attention_weights(attention_matrix, run_name=run_name)
                for f in [att_bar, att_heat, att_csv, att_npz]:
                    if f and os.path.exists(f):
                        mlflow.log_artifact(f, artifact_path="plots/attention")
                        try:
                            os.remove(f)
                        except Exception:
                            pass

                # optional colored scatter (followers)
                last_graph = graphs.monthly_graphs[-2]  # last input month for prediction
                inf_global = torch.tensor(graphs.influencer_indices, dtype=torch.long)
                follower_counts = last_graph.x[inf_global, graphs.follower_feat_idx].cpu().numpy()
                log_followers = np.log1p(follower_counts)

                sc2 = tmpd / "pred_vs_true_by_followers.png"
                generate_scatter_plot(true_scores, predicted_scores, "True", "Pred", sc2.as_posix(),
                                     title="Pred vs True (colored by log followers)", color_data=log_followers, color_label="log(1+followers)")
                mlflow.log_artifact(sc2.as_posix(), artifact_path="plots")

                # prediction table
                pred_csv = tmpd / "predictions.csv"
                df_pred = pd.DataFrame({
                    "influencer_global_idx": graphs.influencer_indices,
                    "username": [graphs.idx_to_node[i] for i in graphs.influencer_indices],
                    "true_score": true_scores,
                    "pred_score": predicted_scores,
                    "baseline_score": baseline_scores,
                })
                df_pred.to_csv(pred_csv, index=False)
                mlflow.log_artifact(pred_csv.as_posix(), artifact_path="predictions")
            finally:
                try:
                    for p in tmpd.glob("*"):
                        p.unlink(missing_ok=True)
                    tmpd.rmdir()
                except Exception:
                    pass

        # ---------------- XAI ----------------
        if do_xai:
            if predicted_scores is None or attention_matrix is None:
                print("[XAI] skipped: need inference outputs. Run with --mode all or --mode infer+xai.")
            else:
                print("\n🧩 [XAI] Running MaskOpt E2E explanations (best to do only for best run).")
                # choose top predicted influencers
                k = max(1, int(xai_top_pred_k))
                top_idx_local = np.argsort(predicted_scores)[::-1][:k].tolist()
                target_globals = [graphs.influencer_indices[i] for i in top_idx_local]
                mlflow.log_param("xai_target_globals", ",".join(map(str, target_globals)))

                # choose time steps per node by sensitivity
                for rank, tg in enumerate(target_globals):
                    try:
                        sens_df, sel_pos, pred_full, alpha = compute_time_step_sensitivity(
                            model=model,
                            input_graphs=graphs.monthly_graphs[:-1],
                            target_node_idx=int(tg),
                            device=device,
                            topk=int(xai_top_pos),
                        )
                        # log sensitivity table
                        sens_csv = f"xai_sensitivity_node_{tg}.csv"
                        sens_df.to_csv(sens_csv, index=False)
                        mlflow.log_artifact(sens_csv, artifact_path="xai/sensitivity")
                        os.remove(sens_csv)

                        for ppos in sel_pos:
                            tag = f"top{rank}_pos{ppos}"
                            df_feat, df_edge, meta = maskopt_e2e_explain(
                                model=model,
                                input_graphs=graphs.monthly_graphs[:-1],
                                idx_to_node=graphs.idx_to_node,
                                target_node_idx=int(tg),
                                explain_pos=int(ppos),
                                feature_names=graphs.feature_names,
                                device=device,
                                epochs=int(params.get("XAI_EPOCHS", 200)),
                                lr=float(params.get("XAI_LR", 0.05)),
                                topk_feat=int(params.get("XAI_TOPK_FEAT", 20)),
                                topk_edge=int(params.get("XAI_TOPK_EDGE", 30)),
                                min_show=float(params.get("XAI_MIN_SHOW", 1e-6)),
                                mlflow_log=True,
                                tag=tag,
                                impact_reference=str(params.get("XAI_IMPACT_REF", "masked")),
                            )
                            mlflow_log_maskopt_plots(df_feat, df_edge, tag=tag)
                    except Exception as e:
                        print(f"[XAI] failed for node={tg}: {e}")

        # finalize
        res = RunResult(run_id=run_id, run_name=run_name, metrics=metrics, params=params)

        # free memory
        del model
        maybe_empty_cache(device)
        gc.collect()

        return res


# -----------------------------------------------------------------------------
# Utilities: sweep grid
# -----------------------------------------------------------------------------
def _parse_float_list(s: Optional[str]):
    if s is None or str(s).strip() == "":
        return None
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(float(part))
    return out or None


def _parse_int_list(s: Optional[str]):
    if s is None or str(s).strip() == "":
        return None
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out or None


def _grid(dict_lists: Dict[str, List[object]]) -> List[Dict[str, object]]:
    keys = list(dict_lists.keys())
    vals = [dict_lists[k] for k in keys]
    if not keys:
        return [dict()]
    out = []
    for combo in itertools.product(*vals):
        d = {}
        for k, v in zip(keys, combo):
            d[k] = v
        out.append(d)
    return out


# -----------------------------------------------------------------------------
# Model loading from MLflow
# -----------------------------------------------------------------------------
def load_model_from_mlflow_run(
    run_id: str,
    graphs: GraphsData,
    device: torch.device,
    artifact_relpath: str = "model/model_state_dict",
) -> HardResidualInfluencerModel:
    """
    Download model artifact from MLflow and load it.
    We search for a .pth under the artifact path prefix.
    """
    # try to download the entire model artifact directory
    local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
    local_dir = Path(local_dir)
    pths = sorted(local_dir.glob("*.pth"))
    if not pths:
        # maybe nested
        pths = sorted(local_dir.rglob("*.pth"))
    if not pths:
        raise FileNotFoundError(f"No .pth found under MLflow artifact 'model' for run_id={run_id}. Downloaded to: {local_dir}")

    state_path = pths[0]
    print(f"[Load] using state_dict={state_path}")

    # We need params to rebuild architecture; store them in MLflow run params ideally.
    # Here we will read MLflow run params; if missing, fallback to defaults.
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    r = client.get_run(run_id)
    p = r.data.params
    gcn_dim = int(p.get("GCN_DIM", 128))
    rnn_dim = int(p.get("RNN_DIM", 128))
    num_gcn_layers = int(p.get("NUM_GCN_LAYERS", 2))
    dropout = float(p.get("DROPOUT_PROB", 0.35))
    proj = int(p.get("PROJECTION_DIM", 64))

    model = HardResidualInfluencerModel(
        feature_dim=graphs.feature_dim,
        gcn_dim=gcn_dim,
        rnn_dim=rnn_dim,
        num_gcn_layers=num_gcn_layers,
        dropout_prob=dropout,
        projection_dim=proj,
    ).to(device)

    sd = torch.load(state_path.as_posix(), map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def write_sweep_csv(all_results, sweep_csv_path):
    """Write sweep params+metrics to CSV for offline inspection."""
    try:
        import json as _json
        from pathlib import Path as _Path
        path = _Path(sweep_csv_path)
        rows = []
        for r in all_results:
            row = {"run_id": getattr(r, "run_id", None), "run_name": getattr(r, "run_name", None)}
            metrics = getattr(r, "metrics", None) or {}
            params = getattr(r, "params", None) or {}
            for mk, mv in metrics.items():
                try:
                    row[f"metric_{mk}"] = float(mv)
                except Exception:
                    row[f"metric_{mk}"] = mv
            for pk, pv in params.items():
                if isinstance(pv, (dict, list, tuple)):
                    row[f"param_{pk}"] = _json.dumps(pv, ensure_ascii=False)
                else:
                    row[f"param_{pk}"] = pv
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"[Sweep] wrote {path.resolve()}")
    except Exception as e:
        print(f"[Sweep] CSV write skipped: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="InfluencerRank (no-image) — Sweep training + eval + optional XAI (MLflow logged)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # execution mode
    parser.add_argument("--mode", type=str, default="train", choices=["prepare", "train", "infer", "xai", "all"],
                        help="prepare=build graphs only; train=train+eval sweep; infer=infer-only from saved model; xai=xai-only; all=train+eval+xai")
    parser.add_argument("--device", type=str, default=_PRE_ARGS.device, help="auto|cpu|mps|<cuda_index>")
    parser.add_argument("--seed", type=int, default=42)

    # MLflow
    parser.add_argument("--mlflow_tracking_uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--mlflow_experiment_name", type=str, default="InfluencerRankSweep", help="MLflow experiment name")
    parser.add_argument("--local_artifact_dir", type=str, default="mlruns_artifacts", help="Local artifact dir for file-based MLflow")

    # Data window
    parser.add_argument("--end_date", type=str, default="2017-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=12, help="Number of monthly snapshots")

    # fixed epochs default 50
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per run (default 50)")
    parser.add_argument("--cache_seq_on_cpu", type=int, default=1, help="1=cache influencer sequences on CPU (recommended for MPS)")

    # sweep controls
    parser.add_argument("--sweep_mode", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--max_runs", type=int, default=0, help="If >0, limit number of runs executed")

    # explicit grid lists (comma separated). If given, overrides sweep_mode.
    parser.add_argument("--lr_list", type=str, default=None)
    parser.add_argument("--dropout_list", type=str, default=None)
    parser.add_argument("--weight_decay_list", type=str, default=None)
    parser.add_argument("--gcn_dim_list", type=str, default=None)
    parser.add_argument("--rnn_dim_list", type=str, default=None)
    parser.add_argument("--proj_dim_list", type=str, default=None)
    parser.add_argument("--num_gcn_layers_list", type=str, default=None)

    # train batch params
    parser.add_argument("--list_size", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--use_sampler", type=int, default=0)
    parser.add_argument("--pointwise_weight", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--infer_batch", type=int, default=1024)

    # choose "best" metric
    parser.add_argument("--select_metric", type=str, default="spearman_corr", choices=["spearman_corr", "rmse", "mae", "pearson_corr"])

    # XAI controls
    parser.add_argument("--xai_on_best", type=int, default=0, help="After sweep, run XAI only on the best run and log into that run if possible")
    parser.add_argument("--xai_top_pred_k", type=int, default=3)
    parser.add_argument("--xai_top_pos", type=int, default=3)
    parser.add_argument("--xai_epochs", type=int, default=200)
    parser.add_argument("--xai_lr", type=float, default=0.05)
    parser.add_argument("--xai_impact_ref", type=str, default="masked", choices=["masked", "unmasked", "both"])

    # infer/xai-only: load model from mlflow run id
    parser.add_argument("--load_run_id", type=str, default=None, help="MLflow run_id to load model from (for infer/xai-only)")

    args = parser.parse_args()

    device = get_device(args.device)
    print("[Device] Using:", device)

    exp_name, exp_id = setup_mlflow_experiment(
        experiment_base_name=args.mlflow_experiment_name,
        tracking_uri=args.mlflow_tracking_uri,
        local_artifact_dir=args.local_artifact_dir,
    )

    graphs = prepare_graph_data(end_date=pd.to_datetime(args.end_date), num_months=int(args.months))
    if graphs is None:
        return 2

    if args.mode == "prepare":
        print("✅ prepare mode done (graphs built).")
        return 0

    # base params (these are the *only* parameters we actually use in the model/training)
    base_params = {
        "EPOCHS": int(args.epochs),
        "LR": 1e-3,
        "WEIGHT_DECAY": 0.0,
        "DROPOUT_PROB": 0.35,
        "GCN_DIM": 128,
        "RNN_DIM": 128,
        "NUM_GCN_LAYERS": 2,
        "PROJECTION_DIM": 64,
        "LIST_SIZE": int(args.list_size),
        "BATCH_SIZE": int(args.batch_size),
        "USE_SAMPLER": bool(args.use_sampler),
        "POINTWISE_LOSS_WEIGHT": float(args.pointwise_weight),
        "GRAD_CLIP": float(args.grad_clip),
        "INFER_BATCH": int(args.infer_batch),
        # XAI params (used only if do_xai)
        "XAI_EPOCHS": int(args.xai_epochs),
        "XAI_LR": float(args.xai_lr),
        "XAI_IMPACT_REF": str(args.xai_impact_ref),
        "XAI_TOPK_FEAT": 20,
        "XAI_TOPK_EDGE": 30,
        "XAI_MIN_SHOW": 1e-6,
    }

    cache_seq_on_cpu = bool(int(args.cache_seq_on_cpu))

    # infer-only / xai-only: load model and run inside a new mlflow run (or reuse existing run by continuing)
    if args.mode in ("infer", "xai"):
        if not args.load_run_id:
            print("❌ --load_run_id is required for infer/xai-only modes.")
            return 2

        # continue the same run if possible
        try:
            mlflow.start_run(run_id=args.load_run_id)
            started_existing = True
        except Exception:
            started_existing = False
            mlflow.start_run(run_name=f"resume_{args.mode}_{args.load_run_id}", experiment_id=exp_id)
            mlflow.log_param("resumed_from_run_id", args.load_run_id)

        try:
            model = load_model_from_mlflow_run(args.load_run_id, graphs, device=device)
            # For infer-only, we call train_and_eval_one_run with do_train=False but do_infer=True.
            # We reuse base_params; architecture is read from MLflow run params in load_model_from_mlflow_run.
            # So here we only run eval and/or xai. We'll do it manually:
            if args.mode == "infer":
                # minimal eval by calling helper
                _ = train_and_eval_one_run(
                    graphs=graphs,
                    params={**base_params, "RUN_NAME": f"infer_only_{args.load_run_id}", "EPOCHS": 0},
                    device=device,
                    exp_id=exp_id,
                    do_train=False,
                    do_infer=True,
                    do_xai=False,
                    seed=int(args.seed),
                    cache_seq_on_cpu=cache_seq_on_cpu,
                )
            else:
                # infer is needed before xai; do both
                _ = train_and_eval_one_run(
                    graphs=graphs,
                    params={**base_params, "RUN_NAME": f"xai_only_{args.load_run_id}", "EPOCHS": 0},
                    device=device,
                    exp_id=exp_id,
                    do_train=False,
                    do_infer=True,
                    do_xai=True,
                    xai_top_pred_k=int(args.xai_top_pred_k),
                    xai_top_pos=int(args.xai_top_pos),
                    seed=int(args.seed),
                    cache_seq_on_cpu=cache_seq_on_cpu,
                )
        finally:
            mlflow.end_run()
        return 0

    # Build sweep list (train mode / all mode)
    explicit = {
        "LR": _parse_float_list(args.lr_list),
        "DROPOUT_PROB": _parse_float_list(args.dropout_list),
        "WEIGHT_DECAY": _parse_float_list(args.weight_decay_list),
        "GCN_DIM": _parse_int_list(args.gcn_dim_list),
        "RNN_DIM": _parse_int_list(args.rnn_dim_list),
        "PROJECTION_DIM": _parse_int_list(args.proj_dim_list),
        "NUM_GCN_LAYERS": _parse_int_list(args.num_gcn_layers_list),
    }
    explicit = {k: v for k, v in explicit.items() if v is not None}

    if explicit:
        sweep_overrides = _grid(explicit)
    else:
        if args.sweep_mode == "small":
            sweep_overrides = _grid({
                "LR": [1e-3, 5e-4],
                "DROPOUT_PROB": [0.2, 0.35],
                "RNN_DIM": [64, 128],
            })
        elif args.sweep_mode == "medium":
            sweep_overrides = _grid({
                "LR": [1e-3, 7e-4, 5e-4],
                "DROPOUT_PROB": [0.2, 0.35, 0.5],
                "WEIGHT_DECAY": [0.0, 1e-4],
                "GCN_DIM": [64, 128],
                "RNN_DIM": [64, 128],
            })
        else:
            sweep_overrides = _grid({
                "LR": [2e-3, 1e-3, 7e-4, 5e-4],
                "DROPOUT_PROB": [0.1, 0.2, 0.35, 0.5],
                "WEIGHT_DECAY": [0.0, 1e-5, 1e-4],
                "GCN_DIM": [64, 128],
                "RNN_DIM": [64, 128],
                "PROJECTION_DIM": [32, 64],
                "NUM_GCN_LAYERS": [2],
            })

    if args.max_runs and int(args.max_runs) > 0:
        sweep_overrides = sweep_overrides[: int(args.max_runs)]

    print(f"[Sweep] total_runs={len(sweep_overrides)} (mode={args.mode}, sweep_mode={args.sweep_mode}, explicit={bool(explicit)})")

    do_xai_each_run = (args.mode == "all")  # if 'all', we do xai for each run (can be slow)
    all_results: List[RunResult] = []

    sweep_csv_path = Path("sweep_metrics.csv")

    for i, ov in enumerate(sweep_overrides, start=1):
        p = dict(base_params)
        p.update(ov)
        p["RUN_NAME"] = f"Sweep_{i:03d}_ep{p['EPOCHS']}_lr{p['LR']}_do{p['DROPOUT_PROB']}_gcn{p['GCN_DIM']}_rnn{p['RNN_DIM']}_proj{p['PROJECTION_DIM']}_L{p['NUM_GCN_LAYERS']}"

        print("\n" + "=" * 90)
        print(f"🚀 Run {i}/{len(sweep_overrides)}: {p['RUN_NAME']}")
        print(json.dumps({k: p[k] for k in ["LR","DROPOUT_PROB","WEIGHT_DECAY","GCN_DIM","RNN_DIM","PROJECTION_DIM","NUM_GCN_LAYERS","EPOCHS"]}, indent=2))

        res = train_and_eval_one_run(
            graphs=graphs,
            params=p,
            device=device,
            exp_id=exp_id,
            do_train=True,
            do_infer=True,
            do_xai=bool(do_xai_each_run),
            xai_top_pred_k=int(args.xai_top_pred_k),
            xai_top_pos=int(args.xai_top_pos),
            seed=int(args.seed),
            cache_seq_on_cpu=cache_seq_on_cpu,
        )
        all_results.append(res)
        write_sweep_csv(all_results, sweep_csv_path)


    # Summarize (best run)
    if all_results:
        metric = args.select_metric

        def key_fn(r: RunResult):
            v = r.metrics.get(metric, None)
            if v is None:
                return (1e9 if metric in ("rmse", "mae") else -1e9)
            # larger is better for corr; smaller for error
            if metric in ("rmse", "mae"):
                return float(v)
            return -float(v)

        all_sorted = sorted(all_results, key=key_fn)
        best = all_sorted[0]
        print("\n" + "#" * 90)
        print(f"[Sweep Summary] Best by {metric}: run_id={best.run_id} name={best.run_name} metrics={best.metrics}")

        # write summary json
        summary = {
            "select_metric": metric,
            "metrics_csv": "sweep_metrics.csv",
            "best": {"run_id": best.run_id, "run_name": best.run_name, "metrics": best.metrics, "params": best.params},
            "results": [{"run_id": r.run_id, "run_name": r.run_name, "metrics": r.metrics, "params": r.params} for r in all_sorted],
        }
        summary_path = Path("sweep_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Sweep Summary] wrote {summary_path.resolve()}")

        # log sweep summary to MLflow (as a new run)
        with mlflow.start_run(run_name=f"sweep_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}", experiment_id=exp_id):
            mlflow.log_param("best_run_id", best.run_id)
            mlflow.log_param("best_run_name", best.run_name)
            for k, v in best.metrics.items():
                mlflow.log_metric(f"best_{k}", float(v))
            mlflow.log_artifact(summary_path.as_posix(), artifact_path="sweep")
            if sweep_csv_path.exists():
                mlflow.log_artifact(sweep_csv_path.as_posix(), artifact_path="sweep")

        # optionally run XAI only on best run, logging into best run if possible
        if int(args.xai_on_best) == 1:
            print("\n🧩 [XAI on best] Running XAI only for best run and logging into best run if possible.")
            try:
                # resume existing best run
                mlflow.start_run(run_id=best.run_id)
                resumed = True
            except Exception:
                resumed = False
                mlflow.start_run(run_name=f"xai_best_{best.run_id}", experiment_id=exp_id)
                mlflow.log_param("resumed_from_run_id", best.run_id)

            try:
                # Load model from the run artifact (robust)
                model = load_model_from_mlflow_run(best.run_id, graphs, device=device)
                # run xai using helper path: call train_and_eval_one_run with do_train=False but do_infer=True+do_xai=True
                _ = train_and_eval_one_run(
                    graphs=graphs,
                    params={**best.params, **base_params, "RUN_NAME": f"xai_best_{best.run_id}", "EPOCHS": 0},
                    device=device,
                    exp_id=exp_id,
                    do_train=False,
                    do_infer=True,
                    do_xai=True,
                    xai_top_pred_k=int(args.xai_top_pred_k),
                    xai_top_pos=int(args.xai_top_pos),
                    seed=int(args.seed),
                    cache_seq_on_cpu=cache_seq_on_cpu,
                )
            finally:
                mlflow.end_run()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
