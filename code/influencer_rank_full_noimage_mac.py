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

    return TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), target_y, baseline_y)


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
        self.T = len(input_graphs)
        self.target_global = int(target_node_idx)
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
        else:
            self.num_edge_params = int(self.ei_exp.size(1))

        self.feature_dim = int(self.x_exp.size(1))

    def num_mask_params(self):
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
    impact_reference="masked",  # "masked" | "unmasked" | "both"
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

    Fdim, Edim = wrapper.num_mask_params()

    feat_logits = nn.Parameter(0.1 * torch.randn(Fdim, device=device))
    edge_logits = nn.Parameter(0.1 * torch.randn(Edim, device=device)) if Edim > 0 else None
    mask_params = [feat_logits] + ([edge_logits] if edge_logits is not None else [])
    opt = torch.optim.Adam(mask_params, lr=lr)

    orig = float(wrapper.original_pred())
    orig_t = torch.tensor(orig, device=device)

    print(f"🧠 [MaskOpt] target_node={int(target_node_idx)} explain_pos={explain_pos}/{T-1} orig={orig:.6f}")
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

    # Feature rows
    feat_rows = []
    df_feat = pd.DataFrame()
    if feat_gate is not None and feat_gate.numel() > 0:
        feat_np = feat_gate.detach().cpu().numpy()
        top_feat_idx = np.argsort(feat_np)[::-1][: int(topk_feat)]
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

            nm = f"edge_group_{gidx}"
            row = {"Type": "Edge", "Name": nm, "Importance": imp}
            for key, diff, direc in refs:
                row[f"Score_Impact({key})"] = float(diff)
                row[f"Direction({key})"] = direc
            edge_rows.append(row)

        df_edge = pd.DataFrame(edge_rows)
        if not df_edge.empty:
            df_edge = df_edge.sort_values("Importance", ascending=False).reset_index(drop=True)

    if mlflow_log:
        try:
            feat_csv = f"maskopt_feat_{tag}_node_{int(target_node_idx)}_pos_{explain_pos}.csv"
            edge_csv = f"maskopt_edge_{tag}_node_{int(target_node_idx)}_pos_{explain_pos}.csv"
            df_feat.to_csv(feat_csv, index=False, float_format="%.8e")
            df_edge.to_csv(edge_csv, index=False, float_format="%.8e")
            mlflow.log_artifact(feat_csv)
            mlflow.log_artifact(edge_csv)
            os.remove(feat_csv)
            os.remove(edge_csv)
        except Exception as e:
            print(f"⚠️ MLflow log failed: {e}")

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


# ===================== Training / Evaluation / Explanation =====================
def run_experiment(params, graphs_data, experiment_id=None):
    # Returns: (run_id, final_test_metrics_dict)
    run_id = None
    final_test_metrics = None

    monthly_graphs, influencer_indices, node_to_idx, feature_dim, follower_feat_idx, static_cols, dynamic_cols = graphs_data
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{params.get('name_prefix', 'Run')}_{current_time_str}"

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(params)
        print(f"\n🚀 Starting MLflow Run: {run_name}")
        if "note" in params:
            print(f"Note: {params['note']}")

        model = HardResidualInfluencerModel(
            feature_dim=feature_dim,
            gcn_dim=params["GCN_DIM"],
            rnn_dim=params["RNN_DIM"],
            num_gcn_layers=params["NUM_GCN_LAYERS"],
            dropout_prob=params["DROPOUT_PROB"],
            projection_dim=params["PROJECTION_DIM"],
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params["LR"])
        criterion_list = ListMLELoss().to(device)
        criterion_mse = nn.MSELoss().to(device)

        train_dataset = get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-2)

        sampler = None
        if params.get("USE_SAMPLER", False):
            targets_for_weight = train_dataset.tensors[1].cpu().numpy()
            weights = [5.0 if t > 0.01 else 1.0 for t in targets_for_weight]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        list_size = int(params["LIST_SIZE"])
        # ensure batch size divisible by LIST_SIZE for ListMLE reshape
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

        train_input_graphs = monthly_graphs[:-2]
        gpu_graphs = [g.to(device) for g in train_input_graphs]

        print("Starting Training...")
        model.train()
        last_epoch = 0
        for epoch in range(int(params["EPOCHS"])):
            last_epoch = epoch + 1
            model.train()
            total_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            seq_emb, raw_emb = [], []
            for g in gpu_graphs:
                p_x = model.projection_layer(g.x)
                gcn_out = model.gcn_encoder(p_x, g.edge_index)
                raw_emb.append(p_x)
                seq_emb.append(gcn_out)

            full_seq = torch.stack(seq_emb).permute(1, 0, 2)
            full_raw = torch.stack(raw_emb)

            loss_sum = None
            num_batches = 0

            for batch in dataloader:
                b_idx, b_target, b_baseline = batch
                b_target = b_target.to(device)
                b_baseline = b_baseline.to(device)

                b_seq = full_seq[b_idx]
                b_raw = full_raw[:, b_idx, :].permute(1, 0, 2)

                preds, _ = model(b_seq, b_raw, baseline_scores=b_baseline)
                preds = preds.view(-1)

                log_target = torch.log1p(b_target * 100.0)
                log_pred = torch.log1p(preds * 100.0)

                loss_rank = criterion_list(
                    preds.view(-1, list_size),
                    log_target.view(-1, list_size),
                )
                loss_mse = criterion_mse(log_pred, log_target)
                loss = loss_rank + loss_mse * float(params.get("POINTWISE_LOSS_WEIGHT", 1.0))

                total_loss += float(loss.item())
                num_batches += 1
                loss_sum = loss if loss_sum is None else (loss_sum + loss)

            if loss_sum is not None:
                (loss_sum / float(num_batches)).backward()
            optimizer.step()

            del full_seq, full_raw, seq_emb, raw_emb
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / max(1, num_batches)
                mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
                print(f"Epoch {epoch+1}/{params['EPOCHS']} Loss: {avg_loss:.4f}")

        # ----- Inference (Dec prediction) -----
        print("\nStarting Inference...")
        model.eval()

        test_dataset = get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-1)
        all_indices = test_dataset.tensors[0]
        all_targets = test_dataset.tensors[1]
        all_baselines = test_dataset.tensors[2]

        inf_input_graphs = monthly_graphs[:-1]

        with torch.no_grad():
            seq_emb_l, raw_emb_l = [], []
            for g in inf_input_graphs:
                g = g.to(device)
                p_x = model.projection_layer(g.x)
                gcn_out = model.gcn_encoder(p_x, g.edge_index)
                raw_emb_l.append(p_x.cpu())
                seq_emb_l.append(gcn_out.cpu())

            f_seq = torch.stack(seq_emb_l)[:, influencer_indices].permute(1, 0, 2)
            f_raw = torch.stack(raw_emb_l)[:, influencer_indices].permute(1, 0, 2)

            preds_all = []
            attn_all = []
            infer_batch_size = 1024
            for i in range(0, len(all_indices), infer_batch_size):
                end = min(i + infer_batch_size, len(all_indices))
                b_seq = f_seq[i:end].to(device)
                b_raw = f_raw[i:end].to(device)
                b_base = all_baselines[i:end].to(device)

                p, attn = model(b_seq, b_raw, b_base)
                preds_all.append(p.cpu())
                attn_all.append(attn.cpu())

            predicted_scores = torch.cat(preds_all).squeeze().numpy()
            attention_matrix = torch.cat(attn_all).squeeze().cpu().numpy()

            true_scores = all_targets.cpu().numpy()
            baseline_scores = all_baselines.cpu().numpy()

        # ----- Plots -----
        print("\n--- 📊 Generating and Logging Plots ---")
        last_input_graph = inf_input_graphs[-1]
        follower_counts = last_input_graph.x[all_indices, follower_feat_idx].cpu().numpy()
        log_follower_counts = np.log1p(follower_counts)

        epsilon = 1e-9
        true_growth = (true_scores - baseline_scores) / (baseline_scores + epsilon)
        pred_growth = (predicted_scores - baseline_scores) / (baseline_scores + epsilon)

        plot_files = []
        plot_files.append(generate_enhanced_scatter_plot(true_scores, predicted_scores, "True Engagement Score", "Predicted Score", run_name, "score_basic"))

        att_bar_file, att_heat_file, att_csv_file, att_raw_file = plot_attention_weights(attention_matrix, run_name)
        for f in [att_bar_file, att_heat_file, att_csv_file, att_raw_file]:
            if f is None:
                continue
            mlflow.log_artifact(f)
            try:
                os.remove(f)
            except Exception:
                pass

        plot_files.append(
            generate_enhanced_scatter_plot(
                true_scores,
                predicted_scores,
                "True Engagement Score",
                "Predicted Score",
                run_name,
                "score_by_followers",
                color_data=log_follower_counts,
                color_label="Log(Followers)",
                title_suffix="(Colored by Followers)",
            )
        )

        plot_files.append(
            generate_enhanced_scatter_plot(
                true_scores,
                predicted_scores,
                "True Engagement Score",
                "Predicted Score",
                run_name,
                "score_by_growth",
                color_data=true_growth,
                color_label="True Growth Rate",
                title_suffix="(Colored by Growth Rate)",
            )
        )

        plot_files.append(
            generate_enhanced_scatter_plot(
                true_growth,
                pred_growth,
                "True Growth Rate",
                "Predicted Growth Rate",
                run_name,
                "growth_by_followers",
                color_data=log_follower_counts,
                color_label="Log(Followers)",
                title_suffix="(Colored by Followers)",
            )
        )

        for f in plot_files:
            if f is not None and os.path.exists(f):
                mlflow.log_artifact(f)
                try:
                    os.remove(f)
                except Exception:
                    pass

        # ----- Metrics -----
        print("\n--- 📊 Evaluation Metrics ---")
        mae = mean_absolute_error(true_scores, predicted_scores)
        rmse = np.sqrt(mean_squared_error(true_scores, predicted_scores))
        p_corr, _ = pearsonr(true_scores, predicted_scores)
        s_corr, _ = spearmanr(true_scores, predicted_scores)

        mlflow.log_metrics({"mae": mae, "rmse": rmse, "pearson_corr": p_corr, "spearman_corr": s_corr})
        print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, Pearson: {p_corr:.4f}, Spearman: {s_corr:.4f}")

        final_test_metrics = {"mae": float(mae), "rmse": float(rmse), "pearson_corr": float(p_corr), "spearman_corr": float(s_corr)}

        # ----- Explanation target: hub influencer in Nov graph -----
        feature_names = static_cols + dynamic_cols
        target_graph = monthly_graphs[-2]  # Nov graph
        edge_index = target_graph.edge_index.to(device)
        d = degree(edge_index[1], num_nodes=target_graph.num_nodes)

        max_degree = -1
        target_node_global_idx = -1
        for idx in influencer_indices:
            deg = float(d[idx].item())
            if deg > max_degree:
                max_degree = deg
                target_node_global_idx = int(idx)

        print(f"\n🎯 Selected Hub User (Node {target_node_global_idx}) with {int(max_degree)} edges.")

        input_graphs = monthly_graphs[:-1]  # Jan..Nov, T=11
        T = len(input_graphs)

        # ensure these exist even if explanation fails
        df_feat = None
        df_edge = None
        meta = None
        explain_pos_last = None

        try:
            # attention row for this node (if exists in inference matrix)
            attn_w = None
            pos_in_all = (all_indices == target_node_global_idx).nonzero(as_tuple=False)
            if pos_in_all.numel() > 0:
                row = int(pos_in_all[0].item())
                attn_w = torch.tensor(attention_matrix[row], dtype=torch.float32)

            sens_df, sens_selected = None, None
            if params.get("explain_use_sensitivity", True):
                try:
                    sens_df, sens_selected, _pred_full, _alpha = compute_time_step_sensitivity(
                        model=model,
                        input_graphs=input_graphs,
                        target_node_idx=target_node_global_idx,
                        device=device,
                        topk=int(params.get("xai_topk_pos", 3)),
                        score_mode=str(params.get("sensitivity_score_mode", "alpha_x_delta")),
                        min_delta=float(params.get("sensitivity_min_delta", 1e-4)),
                    )
                except Exception as e:
                    print(f"[Explain] sensitivity computation skipped: {e}")
                    sens_df, sens_selected = None, None

            if attn_w is None:
                positions_attn = list(range(min(3, T)))
            else:
                positions_attn = _select_positions_by_attention(attn_w, T, topk=int(params.get("xai_topk_pos", 3)), min_w=float(params.get("xai_attn_min_w", 0.0)))

            positions_to_explain = positions_attn
            if sens_selected is not None and len(sens_selected) > 0:
                positions_to_explain = sens_selected[: int(params.get("xai_topk_pos", 3))]

            print(f"[Explain] positions_to_explain={positions_to_explain} / T={T}")

            # export alpha + sensitivity
            if attn_w is not None:
                labels = [f"T-{T-1-i}" for i in range(T)]
                df_alpha = pd.DataFrame(
                    {
                        "pos": list(range(T)),
                        "label": labels,
                        "alpha": attn_w.detach().cpu().numpy().astype(float),
                        "selected_for_explain": [int(i in positions_to_explain) for i in range(T)],
                    }
                )
                if sens_df is not None and (not sens_df.empty):
                    df_alpha = df_alpha.merge(sens_df, on="pos", how="left")

                alpha_csv = f"xai_attention_alpha_node_{int(target_node_global_idx)}_{run_name}.csv"
                df_alpha.to_csv(alpha_csv, index=False, float_format="%.8e")
                mlflow.log_artifact(alpha_csv)
                os.remove(alpha_csv)

            for explain_pos in positions_to_explain:
                explain_pos_last = int(explain_pos)
                tag = f"pos_{explain_pos_last}"

                df_feat, df_edge, meta = maskopt_e2e_explain(
                    model=model,
                    input_graphs=input_graphs,
                    target_node_idx=target_node_global_idx,
                    explain_pos=explain_pos_last,
                    feature_names=feature_names,
                    node_to_idx=node_to_idx,
                    device=device,
                    use_subgraph=True,
                    num_hops=1,
                    edge_mask_scope="incident",
                    fid_weight=2000.0,
                    coeffs={"edge_size": 0.08, "edge_ent": 0.15, "node_feat_size": 0.02, "node_feat_ent": 0.15},
                    budget_feat=10,
                    budget_edge=20,
                    budget_weight=1.0,
                    impact_reference="masked",
                    use_contrastive=False,
                    tag=tag,
                    mlflow_log=True,
                )

                print(tag, meta.get("orig_pred"), meta.get("best_pred"))
                if df_feat is not None and not df_feat.empty:
                    print(df_feat.head(10).to_string(index=False, float_format=lambda v: f"{v:.8e}"))
                if df_edge is not None and not df_edge.empty:
                    print(df_edge.head(10).to_string(index=False, float_format=lambda v: f"{v:.8e}"))

                # log nice plots per explained pos
                try:
                    mlflow_log_maskopt_plots(
                        df_feat=df_feat,
                        df_edge=df_edge,
                        meta=meta,
                        tag=tag,
                        topk_feat=15,
                        topk_edge=15,
                        artifact_path="xai",
                        fname_prefix=f"node_{target_node_global_idx}",
                    )
                except Exception as e:
                    print(f"[Explain] plot logging failed: {e}")

        except Exception as e:
            print(f"💥 Explanation Error: {e}")

        # log pred scatter (always)
        try:
            mlflow_log_pred_scatter(
                y_true=true_scores,
                y_pred=predicted_scores,
                tag="test_dec2017",
                step=last_epoch,
                artifact_path="plots",
            )
        except Exception as e:
            print(f"[Plots] pred scatter logging failed: {e}")

        # Cleanup
        del model, optimizer, criterion_list, criterion_mse
        if "gpu_graphs" in locals():
            del gpu_graphs
        if "f_seq" in locals():
            del f_seq
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🧹 Memory cleared after {run_name}.")

    return run_id, final_test_metrics


# ===================== Main =====================
def main():
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
