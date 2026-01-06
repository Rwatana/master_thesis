#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAI Additional Experiment 4: Fidelity–Sparsity frontier (MaskOpt hyperparameter sweep)

What this script does
---------------------
- Loads your monthly graph sequence (same CSV/txt inputs as your pipeline)
- Loads a *trained checkpoint* (so no re-training is required)
- Runs MaskOpt explanation on a chosen (target_node, explain_pos) while sweeping sparsity regularization λ
- Records fidelity and sparsity metrics for each λ, and generates a frontier curve plot
- Logs artifacts to MLflow (CSV + PNG + per-λ gates NPZ + optional edge-group map)

Key idea
--------
If masks "stick to 0", this frontier helps you separate:
- "λ too strong" (regularization dominates, sparse but low fidelity)
- "optimization unstable" (fidelity fails even at weak λ, or jumps)
- "data/model limitation" (fidelity plateau; sparsity doesn't help)

Usage examples
--------------
(1) Run with a local checkpoint and log to local MLflow:
    python xai_fidelity_sparsity_frontier.py \
      --mode frontier \
      --ckpt ./checkpoints/Run_000_20251226_1547/model_state.pt \
      --end_date 2017-12-31 --num_months 12 \
      --metric_numerator likes_and_comments --metric_denominator followers \
      --explain_pos 4 \
      --lambdas 0,0.001,0.003,0.01,0.03,0.1 \
      --epochs 300 --lr 0.05 \
      --use_subgraph --num_hops 1 --undirected \
      --edge_grouping neighbor \
      --mlflow

(2) Specify target node (global idx) instead of auto "hub influencer":
    ... --target_node 1238805

Notes
-----
- This script is *standalone*: it includes the minimum subset of your model + MaskOpt wrapper.
- It assumes your graph is built on the same global node indexing across months (as in your pipeline).
"""

import os
import sys
import gc
import math
import time
import json
import random
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, coalesce, k_hop_subgraph, degree

# matplotlib only (no seaborn needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow


# -------------------- MLflow setup --------------------
def setup_mlflow_experiment(
    experiment_base_name: str = "InfluencerRankSweep",
    tracking_uri: Optional[str] = None,
    local_artifact_dir: str = "mlruns_artifacts",
):
    """
    Local-first MLflow experiment setup.
    Creates a file-based artifact location when tracking URI is file-based.
    """
    import datetime as _dt
    from pathlib import Path as _Path

    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)

    active_tracking_uri = mlflow.get_tracking_uri()
    is_remote_tracking = active_tracking_uri.startswith("http://") or active_tracking_uri.startswith("https://")

    base_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_base_name)
    exp_name = base_name
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    cwd = _Path.cwd()
    artifact_dir = (cwd / local_artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    def _get_exp(name: str):
        try:
            return mlflow.get_experiment_by_name(name)
        except Exception:
            return None

    exp = _get_exp(exp_name)
    if (not is_remote_tracking) and (exp is not None) and str(exp.artifact_location).startswith("mlflow-artifacts:"):
        exp_name = f"{base_name}_file_{ts}"
        exp = None

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


# -------------------- Device --------------------
def get_device(requested: str):
    requested = (requested or "").lower()
    if requested in ("cpu",):
        return torch.device("cpu")
    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(requested)
        print("[Device] CUDA requested but not available; fallback CPU.")
        return torch.device("cpu")
    if requested in ("mps",):
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                _ = torch.tensor([1.0], device="mps")
                return torch.device("mps")
            except Exception:
                pass
        print("[Device] MPS requested but not available; fallback CPU.")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            _ = torch.tensor([1.0], device="mps")
            return torch.device("mps")
        except Exception:
            pass
    return torch.device("cpu")


# -------------------- Files (same as your pipeline defaults) --------------------
PREPROCESSED_FILE = "dataset_A_active_all.csv"
IMAGE_DATA_FILE   = "image_features_v2_full_fixed.csv"
HASHTAGS_FILE     = "hashtags_2017.csv"
MENTIONS_FILE     = "mentions_2017.csv"
INFLUENCERS_FILE  = "influencers.txt"


# ===================== Data Loading / Graph Building =====================
def load_influencer_profiles():
    """Read influencers.txt (tab-separated)."""
    print(f"Loading influencer profiles from {INFLUENCERS_FILE}...")
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
        df_inf[c] = pd.to_numeric(df_inf[c], errors="coerce").fillna(0.0)
    print(f"Loaded {len(df_inf)} influencer profiles.")
    return df_inf


def prepare_graph_data(
    end_date: pd.Timestamp,
    num_months: int = 12,
    metric_numerator: str = "likes_and_comments",
    metric_denominator: str = "followers",
    use_image_features: bool = False,
):
    """
    Build graph sequence for each month.

    Returns:
      monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")

    # --- 1) Load post data ---
    df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=["datetime"], low_memory=False, dtype={"post_id": str})
    print(f"Loaded {len(df_posts)} posts from {PREPROCESSED_FILE}")
    df_posts["username"] = df_posts["username"].astype(str).str.strip()

    # filter to users active in Dec 2017 (same as your current choice)
    target_month_start = pd.Timestamp("2017-12-01")
    target_month_end   = pd.Timestamp("2017-12-31 23:59:59")
    dec_posts = df_posts[(df_posts["datetime"] >= target_month_start) & (df_posts["datetime"] <= target_month_end)]
    valid_users_dec = set(dec_posts["username"].unique())
    print(f"Users who posted in Dec 2017: {len(valid_users_dec):,}")
    if len(valid_users_dec) == 0:
        raise RuntimeError("No users found who posted in Dec 2017. Check your dataset date range.")

    df_posts = df_posts[df_posts["username"].isin(valid_users_dec)].copy()

    # normalize column names
    col_map = {
        "like_count": "likes",
        "comment_count": "comments",
        "hashtag_count": "tag_count",
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
    for col in ["likes", "comments", "feedback_rate"]:
        if col not in df_posts.columns:
            df_posts[col] = 0.0
        else:
            df_posts[col] = df_posts[col].fillna(0.0)

    # --- 2) Optional image features (disabled by default) ---
    df_objects_slim = pd.DataFrame(columns=["post_id", "username", "image_object"])
    if use_image_features:
        df_image_data = pd.read_csv(IMAGE_DATA_FILE, low_memory=False, dtype={"post_id": str})
        if {"post_id", "username", "detected_object"}.issubset(df_image_data.columns):
            df_objects_slim = df_image_data[["post_id", "username", "detected_object"]].copy()
            df_objects_slim.rename(columns={"detected_object": "image_object"}, inplace=True)
            df_objects_slim["username"] = df_objects_slim["username"].astype(str).str.strip()
            df_objects_slim = df_objects_slim[df_objects_slim["username"].isin(valid_users_dec)]
        if "color_temp" in df_image_data.columns and "color_temp_proxy" not in df_image_data.columns:
            df_image_data.rename(columns={"color_temp": "color_temp_proxy"}, inplace=True)

        image_feature_cols = ["post_id", "brightness", "colorfulness", "color_temp_proxy"]
        for col in image_feature_cols:
            if col not in df_image_data.columns:
                df_image_data[col] = 0.0
        df_image_features = df_image_data[image_feature_cols].copy()
        df_posts = pd.merge(df_posts, df_image_features, on="post_id", how="left")
        for col in ["brightness", "colorfulness", "color_temp_proxy"]:
            df_posts[col] = df_posts[col].fillna(0.0)
    else:
        for col in ["brightness", "colorfulness", "color_temp_proxy"]:
            if col not in df_posts.columns:
                df_posts[col] = 0.0

    # --- 3) Edges from objects/hashtags/mentions ---
    df_object_edges = pd.merge(df_objects_slim, df_posts[["post_id", "datetime"]], on="post_id", how="inner")

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

    # --- 4) Influencer profiles (static features) ---
    df_influencers_external = load_influencer_profiles()
    df_influencers_external = df_influencers_external[df_influencers_external["username"].isin(valid_users_dec)]

    df_active_base = pd.DataFrame({"username": list(valid_users_dec)})
    df_influencers = pd.merge(df_active_base, df_influencers_external, on="username", how="left")
    df_influencers["followers"] = df_influencers["followers"].fillna(0.0)
    df_influencers["followees"] = df_influencers["followees"].fillna(0.0)
    df_influencers["posts_history"] = df_influencers["posts_history"].fillna(0.0)
    df_influencers["category"] = df_influencers["category"].fillna("Unknown")

    # --- 5) Node universe ---
    df_posts["month"] = df_posts["datetime"].dt.to_period("M").dt.start_time

    influencer_set = set(df_influencers["username"].astype(str))
    all_hashtags = set(df_hashtags["hashtag"].astype(str))
    all_mentions = set(df_mentions["mention"].astype(str))
    all_image_objects = set(df_object_edges["image_object"].astype(str))

    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions | all_image_objects))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 6) Static features table ---
    node_df = pd.DataFrame({"username": all_nodes})
    profile_features = pd.merge(
        node_df,
        df_influencers[["username", "followers", "followees", "posts_history", "category"]],
        on="username",
        how="left",
    )

    for col in ["followers", "followees", "posts_history"]:
        profile_features[col] = pd.to_numeric(profile_features[col], errors="coerce").fillna(0.0)
        profile_features[col] = np.log1p(profile_features[col])

    category_dummies = pd.get_dummies(profile_features["category"], prefix="cat", dummy_na=True)
    profile_features = pd.concat([profile_features.drop(columns=["category"]), category_dummies], axis=1)

    node_df["type"] = "unknown"
    node_df.loc[node_df["username"].isin(influencer_set), "type"] = "influencer"
    node_df.loc[node_df["username"].isin(all_hashtags), "type"] = "hashtag"
    node_df.loc[node_df["username"].isin(all_mentions), "type"] = "mention"
    node_df.loc[node_df["username"].isin(all_image_objects), "type"] = "image_object"

    node_type_dummies = pd.get_dummies(node_df["type"], prefix="type")
    static_features = pd.concat([profile_features, node_type_dummies], axis=1)

    static_feature_cols = list(static_features.drop(columns=["username"]).columns)
    follower_feat_idx = static_feature_cols.index("followers") if "followers" in static_feature_cols else 0
    print(f"Follower feature index in static features: {follower_feat_idx}")

    # --- 7) Dynamic features (monthly aggregation) ---
    STATS_AGG = ["mean", "median", "min", "max"]
    required_cols = [
        "brightness", "colorfulness", "color_temp_proxy",
        "tag_count", "mention_count", "emoji_count", "caption_length",
        "caption_sent_pos", "caption_sent_neg", "caption_sent_neu", "caption_sent_compound",
        "feedback_rate", "comment_avg_pos", "comment_avg_neg", "comment_avg_neu", "comment_avg_compound"
    ]
    for col in required_cols:
        if col not in df_posts.columns:
            df_posts[col] = 0.0

    df_posts = df_posts.sort_values(["username", "datetime"])
    df_posts["post_interval_sec"] = df_posts.groupby("username")["datetime"].diff().dt.total_seconds().fillna(0.0)

    if "post_category" not in df_posts.columns:
        post_categories = [f"post_cat_{i}" for i in range(10)]
        df_posts["post_category"] = np.random.choice(post_categories, size=len(df_posts))
    if "is_ad" not in df_posts.columns:
        df_posts["is_ad"] = 0.0

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
    dynamic_agg.rename(columns={"datetime_size": "monthly_post_count", "feedback_rate_mean": "feedback_rate", "is_ad_mean": "ad_rate"}, inplace=True)

    post_category_rate = df_posts.groupby(["username", "month"])["post_category"].value_counts(normalize=True).unstack(fill_value=0)
    post_category_rate.columns = [f"rate_{c}" for c in post_category_rate.columns]
    post_category_rate = post_category_rate.reset_index()

    dynamic_features = pd.merge(dynamic_agg, post_category_rate, on=["username", "month"], how="left")
    dynamic_feature_cols = list(dynamic_features.drop(columns=["username", "month"]).columns)

    # --- 8) Build monthly graphs ---
    monthly_graphs: List[Data] = []
    start_date = end_date - pd.DateOffset(months=num_months - 1)

    feature_columns = static_feature_cols + dynamic_feature_cols
    feature_dim = len(feature_columns)
    print(f"Total feature_dim={feature_dim}")

    for snapshot_date in pd.date_range(start_date, end_date, freq="ME"):
        snapshot_month = snapshot_date.to_period("M").start_time

        current_hashtags = df_hashtags[df_hashtags["datetime"] <= snapshot_date]
        current_mentions = df_mentions[df_mentions["datetime"] <= snapshot_date]
        current_image_objects = df_object_edges[df_object_edges["datetime"] <= snapshot_date]

        edges_io = [(node_to_idx[str(u)], node_to_idx[str(o)])
                    for u, o in zip(current_image_objects.get("username", []), current_image_objects.get("image_object", []))
                    if str(u) in node_to_idx and str(o) in node_to_idx]
        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)])
                    for u, h in zip(current_hashtags.get("username", []), current_hashtags.get("hashtag", []))
                    if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)])
                    for u, m in zip(current_mentions.get("username", []), current_mentions.get("mention", []))
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

        current_dynamic = dynamic_features[dynamic_features["month"] == snapshot_month]
        snapshot_features = pd.merge(static_features, current_dynamic, on="username", how="left")
        snapshot_features = snapshot_features[feature_columns].fillna(0.0)
        x = torch.tensor(snapshot_features.astype(float).values, dtype=torch.float32)

        # target y per month (engagement)
        monthly_posts_period = df_posts[df_posts["datetime"].dt.to_period("M") == snapshot_date.to_period("M")]
        monthly_agg = monthly_posts_period.groupby("username").agg(
            total_likes=("likes", "sum"),
            total_comments=("comments", "sum"),
            post_count=("datetime", "size"),
        ).reset_index()

        if metric_numerator == "likes_and_comments":
            monthly_agg["numerator"] = monthly_agg["total_likes"] + monthly_agg["total_comments"]
        else:
            monthly_agg["numerator"] = monthly_agg["total_likes"]

        if metric_denominator == "followers":
            numer_vals = pd.to_numeric(monthly_agg["numerator"], errors="coerce").fillna(0.0).values.astype(float)
            count_vals = pd.to_numeric(monthly_agg["post_count"], errors="coerce").fillna(0.0).values.astype(float)
            monthly_agg["avg_engagement_per_post"] = np.divide(numer_vals, count_vals, out=np.zeros_like(numer_vals), where=count_vals != 0)

            merged_data = pd.merge(monthly_agg, df_influencers[["username", "followers"]], on="username", how="left")
            numer = merged_data["avg_engagement_per_post"].values.astype(float)
            denom = pd.to_numeric(merged_data["followers"], errors="coerce").fillna(0.0).values.astype(float)
            merged_data["engagement"] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)
        else:
            merged_data = monthly_agg
            numer = merged_data["numerator"].values.astype(float)
            denom = merged_data["post_count"].values.astype(float)
            merged_data["engagement"] = np.divide(numer, denom, out=np.zeros_like(numer), where=denom != 0)

        engagement_data = pd.merge(pd.DataFrame({"username": all_nodes}), merged_data[["username", "engagement"]], on="username", how="left").fillna(0.0)
        y = torch.tensor(engagement_data["engagement"].values, dtype=torch.float32).view(-1, 1)

        monthly_graphs.append(Data(x=x, edge_index=edge_index, y=y))

    return monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols


# ===================== Model =====================
class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList(
            [GCNConv(in_channels, hidden_channels)] +
            [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)]
        )

    def forward(self, x, edge_index, edge_weight=None):
        outs = []
        for conv in self.convs:
            try:
                x = conv(x, edge_index, edge_weight=edge_weight)
            except TypeError:
                x = conv(x, edge_index)
            x = x.relu()
            outs.append(x)
        return torch.cat(outs, dim=1)

class AttentiveRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention_layer = nn.Linear(hidden_dim, 1)

    def forward(self, sequence_of_embeddings):
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_scores = self.attention_layer(rnn_out).tanh()
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        return context_vector, attention_weights

class HardResidualInfluencerModel(nn.Module):
    def __init__(self, feature_dim: int, gcn_dim: int, rnn_dim: int, num_gcn_layers: int = 2, dropout_prob: float = 0.2, projection_dim: int = 128):
        super().__init__()
        self.projection_layer = nn.Sequential(nn.Linear(feature_dim, projection_dim), nn.ReLU())
        self.gcn_encoder = GCNEncoder(projection_dim, gcn_dim, num_gcn_layers)
        combined_dim = gcn_dim * num_gcn_layers
        self.attentive_rnn = AttentiveRNN(combined_dim, rnn_dim)
        self.predictor = nn.Sequential(
            nn.Linear(rnn_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1),
        )

    def forward(self, gcn_embeddings, raw_features, baseline_scores=None):
        final_rep, attention_weights = self.attentive_rnn(gcn_embeddings)
        raw_output = self.predictor(final_rep).squeeze()
        predicted_scores = F.softplus(raw_output)
        return predicted_scores, attention_weights


def load_model_from_ckpt(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "feature_dim" in ckpt and "params" in ckpt):
        raise ValueError("Unsupported checkpoint format. Expected keys: state_dict, feature_dim, params.")
    p = ckpt["params"]
    model = HardResidualInfluencerModel(
        feature_dim=int(ckpt["feature_dim"]),
        gcn_dim=int(p.get("GCN_DIM", 128)),
        rnn_dim=int(p.get("RNN_DIM", 128)),
        num_gcn_layers=int(p.get("NUM_GCN_LAYERS", 2)),
        dropout_prob=float(p.get("DROPOUT_PROB", 0.2)),
        projection_dim=int(p.get("PROJECTION_DIM", 128)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model, int(ckpt["feature_dim"]), p


# ===================== MaskOpt (E2E) =====================
def _gcn_forward_concat(gcn_encoder: GCNEncoder, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
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

def _binary_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

class _DisableCudnn:
    def __enter__(self):
        self.prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
    def __exit__(self, exc_type, exc, tb):
        torch.backends.cudnn.enabled = self.prev
        return False

class E2EMaskOptWrapper(nn.Module):
    def __init__(
        self,
        model: HardResidualInfluencerModel,
        input_graphs: List[Data],
        target_node_idx: int,
        explain_pos: int,
        device: torch.device,
        use_subgraph: bool = True,
        num_hops: int = 1,
        undirected: bool = True,
        feat_mask_scope: str = "target",   # "target" | "subgraph" | "all"
        edge_mask_scope: str = "incident", # "incident" | "subgraph"
        edge_grouping: str = "none",       # "none" | "neighbor"
        idx_to_node: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.model = model
        self.input_graphs = input_graphs
        self.T = len(input_graphs)
        self.target_global = int(target_node_idx)
        self.explain_pos = int(explain_pos)
        self.device = device

        self.use_subgraph = bool(use_subgraph)
        self.num_hops = int(num_hops)
        self.undirected = bool(undirected)
        self.feat_mask_scope = str(feat_mask_scope)
        self.edge_mask_scope = str(edge_mask_scope)

        self.edge_grouping = str(edge_grouping)
        self.idx_to_node = idx_to_node or {}

        self.edge_group_names = None
        self.edge_group_members = None
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
                num_nodes=x_full.size(0),
            )
            x_sub = x_full[subset]
            if self.undirected and ei_sub.numel() > 0:
                ei_sub = to_undirected(ei_sub, num_nodes=x_sub.size(0))
                ei_sub = coalesce(ei_sub, num_nodes=x_sub.size(0))
            self.x_exp = x_sub
            self.ei_exp = ei_sub
            self.target_local = int(mapping.item())
            self.local2global = subset

        # incident edges
        if self.ei_exp.numel() == 0:
            self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            src, dst = self.ei_exp
            incident = (src == self.target_local) | (dst == self.target_local)
            self.incident_edge_idx = torch.where(incident)[0]

        # neighbor grouping (incident scope)
        if (self.edge_mask_scope == "incident") and (self.edge_grouping == "neighbor") and (self.incident_edge_idx.numel() > 0):
            src, dst = self.ei_exp
            groups = {}
            for epos in self.incident_edge_idx.detach().cpu().tolist():
                s = int(src[epos].item())
                d = int(dst[epos].item())
                nbr_local = d if s == self.target_local else s
                if self.local2global is not None:
                    nbr_global = int(self.local2global[nbr_local].item())
                else:
                    nbr_global = int(nbr_local)
                groups.setdefault(nbr_global, []).append(epos)

            keys = sorted(groups.keys())
            self.edge_group_members = [groups[k] for k in keys]
            self.edge_group_names = [str(self.idx_to_node.get(k, f"node_{k}")) for k in keys]
            self.edge_group_meta = [{
                "group_id": int(i),
                "neighbor_global": int(k),
                "neighbor_name": str(self.idx_to_node.get(k, f"node_{k}")),
                "num_edges_in_group": int(len(groups[k])),
            } for i, k in enumerate(keys)]
            self.num_edge_params = int(len(self.edge_group_members))
        else:
            if self.edge_mask_scope == "incident":
                self.num_edge_params = int(self.incident_edge_idx.numel())
            else:
                self.num_edge_params = int(self.ei_exp.size(1))

        self.feature_dim = int(self.x_exp.size(1))

    def num_mask_params(self) -> Tuple[int, int]:
        return self.feature_dim, self.num_edge_params

    def _apply_feature_gate(self, x: torch.Tensor, feat_gate: torch.Tensor) -> torch.Tensor:
        if self.feat_mask_scope in ("all", "subgraph"):
            return x * feat_gate.view(1, -1)
        n = x.size(0)
        sel = F.one_hot(torch.tensor(self.target_local, device=x.device), num_classes=n).to(x.dtype).unsqueeze(1)  # [N,1]
        return x + sel * x * (feat_gate.view(1, -1) - 1.0)

    def _make_edge_weight(self, edge_gate: Optional[torch.Tensor]) -> torch.Tensor:
        E = int(self.ei_exp.size(1))
        w = torch.ones(E, device=self.device)
        if E == 0 or edge_gate is None or edge_gate.numel() == 0:
            return w

        if (self.edge_mask_scope == "incident") and (self.edge_grouping == "neighbor") and (self.edge_group_members is not None):
            w = w.clone()
            for g, epos_list in enumerate(self.edge_group_members):
                if not epos_list:
                    continue
                idx = torch.tensor(epos_list, device=self.device, dtype=torch.long)
                w[idx] = edge_gate[g]
            return w

        if self.edge_mask_scope == "incident":
            w = w.clone()
            if self.incident_edge_idx.numel() > 0:
                w[self.incident_edge_idx] = edge_gate
            return w

        return edge_gate

    def predict_with_gates(self, feat_gate: torch.Tensor, edge_gate: Optional[torch.Tensor]) -> torch.Tensor:
        seq_gcn, seq_raw = [], []
        for t in range(self.T):
            if t != self.explain_pos:
                seq_gcn.append(self.cached_gcn[t])
                seq_raw.append(self.cached_proj[t])
                continue

            x_masked = self._apply_feature_gate(self.x_exp, feat_gate)
            ew = self._make_edge_weight(edge_gate)
            p = self.model.projection_layer(x_masked)
            out = _gcn_forward_concat(self.model.gcn_encoder, p, self.ei_exp, edge_weight=ew)
            seq_gcn.append(out[self.target_local])
            seq_raw.append(p[self.target_local])

        seq_gcn = torch.stack(seq_gcn, dim=0).unsqueeze(0)
        seq_raw = torch.stack(seq_raw, dim=0).unsqueeze(0)
        pred, _ = self.model(seq_gcn, seq_raw, baseline_scores=None)
        return pred.view(())

    @torch.no_grad()
    def original_pred(self) -> float:
        ones_feat = torch.ones(self.feature_dim, device=self.device)
        ones_edge = torch.ones(self.num_edge_params, device=self.device) if self.num_edge_params > 0 else None
        return float(self.predict_with_gates(ones_feat, ones_edge).item())


@torch.no_grad()
def _gate_stats(gate: Optional[torch.Tensor], thr: float = 0.01) -> Dict[str, float]:
    if gate is None or gate.numel() == 0:
        return {"mean": 0.0, "sum": 0.0, "min": float("nan"), "max": float("nan"), "le_thr": 0.0, "ge_1mthr": 0.0}
    g = gate.detach().float()
    le = float((g <= thr).sum().item())
    ge = float((g >= (1.0 - thr)).sum().item())
    return {
        "mean": float(g.mean().item()),
        "sum": float(g.sum().item()),
        "min": float(g.min().item()),
        "max": float(g.max().item()),
        "le_thr": le,
        "ge_1mthr": ge,
    }


def run_maskopt_single(
    model: HardResidualInfluencerModel,
    input_graphs: List[Data],
    target_node_idx: int,
    explain_pos: int,
    device: torch.device,
    *,
    coeffs: Dict[str, float],
    fid_weight: float = 2000.0,
    epochs: int = 300,
    lr: float = 0.05,
    feat_mask_scope: str = "target",
    edge_mask_scope: str = "incident",
    use_subgraph: bool = True,
    num_hops: int = 1,
    undirected: bool = True,
    edge_grouping: str = "neighbor",
    node_to_idx: Optional[Dict[str, int]] = None,
    tag: str = "pos_0",
    mlflow_log: bool = True,
    mask_thr: float = 0.01,
):
    """
    MaskOpt optimization for a single (node, pos), returning summary + gates + optional MLflow artifacts.
    """
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    idx_to_node = None
    if node_to_idx is not None and isinstance(node_to_idx, dict):
        idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

    wrapper = E2EMaskOptWrapper(
        model=model,
        input_graphs=input_graphs,
        target_node_idx=int(target_node_idx),
        explain_pos=int(explain_pos),
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
    feat_logits = nn.Parameter(0.1 * torch.randn(Fdim, device=device))
    edge_logits = nn.Parameter(0.1 * torch.randn(Edim, device=device)) if Edim > 0 else None
    params = [feat_logits] + ([edge_logits] if edge_logits is not None else [])
    opt = torch.optim.Adam(params, lr=float(lr))

    orig = wrapper.original_pred()
    orig_t = torch.tensor(orig, device=device)

    cudnn_ctx = _DisableCudnn()

    best_loss = float("inf")
    best_feat = None
    best_edge = None
    best_pred = None

    for ep in range(1, int(epochs) + 1):
        opt.zero_grad(set_to_none=True)
        feat_gate = torch.sigmoid(feat_logits)
        edge_gate = torch.sigmoid(edge_logits) if edge_logits is not None else None

        with cudnn_ctx:
            pred = wrapper.predict_with_gates(feat_gate, edge_gate)

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
            + float(coeffs.get("node_feat_size", 0.0)) * loss_feat_size
            + float(coeffs.get("node_feat_ent", 0.0)) * loss_feat_ent
            + float(coeffs.get("edge_size", 0.0)) * loss_edge_size
            + float(coeffs.get("edge_ent", 0.0)) * loss_edge_ent
        )

        loss.backward()
        opt.step()

        lval = float(loss.item())
        if lval < best_loss:
            best_loss = lval
            best_feat = feat_gate.detach().clone()
            best_edge = edge_gate.detach().clone() if edge_gate is not None else None
            best_pred = float(pred.detach().item())

    # final stats
    ones_feat = torch.ones(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None
    with torch.no_grad():
        with cudnn_ctx:
            pred_unmasked = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())
            pred_masked = float(wrapper.predict_with_gates(best_feat, best_edge).item()) if best_feat is not None else pred_unmasked

    fid_abs = abs(pred_masked - orig)
    fid_sq = (pred_masked - orig) ** 2

    fs = _gate_stats(best_feat, thr=float(mask_thr))
    es = _gate_stats(best_edge, thr=float(mask_thr))

    # Define "sparsity" as 1 - mean_gate (higher is sparser)
    feat_sparsity = 1.0 - float(fs["mean"])
    edge_sparsity = 1.0 - float(es["mean"]) if (best_edge is not None and best_edge.numel() > 0) else float("nan")
    # combined (ignore nan by simple rule)
    if math.isnan(edge_sparsity):
        sparsity = feat_sparsity
    else:
        sparsity = 0.5 * (feat_sparsity + edge_sparsity)

    summary = {
        "tag": str(tag),
        "target_node": int(target_node_idx),
        "explain_pos": int(explain_pos),
        "orig_pred": float(orig),
        "pred_masked": float(pred_masked),
        "pred_unmasked": float(pred_unmasked),
        "fidelity_abs": float(fid_abs),
        "fidelity_sq": float(fid_sq),
        "best_loss": float(best_loss),
        "feat_mean_gate": float(fs["mean"]),
        "edge_mean_gate": float(es["mean"]) if not math.isnan(es["mean"]) else float("nan"),
        "feat_sparsity": float(feat_sparsity),
        "edge_sparsity": float(edge_sparsity) if not math.isnan(edge_sparsity) else float("nan"),
        "sparsity": float(sparsity),
        "feat_le_thr": float(fs["le_thr"]),
        "edge_le_thr": float(es["le_thr"]),
        "feat_ge_1mthr": float(fs["ge_1mthr"]),
        "edge_ge_1mthr": float(es["ge_1mthr"]),
        "Fdim": int(Fdim),
        "Edim": int(Edim),
        "coeff_node_feat_size": float(coeffs.get("node_feat_size", 0.0)),
        "coeff_edge_size": float(coeffs.get("edge_size", 0.0)),
        "coeff_node_feat_ent": float(coeffs.get("node_feat_ent", 0.0)),
        "coeff_edge_ent": float(coeffs.get("edge_ent", 0.0)),
        "fid_weight": float(fid_weight),
        "epochs": int(epochs),
        "lr": float(lr),
        "edge_grouping": str(edge_grouping),
    }

    # ---- artifacts (NPZ gates + optional edge group map) ----
    if mlflow_log and mlflow.active_run() is not None:
        gates_npz = f"frontier_gates_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.npz"
        np.savez_compressed(
            gates_npz,
            feat_gate=best_feat.detach().cpu().numpy().astype(np.float32) if best_feat is not None else None,
            edge_gate=best_edge.detach().cpu().numpy().astype(np.float32) if best_edge is not None else None,
            summary=summary,
        )
        mlflow.log_artifact(gates_npz, artifact_path="xai/frontier")
        try:
            os.remove(gates_npz)
        except Exception:
            pass

        if (edge_grouping == "neighbor") and (wrapper.edge_group_meta is not None) and len(wrapper.edge_group_meta) > 0:
            m = pd.DataFrame(wrapper.edge_group_meta)
            map_csv = f"frontier_edge_group_map_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
            m.to_csv(map_csv, index=False)
            mlflow.log_artifact(map_csv, artifact_path="xai/frontier")
            try:
                os.remove(map_csv)
            except Exception:
                pass

    return summary, best_feat, best_edge


# ===================== Frontier sweep =====================
def frontier_sweep(
    model: HardResidualInfluencerModel,
    monthly_graphs: List[Data],
    node_to_idx: Dict[str, int],
    influencer_indices: List[int],
    target_node_idx: int,
    explain_pos: int,
    device: torch.device,
    *,
    lambdas: List[float],
    base_coeffs: Dict[str, float],
    fid_weight: float,
    epochs: int,
    lr: float,
    feat_mask_scope: str,
    edge_mask_scope: str,
    use_subgraph: bool,
    num_hops: int,
    undirected: bool,
    edge_grouping: str,
    mask_thr: float,
    mlflow_log: bool,
    tag_prefix: str,
):
    """
    Sweep λ by scaling size-regularizers:
      node_feat_size = base_node_feat_size * λ
      edge_size      = base_edge_size      * λ
    (entropy coefficients are kept as provided in base_coeffs unless you change them too.)
    """
    assert len(monthly_graphs) >= 2, "monthly_graphs must have at least 2 months."
    input_graphs = monthly_graphs[:-1]  # Jan..Nov (T=11 if 12 months)

    rows = []
    for lam in lambdas:
        coeffs = dict(base_coeffs)
        coeffs["node_feat_size"] = float(base_coeffs.get("node_feat_size", 0.02)) * float(lam)
        coeffs["edge_size"] = float(base_coeffs.get("edge_size", 0.08)) * float(lam)
        tag = f"{tag_prefix}_lam_{lam:g}"

        print(f"\n[Frontier] λ={lam:g} coeff_node_feat_size={coeffs['node_feat_size']:.6g} coeff_edge_size={coeffs['edge_size']:.6g}")
        summary, _fg, _eg = run_maskopt_single(
            model=model,
            input_graphs=input_graphs,
            target_node_idx=int(target_node_idx),
            explain_pos=int(explain_pos),
            device=device,
            coeffs=coeffs,
            fid_weight=float(fid_weight),
            epochs=int(epochs),
            lr=float(lr),
            feat_mask_scope=feat_mask_scope,
            edge_mask_scope=edge_mask_scope,
            use_subgraph=use_subgraph,
            num_hops=int(num_hops),
            undirected=bool(undirected),
            edge_grouping=edge_grouping,
            node_to_idx=node_to_idx,
            tag=str(tag),
            mlflow_log=mlflow_log,
            mask_thr=float(mask_thr),
        )
        summary["lambda"] = float(lam)
        rows.append(summary)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # Sort by lambda ascending for plotting
    df = df.sort_values("lambda", ascending=True).reset_index(drop=True)

    # Save CSV
    csv_name = f"frontier_summary_node_{int(target_node_idx)}_pos_{int(explain_pos)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(csv_name, index=False, float_format="%.8e")

    # Plot frontier: fidelity (abs) vs sparsity (combined)
    png_name = f"frontier_plot_node_{int(target_node_idx)}_pos_{int(explain_pos)}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.figure(figsize=(7.5, 6))
    plt.plot(df["sparsity"].values, df["fidelity_abs"].values, marker="o")
    for _, r in df.iterrows():
        plt.text(float(r["sparsity"]), float(r["fidelity_abs"]), f"λ={r['lambda']:g}", fontsize=8, ha="left", va="bottom")
    plt.xlabel("Sparsity (1 - mean_gate)  ↑ sparser")
    plt.ylabel("Fidelity error |pred_masked - pred_orig|  ↓ better")
    plt.title("Fidelity–Sparsity frontier (MaskOpt λ sweep)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(png_name, dpi=220)
    plt.close()

    if mlflow_log and mlflow.active_run() is not None:
        mlflow.log_artifact(csv_name, artifact_path="xai/frontier")
        mlflow.log_artifact(png_name, artifact_path="xai/frontier")

        # log main scalar series to compare runs quickly
        # (use step as lambda index)
        for i, r in df.iterrows():
            mlflow.log_metric("frontier_fidelity_abs", float(r["fidelity_abs"]), step=int(i))
            mlflow.log_metric("frontier_sparsity", float(r["sparsity"]), step=int(i))
            mlflow.log_metric("frontier_feat_mean_gate", float(r["feat_mean_gate"]), step=int(i))
            if not math.isnan(float(r.get("edge_mean_gate", float("nan")))):
                mlflow.log_metric("frontier_edge_mean_gate", float(r["edge_mean_gate"]), step=int(i))

    # cleanup local files (keep only in mlflow if logging)
    if mlflow_log:
        try:
            os.remove(csv_name)
            os.remove(png_name)
        except Exception:
            pass

    return df


# ===================== Target selection helper =====================
def pick_hub_influencer(monthly_graphs: List[Data], influencer_indices: List[int], device: torch.device) -> int:
    """
    Pick influencer node with maximum degree in the last history graph (month -2).
    """
    target_graph = monthly_graphs[-2]  # last history month
    ei = target_graph.edge_index.to(device)
    d = degree(ei[1], num_nodes=target_graph.num_nodes)
    max_degree = -1.0
    best_idx = -1
    for idx in influencer_indices:
        deg = float(d[idx].item())
        if deg > max_degree:
            max_degree = deg
            best_idx = int(idx)
    print(f"[Target] Hub influencer global_idx={best_idx} deg={int(max_degree)}")
    return best_idx


# ===================== Main =====================
def parse_float_list(s: str) -> List[float]:
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).split(",") if str(x).strip() != ""]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["frontier"], default="frontier")

    ap.add_argument("--ckpt", type=str, required=True, help="Path to saved checkpoint (model_state.pt/.pth).")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|mps|cuda:0 ...")

    ap.add_argument("--end_date", type=str, default="2017-12-31")
    ap.add_argument("--num_months", type=int, default=12)
    ap.add_argument("--metric_numerator", type=str, default="likes_and_comments")
    ap.add_argument("--metric_denominator", type=str, default="followers")
    ap.add_argument("--use_image_features", action="store_true")

    ap.add_argument("--target_node", type=int, default=None, help="Global node idx. If omitted, auto-pick hub influencer.")
    ap.add_argument("--explain_pos", type=int, default=4, help="0..T-1 position (T=num_months-1). e.g. 4 means 5th month in history.")

    # MaskOpt setup
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--fid_weight", type=float, default=2000.0)
    ap.add_argument("--feat_mask_scope", type=str, default="target", choices=["target", "subgraph", "all"])
    ap.add_argument("--edge_mask_scope", type=str, default="incident", choices=["incident", "subgraph"])
    ap.add_argument("--use_subgraph", action="store_true")
    ap.add_argument("--num_hops", type=int, default=1)
    ap.add_argument("--undirected", action="store_true")
    ap.add_argument("--edge_grouping", type=str, default="neighbor", choices=["none", "neighbor"])

    # Frontier sweep
    ap.add_argument("--lambdas", type=str, default="0,0.001,0.003,0.01,0.03,0.1", help="Comma-separated λ values.")
    ap.add_argument("--base_node_feat_size", type=float, default=0.02, help="Base coefficient multiplied by λ.")
    ap.add_argument("--base_edge_size", type=float, default=0.08, help="Base coefficient multiplied by λ.")
    ap.add_argument("--node_feat_ent", type=float, default=0.15, help="Entropy regularizer coefficient (kept constant across λ).")
    ap.add_argument("--edge_ent", type=float, default=0.15, help="Entropy regularizer coefficient (kept constant across λ).")
    ap.add_argument("--mask_thr", type=float, default=0.01, help="Threshold for near-zero/near-one counts in diagnostics.")

    # logging
    ap.add_argument("--mlflow", action="store_true", help="Log to MLflow.")
    ap.add_argument("--mlflow_experiment", type=str, default="InfluencerRankSweep")
    ap.add_argument("--mlflow_tracking_uri", type=str, default=None)
    ap.add_argument("--mlflow_artifact_dir", type=str, default="mlruns_artifacts")

    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = get_device(args.device)
    print(f"[Device] Using {device}")

    # MLflow
    exp_id = None
    if args.mlflow:
        os.environ["MLFLOW_EXPERIMENT_NAME"] = str(args.mlflow_experiment)
        setup_mlflow_experiment(
            experiment_base_name=str(args.mlflow_experiment),
            tracking_uri=args.mlflow_tracking_uri,
            local_artifact_dir=str(args.mlflow_artifact_dir),
        )
        exp = mlflow.get_experiment_by_name(str(args.mlflow_experiment))
        exp_id = exp.experiment_id if exp is not None else None

    # build graphs
    end_date = pd.to_datetime(args.end_date)
    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prepare_graph_data(
        end_date=end_date,
        num_months=int(args.num_months),
        metric_numerator=str(args.metric_numerator),
        metric_denominator=str(args.metric_denominator),
        use_image_features=bool(args.use_image_features),
    )
    feature_dim = int(monthly_graphs[0].x.size(1))
    print(f"[Data] feature_dim={feature_dim} nodes={int(monthly_graphs[0].num_nodes)} months={len(monthly_graphs)}")

    # load model
    model, ckpt_feat_dim, ckpt_params = load_model_from_ckpt(args.ckpt, device=device)
    if int(ckpt_feat_dim) != int(feature_dim):
        raise RuntimeError(f"feature_dim mismatch: ckpt={ckpt_feat_dim} vs graph={feature_dim}. Rebuild features identically or use matching ckpt.")
    print(f"[Model] loaded ckpt={args.ckpt}")

    # pick target node
    if args.target_node is None:
        target_node = pick_hub_influencer(monthly_graphs, influencer_indices, device=device)
    else:
        target_node = int(args.target_node)
        print(f"[Target] using provided global_idx={target_node}")

    # guard explain_pos
    input_graphs = monthly_graphs[:-1]
    T = len(input_graphs)
    if not (0 <= int(args.explain_pos) < T):
        raise ValueError(f"explain_pos must be in [0, {T-1}] but got {args.explain_pos}")

    lambdas = parse_float_list(args.lambdas)
    if len(lambdas) == 0:
        raise ValueError("No lambdas provided. Use --lambdas like '0,0.01,0.1'.")

    base_coeffs = {
        "node_feat_size": float(args.base_node_feat_size),
        "edge_size": float(args.base_edge_size),
        "node_feat_ent": float(args.node_feat_ent),
        "edge_ent": float(args.edge_ent),
    }

    run_name = f"Frontier_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_node_{target_node}_pos_{int(args.explain_pos)}"
    if args.mlflow:
        with mlflow.start_run(run_name=run_name, experiment_id=exp_id):
            mlflow.log_params({
                "mode": "frontier",
                "ckpt": str(args.ckpt),
                "end_date": str(args.end_date),
                "num_months": int(args.num_months),
                "metric_numerator": str(args.metric_numerator),
                "metric_denominator": str(args.metric_denominator),
                "target_node": int(target_node),
                "explain_pos": int(args.explain_pos),
                "lambdas": str(args.lambdas),
                "fid_weight": float(args.fid_weight),
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "feat_mask_scope": str(args.feat_mask_scope),
                "edge_mask_scope": str(args.edge_mask_scope),
                "use_subgraph": bool(args.use_subgraph),
                "num_hops": int(args.num_hops),
                "undirected": bool(args.undirected),
                "edge_grouping": str(args.edge_grouping),
                "mask_thr": float(args.mask_thr),
                "base_node_feat_size": float(args.base_node_feat_size),
                "base_edge_size": float(args.base_edge_size),
                "node_feat_ent": float(args.node_feat_ent),
                "edge_ent": float(args.edge_ent),
            })

            df = frontier_sweep(
                model=model,
                monthly_graphs=monthly_graphs,
                node_to_idx=node_to_idx,
                influencer_indices=influencer_indices,
                target_node_idx=int(target_node),
                explain_pos=int(args.explain_pos),
                device=device,
                lambdas=lambdas,
                base_coeffs=base_coeffs,
                fid_weight=float(args.fid_weight),
                epochs=int(args.epochs),
                lr=float(args.lr),
                feat_mask_scope=str(args.feat_mask_scope),
                edge_mask_scope=str(args.edge_mask_scope),
                use_subgraph=bool(args.use_subgraph),
                num_hops=int(args.num_hops),
                undirected=bool(args.undirected),
                edge_grouping=str(args.edge_grouping),
                mask_thr=float(args.mask_thr),
                mlflow_log=True,
                tag_prefix="frontier",
            )
            print(df[["lambda", "fidelity_abs", "sparsity", "feat_mean_gate", "edge_mean_gate"]].to_string(index=False))
    else:
        df = frontier_sweep(
            model=model,
            monthly_graphs=monthly_graphs,
            node_to_idx=node_to_idx,
            influencer_indices=influencer_indices,
            target_node_idx=int(target_node),
            explain_pos=int(args.explain_pos),
            device=device,
            lambdas=lambdas,
            base_coeffs=base_coeffs,
            fid_weight=float(args.fid_weight),
            epochs=int(args.epochs),
            lr=float(args.lr),
            feat_mask_scope=str(args.feat_mask_scope),
            edge_mask_scope=str(args.edge_mask_scope),
            use_subgraph=bool(args.use_subgraph),
            num_hops=int(args.num_hops),
            undirected=bool(args.undirected),
            edge_grouping=str(args.edge_grouping),
            mask_thr=float(args.mask_thr),
            mlflow_log=False,
            tag_prefix="frontier",
        )
        print(df[["lambda", "fidelity_abs", "sparsity", "feat_mean_gate", "edge_mean_gate"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
