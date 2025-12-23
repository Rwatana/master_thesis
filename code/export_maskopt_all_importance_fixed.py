#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Export ALL importance + score_impact for (features + neighbor-nodes via grouped edges).
- No training / no MLflow / no plots. XAI-only.
- For each target node & explain_pos:
    1) Optimize gates (MaskOpt)
    2) Export per-feature: importance + score_impact (ablate gate -> 0)
    3) Export per-neighbor(node): importance + score_impact (ablate group gate -> 0)
    4) Save CSVs + a summary CSV (corr between importance and abs(score_impact))

Recommended:
  - edge_mask_scope=incident
  - edge_grouping=neighbor  (=> "node importance" for neighbors)

Example (Mac / MPS):
  python export_maskopt_all_importance_fixed.py --model-path ./model.pth --device auto

Example (NVIDIA single GPU):
  python export_maskopt_all_importance_fixed.py --model-path ./model.pth --device cuda --cuda 0 --visible 0

Example (NVIDIA multi GPU parallel):
  python export_maskopt_all_importance_fixed.py --model-path ./model.pth --device cuda --visible 0,1 --devices 0,1
"""

import os
import gc
import math
import random
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- pre-torch args: set env vars BEFORE importing torch ----
def _parse_pre_torch_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--visible", type=str, default=None,
                   help="Set CUDA_VISIBLE_DEVICES before torch import, e.g. '0' or '0,1'. (CUDA only)")
    p.add_argument("--mps_fallback", action="store_true",
                   help="Set PYTORCH_ENABLE_MPS_FALLBACK=1 before torch import (Mac/MPS only).")
    args, _ = p.parse_known_args()
    if args.visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible)
    if args.mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return args

_PRE = _parse_pre_torch_args()
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, coalesce, k_hop_subgraph


# ===================== Device selection (Mac MPS / NVIDIA CUDA) =====================
def setup_device(args) -> torch.device:
    """
    Device selection:
      - --device auto: prefer CUDA -> MPS -> CPU
      - --device cuda/mps/cpu: force it (error if unavailable)

    CUDA:
      - --visible: sets CUDA_VISIBLE_DEVICES (already applied pre-torch)
      - --cuda: selects cuda:<index> within the *visible* set

    MPS:
      - --mps_fallback: sets PYTORCH_ENABLE_MPS_FALLBACK=1 (already applied pre-torch)
    """
    want = getattr(args, "device", "auto").lower()

    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available() and torch.backends.mps.is_built()

    if want == "cuda":
        if not cuda_ok:
            raise RuntimeError("device=cuda was requested but torch.cuda.is_available() is False.")
        cuda_index = int(getattr(args, "cuda", 0))
        return torch.device(f"cuda:{cuda_index}")

    if want == "mps":
        if not mps_ok:
            raise RuntimeError("device=mps was requested but torch.backends.mps.is_available() is False.")
        return torch.device("mps")

    if want == "cpu":
        return torch.device("cpu")

    # auto: CUDA -> MPS -> CPU
    if cuda_ok:
        cuda_index = int(getattr(args, "cuda", 0))
        return torch.device(f"cuda:{cuda_index}")
    if mps_ok:
        return torch.device("mps")
    return torch.device("cpu")


def print_device_info(dev: torch.device):
    if dev.type == "cuda":
        i = dev.index if dev.index is not None else 0
        name = torch.cuda.get_device_name(i)
        print(f"[Device] Using CUDA: cuda:{i} ({name})")
    elif dev.type == "mps":
        print("[Device] Using MPS (Apple Silicon)")
    else:
        print("[Device] Using CPU")


# ===================== Your data builder (reuse) =====================
PREPROCESSED_FILE = 'dataset_A_active_all.csv'
IMAGE_DATA_FILE   = 'image_features_v2_full_fixed.csv'
HASHTAGS_FILE     = 'hashtags_2017.csv'
MENTIONS_FILE     = 'mentions_2017.csv'
INFLUENCERS_FILE  = 'influencers.txt'

def load_influencer_profiles():
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
    return df_inf


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

    # Use 'ME' (month end) for new pandas (replaces deprecated 'M')
    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq="ME"), desc="Building monthly graphs"):
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


# ===================== Model (same as your script) =====================
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
            try:
                x = conv(x, edge_index, edge_weight=edge_weight)
            except TypeError:
                x = conv(x, edge_index)
            x = x.relu()
            outs.append(x)
        return torch.cat(outs, dim=1)


class AttentiveRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
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
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.2, projection_dim=128):
        super().__init__()
        self.projection_layer = nn.Sequential(nn.Linear(feature_dim, projection_dim), nn.ReLU())
        self.gcn_encoder = GCNEncoder(projection_dim, gcn_dim, num_gcn_layers)
        combined_dim = (gcn_dim * num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(combined_dim, rnn_dim)
        self.predictor = nn.Sequential(
            nn.Linear(rnn_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, 1)
        )

    def forward(self, gcn_embeddings, raw_features, baseline_scores=None):
        final_rep, attention_weights = self.attentive_rnn(gcn_embeddings)
        raw_output = self.predictor(final_rep).squeeze()
        predicted_scores = F.softplus(raw_output)
        return predicted_scores, attention_weights


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


# ===================== Wrapper with neighbor grouping =====================
class E2EMaskOptWrapper(nn.Module):
    """
    edge_mask_scope:
      - "incident": mask only edges incident to target node (in explain graph)
    edge_grouping:
      - "none": each incident edge is a parameter
      - "neighbor": group incident edges by neighbor node (=> node-importance proxy)
    """
    def __init__(
        self,
        model,
        input_graphs,
        target_node_idx,
        explain_pos,
        device,
        use_subgraph=True,
        num_hops=1,
        undirected=True,
        feat_mask_scope="target",
        edge_mask_scope="incident",
        edge_grouping="neighbor",
        idx_to_node=None,  # list: global_idx -> name
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
        self.idx_to_node = idx_to_node

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

        self.feature_dim = int(self.x_exp.size(1))

        # incident edges
        if self.ei_exp.numel() == 0:
            self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            src, dst = self.ei_exp
            incident = (src == self.target_local) | (dst == self.target_local)
            self.incident_edge_idx = torch.where(incident)[0]

        # neighbor grouping (node-importance proxy)
        self.edge_group_names = None
        self.edge_group_global = None
        self.edge_group_sizes = None
        self._incident_edge_to_group = None  # for neighbor grouping

        if self.edge_mask_scope != "incident":
            # not supported in this exporter
            self.num_edge_params = int(self.ei_exp.size(1))
            self.edge_grouping = "none"
            return

        if self.incident_edge_idx.numel() == 0:
            self.num_edge_params = 0
            return

        if self.edge_grouping == "neighbor":
            src, dst = self.ei_exp
            inc = self.incident_edge_idx
            nbr_local = torch.where(src[inc] == self.target_local, dst[inc], src[inc])  # [E_inc]
            uniq_nbr, inv = torch.unique(nbr_local, sorted=True, return_inverse=True)   # inv: [E_inc] -> group id
            self._incident_edge_to_group = inv
            self.num_edge_params = int(uniq_nbr.numel())

            # convert to global ids if subgraph used
            if self.local2global is not None:
                nbr_global = self.local2global[uniq_nbr]
            else:
                nbr_global = uniq_nbr

            self.edge_group_global = nbr_global.detach().cpu().numpy().astype(int).tolist()

            # names
            names = []
            for gid in self.edge_group_global:
                if (self.idx_to_node is not None) and (0 <= gid < len(self.idx_to_node)):
                    names.append(str(self.idx_to_node[gid]))
                else:
                    names.append(f"node_{gid}")
            self.edge_group_names = names

            # group sizes
            sizes = torch.bincount(inv, minlength=self.num_edge_params).detach().cpu().numpy().astype(int).tolist()
            self.edge_group_sizes = sizes
        else:
            # each incident edge has its own param
            self.num_edge_params = int(self.incident_edge_idx.numel())
            self.edge_group_names = [f"edge_{i}" for i in range(self.num_edge_params)]

    def num_mask_params(self):
        return self.feature_dim, self.num_edge_params

    def _apply_feature_gate(self, x, feat_gate):
        if self.feat_mask_scope in ("all", "subgraph"):
            return x * feat_gate.view(1, -1)

        # target-only
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
            inc = self.incident_edge_idx
            if inc.numel() == 0:
                return w

            if self.edge_grouping == "neighbor":
                gid = self._incident_edge_to_group  # [E_inc]
                w[inc] = edge_gate[gid]
            else:
                w[inc] = edge_gate
            return w

        return edge_gate  # not used here

    def predict_with_gates(self, feat_gate, edge_gate, x_override=None, edge_weight_override=None):
        seq_gcn, seq_raw = [], []
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

        seq_gcn = torch.stack(seq_gcn, dim=0).unsqueeze(0)  # [1,T,D]
        seq_raw = torch.stack(seq_raw, dim=0).unsqueeze(0)  # [1,T,P]
        pred, _ = self.model(seq_gcn, seq_raw, baseline_scores=None)
        return pred.view(())

    @torch.no_grad()
    def pred_unmasked(self):
        feat = torch.ones(self.feature_dim, device=self.device)
        edge = torch.ones(self.num_edge_params, device=self.device) if self.num_edge_params > 0 else None
        return float(self.predict_with_gates(feat, edge).item())


# ===================== MaskOpt (optimize gates) =====================
def _binary_entropy(p, eps=1e-12):
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

@dataclass
class MaskOptConfig:
    epochs: int = 200
    lr: float = 0.05
    fid_weight: float = 2000.0
    edge_size: float = 0.08
    edge_ent: float = 0.15
    feat_size: float = 0.02
    feat_ent: float = 0.15
    budget_feat: Optional[int] = 10
    budget_edge: Optional[int] = 20
    budget_weight: float = 1.0
    early_stop_patience: int = 30
    early_stop_fid: float = 1e-10


def optimize_gates(wrapper: E2EMaskOptWrapper, cfg: MaskOptConfig):
    device = wrapper.device
    Fdim, Edim = wrapper.num_mask_params()

    feat_logits = nn.Parameter(0.1 * torch.randn(Fdim, device=device))
    edge_logits = nn.Parameter(0.1 * torch.randn(Edim, device=device)) if Edim > 0 else None
    params = [feat_logits] + ([edge_logits] if edge_logits is not None else [])
    opt = torch.optim.Adam(params, lr=cfg.lr)

    orig = wrapper.pred_unmasked()
    orig_t = torch.tensor(orig, device=device)

    best = {"loss": float("inf"), "feat": None, "edge": None, "pred": None, "fid": None}
    stall = 0

    def budget_loss(gate, budget, denom):
        if budget is None or gate is None or gate.numel() == 0:
            return gate.new_zeros(())
        return ((gate.sum() - float(budget)) / float(max(1, denom))) ** 2

    for ep in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)
        feat_gate = torch.sigmoid(feat_logits)
        edge_gate = torch.sigmoid(edge_logits) if edge_logits is not None else None

        pred = wrapper.predict_with_gates(feat_gate, edge_gate)
        fid = (pred - orig_t) ** 2

        lf_size = feat_gate.mean()
        lf_ent  = _binary_entropy(feat_gate).mean()
        if edge_gate is not None and edge_gate.numel() > 0:
            le_size = edge_gate.mean()
            le_ent  = _binary_entropy(edge_gate).mean()
        else:
            le_size = pred.new_zeros(())
            le_ent  = pred.new_zeros(())

        lb = pred.new_zeros(())
        if cfg.budget_weight > 0.0:
            lb = budget_loss(feat_gate, cfg.budget_feat, Fdim) + budget_loss(edge_gate, cfg.budget_edge, max(1, Edim))

        loss = (
            cfg.fid_weight * fid
            + cfg.budget_weight * lb
            + cfg.feat_size * lf_size
            + cfg.feat_ent  * lf_ent
            + cfg.edge_size * le_size
            + cfg.edge_ent  * le_ent
        )

        loss.backward()
        opt.step()

        lval = float(loss.item())
        fidv = float(fid.item())
        if lval < best["loss"]:
            best.update({
                "loss": lval,
                "feat": feat_gate.detach().clone(),
                "edge": edge_gate.detach().clone() if edge_gate is not None else None,
                "pred": float(pred.detach().item()),
                "fid": fidv,
            })
            stall = 0
        else:
            stall += 1

        if fidv <= cfg.early_stop_fid and stall >= cfg.early_stop_patience:
            break

    return best, orig


# ===================== Export: ALL elements =====================
@torch.no_grad()
def export_all_elements(
    wrapper: E2EMaskOptWrapper,
    feat_gate: torch.Tensor,
    edge_gate: Optional[torch.Tensor],
    feature_names: list[str],
    impact_reference: str = "masked",   # "masked" or "unmasked"
    min_importance: float = 0.0,
    chunk_log_every: int = 200,
):
    """
    Returns:
      df_feat_all, df_node_all, meta
    df_node_all is "neighbor-node importance" (via grouped edges).
    """
    device = wrapper.device
    Fdim, Edim = wrapper.num_mask_params()

    ones_feat = torch.ones(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None

    if impact_reference == "unmasked":
        base_feat = ones_feat
        base_edge = ones_edge
        pred_base = float(wrapper.predict_with_gates(base_feat, base_edge).item())
    else:
        base_feat = feat_gate
        base_edge = edge_gate
        pred_base = float(wrapper.predict_with_gates(base_feat, base_edge).item())

    # ---------- Features ----------
    feat_np = feat_gate.detach().cpu().numpy().astype(float)
    rows_feat = []
    for j in range(Fdim):
        imp = float(feat_np[j])
        if imp < min_importance:
            continue
        ab_f = base_feat.clone()
        ab_f[j] = 0.0
        pred_abl = float(wrapper.predict_with_gates(ab_f, base_edge).item())
        impact = float(pred_base - pred_abl)
        name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
        rows_feat.append({
            "Type": "Feature",
            "Index": int(j),
            "Name": str(name),
            "Importance": imp,
            "Pred_Base": float(pred_base),
            "Pred_Ablated": float(pred_abl),
            "Score_Impact": impact,
            "Abs_Impact": abs(impact),
        })
        if (j + 1) % chunk_log_every == 0:
            pass

    df_feat = pd.DataFrame(rows_feat)
    if not df_feat.empty:
        df_feat.sort_values(["Abs_Impact", "Importance"], ascending=False, inplace=True)

    # ---------- Neighbor nodes via grouped edges ----------
    rows_node = []
    if Edim > 0 and edge_gate is not None:
        edge_np = edge_gate.detach().cpu().numpy().astype(float)

        names = wrapper.edge_group_names if wrapper.edge_group_names is not None else [f"group_{i}" for i in range(Edim)]
        gids  = wrapper.edge_group_global if wrapper.edge_group_global is not None else [None] * Edim
        sizes = wrapper.edge_group_sizes if wrapper.edge_group_sizes is not None else [None] * Edim

        for g in range(Edim):
            imp = float(edge_np[g])
            if imp < min_importance:
                continue
            ab_e = base_edge.clone() if base_edge is not None else None
            if ab_e is not None:
                ab_e[g] = 0.0
            pred_abl = float(wrapper.predict_with_gates(base_feat, ab_e).item())
            impact = float(pred_base - pred_abl)

            rows_node.append({
                "Type": "NeighborNode",
                "Index": int(g),
                "NeighborGlobalIdx": None if gids[g] is None else int(gids[g]),
                "NeighborName": str(names[g]),
                "GroupSizeEdges": None if sizes[g] is None else int(sizes[g]),
                "Importance": imp,
                "Pred_Base": float(pred_base),
                "Pred_Ablated": float(pred_abl),
                "Score_Impact": impact,
                "Abs_Impact": abs(impact),
            })

    df_node = pd.DataFrame(rows_node)
    if not df_node.empty:
        df_node.sort_values(["Abs_Impact", "Importance"], ascending=False, inplace=True)

    # ---------- meta / quick validity ----------
    def corr(a, b):
        if len(a) < 2:
            return (np.nan, np.nan)
        try:
            pa = np.corrcoef(a, b)[0, 1]
        except Exception:
            pa = np.nan
        try:
            ra = pd.Series(a).rank().to_numpy()
            rb = pd.Series(b).rank().to_numpy()
            sp = np.corrcoef(ra, rb)[0, 1]
        except Exception:
            sp = np.nan
        return (float(pa), float(sp))

    meta = {
        "pred_base": float(pred_base),
        "impact_reference": str(impact_reference),
        "pred_unmasked": float(wrapper.pred_unmasked()),
        "feat_count_exported": int(0 if df_feat.empty else len(df_feat)),
        "node_count_exported": int(0 if df_node.empty else len(df_node)),
    }

    if not df_feat.empty:
        pa, sp = corr(df_feat["Importance"].to_numpy(), df_feat["Abs_Impact"].to_numpy())
        meta["feat_pearson(imp,absImpact)"] = pa
        meta["feat_spearman(imp,absImpact)"] = sp

    if not df_node.empty:
        pa, sp = corr(df_node["Importance"].to_numpy(), df_node["Abs_Impact"].to_numpy())
        meta["node_pearson(imp,absImpact)"] = pa
        meta["node_spearman(imp,absImpact)"] = sp

    return df_feat, df_node, meta


# ===================== Model loader =====================
def _load_model(model_path, feature_dim, device, gcn_dim=128, rnn_dim=128, num_gcn_layers=2, dropout=0.2, proj=128):
    m = HardResidualInfluencerModel(
        feature_dim=feature_dim,
        gcn_dim=gcn_dim,
        rnn_dim=rnn_dim,
        num_gcn_layers=num_gcn_layers,
        dropout_prob=dropout,
        projection_dim=proj
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    m.load_state_dict(ckpt, strict=False)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


# ===================== Worker (single device) =====================
def run_export_for_targets(
    device: torch.device,
    args,
    graphs_payload,
    targets: list[int],
    explain_positions: list[int],
):
    if device.type == "cuda":
        torch.cuda.set_device(device)

    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = graphs_payload
    idx_to_node = [""] * (len(node_to_idx))
    for k, v in node_to_idx.items():
        idx_to_node[v] = k

    feature_names = list(static_cols) + list(dynamic_cols)
    feature_dim = monthly_graphs[0].x.shape[1]
    model = _load_model(
        args.model_path,
        feature_dim=feature_dim,
        device=device,
        gcn_dim=args.gcn_dim,
        rnn_dim=args.rnn_dim,
        num_gcn_layers=args.num_gcn_layers,
        dropout=args.dropout,
        proj=args.proj_dim
    )

    input_graphs = monthly_graphs[:-1]  # Jan..Nov (T=11) if you built 12 months ending Dec

    out_dir = args.outdir
    os.makedirs(out_dir, exist_ok=True)

    summary_rows = []

    for tgt in tqdm(targets, desc=f"[{str(device)}] targets", leave=False):
        for pos in explain_positions:
            wrapper = E2EMaskOptWrapper(
                model=model,
                input_graphs=input_graphs,
                target_node_idx=int(tgt),
                explain_pos=int(pos),
                device=device,
                use_subgraph=not args.no_subgraph,
                num_hops=args.num_hops,
                undirected=True,
                feat_mask_scope="target",
                edge_mask_scope="incident",
                edge_grouping="neighbor",
                idx_to_node=idx_to_node,
            )

            cfg = MaskOptConfig(
                epochs=args.epochs,
                lr=args.lr,
                fid_weight=args.fid_weight,
                edge_size=args.edge_size,
                edge_ent=args.edge_ent,
                feat_size=args.feat_size,
                feat_ent=args.feat_ent,
                budget_feat=None if args.budget_feat < 0 else args.budget_feat,
                budget_edge=None if args.budget_edge < 0 else args.budget_edge,
                budget_weight=args.budget_weight,
                early_stop_patience=args.early_stop_patience,
                early_stop_fid=args.early_stop_fid,
            )

            best, _pred_unmasked = optimize_gates(wrapper, cfg)
            feat_gate = best["feat"]
            edge_gate = best["edge"]

            df_feat, df_node, meta = export_all_elements(
                wrapper=wrapper,
                feat_gate=feat_gate,
                edge_gate=edge_gate,
                feature_names=feature_names,
                impact_reference=args.impact_reference,
                min_importance=args.min_importance,
            )

            # add identifiers
            tgt_name = idx_to_node[tgt] if 0 <= tgt < len(idx_to_node) else f"node_{tgt}"
            for d in [df_feat, df_node]:
                if d is not None and not d.empty:
                    d.insert(0, "TargetGlobalIdx", int(tgt))
                    d.insert(1, "TargetName", str(tgt_name))
                    d.insert(2, "ExplainPos", int(pos))

            tag = f"tgt_{tgt}_pos_{pos}_dev_{str(device).replace(':','_')}"
            feat_csv = os.path.join(out_dir, f"feat_{tag}.csv")
            node_csv = os.path.join(out_dir, f"neighbor_{tag}.csv")

            if df_feat is not None:
                df_feat.to_csv(feat_csv, index=False, float_format="%.8e")
            if df_node is not None:
                df_node.to_csv(node_csv, index=False, float_format="%.8e")

            summary = {
                "device": str(device),
                "TargetGlobalIdx": int(tgt),
                "TargetName": str(tgt_name),
                "ExplainPos": int(pos),
                "pred_unmasked": float(meta.get("pred_unmasked", np.nan)),
                "pred_base": float(meta.get("pred_base", np.nan)),
                "impact_reference": str(meta.get("impact_reference", "")),
                "feat_count_exported": int(meta.get("feat_count_exported", 0)),
                "node_count_exported": int(meta.get("node_count_exported", 0)),
                "feat_pearson(imp,absImpact)": meta.get("feat_pearson(imp,absImpact)", np.nan),
                "feat_spearman(imp,absImpact)": meta.get("feat_spearman(imp,absImpact)", np.nan),
                "node_pearson(imp,absImpact)": meta.get("node_pearson(imp,absImpact)", np.nan),
                "node_spearman(imp,absImpact)": meta.get("node_spearman(imp,absImpact)", np.nan),
                "maskopt_best_loss": float(best.get("loss", np.nan)),
                "maskopt_best_fid": float(best.get("fid", np.nan)),
            }
            summary_rows.append(summary)

            del wrapper, df_feat, df_node
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df_sum = pd.DataFrame(summary_rows)
    dev_tag = str(device).replace(":", "_")
    sum_path = os.path.join(out_dir, f"summary_dev_{dev_tag}.csv")
    df_sum.to_csv(sum_path, index=False)
    return sum_path


# ===================== CLI / Main =====================
def parse_args():
    p = argparse.ArgumentParser()

    # Device selection (single device)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto: CUDA->MPS->CPU")
    p.add_argument("--cuda", type=int, default=0,
                   help="CUDA device index (within CUDA_VISIBLE_DEVICES)")
    p.add_argument("--visible", default=None,
                   help="Set CUDA_VISIBLE_DEVICES (e.g., 0 or 0,1). CUDA only.")
    p.add_argument("--mps_fallback", action="store_true",
                   help="Enable PYTORCH_ENABLE_MPS_FALLBACK=1 for unsupported ops on MPS")

    # Model / output
    p.add_argument("--model-path", type=str, required=True, help="Path to trained .pth")
    p.add_argument("--outdir", type=str, default="xai_export_out", help="Output dir for CSVs")

    # targets / positions
    p.add_argument("--targets", type=str, default="all", choices=["all", "hub", "topk"],
                   help="Which target nodes to export.")
    p.add_argument("--topk-targets", type=int, default=50, help="Used when --targets topk")
    p.add_argument("--target-months", type=str, default="pos_0,pos_1,pos_2",
                   help="Explain positions, e.g. 'pos_0,pos_1,pos_2' (indices in input_graphs).")

    # graph explain settings
    p.add_argument("--no-subgraph", action="store_true", help="Disable k-hop subgraph (slower if graph big).")
    p.add_argument("--num-hops", type=int, default=1)

    # MaskOpt
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--fid-weight", type=float, default=2000.0)
    p.add_argument("--edge-size", type=float, default=0.08)
    p.add_argument("--edge-ent", type=float, default=0.15)
    p.add_argument("--feat-size", type=float, default=0.02)
    p.add_argument("--feat-ent", type=float, default=0.15)
    p.add_argument("--budget-feat", type=int, default=10, help="Set -1 to disable")
    p.add_argument("--budget-edge", type=int, default=20, help="Set -1 to disable")
    p.add_argument("--budget-weight", type=float, default=1.0)
    p.add_argument("--early-stop-patience", type=int, default=30)
    p.add_argument("--early-stop-fid", type=float, default=1e-10)

    # export options
    p.add_argument("--impact-reference", type=str, default="masked", choices=["masked", "unmasked"],
                   help="score_impact base prediction reference.")
    p.add_argument("--min-importance", type=float, default=0.0,
                   help="If >0, skip very small importance elements (for speed).")

    # Multi-GPU parallel (CUDA only!)
    p.add_argument("--devices", type=str, default=None,
                   help="CUDA-only parallel: comma-separated cuda indices (e.g. '0,1,2'). "
                        "If omitted, uses --cuda. If set to 'auto', uses single selected device. "
                        "Ignored when device is not CUDA.")
    p.add_argument("--seed", type=int, default=42)

    # Model dims (must match your checkpoint)
    p.add_argument("--gcn-dim", type=int, default=128)
    p.add_argument("--rnn-dim", type=int, default=128)
    p.add_argument("--num-gcn-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--proj-dim", type=int, default=128)

    return p.parse_args()


def _parse_cuda_devices(devices_str: Optional[str], default_cuda: int) -> list[int]:
    if devices_str is None:
        return [int(default_cuda)]
    s = devices_str.strip().lower()
    if s in ("", "none"):
        return [int(default_cuda)]
    if s == "auto":
        return [int(default_cuda)]
    parts = [p.strip() for p in devices_str.split(",") if p.strip()]
    out = []
    for p in parts:
        if p.lower().startswith("cuda:"):
            p = p.split(":", 1)[1]
        out.append(int(p))
    if not out:
        out = [int(default_cuda)]
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Make sure CUDA_VISIBLE_DEVICES is applied (pre-torch already did it)
    if args.visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible)
    if args.mps_fallback:
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    device0 = setup_device(args)
    print_device_info(device0)

    # ---- build graphs ----
    target_date = pd.to_datetime('2017-12-31')
    payload = prepare_graph_data(
        end_date=target_date,
        num_months=12,
        metric_numerator='likes_and_comments',
        metric_denominator='followers'
    )
    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = payload

    # parse explain positions
    T = len(monthly_graphs[:-1])
    pos_list = []
    for token in args.target_months.split(","):
        token = token.strip()
        if not token:
            continue
        if token.startswith("pos_"):
            pos = int(token.replace("pos_", ""))
        else:
            pos = int(token)
        if pos < 0 or pos >= T:
            raise ValueError(f"ExplainPos out of range: {pos} (T={T})")
        pos_list.append(pos)
    pos_list = sorted(list(set(pos_list)))

    # choose targets
    if args.targets == "all":
        targets = list(map(int, influencer_indices))
    elif args.targets == "hub":
        g = monthly_graphs[-2]
        ei = g.edge_index
        deg = torch.zeros(g.num_nodes, dtype=torch.long)
        if ei.numel() > 0:
            deg.scatter_add_(0, ei[1].cpu(), torch.ones(ei.size(1), dtype=torch.long))
        best = None
        best_d = -1
        for idx in influencer_indices:
            d = int(deg[idx].item())
            if d > best_d:
                best_d = d
                best = int(idx)
        targets = [best]
    else:
        targets = list(map(int, influencer_indices))[: max(1, int(args.topk_targets))]

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Parallel only if CUDA and multiple devices ----
    if device0.type != "cuda":
        # MPS/CPU: ignore --devices and run single-process
        sum_path = run_export_for_targets(device0, args, payload, targets, pos_list)
        print(f"[DONE] summary: {sum_path}")
        return 0

    cuda_ids = _parse_cuda_devices(args.devices, default_cuda=args.cuda)

    if len(cuda_ids) == 1:
        dev = torch.device(f"cuda:{cuda_ids[0]}")
        sum_path = run_export_for_targets(dev, args, payload, targets, pos_list)
        print(f"[DONE] summary: {sum_path}")
        return 0

    # Multi-GPU parallel (CUDA only)
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    chunks = [[] for _ in cuda_ids]
    for i, t in enumerate(targets):
        chunks[i % len(cuda_ids)].append(t)

    ctx = mp.get_context("spawn")
    procs = []
    mgr = ctx.Manager()
    ret = mgr.list()

    def _worker(cuda_idx: int, sub_targets, ret_list):
        dev = torch.device(f"cuda:{int(cuda_idx)}")
        sp = run_export_for_targets(dev, args, payload, sub_targets, pos_list)
        ret_list.append(sp)

    for cuda_idx, sub in zip(cuda_ids, chunks):
        if len(sub) == 0:
            continue
        p = ctx.Process(target=_worker, args=(cuda_idx, sub, ret))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    sum_paths = list(ret)
    dfs = []
    for sp in sum_paths:
        if os.path.exists(sp):
            dfs.append(pd.read_csv(sp))
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        merged = os.path.join(args.outdir, "summary_merged.csv")
        df_all.to_csv(merged, index=False)
        print(f"[DONE] merged summary: {merged}")
    else:
        print("[DONE] (no summaries?)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
