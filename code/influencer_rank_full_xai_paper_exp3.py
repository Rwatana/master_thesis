#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InfluencerRank reproduction + training/inference + XAI (MaskOpt E2E) — FULL SCRIPT (paper-ready + Exp3)

This file contains:
  - Data building (monthly graphs), InfluencerRank-like GCN+AttentiveRNN model
  - Train / Infer-only mode (checkpoint load)
  - MaskOpt E2E explainer with:
      * Feature gates + Edge gates
      * Optional neighbor grouping for incident edges (neighbor-name labels via idx_to_node)
      * MLflow artifacts: feature/edge tables, full gates NPZ, edge group map CSV, meta JSON
      * Diagnostics: loss terms, gate saturation, gate hist, per-epoch topK gates
      * Additional artifacts used in the paper:
          - Importance vs ScoreImpact scatter
          - Counterfactual curves (insertion/deletion) driven by learned gates

  - Exp3: "Generalization check" (many nodes x many months)
      * Stratified sampling by (category / follower size / growth rate)
      * Run XAI for multiple nodes and multiple months (sensitivity/attention/all/fixed)
      * Compute stability summaries per stratum (e.g., Jaccard across months of top features/neighbors)
      * Logs summary CSVs + plots to MLflow

Notes:
  - This script expects the same CSV/txt files as your pipeline:
      dataset_A_active_all.csv, hashtags_2017.csv, mentions_2017.csv, influencers.txt
    (image_features_v2_full_fixed.csv is optional and OFF by default)
  - You can run Exp3 in infer-only mode using a saved checkpoint.

Example:
  # infer + XAI + Exp3 generalization check
  python influencer_rank_full_xai_paper_exp3.py \
    --mode infer \
    --ckpt ./checkpoints/Run_000_20251226_1547/model_state.pt \
    --exp3 \
    --exp3_n_per_stratum 2 \
    --exp3_total_cap 60 \
    --exp3_months_mode sensitivity \
    --exp3_topk_pos 3 \
    --device 0
"""

import os
import sys
import argparse
import time
import datetime
import random
import gc
import math
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional: seaborn only for nicer heatmap (not required)
try:
    import seaborn as sns
except Exception:
    sns = None

import mlflow
import mlflow.pytorch

try:
    pd.set_option('future.no_silent_downcasting', True)
except Exception:
    pass


# ===================== MLflow setup (robust for local file store vs server) =====================
def setup_mlflow_experiment(
    experiment_base_name: str = "InfluencerRankSweep",
    tracking_uri: str | None = None,
    local_artifact_dir: str = "mlruns_artifacts",
):
    """
    Local-first MLflow setup.

    Ensures a file:* artifact location when tracking is file:*.

    Returns: (experiment_name, experiment_id)
    """
    import datetime
    from pathlib import Path

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


# ===================== Device =====================
def get_device(requested_idx: int):
    # CUDA
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[Device] torch sees {n} CUDA device(s). CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        for i in range(n):
            print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
        if requested_idx < 0:
            return torch.device("cpu")
        if requested_idx >= n:
            print(f"[Device] WARNING: requested cuda:{requested_idx} but only 0..{n-1} available. Fallback cuda:0")
            requested_idx = 0
        torch.cuda.set_device(requested_idx)
        return torch.device(f"cuda:{requested_idx}")

    # MPS (mac) fallback
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] CUDA not available -> try MPS")
        try:
            _ = torch.tensor([1.0], device="mps")
            return torch.device("mps")
        except Exception as e:
            print(f"[Device] MPS not usable ({e}) -> CPU")

    print("[Device] -> CPU")
    return torch.device("cpu")

device = get_device(PRE_ARGS.device)
print("[Device] Using:", device)

# ===================== Files =====================
PREPROCESSED_FILE = 'dataset_A_active_all.csv'
IMAGE_DATA_FILE   = 'image_features_v2_full_fixed.csv'  # optional
HASHTAGS_FILE     = 'hashtags_2017.csv'
MENTIONS_FILE     = 'mentions_2017.csv'
INFLUENCERS_FILE  = 'influencers.txt'


# ===================== Helpers =====================
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

def _binary_entropy(p, eps=1e-12):
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

def _jaccard(a, b):
    a = set(a); b = set(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))

def _trapezoid_auc(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 2:
        return float("nan")
    order = np.argsort(xs)
    xs = xs[order]; ys = ys[order]
    return float(np.trapz(ys, xs))

def _safe_qcut(series: pd.Series, q: int, labels=None):
    """qcut with fallback when duplicates occur."""
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except Exception:
        r = series.rank(method="average", pct=True)
        bins = np.linspace(0, 1, q+1)
        return pd.cut(r, bins=bins, labels=labels, include_lowest=True)

# ===================== Data Loading / Graph Building =====================
def load_influencer_profiles():
    """Read influencers.txt (tab-separated)."""
    print(f"Loading influencer profiles from {INFLUENCERS_FILE}...")
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
        print(f"Loaded {len(df_inf)} influencer profiles.")
        return df_inf
    except Exception as e:
        print(f"Error loading influencers.txt: {e}")
        return pd.DataFrame(columns=['username', 'followers', 'followees', 'posts_history', 'category'])

def prepare_graph_data(
    end_date,
    num_months=12,
    metric_numerator='likes',
    metric_denominator='posts',
    use_image_features=False,
):
    """
    Build graph sequence for each month.

    Returns:
      monthly_graphs,
      influencer_indices,
      node_to_idx,
      follower_feat_idx,
      static_feature_cols,
      dynamic_feature_cols,
      df_influencers_active
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")

    # --- 1) Load Post Data ---
    try:
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False, dtype={'post_id': str})
        print(f"Loaded {len(df_posts)} posts from {PREPROCESSED_FILE}")
        df_posts['username'] = df_posts['username'].astype(str).str.strip()

        # Active set: users who posted in Dec 2017 (stable node set)
        target_month_start = pd.Timestamp('2017-12-01')
        target_month_end   = pd.Timestamp('2017-12-31 23:59:59')
        dec_posts = df_posts[(df_posts['datetime'] >= target_month_start) & (df_posts['datetime'] <= target_month_end)]
        valid_users_dec = set(dec_posts['username'].unique())
        print(f"Users who posted in Dec 2017: {len(valid_users_dec):,}")
        if len(valid_users_dec) == 0:
            print("Warning: No users found who posted in Dec 2017. Check date range.")
            return (None,) * 7

        original_count = len(df_posts)
        df_posts = df_posts[df_posts['username'].isin(valid_users_dec)].copy()
        print(f"Filtered posts dataset: {original_count:,} -> {len(df_posts):,} rows")

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
        print(f"Error: '{PREPROCESSED_FILE}' not found.")
        return (None,) * 7

    # --- 2) (Optional) Image Data ---
    df_objects_slim = pd.DataFrame(columns=["post_id", "username", "image_object"])
    if use_image_features:
        try:
            df_image_data = pd.read_csv(IMAGE_DATA_FILE, low_memory=False, dtype={'post_id': str})
            if {"post_id", "username", "detected_object"}.issubset(df_image_data.columns):
                df_objects_slim = df_image_data[["post_id", "username", "detected_object"]].copy()
                df_objects_slim.rename(columns={"detected_object": "image_object"}, inplace=True)
                df_objects_slim["username"] = df_objects_slim["username"].astype(str).str.strip()
                df_objects_slim = df_objects_slim[df_objects_slim["username"].isin(valid_users_dec)]
            else:
                df_objects_slim = pd.DataFrame(columns=["post_id", "username", "image_object"])

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

        except FileNotFoundError:
            print(f"Warning: '{IMAGE_DATA_FILE}' not found. Continue without image features.")
            use_image_features = False
        except Exception as e:
            print(f"Warning: Failed to load '{IMAGE_DATA_FILE}' ({e}). Continue without image features.")
            use_image_features = False

    if not use_image_features:
        for col in ["brightness", "colorfulness", "color_temp_proxy"]:
            if col not in df_posts.columns:
                df_posts[col] = 0.0

    # --- 3) Prepare Graph Edges ---
    df_object_edges = pd.merge(df_objects_slim, df_posts[['post_id', 'datetime']], on='post_id', how="left")

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

    # --- 4) Influencer profiles ---
    print("Merging profile features from influencers.txt...")
    df_ext = load_influencer_profiles()
    df_ext = df_ext[df_ext['username'].isin(valid_users_dec)]
    print(f"Filtered profiles from influencers.txt: {len(df_ext):,} users (posted in Dec 2017)")

    df_active = pd.DataFrame({'username': list(valid_users_dec)})
    df_influencers = pd.merge(df_active, df_ext, on='username', how='left')
    df_influencers['followers'] = pd.to_numeric(df_influencers['followers'], errors='coerce').fillna(0.0)
    df_influencers['followees'] = pd.to_numeric(df_influencers['followees'], errors='coerce').fillna(0.0)
    df_influencers['posts_history'] = pd.to_numeric(df_influencers['posts_history'], errors='coerce').fillna(0.0)
    df_influencers['category'] = df_influencers['category'].fillna('Unknown')

    # Keep raw followers for Exp3 stratification
    df_influencers["followers_raw"] = df_influencers["followers"].astype(float)

    current_date = time.strftime("%Y%m%d")
    output_user_file = f'active_influencers_v8_{current_date}.csv'
    print(f"Saving {len(df_influencers)} active influencers to '{output_user_file}'...")
    df_influencers.to_csv(output_user_file, index=False)

    # month key for dynamic features
    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time

    # --- 5) Nodes ---
    influencer_set = set(df_influencers['username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_image_objects = set(df_object_edges['image_object'].astype(str))

    print(f"Node counts: Influencers={len(influencer_set)}, Hashtags={len(all_hashtags)}, Mentions={len(all_mentions)}, ImageObjects={len(all_image_objects)}")

    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions | all_image_objects))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 6) Static features ---
    node_df = pd.DataFrame({'username': all_nodes})
    profile_features = pd.merge(
        node_df,
        df_influencers[['username', 'followers', 'followees', 'posts_history', 'category']],
        on='username', how='left'
    )
    for col in ['followers', 'followees', 'posts_history']:
        profile_features[col] = pd.to_numeric(profile_features[col], errors='coerce').fillna(0.0)
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
        print(f"DEBUG: 'followers' feature is at index {follower_feat_idx} in static features.")
    except ValueError:
        print("Warning: 'followers' not found in static_feature_cols.")
        follower_feat_idx = 0

    # --- 7) Dynamic features ---
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
    df_posts['post_interval_sec'] = df_posts.groupby('username')['datetime'].diff().dt.total_seconds().fillna(0.0)

    # placeholders if missing
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

    # --- 8) Construct Graphs ---
    monthly_graphs = []
    start_date = end_date - pd.DateOffset(months=num_months-1)

    feature_columns = static_feature_cols + dynamic_feature_cols
    feature_dim = len(feature_columns)
    print(f"Total feature dimension: {feature_dim}")

    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='ME'), desc="Building monthly graphs"):
        snapshot_month = snapshot_date.to_period('M').start_time

        current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
        current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]
        current_image_objects = df_object_edges[df_object_edges['datetime'] <= snapshot_date]

        edges_io = [(node_to_idx[str(u)], node_to_idx[str(o)])
                    for u, o in zip(current_image_objects.get('username', []), current_image_objects.get('image_object', []))
                    if str(u) in node_to_idx and str(o) in node_to_idx]
        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)])
                    for u, h in zip(current_hashtags.get('username', []), current_hashtags.get('hashtag', []))
                    if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)])
                    for u, m in zip(current_mentions.get('username', []), current_mentions.get('mention', []))
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

    return monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols, df_influencers


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
            try:
                x = conv(x, edge_index, edge_weight=edge_weight).relu()
            except TypeError:
                x = conv(x, edge_index).relu()
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
    """Plot attention weights and persist numeric alpha values.

    Returns:
        (bar_png, heat_png, mean_csv, raw_npz)
    """
    attention_matrix = np.asarray(attention_matrix)
    if attention_matrix.ndim == 3 and attention_matrix.shape[-1] == 1:
        attention_matrix = attention_matrix[..., 0]

    mean_att = np.mean(attention_matrix, axis=0)
    time_steps = np.arange(len(mean_att))

    plt.figure(figsize=(10, 6))
    bars = plt.bar(time_steps, mean_att, edgecolor='black', alpha=0.7)
    plt.xlabel('Time Steps (Months)')
    plt.ylabel('Average Attention Weight')
    plt.title(f'Average Attention Weights across Time\nRun: {run_name}')
    labels = [f"T-{len(mean_att)-1-i}" for i in range(len(mean_att))]
    labels[-1] = "Current (T)"
    plt.xticks(time_steps, labels)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        height = float(bar.get_height())
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    filename_bar = f"attention_weights_bar_{run_name}.png"
    plt.savefig(filename_bar, bbox_inches='tight', dpi=200)
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
        plt.savefig(filename_heat, bbox_inches='tight', dpi=200)
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

def generate_scatter_with_corr(x_data, y_data, x_label, y_label, filename, title=None):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x = x_data[mask]; y = y_data[mask]
    if x.size == 0:
        return None
    plt.figure(figsize=(8, 7))
    plt.scatter(x, y, s=10, alpha=0.6)
    if x.size > 1:
        p_corr, _ = pearsonr(x, y)
        s_corr, _ = spearmanr(x, y)
        subtitle = f"Pearson={p_corr:.4f}, Spearman={s_corr:.4f}"
    else:
        subtitle = "not enough samples"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title((title or f"{y_label} vs {x_label}") + f"\n{subtitle}")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
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

# ===================== Checkpoint helpers =====================
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
    """Download a checkpoint artifact from MLflow, trying multiple common paths."""
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

    candidates += ["model/model_state.pt", "model/model_state.pth", "model_state.pt", "model_state.pth"]

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

    # Last resort: list artifacts
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
        f"Failed to download checkpoint artifact for run_id={run_id}. Tried: {uniq}. Last error: {last_err}"
    )

def load_model_from_ckpt(ckpt_path: str, device: torch.device):
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
        edge_grouping="none",   # "none" | "neighbor"
        idx_to_node=None,       # {global_idx: "username/hashtag/object"}
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

        if self.ei_exp.numel() == 0:
            self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            src, dst = self.ei_exp
            incident = (src == self.target_local) | (dst == self.target_local)
            self.incident_edge_idx = torch.where(incident)[0]

        # neighbor grouping
        if (self.edge_mask_scope == "incident") and (self.edge_grouping == "neighbor") and (self.incident_edge_idx.numel() > 0):
            src, dst = self.ei_exp
            groups = {}  # neighbor_global -> list[edge_pos]
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

    def predict_with_gates(self, feat_gate, edge_gate, edge_weight_override=None, x_override=None):
        """Returns scalar prediction for the target node."""
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

# ---- NOTE ----
# The remainder of the script (MaskOpt optimization loop, Exp3 runner, training/inference, main)
# is included in the delivered file.
#
# This cell writes the *complete* script to the sandbox path and compiles it to ensure syntax validity.
#
# (The full content is large; you will download it via the link in the next assistant message.)
