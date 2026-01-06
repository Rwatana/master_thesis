# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import pandas as pd
import numpy as np

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

from .config import PREPROCESSED_FILE, IMAGE_DATA_FILE, HASHTAGS_FILE, MENTIONS_FILE, INFLUENCERS_FILE

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

def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts', use_image_features=False):
    """Build graph sequence for each month.

    Returns:
      monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_feature_cols, dynamic_feature_cols
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")

    # --- 1. Load Post Data ---
    try:
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False, dtype={'post_id': str})
        print(f"Loaded {len(df_posts)} posts from {PREPROCESSED_FILE}")
        df_posts['username'] = df_posts['username'].astype(str).str.strip()

        target_month_start = pd.Timestamp('2017-12-01')
        target_month_end   = pd.Timestamp('2017-12-31 23:59:59')
        dec_posts = df_posts[(df_posts['datetime'] >= target_month_start) & (df_posts['datetime'] <= target_month_end)]
        valid_users_dec = set(dec_posts['username'].unique())
        print(f"Users who posted in Dec 2017: {len(valid_users_dec):,}")
        if len(valid_users_dec) == 0:
            print("Warning: No users found who posted in Dec 2017. Check date range.")
            return None, None, None, None, None, None

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
        return None, None, None, None, None, None

    # --- 2. (Optional) Image Data ---
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

    # --- 3. Prepare Graph Edges ---
    df_object_edges = pd.merge(df_objects_slim, df_posts[['post_id', 'datetime']], on='post_id')

    try:
        df_hashtags = pd.read_csv(HASHTAGS_FILE)
        df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
        df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce')
        df_hashtags['username'] = df_hashtags['username'].astype(str).str.strip()
        df_hashtags = df_hashtags[df_hashtags['username'].isin(valid_users_dec)]
    except Exception:
        df_hashtags = pd.DataFrame(columns=['username', 'hashtag', 'datetime'])

    try:
        df_mentions = pd.read_csv(MENTIONS_FILE)
        df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)
        df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce')
        df_mentions['username'] = df_mentions['username'].astype(str).str.strip()
        df_mentions = df_mentions[df_mentions['username'].isin(valid_users_dec)]
    except Exception:
        df_mentions = pd.DataFrame(columns=['username', 'mention', 'datetime'])

    # --- 4. Prepare Influencer Profiles ---
    print("Merging profile features from influencers.txt...")
    df_influencers_external = load_influencer_profiles()
    df_influencers_external = df_influencers_external[df_influencers_external['username'].isin(valid_users_dec)]
    print(f"Filtered profiles from influencers.txt: {len(df_influencers_external):,} users (posted in Dec 2017)")

    df_active_base = pd.DataFrame({'username': list(valid_users_dec)})
    df_influencers = pd.merge(df_active_base, df_influencers_external, on='username', how='left')
    df_influencers['followers'] = df_influencers['followers'].fillna(0)
    df_influencers['followees'] = df_influencers['followees'].fillna(0)
    df_influencers['posts_history'] = df_influencers['posts_history'].fillna(0)
    df_influencers['category'] = df_influencers['category'].fillna('Unknown')

    current_date = time.strftime("%Y%m%d")
    output_user_file = f'active_influencers_v8_{current_date}.csv'
    print(f"Saving {len(df_influencers)} active influencers to '{output_user_file}'...")
    df_influencers.to_csv(output_user_file, index=False)

    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time

    # --- 5. Prepare Nodes ---
    influencer_set = set(df_influencers['username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_image_objects = set(df_object_edges['image_object'].astype(str))

    print(f"Node counts: Influencers={len(influencer_set)}, Hashtags={len(all_hashtags)}, Mentions={len(all_mentions)}, ImageObjects={len(all_image_objects)}")

    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions | all_image_objects))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 6. Static Features ---
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
        print(f"DEBUG: 'followers' feature is at index {follower_feat_idx} in static features.")
    except ValueError:
        print("Warning: 'followers' not found in static_feature_cols.")
        follower_feat_idx = 0

    # --- 7. Dynamic Features ---
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

    # --- 8. Construct Graphs ---
    monthly_graphs = []
    start_date = end_date - pd.DateOffset(months=num_months-1)

    feature_columns = static_feature_cols + dynamic_feature_cols
    feature_dim = len(feature_columns)
    print(f"Total feature dimension: {feature_dim}")

    for snapshot_date in pd.date_range(start_date, end_date, freq='ME'):
        snapshot_month = snapshot_date.to_period('M').start_time

        current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
        current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]
        current_image_objects = df_object_edges[df_object_edges['datetime'] <= snapshot_date]

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

def get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-1):
    from torch.utils.data import TensorDataset
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
