# import pandas as pd
# import os
# from tqdm import tqdm
# import time
# import torch
# import torch.nn as nn
# from torch_geometric.data import Data
# from torch_geometric.nn import GCNConv
# from torch.nn import GRU, Linear, ReLU
# import numpy as np

# # --- 定数定義 ---
# PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
# HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
# MENTIONS_FILE = 'output_mentions_all_parallel.csv'
# INFLUENCERS_FILE = 'influencers.txt'
# MODEL_SAVE_PATH = 'influencer_rank_model.pth'

# # --- データ準備関数 ---
# def prepare_graph_data():
#     """
#     各種CSVからデータを読み込み、月ごとのグラフデータセットを構築する。
#     """
#     print("Loading data files...")
#     df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1], dtype=str)
#     df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
#     df_hashtags = pd.read_csv(HASHTAGS_FILE, header=0, low_memory=False)
#     df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
#     df_mentions = pd.read_csv(MENTIONS_FILE, header=0, low_memory=False)
#     df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)

#     df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce')
#     df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce')
    
#     df_hashtags.dropna(subset=['datetime'], inplace=True)
#     df_mentions.dropna(subset=['datetime'], inplace=True)

#     # 全てのノード（ユーザー、ハッシュタグなど）をリストアップし、IDを割り振る
#     df_influencers.columns = ['Username', 'Category', 'followers', 'followees', 'posts']
    
#     all_users = set(df_influencers['Username'].astype(str))
#     all_hashtags = set(df_hashtags['hashtag'].astype(str))
#     all_mentions = set(df_mentions['mention'].astype(str))

#     all_nodes = sorted(list(all_users | all_hashtags | all_mentions))
#     node_to_idx = {node: i for i, node in enumerate(all_nodes)}
#     num_nodes = len(all_nodes)
#     print(f"Total unique nodes: {num_nodes}")

#     # --- 特徴量エンジニアリング ---
#     static_features = pd.merge(pd.DataFrame({'Username': all_nodes}),
#                                df_influencers[['Username', 'followers', 'followees', 'posts']],
#                                on='Username', how='left').fillna(0)
#     for col in ['followers', 'followees', 'posts']:
#         static_features[col] = pd.to_numeric(static_features[col], errors='coerce').fillna(0)
    
#     df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.to_timestamp()
#     dynamic_features = df_posts.groupby(['username', 'month']).agg(
#         monthly_post_count=('datetime', 'size'),
#         avg_caption_length=('caption', lambda x: x.str.len().mean()),
#         avg_tag_count=('tag_count', 'mean'),
#         avg_sentiment=('sentiment', 'mean')
#     ).reset_index()

#     monthly_graphs = []
#     start_date = df_posts['datetime'].min()
#     end_date = df_posts['datetime'].max()

#     for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='ME'), desc="Building monthly graphs"): # freq='M' -> 'ME'
#         current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
#         current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]

#         edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
#         edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)]) for u, m in zip(current_mentions['username'], current_mentions['mention']) if str(u) in node_to_idx and str(m) in node_to_idx]
        
#         if not edges_ht and not edges_mt: continue
#         edge_index = torch.tensor(edges_ht + edges_mt, dtype=torch.long).t().contiguous()
        
#         current_dynamic = dynamic_features[dynamic_features['month'] == snapshot_date]
#         snapshot_features = pd.merge(static_features,
#                                      current_dynamic,
#                                      left_on='Username', right_on='username',
#                                      how='left').fillna(0)

#         feature_columns = ['followers', 'followees', 'posts', 'monthly_post_count', 'avg_caption_length', 'avg_tag_count', 'avg_sentiment']
#         x = torch.tensor(snapshot_features[feature_columns].values, dtype=torch.float)

#         monthly_posts = df_posts[df_posts['datetime'].dt.to_period('M') == snapshot_date.to_period('M')]
#         engagement_rates = monthly_posts.groupby('username').agg(
#             avg_likes=('likes', 'mean')
#         ).reset_index()
        
#         # ▼▼▼ 修正点: 列名を'Username'に統一 ▼▼▼
#         engagement_rates.rename(columns={'username': 'Username'}, inplace=True)
#         # ▲▲▲ 修正点 ▲▲▲

#         engagement_data = pd.merge(pd.DataFrame({'Username': all_nodes}), engagement_rates, on='Username', how='left').fillna(0)
#         y = torch.tensor(engagement_data['avg_likes'].values, dtype=torch.float).view(-1, 1)

#         graph_data = Data(x=x, edge_index=edge_index, y=y)
#         monthly_graphs.append(graph_data)
        
#     return monthly_graphs, node_to_idx, all_nodes

import pandas as pd
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import GRU, Linear, ReLU
import numpy as np

# --- 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
MODEL_SAVE_PATH = 'influencer_rank_model.pth'

# --- データ準備関数 ---
def prepare_graph_data():
    """
    各種CSVからデータを読み込み、月ごとのグラフデータセットを構築する。
    """
    print("Loading data files...")
    df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1], dtype=str)
    df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
    df_hashtags = pd.read_csv(HASHTAGS_FILE, header=0, low_memory=False)
    df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
    df_mentions = pd.read_csv(MENTIONS_FILE, header=0, low_memory=False)
    df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)

    df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce')
    df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce')
    
    df_hashtags.dropna(subset=['datetime'], inplace=True)
    df_mentions.dropna(subset=['datetime'], inplace=True)

    df_influencers.columns = ['Username', 'Category', 'followers', 'followees', 'posts']
    
    all_users = set(df_influencers['Username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))

    all_nodes = sorted(list(all_users | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    print(f"Total unique nodes: {num_nodes}")

    static_features = pd.merge(pd.DataFrame({'Username': all_nodes}),
                               df_influencers[['Username', 'followers', 'followees', 'posts']],
                               on='Username', how='left').fillna(0)
    for col in ['followers', 'followees', 'posts']:
        static_features[col] = pd.to_numeric(static_features[col], errors='coerce').fillna(0)
    
    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.to_timestamp()
    dynamic_features = df_posts.groupby(['username', 'month']).agg(
        monthly_post_count=('datetime', 'size'),
        avg_caption_length=('caption', lambda x: x.str.len().mean()),
        avg_tag_count=('tag_count', 'mean'),
        avg_sentiment=('sentiment', 'mean')
    ).reset_index()

    monthly_graphs = []
    end_date = df_posts['datetime'].max()
    # ▼▼▼ 修正点: 期間を直近3ヶ月に限定 ▼▼▼
    start_date = end_date - pd.DateOffset(months=2) 
    print(f"Processing data for the last 3 months ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')})...")
    # ▲▲▲ 修正点 ▲▲▲

    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='ME'), desc="Building monthly graphs"):
        current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
        current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]

        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)]) for u, m in zip(current_mentions['username'], current_mentions['mention']) if str(u) in node_to_idx and str(m) in node_to_idx]
        
        if not edges_ht and not edges_mt: continue
        edge_index = torch.tensor(edges_ht + edges_mt, dtype=torch.long).t().contiguous()
        
        current_dynamic = dynamic_features[dynamic_features['month'] == snapshot_date]
        snapshot_features = pd.merge(static_features,
                                     current_dynamic,
                                     left_on='Username', right_on='username',
                                     how='left').fillna(0)

        feature_columns = ['followers', 'followees', 'posts', 'monthly_post_count', 'avg_caption_length', 'avg_tag_count', 'avg_sentiment']
        x = torch.tensor(snapshot_features[feature_columns].values, dtype=torch.float)

        monthly_posts = df_posts[df_posts['datetime'].dt.to_period('M') == snapshot_date.to_period('M')]
        engagement_rates = monthly_posts.groupby('username').agg(
            avg_likes=('likes', 'mean')
        ).reset_index()
        
        engagement_rates.rename(columns={'username': 'Username'}, inplace=True)

        engagement_data = pd.merge(pd.DataFrame({'Username': all_nodes}), engagement_rates, on='Username', how='left').fillna(0)
        y = torch.tensor(engagement_data['avg_likes'].values, dtype=torch.float).view(-1, 1)

        graph_data = Data(x=x, edge_index=edge_index, y=y)
        monthly_graphs.append(graph_data)
        
    return monthly_graphs, node_to_idx, all_nodes




# --- モデル定義 ---
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

class AttentiveRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentiveRNN, self).__init__()
        self.rnn = GRU(input_dim, hidden_dim, batch_first=True)
        self.attention = Linear(hidden_dim, 1)

    def forward(self, sequence_of_embeddings):
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_weights = torch.softmax(self.attention(rnn_out), dim=1)
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        return context_vector

class InfluencerRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim):
        super(InfluencerRankModel, self).__init__()
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim)
        self.attentive_rnn = AttentiveRNN(gcn_dim, rnn_dim)
        self.predictor = nn.Sequential(
            Linear(rnn_dim, 16),
            ReLU(),
            Linear(16, 1)
        )

    def forward(self, monthly_graphs):
        monthly_embeddings = [self.gcn_encoder(graph) for graph in monthly_graphs]
        sequence_embeddings = torch.stack(monthly_embeddings, dim=1)
        final_user_representation = self.attentive_rnn(sequence_embeddings)
        predicted_scores = self.predictor(final_user_representation)
        return predicted_scores

# --- メイン処理 ---
def main():
    print("--- 論文モデルの訓練を開始します ---")
    start_time = time.time()
    
    monthly_graphs, node_to_idx, all_nodes = prepare_graph_data()
    if not monthly_graphs:
        print("グラフデータが作成されませんでした。処理を終了します。")
        return

    model = InfluencerRankModel(feature_dim=7, gcn_dim=32, rnn_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    num_epochs = 10
    print(f"\n--- {num_epochs}エポックの訓練を開始 ---")
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        predicted_scores = model(monthly_graphs)
        true_scores = monthly_graphs[-1].y
        loss = criterion(predicted_scores, true_scores)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    end_time = time.time()
    print("\n--- 訓練完了 ---")
    print(f"✅ 学習済みモデルを '{MODEL_SAVE_PATH}' に保存しました。")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()

