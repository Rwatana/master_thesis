import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# PyTorch and PyTorch Geometricのインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

# --- 1. サンプルデータセットの生成 ---
def create_dummy_dataset(num_influencers=20, posts_per_influencer=50, num_months=12):
    """論文にあるような構造のダミーデータセットを作成する"""
    print("--- 1. サンプルデータセットを生成します ---")
    start_date = datetime(2023, 1, 1)
    influencers = [f'influencer_{i}' for i in range(num_influencers)]
    hashtags = [f'#tag{i}' for i in range(50)]
    users = [f'@user{i}' for i in range(100)]
    
    data = []
    for influencer in influencers:
        followers = np.random.randint(10000, 100000)
        for i in range(posts_per_influencer):
            post_date = start_date + timedelta(days=np.random.randint(0, 30 * num_months))
            caption_tags = ' '.join(np.random.choice(hashtags, size=np.random.randint(1, 4), replace=False))
            caption_users = ' '.join(np.random.choice(users, size=np.random.randint(0, 3), replace=False))
            caption = f"Check this out! {caption_tags} {caption_users}"
            likes = followers * np.random.uniform(0.01, 0.08)
            data.append([influencer, post_date, caption, likes, followers])
            
    df = pd.DataFrame(data, columns=['influencer_id', 'timestamp', 'caption', 'likes', 'followers'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M')
    print("生成されたデータのサンプル:")
    print(df.head())
    print("\n")
    return df

# --- 2. 異種混合ネットワークの構築 ---
# CORRECTED build_hetero_graphs FUNCTION
def build_hetero_graphs(df):
    """
    【修正版】
    月ごとのデータから双方向の異種混合グラフのリストを構築する
    """
    print("--- 2. 月ごとの異種混合ネットワークを構築します ---")
    graphs = {}
    
    influencer_nodes = {name: i for i, name in enumerate(df['influencer_id'].unique())}
    hashtag_nodes = {}
    user_nodes = {}

    unique_months = sorted(df['month'].unique())
    
    for month in unique_months:
        monthly_df = df[df['month'] == month]
        if monthly_df.empty:
            continue

        data = HeteroData()
        
        edge_influencer_hashtag = []
        edge_influencer_user = []

        for _, row in monthly_df.iterrows():
            influencer_idx = influencer_nodes[row['influencer_id']]
            
            tags = [tag for tag in row['caption'].split() if tag.startswith('#')]
            for tag in tags:
                if tag not in hashtag_nodes: hashtag_nodes[tag] = len(hashtag_nodes)
                tag_idx = hashtag_nodes[tag]
                edge_influencer_hashtag.append([influencer_idx, tag_idx])

            mentioned_users = [user for user in row['caption'].split() if user.startswith('@')]
            for user in mentioned_users:
                if user not in user_nodes: user_nodes[user] = len(user_nodes)
                user_idx = user_nodes[user]
                edge_influencer_user.append([influencer_idx, user_idx])
        
        data['influencer'].x = torch.randn(len(influencer_nodes), 32)
        if hashtag_nodes: data['hashtag'].x = torch.randn(len(hashtag_nodes), 16)
        if user_nodes: data['user'].x = torch.randn(len(user_nodes), 16)

        # 元の方向のエッジを追加
        if edge_influencer_hashtag:
            edge_index_ih = torch.tensor(edge_influencer_hashtag, dtype=torch.long).t().contiguous()
            data['influencer', 'uses', 'hashtag'].edge_index = edge_index_ih
            # --- 修正点: 逆方向のエッジを追加 ---
            data['hashtag', 'used_by', 'influencer'].edge_index = edge_index_ih.flip([0])

        if edge_influencer_user:
            edge_index_iu = torch.tensor(edge_influencer_user, dtype=torch.long).t().contiguous()
            data['influencer', 'mentions', 'user'].edge_index = edge_index_iu
            # --- 修正点: 逆方向のエッジを追加 ---
            data['user', 'mentioned_by', 'influencer'].edge_index = edge_index_iu.flip([0])
        
        graphs[month] = data

    print(f"構築された月次グラフの数: {len(graphs)}")
    if graphs:
        print(f"最初の月のグラフ構造: {list(graphs.values())[0]}")
    print("\n")
    return graphs, influencer_nodes

# --- 3. モデルの定義 ---
class GCNEncoder(nn.Module):
    """
    異種混合グラフに変換される「元」となる、単純な（同種混合グラフ用の）GCNエンコーダ
    """
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class InfluencerRank(nn.Module):
    """
    論文のアーキテクチャに基づいたInfluencerRankモデル
    """
    def __init__(self, metadata, hidden_channels=64, gcn_out_channels=32):
        super().__init__()
        
        self.gcn_encoder = to_hetero(
            GCNEncoder(hidden_channels, gcn_out_channels), 
            metadata, 
            aggr='sum'
        )
        self.rnn = nn.GRU(input_size=gcn_out_channels, hidden_size=hidden_channels, batch_first=True)
        self.attention_layer = nn.Linear(hidden_channels, 1)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, graph_sequence, target_influencer_idx):
        influencer_embeddings = []
        for graph in graph_sequence:
            node_embeddings = self.gcn_encoder(graph.x_dict, graph.edge_index_dict)
            influencer_embeddings.append(node_embeddings['influencer'][target_influencer_idx])
            
        sequence_tensor = torch.stack(influencer_embeddings).unsqueeze(0)
        
        rnn_outputs, _ = self.rnn(sequence_tensor)
        rnn_outputs = rnn_outputs.squeeze(0)
        
        attention_weights = F.softmax(self.attention_layer(rnn_outputs), dim=0)
        context_vector = torch.sum(attention_weights * rnn_outputs, dim=0)
        engagement_score = self.prediction_head(context_vector)
        
        return engagement_score.squeeze()

def train_and_evaluate():
    """学習と評価の全プロセスを実行する"""
    dataset_df = create_dummy_dataset()
    monthly_graphs, influencer_mapping = build_hetero_graphs(dataset_df)
    if not monthly_graphs:
        print("グラフを構築できませんでした。処理を終了します。")
        return

    print("--- 3. InfluencerRankモデルを定義します ---")
    model = InfluencerRank(metadata=list(monthly_graphs.values())[0].metadata())
    print(model)
    print("\n")
    
    print("--- 4. モデルの学習を開始します ---")
    influencer_sequences = defaultdict(list)
    influencer_targets = {}

    all_months = sorted(dataset_df['month'].unique())
    if len(all_months) < 2:
        print("学習に必要な期間（2ヶ月以上）のデータがありません。")
        return
        
    train_months = all_months[:-1]
    test_month = all_months[-1]

    for influencer_name, idx in influencer_mapping.items():
        seq = [monthly_graphs[m] for m in train_months if m in monthly_graphs]
        if seq: influencer_sequences[idx] = seq
        
        target_df = dataset_df[(dataset_df['influencer_id'] == influencer_name) & (dataset_df['month'] == test_month)]
        if not target_df.empty:
            engagement_rate = (target_df['likes'].mean() / target_df['followers'].mean())
            influencer_targets[idx] = torch.tensor(engagement_rate, dtype=torch.float32)

    trainable_influencers = [idx for idx in influencer_sequences if idx in influencer_targets]
    if not trainable_influencers:
        print("学習対象のインフルエンサーがいません。")
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(50):
        total_loss = 0
        for influencer_idx in trainable_influencers:
            optimizer.zero_grad()
            sequence = influencer_sequences[influencer_idx]
            target = influencer_targets[influencer_idx]
            predicted_score = model(sequence, influencer_idx)
            loss = loss_fn(predicted_score, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {total_loss / len(trainable_influencers):.6f}")
    print("\n学習が完了しました。")

    print("\n--- 5. 学習済みモデルでインフルエンサーのスコアを予測します ---")
    model.eval()
    with torch.no_grad():
        results = []
        for influencer_idx in trainable_influencers:
            sequence = influencer_sequences[influencer_idx]
            score = model(sequence, influencer_idx).item()
            influencer_name = [name for name, idx in influencer_mapping.items() if idx == influencer_idx][0]
            actual_engagement = influencer_targets[influencer_idx].item()
            results.append((influencer_name, score, actual_engagement))

    results.sort(key=lambda x: x[1], reverse=True)
    print("\nインフルエンサーランキング (予測スコア順):")
    print(f"{'Rank':<5} {'Influencer ID':<15} {'Predicted Score':<20} {'Actual Engagement':<20}")
    print("-" * 65)
    for i, (name, score, actual) in enumerate(results, 1):
        print(f"{i:<5} {name:<15} {score:<20.4f} {actual:<20.4f}")

if __name__ == '__main__':
    train_and_evaluate()