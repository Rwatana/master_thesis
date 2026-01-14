import pandas as pd
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import GRU, Linear, ReLU, Tanh
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- 1. 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
MODEL_SAVE_PATH = 'influencer_rank_model_listwise_cpu_corrected.pth' # 修正版モデルとして保存

# --- 2. データ準備関数 ---
def prepare_graph_data():
    """
    各種CSVからデータを読み込み,月ごとのグラフデータセットを構築する。
    """
    print("Loading data files...")
    # --- 堅牢なファイル読み込み ---
    try:
        with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [line for line in lines if '===' not in line]
        has_header = 'Username' in lines[0] or '#Followers' in lines[0]
        from io import StringIO
        cleaned_file_like = StringIO("".join(lines))
        if has_header:
            df_influencers = pd.read_csv(cleaned_file_like, sep='\t', dtype=str)
        else:
            df_influencers = pd.read_csv(cleaned_file_like, sep='\t', header=None, dtype=str)
            df_influencers.columns = ['Username', 'Category', 'followers', 'followees', 'posts']
        df_influencers.rename(columns={'#Followers': 'followers', '#Followees': 'followees', '#Posts': 'posts'}, inplace=True)
    except FileNotFoundError:
        print(f"Error: {INFLUENCERS_FILE} not found.")
        return None, None
    df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
    df_hashtags = pd.read_csv(HASHTAGS_FILE, header=0, low_memory=False)
    df_mentions = pd.read_csv(MENTIONS_FILE, header=0, low_memory=False)

    df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
    df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)
    df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce')
    df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce')
    df_hashtags.dropna(subset=['datetime'], inplace=True)
    df_mentions.dropna(subset=['datetime'], inplace=True)

    # --- ノードの準備 ---
    influencer_set = set(df_influencers['Username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    print(f"Total unique nodes: {num_nodes}")
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 特徴量エンジニアリング ---
    static_features = pd.merge(pd.DataFrame({'Username': all_nodes}),
                               df_influencers[['Username', 'followers', 'followees', 'posts']],
                               on='Username', how='left').fillna(0)
    for col in ['followers', 'followees', 'posts']:
        static_features[col] = pd.to_numeric(static_features[col], errors='coerce').fillna(0)

    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time
    dynamic_features = df_posts.groupby(['username', 'month']).agg(
        monthly_post_count=('datetime', 'size'),
        avg_caption_length=('caption', lambda x: x.astype(str).str.len().mean()),
        avg_tag_count=('tag_count', 'mean'),
        avg_sentiment=('sentiment', 'mean')
    ).reset_index()

    # --- グラフ時系列データの構築 ---
    monthly_graphs = []
    end_date = df_posts['datetime'].max()
    start_date = end_date - pd.DateOffset(months=11)

    print(f"Building graph sequence for 12 months ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')})...")
    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='ME'), desc="Building monthly graphs"):
        snapshot_month = snapshot_date.to_period('M').start_time
        current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
        current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]
        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)]) for u, m in zip(current_mentions['username'], current_mentions['mention']) if str(u) in node_to_idx and str(m) in node_to_idx]
        if not edges_ht and not edges_mt: continue
        edge_index = torch.tensor(list(set(edges_ht + edges_mt)), dtype=torch.long).t().contiguous()
        current_dynamic = dynamic_features[dynamic_features['month'] == snapshot_month]
        snapshot_features = pd.merge(static_features, current_dynamic, left_on='Username', right_on='username', how='left').fillna(0)
        feature_columns = ['followers', 'followees', 'posts', 'monthly_post_count', 'avg_caption_length', 'avg_tag_count', 'avg_sentiment']
        x = torch.tensor(snapshot_features[feature_columns].values, dtype=torch.float)
        monthly_posts = df_posts[df_posts['datetime'].dt.to_period('M') == snapshot_date.to_period('M')]
        engagement_rates = monthly_posts.groupby('username').agg(avg_likes=('likes', 'mean')).reset_index()
        engagement_rates.rename(columns={'username': 'Username'}, inplace=True)
        engagement_data = pd.merge(pd.DataFrame({'Username': all_nodes}), engagement_rates, on='Username', how='left').fillna(0)
        y = torch.tensor(engagement_data['avg_likes'].values, dtype=torch.float).view(-1, 1)
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        monthly_graphs.append(graph_data)

    return monthly_graphs, influencer_indices

# --- 3. モデルと損失関数の定義 (変更なし) ---
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        layer_outputs = []
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index).relu()
            layer_outputs.append(x)
        return torch.cat(layer_outputs, dim=1)

class AttentiveRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentiveRNN, self).__init__()
        self.rnn = GRU(input_dim, hidden_dim, batch_first=True)
        self.attention_layer = Linear(hidden_dim, 1)

    def forward(self, sequence_of_embeddings):
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_scores = self.attention_layer(rnn_out).tanh()
        attention_weights = torch.softmax(attention_scores, dim=1)
        return torch.sum(rnn_out * attention_weights, dim=1)

class InfluencerRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2):
        super(InfluencerRankModel, self).__init__()
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim, num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        self.predictor = nn.Sequential(Linear(rnn_dim, 16), ReLU(), Linear(16, 1))

    def forward(self, monthly_graphs):
        # このフォワードパスは,バッチ処理前の事前計算でのみ使用
        monthly_embeddings = [self.gcn_encoder(graph.x, graph.edge_index) for graph in monthly_graphs]
        return torch.stack(monthly_embeddings, dim=1)

class ListwiseRankingLoss(nn.Module):
    def __init__(self):
        super(ListwiseRankingLoss, self).__init__()
    def forward(self, pred_scores, true_scores):
        pred_probs = F.softmax(pred_scores.squeeze(), dim=0)
        true_probs = F.softmax(true_scores.squeeze(), dim=0)
        return -torch.sum(true_probs * torch.log(pred_probs + 1e-9))

# --- 4. メイン実行ブロック ---
def main():
    print("--- Training Paper-Aligned Model on CPU ---")
    start_time = time.time()

    monthly_graphs, influencer_indices = prepare_graph_data()
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return

    NUM_GCN_LAYERS = 2
    model = InfluencerRankModel(feature_dim=7, gcn_dim=32, rnn_dim=64, num_gcn_layers=NUM_GCN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = ListwiseRankingLoss()

    # --- ✅ 変更点 1: GCNによる特徴量埋め込みを事前に計算 ---
    # 論文のフレームワーク通り,まず全期間のグラフデータをGCNに通し,時系列のノード埋め込みを作成します。
    # これを事前計算することで,訓練ループ中の計算量を大幅に削減します。
    print("Pre-computing node embeddings for all months on CPU...")
    model.gcn_encoder.eval() # GCN部分は推論モードで実行
    with torch.no_grad(): # 勾配計算をオフにしてメモリを節約
        # 全ての月のグラフをGCNに通し,結果をリストに格納
        monthly_embeddings_list = [model.gcn_encoder(g.x, g.edge_index) for g in tqdm(monthly_graphs, desc="GCN Encoding")]
        # リストをテンソルに変換 [ノード数, 月数, 埋め込み次元数]
        sequence_embeddings = torch.stack(monthly_embeddings_list, dim=1)

    print(f"Pre-computation complete. Embeddings shape: {sequence_embeddings.shape}")

    # --- データローダーの準備 ---
    true_scores = monthly_graphs[-1].y
    influencer_true_scores = true_scores[influencer_indices]
    dataset = TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), influencer_true_scores)
    dataloader = DataLoader(dataset, batch_size=2048, shuffle=True)

    num_epochs = 20
    print(f"\n--- Starting training for {num_epochs} epochs with batching ---")

    # ✅ 変更点 2: モデルの訓練対象をRNNと予測器に限定
    # 事前計算した特徴量を使うため,GCN部分は固定（訓練しない）し,RNN以降の部分のみを訓練します。
    model.attentive_rnn.train()
    model.predictor.train()
    model.gcn_encoder.eval() # GCN部分は評価モードのまま

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            # ✅ 変更点 3: ダミーデータの代わりに,事前計算した本物の時系列データを入力
            # バッチに含まれるインフルエンサーのインデックスを使い,対応する時系列埋め込みを取得します。
            batch_sequence_embeddings = sequence_embeddings[batch_indices]

            final_user_representation = model.attentive_rnn(batch_sequence_embeddings)
            predicted_scores = model.predictor(final_user_representation)

            loss = criterion(predicted_scores, batch_true_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Average Batch Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"✅ Model saved to '{MODEL_SAVE_PATH}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
