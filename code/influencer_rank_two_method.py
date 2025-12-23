import pandas as pd
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import GRU, Linear, ReLU, Tanh, Dropout
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --- 1. 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
# Datetime added to MODEL_SAVE_PATH to make the model version unique year month day time
MODEL_SAVE_PATH = f'influencer_rank_model_{time.strftime("%Y%m%d")}.pth' # 学習済みモデルの保存先

# --- 2. データ準備関数 (変更なし) ---
def prepare_graph_data():
    """
    各種CSVからデータを読み込み、月ごとのグラフデータセットを構築する。
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
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5):
        super(InfluencerRankModel, self).__init__()
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim, num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        self.predictor = nn.Sequential(
            Linear(rnn_dim, 16),
            ReLU(),
            Dropout(dropout_prob),
            Linear(16, 1)
        )

    def forward(self, monthly_graphs):
        # このフォワードパスは、バッチ処理前の事前計算でのみ使用
        monthly_embeddings = [self.gcn_encoder(graph.x, graph.edge_index) for graph in monthly_graphs]
        return torch.stack(monthly_embeddings, dim=1)

class BatchedListwiseRankingLoss(nn.Module):
    def __init__(self):
        super(BatchedListwiseRankingLoss, self).__init__()
    def forward(self, pred_scores, true_scores):
        pred_probs = F.softmax(pred_scores, dim=1)
        true_probs = F.softmax(true_scores, dim=1)
        log_pred_probs = torch.log(pred_probs + 1e-9)
        loss_per_list = -torch.sum(true_probs * log_pred_probs, dim=1)
        return torch.mean(loss_per_list)


# --- 4. メイン実行ブロック ---
def main():
    # ✅✅✅ --- 戦略選択フラグ --- ✅✅✅
    # True: エンドツーエンド学習 (低速・高負荷) / False: 二段階学習 (高速)
    END_TO_END_TRAINING = False
    
    # --- ハイパーパラメータ定義 ---
    GCN_DIM = 128
    NUM_GCN_LAYERS = 2
    RNN_DIM = 64
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.5
    NUM_EPOCHS = 20
    LISTS_PER_BATCH = 1024
    LIST_SIZE = 10
    BATCH_SIZE = LISTS_PER_BATCH * LIST_SIZE

    print(f"--- Starting Training ---")
    print(f"Mode: {'End-to-End' if END_TO_END_TRAINING else 'Two-Stage'}")
    start_time = time.time()

    # --- データ準備 ---
    monthly_graphs, influencer_indices = prepare_graph_data()
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return

    # --- モデルとオプティマイザ、損失関数の初期化 ---
    model = InfluencerRankModel(
        feature_dim=7, gcn_dim=GCN_DIM, rnn_dim=RNN_DIM,
        num_gcn_layers=NUM_GCN_LAYERS, dropout_prob=DROPOUT_PROB
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BatchedListwiseRankingLoss()
    
    # --- データローダーの準備 ---
    true_scores = monthly_graphs[-1].y
    influencer_true_scores = true_scores[influencer_indices]
    dataset = TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), influencer_true_scores)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # =================================================================================
    # --- ✅ 戦略に応じた訓練プロセスの分岐 ✅ ---
    # =================================================================================

    if not END_TO_END_TRAINING:
        # --- 戦略1: 二段階学習 (Two-Stage Learning) ---
        print("\n--- Strategy: Two-Stage Learning (Fast) ---")
        
        # 1. 事前計算: 訓練ループの前にGCNで特徴量を一度だけ計算
        print("Pre-computing node embeddings...")
        model.gcn_encoder.eval()
        with torch.no_grad():
            monthly_embeddings_list = [model.gcn_encoder(g.x, g.edge_index) for g in tqdm(monthly_graphs, desc="GCN Encoding")]
            sequence_embeddings = torch.stack(monthly_embeddings_list, dim=1)
        
        # 2. 訓練: RNNと予測器のみを訓練
        model.attentive_rnn.train()
        model.predictor.train()
        
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                optimizer.zero_grad()
                
                # 事前計算した特徴量を使用
                batch_sequence_embeddings = sequence_embeddings[batch_indices]

                final_user_representation = model.attentive_rnn(batch_sequence_embeddings)
                predicted_scores = model.predictor(final_user_representation)

                predicted_scores_reshaped = predicted_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                loss = criterion(predicted_scores_reshaped, batch_true_scores_reshaped)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Batch Loss: {total_loss / len(dataloader):.4f}")

    else:
        # --- 戦略2: エンドツーエンド学習 (End-to-End Learning) ---
        print("\n--- Strategy: End-to-End Learning (Slow, High-Memory) ---")
        
        # 1. モデル全体を訓練モードに設定
        model.train()
        
        for epoch in range(NUM_EPOCHS):
            # 2. GCN計算をエポックの開始時に実行
            # (注意: 本来はバッチごとだが、CPUでの実行可能性を考慮しエポックごとに行う妥協案)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Performing GCN forward pass for this epoch...")
            monthly_embeddings_list = [model.gcn_encoder(g.x, g.edge_index) for g in monthly_graphs]
            sequence_embeddings = torch.stack(monthly_embeddings_list, dim=1)
            
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Training Batches"):
                optimizer.zero_grad()
                
                # エポックごとに計算した特徴量を使用
                batch_sequence_embeddings = sequence_embeddings[batch_indices]

                final_user_representation = model.attentive_rnn(batch_sequence_embeddings)
                predicted_scores = model.predictor(final_user_representation)

                predicted_scores_reshaped = predicted_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                loss = criterion(predicted_scores_reshaped, batch_true_scores_reshaped)
                
                # 誤差がGCNまで逆伝播し、モデル全体の重みが更新される
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Batch Loss: {total_loss / len(dataloader):.4f}")


    # --- 訓練完了 ---
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"✅ Model saved to '{MODEL_SAVE_PATH}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()