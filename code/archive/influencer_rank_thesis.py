import pandas as pd
import os
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.nn import GRU, Linear, ReLU, Tanh
import numpy as np

# --- 1. Constant Definitions ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
MODEL_SAVE_PATH = 'influencer_rank_model_paper_aligned.pth'

# --- 2. Data Preparation Function ---
def prepare_graph_data():
    """
    EN: This function implements the "Heterogeneous Information Networks" section.
        It constructs a sequence of k heterogeneous networks G = {G1, G2, ..., Gk},
        where each graph Gt is represented by its feature matrix (Xt) and adjacency structure (At).
    JA: この関数は論文の「Heterogeneous Information Networks」セクションを実装します。
        k個の異種ネットワークの時系列データ G = {G1, G2, ..., Gk} を構築します。
        各グラフGtは,特徴量行列(Xt)と隣接構造(At)によって表現されます。
    """
    print("Loading data files...")
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1], dtype=str)
    df_influencers.rename(columns={
            '#Followers': 'followers',
            '#Followees': 'followees',
            '#Posts': 'posts'
        }, inplace=True)
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

    influencer_set = set(df_influencers['Username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    num_nodes = len(all_nodes)
    print(f"Total unique nodes: {num_nodes}")

    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

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
    start_date = end_date - pd.DateOffset(months=11)
    
    print(f"Building graph sequence for 12 months ({start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')})...")
    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='ME'), desc="Building monthly graphs"):
        current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
        current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]

        # EN: Corresponds to "Edge Construction and Adjacency Matrix At".
        # JA: 「Edge Construction and Adjacency Matrix At」に対応します。
        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)]) for u, m in zip(current_mentions['username'], current_mentions['mention']) if str(u) in node_to_idx and str(m) in node_to_idx]
        
        if not edges_ht and not edges_mt: continue
        edge_index = torch.tensor(list(set(edges_ht + edges_mt)), dtype=torch.long).t().contiguous()

        # EN: Corresponds to "Heterogeneous Nodes and Embedded Features Xt".
        # JA: 「Heterogeneous Nodes and Embedded Features Xt」に対応します。
        current_dynamic = dynamic_features[dynamic_features['month'] == snapshot_date]
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
        
    return monthly_graphs, node_to_idx, all_nodes, influencer_indices

# --- 3. Model Definitions ---

class GCNEncoder(nn.Module):
    """
    EN: Implements the "Graph Convolutional Networks" section of the paper.
        Generates node representation Rt from each graph Gt.
    JA: 論文の「Graph Convolutional Networks」セクションを実装。
        各グラフGtから,ノード表現Rtを生成します。
    """
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        # EN: "GCNs then stack multiple GCN layers..."
        # JA: 「GCNs then stack multiple GCN layers...」
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        layer_outputs = []
        
        # EN: "The i-th layer in GCNs then outputs F(i) = σ(Â * F(i-1) * W(i-1))"
        # JA: 「The i-th layer in GCNs then outputs F(i) = σ(Â * F(i-1) * W(i-1))」
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index).relu()
            layer_outputs.append(x)
        
        # EN: "The final output of the GCNs Rt... can be represented as: Rt = [F(1), F(2), ..., F(e)]"
        #     This is the concatenation of all layer outputs.
        # JA: 「The final output of the GCNs Rt... can be represented as: Rt = [F(1), F(2), ..., F(e)]」
        #     これは全層の出力の連結（concatenate）を意味します。
        final_representation = torch.cat(layer_outputs, dim=1)
        
        return final_representation

class AttentiveRNN(nn.Module):
    """
    EN: Implements the "Attentive Recurrent Neural Networks" section of the paper.
    JA: 論文の「Attentive Recurrent Neural Networks」セクションを実装。
    """
    def __init__(self, input_dim, hidden_dim):
        super(AttentiveRNN, self).__init__()
        # EN: "...we employ Gated Recurrent Units (GRUs)..."
        # JA: 「...we employ Gated Recurrent Units (GRUs)...」
        self.rnn = GRU(input_dim, hidden_dim, batch_first=True)
        # EN: "where Fa(·) is a fully-connected layer" (from Eq. 4)
        # JA: 「where Fa(·) is a fully-connected layer」（数式4より）
        self.attention_layer = Linear(hidden_dim, 1)
        self.tanh = Tanh()

    def forward(self, sequence_of_embeddings):
        # EN: rnn_out corresponds to the sequence of hidden states S = [H1, H2, ..., Hk]
        # JA: rnn_outは隠れ状態の時系列データ S = [H1, H2, ..., Hk] に対応します
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        
        # EN: Equation (4): τt = tanh(Fa(Ht))
        # JA: 数式 (4): τt = tanh(Fa(Ht))
        attention_scores = self.attention_layer(rnn_out).tanh()
        
        # EN: Equation (5): αt = exp(τt) / Σ exp(τi)
        # JA: 数式 (5): αt = exp(τt) / Σ exp(τi)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # EN: Equation (6): c = Σ αi · Hi
        # JA: 数式 (6): c = Σ αi · Hi
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        return context_vector

class InfluencerRankModel(nn.Module):
    """
    EN: The overall model framework from Figure 2, integrating all components.
    JA: 図2の全体フレームワーク。全てのコンポーネントを統合したモデルです。
    """
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2):
        super(InfluencerRankModel, self).__init__()
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim, num_gcn_layers)
        
        # EN: The RNN's input dimension must match the GCN's concatenated output dimension.
        # JA: RNNの入力次元は,GCNの連結された出力次元と一致する必要があります。
        rnn_input_dim = gcn_dim * num_gcn_layers
        self.attentive_rnn = AttentiveRNN(rnn_input_dim, rnn_dim)
        
        # EN: Implements "Engagement Score Estimation", Equation (7): ŷu = Fc(ReLU(Fb(cu)))
        # JA: 「Engagement Score Estimation」を実装,数式 (7): ŷu = Fc(ReLU(Fb(cu)))
        self.predictor = nn.Sequential(
            Linear(rnn_dim, 16), # Fb(·)
            ReLU(),
            Linear(16, 1)       # Fc(·)
        )

    def forward(self, monthly_graphs):
        # EN: Generate the sequence of GCN-Encoded Representations, [R1, ..., Rk].
        # JA: GCNでエンコードされた表現の時系列データ [R1, ..., Rk] を生成します。
        monthly_embeddings = [self.gcn_encoder(graph) for graph in monthly_graphs]
        sequence_embeddings = torch.stack(monthly_embeddings, dim=1)
        
        # EN: Get the final context vector 'c' using the attentive RNN.
        # JA: アテンション付きRNNを使い,最終的な文脈ベクトル 'c' を取得します。
        final_user_representation = self.attentive_rnn(sequence_embeddings)
        
        # EN: Predict the final engagement score 'ŷ'.
        # JA: 最終的なエンゲージメントスコア 'ŷ' を予測します。
        predicted_scores = self.predictor(final_user_representation)
        return predicted_scores

# --- 4. Main Execution Block ---
def main():
    print("--- Training Paper-Aligned Model (GCN Concat + Ranking Loss) ---")
    start_time = time.time()
    
    monthly_graphs, node_to_idx, all_nodes, influencer_indices = prepare_graph_data()
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return

    NUM_GCN_LAYERS = 2
    model = InfluencerRankModel(feature_dim=7, gcn_dim=32, rnn_dim=64, num_gcn_layers=NUM_GCN_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    # EN: Implements "List-wise Ranking and Optimization". While the paper proposes a list-wise loss (Eq. 8),
    #     this code uses MarginRankingLoss, a common pairwise alternative that also learns relative order.
    # JA: 「List-wise Ranking and Optimization」を実装。論文はリストワイズ損失 (数式 8) を提案していますが,
    #     このコードでは同様に順序関係を学習する,一般的なペアワイズ損失の代替であるMarginRankingLossを使用します。
    criterion = nn.MarginRankingLoss(margin=1.0)

    num_epochs = 10
    pairs_per_epoch = 5000
    print(f"\n--- Starting training for {num_epochs} epochs ---")
    
    influencer_indices_tensor = torch.tensor(influencer_indices, dtype=torch.long)
    
    for epoch in range(num_epochs):
        model.train()
        
        all_predicted_scores = model(monthly_graphs)
        true_scores = monthly_graphs[-1].y

        positive_indices = torch.randint(0, len(influencer_indices_tensor), (pairs_per_epoch,))
        negative_indices = torch.randint(0, len(influencer_indices_tensor), (pairs_per_epoch,))
        pos_inf_indices = influencer_indices_tensor[positive_indices]
        neg_inf_indices = influencer_indices_tensor[negative_indices]
        pos_preds = all_predicted_scores[pos_inf_indices]
        neg_preds = all_predicted_scores[neg_inf_indices]
        pos_true = true_scores[pos_inf_indices]
        neg_true = true_scores[neg_inf_indices]
        
        target = torch.sign(pos_true - neg_true)
        valid_pairs = target != 0
        
        optimizer.zero_grad()
        loss = criterion(pos_preds[valid_pairs], neg_preds[valid_pairs], target[valid_pairs])
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Ranking Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"✅ Model saved to '{MODEL_SAVE_PATH}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()