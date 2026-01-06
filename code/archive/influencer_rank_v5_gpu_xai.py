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
from sklearn.metrics import ndcg_score
import emoji
from copy import deepcopy # ✅ XAI: 順列重要度のために追加
import matplotlib.pyplot as plt # ✅ XAI: 可視化のために追加
import seaborn as sns # ✅ XAI: 可視化のために追加

# --- 1. 定数定義 ---
PREPROCESSED_FILE = 'posts_2017.csv'
HASHTAGS_FILE = 'hashtags_2017.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
MODEL_SAVE_PATH = f'influencer_rank_model_{time.strftime("%Y%m%d")}_rich_features_2017_3rd_v5.pth'
FEATURE_NAMES_FILE = 'feature_names.txt' # ✅ XAI: 特徴量名を保存するファイル

# ✅ GPU対応 (1): グローバルデバイスの定義
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- 💻 Using device: {DEVICE} ---")


# --- 2. データ準備関数 (特徴量リストを返すよう変更) ---
def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts'):
    """
    指定された終了日までのNヶ月間のグラフデータセットを構築する。
    ✅ XAI: 戻り値に feature_columns を追加
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")
    
    # --- データ読み込み ---
    df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
    if 'comments' not in df_posts.columns: df_posts['comments'] = 0
    df_hashtags = pd.read_csv(HASHTAGS_FILE, header=0, low_memory=False)
    df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
    df_mentions = pd.read_csv(MENTIONS_FILE, header=0, low_memory=False)
    df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)
    with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f: lines = f.readlines()
    lines = [line for line in lines if '===' not in line]
    from io import StringIO
    df_influencers_master = pd.read_csv(StringIO("".join(lines)), sep='\t', dtype=str)
    df_influencers_master.rename(columns={'#Followers': 'followers', '#Followees': 'followees', '#Posts': 'posts', 'Username': 'username', 'Category': 'category'}, inplace=True)
    df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce').dropna()
    df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce').dropna()
    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time
    active_influencers_set = set(df_posts['username'].unique())
    print(f"Found {len(active_influencers_set):,} active influencers in {PREPROCESSED_FILE}.")
    df_influencers = df_influencers_master[df_influencers_master['username'].isin(active_influencers_set)].copy()
    influencer_set = set(df_influencers['username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 特徴量エンジニアリング ---
    node_df = pd.DataFrame({'username': all_nodes})
    profile_features = pd.merge(node_df, df_influencers[['username', 'followers', 'followees', 'posts', 'category']], on='username', how='left')
    for col in ['followers', 'followees', 'posts']:
        profile_features[col] = pd.to_numeric(profile_features[col], errors='coerce').fillna(0)
    category_dummies = pd.get_dummies(profile_features['category'], prefix='cat', dummy_na=True)
    profile_features = pd.concat([profile_features, category_dummies], axis=1).drop(columns=['category'])
    node_df['type'] = 'other_user'
    node_df.loc[node_df['username'].isin(influencer_set), 'type'] = 'influencer'
    node_df.loc[node_df['username'].isin(all_hashtags), 'type'] = 'hashtag'
    node_type_dummies = pd.get_dummies(node_df['type'], prefix='type')
    static_features = pd.concat([profile_features, node_type_dummies], axis=1)
    df_posts['emoji_count'] = df_posts['caption'].astype(str).apply(emoji.emoji_count)
    df_posts.sort_values(by=['username', 'datetime'], inplace=True)
    df_posts['post_interval_sec'] = df_posts.groupby('username')['datetime'].diff().dt.total_seconds()
    post_categories = [f'post_cat_{i}' for i in range(10)]
    df_posts['post_category'] = np.random.choice(post_categories, size=len(df_posts))
    df_posts['is_ad'] = np.random.choice([0, 1], size=len(df_posts), p=[0.9, 0.1])
    dynamic_agg = df_posts.groupby(['username', 'month']).agg(
        monthly_post_count=('datetime', 'size'), avg_caption_length=('caption', lambda x: x.astype(str).str.len().mean()),
        avg_tag_count=('tag_count', 'mean'), avg_sentiment=('sentiment', 'mean'),
        avg_emoji_count=('emoji_count', 'mean'), avg_post_interval=('post_interval_sec', 'mean'),
        ad_rate=('is_ad', 'mean')).reset_index()
    post_category_rate = df_posts.groupby(['username', 'month'])['post_category'].value_counts(normalize=True).unstack(fill_value=0)
    post_category_rate.columns = [f'rate_{col}' for col in post_category_rate.columns]
    dynamic_features = pd.merge(dynamic_agg, post_category_rate, on=['username', 'month'], how='left')
    
    monthly_graphs = []
    start_date = end_date - pd.DateOffset(months=num_months-1)
    
    feature_columns = list(static_features.drop('username', axis=1).columns) + list(dynamic_features.drop(['username', 'month'], axis=1).columns) + ['feedback_rate']
    
    global FEATURE_DIM
    FEATURE_DIM = len(feature_columns)
    print(f"Total raw feature dimension: {FEATURE_DIM}") 

    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='ME'), desc="Building monthly graphs"):
        snapshot_month = snapshot_date.to_period('M').start_time
        current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
        current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]
        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)]) for u, m in zip(current_mentions['username'], current_mentions['mention']) if str(u) in node_to_idx and str(m) in node_to_idx]
        if not edges_ht and not edges_mt: continue
        edge_index = torch.tensor(list(set(edges_ht + edges_mt)), dtype=torch.long).t().contiguous()
        
        current_dynamic = dynamic_features[dynamic_features['month'] == snapshot_month]
        snapshot_features = pd.merge(static_features, current_dynamic, on='username', how='left')
        snapshot_features['feedback_rate'] = 0.0
        snapshot_features = snapshot_features[feature_columns].fillna(0)
        
        x = torch.tensor(snapshot_features.astype(float).values, dtype=torch.float)        
        monthly_posts_period = df_posts[df_posts['datetime'].dt.to_period('M') == snapshot_date.to_period('M')]
        monthly_agg = monthly_posts_period.groupby('username').agg(
            total_likes=('likes', 'sum'), total_comments=('comments', 'sum'), post_count=('datetime', 'size')).reset_index()
        
        if metric_numerator == 'likes_and_comments': monthly_agg['numerator'] = monthly_agg['total_likes'] + monthly_agg['total_comments']
        else: monthly_agg['numerator'] = monthly_agg['total_likes']
            
        if metric_denominator == 'followers':
            monthly_agg['avg_engagement_per_post'] = (monthly_agg['numerator'] / monthly_agg['post_count']).where(monthly_agg['post_count'] > 0, 0)
            merged_data = pd.merge(monthly_agg, static_features[['username', 'followers']], on='username', how='left')
            merged_data['engagement'] = (merged_data['avg_engagement_per_post'] / merged_data['followers']).where(merged_data['followers'] > 0, 0)
        else:
            merged_data = monthly_agg
            merged_data['engagement'] = (merged_data['numerator'] / merged_data['post_count']).where(merged_data['post_count'] > 0, 0)
        
        engagement_data = pd.merge(pd.DataFrame({'username': all_nodes}), merged_data[['username', 'engagement']], on='username', how='left').fillna(0)
        y = torch.tensor(engagement_data['engagement'].values, dtype=torch.float).view(-1, 1)
        
        graph_data = Data(x=x, edge_index=edge_index, y=y) 
        monthly_graphs.append(graph_data)
        
    # ✅ XAI: feature_columns を返す
    return monthly_graphs, influencer_indices, node_to_idx, feature_columns


# --- GCNEncoder (変更なし) ---
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super(GCNEncoder, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList([GCNConv(in_channels, hidden_channels)] + [GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
    def forward(self, x, edge_index):
        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index).relu()
            layer_outputs.append(x)
        return torch.cat(layer_outputs, dim=1)

# --- AttentiveRNN (✅ XAI: アテンションの重みを返すよう変更) ---
class AttentiveRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentiveRNN, self).__init__()
        self.rnn = GRU(input_dim, hidden_dim, batch_first=True)
        self.attention_layer = Linear(hidden_dim, 1)
    def forward(self, sequence_of_embeddings):
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_scores = self.attention_layer(rnn_out).tanh()
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        final_representation = torch.sum(rnn_out * attention_weights, dim=1)
        
        # ✅ XAI: 最終表現とアテンションの重み [Batch, Seq] を両方返す
        return final_representation, attention_weights.squeeze(-1)

# --- InfluencerRankModel (✅ XAI: アテンションの重みを返すよう変更) ---
class InfluencerRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5, projection_dim=128):
        super(InfluencerRankModel, self).__init__()
        print(f"\nInitializing Model:")
        print(f"  Raw Features (Input): {feature_dim}")
        print(f"  Projection Dim (GCN Input): {projection_dim}")
        print(f"  GCN Hidden Dim: {gcn_dim} (x{num_gcn_layers} layers -> Output: {gcn_dim * num_gcn_layers})")
        print(f"  RNN Hidden Dim (Predictor Input): {rnn_dim}")

        self.projection_layer = nn.Sequential(
            Linear(feature_dim, projection_dim),
            ReLU()
        )
        self.gcn_encoder = GCNEncoder(projection_dim, gcn_dim, num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        self.predictor = nn.Sequential(Linear(rnn_dim, 16), ReLU(), Dropout(dropout_prob), Linear(16, 1))

    # ✅ GPU対応 (2): forwardメソッドで `device` を受け取るように変更
    def forward(self, graph_sequence, target_indices, device, debug_print=False):
        """
        モデルのフォワードパス全体。
        ✅ XAI: 予測スコアとアテンションの重みを返す
        """
        if debug_print: 
            print(f"\n--- 🐛 DEBUG: Model Forward Pass (BatchSize={len(target_indices)}) ---")
            print(f"Input: {len(graph_sequence)} graphs, {len(target_indices)} target indices. Target Device: {device}")

        gcn_inputs = []
        # --- 1. 射影層 (グラフごと) ---
        for i, g in enumerate(graph_sequence):
            # ✅ GPU対応 (3): グラフの 'x' テンソルをGPUに転送
            g_x = g.x.to(device) 
            if i == 0 and debug_print: print(f"[1] Projection Layer Input (g.x shape, T=0): {g_x.shape} (on {g_x.device})")
            projected_x = self.projection_layer(g_x)
            if i == 0 and debug_print: print(f"[1] Projection Layer Output (shape, T=0):      {projected_x.shape}")
            gcn_inputs.append(projected_x)

        # --- 2. GCNエンコーダ (グラフごと) ---
        sequence_embeddings_list = []
        for i, (g, projected_x) in enumerate(zip(graph_sequence, gcn_inputs)):
            # ✅ GPU対応 (4): グラフの 'edge_index' テンソルをGPUに転送
            g_edge_index = g.edge_index.to(device)
            if i == 0 and debug_print: print(f"\n[2] GCN Encoder Input (projected_x, T=0): {projected_x.shape}")
            if i == 0 and debug_print: print(f"[2] GCN Encoder Input (edge_index, T=0):  {g_edge_index.shape} (on {g_edge_index.device})")
            gcn_out = self.gcn_encoder(projected_x, g_edge_index)
            if i == 0 and debug_print: print(f"[2] GCN Encoder Output (shape, T=0):      {gcn_out.shape}")
            sequence_embeddings_list.append(gcn_out)
        
        # [Seq_Len, Num_Nodes, GCN_Out_Feat]
        sequence_embeddings = torch.stack(sequence_embeddings_list)
        if debug_print: print(f"\n[3] Stacked GCN Embeddings (Seq, AllNodes, Feat): {sequence_embeddings.shape} (on {sequence_embeddings.device})")
        
        # --- 3. ターゲット選択 & 転置 ---
        # [Batch_Size, Seq_Len, GCN_Out_Feat]
        target_embeddings = sequence_embeddings[:, target_indices].permute(1, 0, 2)
        if debug_print: print(f"[3] Target Embeddings (Batch, Seq, Feat):        {target_embeddings.shape}")

        # --- 4. Attentive RNN ---
        if debug_print: print(f"\n[4] Attentive RNN Input (Batch, Seq, Feat):  {target_embeddings.shape}")
        # ✅ XAI: アテンションの重みも受け取る
        final_representation, attention_weights = self.attentive_rnn(target_embeddings)
        if debug_print: print(f"[4] Attentive RNN Output (Batch, Feat): {final_representation.shape}")
        if debug_print: print(f"[4] Attention Weights (Batch, Seq): {attention_weights.shape}")

        # --- 5. 予測層 ---
        if debug_print: print(f"\n[5] Predictor Input (Batch, Feat):  {final_representation.shape}")
        predicted_scores = self.predictor(final_representation)
        if debug_print: print(f"[5] Predictor Output (Batch, 1): {predicted_scores.shape}")
        if debug_print: print(f"--- 🐛 End Debug ---")

        # ✅ XAI: スコアとアテンションの重みを返す
        return predicted_scores, attention_weights


# --- 損失関数と評価関数 ---
class BatchedListwiseRankingLoss(nn.Module):
    def __init__(self):
        super(BatchedListwiseRankingLoss, self).__init__()
    def forward(self, pred_scores, true_scores):
        pred_probs = F.softmax(pred_scores, dim=1)
        true_probs = F.softmax(true_scores, dim=1)
        return -torch.sum(true_probs * torch.log(pred_probs + 1e-9), dim=1).mean()

class PointwiseRankingLoss(nn.Module):
    def __init__(self):
        super(PointwiseRankingLoss, self).__init__()
    def forward(self, pred_scores, true_scores):
        return F.mse_loss(pred_scores, true_scores)
        
def display_relevance_distribution(scores, title):
    scores_series = pd.Series(scores)
    relevance_series = scores_series.apply(assign_relevance_levels)
    counts = relevance_series.value_counts().sort_index()
    percentages = relevance_series.value_counts(normalize=True).sort_index() * 100
    dist_df = pd.DataFrame({'Relevance': counts.index, 'Count': counts.values, 'Percentage': percentages.values}).set_index('Relevance')
    dist_df = dist_df.reindex(range(6), fill_value=0)
    dist_df['Percentage'] = dist_df['Percentage'].map('{:.2f}%'.format)
    print(f"\n--- {title} ---")
    print(dist_df)

def assign_relevance_levels(engagement_rate):
    """ 論文(Table 2)に基づき、単一のエンゲージメント率を関連レベルに変換する """
    if engagement_rate >= 0.10: return 5
    if engagement_rate >= 0.07: return 4
    if engagement_rate >= 0.05: return 3
    if engagement_rate >= 0.03: return 2
    if engagement_rate >= 0.01: return 1
    return 0

def calculate_rbp(true_scores_in_predicted_order, p=0.95):
    rbp_score = 0
    max_score = true_scores_in_predicted_order.max()
    if max_score == 0: return 0.0
    normalized_scores = true_scores_in_predicted_order / max_score
    for i, relevance in enumerate(normalized_scores):
        rbp_score += (p ** i) * relevance
    return (1 - p) * rbp_score


# --- train_and_save_model (✅ XAI: 特徴量名を保存するよう変更) ---
def train_and_save_model():
    END_TO_END_TRAINING = False
    GCN_DIM = 128
    NUM_GCN_LAYERS = 2
    RNN_DIM = 64
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.5
    NUM_EPOCHS = 200 # エポック数を増やす (論文に合わせて)
    LISTS_PER_BATCH = 1024 # 論文の設定
    LIST_SIZE = 10 # 論文の設定
    BATCH_SIZE = LISTS_PER_BATCH * LIST_SIZE
    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers' # 論文の定義 (Eq 1) に厳密に合わせる
    PROJECTION_DIM = 128

    print(f"--- Starting Training ---")
    start_time = time.time()
    
    latest_date = pd.to_datetime('2017-12-31')
    # ✅ XAI: feature_columns を受け取る
    monthly_graphs, influencer_indices, _, feature_columns = prepare_graph_data(
        end_date=latest_date, num_months=12, 
        metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR
    )
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return

    model = InfluencerRankModel(
        feature_dim=FEATURE_DIM, 
        gcn_dim=GCN_DIM, 
        rnn_dim=RNN_DIM, 
        num_gcn_layers=NUM_GCN_LAYERS, 
        dropout_prob=DROPOUT_PROB,
        projection_dim=PROJECTION_DIM
    )
    # ✅ GPU対応 (5): モデルを定義したデバイス(GPU)に転送
    model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_listwise = BatchedListwiseRankingLoss()
    criterion_pointwise = PointwiseRankingLoss() # MSELoss
    alpha = 1 # ListwiseとPointwiseのバランス (調整可能)
    
    true_scores = monthly_graphs[-1].y[influencer_indices]
    # .cpu() は、GPUテンソルからNumpy配列に変換する前に必要
    display_relevance_distribution(true_scores.squeeze().cpu().numpy(), "📊 Training Data Ground Truth Distribution")
    
    # DataLoader は CPU 上でインデックスとスコアを保持します
    dataset = TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), true_scores)
    # pin_memory=True は、CPUからGPUへのデータ転送を高速化する（オプション）
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True if DEVICE.type == 'cuda' else False)
    
    if not END_TO_END_TRAINING:
        # --- 戦略1: 2段階学習 ---
        print("\n--- Strategy: Two-Stage Learning (Fast) ---")
        model.projection_layer.eval()
        model.gcn_encoder.eval()
        
        with torch.no_grad():
            sequence_embeddings_list = []
            print("Running Projection + GCN Encoding (Two-Stage)...")
            debug_print_2stage = True 
            
            for i, g in enumerate(tqdm(monthly_graphs, desc="Projection + GCN Encoding")):
                # ✅ GPU対応 (6): 2段階学習の前計算でもデータをGPUに転送
                g_x = g.x.to(DEVICE)
                g_edge_index = g.edge_index.to(DEVICE)
                
                if i == 0 and debug_print_2stage: print(f"[1] Projection Input (T=0): {g_x.shape} (on {g_x.device})")
                projected_x = model.projection_layer(g_x)
                if i == 0 and debug_print_2stage: print(f"[1] Projection Output (T=0): {projected_x.shape}")
                
                if i == 0 and debug_print_2stage: print(f"[2] GCN Input (T=0): {projected_x.shape}")
                gcn_out = model.gcn_encoder(projected_x, g_edge_index)
                if i == 0 and debug_print_2stage: print(f"[2] GCN Output (T=0): {gcn_out.shape}")
                
                sequence_embeddings_list.append(gcn_out)
                
            # この時点で sequence_embeddings は GPU 上にある
            sequence_embeddings = torch.stack(sequence_embeddings_list)
            if debug_print_2stage: print(f"[3] Stacked GCN Embeddings (Seq, AllNodes, Feat): {sequence_embeddings.shape} (on {sequence_embeddings.device})")

        model.attentive_rnn.train()
        model.predictor.train()
        
        print("\n--- 🐛 DEBUG: Two-Stage RNN/Predictor (1 Batch) ---")
        batch_indices, _ = next(iter(dataloader))
        # batch_indices (CPU) を使って sequence_embeddings (GPU) からスライスする
        debug_target_embeddings = sequence_embeddings[:, batch_indices].permute(1, 0, 2)
        print(f"[3] Target Embeddings (Batch, Seq, Feat): {debug_target_embeddings.shape} (on {debug_target_embeddings.device})")
        # ✅ XAI: 戻り値を2つ受け取る
        debug_rnn_out, debug_attn = model.attentive_rnn(debug_target_embeddings)
        print(f"[4] Attentive RNN Output (Batch, Feat): {debug_rnn_out.shape}")
        print(f"[4] Attention Weights (Batch, Seq): {debug_attn.shape}")
        debug_pred_out = model.predictor(debug_rnn_out)
        print(f"[5] Predictor Output (Batch, 1): {debug_pred_out.shape}")
        print("--- 🐛 End Debug ---")

        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                # ✅ GPU対応 (7): バッチデータをGPUに転送
                # (sequence_embeddings は既にGPU上にある)
                batch_indices = batch_indices.to(DEVICE)
                batch_true_scores = batch_true_scores.to(DEVICE)
                
                optimizer.zero_grad()
                batch_sequence_embeddings = sequence_embeddings[:, batch_indices].permute(1, 0, 2)
                # ✅ XAI: 戻り値を2つ受け取る (アテンションは使わない)
                final_user_representation, _ = model.attentive_rnn(batch_sequence_embeddings)
                predicted_scores = model.predictor(final_user_representation)
                
                predicted_scores_reshaped = predicted_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                
                loss_listwise = criterion_listwise(predicted_scores_reshaped, batch_true_scores_reshaped)
                loss_pointwise = criterion_pointwise(predicted_scores.squeeze(), batch_true_scores.squeeze())
                loss = loss_listwise + alpha * loss_pointwise
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Batch Loss: {total_loss / len(dataloader):.4f}")
    else:
        # --- 戦略2: エンドツーエンド学習 ---
        print("\n--- Strategy: End-to-End Learning (Slow, High-Memory) ---")
        model.train() 
        
        batch_indices_debug, _ = next(iter(dataloader))
        print("\n--- 🐛 DEBUG: End-to-End (1 Batch) ---")
        # ✅ GPU対応 (8): model.forward に `device=DEVICE` を渡す
        # ✅ XAI: 戻り値を2つ受け取る
        _, _ = model(monthly_graphs, batch_indices_debug.to(DEVICE), device=DEVICE, debug_print=True) 
        print("--- 🐛 End Debug ---")
            
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                # ✅ GPU対応 (7): バッチデータをGPUに転送
                batch_indices = batch_indices.to(DEVICE)
                batch_true_scores = batch_true_scores.to(DEVICE)
                
                optimizer.zero_grad()
                
                # ✅ GPU対応 (8): model.forward に `device=DEVICE` を渡す
                # ✅ XAI: 戻り値を2つ受け取る (アテンションは使わない)
                predicted_scores, _ = model(monthly_graphs, batch_indices, device=DEVICE, debug_print=False) 

                predicted_scores_reshaped = predicted_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                loss_listwise = criterion_listwise(predicted_scores_reshaped, batch_true_scores_reshaped)
                loss_pointwise = criterion_pointwise(predicted_scores.squeeze(), batch_true_scores.squeeze())
                loss = loss_listwise + alpha * loss_pointwise

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Batch Loss: {total_loss / len(dataloader):.4f}")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    # ✅ XAI: 特徴量名をファイルに保存
    with open(FEATURE_NAMES_FILE, 'w', encoding='utf-8') as f:
        for feature_name in feature_columns:
            f.write(f"{feature_name}\n")
    print(f"✅ Feature names saved to '{FEATURE_NAMES_FILE}'")

    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"✅ Model saved to '{MODEL_SAVE_PATH}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")


# --- ✅ XAI: 順列重要度（Permutation Importance）の分析関数 ---
def run_permutation_importance(model, base_graphs, target_indices, feature_names, ground_truth_graph, base_ndcg_100, device):
    """
    順列重要度（Permutation Importance）を計算し、表示する。
    NDCG@100 の低下幅を重要度スコアとする。
    """
    print("\n" + "="*50)
    print("🔬 D. FEATURE IMPORTANCE (PERMUTATION)")
    print("="*50)
    print(f"Calculating importance for {len(feature_names)} features...")
    print(f"Baseline NDCG@100: {base_ndcg_100:.4f}")
    
    model.eval()
    importances = {}
    
    # ---
    # ✅ FIX: ここが修正点です。
    # assign_relevance_levelsが配列全体に適用されエラーになっていたのを修正。
    # pd.Series.apply() を使って、各要素（スコア）ごとに関数を適用します。
    # ---
    # 1. スコアのNumPy配列を取得
    true_scores_numpy = ground_truth_graph.y[target_indices].squeeze().cpu().numpy()
    # 2. Pandas Seriesに変換し、.apply()で各要素に関数を適用
    true_relevance_series = pd.Series(true_scores_numpy).apply(assign_relevance_levels)
    # 3. ndcg_scoreが期待する 2D-array 形式に変換
    true_relevance_for_ndcg = true_relevance_series.values.reshape(1, -1)

    # ターゲットインデックスをGPUテンソルに変換（フォワードパス用）
    target_indices_tensor = torch.tensor(target_indices, dtype=torch.long).to(device)

    for i, feature_name in enumerate(tqdm(feature_names, desc="Permutation Importance")):
        # グラフのリストをディープコピーして、元のデータを変更しないようにする
        shuffled_graphs = deepcopy(base_graphs)
        
        # --- i番目の特徴量を、全ノード・全タイムステップでシャッフル ---
        for g in shuffled_graphs:
            # g.x は [Num_Nodes, Num_Features]
            feature_column = g.x[:, i].clone()
            
            # この特徴量列をランダムに並び替える
            permuted_indices = torch.randperm(feature_column.shape[0])
            g.x[:, i] = feature_column[permuted_indices]
        
        # --- シャッフルしたグラフで推論を実行 ---
        with torch.no_grad():
            # 戻り値は (スコア, アテンション)
            predicted_scores_shuffled, _ = model(shuffled_graphs, target_indices_tensor, device=device, debug_print=False)
        
        # --- 新しいNDCGを計算 ---
        predicted_scores_for_ndcg = predicted_scores_shuffled.squeeze().cpu().numpy().reshape(1, -1)
        
        ndcg_k = 100
        if ndcg_k > len(target_indices):
            ndcg_k = len(target_indices)
            
        new_ndcg_100 = ndcg_score(true_relevance_for_ndcg, predicted_scores_for_ndcg, k=ndcg_k)
        
        # 重要度 = ベースラインからのスコア低下幅
        importance_score = base_ndcg_100 - new_ndcg_100
        importances[feature_name] = importance_score

    # --- 結果を表示 ---
    df_importance = pd.DataFrame(importances.items(), columns=['Feature', 'Importance (NDCG@100 Drop)'])
    df_importance = df_importance.sort_values(by='Importance (NDCG@100 Drop)', ascending=False)
    
    print("\n--- Top 15 Most Important Features ---")
    print(df_importance.head(15).to_string(index=False))
    
    print("\n--- Top 10 Least Important Features ---")
    print(df_importance.tail(10).to_string(index=False))
    
    return df_importance

# --- run_inference (✅ XAI: 分析と可視化を追加) ---
def run_inference():
    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'
    PROJECTION_DIM = 128
    
    print("--- 📈 Starting Inference Process ---")
    start_time = time.time()
    params = {'GCN_DIM': 128, 'NUM_GCN_LAYERS': 2, 'RNN_DIM': 64, 'DROPOUT_PROB': 0.5}

    # ✅ XAI: 特徴量名はファイルからロードするので、_ で受ける
    latest_date = pd.to_datetime('2017-12-31')
    predict_graphs, predict_indices, node_to_idx, _ = prepare_graph_data(
        end_date=latest_date, num_months=12, 
        metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR
    )
    
    # ✅ XAI: 特徴量名をファイルからロード
    try:
        with open(FEATURE_NAMES_FILE, 'r', encoding='utf-8') as f:
            feature_columns = [line.strip() for line in f]
        print(f"Successfully loaded {len(feature_columns)} feature names from '{FEATURE_NAMES_FILE}'.")
    except FileNotFoundError:
        print(f"Error: Feature file '{FEATURE_NAMES_FILE}' not found. Please run training first.")
        return

    model = InfluencerRankModel(
        feature_dim=FEATURE_DIM, # グローバル変数
        gcn_dim=params['GCN_DIM'], 
        rnn_dim=params['RNN_DIM'], 
        num_gcn_layers=params['NUM_GCN_LAYERS'], 
        dropout_prob=params['DROPOUT_PROB'],
        projection_dim=PROJECTION_DIM
    )
    
    # ✅ GPU対応 (9): モデルをロードする前に、まずGPUに転送する
    model.to(DEVICE) 
    
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        print(f"Successfully loaded model from '{MODEL_SAVE_PATH}' (on {DEVICE})")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_SAVE_PATH}'. Please run training first.")
        return

    # 予測に使うのは11ヶ月分のグラフ (T=1...11)
    inference_input_graphs = predict_graphs[:-1] 
    # 正解ラベルは12ヶ月目のグラフ (T=12)
    ground_truth_graph = predict_graphs[-1]

    model.eval()
    with torch.no_grad():
        print("\n---  BUG DEBUG: Inference ---")
        
        # ターゲットインデックスをPythonリストからTensorに変換
        # (forwardメソッドは内部でリストを処理できるが、明示的にTensorにしても良い)
        # ここでは元のPythonリスト `predict_indices` をそのまま使う
        
        # ✅ GPU対応 (10): 推論時も model.forward に `device=DEVICE` を渡す
        # predict_indices はPythonリストなので、そのまま渡してOK
        # ✅ XAI: 予測スコアとアテンションの重みを受け取る
        predicted_scores, attention_weights = model(
            inference_input_graphs, predict_indices, device=DEVICE, debug_print=True
        )
        print("--- 🐛 End Debug ---")

    
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    influencer_usernames = [idx_to_node[idx] for idx in predict_indices]
    
    # ✅ GPU対応 (11): 結果をNumpy/Pandasで処理する前に .cpu() でCPUに戻す
    predicted_scores_cpu = predicted_scores.squeeze().cpu().numpy()
    true_scores_cpu = ground_truth_graph.y[predict_indices].squeeze().cpu().numpy()
    
    df_results = pd.DataFrame({
        'Username': influencer_usernames,
        'Predicted_Score': predicted_scores_cpu,
        'True_Score': true_scores_cpu
    })
    
    mae = (df_results['Predicted_Score'] - df_results['True_Score']).abs().mean()
    mse = ((df_results['Predicted_Score'] - df_results['True_Score']) ** 2).mean()
    rmse = np.sqrt(mse)
    
    # NDCG計算のために、予測スコアと真の関連性レベルを準備
    df_results['Relevance'] = df_results['True_Score'].apply(assign_relevance_levels)
    true_relevance = df_results['Relevance'].values.reshape(1, -1)
    predicted_scores_for_ndcg = df_results['Predicted_Score'].values.reshape(1, -1)
    
    ndcg_results = {}
    k_values = [1, 10, 50, 100, 200]
    for k in k_values:
        if k > len(df_results): continue
        ndcg_results[f'NDCG@{k}'] = ndcg_score(true_relevance, predicted_scores_for_ndcg, k=k)
    
    df_sorted_by_pred = df_results.sort_values(by='Predicted_Score', ascending=False)
    true_scores_in_pred_order = df_sorted_by_pred['True_Score'].values
    rbp_val = calculate_rbp(true_scores_in_pred_order, p=0.95)

    df_results['Predicted_Rank'] = df_results['Predicted_Score'].rank(ascending=False, method='first').astype(int)
    
    print("\n🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆")
    print(df_results.sort_values(by='Predicted_Rank')[['Username', 'Predicted_Rank', 'Predicted_Score', 'True_Score']].head(20).to_string(index=False))
    
    
    print("\n\n" + "="*50)
    print("📊 MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    display_relevance_distribution(df_results['True_Score'], "📈 Inference Data Ground Truth Distribution")
    display_relevance_distribution(df_results['Predicted_Score'], "🤖 Inference Data Predicted Distribution")

    print("\n🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---")
    print(f"    - **MAE (平均絶対誤差)**: {mae:.4f}")
    print(f"    - **RMSE (二乗平均平方根誤差)**: {rmse:.4f}")

    print("\n🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---")
    print(f"    - **NDCG@K (正規化割引累積利得)**:")
    for k_str, score in ndcg_results.items():
        print(f"      - {k_str:<8}: {score:.4f}")

    print(f"\n    - **RBP (ランクバイアス適合率)**: {rbp_val:.4f}")
    
    # --- ✅ XAI (1): アテンション可視化 ---
    print("\n" + "="*50)
    print("🎨 C. ATTENTION VISUALIZATION (Top 5)")
    print("="*50)
    # 予測ランク上位5名のインフルエンサーのインデックスを取得
    top_5_indices_in_df = df_results.sort_values(by='Predicted_Rank').head(5).index
    
    # attention_weights は [Batch_Size, Seq_Len(11)]
    # df_results の index がそのまま Batch_Size の index に対応する
    top_5_usernames = df_results.loc[top_5_indices_in_df, 'Username'].values
    top_5_attentions = attention_weights[top_5_indices_in_df].cpu().numpy()
    
    try:
        # attention_weights は 11ヶ月分 (T=1...11)
        num_months_attention = top_5_attentions.shape[1]
        # 論文のグラフ(Fig 1)に合わせて Jan から Dec でラベル付け (予測対象はDec)
        month_labels = pd.date_range(start=latest_date - pd.DateOffset(months=num_months_attention), periods=num_months_attention, freq='M').strftime('%b')
        
        df_attention = pd.DataFrame(top_5_attentions, index=top_5_usernames, columns=month_labels)
        
        plt.figure(figsize=(12, 4))
        sns.heatmap(df_attention, annot=True, fmt=".3f", cmap="viridis", linewidths=.5)
        plt.title("Attention Weights for Top 5 Predicted Influencers (When Predicting Dec)")
        plt.xlabel("Input Month (T=1 to T=11)")
        plt.ylabel("Influencer")
        plt.yticks(rotation=0)
        img_path = "attention_heatmap.png"
        plt.savefig(img_path)
        print(f"✅ Attention heatmap saved to '{img_path}'")
        plt.close()

    except ImportError:
        print("⚠️ Matplotlib/Seaborn not found. Skipping heatmap visualization.")
        print("Top 5 Attentions (Raw):")
        for i, user in enumerate(top_5_usernames):
            print(f"  - {user}: {np.round(top_5_attentions[i], 3)}")
    
    # --- ✅ XAI (2): 順列重要度の実行 ---
    # ベースラインのNDCG@100スコアを取得
    base_ndcg_100 = ndcg_results.get('NDCG@100')
    if base_ndcg_100 is None:
        # 100人未満の場合、計算可能な最大のKで代用
        max_k = max([k for k in k_values if k <= len(df_results)])
        base_ndcg_100 = ndcg_results.get(f'NDCG@{max_k}', 0.0)

    # 順列重要度を計算・表示
    _ = run_permutation_importance(
        model=model,
        base_graphs=inference_input_graphs,
        target_indices=predict_indices,
        feature_names=feature_columns,
        ground_truth_graph=ground_truth_graph,
        base_ndcg_100=base_ndcg_100,
        device=DEVICE
    )
    
    end_time = time.time()
    print(f"\nTotal inference time: {end_time - start_time:.2f} seconds")

# --- 乱数シード設定関数 ---
def set_seed(seed_value=42):
    np.random.seed(seed_value) 
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- mainブロック ---
if __name__ == '__main__':
    set_seed(42) 
    train_and_save_model() # 訓練と特徴量名の保存
    run_inference()          # 推論とXAI分析の実行