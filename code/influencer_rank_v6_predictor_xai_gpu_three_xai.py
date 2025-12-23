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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from io import StringIO

# --- 🚀 XAIライブラリのインポート ---
try:
    import shap
    from captum.attr import IntegratedGradients, LRP
    print("Successfully imported SHAP and Captum.")
except ImportError:
    print("Error: 'shap' or 'captum' library not found.")
    print("Please install them using: pip install shap captum")
    # 続けるが、XAI分析はスキップされる
    shap = None
    IntegratedGradients = None
    LRP = None

# --- 1. 定数定義 ---
PREPROCESSED_FILE = 'posts_2017.csv'
HASHTAGS_FILE = 'hashtags_2017.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
MODEL_SAVE_PATH = f'influencer_rank_model_{time.strftime("%Y%m%d")}_rich_features_2017_final.pth'

# --- 使用するGPUデバイス番号 ---
DEVICE_NUMBER = 0 

# --- 2. データ準備関数 ---
def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts'):
    """
    指定された終了日までのNヶ月間のグラフデータセットを構築する。
    指定された期間に活動のあったインフルエンサーのみを対象とする。
    (この段階ではテンソルはCPU上に作成されます)
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
    df_influencers_master = pd.read_csv(StringIO("".join(lines)), sep='\t', dtype=str)
    df_influencers_master.rename(columns={'#Followers': 'followers', '#Followees': 'followees', '#Posts': 'posts', 'Username': 'username', 'Category': 'category'}, inplace=True)

    df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce').dropna()
    df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce').dropna()
    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time

    active_influencers_set = set(df_posts['username'].unique())
    print(f"Found {len(active_influencers_set):,} active influencers in {PREPROCESSED_FILE}.")
    df_influencers = df_influencers_master[df_influencers_master['username'].isin(active_influencers_set)].copy()
    
    # --- ノードの準備 ---
    influencer_set = set(df_influencers['username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 静的特徴量 ---
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

    # --- 動的特徴量 ---
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
    print(f"Total feature dimension: {FEATURE_DIM}")

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
        
    return monthly_graphs, influencer_indices, node_to_idx

# --- 3. モデル定義 ---
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

class AttentiveRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentiveRNN, self).__init__()
        self.rnn = GRU(input_dim, hidden_dim, batch_first=True)
        self.attention_layer = Linear(hidden_dim, 1)
    def forward(self, sequence_of_embeddings):
        # sequence_of_embeddings: (Batch, Time, Features)
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_scores = self.attention_layer(rnn_out).tanh()
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        return context_vector, attention_weights

class InfluencerRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5):
        super(InfluencerRankModel, self).__init__()
        # GCNエンコーダはモデルの一部として保持
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim, num_gcn_layers)
        
        # RNN + Predictor
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        self.predictor = nn.Sequential(Linear(rnn_dim, 16), ReLU(), Dropout(dropout_prob), Linear(16, 1))
    
    # 🚀 OOMエラー回避のためのリファクタリング
    # GCNエンコードは `forward` の *外* で実行する。
    # この `forward` は、エンコード済みの時系列埋め込み (target_embeddings) を受け取る。
    def forward(self, target_embeddings):
        """
        GCNエンコード済みの時系列埋め込みを受け取り、RNNとPredictorを実行する。
        :param target_embeddings: (Batch_Size, Time_Steps, GCN_Features)
        :return: predicted_scores, attention_weights
        """
        final_representation, attention_weights = self.attentive_rnn(target_embeddings)
        predicted_scores = self.predictor(final_representation)
        return predicted_scores, attention_weights

# --- 4. 損失関数と評価関数 ---
class BatchedListwiseRankingLoss(nn.Module):
    def __init__(self):
        super(BatchedListwiseRankingLoss, self).__init__()
    def forward(self, pred_scores, true_scores):
        pred_probs = F.softmax(pred_scores, dim=1)
        true_probs = F.softmax(true_scores, dim=1)
        return -torch.sum(true_probs * torch.log(pred_probs + 1e-9), dim=1).mean()

def display_relevance_distribution(scores, title):
    # この関数は Numpy 配列を期待
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
    if engagement_rate >= 0.10: return 5
    if engagement_rate >= 0.07: return 4
    if engagement_rate >= 0.05: return 3
    if engagement_rate >= 0.03: return 2
    if engagement_rate >= 0.01: return 1
    return 0

def calculate_rbp(true_scores_in_predicted_order, p=0.95):
    # この関数は Numpy 配列を期待
    rbp_score = 0
    max_score = true_scores_in_predicted_order.max()
    if max_score == 0: return 0.0
    normalized_scores = true_scores_in_predicted_order / max_score
    for i, relevance in enumerate(normalized_scores):
        rbp_score += (p ** i) * relevance
    return (1 - p) * rbp_score

# --- 🚀 新設: XAIヒートマップ描画関数 ---
def plot_xai_heatmap(contributions, usernames, method_name, rnn_dim, top_n):
    """
    XAI手法による貢献度 (Numpy配列) を受け取り、ヒートマップをプロットして保存する
    """
    print(f"Generating {method_name} contribution heatmap...")
    # contributions は (top_n, rnn_dim) の Numpy 配列であること
    df_contribution = pd.DataFrame(contributions, 
                                   index=usernames, 
                                   columns=[f'Dim_{i}' for i in range(rnn_dim)])
    
    # ヒートマップ用にスケーリング
    scaler = StandardScaler()
    scaled_contribution = scaler.fit_transform(df_contribution)
    df_contribution_scaled = pd.DataFrame(scaled_contribution, index=df_contribution.index, columns=df_contribution.columns)
    
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df_contribution_scaled, 
        yticklabels=df_contribution_scaled.index, 
        cmap="vlag", # 0を中央とする発散色マップ
        center=0,     # 0を中心に色付け
        annot=False,  # 64次元もあるので注釈は不可
        cbar_kws={'label': f'Scaled Contribution ({method_name})'}
    )
    plt.title(f'RNN Feature Contribution ({method_name}) for Top {top_n} Influencers ({rnn_dim} Dims)', fontsize=16)
    plt.xlabel(f'RNN Hidden Dimension ({rnn_dim})', fontsize=12)
    plt.ylabel('Influencer Username', fontsize=12)
    plt.tight_layout()
    
    # ファイル名に月日付と時間を付与して保存
    current_time_str = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f'{method_name}_contribution_heatmap_{current_time_str}.png'
    plt.savefig(output_filename)
    print(f"✅ {method_name} contribution heatmap saved to '{output_filename}'")
    plt.close() # メモリを解放

# --- 5. 学習実行関数 ---
def train_and_save_model():
    END_TO_END_TRAINING = False
    GCN_DIM = 128
    NUM_GCN_LAYERS = 2
    RNN_DIM = 64
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.5
    NUM_EPOCHS = 200
    LISTS_PER_BATCH = 1024
    LIST_SIZE = 10
    BATCH_SIZE = LISTS_PER_BATCH * LIST_SIZE
    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'

    # --- 🚀 デバイス設定 (DEVICE_NUMBER を使用) ---
    device_string = f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f"Error: Device cuda:{DEVICE_NUMBER} was requested, but CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    elif device.type == 'cuda':
        try:
            torch.cuda.get_device_name(device)
        except Exception:
            print(f"Error: Device cuda:{DEVICE_NUMBER} not found or invalid. Defaulting to cuda:0.")
            device = torch.device("cuda:0") # デフォルト(0番)にフォールバック

    print(f"--- Starting Training ---")
    print(f"Using device: {device}")
    start_time = time.time()
    
    latest_date = pd.to_datetime('2017-12-31')
    monthly_graphs, influencer_indices, _ = prepare_graph_data(end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return

    # --- 🚀 モデルをGPUに転送 ---
    model = InfluencerRankModel(feature_dim=FEATURE_DIM, gcn_dim=GCN_DIM, rnn_dim=RNN_DIM, num_gcn_layers=NUM_GCN_LAYERS, dropout_prob=DROPOUT_PROB).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_listwise = BatchedListwiseRankingLoss().to(device)
    criterion_pointwise = nn.MSELoss().to(device)
    alpha = 1.0
    
    true_scores = monthly_graphs[-1].y[influencer_indices]
    display_relevance_distribution(true_scores.squeeze().cpu().numpy(), "📊 Training Data Ground Truth Distribution")
    
    dataset = TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), true_scores)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    if not END_TO_END_TRAINING:
        print("\n--- Strategy: Two-Stage Learning (Fast) ---")
        model.gcn_encoder.eval()
        
        # --- 🚀 修正 (Patch 1): GCNエンコード -> CPUに退避 ---
        with torch.no_grad():
            print("Pre-computing GCN embeddings (GPU -> CPU cache)...")
            # グラフデータをGPUでエンコードし、結果は .cpu() でCPUメモリに即時退避
            sequence_embeddings = torch.stack([
                model.gcn_encoder(g.x.to(device), g.edge_index.to(device)).cpu() 
                for g in tqdm(monthly_graphs, desc="GCN Encoding (GPU) -> CPU")
            ])
            # sequence_embeddings は CPU 上に (Time, All_Nodes, Features) のShapeで保持
            
        model.attentive_rnn.train()
        model.predictor.train()
        
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                optimizer.zero_grad()
                
                # --- 🚀 修正 (Patch 2): バッチをGPUに転送 ---
                # sequence_embeddings (CPU) からスライスし、.to(device) でGPUに転送
                batch_sequence_embeddings = sequence_embeddings[:, batch_indices].permute(1, 0, 2).to(device)
                batch_true_scores = batch_true_scores.to(device)
                
                # --- 🚀 修正 (Refactor): model.forward を呼ぶ ---
                predicted_scores, _ = model(batch_sequence_embeddings)
                
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
        # (End-to-End 学習は OOM 対策の修正が複雑なため、Two-Stage を推奨)
        print("\n--- Strategy: End-to-End Learning (Skipped) ---")
        print("End-to-End is complex to manage memory for. Please use Two-Stage (END_TO_END_TRAINING = False).")
        pass 

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"✅ Model saved to '{MODEL_SAVE_PATH}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")

# --- 6. 推論実行関数 ---
def run_inference():
    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'
    
    print("\n\n" + "="*50)
    print("📈 STARTING INFERENCE PROCESS")
    print("="*50)

    # --- 🚀 デバイス設定 (DEVICE_NUMBER を使用) ---
    device_string = f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f"Error: Device cuda:{DEVICE_NUMBER} was requested, but CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    elif device.type == 'cuda':
        try:
            torch.cuda.get_device_name(device)
        except Exception:
            print(f"Error: Device cuda:{DEVICE_NUMBER} not found or invalid. Defaulting to cuda:0.")
            device = torch.device("cuda:0")

    print(f"Using device: {device}")
    start_time = time.time()
    
    params = {'GCN_DIM': 128, 'NUM_GCN_LAYERS': 2, 'RNN_DIM': 64, 'DROPOUT_PROB': 0.5}

    latest_date = pd.to_datetime('2017-12-31')
    predict_graphs, predict_indices, node_to_idx = prepare_graph_data(
        end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
    
    model = InfluencerRankModel(feature_dim=FEATURE_DIM, gcn_dim=params['GCN_DIM'], rnn_dim=params['RNN_DIM'], num_gcn_layers=params['NUM_GCN_LAYERS'], dropout_prob=params['DROPOUT_PROB']).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Successfully loaded model from '{MODEL_SAVE_PATH}'")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_SAVE_PATH}'. Please run training first.")
        return

    inference_input_graphs = predict_graphs[:-1]
    ground_truth_graph = predict_graphs[-1]

    model.eval()
    with torch.no_grad():
        # --- 🚀 修正 (Refactor): GCNエンコードを外で実行 (CPUキャッシュ) ---
        print("Running GCN Encoding for inference (results to CPU)...")
        sequence_embeddings_cpu = torch.stack([
            model.gcn_encoder(g.x.to(device), g.edge_index.to(device)).cpu()
            for g in tqdm(inference_input_graphs, desc="GCN Encoding (Inference)")
        ])
        
        # --- 🚀 修正 (Refactor): 必要な埋め込みをスライス ---
        target_embeddings_cpu = sequence_embeddings_cpu[:, predict_indices].permute(1, 0, 2)
        
        # --- 🚀 修正 (Refactor): RNN+Predictor を一括でGPU実行 ---
        # (推論時のVRAMが不足する場合は、ここもバッチ処理が必要になります)
        print(f"Running RNN + Predictor for {len(predict_indices)} influencers...")
        try:
            target_embeddings_gpu = target_embeddings_cpu.to(device)
            predicted_scores, _ = model(target_embeddings_gpu)
        except torch.cuda.OutOfMemoryError:
            print("OOM Error during full inference pass. Trying batched inference...")
            predicted_scores_list = []
            inf_batch_size = 1024 # VRAMに応じて調整
            for i in tqdm(range(0, len(target_embeddings_cpu), inf_batch_size), desc="Batched Inference"):
                batch_embeddings_gpu = target_embeddings_cpu[i:i+inf_batch_size].to(device)
                batch_scores, _ = model(batch_embeddings_gpu)
                predicted_scores_list.append(batch_scores.cpu())
            predicted_scores = torch.cat(predicted_scores_list, dim=0).to(device)

    idx_to_node = {i: node for node, i in node_to_idx.items()}
    influencer_usernames = [idx_to_node[idx] for idx in predict_indices]
    
    true_scores = ground_truth_graph.y[predict_indices] # CPU上
    
    # --- 🚀 結果をCPUに戻してDataFrame作成 ---
    df_results = pd.DataFrame({
        'Username': influencer_usernames,
        'Predicted_Score': predicted_scores.squeeze().cpu().numpy(),
        'True_Score': true_scores.squeeze().cpu().numpy()
    })
    
    # (以降の Numpy / Pandas 処理は変更なし)
    mae = (df_results['Predicted_Score'] - df_results['True_Score']).abs().mean()
    mse = ((df_results['Predicted_Score'] - df_results['True_Score']) ** 2).mean()
    rmse = np.sqrt(mse)
    
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
    print(df_results.sort_values(by='Predicted_Rank')[['Username', 'Predicted_Score', 'True_Score']].head(20).to_string(index=False))
    
    print("\n\n" + "="*50)
    print("📊 MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    display_relevance_distribution(df_results['True_Score'], "📈 Inference Data Ground Truth Distribution")
    display_relevance_distribution(df_results['Predicted_Score'], "🤖 Inference Data Predicted Distribution")

    print("\n🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---")
    print(f"   - **MAE (平均絶対誤差)**: {mae:.4f}")
    print(f"   - **RMSE (二乗平均平方根誤差)**: {rmse:.4f}")

    print("\n🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---")
    print(f"   - **NDCG@K (正規化割引累積利得)**:")
    for k_str, score in ndcg_results.items():
        print(f"     - {k_str:<8}: {score:.4f}")

    print(f"\n   - **RBP (ランクバイアス適合率)**: {rbp_val:.4f}")
    
    end_time = time.time()
    print(f"\nTotal inference time: {end_time - start_time:.2f} seconds")

# --- 7. アテンション可視化・分析関数 ---
def analyze_and_visualize_attention(top_n=20):
    """
    推論を実行し、アテンション、Grad*Input, SHAP, IG, LRP の貢献度を可視化・保存する。
    """
    print("\n\n" + "="*50)
    print("🧠 STARTING XAI ANALYSIS & VISUALIZATION")
    print("="*50)
    
    # --- 🚀 XAIライブラリのチェック ---
    if shap is None or IntegratedGradients is None or LRP is None:
        print("SHAP or Captum libraries not found. Skipping advanced XAI analysis.")
        return

    # --- 🚀 デバイス設定 (DEVICE_NUMBER を使用) ---
    device_string = f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_string)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        print(f"Error: Device cuda:{DEVICE_NUMBER} was requested, but CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    elif device.type == 'cuda':
        try:
            torch.cuda.get_device_name(device)
        except Exception:
            print(f"Error: Device cuda:{DEVICE_NUMBER} not found or invalid. Defaulting to cuda:0.")
            device = torch.device("cuda:0")

    print(f"Using device: {device}")

    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'
    params = {'GCN_DIM': 128, 'NUM_GCN_LAYERS': 2, 'RNN_DIM': 64, 'DROPOUT_PROB': 0.5}

    latest_date = pd.to_datetime('2017-12-31')
    predict_graphs, predict_indices, node_to_idx = prepare_graph_data(
        end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)

    model = InfluencerRankModel(feature_dim=FEATURE_DIM, gcn_dim=params['GCN_DIM'], rnn_dim=params['RNN_DIM'], num_gcn_layers=params['NUM_GCN_LAYERS'], dropout_prob=params['DROPOUT_PROB']).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Successfully loaded model from '{MODEL_SAVE_PATH}' for attention analysis.")
    except FileNotFoundError:
        print("Model file not found. Cannot perform attention analysis.")
        return

    inference_input_graphs = predict_graphs[:-1]
    
    # --- 1. GCNエンコードの事前計算 (CPUキャッシュ) ---
    # 貢献度分析 (backward) のため、 no_grad の *外* でGCNを実行すると
    # メモリを大量消費するため、GCNは no_grad で実行し、
    # 貢献度分析は「RNNの入力 (final_rep)」から開始します。
    print("Running GCN Encoding for analysis (results to CPU)...")
    model.gcn_encoder.eval()
    with torch.no_grad():
        sequence_embeddings_cpu = torch.stack([
            model.gcn_encoder(g.x.to(device), g.edge_index.to(device)).cpu()
            for g in tqdm(inference_input_graphs, desc="GCN Encoding (Analysis)")
        ])
    
    # 必要な埋め込みをスライス (CPU)
    target_embeddings_cpu = sequence_embeddings_cpu[:, predict_indices].permute(1, 0, 2)
    
    # --- 2. (no_grad) でのアテンションと予測の取得 ---
    model.eval() # model 全体を eval モードに (Dropout を無効化)
    
    # RNN+Predictor の実行 (バッチ処理)
    print(f"Running RNN + Predictor (no_grad) for {len(predict_indices)} influencers...")
    predicted_scores_list = []
    attention_weights_list = []
    
    # 貢献度分析の対象 (final_rep) もここで取得
    final_rep_list = [] 
    
    inf_batch_size = 1024 # VRAMに応じて調整
    with torch.no_grad():
        for i in tqdm(range(0, len(target_embeddings_cpu), inf_batch_size), desc="Batched Analysis Pass"):
            batch_embeddings_gpu = target_embeddings_cpu[i:i+inf_batch_size].to(device)
            
            # RNN部分だけ実行して final_rep を取得
            final_rep_gpu, batch_att_weights_gpu = model.attentive_rnn(batch_embeddings_gpu)
            
            # Predictor部分を実行
            batch_scores_gpu = model.predictor(final_rep_gpu)
            
            predicted_scores_list.append(batch_scores_gpu.cpu())
            attention_weights_list.append(batch_att_weights_gpu.cpu())
            final_rep_list.append(final_rep_gpu.cpu()) # 貢献度分析の対象

    predicted_scores = torch.cat(predicted_scores_list, dim=0)
    attention_weights = torch.cat(attention_weights_list, dim=0)
    final_rep_all_cpu = torch.cat(final_rep_list, dim=0) # (Num_Influencers, 64)

    idx_to_node = {i: node for node, i in node_to_idx.items()}
    influencer_usernames = [idx_to_node[idx] for idx in predict_indices]

    df_results_with_idx = pd.DataFrame({
        'Username': influencer_usernames,
        'Predicted_Score': predicted_scores.squeeze().cpu().numpy()
    }).reset_index()

    df_sorted = df_results_with_idx.sort_values(by='Predicted_Score', ascending=False).reset_index(drop=True)

    # --- 3. アテンションの可視化 ---
    print("\n--- 1. Attention Analysis ---")
    attention_weights_np = attention_weights.squeeze().cpu().numpy()
    df_attention = pd.DataFrame(attention_weights_np, index=influencer_usernames)

    top_n_indices_in_batch = df_sorted.head(top_n)['index'].values
    top_influencers_usernames = df_sorted.head(top_n)['Username'].tolist()
    
    df_attention_top_n = df_attention.iloc[top_n_indices_in_batch]

    plt.figure(figsize=(12, 8))
    month_labels = [d.strftime('%Y-%m') for d in pd.date_range(start='2017-01-01', periods=11, freq='MS')]
    sns.heatmap(df_attention_top_n, xticklabels=month_labels, yticklabels=top_influencers_usernames, cmap="viridis", annot=True, fmt=".2f")
    plt.title(f'Attention Weights for Top {top_n} Predicted Influencers', fontsize=16)
    plt.xlabel('Time (Month)', fontsize=12)
    plt.ylabel('Influencer Username', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_filename = f'attention_heatmap_top{top_n}.png'
    plt.savefig(output_filename)
    print(f"\n✅ Attention heatmap saved to '{output_filename}'")
    plt.close()

    # --- 4. 貢献度分析 (Grad * Input) ---
    print("\n--- 2. Grad * Input Analysis ---")
    
    # 貢献度分析の対象 (TopN) をCPUからGPUに戻す
    top_n_final_rep_gpu = final_rep_all_cpu[top_n_indices_in_batch].to(device).requires_grad_(True)

    # フックは不要。model.predictor に直接入力する
    # model.predictor は model.eval() のまま (Dropout無効)
    
    # フォワードパスを実行
    predicted_scores_for_grad = model.predictor(top_n_final_rep_gpu)
    
    # TopNのスコア合計に対して backward
    top_n_scores_sum = predicted_scores_for_grad.sum()
    
    model.zero_grad()
    top_n_scores_sum.backward()
    
    # 勾配を取得
    final_rep_grad_gpu = top_n_final_rep_gpu.grad
    
    # CPUに戻して計算
    top_n_final_rep_cpu = top_n_final_rep_gpu.cpu().detach().numpy()
    top_n_final_rep_grad_cpu = final_rep_grad_gpu.cpu().detach().numpy()
    
    contribution_scores = top_n_final_rep_cpu * top_n_final_rep_grad_cpu
    
    # ヒートマップ描画関数を呼び出し
    plot_xai_heatmap(contribution_scores, 
                     top_influencers_usernames, 
                     "Grad_x_Input", 
                     params['RNN_DIM'], 
                     top_n)

    print("--- RNN Contribution Analysis (Grad*Input) Complete ---")
    
    # (Top 1 のコンソール出力)
    top_1_username = top_influencers_usernames[0]
    top_1_score = df_sorted[df_sorted['Username'] == top_1_username].iloc[0]['Predicted_Score']
    top_1_contributions = pd.Series(contribution_scores[0], index=[f'Dim_{i}' for i in range(params['RNN_DIM'])])
    
    print(f"\nShowing contribution details for Top 1 Influencer: {top_1_username} (Score: {top_1_score:.4f})")
    print("\n--- Top 5 Positive Contributors (Grad*Input) ---")
    print(top_1_contributions.sort_values(ascending=False).head(5).to_string())
    print("\n--- Top 5 Negative Contributors (Grad*Input) ---")
    print(top_1_contributions.sort_values(ascending=True).head(5).to_string())


    # --- 🚀 5. 高度なXAI分析 (SHAP, IG, LRP) ---
    print("\n\n" + "="*50)
    print("🧠 STARTING ADVANCED XAI ANALYSIS (SHAP, IG, LRP)")
    print("="*50)

    # 入力テンソル (top_n, 64) とベースライン (1, 64) を準備
    # (top_n_final_rep_gpu は上で requires_grad=True に設定済み)
    baselines_gpu = torch.zeros(1, params['RNN_DIM']).to(device)

    # --- 5A. Integrated Gradients (IG) ---
    try:
        print("Running Integrated Gradients (IG)...")
        explainer_ig = IntegratedGradients(model.predictor)
        attributions_ig = explainer_ig.attribute(top_n_final_rep_gpu, 
                                                 baselines=baselines_gpu, 
                                                 n_steps=50) # n_steps は精度と速度のトレードオフ
        
        plot_xai_heatmap(attributions_ig.cpu().detach().numpy(), 
                         top_influencers_usernames, 
                         "IntegratedGradients", 
                         params['RNN_DIM'], 
                         top_n)
    except Exception as e:
        print(f"Error during Integrated Gradients: {e}")

    # --- 5B. LRP (Layer-wise Relevance Propagation) ---
    try:
        print("Running LRP...")
        explainer_lrp = LRP(model.predictor)
        # LRPは (batch, features) の入力を期待する
        # model.eval() モードなので Dropout は無視される
        attributions_lrp = explainer_lrp.attribute(top_n_final_rep_gpu)
        
        plot_xai_heatmap(attributions_lrp.cpu().detach().numpy(), 
                         top_influencers_usernames, 
                         "LRP", 
                         params['RNN_DIM'], 
                         top_n)
    except Exception as e:
        print(f"Error during LRP: {e}")
        print("Note: LRP in Captum works best on simple Linear/ReLU models.")

    # --- 5C. SHAP (GradientExplainer) ---
    try:
        print("Running SHAP (GradientExplainer)...")
        # SHAPは (background_samples, features) と (input_samples, features) を期待
        explainer_shap = shap.GradientExplainer(model.predictor, baselines_gpu)
        shap_values = explainer_shap.shap_values(top_n_final_rep_gpu) # (N, 64) の numpy array
        
        plot_xai_heatmap(shap_values, 
                         top_influencers_usernames, 
                         "SHAP_Gradient", 
                         params['RNN_DIM'], 
                         top_n)
    except Exception as e:
        print(f"Error during SHAP: {e}")
    
    print("--- Advanced XAI Analysis Complete ---")

# --- 8. ユーティリティと実行ブロック ---
def set_seed(seed_value=42):
    np.random.seed(seed_value) 
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # 乱数シードを固定
    set_seed(42) 
    
    # モデルの学習と保存 (GPU使用, OOM対策済み)
    train_and_save_model()
    
    # 通常の推論と評価 (GPU使用, OOM対策済み)
    run_inference()
    
    # アテンションの分析と可視化 (GPU使用, OOM対策済み, XAI追加)
    analyze_and_visualize_attention(top_n=20)