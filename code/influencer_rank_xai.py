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
from io import StringIO

# --- 1. 定数定義 ---
PREPROCESSED_FILE = 'posts_2017.csv'
HASHTAGS_FILE = 'hashtags_2017.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
MODEL_SAVE_PATH = f'influencer_rank_model_{time.strftime("%Y%m%d")}_rich_features_2017_final.pth'

# --- 2. データ準備関数 ---
def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts'):
    """
    指定された終了日までのNヶ月間のグラフデータセットを構築する。
    指定された期間に活動のあったインフルエンサーのみを対象とする。
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
        rnn_out, _ = self.rnn(sequence_of_embeddings)
        attention_scores = self.attention_layer(rnn_out).tanh()
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(rnn_out * attention_weights, dim=1)
        return context_vector, attention_weights

class InfluencerRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5):
        super(InfluencerRankModel, self).__init__()
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim, num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        self.predictor = nn.Sequential(Linear(rnn_dim, 16), ReLU(), Dropout(dropout_prob), Linear(16, 1))
    def forward(self, graph_sequence, target_indices):
        sequence_embeddings = torch.stack([self.gcn_encoder(g.x, g.edge_index) for g in graph_sequence])
        target_embeddings = sequence_embeddings[:, target_indices].permute(1, 0, 2)
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
    rbp_score = 0
    max_score = true_scores_in_predicted_order.max()
    if max_score == 0: return 0.0
    normalized_scores = true_scores_in_predicted_order / max_score
    for i, relevance in enumerate(normalized_scores):
        rbp_score += (p ** i) * relevance
    return (1 - p) * rbp_score

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

    print(f"--- Starting Training ---")
    start_time = time.time()
    
    latest_date = pd.to_datetime('2017-12-31')
    monthly_graphs, influencer_indices, _ = prepare_graph_data(end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return

    model = InfluencerRankModel(feature_dim=FEATURE_DIM, gcn_dim=GCN_DIM, rnn_dim=RNN_DIM, num_gcn_layers=NUM_GCN_LAYERS, dropout_prob=DROPOUT_PROB)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_listwise = BatchedListwiseRankingLoss()
    criterion_pointwise = nn.MSELoss() 
    alpha = 1.0
    
    true_scores = monthly_graphs[-1].y[influencer_indices]
    display_relevance_distribution(true_scores.squeeze().cpu().numpy(), "📊 Training Data Ground Truth Distribution")
    dataset = TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), true_scores)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    if not END_TO_END_TRAINING:
        print("\n--- Strategy: Two-Stage Learning (Fast) ---")
        model.gcn_encoder.eval()
        with torch.no_grad():
            sequence_embeddings = torch.stack([model.gcn_encoder(g.x, g.edge_index) for g in tqdm(monthly_graphs, desc="GCN Encoding")])
        model.attentive_rnn.train()
        model.predictor.train()
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
                optimizer.zero_grad()
                batch_sequence_embeddings = sequence_embeddings[:, batch_indices].permute(1, 0, 2)
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
        print("\n--- Strategy: End-to-End Learning (Slow, High-Memory) ---")
        model.train()
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Performing GCN forward pass for this epoch...")
            sequence_embeddings = torch.stack([model.gcn_encoder(g.x, g.edge_index) for g in monthly_graphs])
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Training Batches"):
                optimizer.zero_grad()
                batch_sequence_embeddings = sequence_embeddings[:, batch_indices].permute(1, 0, 2)
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

    start_time = time.time()
    params = {'GCN_DIM': 128, 'NUM_GCN_LAYERS': 2, 'RNN_DIM': 64, 'DROPOUT_PROB': 0.5}

    latest_date = pd.to_datetime('2017-12-31')
    predict_graphs, predict_indices, node_to_idx = prepare_graph_data(
        end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
    
    model = InfluencerRankModel(feature_dim=FEATURE_DIM, gcn_dim=params['GCN_DIM'], rnn_dim=params['RNN_DIM'], num_gcn_layers=params['NUM_GCN_LAYERS'], dropout_prob=params['DROPOUT_PROB'])
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
        predicted_scores, _ = model(inference_input_graphs, predict_indices)

    idx_to_node = {i: node for node, i in node_to_idx.items()}
    influencer_usernames = [idx_to_node[idx] for idx in predict_indices]
    true_scores = ground_truth_graph.y[predict_indices]
    
    df_results = pd.DataFrame({
        'Username': influencer_usernames,
        'Predicted_Score': predicted_scores.squeeze().cpu().numpy(),
        'True_Score': true_scores.squeeze().cpu().numpy()
    })
    
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
    推論を実行し、トップNインフルエンサーのアテンションの重みを可視化する。
    また、アテンションパターンの類似度を分析する。
    """
    print("\n\n" + "="*50)
    print("🧠 STARTING ATTENTION ANALYSIS & VISUALIZATION")
    print("="*50)

    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'
    params = {'GCN_DIM': 128, 'NUM_GCN_LAYERS': 2, 'RNN_DIM': 64, 'DROPOUT_PROB': 0.5}

    latest_date = pd.to_datetime('2017-12-31')
    predict_graphs, predict_indices, node_to_idx = prepare_graph_data(
        end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)

    model = InfluencerRankModel(feature_dim=FEATURE_DIM, gcn_dim=params['GCN_DIM'], rnn_dim=params['RNN_DIM'], num_gcn_layers=params['NUM_GCN_LAYERS'], dropout_prob=params['DROPOUT_PROB'])
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Successfully loaded model from '{MODEL_SAVE_PATH}' for attention analysis.")
    except FileNotFoundError:
        print("Model file not found. Cannot perform attention analysis.")
        return

    inference_input_graphs = predict_graphs[:-1]
    model.eval()
    with torch.no_grad():
        predicted_scores, attention_weights = model(inference_input_graphs, predict_indices)

    idx_to_node = {i: node for node, i in node_to_idx.items()}
    influencer_usernames = [idx_to_node[idx] for idx in predict_indices]
    df_results = pd.DataFrame({
        'Username': influencer_usernames,
        'Predicted_Score': predicted_scores.squeeze().cpu().numpy()
    })
    df_sorted = df_results.sort_values(by='Predicted_Score', ascending=False).reset_index(drop=True)

    attention_weights_np = attention_weights.squeeze().cpu().numpy()
    df_attention = pd.DataFrame(attention_weights_np, index=influencer_usernames)

    top_influencers = df_sorted.head(top_n)['Username'].tolist()
    df_attention_top_n = df_attention.loc[top_influencers]

    # --- ヒートマップのプロットと保存 ---
    plt.figure(figsize=(12, 8))
    month_labels = [d.strftime('%Y-%m') for d in pd.date_range(start='2017-01-01', periods=11, freq='MS')]
    sns.heatmap(df_attention_top_n, xticklabels=month_labels, yticklabels=df_attention_top_n.index, cmap="viridis", annot=True, fmt=".2f")
    plt.title(f'Attention Weights for Top {top_n} Predicted Influencers', fontsize=16)
    plt.xlabel('Time (Month)', fontsize=12)
    plt.ylabel('Influencer Username', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # ✅ 変更点: plt.savefig() を使ってファイルに保存します
    output_filename = f'attention_heatmap_top{top_n}.png'
    plt.savefig(output_filename)
    print(f"\n✅ Attention heatmap saved to '{output_filename}'")
    
    # plt.show() はスクリプト実行時に不要なためコメントアウトまたは削除
    # plt.show()
    plt.close() # メモリを解放するためにプロットを閉じます

    # --- 類似度分析 ---
    print("\n🔍 --- Analyzing Similarity of Attention Patterns ---")
    
    similarity_matrix = cosine_similarity(df_attention)
    df_similarity = pd.DataFrame(similarity_matrix, index=df_attention.index, columns=df_attention.index)
    
    np.fill_diagonal(df_similarity.values, 0)
    
    similar_pairs = df_similarity.unstack().sort_values(ascending=False)
    
    print("Top 10 Most Similar Influencer Pairs (based on attention pattern):")
    displayed_pairs = set()
    count = 0
    for (user1, user2), score in similar_pairs.items():
        if user1 != user2 and tuple(sorted((user1, user2))) not in displayed_pairs:
            print(f"  - {user1:<20} & {user2:<20} | Similarity: {score:.4f}")
            displayed_pairs.add(tuple(sorted((user1, user2))))
            count += 1
        if count >= 10:
            break

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
    
    # モデルの学習と保存
    train_and_save_model()
    
    # 通常の推論と評価
    run_inference()
    
    # アテンションの分析と可視化
    analyze_and_visualize_attention(top_n=20)