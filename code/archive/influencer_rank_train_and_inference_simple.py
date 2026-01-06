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

# --- 1. 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
HASHTAGS_FILE = 'output_hashtags_all_parallel.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
MODEL_SAVE_PATH = f'influencer_rank_model_{time.strftime("%Y%m%d")}_metric.pth'

def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts'):
    """
    指定された終了日までのNヶ月間のグラフデータセットを構築する。
    エンゲージメントメトリックを選択可能にする。
    
    Args:
        end_date (pd.Timestamp): グラフ構築の終了日
        num_months (int): 構築する月数
        metric_numerator (str): 'likes' または 'likes_and_comments'
        metric_denominator (str): 'posts' または 'followers'
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")
    
    # --- データ読み込み ---
    df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
    if 'comments' not in df_posts.columns:
        print("Warning: 'comments' column not found in df_posts. Defaulting to 0.")
        df_posts['comments'] = 0
        
    df_hashtags = pd.read_csv(HASHTAGS_FILE, header=0, low_memory=False)
    df_mentions = pd.read_csv(MENTIONS_FILE, header=0, low_memory=False)
    
    df_hashtags.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
    df_mentions.rename(columns={'source': 'username', 'target': 'mention'}, inplace=True)
    with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f: lines = f.readlines()
    lines = [line for line in lines if '===' not in line]
    from io import StringIO
    df_influencers = pd.read_csv(StringIO("".join(lines)), sep='\t', dtype=str)
    df_influencers.rename(columns={'#Followers': 'followers', '#Followees': 'followees', '#Posts': 'posts'}, inplace=True)
    
    df_hashtags['datetime'] = pd.to_datetime(df_hashtags['timestamp'], unit='s', errors='coerce').dropna()
    df_mentions['datetime'] = pd.to_datetime(df_mentions['timestamp'], unit='s', errors='coerce').dropna()
    df_posts['month'] = df_posts['datetime'].dt.to_period('M').dt.start_time

    # --- ノードの準備 ---
    influencer_set = set(df_influencers['Username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- 特徴量エンジニアリング ---
    static_features = pd.merge(pd.DataFrame({'Username': all_nodes}),
                               df_influencers[['Username', 'followers', 'followees', 'posts']],
                               on='Username', how='left').fillna(0)
    for col in ['followers', 'followees', 'posts']:
        static_features[col] = pd.to_numeric(static_features[col], errors='coerce').fillna(0)

    dynamic_features = df_posts.groupby(['username', 'month']).agg(
        monthly_post_count=('datetime', 'size'),
        avg_caption_length=('caption', lambda x: x.astype(str).str.len().mean()),
        avg_tag_count=('tag_count', 'mean'),
        avg_sentiment=('sentiment', 'mean')).reset_index()

    # --- グラフ時系列データの構築 ---
    monthly_graphs = []
    start_date = end_date - pd.DateOffset(months=num_months-1)
    for snapshot_date in tqdm(pd.date_range(start_date, end_date, freq='ME'), desc="Building monthly graphs"):
        snapshot_month = snapshot_date.to_period('M').start_time
        current_hashtags = df_hashtags[df_hashtags['datetime'] <= snapshot_date]
        current_mentions = df_mentions[df_mentions['datetime'] <= snapshot_date]
        edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
        edges_mt = [(node_to_idx[str(u)], node_to_idx[str(m)]) for u, m in zip(current_mentions['username'], current_mentions['mention']) if str(u) in node_to_idx and str(m) in node_to_idx]
        if not edges_ht and not edges_mt: continue
        edge_index = torch.tensor(list(set(edges_ht + edges_mt)), dtype=torch.long).t().contiguous()
        
        # --- x (特徴量) の作成 ---
        current_dynamic = dynamic_features[dynamic_features['month'] == snapshot_month]
        snapshot_features = pd.merge(static_features, current_dynamic, left_on='Username', right_on='username', how='left').fillna(0)
        feature_columns = ['followers', 'followees', 'posts', 'monthly_post_count', 'avg_caption_length', 'avg_tag_count', 'avg_sentiment']
        x = torch.tensor(snapshot_features[feature_columns].values, dtype=torch.float)
        
        # --- y (正解ラベル) の作成 ---
        monthly_posts_period = df_posts[df_posts['datetime'].dt.to_period('M') == snapshot_date.to_period('M')]
        
        monthly_agg = monthly_posts_period.groupby('username').agg(
            total_likes=('likes', 'sum'),
            total_comments=('comments', 'sum'),
            post_count=('datetime', 'size')
        ).reset_index()
        monthly_agg.rename(columns={'username': 'Username'}, inplace=True)
        
        if metric_numerator == 'likes_and_comments':
            monthly_agg['numerator'] = monthly_agg['total_likes'] + monthly_agg['total_comments']
        else:
            monthly_agg['numerator'] = monthly_agg['total_likes']
            
        # ✅✅✅ ここからが修正箇所 ✅✅✅
        if metric_denominator == 'followers':
            # 論文の定義 E = (avg_likes) / followers に従う
            # 1. まず、月間の「平均」エンゲージメントを計算 (合計エンゲージメント / 投稿数)
            monthly_agg['avg_engagement_per_post'] = 0.0
            post_count_mask = monthly_agg['post_count'] > 0
            monthly_agg.loc[post_count_mask, 'avg_engagement_per_post'] = monthly_agg.loc[post_count_mask, 'numerator'] / monthly_agg.loc[post_count_mask, 'post_count']
            
            # 2. 次に、その平均値をフォロワー数で割る
            merged_data = pd.merge(monthly_agg, static_features[['Username', 'followers']], on='Username', how='left')
            merged_data['engagement'] = 0.0
            followers_mask = merged_data['followers'] > 0
            # 0除算を避ける
            merged_data.loc[followers_mask, 'engagement'] = merged_data.loc[followers_mask, 'avg_engagement_per_post'] / merged_data.loc[followers_mask, 'followers']
            
        else: # デフォルトは 'posts'
            # こちらは「1投稿あたりの平均」なので、元の計算で正しい
            merged_data = monthly_agg
            merged_data['engagement'] = 0.0
            mask = merged_data['post_count'] > 0
            merged_data.loc[mask, 'engagement'] = merged_data.loc[mask, 'numerator'] / merged_data.loc[mask, 'post_count']
        
        engagement_data = pd.merge(pd.DataFrame({'Username': all_nodes}), merged_data[['Username', 'engagement']], on='Username', how='left').fillna(0)
        y = torch.tensor(engagement_data['engagement'].values, dtype=torch.float).view(-1, 1)
        
        graph_data = Data(x=x, edge_index=edge_index, y=y)
        monthly_graphs.append(graph_data)
        
    return monthly_graphs, influencer_indices, node_to_idx

# --- 3. モデル定義 (変更なし) ---
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
        return torch.sum(rnn_out * attention_weights, dim=1)

class InfluencerRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5):
        super(InfluencerRankModel, self).__init__()
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim, num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        #TODO ReLU活性化関数を追加したから修正の可能性あるかも
        self.predictor = nn.Sequential(Linear(rnn_dim, 16), ReLU(), Dropout(dropout_prob), Linear(16, 1), ReLU())

class BatchedListwiseRankingLoss(nn.Module):
    def __init__(self):
        super(BatchedListwiseRankingLoss, self).__init__()
    def forward(self, pred_scores, true_scores):
        pred_probs = F.softmax(pred_scores, dim=1)
        true_probs = F.softmax(true_scores, dim=1)
        return -torch.sum(true_probs * torch.log(pred_probs + 1e-9), dim=1).mean()

# --- 4. 学習・推論関数 ---

# ✅ 元のmain関数を学習専用の関数として名前変更 (中身は変更なし)
def train_and_save_model():
    """モデルを学習させ、重みをファイルに保存する元のコード"""
    END_TO_END_TRAINING = False
    GCN_DIM = 128
    NUM_GCN_LAYERS = 2
    RNN_DIM = 64
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.5
    NUM_EPOCHS = 20
    LISTS_PER_BATCH = 1024
    LIST_SIZE = 10
    BATCH_SIZE = LISTS_PER_BATCH * LIST_SIZE
    # 分子: 'likes' または 'likes_and_comments'
    METRIC_NUMERATOR = 'likes_and_comments'
    # 分母: 'posts' または 'followers'
    METRIC_DENOMINATOR = 'followers'

    print(f"--- Starting Training ---")
    print(f"Mode: {'End-to-End' if END_TO_END_TRAINING else 'Two-Stage'}")
    start_time = time.time()

    df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
    # latest_date = sorted(df_posts['datetime'].dt.to_period('M').dt.start_time.unique())[-1]
    latest_date = pd.to_datetime('2017-12-31')
    monthly_graphs, influencer_indices, _ = prepare_graph_data(end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return

    model = InfluencerRankModel(feature_dim=7, gcn_dim=GCN_DIM, rnn_dim=RNN_DIM, num_gcn_layers=NUM_GCN_LAYERS, dropout_prob=DROPOUT_PROB)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = BatchedListwiseRankingLoss()
    
    true_scores = monthly_graphs[-1].y[influencer_indices]
    display_relevance_distribution(
        true_scores.squeeze().cpu().numpy(), 
        "📊 Training Data Ground Truth Distribution"
    )
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
        print("\n--- Strategy: End-to-End Learning (Slow, High-Memory) ---")
        model.train()
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}: Performing GCN forward pass for this epoch...")
            sequence_embeddings = torch.stack([model.gcn_encoder(g.x, g.edge_index) for g in monthly_graphs])
            total_loss = 0
            for batch_indices, batch_true_scores in tqdm(dataloader, desc=f"Training Batches"):
                optimizer.zero_grad()
                batch_sequence_embeddings = sequence_embeddings[:, batch_indices].permute(1, 0, 2)
                final_user_representation = model.attentive_rnn(batch_sequence_embeddings)
                predicted_scores = model.predictor(final_user_representation)
                predicted_scores_reshaped = predicted_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                loss = criterion(predicted_scores_reshaped, batch_true_scores_reshaped)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Batch Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"✅ Model saved to '{MODEL_SAVE_PATH}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")


# ✅✅✅ --- 新しく追加したヘルパー関数 --- ✅✅✅
def display_relevance_distribution(scores, title):
    """
    エンゲージメントスコアのリストを受け取り、
    Table 2に基づいた関連性レベルの分布を表示する。
    """
    # Relevance Engagement rate E(·)
    # 5: E(·) ≥ 0.10
    # 4: 0.10 > E(·) ≥ 0.07
    # 3: 0.07 > E(·) ≥ 0.05
    # 2: 0.05 > E(·) ≥ 0.03
    # 1: 0.03 > E(·) ≥ 0.01
    # 0: 0.01 > E(·)
    
    # スコアを pandas Series に変換して処理
    scores_series = pd.Series(scores)
    relevance_series = scores_series.apply(assign_relevance_levels)
    
    counts = relevance_series.value_counts().sort_index()
    percentages = relevance_series.value_counts(normalize=True).sort_index() * 100
    
    # 分布をまとめたDataFrameを作成
    dist_df = pd.DataFrame({
        'Relevance': counts.index,
        'Count': counts.values,
        'Percentage': percentages.values
    }).set_index('Relevance')
    
    # 存在しないレベルも表示するために reindex
    dist_df = dist_df.reindex(range(6), fill_value=0)
    dist_df['Percentage'] = dist_df['Percentage'].map('{:.2f}%'.format)

    print(f"\n--- {title} ---")
    print(dist_df)

# --- 論文の基準(Table 2)に基づいて関連性レベルを割り当てる関数 ---
def assign_relevance_levels(engagement_rate):
    """エンゲージメント率を0から5の関連性スコアに変換する"""
    if engagement_rate >= 0.10: return 5
    if engagement_rate >= 0.07: return 4
    if engagement_rate >= 0.05: return 3
    if engagement_rate >= 0.03: return 2
    if engagement_rate >= 0.01: return 1
    return 0

# --- Rank-Biased Precision (RBP) を計算する関数 ---
def calculate_rbp(true_scores_in_predicted_order, p=0.95):
    """
    予測順に並べた実際のスコアリストからRBPを計算する。
    pはpersistence（持続性）パラメータ。
    """
    rbp_score = 0
    max_score = true_scores_in_predicted_order.max()
    if max_score == 0: return 0.0
    
    normalized_scores = true_scores_in_predicted_order / max_score
    
    for i, relevance in enumerate(normalized_scores):
        rbp_score += (p ** i) * relevance
        
    return (1 - p) * rbp_score


# ✅✅✅ --- NDCG@Kの計算を追加した推論関数 --- ✅✅✅
def run_inference():
    """学習済みモデルをロードし、最新データで推論を行い、各種評価指標を計算する"""
    METRIC_NUMERATOR = 'likes_and_comments'
    # 分母: 'posts' または 'followers'
    METRIC_DENOMINATOR = 'followers'
    
    print("--- 📈 Starting Inference Process ---")
    start_time = time.time()
    params = {'GCN_DIM': 128, 'NUM_GCN_LAYERS': 2, 'RNN_DIM': 64, 'DROPOUT_PROB': 0.5}

    # 1. モデルのインスタンス化と重みのロード
    model = InfluencerRankModel(feature_dim=7, gcn_dim=params['GCN_DIM'], rnn_dim=params['RNN_DIM'], num_gcn_layers=params['NUM_GCN_LAYERS'], dropout_prob=params['DROPOUT_PROB'])
    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Successfully loaded model from '{MODEL_SAVE_PATH}'")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_SAVE_PATH}'.")
        print("Please run the training process first by calling train_and_save_model().")
        return

    # 2. 推論用データと正解データの準備
    df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
    # latest_date = sorted(df_posts['datetime'].dt.to_period('M').dt.start_time.unique())[-1]
    latest_date = pd.to_datetime('2017-12-31')

    predict_graphs, predict_indices, node_to_idx = prepare_graph_data(
        end_date=latest_date, 
        num_months=12,
        metric_numerator=METRIC_NUMERATOR,
        metric_denominator=METRIC_DENOMINATOR
    )
    inference_input_graphs = predict_graphs[:-1]
    ground_truth_graph = predict_graphs[-1]

    # 3. 予測の実行
    model.eval()
    with torch.no_grad():
        sequence_embeddings = torch.stack([model.gcn_encoder(g.x, g.edge_index) for g in tqdm(inference_input_graphs, desc="GCN Encoding for Inference")])
        influencer_embeddings = sequence_embeddings[:, predict_indices].permute(1, 0, 2)
        final_representation = model.attentive_rnn(influencer_embeddings)
        predicted_scores = model.predictor(final_representation)

    # 4. 結果の集計
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    influencer_usernames = [idx_to_node[idx] for idx in predict_indices]
    true_scores = ground_truth_graph.y[predict_indices]
    
    df_results = pd.DataFrame({
        'Username': influencer_usernames,
        'Predicted_Score': predicted_scores.squeeze().cpu().numpy(),
        'True_Score': true_scores.squeeze().cpu().numpy()
    })
    
    # 5. 評価指標の計算
    
    # --- 5.1 MAE/MSE (予測値の正確さ) ---
    mae = (df_results['Predicted_Score'] - df_results['True_Score']).abs().mean()
    mse = ((df_results['Predicted_Score'] - df_results['True_Score']) ** 2).mean()
    rmse = np.sqrt(mse)

    # --- 5.2 NDCG@K (ランキングの順序評価) --- ✅ 変更箇所
    df_results['Relevance'] = df_results['True_Score'].apply(assign_relevance_levels)
    true_relevance = df_results['Relevance'].values.reshape(1, -1)
    predicted_scores_for_ndcg = df_results['Predicted_Score'].values.reshape(1, -1)
    
    ndcg_results = {}
    k_values = [1, 10, 50, 100, 200]
    for k in k_values:
        # インフルエンサーの総数よりKが大きい場合は計算しない
        if k > len(df_results):
            continue
        ndcg_results[f'NDCG@{k}'] = ndcg_score(true_relevance, predicted_scores_for_ndcg, k=k)
    
    # --- 5.3 RBP (ランキングの順序評価) ---
    df_sorted_by_pred = df_results.sort_values(by='Predicted_Score', ascending=False)
    true_scores_in_pred_order = df_sorted_by_pred['True_Score'].values
    rbp_val = calculate_rbp(true_scores_in_pred_order, p=0.95)

    # 6. 結果の表示
    df_results['Predicted_Rank'] = df_results['Predicted_Score'].rank(ascending=False, method='first').astype(int)
    
    print("\n🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆")
    print(df_results.sort_values(by='Predicted_Rank')[['Username', 'Predicted_Score', 'True_Score']].head(20).to_string(index=False))
    
    print("\n\n" + "="*50)
    print("📊 MODEL PERFORMANCE EVALUATION")
    print("="*50)
    
    # ✅✅✅ 分布表示を追加 ✅✅✅
    # --- 6.1 分布の表示 ---
    # 推論時の正解データの分布
    display_relevance_distribution(df_results['True_Score'], "📈 Inference Data Ground Truth Distribution")
    # モデルが予測したスコアの分布
    display_relevance_distribution(df_results['Predicted_Score'], "🤖 Inference Data Predicted Distribution")

    print("\n🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---")
    print(f"   - **MAE (平均絶対誤差)**: {mae:.4f}")
    print(f"     (予測が平均してどれくらい外れているか)")
    print(f"   - **RMSE (二乗平均平方根誤差)**: {rmse:.4f}")
    print(f"     (大きな外れをより重視した誤差)")

    print("\n🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---")
    # ✅ 変更箇所: NDCG@Kの結果をループで表示
    print(f"   - **NDCG@K (正規化割引累積利得)**:")
    for k_str, score in ndcg_results.items():
        print(f"     - {k_str:<8}: {score:.4f}")
    print(f"     (予測リストの上位K件における順序の正しさ。1に近いほど良い)")

    print(f"\n   - **RBP (ランクバイアス適合率)**: {rbp_val:.4f}")
    print(f"     (ユーザがリスト上位を重視する傾向を考慮した順序の正しさ)")


    end_time = time.time()
    print(f"\nTotal inference time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    # ----------------------------------------------------------------
    # --- 実行するプロセスを選択 ---
    # ----------------------------------------------------------------
    
    # モデルを学習させたい場合は、こちらを実行
    # train_and_save_model()
    
    # 学習済みモデルで推論（予測）だけを行いたい場合は、こちらをコメント解除して実行
    run_inference()