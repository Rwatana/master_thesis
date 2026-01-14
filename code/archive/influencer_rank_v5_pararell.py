import pandas as pd
import os
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
import emoji # ✅ 絵文字カウントのために追加
from tqdm import tqdm # ✅ tqdmをトップレベルでインポート

# --- 1. 並列実行のためのライブラリ ---
import multiprocessing
import traceback # エラー詳細表示用

# --- 2. 定数定義 ---
PREPROCESSED_FILE = 'posts_2017.csv'
HASHTAGS_FILE = 'hashtags_2017.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'

# ✅ 保存先ディレクトリを定義 (グローバルパスは削除)
MODEL_DIR = 'model_v5'

# --- 3. データ準備関数 (変更) ---
def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts'):
    """
    指定された終了日までのNヶ月間のグラフデータセットを構築する。
    指定された期間に活動のあったインフルエンサーのみを対象とする。
    ✅ 最後に feature_dim を返すように変更
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
    
    # ✅ グローバル変数を削除し,ローカル変数として定義
    feature_dim = len(feature_columns)
    print(f"Total raw feature dimension: {feature_dim}")

    for snapshot_date in pd.date_range(start_date, end_date, freq='ME'): # tqdm削除 (ワーカー内でtqdmを使うため)
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
    
    # ✅ feature_dim を返す
    return monthly_graphs, influencer_indices, node_to_idx, feature_dim


# --- 4. モデル定義とその他の関数 (変更なし) ---
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
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5, projection_dim=128):
        super(InfluencerRankModel, self).__init__()
        print(f"\nInitializing Model:")
        print(f"  Raw Features (Input): {feature_dim}")
        print(f"  Projection Dim (GCN Input): {projection_dim}")
        print(f"  GCN Hidden Dim: {gcn_dim} (x{num_gcn_layers} layers -> Output: {gcn_dim * num_gcn_layers})")
        print(f"  RNN Hidden Dim (Predictor Input): {rnn_dim}")
        self.projection_layer = nn.Sequential(Linear(feature_dim, projection_dim), ReLU())
        self.gcn_encoder = GCNEncoder(projection_dim, gcn_dim, num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        self.predictor = nn.Sequential(Linear(rnn_dim, 16), ReLU(), Dropout(dropout_prob), Linear(16, 1))

    def forward(self, graph_sequence, target_indices, debug_print=False):
        if debug_print: 
            print(f"\n--- 🐛 DEBUG: Model Forward Pass (BatchSize={len(target_indices)}) ---")
            print(f"Input: {len(graph_sequence)} graphs, {len(target_indices)} target indices")
        gcn_inputs = []
        for i, g in enumerate(graph_sequence):
            if i == 0 and debug_print: print(f"[1] Projection Layer Input (g.x shape, T=0): {g.x.shape}")
            projected_x = self.projection_layer(g.x)
            if i == 0 and debug_print: print(f"[1] Projection Layer Output (shape, T=0):      {projected_x.shape}")
            gcn_inputs.append(projected_x)
        sequence_embeddings_list = []
        for i, (g, projected_x) in enumerate(zip(graph_sequence, gcn_inputs)):
            if i == 0 and debug_print: print(f"\n[2] GCN Encoder Input (projected_x, T=0): {projected_x.shape}")
            if i == 0 and debug_print: print(f"[2] GCN Encoder Input (edge_index, T=0):  {g.edge_index.shape}")
            gcn_out = self.gcn_encoder(projected_x, g.edge_index)
            if i == 0 and debug_print: print(f"[2] GCN Encoder Output (shape, T=0):      {gcn_out.shape}")
            sequence_embeddings_list.append(gcn_out)
        
        sequence_embeddings = torch.stack(sequence_embeddings_list)
        if debug_print: print(f"\n[3] Stacked GCN Embeddings (Seq, AllNodes, Feat): {sequence_embeddings.shape}")
        
        target_embeddings = sequence_embeddings[:, target_indices].permute(1, 0, 2)
        if debug_print: print(f"[3] Target Embeddings (Batch, Seq, Feat):       {target_embeddings.shape}")
        if debug_print: print(f"\n[4] Attentive RNN Input (Batch, Seq, Feat):  {target_embeddings.shape}")
        final_representation = self.attentive_rnn(target_embeddings)
        if debug_print: print(f"[4] Attentive RNN Output (Batch, Feat): {final_representation.shape}")
        if debug_print: print(f"\n[5] Predictor Input (Batch, Feat):  {final_representation.shape}")
        predicted_scores = self.predictor(final_representation)
        if debug_print: print(f"[5] Predictor Output (Batch, 1): {predicted_scores.shape}")
        if debug_print: print(f"--- 🐛 End Debug ---")
        return predicted_scores

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


# --- 5. train_and_save_model (変更) ---
# ✅ 引数に model_save_path を追加
# ✅ 最後に feature_dim を return する
def train_and_save_model(model_save_path):
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
    PROJECTION_DIM = 128

    print(f"--- Starting Training ---")
    print(f"Saving model to: {model_save_path}")
    start_time = time.time()
    
    latest_date = pd.to_datetime('2017-12-31')
    
    # ✅ feature_dim を受け取る
    monthly_graphs, influencer_indices, _, feature_dim = prepare_graph_data(
        end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
    
    if not monthly_graphs:
        print("No graph data was created. Exiting.")
        return None # ✅ 失敗時にNoneを返す

    # ✅ 受け取った feature_dim をモデルに渡す
    model = InfluencerRankModel(
        feature_dim=feature_dim, 
        gcn_dim=GCN_DIM, 
        rnn_dim=RNN_DIM, 
        num_gcn_layers=NUM_GCN_LAYERS, 
        dropout_prob=DROPOUT_PROB,
        projection_dim=PROJECTION_DIM 
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_listwise = BatchedListwiseRankingLoss()
    criterion_pointwise = nn.MSELoss() 
    alpha = 1 
    
    true_scores = monthly_graphs[-1].y[influencer_indices]
    display_relevance_distribution(true_scores.squeeze().cpu().numpy(), "📊 Training Data Ground Truth Distribution")
    dataset = TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), true_scores)
    
    # drop_last=True が重要 (バッチサイズが揃わないと Two-Stage でエラーになる)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # (Tqdmのdescを簡潔に)
    if not END_TO_END_TRAINING:
        print("\n--- Strategy: Two-Stage Learning (Fast) ---")
        model.projection_layer.eval()
        model.gcn_encoder.eval()
        with torch.no_grad():
            sequence_embeddings_list = []
            print("Running Projection + GCN Encoding (Two-Stage)...")
            debug_print_2stage = True 
            
            for i, g in enumerate(monthly_graphs): # tqdm削除
                if i == 0 and debug_print_2stage: print(f"[1] Projection Input (T=0): {g.x.shape}")
                projected_x = model.projection_layer(g.x)
                if i == 0 and debug_print_2stage: print(f"[1] Projection Output (T=0): {projected_x.shape}")
                if i == 0 and debug_print_2stage: print(f"[2] GCN Input (T=0): {projected_x.shape}")
                gcn_out = model.gcn_encoder(projected_x, g.edge_index)
                if i == 0 and debug_print_2stage: print(f"[2] GCN Output (T=0): {gcn_out.shape}")
                sequence_embeddings_list.append(gcn_out)
                
            sequence_embeddings = torch.stack(sequence_embeddings_list)
            if debug_print_2stage: print(f"[3] Stacked GCN Embeddings (Seq, AllNodes, Feat): {sequence_embeddings.shape}")

        model.attentive_rnn.train()
        model.predictor.train()
        
        # (デバッグプリントを削除... ログが膨大になるため)

        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            # ✅ tqdmをエポックループ内に配置
            for batch_indices, batch_true_scores in dataloader: # tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
                optimizer.zero_grad()
                batch_sequence_embeddings = sequence_embeddings[:, batch_indices].permute(1, 0, 2)
                final_user_representation = model.attentive_rnn(batch_sequence_embeddings)
                predicted_scores = model.predictor(final_user_representation)
                predicted_scores_reshaped = predicted_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                loss_listwise = criterion_listwise(predicted_scores_reshaped, batch_true_scores_reshaped)
                loss_pointwise = criterion_pointwise(predicted_scores.squeeze(), batch_true_scores.squeeze())
                loss = loss_listwise + alpha * loss_pointwise
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # 10エポックごとに表示 (ログ削減のため)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Batch Loss: {total_loss / len(dataloader):.4f}")
    else:
        print("\n--- Strategy: End-to-End Learning (Slow, High-Memory) ---")
        model.train() 
        for epoch in range(NUM_EPOCHS):
            total_loss = 0
            # ✅ tqdmをエポックループ内に配置
            for batch_indices, batch_true_scores in dataloader: # tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
                optimizer.zero_grad()
                predicted_scores = model(monthly_graphs, batch_indices, debug_print=False) 
                predicted_scores_reshaped = predicted_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                loss_listwise = criterion_listwise(predicted_scores_reshaped, batch_true_scores_reshaped)
                loss_pointwise = criterion_pointwise(predicted_scores.squeeze(), batch_true_scores.squeeze())
                loss = loss_listwise + alpha * loss_pointwise
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # 10エポックごとに表示 (ログ削減のため)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Batch Loss: {total_loss / len(dataloader):.4f}")
    
    # ✅ 引数で受け取ったパスに保存
    torch.save(model.state_dict(), model_save_path)
    end_time = time.time()
    print("\n--- Training Complete ---")
    print(f"✅ Model saved to '{model_save_path}'")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    
    # ✅ 学習に使った feature_dim を返す
    return feature_dim


# --- 6. run_inference (変更) ---
# ✅ 引数に model_save_path と trained_feature_dim を追加
# ✅ 最後にメトリクスの辞書を return する
def run_inference(model_save_path, trained_feature_dim):
    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'
    PROJECTION_DIM = 128
    
    print("--- 📈 Starting Inference Process ---")
    start_time = time.time()
    params = {'GCN_DIM': 128, 'NUM_GCN_LAYERS': 2, 'RNN_DIM': 64, 'DROPOUT_PROB': 0.5}

    latest_date = pd.to_datetime('2017-12-31')
    
    # ✅ feature_dim を捨てる (データ準備のみが目的)
    predict_graphs, predict_indices, node_to_idx, _ = prepare_graph_data(
        end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
    
    if not predict_graphs or len(predict_graphs) < 2:
        print("Not enough graph data for inference. Skipping.")
        return None

    # ✅ 引数で渡された trained_feature_dim を使ってモデルを初期化
    model = InfluencerRankModel(
        feature_dim=trained_feature_dim, 
        gcn_dim=params['GCN_DIM'], 
        rnn_dim=params['RNN_DIM'], 
        num_gcn_layers=params['NUM_GCN_LAYERS'], 
        dropout_prob=params['DROPOUT_PROB'],
        projection_dim=PROJECTION_DIM
    )
    try:
        # ✅ 引数の model_save_path から読み込み
        model.load_state_dict(torch.load(model_save_path))
        print(f"Successfully loaded model from '{model_save_path}'")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{model_save_path}'. Please run training first.")
        return None # ✅ 失敗時にNoneを返す
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print(f"Model was initialized with feature_dim={trained_feature_dim}. Check if this matches the saved model.")
        return None

    inference_input_graphs = predict_graphs[:-1] 
    ground_truth_graph = predict_graphs[-1]

    model.eval()
    with torch.no_grad():
        # デバッグプリントはオフにする (ログ削減)
        predicted_scores = model(inference_input_graphs, predict_indices, debug_print=False)

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

    # ✅ メトリクスを辞書として返す
    results = {
        'mae': mae, 
        'rmse': rmse, 
        'rbp': rbp_val,
        **ndcg_results
    }
    return results

# --- 7. 乱数シード設定関数 (変更なし) ---
def set_seed(seed_value=42):
    np.random.seed(seed_value) 
    torch.manual_seed(seed_value)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 8. ✅ 並列実行のためのワーカー関数 ---
def run_experiment(run_args):
    """
    1回の「学習＋推論」を実行するワーカー関数。
    multiprocessing.Pool から呼び出される。
    """
    run_id, seed = run_args
    
    # 実行IDとシードでログを区別
    print(f"\n--- 🚀 STARTING RUN {run_id} (Seed: {seed}) 🚀 ---")
    
    try:
        # 1. このプロセス固有のシードを設定
        set_seed(seed)
        
        # 2. このプロセス固有のモデル保存パスを決定
        model_save_path = os.path.join(MODEL_DIR, f'model_run_{run_id}_seed_{seed}.pth')
        
        # 3. 学習を実行し,学習時
        # の特徴量次元数を取得
        trained_feature_dim = train_and_save_model(model_save_path)
        
        if trained_feature_dim is None:
            print(f"[Run {run_id}] FAILED during training.")
            return None

        # 4. 推論を実行
        # 学習時と同じ特徴量次元数を渡してモデルアーキテクチャを揃える
        metrics = run_inference(model_save_path, trained_feature_dim)
        
        if metrics is None:
            print(f"[Run {run_id}] FAILED during inference.")
            return None
            
        # 5. 結果にIDとシードを追加して返す
        metrics['run_id'] = run_id
        metrics['seed'] = seed
        
        print(f"--- ✅ FINISHED RUN {run_id} (Seed: {seed}) | MAE: {metrics['mae']:.4f} ---")
        return metrics

    except Exception as e:
        # エラーが発生しても他のプロセスを止めない
        print(f"\n--- ❌ CRITICAL FAILURE IN RUN {run_id} (Seed: {seed}) ❌ ---")
        print(f"Error: {e}")
        traceback.print_exc() # エラーの詳細を出力
        return None

# --- 9. ✅ メイン実行ブロック (全面的に書き換え) ---
if __name__ == '__main__':
    # --- 実験設定 ---
    NUM_TOTAL_RUNS = 20 # 実行したい総回数
    START_SEED = 42      # 最初のシード値 (42, 43, 44... と増えていきます)
    
    # 警告: この値を大きくしすぎないでください！
    # PCのCPUコア数 / 2,またはGPUメモリが許す数 (例: 2, 4, 8) を推奨します。
    # 100に設定するとシステムがクラッシュします。
    NUM_PARALLEL_WORKERS = 4 
    
    print("="*60)
    print(f"STARTING EXPERIMENT SUITE: {NUM_TOTAL_RUNS} runs")
    print(f"MAX PARALLEL WORKERS: {NUM_PARALLEL_WORKERS}")
    print(f"MODEL DIRECTORY: {MODEL_DIR}")
    print("="*60)

    # 1. 保存先フォルダを作成
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. 実行する100個のタスク (run_id, seed) のリストを作成
    # 例: [(0, 42), (1, 43), (2, 44), ..., (99, 141)]
    jobs = [(i, START_SEED + i) for i in range(NUM_TOTAL_RUNS)]

    # 3. プロセスプールを作成して並列実行
    # `imap` を使うと,タスクが完了した順に結果を処理できます
    # `tqdm` で全体の進捗を表示
    all_results = []
    
    # Windows/macOS で PyTorch を multiprocessing するときの "fork" 問題を回避
    multiprocessing.set_start_method('spawn', force=True) 
    
    with multiprocessing.Pool(processes=NUM_PARALLEL_WORKERS) as pool:
        # pool.imap(関数, 引数のリスト)
        # tqdm でラップして進捗を表示
        with tqdm(total=NUM_TOTAL_RUNS, desc="Processing Experiments") as pbar:
            for result in pool.imap(run_experiment, jobs):
                if result is not None:
                    all_results.append(result)
                pbar.update(1) # 進捗バーを1つ進める

    print("\n\n" + "="*60)
    print("🚀 ALL RUNS COMPLETE - FINAL RESULTS SUMMARY 🚀")
    print("="*60)

    if not all_results:
        print("No results were collected. All runs may have failed.")
    else:
        # 4. 結果をDataFrameにまとめて集計
        df_results = pd.DataFrame(all_results)
        
        # カラムの順序を整える
        cols = ['run_id', 'seed', 'mae', 'rmse', 'rbp'] + [k for k in df_results.columns if 'NDCG' in k]
        df_results = df_results[cols]

        print("\n--- 📈 Individual Run Results ---")
        print(df_results.to_string(index=False))

        # 5. 平均と標準偏差を計算
        df_summary_mean = df_results.drop(columns=['run_id', 'seed']).mean()
        df_summary_std = df_results.drop(columns=['run_id', 'seed']).std()
        
        summary_df = pd.DataFrame({
            'Mean': df_summary_mean,
            'StdDev': df_summary_std
        })

        print("\n\n--- 📊 Aggregate Metrics (Mean & StdDev over {len(df_results)} runs) ---")
        print(summary_df.to_string())
        
        # 6. 結果をCSVに保存
        output_csv_path = 'experiment_results_v5.csv'
        df_results.to_csv(output_csv_path, index=False)
        print(f"\n✅ Full results saved to '{output_csv_path}'")