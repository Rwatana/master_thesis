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
from io import StringIO

# XAI and visualization libraries
import captum.attr as captum_attr
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# --- 1. Constants ---
# ==============================================================================
PREPROCESSED_FILE = '../posts_2017.csv'
HASHTAGS_FILE = '../hashtags_2017.csv'
MENTIONS_FILE = '../output_mentions_all_parallel.csv'
INFLUENCERS_FILE = '../influencers.txt'
MODEL_SAVE_PATH = f'influencer_rank_model_{time.strftime("%Y%m%d")}_rich_features_2017_xai.pth'
FEATURE_DIM = 0 # Global variable for feature dimension

# ==============================================================================
# --- 2. Data Preparation ---
# ==============================================================================
def prepare_graph_data(end_date, num_months=12, metric_numerator='likes', metric_denominator='posts'):
    """
    Builds a graph dataset for N months up to the specified end date.
    Only includes influencers with activity during the specified period.
    """
    print(f"\nBuilding graph sequence for {num_months} months ending on {end_date.strftime('%Y-%m')}...")
    print(f"Using Engagement Metric: {metric_numerator} / {metric_denominator}")

    # --- Load Data ---
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

    # --- Filter for Active Influencers ---
    active_influencers_set = set(df_posts['username'].unique())
    print(f"Found {len(active_influencers_set):,} active influencers in {PREPROCESSED_FILE}.")
    df_influencers = df_influencers_master[df_influencers_master['username'].isin(active_influencers_set)].copy()

    # --- Prepare Nodes ---
    influencer_set = set(df_influencers['username'].astype(str))
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
    all_mentions = set(df_mentions['mention'].astype(str))
    all_nodes = sorted(list(influencer_set | all_hashtags | all_mentions))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    influencer_indices = [node_to_idx[inf] for inf in influencer_set if inf in node_to_idx]

    # --- Feature Engineering ---
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

# ==============================================================================
# --- 3. Model & Utility Functions ---
# ==============================================================================
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
        final_representation = torch.sum(rnn_out * attention_weights, dim=1)
        return final_representation, attention_weights

class InfluencerRankModel(nn.Module):
    def __init__(self, feature_dim, gcn_dim, rnn_dim, num_gcn_layers=2, dropout_prob=0.5):
        super(InfluencerRankModel, self).__init__()
        self.gcn_encoder = GCNEncoder(feature_dim, gcn_dim, num_gcn_layers)
        self.attentive_rnn = AttentiveRNN(gcn_dim * num_gcn_layers, rnn_dim)
        self.predictor = nn.Sequential(Linear(rnn_dim, 16), ReLU(), Dropout(dropout_prob), Linear(16, 1))
    def forward(self, graph_sequence, target_indices):
        sequence_embeddings = torch.stack([self.gcn_encoder(g.x, g.edge_index) for g in graph_sequence])
        target_embeddings = sequence_embeddings[:, target_indices].permute(1, 0, 2)
        final_representation, _ = self.attentive_rnn(target_embeddings)
        predicted_scores = self.predictor(final_representation)
        return predicted_scores

class BatchedListwiseRankingLoss(nn.Module):
    def __init__(self):
        super(BatchedListwiseRankingLoss, self).__init__()
    def forward(self, pred_scores, true_scores):
        pred_probs = F.softmax(pred_scores, dim=1)
        true_probs = F.softmax(true_scores, dim=1)
        return -torch.sum(true_probs * torch.log(pred_probs + 1e-9), dim=1).mean()

def assign_relevance_levels(engagement_rate):
    if engagement_rate >= 0.10: return 5
    if engagement_rate >= 0.07: return 4
    if engagement_rate >= 0.05: return 3
    if engagement_rate >= 0.03: return 2
    if engagement_rate >= 0.01: return 1
    return 0

def set_seed(seed_value=42):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==============================================================================
# --- 4. Training & Inference ---
# ==============================================================================
def train_and_save_model():
    """Trains the model and saves its state dict."""
    # (This function is provided for completeness but not called in the final script)
    END_TO_END_TRAINING = False
    GCN_DIM = 128
    NUM_GCN_LAYERS = 2
    RNN_DIM = 64
    LEARNING_RATE = 0.001
    DROPOUT_PROB = 0.5
    NUM_EPOCHS = 200 # For a real run, might need more
    LISTS_PER_BATCH = 1024
    LIST_SIZE = 10
    BATCH_SIZE = LISTS_PER_BATCH * LIST_SIZE
    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'

    print(f"--- Starting Training ---")
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
    dataset = TensorDataset(torch.tensor(influencer_indices, dtype=torch.long), true_scores)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
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

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model saved to '{MODEL_SAVE_PATH}'")

def run_inference():
    """Runs inference and returns necessary objects for XAI analysis."""
    METRIC_NUMERATOR = 'likes_and_comments'
    METRIC_DENOMINATOR = 'followers'
    
    print("--- 📈 Starting Inference Process ---")
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
        return None, None, None, None, None

    inference_input_graphs = predict_graphs[:-1] # Use 11 months for prediction
    ground_truth_graph = predict_graphs[-1] # Last month for ground truth

    model.eval()
    with torch.no_grad():
        predicted_scores = model(inference_input_graphs, predict_indices)

    idx_to_node = {i: node for node, i in node_to_idx.items()}
    influencer_usernames = [idx_to_node[idx] for idx in predict_indices]
    true_scores = ground_truth_graph.y[predict_indices]
    
    df_results = pd.DataFrame({
        'Username': influencer_usernames,
        'Predicted_Score': predicted_scores.squeeze().cpu().numpy(),
        'True_Score': true_scores.squeeze().cpu().numpy()
    })
    
    print("\n🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆")
    print(df_results.sort_values(by='Predicted_Score', ascending=False).head(20).to_string(index=False))

    return model, predict_graphs, predict_indices, node_to_idx, df_results

# ==============================================================================
# --- 5. XAI Analysis ---
# ==============================================================================
class RnnExplainerModel(nn.Module):
    """Wrapper model for XAI to explain RNN and Predictor parts."""
    def __init__(self, attentive_rnn, predictor):
        super().__init__()
        self.attentive_rnn = attentive_rnn
        self.predictor = predictor
    def forward(self, rnn_input_sequence):
        final_representation, _ = self.attentive_rnn(rnn_input_sequence)
        prediction = self.predictor(final_representation)
        return prediction

def plot_contributions(df_plot, title, username):
    """Visualizes contributions and attention, saving the plot to a file."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    sns.barplot(x='Month', y='Contribution', hue='Method', data=df_plot[df_plot['Method'] != 'Attention'], ax=ax1, palette='viridis')
    ax1.set_ylabel('Contribution Score (from XAI methods)')
    ax1.set_xlabel('Month')
    
    ax2 = ax1.twinx()
    sns.lineplot(x='Month', y='Contribution', data=df_plot[df_plot['Method'] == 'Attention'], 
                 ax=ax2, color='red', marker='o', label='Attention Weight')
    ax2.set_ylabel('Attention Weight', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, df_plot[df_plot['Method'] == 'Attention']['Contribution'].max() * 1.5)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, title='Method')
    ax2.get_legend().remove()

    plt.title(title, fontsize=14, pad=20)
    fig.tight_layout()
    
    output_filename = f"xai_analysis_{username}.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Graph saved to: {output_filename}")
    plt.close(fig)

def run_enhanced_xai_analysis(model, graph_sequence, influencer_indices, node_to_idx, df_results):
    """Analyzes and visualizes RNN contributions for multiple influencers."""
    print("\n\n" + "="*50)
    print("🧠 ENHANCED XAI ANALYSIS: Explaining & Visualizing RNN Contributions")
    print("="*50)

    df_sorted = df_results.sort_values(by='Predicted_Score', ascending=False).reset_index(drop=True)
    targets = {
        'Top-Ranked': df_sorted.iloc[0]['Username'],
        'Bottom-Ranked': df_sorted.iloc[-1]['Username'],
        'Mid-Ranked': df_sorted.iloc[len(df_sorted) // 2]['Username']
    }

    model.eval()
    with torch.no_grad():
        print("Pre-computing GCN embeddings for all influencers...")
        sequence_embeddings = torch.stack([model.gcn_encoder(g.x, g.edge_index) for g in tqdm(graph_sequence)])
    
    explainer_model = RnnExplainerModel(model.attentive_rnn, model.predictor)
    
    background_indices = np.random.choice(len(influencer_indices), 50, replace=False)
    background_data = sequence_embeddings[:, influencer_indices].permute(1, 0, 2)[background_indices]
    
    ig = captum_attr.IntegratedGradients(explainer_model)
    explainer_shap = shap.GradientExplainer(explainer_model, background_data)

    for rank_type, username in targets.items():
        print(f"\n--- Analyzing for {rank_type} Influencer: @{username} ---")
        target_global_idx = node_to_idx[username]
        input_for_rnn = sequence_embeddings[:, target_global_idx].unsqueeze(0)

        with torch.no_grad():
            _, attention_weights = model.attentive_rnn(input_for_rnn)
        monthly_attention = attention_weights.squeeze().cpu().numpy()

        attributions_ig, _ = ig.attribute(input_for_rnn, return_convergence_delta=True)
        monthly_attr_ig = attributions_ig.squeeze(0).sum(dim=1).cpu().numpy()
        
        shap_values = explainer_shap.shap_values(input_for_rnn)
        monthly_attr_shap = shap_values.sum(axis=2).flatten()

        num_months = len(monthly_attention)
        df_plot = pd.DataFrame({
            'Month': list(range(1, num_months + 1)) * 3,
            'Contribution': np.concatenate([monthly_attr_ig, monthly_attr_shap, monthly_attention]),
            'Method': ['Integrated Gradients'] * num_months + ['SHAP'] * num_months + ['Attention'] * num_months
        })
        
        pred_score = df_results[df_results['Username']==username]['Predicted_Score'].values[0]
        true_score = df_results[df_results['Username']==username]['True_Score'].values[0]
        plot_title = (f"Monthly Contributions for {rank_type} Influencer: @{username}\n"
                      f"Predicted Score: {pred_score:.4f} | True Score: {true_score:.4f}")
        
        plot_contributions(df_plot, plot_title, username)

# ==============================================================================
# --- 6. Main Execution Block ---
# ==============================================================================
if __name__ == '__main__':
    set_seed(42)
    
    # NOTE: Training is commented out. The script assumes a pre-trained model exists.
    # To train the model first, uncomment the line below.
    # train_and_save_model()
    
    # Run inference to get predictions and necessary objects for XAI
    model, graphs, indices, node_map, results_df = run_inference()
    
    # If inference was successful, proceed with XAI analysis
    if model and graphs:
        # The graph sequence for prediction is the first 11 months
        run_enhanced_xai_analysis(model, graphs[:-1], indices, node_map, results_df)