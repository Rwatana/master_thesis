import pandas as pd
import os
from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import ndcg_score

# --- 定数定義 ---
GROWTH_RATES_FILE = 'growth_rates_normalized.csv'
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
MENTIONS_FILE = 'output_mentions_all_parallel.csv'
INFLUENCERS_FILE = 'influencers.txt'
OUTPUT_FILE = 'ranking_evaluation_results.csv' # 出力ファイル名を変更

# 評価設定
GROUND_TRUTH_PERCENTILE = 0.90
K_VALUES = [1, 10, 50, 100, 200]
RBP_P = 0.95

# --- データ読み込み・特徴量エンジニアリング ---
def prepare_features():
    """
    分析に必要な全てのデータを読み込み、特徴量を計算して単一のDataFrameにまとめる。
    """
    print("--- 1. データの読み込みと特徴量エンジニアリングを開始 ---")
    
    try:
        df_growth = pd.read_csv(GROWTH_RATES_FILE)
        # 評価に使用する 'normalized_likes_growth_pct' カラムの存在チェック
        if 'normalized_likes_growth_pct' not in df_growth.columns:
             print(f"エラー: {GROWTH_RATES_FILE} に 'normalized_likes_growth_pct' カラムが必要です。")
             return None
        df_posts = pd.read_csv(PREPROCESSED_FILE, usecols=['username', 'caption', 'sentiment'])
        df_mentions = pd.read_csv(MENTIONS_FILE, header=0)
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
        df_influencers.columns = ['Username', 'Category', '#Followers', '#Followees', '#Posts']
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。 {e}")
        return None

    # --- 特徴量の計算 ---
    print("各種特徴量を計算中...")
    df_mentions.rename(columns={'target': 'mention'}, inplace=True)
    in_degree_counts = df_mentions['mention'].value_counts().reset_index()
    in_degree_counts.columns = ['username', 'in_degree']
    content_features = df_posts.groupby('username').agg(
        avg_sentiment=('sentiment', 'mean'),
        avg_caption_length=('caption', lambda x: x.str.len().mean())
    ).reset_index()
    activity_features = df_influencers[['Username', '#Posts']].rename(columns={'Username': 'username', '#Posts': 'total_posts'})

    # --- 全ての特徴量を結合 ---
    print("全ての特徴量を結合しています...")
    df_features = pd.merge(df_growth, df_influencers, left_on='username', right_on='Username', how='inner')
    df_features = pd.merge(df_features, in_degree_counts, on='username', how='left')
    df_features = pd.merge(df_features, content_features, on='username', how='left')
    df_features = pd.merge(df_features, activity_features, on='username', how='left')
    df_features.fillna(0, inplace=True)

    # --- 「正解」の定義 (参考情報用) ---
    growth_threshold = df_features['normalized_likes_growth_pct'].quantile(GROUND_TRUTH_PERCENTILE)
    df_features['is_ground_truth'] = (df_features['normalized_likes_growth_pct'] >= growth_threshold).astype(int)
    
    # relevance_score (qcut) の行を削除。 'normalized_likes_growth_pct' を直接使用する。
    
    print(f"正解ユーザー数: {df_features['is_ground_truth'].sum()} 人")
    return df_features

# --- 評価関数 ---
def calculate_rbp(true_scores_in_predicted_order, p=0.95):
    """Rank-Biased Precision (RBP) を計算する (段階的関連度バージョン)"""
    rbp_score = 0
    
    # 入力がpandas Seriesの場合も考慮して .values を使って numpy 配列を取得
    scores_array = np.asarray(true_scores_in_predicted_order)
    
    # RBPやNDCGは非負の関連度を想定するため、0未満のスコアを0にクリップする
    scores_array = np.clip(scores_array, 0, None)
    
    max_score = scores_array.max()
    if max_score == 0: return 0.0
    
    # 関連度スコアを正規化
    normalized_scores = scores_array / max_score
    
    for i, relevance in enumerate(normalized_scores):
        rbp_score += (p ** i) * relevance
    return (1 - p) * rbp_score

# --- メイン処理 ---
def main():
    start_time = time.time()
    
    df_features = prepare_features()
    if df_features is None: return
        
    # 段階的RBPでは ground_truth_set は不要になったため削除 (is_ground_truth は prepare_features 内で表示用に残す)

    print("\n--- 2. 各手法によるランキングと精度の評価を開始 ---")
    
    all_results = []
    
    # 評価する手法のリスト
    methods = {
        'User Popularity (Followers)': 'followers',
        'Post Popularity (Avg Likes)': 'average_likes',
        'User Activity (Posts)': 'total_posts',
        'Network Centrality (In-Degree)': 'in_degree',
    }

    # --- 標準的な手法の評価 ---
    for method_name, sort_col in methods.items():
        print(f"評価中: {method_name}")
        ranked_df = df_features.sort_values(sort_col, ascending=False).reset_index()
        
        # 評価指標の計算
        # 関連度スコアとして 'normalized_likes_growth_pct' を渡すように変更
        rbp_score = calculate_rbp(ranked_df['normalized_likes_growth_pct'], p=RBP_P)
        
        # NDCG用のスコア準備
        # 関連度スコアとして 'normalized_likes_growth_pct' を渡すように変更
        # 0未満の値をクリップ
        true_relevance_scores = np.clip(ranked_df['normalized_likes_growth_pct'].values, 0, None)
        true_relevance = [true_relevance_scores]
        predicted_scores = [np.arange(len(ranked_df), 0, -1)] # ランクをスコアとして使用
        
        ndcg_scores = {f'NDCG@{k}': ndcg_score(true_relevance, predicted_scores, k=k) for k in K_VALUES if k <= len(ranked_df)}

        result_row = {'Method': method_name, 'RBP': rbp_score, **ndcg_scores}
        all_results.append(result_row)

    # --- MIVモデルの評価 ---
    print("評価中: MIV Model (総合スコア)")
    df_miv = df_features.copy()
    miv_cols = ['followers', 'average_likes', 'total_posts', 'in_degree', 'avg_sentiment', 'avg_caption_length']
    for col in miv_cols:
        # ゼロ除算を避けるためのチェック
        col_min = df_miv[col].min()
        col_max = df_miv[col].max()
        if col_max - col_min > 0:
            df_miv[col] = (df_miv[col] - col_min) / (col_max - col_min)
        else:
            df_miv[col] = 0.0 # 全て同じ値の場合は0にする
            
    df_miv['miv_score'] = df_miv[miv_cols].sum(axis=1)
    ranked_miv = df_miv.sort_values('miv_score', ascending=False).reset_index()
    
    # 呼び出しを 'normalized_likes_growth_pct' を使うように変更
    rbp_miv = calculate_rbp(ranked_miv['normalized_likes_growth_pct'], p=RBP_P)
    
    # 関連度スコアとして 'normalized_likes_growth_pct' を渡すように変更
    # 0未満の値をクリップ
    true_rel_miv_scores = np.clip(ranked_miv['normalized_likes_growth_pct'].values, 0, None)
    true_rel_miv = [true_rel_miv_scores]
    pred_scores_miv = [np.arange(len(ranked_miv), 0, -1)]
    ndcg_miv = {f'NDCG@{k}': ndcg_score(true_rel_miv, pred_scores_miv, k=k) for k in K_VALUES if k <= len(ranked_miv)}
    all_results.append({'Method': 'MIV Model (Composite Score)', 'RBP': rbp_miv, **ndcg_miv})
    
    # --- 結果の表示と保存 ---
    df_results = pd.DataFrame(all_results)
    df_sorted_results = df_results.sort_values('RBP', ascending=False)
    
    print("\n--- 3. 評価結果 ---")
    print(df_sorted_results.to_string())
    
    df_sorted_results.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 評価結果を '{OUTPUT_FILE}' に保存しました。")
    
    end_time = time.time()
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    main()

