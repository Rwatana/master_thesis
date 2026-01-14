import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import concurrent.futures
from tqdm import tqdm
import time
import os
import multiprocessing
from growth_analyser_worker import calculate_growth_for_user
# --- 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
INFLUENCERS_FILE = 'influencers.txt'
OUTPUT_FILE = 'growth_rates_normalized.csv'

# グローバル変数としてデータフレームを定義
df_posts_global = None

# --- ワーカー関数 ---
# def calculate_growth_for_user(user_data):
#     """単一ユーザーのデータフレームを受け取り,成長率を計算する"""
#     username, user_df = user_data
#     if len(user_df) < 2:
#         return None

#     user_df = user_df.sort_values('datetime').copy()
#     user_df['days_since_start'] = (user_df['datetime'] - user_df['datetime'].min()).dt.days
    
#     model = LinearRegression()
#     X = user_df[['days_since_start']]
    
#     # likesの傾き
#     model.fit(X, user_df['likes'])
#     likes_slope = model.coef_[0]
    
#     # commentsの傾き
#     model.fit(X, user_df['comments'])
#     comments_slope = model.coef_[0]

#     return {
#         'username': username,
#         'likes_growth_rate': likes_slope,
#         'comments_growth_rate': comments_slope
#     }

# --- メイン処理 ---
def main():
    """メインの処理を実行する関数"""
    global df_posts_global

    print("--- 成長率とインフルエンサータイプの計算を開始します ---")

    # --- データの読み込み ---
    try:
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
        df_posts_global = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], usecols=['username', 'datetime', 'likes', 'comments'])
    except FileNotFoundError as e:
        print(f"エラー: 必要なファイルが見つかりません。 {e}")
        return
        
    print(f"'{PREPROCESSED_FILE}' から {len(df_posts_global)} 件の投稿データを読み込みました。")

    # --- 成長率の並列計算 ---
    user_groups = list(df_posts_global.groupby('username'))
    print(f"{len(user_groups)} 人のユーザーを対象に成長率を計算します...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(calculate_growth_for_user, user_groups)
        results = list(tqdm(results_iterator, total=len(user_groups), desc="成長率を計算中"))
    
    df_growth = pd.DataFrame([r for r in results if r is not None])

    # --- 平均いいね数の計算 ---
    df_avg_likes = df_posts_global.groupby('username')['likes'].mean().rename('average_likes').reset_index()

    # --- データの結合 ---
    df_features = pd.merge(df_growth, df_avg_likes, on='username', how='inner')
    df_full_features = pd.merge(df_features, df_influencers[['Username', '#Followers']], left_on='username', right_on='Username', how='left')
    df_full_features.rename(columns={'#Followers': 'followers'}, inplace=True)
    
    # --- 正規化成長率の計算 ---
    # 平均いいね数が0の場合の0除算を避ける
    safe_avg_likes = df_full_features['average_likes'].replace(0, 1)
    df_full_features['normalized_likes_growth_pct'] = (df_full_features['likes_growth_rate'] / safe_avg_likes) * 100
    df_full_features['normalized_comments_growth_pct'] = (df_full_features['comments_growth_rate'] / safe_avg_likes) * 100

    # ▼▼▼ 修正点: 新しい分類基準を適用 ▼▼▼
    # --- インフルエンサータイプの分類 ---
    print("新しい基準でインフルエンサータイプを分類しています...")
    bins = [1000, 10000, 100000, 1000000, float('inf')]
    labels = ['Nano', 'Micro', 'Macro', 'Mega']
    df_full_features['influencer_type'] = pd.cut(df_full_features['followers'], bins=bins, labels=labels, right=False)
    # ▲▲▲ 修正点 ▲▲▲

    # --- CSVに保存 ---
    df_full_features.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ 計算結果を '{OUTPUT_FILE}' に保存しました。")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\n--- 全ての処理が完了しました ---")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

