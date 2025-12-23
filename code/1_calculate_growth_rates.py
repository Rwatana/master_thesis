import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import concurrent.futures
from tqdm import tqdm
import time
import multiprocessing

# --- 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata_clean.csv'
INFLUENCERS_FILE = 'influencers.txt'
OUTPUT_FILE = 'growth_rates_normalized.csv'

def calculate_growth_for_user(user_data_with_followers):
    """
    単一ユーザーのデータを受け取り、正規化成長率とカテゴリを計算するワーカー関数。
    """
    username, user_df, follower_count = user_data_with_followers
    
    # 統計的に意味のある回帰を行うため、最低5投稿は必要とする
    if len(user_df) < 5:
        return None

    user_df = user_df.sort_values('datetime').copy()
    start_date = user_df['datetime'].min()
    user_df['days_since_start'] = (user_df['datetime'] - start_date).dt.days
    
    model = LinearRegression()
    X = user_df[['days_since_start']]
    
    # --- いいね数の分析 ---
    y_likes = user_df['likes']
    model.fit(X, y_likes)
    likes_slope = model.coef_[0]
    average_likes = y_likes.mean()
    # 0除算を避けつつ、正規化成長率をパーセントで計算
    normalized_likes_growth = (likes_slope / average_likes) * 100 if average_likes > 1 else 0

    # --- コメント数の分析 ---
    y_comments = user_df['comments']
    model.fit(X, y_comments)
    comments_slope = model.coef_[0]
    average_comments = y_comments.mean()
    normalized_comments_growth = (comments_slope / average_comments) * 100 if average_comments > 1 else 0
    
    # --- インフルエンサーのカテゴリ分類 ---
    if follower_count < 50000:
        influencer_type = 'Micro'
    elif 50000 <= follower_count < 100000:
        influencer_type = 'Meso'
    elif 100000 <= follower_count < 1000000:
        influencer_type = 'Macro'
    else:
        influencer_type = 'Mega'

    return {
        'username': username,
        'influencer_type': influencer_type,
        'likes_growth_rate': likes_slope,
        'normalized_likes_growth_pct': normalized_likes_growth,
        'comments_growth_rate': comments_slope,
        'normalized_comments_growth_pct': normalized_comments_growth,
        'average_likes': average_likes
    }

def main():
    """データ処理を駆動するメイン関数"""
    print("--- 成長率とインフルエンサータイプの計算を開始します ---")

    # データの読み込み
    try:
        print(f"'{PREPROCESSED_FILE}' を読み込み中...")
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'])
        print(f"'{INFLUENCERS_FILE}' を読み込み中...")
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
    except FileNotFoundError as e:
        print(f"エラー: {e}")
        return

    # ユーザーごとのフォロワー数を投稿データに結合
    df_merged = pd.merge(df_posts, df_influencers[['Username', '#Followers']], left_on='username', right_on='Username', how='left')
    df_merged.dropna(subset=['#Followers'], inplace=True)
    df_merged['#Followers'] = df_merged['#Followers'].astype(int)

    # 並列処理用の入力データを作成: (ユーザー名, そのユーザーの投稿DF, フォロワー数)
    user_groups = df_merged.groupby('username')
    data_for_processing = [
        (name, group, group['#Followers'].iloc[0]) for name, group in user_groups
    ]
    
    print(f"{len(data_for_processing)}人のユーザーを対象に並列処理を開始します...")

    # 並列処理の実行
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results_iterator = executor.map(calculate_growth_for_user, data_for_processing)
        results = list(tqdm(results_iterator, total=len(data_for_processing), desc="ユーザーの成長率を計算中"))

    # 結果を整形して保存
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        print("有効な結果がありませんでした。")
        return

    df_final = pd.DataFrame(valid_results)
    df_final.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n--- 処理完了 ---")
    print(f"✅ {len(df_final)}人のユーザーデータを'{OUTPUT_FILE}'に保存しました。")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

