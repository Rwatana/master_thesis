# import pandas as pd

# # --- ファイルパス定義 ---
# PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
# INFLUENCERS_FILE = 'influencers.txt'

# def analyze_influencer_activity_2017():
#     """
#     2017年を通して活動していたインフルエンサーの数を計算する。
#     """
#     print("--- 🔍 Analyzing influencer activity for 2017 ---")

#     # --- 1. データ読み込み ---
#     print("[Step 1/3] Loading data...")
#     try:
#         # 投稿データ
#         df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
        
#         # インフルエンサーリスト
#         with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f: lines = f.readlines()
#         lines = [line for line in lines if '===' not in line]
#         from io import StringIO
#         df_influencers = pd.read_csv(StringIO("".join(lines)), sep='\t', dtype=str)
#         influencer_set = set(df_influencers['Username'])
        
#     except FileNotFoundError as e:
#         print(f"Error: Could not find the file {e.filename}. Please check the file path.")
#         return

#     # --- 2. データの前処理 ---
#     print("[Step 2/3] Filtering data for 2017...")
#     # インフルエンサーの投稿のみに絞る
#     df_posts_influencers = df_posts[df_posts['username'].isin(influencer_set)]
    
#     # 2017年の投稿のみに絞る
#     df_2017 = df_posts_influencers[df_posts_influencers['datetime'].dt.year == 2017].copy()
    
#     if df_2017.empty:
#         print("No posts found for 2017.")
#         return
        
#     # 月情報を追加
#     df_2017['month'] = df_2017['datetime'].dt.month

#     # --- 3. アクティブなインフルエンサーの集計 ---
#     print("[Step 3/3] Aggregating active influencers...")
    
#     # 各インフルエンサーの活動月をリスト化
#     monthly_activity = df_2017.groupby('username')['month'].unique().apply(list)
    
#     # --- 定義A: 第1四半期 AND 第4四半期 ---
#     q1_months = {1, 2, 3}
#     q4_months = {10, 11, 12}
    
#     active_q1_and_q4 = monthly_activity.apply(
#         lambda months: any(m in q1_months for m in months) and any(m in q4_months for m in months)
#     )
#     count_q1_q4 = active_q1_and_q4.sum()

#     # --- 定義B: 1月 AND 12月 ---
#     active_jan_and_dec = monthly_activity.apply(
#         lambda months: 1 in months and 12 in months
#     )
#     count_jan_dec = active_jan_and_dec.sum()
    
#     # --- 結果表示 ---
#     print("\n" + "="*50)
#     print("--- 📊 Results ---")
#     print("="*50)
#     print(f"Total influencers in 'influencers.txt': {len(influencer_set):,}")
#     print(f"Influencers with any post in 2017:      {len(monthly_activity):,}")
#     print("\n--- Activity Definitions ---")
#     print(f"✅ Active in Q1 (Jan-Mar) AND Q4 (Oct-Dec): {count_q1_q4:,} users")
#     print(f"   (This is the main answer)")
#     print(f"✅ Active in January AND December:          {count_jan_dec:,} users")
#     print(f"   (Stricter definition)")


# if __name__ == '__main__':
#     analyze_influencer_activity_2017()

import pandas as pd

# --- ファイルパス定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata.csv'
INFLUENCERS_FILE = 'influencers.txt'

def find_best_threshold():
    """
    最低投稿数の閾値を複数試行し,それぞれの結果を一覧で表示する。
    """
    print("--- 🔬 Finding the best threshold to match the paper's stats ---")

    # --- 1. データ読み込み ---
    print("[Step 1/3] Loading data files...")
    try:
        df_posts = pd.read_csv(PREPROCESSED_FILE, parse_dates=['datetime'], low_memory=False)
        with open(INFLUENCERS_FILE, 'r', encoding='utf-8') as f:
            lines = [line for line in f if '===' not in line]
        from io import StringIO
        df_influencers = pd.read_csv(StringIO("".join(lines)), sep='\t', dtype=str)
        master_influencer_set = set(df_influencers['Username'])
    except FileNotFoundError as e:
        print(f"Error: Could not find the file {e.filename}. Please check the file path.")
        return

    # --- 2. 2017年のインフルエンサー投稿を抽出 ---
    print("[Step 2/3] Filtering data for 2017 and counting posts per influencer...")
    df_posts_filtered = df_posts[df_posts['username'].isin(master_influencer_set)]
    df_2017 = df_posts_filtered[df_posts_filtered['datetime'].dt.year == 2017]

    if df_2017.empty:
        print("No posts found for 2017.")
        return

    # インフルエンサーごとの投稿数を事前に計算（ループの外で一度だけ）
    post_counts = df_2017['username'].value_counts()

    # --- 3. forループで各閾値をテスト ---
    print("[Step 3/3] Testing various thresholds...")

    # ✅✅✅ ここで試したい「最低投稿数」のリストを定義 ✅✅✅
    thresholds_to_test = [1, 5, 10, 12, 15, 18, 20, 25, 30]
    
    results = []

    for threshold in thresholds_to_test:
        # 現在の閾値でインフルエンサーをフィルタリング
        active_influencers = post_counts[post_counts >= threshold].index
        
        # フィルタリング後の投稿データセットを作成
        final_posts_df = df_2017[df_2017['username'].isin(active_influencers)]
        
        # 結果を格納
        results.append({
            'Min Posts': threshold,
            'Influencer Count': len(active_influencers),
            'Post Count': len(final_posts_df)
        })

    # --- 結果をまとめて表示 ---
    results_df = pd.DataFrame(results)
    
    # 見やすいようにフォーマット
    results_df['Influencer Count'] = results_df['Influencer Count'].map('{:,}'.format)
    results_df['Post Count'] = results_df['Post Count'].map('{:,}'.format)

    print("\n" + "="*65)
    print("--- 📊 Results of Threshold Testing ---")
    print("="*65)
    print(results_df.to_string(index=False))
    print("="*65)

    print("\n--- 📜 Paper's Stats for Reference ---")
    print(f"Target Influencer Count: 18,397")


if __name__ == '__main__':
    find_best_threshold()
    