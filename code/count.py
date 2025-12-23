import pandas as pd
import os
import time

# --- 定数定義 ---
# YEAR_TO_PROCESS = 2017 # 2017年のデータを集計する場合
YEAR_TO_PROCESS = 2018 # 2018年のデータを集計する場合

BASE_DIR = f'dataset'
TRAIN_FILE = os.path.join(BASE_DIR, 'training', 'posts.csv')
TEST_FILE = os.path.join(BASE_DIR, 'testing', 'posts.csv')

def count_posts_in_split(filepath, split_name):
    """
    指定されたposts.csvファイルを読み込み、ユーザーごとの投稿数を集計して表示する。
    """
    print(f"\n--- {split_name} データセットの投稿数を集計しています ---")
    try:
        df = pd.read_csv(filepath, usecols=['username'])
        print(f"'{filepath}' から {len(df)} 件の投稿を読み込みました。")
        
        # ユーザーごとの投稿数をカウント
        user_counts = df['username'].value_counts().reset_index()
        user_counts.columns = ['username', 'post_count']
        
        print(f"{split_name} データセット内のユニークユーザー数: {len(user_counts)} 人")
        print("投稿数が多いユーザー TOP 5:")
        print(user_counts.head())
        
        return user_counts
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
        return pd.DataFrame(columns=['username', 'post_count'])

def main():
    """
    メインの処理を実行する関数。
    """
    start_time = time.time()
    
    # 訓練データとテストデータの両方を処理
    train_counts = count_posts_in_split(TRAIN_FILE, "訓練 (Training)")
    test_counts = count_posts_in_split(TEST_FILE, "テスト (Testing)")

    end_time = time.time()
    print("\n--- 全ての集計が完了しました ---")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")


if __name__ == '__main__':
    main()
