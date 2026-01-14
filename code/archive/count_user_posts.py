import pandas as pd
import os
from collections import Counter
from tqdm import tqdm
import time

# --- 定数定義 ---
INFO_DIR = 'posts_info/unzipped_data_7z/info/'
INFLUENCERS_FILE = 'influencers.txt' # influencers.txtのパスを追加
OUTPUT_CSV_FILE = 'user_post_counts.csv'

def count_posts_per_user():
    """
    指定されたディレクトリ内のファイル名を解析し,ユーザーごとのファイル数を集計してCSVに保存する。
    さらに,influencers.txtとの差分を計算して表示する。
    """
    print(f"--- ユーザーごとの投稿ファイル数の集計を開始します ---")
    start_time = time.time()

    # --- 1. .infoファイルからユーザーリストを取得 ---
    try:
        all_files = os.listdir(INFO_DIR)
        print(f"ディレクトリ '{INFO_DIR}' から {len(all_files)} 個のファイルを検出しました。")
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{INFO_DIR}' が見つかりません。")
        return

    usernames_from_dir = []
    for filename in tqdm(all_files, desc="ファイル名を解析中"):
        if '-' in filename:
            username = filename.split('-', 1)[0]
            usernames_from_dir.append(username)

    if not usernames_from_dir:
        print("解析対象のファイルが見つかりませんでした。")
        return

    user_counts = Counter(usernames_from_dir)
    info_dir_users = set(user_counts.keys()) # Setに変換
    print(f".infoファイルから {len(info_dir_users)} 人のユニークユーザーが見つかりました。")

    # --- 2. influencers.txtからユーザーリストを取得 ---
    try:
        df_influencers = pd.read_csv(INFLUENCERS_FILE, sep='\t', skiprows=[1])
        influencer_txt_users = set(df_influencers['Username'].unique())
        print(f"'{INFLUENCERS_FILE}' から {len(influencer_txt_users)} 人のユニークユーザーが見つかりました。")
    except FileNotFoundError:
        print(f"警告: '{INFLUENCERS_FILE}' が見つかりませんでした。差分計算はスキップします。")
        influencer_txt_users = set() # 空のセットとして初期化
    
    # --- 3. 差分の計算と表示 ---
    if influencer_txt_users:
        print("\n--- ユーザーリストの差分分析 ---")
        
        # influencers.txtにのみ存在するユーザー
        only_in_txt = influencer_txt_users - info_dir_users
        if only_in_txt:
            print(f"✅ '{INFLUENCERS_FILE}' にのみ存在するユーザーが {len(only_in_txt)} 人見つかりました。")
            print("   (プロフィール情報はあるが,投稿ファイルがないユーザー)")
            print(f"   例: {list(only_in_txt)[:5]}") # 最初の5人を例として表示
        else:
            print(f"✅ '{INFLUENCERS_FILE}' の全ユーザーは投稿ファイルを持っています。")

        # .infoディレクトリにのみ存在するユーザー
        only_in_dir = info_dir_users - influencer_txt_users
        if only_in_dir:
            print(f"✅ 投稿ファイルのみが存在するユーザーが {len(only_in_dir)} 人見つかりました。")
            print("   (投稿ファイルはあるが,'influencers.txt' にプロフィール情報がないユーザー)")
            print(f"   例: {list(only_in_dir)[:5]}") # 最初の5人を例として表示
        else:
            print(f"✅ 投稿ファイルを持つ全ユーザーは '{INFLUENCERS_FILE}' にも記載されています。")

    # --- 4. 既存の集計とCSV保存 ---
    df_counts = pd.DataFrame(user_counts.items(), columns=['username', 'post_count'])
    df_sorted = df_counts.sort_values('post_count', ascending=False).reset_index(drop=True)
    df_sorted.to_csv(OUTPUT_CSV_FILE, index=False)

    end_time = time.time()
    print(f"\n--- 処理完了 ---")
    print(f"✅ 集計結果を '{OUTPUT_CSV_FILE}' に保存しました。")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")

if __name__ == '__main__':
    count_posts_per_user()

