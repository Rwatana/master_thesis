import pandas as pd
import os
from tqdm import tqdm
import concurrent.futures
import time
import multiprocessing

# --- 定数定義 ---
PREPROCESSED_FILE = 'preprocessed_posts_with_metadata_clean.csv'
OUTPUT_DIR = 'user_data'  # ユーザーごとのファイル出力先ディレクトリ

# グローバル変数としてデータフレームを定義
df_posts = None

def save_user_file(username):
    """
    単一ユーザーのデータフレームをフィルタリングし,CSVとして保存するワーカー関数。
    """
    try:
        # グローバルなdf_postsから該当ユーザーのデータを抽出
        df_user = df_posts[df_posts['username'] == username]
        
        if not df_user.empty:
            output_filename = f"{username}.csv"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            
            # 日時でソートして保存
            df_user.sort_values('datetime', ascending=False).to_csv(output_path, index=False)
            return f"✅ ユーザー '{username}': {len(df_user)} 件を保存しました。"
        else:
            return f"⚠️ ユーザー '{username}': データがありませんでした。"
    except Exception as e:
        return f"❌ ユーザー '{username}': エラーが発生しました - {e}"

def main():
    """
    メインの処理を実行する関数。
    """
    global df_posts # グローバル変数を更新することを明示

    print("--- ユーザーごとのファイル集計処理を開始します ---")

    # --- 1. データの読み込み ---
    try:
        print(f"'{PREPROCESSED_FILE}'を読み込んでいます...")
        df_posts = pd.read_csv(PREPROCESSED_FILE)
    except FileNotFoundError:
        print(f"エラー: '{PREPROCESSED_FILE}' が見つかりません。先に preprocess_data.py を実行してください。")
        return

    # --- 2. ユーザーごとに並列処理で分割・保存 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_users = df_posts['username'].unique()
    
    print(f"見つかったユニークユーザー数: {len(all_users)}人")
    num_processes = os.cpu_count()
    print(f"ユーザー処理のために {num_processes} 個のプロセスを開始します...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        # tqdmを使って進捗を表示
        results = list(tqdm(executor.map(save_user_file, all_users), total=len(all_users), desc="ユーザーごとにファイルを保存中"))

    print("\n--- 処理結果（一部抜粋） ---")
    # 結果が多すぎる場合があるので,最初と最後の数件だけ表示
    for res in results[:5]:
        print(res)
    if len(results) > 10:
        print("...")
    for res in results[-5:]:
        print(res)

if __name__ == '__main__':
    multiprocessing.freeze_support() # Windows/macOSで安全に実行するために必要
    start_time = time.time()
    main()
    end_time = time.time()
    print("\n--- 全てのファイルの集計が完了しました ---")
    print(f"'{OUTPUT_DIR}' ディレクトリに {len(os.listdir(OUTPUT_DIR))} 個のファイルが作成されました。")
    print(f"合計処理時間: {end_time - start_time:.2f} 秒")