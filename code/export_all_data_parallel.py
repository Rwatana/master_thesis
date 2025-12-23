import pandas as pd
import os
import json
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- グローバル変数として定義 ---
INFO_DIRECTORY = '../influencer_info_v2/info/'

def process_file(filename):
    """
    単一の.infoファイルを処理して、ハッシュタグとメンションのリストを返すワーカー関数。
    """
    source_user = filename.split('-')[0]
    hashtags = []
    mentions = []

    file_path = os.path.join(INFO_DIRECTORY, filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            post_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError, UnicodeDecodeError):
        return hashtags, mentions

    timestamp = post_data.get('taken_at_timestamp', 0)
    
    # ハッシュタグを抽出
    caption_edges = post_data.get('edge_media_to_caption', {}).get('edges', [])
    if caption_edges:
        caption = caption_edges[0].get('node', {}).get('text', "")
        found_hashtags = re.findall(r"#(\w+)", caption)
        for tag in found_hashtags:
            hashtags.append({
                'source': source_user,
                'target': f"#{tag}",
                'timestamp': timestamp
            })

    # メンションを抽出
    tagged_users_edges = post_data.get('edge_media_to_tagged_user', {}).get('edges', [])
    for edge in tagged_users_edges:
        tagged_user = edge.get('node', {}).get('user', {}).get('username')
        if tagged_user:
            mentions.append({
                'source': source_user,
                'target': tagged_user,
                'timestamp': timestamp
            })
            
    return hashtags, mentions

if __name__ == "__main__":
    print("--- 全インフルエンサーのデータ抽出を開始します (並列処理) ---")

    # ディレクトリの存在確認
    if not os.path.exists(INFO_DIRECTORY):
        print(f"エラー: ディレクトリ '{INFO_DIRECTORY}' が見つかりません。")
        exit()

    all_files = [f for f in os.listdir(INFO_DIRECTORY) if f.endswith('.info')]
    if not all_files:
        print("対象ファイル (.info) が見つかりません。")
        exit()

    # CPUコア数を取得し、プロセスプールを作成
    num_processes = cpu_count()
    print(f"{num_processes}個のCPUコアを使用して処理します。")

    hashtags_results = []
    mentions_results = []

    # Poolを使って並列処理を実行し、tqdmで進捗を表示
    with Pool(processes=num_processes) as pool:
        # imapを使用することで進捗バーを正確に表示
        results = list(tqdm(pool.imap(process_file, all_files), total=len(all_files), desc="全投稿データを処理中"))

    # 結果をフラットなリストにまとめる
    print("データを集計中...")
    for hashtags, mentions in results:
        hashtags_results.extend(hashtags)
        mentions_results.extend(mentions)

    # DataFrameの作成
    df_hashtags = pd.DataFrame(hashtags_results)
    df_mentions = pd.DataFrame(mentions_results)

    # CSVファイルに保存
    hashtags_csv_path = 'output_hashtags_all_parallel.csv'
    mentions_csv_path = 'output_mentions_all_parallel.csv'
    
    if not df_hashtags.empty:
        df_hashtags.to_csv(hashtags_csv_path, index=False)
        print(f"ハッシュタグデータ ({len(df_hashtags)}件) を '{hashtags_csv_path}' に保存しました。")
    else:
        print("ハッシュタグデータは見つかりませんでした。")

    if not df_mentions.empty:
        df_mentions.to_csv(mentions_csv_path, index=False)
        print(f"メンションデータ ({len(df_mentions)}件) を '{mentions_csv_path}' に保存しました。")
    else:
        print("メンションデータは見つかりませんでした。")
    
    print("\n--- 全処理が完了しました ---")