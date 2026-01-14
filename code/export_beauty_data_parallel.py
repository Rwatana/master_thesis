import pandas as pd
import os
import json
import re
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- グローバル変数として定義 ---
INFO_DIRECTORY = 'posts_info/unzipped_data_7z/info/'

def get_beauty_influencers(filepath):
    """'beauty'カテゴリのインフルエンサーリストを取得する"""
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=[1])
        beauty_users = set(df[df['Category'] == 'beauty']['Username'])
        return beauty_users
    except FileNotFoundError:
        print(f"エラー: '{filepath}' が見つかりません。")
        return None

def process_file(filename):
    """
    単一の.infoファイルを処理して,ハッシュタグとメンションのリストを返すワーカー関数。
    """
    source_user = filename.split('-')[0]
    hashtags = []
    mentions = []
    
    try:
        with open(os.path.join(INFO_DIRECTORY, filename), 'r', encoding='utf-8') as f:
            post_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return hashtags, mentions

    timestamp = post_data.get('taken_at_timestamp', 0)
    
    caption_edges = post_data.get('edge_media_to_caption', {}).get('edges', [])
    caption = caption_edges[0]['node']['text'] if caption_edges else ""
    found_hashtags = re.findall(r"#(\w+)", caption)
    for tag in found_hashtags:
        hashtags.append({
            'source': source_user,
            'target': f"#{tag}",
            'timestamp': timestamp
        })

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
    print("--- 'beauty'カテゴリのインフルエンサーのデータ抽出を開始します (並列処理) ---")
    
    INFLUENCERS_FILE = 'influencers.txt'
    beauty_users = get_beauty_influencers(INFLUENCERS_FILE)
    
    if not beauty_users:
        exit()
        
    print(f"{len(beauty_users)}人のbeautyインフルエンサーを対象とします。")

    try:
        # beautyカテゴリのユーザーのファイルのみを対象にする
        all_files = [f for f in os.listdir(INFO_DIRECTORY) if f.endswith('.info') and f.split('-')[0] in beauty_users]
        if not all_files:
            print("対象ファイルが見つかりません。")
            exit()
    except FileNotFoundError:
        print(f"エラー: ディレクトリ '{INFO_DIRECTORY}' が見つかりません。")
        exit()

    num_processes = cpu_count()
    print(f"{num_processes}個のCPUコアを使用して処理します。")

    hashtags_results = []
    mentions_results = []

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_file, all_files), total=len(all_files), desc="beautyカテゴリの投稿を処理中"))

    for hashtags, mentions in results:
        hashtags_results.extend(hashtags)
        mentions_results.extend(mentions)

    df_hashtags = pd.DataFrame(hashtags_results)
    df_mentions = pd.DataFrame(mentions_results)

    hashtags_csv_path = 'output_hashtags_beauty_parallel.csv'
    mentions_csv_path = 'output_mentions_beauty_parallel.csv'
    
    df_hashtags.to_csv(hashtags_csv_path, index=False)
    df_mentions.to_csv(mentions_csv_path, index=False)
    
    print("\n--- 処理完了 ---")
    print(f"ハッシュタグデータ ({len(df_hashtags)}件) を '{hashtags_csv_path}' に保存しました。")
    print(f"メンションデータ ({len(df_mentions)}件) を '{mentions_csv_path}' に保存しました。")
