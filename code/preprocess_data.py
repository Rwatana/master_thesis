import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import multiprocessing  # 並列処理ライブラリをインポート

# --- グローバル変数として定義 ---
# 情報系コンテンツを判定するためのキーワード
INFORMATIVE_KEYWORDS = [
    'レビュー', 'review', '比較', 'レポ', 'repo', '使い方', 'howto',
    'コスメ紹介', '解説', 'まとめ', 'スウォッチ', 'swatch', 'tips', '方法'
]
REGEX_PATTERN = '|'.join(INFORMATIVE_KEYWORDS)
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
INFO_DIR = 'posts_info/unzipped_data_7z/info/'

def process_file(filename):
    """
    単一の.infoファイルを処理する関数。
    この関数が各CPUコアで並行して実行される。
    """
    try:
        with open(os.path.join(INFO_DIR, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)

        username = data.get('owner', {}).get('username', '')
        caption_edges = data.get('edge_media_to_caption', {}).get('edges', [])
        caption = caption_edges[0]['node']['text'] if caption_edges else ""
        
        # コンテンツタイプの分類
        content_type = '情報系コンテンツ' if pd.Series(caption).str.contains(REGEX_PATTERN, case=False, na=False).any() else '一般コンテンツ'
        
        # 感情スコアを計算
        sentiment_score = SENTIMENT_ANALYZER.polarity_scores(caption)['compound']

        return {
            'username': username,
            'datetime': datetime.fromtimestamp(data.get('taken_at_timestamp', 0)),
            'likes': data.get('edge_media_preview_like', {}).get('count', 0),
            'comments': data.get('edge_media_to_parent_comment', {}).get('count', 0),
            'caption': caption,
            'tag_count': len(data.get('edge_media_to_tagged_user', {}).get('edges', [])),
            'content_type': content_type,
            'sentiment': sentiment_score
        }
    except (json.JSONDecodeError, KeyError, TypeError):
        # print(f"Warning: Skipping malformed file {filename}")
        return None

def create_aggregated_post_data_parallel():
    """
    並列処理を使って,全ての.infoファイルから集計CSVを作成する。
    """
    output_filename = 'preprocessed_posts_with_metadata_data_check.csv'
    
    try:
        all_files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info')]
        print(f"Found {len(all_files)} .info files to process...")
    except FileNotFoundError:
        print(f"Error: Directory '{INFO_DIR}' not found.")
        return

    # 利用可能なCPUコア数を取得（-1することで少し余裕を持たせることも可能）
    num_processes = multiprocessing.cpu_count()
    print(f"Starting parallel processing with {num_processes} cores...")

    # プロセスプールを作成
    with multiprocessing.Pool(processes=num_processes) as pool:
        # pool.imap_unorderedを使って各ファイルにprocess_file関数を適用
        # tqdmで進捗を表示
        results = list(tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files)))

    # Noneが返された（エラーがあった）結果を除外
    all_post_details = [r for r in results if r is not None]

    if not all_post_details:
        print("No data was processed.")
        return
        
    df_processed = pd.DataFrame(all_post_details)
    df_processed.to_csv(output_filename, index=False)
    
    print(f"\nSuccessfully created '{output_filename}' with {len(df_processed)} records.")


# ▼▼▼ 重要: if __name__ == '__main__': で囲む ▼▼▼
if __name__ == '__main__':
    # WindowsやmacOSで並列処理を安全に実行するために必須
    multiprocessing.freeze_support() 
    create_aggregated_post_data_parallel()
