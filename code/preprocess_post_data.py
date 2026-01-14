import pandas as pd
import os
import json
import re
import emoji  # pip install emoji
from datetime import datetime
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import multiprocessing
import numpy as np

# --- 設定 ---
INFO_DIR = '../influencer_info_v2/info/'
OUTPUT_FILE = 'preprocessed_posts_detailed.csv'

# 分析器の初期化（各プロセスで初期化するためグローバルでは定義のみ）
analyzer = None

def init_worker():
    """マルチプロセス用ワーカーの初期化"""
    global analyzer
    analyzer = SentimentIntensityAnalyzer()

def extract_text_features(text):
    """テキストから特徴量を抽出する"""
    if not text:
        return {
            'char_length': 0,
            'hashtag_count': 0,
            'mention_count': 0,
            'emoji_count': 0,
            'sentiment_pos': 0.0,
            'sentiment_neg': 0.0,
            'sentiment_neu': 0.0,
            'sentiment_compound': 0.0
        }
    
    # VADER感情分析
    vs = analyzer.polarity_scores(text)
    
    return {
        'char_length': len(text),
        'hashtag_count': len(re.findall(r'#\w+', text)),
        'mention_count': len(re.findall(r'@\w+', text)),
        'emoji_count': emoji.emoji_count(text),
        'sentiment_pos': vs['pos'],
        'sentiment_neg': vs['neg'],
        'sentiment_neu': vs['neu'],
        'sentiment_compound': vs['compound']
    }

def get_comment_sentiment(comments_data):
    """コメントデータのリストから感情スコアの平均を算出する"""
    if not comments_data:
        return {'comment_pos': 0, 'comment_neg': 0, 'comment_neu': 0, 'comment_compound': 0}

    scores = {'pos': [], 'neg': [], 'neu': [], 'compound': []}
    
    # コメントデータ構造の解析（Instaloaderの形式に準拠）
    # edges -> node -> text
    edges = comments_data.get('edges', [])
    if not edges:
        return {'comment_pos': 0, 'comment_neg': 0, 'comment_neu': 0, 'comment_compound': 0}

    for edge in edges:
        text = edge.get('node', {}).get('text', '')
        if text:
            vs = analyzer.polarity_scores(text)
            for k in scores:
                scores[k].append(vs[k])
    
    # 平均値を返す（コメントがない場合は0）
    if not scores['pos']:
        return {'comment_pos': 0, 'comment_neg': 0, 'comment_neu': 0, 'comment_compound': 0}

    return {f'comment_{k}': np.mean(v) for k, v in scores.items()}

def process_file(filename):
    """単一ファイルの処理"""
    try:
        filepath = os.path.join(INFO_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # --- 0. ID & Basic Info ---
        # ファイル名からIDを取得（拡張子除去）
        post_id = os.path.splitext(filename)[0]
        owner = data.get('owner', {})
        
        username = owner.get('username', '')
        owner_id = owner.get('id', '')
        
        # キャプション取得
        caption_edges = data.get('edge_media_to_caption', {}).get('edges', [])
        caption = caption_edges[0]['node']['text'] if caption_edges else ""

        # --- 2. Profile Features (Raw) ---
        # 統計値算出のために生の値を取得
        followers = data.get('edge_followed_by', {}).get('count', 0)  # JSON構造によってはowner内にある場合あり
        if followers == 0 and 'edge_followed_by' in owner:
             followers = owner['edge_followed_by']['count']
             
        following = data.get('edge_follow', {}).get('count', 0)
        if following == 0 and 'edge_follow' in owner:
            following = owner['edge_follow']['count']
            
        media_count = owner.get('edge_owner_to_timeline_media', {}).get('count', 0)
        
        # カテゴリ（ビジネスアカウントなどのカテゴリ名）
        # "8つのカテゴリ"へのマッピングは後処理で行うため,まずは生データを取得
        category_name = data.get('category_name') or owner.get('category_name', 'Unknown')
        
        # --- 4. Text Features (Raw) ---
        text_feats = extract_text_features(caption)
        
        # --- 5. Posting Features (Raw) ---
        timestamp = data.get('taken_at_timestamp', 0)
        dt = datetime.fromtimestamp(timestamp)
        
        # 広告判定 (簡易ロジック: PRタグやis_adフラグ)
        is_ad = data.get('is_ad', False)
        if not is_ad and ('#pr' in caption.lower() or '#ad' in caption.lower() or 'タイアップ' in caption):
            is_ad = True
            
        # 投稿カテゴリ（後で分類するためにテキスト情報を保持,あるいはここで簡易分類）
        # ここでは以前のロジックを踏襲しつつフラグ化
        is_informative = 1 if re.search(r'レビュー|review|比較|レポ|howto|解説|まとめ|swatch', caption, re.IGNORECASE) else 0
        
        # フィードバック率計算用（コメント返信など）
        # ※詳細な返信率はコメントデータ全量が必要だが,ここでは簡易的にコメント機能が有効かを保持
        comments_disabled = data.get('comments_disabled', False)
        
        # --- 6. Reaction Features (Raw) ---
        likes = data.get('edge_media_preview_like', {}).get('count', 0)
        comment_count = data.get('edge_media_to_parent_comment', {}).get('count', 0)
        
        # コメントの感情分析（JSON内にコメントテキストがある場合のみ有効）
        comment_data = data.get('edge_media_to_parent_comment', {})
        comment_sentiment_feats = get_comment_sentiment(comment_data)

        # --- 結果の統合 ---
        row = {
            'post_id': post_id,
            'username': username,
            'owner_id': owner_id,
            'user_category_raw': category_name, # 後でOne-hot化
            'datetime': dt,
            'timestamp': timestamp,
            
            # Profile Stats (Raw)
            'user_followers': followers,
            'user_following': following,
            'user_media_count': media_count,
            
            # Text Stats (Raw)
            'caption': caption, # 確認用
            'caption_len': text_feats['char_length'],
            'hashtag_count': text_feats['hashtag_count'],
            'mention_count': text_feats['mention_count'],
            'emoji_count': text_feats['emoji_count'],
            
            # Text Sentiment (Raw)
            'sentiment_pos': text_feats['sentiment_pos'],
            'sentiment_neg': text_feats['sentiment_neg'],
            'sentiment_neu': text_feats['sentiment_neu'],
            'sentiment_compound': text_feats['sentiment_compound'],
            
            # Posting Stats (Raw)
            'is_ad': 1 if is_ad else 0,
            'is_informative': is_informative,
            'comments_disabled': 1 if comments_disabled else 0,
            
            # Reaction Stats (Raw)
            'like_count': likes,
            'comment_count': comment_count,
            'comment_sentiment_pos': comment_sentiment_feats['comment_pos'],
            'comment_sentiment_neg': comment_sentiment_feats['comment_neg'],
            'comment_sentiment_neu': comment_sentiment_feats['comment_neu'],
            'comment_sentiment_compound': comment_sentiment_feats['comment_compound'],
        }
        
        return row

    except Exception as e:
        # デバッグ時は以下を有効化
        # print(f"Error processing {filename}: {e}")
        return None

def main():
    if not os.path.exists(INFO_DIR):
        print(f"Error: Directory '{INFO_DIR}' not found.")
        return

    all_files = [f for f in os.listdir(INFO_DIR) if f.endswith('.info') or f.endswith('.json')]
    print(f"Found {len(all_files)} files to process.")
    
    # 並列処理設定
    num_processes = max(1, multiprocessing.cpu_count() - 1) # 1コア余力を残す
    print(f"Starting extraction with {num_processes} cores...")

    # pool作成時に init_worker を呼んで VADER を各プロセスで初期化
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, all_files), total=len(all_files)))

    # Noneを除外
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(valid_results)
    
    # datetimeでソート（時系列計算の準備のため）
    df = df.sort_values(['username', 'datetime'])
    
    # CSV出力
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved detailed data to '{OUTPUT_FILE}' with {len(df)} records.")
    print("Columns:", list(df.columns))

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()