import os
import json
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import emoji
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import re
from datetime import datetime

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NLTKのVADER辞書をダウンロード（初回のみ）
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    logging.info("VADER lexicon not found. Downloading...")
    nltk.download('vader_lexicon', quiet=True)
    logging.info("VADER lexicon downloaded.")

# === 1. 投稿カテゴリ分析の定義 (★ Ryomaさんによる編集が必要) ===

# カテゴリの優先順位リスト
# 複数のキーワードが一致した場合,このリストの順序が優先されます。
# （例：'food' と 'travel' が両方あれば,'food' が優先される）
CATEGORY_PRIORITY = [
    'food', 
    'travel', 
    'pet', 
    'fitness', 
    'family', 
    'interior', 
    'fashion', 
    'beauty', 
    'other'
]

# 投稿カテゴリ判定用のキーワード辞書
# ★★★ このキーワードリストを充実させてください ★★★
POST_CATEGORY_KEYWORDS = {
    'food': ['recipe', 'food', 'yummy', 'delicious', 'restaurant', 'eat', 'cooking', 'cafe', 
             '料理', 'グルメ', 'カフェ', '美味しい', 'レストラン', '食べ', 'ごはん'],
    'travel': ['travel', 'trip', 'vacation', 'adventure', 'explore', 'passport', 'airport', 
               '旅行', '観光', '空港', '旅', '海外', '国内'],
    'pet': ['pet', 'dog', 'cat', 'puppy', 'kitten', 'animal', 
            'ペット', '犬', '猫', 'ワンコ', 'ニャンコ', '愛犬', '愛猫'],
    'fitness': ['fitness', 'gym', 'workout', 'muscle', 'training', 'fit', 'health', 
                'フィットネス', 'ジム', '筋トレ', 'ワークアウト', 'トレーニング', '筋肉'],
    'family': ['family', 'kids', 'baby', 'son', 'daughter', 'husband', 'wife', 
               '家族', '子供', '赤ちゃん', '息子', '娘', '夫', '妻', '育児'],
    'interior': ['interior', 'design', 'home', 'decor', 'furniture', 'room', 
                 'インテリア', '家具', '部屋', '内装', 'デザイン', '雑貨'],
    'fashion': ['fashion', 'style', 'ootd', 'outfit', 'clothes', 'shopping', 'dress', 
                'ファッション', 'コーデ', '服', 'スタイル', '今日のコーデ', '洋服', '買い物'],
    'beauty': ['beauty', 'makeup', 'skincare', 'hair', 'cosmetics', 'lipstick', 'nail', 
               '美容', 'メイク', 'コスメ', 'スキンケア', 'ネイル', 'ヘア', '化粧品'],
    'other': ['art', 'music', 'book', 'movie', 'hobby', 'diy', 
              'アート', '音楽', '映画', '本', '趣味', '読書'],
}

# フォルダ名（インフルエンサーカテゴリ）を正規化する辞書
CATEGORY_NORMALIZATION_MAP = {
    'beauty': 'beauty',
    'family': 'family',
    'fashion': 'fashion',
    'fashion 0.5': 'fashion', # 'ls'で表示されたもの
    'fasion': 'fashion',      # 'ls'で表示されたもの
    'fitness': 'fitness',
    'food': 'food',
    'interior': 'interior',
    'other': 'other',
    'pet': 'pet',
    'travel': 'travel',
    '_unknown_category': '_unknown_category'
}

# --- 2. ヘルパー関数 ---

def normalize_influencer_category(category_name):
    """フォルダ名を正規のカテゴリ名に変換する"""
    return CATEGORY_NORMALIZATION_MAP.get(category_name, '_unknown_category')

def determine_post_category(caption, normalized_influencer_category):
    """キャプションのキーワードに基づいて投稿カテゴリを決定する"""
    if not caption:
        return normalized_influencer_category # キャプションがなければインフルエンサーカテゴリ

    lower_caption = caption.lower()
    
    # 優先順位リストに基づいてキーワードをチェック
    for category in CATEGORY_PRIORITY:
        keywords = POST_CATEGORY_KEYWORDS.get(category, [])
        for keyword in keywords:
            # 正規表現で単語として一致するかをチェック（例：'food' が 'foodie' に誤爆しないよう）
            # ただし,日本語も考慮し,単純な 'in' 演算子の方が広範囲を拾える
            # ここでは単純な 'in' を使います（精度を上げるなら要調整）
            if keyword in lower_caption:
                # キーワードが見つかったら,そのカテゴリを返す
                return category
    
    # どのキーワードも見つからなければ,インフルエンサーの正規化カテゴリを返す
    return normalized_influencer_category

def count_emojis(text):
    """テキスト内の絵文字数をカウントする"""
    if not text:
        return 0
    return emoji.emoji_count(text)

def calculate_comment_stats(comments_data, post_owner_username, vader_analyzer):
    """
    1つの投稿に含まれるコメントを分析する
    - オーナーによる返信の有無 (has_replied)
    - ユーザーコメントの感情スコアの平均
    """
    has_replied = False
    user_comment_scores = {
        'pos': [], 'neg': [], 'neu': [], 'compound': []
    }
    
    for comment_node in comments_data:
        try:
            comment_owner = comment_node['node']['owner']['username']
            comment_text = comment_node['node']['text']
            
            if comment_owner == post_owner_username:
                has_replied = True
            else:
                # ユーザー（オーナー以外）のコメントの感情を分析
                scores = vader_analyzer.polarity_scores(comment_text)
                user_comment_scores['pos'].append(scores['pos'])
                user_comment_scores['neg'].append(scores['neg'])
                user_comment_scores['neu'].append(scores['neu'])
                user_comment_scores['compound'].append(scores['compound'])
        except (KeyError, TypeError):
            continue # コメントデータが不正な場合はスキップ

    # ユーザーコメントの感情スコアの平均を計算
    mean_scores = {}
    for key, values in user_comment_scores.items():
        if values:
            mean_scores[f'comment_vader_{key}_mean'] = np.mean(values)
        else:
            mean_scores[f'comment_vader_{key}_mean'] = np.nan
            
    return has_replied, mean_scores


def process_post_file(file_path_tuple):
    """
    単一の.infoファイルを処理し,1行分のデータを辞書として返す
    """
    file_path, influencer_category_raw, vader_analyzer = file_path_tuple
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        # logging.warning(f"Skipping file (corrupt JSON?): {file_path} - {e}")
        return None

    try:
        # --- 必須情報 ---
        post_id = data['id']
        username = data['owner']['username']
        timestamp = data['taken_at_timestamp']
        # タイムスタンプを人間が読める形式にも変換
        datetime_utc = datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%dT%H:%M:%SZ')

        # --- 5. Posting ---
        is_ad = data.get('is_ad', False)

        # --- 4. Text (Caption) ---
        caption = ""
        if data.get('edge_media_to_caption', {}).get('edges'):
            try:
                caption = data['edge_media_to_caption']['edges'][0]['node']['text']
            except (IndexError, KeyError, TypeError):
                caption = "" # Handle empty/malformed caption
        
        caption_length = len(caption)
        hashtag_count = caption.count('#')
        mention_count = caption.count('@')
        emoji_count = count_emojis(caption)
        
        # キャプションの感情分析
        if caption_length > 0:
            caption_vader_scores = vader_analyzer.polarity_scores(caption)
        else:
            caption_vader_scores = {'pos': np.nan, 'neg': np.nan, 'neu': np.nan, 'compound': np.nan}

        # --- 5. Posting (Category Analysis) ★新機能★ ---
        # フォルダ名を正規化
        normalized_influencer_cat = normalize_influencer_category(influencer_category_raw)
        
        # 投稿カテゴリをキーワードで分析
        analyzed_post_category = determine_post_category(caption, normalized_influencer_cat)

        # --- 5. Posting (Feedback) & 6. Reaction (Comments) ---
        comments_data = data.get('edge_media_to_parent_comment', {}).get('edges', [])
        has_replied, comment_mean_vader_scores = calculate_comment_stats(
            comments_data, 
            username, 
            vader_analyzer
        )
        
        # 1行のデータを作成
        row = {
            'username': username,
            'post_id': post_id,
            'timestamp': timestamp,
            'datetime_utc': datetime_utc,
            'influencer_category': normalized_influencer_cat, # インフルエンサーのカテゴリ
            'analyzed_post_category': analyzed_post_category, # ★分析された投稿カテゴリ
            'is_ad_post': is_ad,
            'has_owner_replied': has_replied,
            'caption_length': caption_length,
            'hashtag_count': hashtag_count,
            'mention_count': mention_count,
            'emoji_count': emoji_count,
            'caption_vader_pos': caption_vader_scores['pos'],
            'caption_vader_neg': caption_vader_scores['neg'],
            'caption_vader_neu': caption_vader_scores['neu'],
            'caption_vader_comp': caption_vader_scores['compound'],
            **comment_mean_vader_scores # コメントの平均VADERスコアを展開して追加
        }
        return row
        
    except KeyError as e:
        # logging.warning(f"Skipping file (missing key {e}): {file_path}")
        return None
    except Exception as e:
        # logging.error(f"Unexpected error processing file {file_path}: {e}")
        return None

# --- 3. メイン実行関数 ---
def main():
    base_dir = 'organized_posts'
    output_file = 'instagram_post_features_analyzed.csv' # 出力ファイル名を変更
    
    if not os.path.isdir(base_dir):
        logging.error(f"エラー: ディレクトリ '{base_dir}' が見つかりません。")
        logging.error(f"スクリプトは 'organized_posts' フォルダと同じ階層で実行してください。")
        return

    # VADERをメインプロセスで一度だけ初期化
    vader = SentimentIntensityAnalyzer()

    # 1. 処理対象のファイルリストを作成
    tasks = []
    logging.info("スキャン中: 'organized_posts' ディレクトリ...")
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        for username in os.listdir(category_path):
            username_path = os.path.join(category_path, username)
            if not os.path.isdir(username_path):
                continue
                
            for fname in os.listdir(username_path):
                if fname.endswith('.info'):
                    file_path = os.path.join(username_path, fname)
                    # (ファイルパス, フォルダカテゴリ名, VADERインスタンス) をタプルで渡す
                    tasks.append((file_path, category, vader))

    if not tasks:
        logging.error("エラー: .info ファイルが1つも見つかりませんでした。")
        return

    logging.info(f"合計 {len(tasks)} 件の投稿ファイルを検出しました。並列処理を開始します...")

    # 2. 並列処理
    num_workers = max(1, cpu_count() - 1)
    results = []

    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="投稿データを処理中", unit="file") as pbar:
            for result in pool.imap_unordered(process_post_file, tasks):
                if result:
                    results.append(result)
                pbar.update()

    logging.info("すべての処理が完了しました。")

    if not results:
        logging.error("エラー: 処理に成功したデータがありませんでした。")
        return

    # 3. DataFrameに変換してCSVに保存
    logging.info(f"DataFrameを作成中 (合計 {len(results)} 行)...")
    df = pd.DataFrame(results)

    df = df.sort_values(by=['username', 'timestamp'])
    
    # カラムの順序を整理
    columns_order = [
        'username', 'post_id', 'timestamp', 'datetime_utc', 
        'influencer_category', 'analyzed_post_category', # 2つのカテゴリ列
        'is_ad_post', 'has_owner_replied',
        'caption_length', 'hashtag_count', 'mention_count', 'emoji_count',
        'caption_vader_pos', 'caption_vader_neg', 'caption_vader_neu', 'caption_vader_comp',
        'comment_vader_pos_mean', 'comment_vader_neg_mean', 'comment_vader_neu_mean', 'comment_vader_comp_mean'
    ]
    
    final_columns = [col for col in columns_order if col in df.columns]
    final_columns.extend([col for col in df.columns if col not in final_columns])
    df = df[final_columns]

    try:
        # 新しいファイル名で保存
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"✅ 成功: データを '{output_file}' に保存しました。")
        
        logging.info("\n--- データプレビュー (先頭5行) ---")
        print(df.head().to_string())
        logging.info("----------------------------------")

    except Exception as e:
        logging.error(f"エラー: CSVファイルへの保存に失敗しました。- {e}")


if __name__ == "__main__":
    main()
