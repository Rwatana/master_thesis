import pandas as pd
import os

# --- 設定: 入力ファイル名と出力ファイル名 ---
INPUT_POSTS_FILE = 'preprocessed_posts_detailed_v7.csv'
OUTPUT_POSTS_FILE = 'preprocessed_posts_detailed_v7_fixed.csv' # 新しいファイル名

INPUT_IMAGES_FILE = 'image_features_v2_full.csv'
OUTPUT_IMAGES_FILE = 'image_features_v2_full_fixed.csv' # 新しいファイル名

def fix_and_save_new_files():
    print("--- Starting CSV ID Repair (Saving as NEW files) ---")
    
    df_posts = None
    df_images = None

    # 1. Postsファイルの修正
    if os.path.exists(INPUT_POSTS_FILE):
        print(f"\nProcessing {INPUT_POSTS_FILE}...")
        df_posts = pd.read_csv(INPUT_POSTS_FILE, dtype={'post_id': str})
        
        # データの安全のためコピーを作成
        df_posts = df_posts.copy()

        original_sample = df_posts['post_id'].iloc[0] if not df_posts.empty else "N/A"
        
        # 修正ロジック: "username-123" -> "123" (数値部分のみ抽出)
        # 例: 'instagram_user-12345' -> '12345'
        df_posts['post_id'] = df_posts['post_id'].apply(lambda x: str(x).split('-')[-1].split('.')[0])
        
        new_sample = df_posts['post_id'].iloc[0] if not df_posts.empty else "N/A"
        print(f"  [Posts] ID Example: '{original_sample}' -> '{new_sample}'")
        print(f"  [Posts] Total rows: {len(df_posts)}")
        
        # 新しいファイルとして保存
        df_posts.to_csv(OUTPUT_POSTS_FILE, index=False)
        print(f"  Saved new file: {OUTPUT_POSTS_FILE}")
    else:
        print(f"Warning: {INPUT_POSTS_FILE} not found.")

    # 2. Imagesファイルの修正
    if os.path.exists(INPUT_IMAGES_FILE):
        print(f"\nProcessing {INPUT_IMAGES_FILE}...")
        df_images = pd.read_csv(INPUT_IMAGES_FILE, dtype={'post_id': str})
        
        # データの安全のためコピーを作成
        df_images = df_images.copy()

        original_sample = df_images['post_id'].iloc[0] if not df_images.empty else "N/A"
        
        # 修正ロジック: "123.0" -> "123" (小数点を削除)
        # 例: '12345.0' -> '12345'
        df_images['post_id'] = df_images['post_id'].apply(lambda x: str(x).split('.')[0])
        
        new_sample = df_images['post_id'].iloc[0] if not df_images.empty else "N/A"
        print(f"  [Images] ID Example: '{original_sample}' -> '{new_sample}'")
        print(f"  [Images] Total rows: {len(df_images)}")
        
        # 新しいファイルとして保存
        df_images.to_csv(OUTPUT_IMAGES_FILE, index=False)
        print(f"  Saved new file: {OUTPUT_IMAGES_FILE}")
    else:
        print(f"Warning: {INPUT_IMAGES_FILE} not found.")

    # --- 3. 整合性チェック（追加機能） ---
    if df_posts is not None and df_images is not None:
        print("\n--- Intersection Check ---")
        posts_ids = set(df_posts['post_id'])
        images_ids = set(df_images['post_id'])
        
        common_ids = posts_ids.intersection(images_ids)
        print(f"Unique IDs in Posts: {len(posts_ids)}")
        print(f"Unique IDs in Images: {len(images_ids)}")
        print(f"Matching IDs (Ready to Merge): {len(common_ids)}")
        
        if len(common_ids) == 0:
            print("WARNING: No matching IDs found. Please check the ID formats again.")
        else:
            print("SUCCESS: IDs match! You can proceed to merging.")

    print("\n--- Done! ---")
    print("Important: Please update your training script to use the new filenames:")
    print(f"1. {OUTPUT_POSTS_FILE}")
    print(f"2. {OUTPUT_IMAGES_FILE}")

if __name__ == '__main__':
    fix_and_save_new_files()