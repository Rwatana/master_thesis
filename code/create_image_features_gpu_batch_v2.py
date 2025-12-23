import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from transformers import pipeline
import torch # PyTorchを直接インポート

# --- 基本設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
Image.MAX_IMAGE_PIXELS = None

# --- バッチサイズ設定 (GPUメモリに応じて調整) ---
# H100 (80GB VRAM) のため、512から開始。
# OOM (Out of Memory) エラーが出る場合は 256 に、余裕がありそうなら 1024 に調整してください。
BATCH_SIZE = 2048

def parse_filename(file_path):
    """
    ファイルパスから username と post_id を抽出する
    """
    try:
        fname = os.path.basename(file_path)
        fname_no_ext = os.path.splitext(fname)[0]
        dir_username = os.path.basename(os.path.dirname(file_path))
        
        if fname_no_ext.startswith(dir_username + '-'):
            post_id = fname_no_ext[len(dir_username) + 1:]
            return dir_username, post_id
        else:
            return dir_username, fname_no_ext
    except Exception:
        return None, None

# ★ 変更: 'image-classification' の出力形式に合わせて関数を修正
def process_results(batch_results, task_info_batch):
    """
    パイプラインからのバッチ結果を処理してCSVの行リストに変換する
    (image-classification 版)
    
    batch_results は [ [top_k_dict_1, ...], [top_k_dict_2, ...], ... ] という形式
    """
    rows = []
    # 'batch_results' は画像ごとの結果リストのリスト
    # 'image_results' は単一の画像に対する [ {'label': '...', 'score': ...}, ... ] (top_k個)
    for i, image_results in enumerate(batch_results):
        try:
            # タスク情報（メタデータ）を取得
            file_path, category, username, post_id = task_info_batch[i]
            
            # image_results は既にスコアでソートされている
            labels = [res['label'] for res in image_results]
            scores = [res['score'] for res in image_results]

            row = {
                'username': username,
                'post_id': post_id,
                'user_category': category,
                'top1_label': labels[0] if len(labels) > 0 else np.nan,
                'top1_score': scores[0] if len(scores) > 0 else np.nan,
                'top2_label': labels[1] if len(labels) > 1 else np.nan,
                'top2_score': scores[1] if len(scores) > 1 else np.nan,
                'top3_label': labels[2] if len(labels) > 2 else np.nan,
                'top3_score': scores[2] if len(scores) > 2 else np.nan,
                'all_top3_labels': ','.join(labels)
            }
            rows.append(row)
        except Exception as e:
            logging.warning(f"結果の処理中にエラー: {e} - file: {task_info_batch[i][0]}")
    return rows

# --- メイン実行関数 ---
def main():
    base_dir = 'organized_images'
    # ★ 変更: 出力ファイル名を変更
    output_file = 'image_classification_features_gpu.csv'
    
    # --- GPUの確認 ---
    if not torch.cuda.is_available():
        logging.error("="*50)
        logging.error("エラー: GPU (CUDA) が利用できません。")
        logging.error("このスクリプトはGPU専用です。")
        logging.error("="*50)
        return
        
    logging.info(f"GPU ({torch.cuda.get_device_name(0)}) が利用可能です。")

    if not os.path.isdir(base_dir):
        logging.error(f"エラー: ディレクトリ '{base_dir}' が見つかりません。")
        return

    # 1. 処理対象のファイルリストを作成
    tasks = []
    logging.info("スキャン中: 'organized_images' ディレクトリ...")
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if not os.path.isdir(category_path): continue
        
        for username_dir in os.listdir(category_path):
            username_path = os.path.join(category_path, username_dir)
            if not os.path.isdir(username_path): continue
                
            for fname in os.listdir(username_path):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(username_path, fname)
                    username, post_id = parse_filename(file_path)
                    if username and post_id:
                        tasks.append((file_path, category, username, post_id))

    if not tasks:
        logging.error("エラー: .jpg ファイルが1つも見つかりませんでした。")
        return

    logging.info(f"合計 {len(tasks)} 件の画像ファイルを検出しました。")

    # 2. モデルをGPUにロード (一度だけ)
    # ★ 変更: タスクとモデルを変更
    logging.info("モデル (google/vit-base-patch16-224) をGPUにロード中...")
    try:
        # device=1 で 2番目のGPUを指定
        classifier = pipeline(
            "image-classification",                # ★ 変更: タスク
            model="google/vit-base-patch16-224", # ★ 変更: モデル (ImageNet-1k)
            device=1                               # 2台目のGPUを使用
        )
        logging.info("モデルのロード完了。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗: {e}")
        return

    # 3. バッチ処理の実行
    logging.info(f"GPUバッチ処理を開始します (Batch Size = {BATCH_SIZE})...")
    all_results_rows = []
    
    with tqdm(total=len(tasks), desc="GPUバッチ処理中", unit="image") as pbar:
        for i in range(0, len(tasks), BATCH_SIZE):
            task_info_batch = tasks[i:i + BATCH_SIZE]
            image_path_batch = [task[0] for task in task_info_batch]
            
            try:
                # ★ 変更: パイプラインに top_k=3 を渡して上位3件のラベルを取得
                batch_results = classifier(image_path_batch, top_k=3)
                
                rows = process_results(batch_results, task_info_batch)
                all_results_rows.extend(rows)
                
            except Exception as e:
                logging.error(f"バッチ {i//BATCH_SIZE} の処理中にエラー: {e}")
                logging.error(f"エラーが発生したファイル: {image_path_batch}")
            
            pbar.update(len(image_path_batch))

    logging.info("すべてのバッチ処理が完了しました。")

    if not all_results_rows:
        logging.error("エラー: 処理に成功したデータがありませんでした。")
        return

    # 4. DataFrameに変換してCSVに保存
    logging.info(f"DataFrameを作成中 (合計 {len(all_results_rows)} 行)...")
    df = pd.DataFrame(all_results_rows)
    
    # ★ 変更: 出力カラムを分類結果に合わせる
    columns_order = [
        'username', 'post_id', 'user_category', 
        'top1_label', 'top1_score', 
        'top2_label', 'top2_score', 
        'top3_label', 'top3_score', 
        'all_top3_labels'
    ]
    df = df[columns_order]

    try:
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logging.info(f"✅ 成功: データを '{output_file}' に保存しました。")
        
        logging.info("\n--- データプレビュー (先頭5行) ---")
        print(df.head().to_string())
        logging.info("----------------------------------")

    except Exception as e:
        logging.error(f"エラー: CSVファイルへの保存に失敗しました。 - {e}")

if __name__ == "__main__":
    main()
    