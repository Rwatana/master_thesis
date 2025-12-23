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
# VRAM 12GB程度なら 8 or 16
# VRAM 24GB程度なら 32 or 64
BATCH_SIZE = 512

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

def process_results(batch_results, task_info_batch):
    """
    パイプラインからのバッチ結果を処理してCSVの行リストに変換する
    """
    rows = []
    for i, results in enumerate(batch_results):
        try:
            # タスク情報（メタデータ）を取得
            file_path, category, username, post_id = task_info_batch[i]
            
            # 結果を信頼度でソート
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            objects = [res['label'] for res in sorted_results]

            row = {
                'username': username,
                'post_id': post_id,
                'user_category': category,
                'first_object': objects[0] if len(objects) > 0 else np.nan,
                'second_object': objects[1] if len(objects) > 1 else np.nan,
                'third_object': objects[2] if len(objects) > 2 else np.nan,
                'detected_object_count': len(objects),
                'all_objects_detected': ','.join(objects)
            }
            rows.append(row)
        except Exception as e:
            logging.warning(f"結果の処理中にエラー: {e} - file: {task_info_batch[i][0]}")
    return rows

# --- メイン実行関数 ---
def main():
    base_dir = 'organized_images'
    output_file = 'image_object_features_gpu.csv'
    
    # --- GPUの確認 ---
    if not torch.cuda.is_available():
        logging.error("="*50)
        logging.error("エラー: GPU (CUDA) が利用できません。")
        logging.error("このスクリプトはGPU専用です。")
        logging.error("CPUで実行する場合は `create_image_features_cpu.py` を使用してください。")
        logging.error("="*50)
        return
        
    logging.info(f"GPU ({torch.cuda.get_device_name(0)}) が利用可能です。")

    if not os.path.isdir(base_dir):
        logging.error(f"エラー: ディレクトリ '{base_dir}' が見つかりません。")
        return

    # 1. 処理対象のファイルリストを作成
    #    今回はファイルパスだけでなく、メタデータも一緒にスキャン
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
                    # ファイル名解析も先に行う
                    username, post_id = parse_filename(file_path)
                    if username and post_id:
                        # (ファイルパス, カテゴリ, ユーザー名, postID)
                        tasks.append((file_path, category, username, post_id))

    if not tasks:
        logging.error("エラー: .jpg ファイルが1つも見つかりませんでした。")
        return

    logging.info(f"合計 {len(tasks)} 件の画像ファイルを検出しました。")

    # 2. モデルをGPUにロード (一度だけ)
    logging.info("モデル (facebook/detr-resnet-50) をGPUにロード中...")
    try:
        # device=0 で 1番目のGPUを指定
        detector = pipeline(
            "object-detection", 
            model="facebook/detr-resnet-50", 
            device=1
        )
        logging.info("モデルのロード完了。")
    except Exception as e:
        logging.error(f"モデルのロードに失敗: {e}")
        return

    # 3. バッチ処理の実行
    logging.info(f"GPUバッチ処理を開始します (Batch Size = {BATCH_SIZE})...")
    all_results_rows = []
    
    # tqdmで進捗を表示
    with tqdm(total=len(tasks), desc="GPUバッチ処理中", unit="image") as pbar:
        # tasksリストを BATCH_SIZE ごとのかたまり（chunk）に分割して処理
        for i in range(0, len(tasks), BATCH_SIZE):
            # i から i + BATCH_SIZE までのタスク情報
            task_info_batch = tasks[i:i + BATCH_SIZE]
            
            # タスク情報からファイルパスのリストだけを抽出
            image_path_batch = [task[0] for task in task_info_batch]
            
            try:
                # ★ パイプラインにファイルパスの *リスト* を渡す
                # これにより、パイプラインが内部で最適化されたバッチ処理を行う
                batch_results = detector(image_path_batch)
                
                # 結果をCSV行に変換
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
    
    columns_order = [
        'username', 'post_id', 'user_category', 
        'first_object', 'second_object', 'third_object',
        'detected_object_count', 'all_objects_detected'
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