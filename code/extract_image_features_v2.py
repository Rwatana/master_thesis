import os
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing

# --- 設定: H100 x 2 推奨設定 ---
IMAGE_ROOT = os.path.expanduser('~/graph_dataset/instagram_influencer/organized_images')
OUTPUT_FILE = 'image_features_v2_full.csv'

# H100 (80GB VRAM) 用に大きく設定
BATCH_SIZE = 512  
# CPU 96コアを有効活用 (前処理負荷が高いため多めに)
NUM_WORKERS = 24  

# メモリ保護のため、この枚数ごとにCSVへ書き出してメモリを解放する
WRITE_CHUNK_SIZE = 5000 

# --- デバイス設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- ヘルパー関数: 処理済みチェック ---
def get_processed_post_ids(csv_path):
    """中断しても続きから再開できるように、既にCSVにあるIDを取得する"""
    if not os.path.exists(csv_path):
        return set()
    try:
        # post_id列のみ読み込む (高速化)
        df = pd.read_csv(csv_path, usecols=['post_id'], dtype={'post_id': str})
        return set(df['post_id'].values)
    except Exception:
        return set()

# --- ヘルパー関数: 画像統計量 (CPU処理) ---
def calculate_pixel_stats(image_pil):
    """
    画像の輝度・色彩度・色温度を計算する。
    Datasetの__getitem__内で呼ばれ、マルチプロセス(num_workers)で並列実行される。
    """
    # 1. 輝度 (Brightness)
    gray_img = image_pil.convert('L')
    stat = ImageStat.Stat(gray_img)
    brightness = stat.mean[0]

    # 計算用にnumpy配列化
    img_np = np.array(image_pil).astype(float)
    if len(img_np.shape) == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    
    R, G, B = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]

    # 2. 色彩度 (Colorfulness) - Hasler and Suesstrunk
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    colorfulness = std_root + 0.3 * mean_root

    # 3. 色温度 (Color Temperature) - McCamy's Formula近似
    mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
    
    # CIE 1931 XYZ変換
    X = 0.4124 * mean_R + 0.3576 * mean_G + 0.1805 * mean_B
    Y = 0.2126 * mean_R + 0.7152 * mean_G + 0.0722 * mean_B
    Z = 0.0193 * mean_R + 0.1192 * mean_G + 0.9505 * mean_B
    
    if (X + Y + Z) == 0:
        cct = 0
    else:
        x = X / (X + Y + Z)
        y = Y / (X + Y + Z)
        denom = (0.1858 - y)
        if denom == 0:
             cct = 0
        else:
            n = (x - 0.3320) / denom
            cct = -449 * (n**3) + 3525 * (n**2) - 6823.3 * n + 5520.33

    return {
        'brightness': brightness,
        'colorfulness': colorfulness,
        'color_temp': cct
    }

# --- データセット定義 ---
class InstagramImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, processed_ids=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        print("Scanning directory structure...")
        skip_count = 0
        
        for category in os.listdir(root_dir):
            cat_path = os.path.join(root_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'): continue
            
            for username in os.listdir(cat_path):
                user_path = os.path.join(cat_path, username)
                if not os.path.isdir(user_path) or username.startswith('.'): continue
                
                for filename in os.listdir(user_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        post_id = self._extract_post_id(filename)
                        
                        # 既にCSVにあるIDならスキップ
                        if processed_ids and post_id in processed_ids:
                            skip_count += 1
                            continue

                        self.image_files.append({
                            'path': os.path.join(user_path, filename),
                            'category': category,
                            'username': username,
                            'post_id': post_id,
                            'filename': filename
                        })
        
        print(f"Found {len(self.image_files) + skip_count} total images.")
        if skip_count > 0:
            print(f"Skipping {skip_count} processed images. Remaining: {len(self.image_files)}")

    def _extract_post_id(self, filename):
        base = os.path.splitext(filename)[0]
        match = re.search(r'-(\d+)$', base)
        return match.group(1) if match else base

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        info = self.image_files[idx].copy()
        image_path = info['path']
        
        try:
            image = Image.open(image_path).convert('RGB')
            
            # --- ピクセル統計量の計算 (ここで計算することでCPU並列化される) ---
            stats = calculate_pixel_stats(image)
            info.update(stats)
            
            # ResNet用の前処理
            if self.transform:
                image_tensor = self.transform(image)
                
            return image_tensor, info
        except Exception:
            # 読み込みエラー時はNoneを返す
            return None, info

def collate_fn(batch):
    """None(エラー画像)を除外してバッチを作成"""
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None, None
    images = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return images, metadata

def flush_results_to_csv(results_buffer, output_file):
    """バッファをCSVに追記し、メモリを解放する"""
    if not results_buffer:
        return []
    
    df = pd.DataFrame(results_buffer)
    
    # Post IDを文字列化 (指数表記防止)
    df['post_id'] = df['post_id'].astype(str)
    
    # ファイルが存在しない場合のみヘッダーを書く
    header = not os.path.exists(output_file)
    
    # 追記モード
    df.to_csv(output_file, mode='a', index=False, header=header)
    
    # メモリ解放
    del df
    return []

# --- メイン処理 ---
def main():
    print(f"Using Device: {device}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    # 1. Resume用チェック
    processed_ids = get_processed_post_ids(OUTPUT_FILE)
    
    # 2. モデル準備
    print("Loading ResNet50...")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    transform = weights.transforms()
    class_names = weights.meta["categories"]
    
    # ★ マルチGPU化 (H100 x 2) ★
    if torch.cuda.device_count() > 1:
        print(f"Activating DataParallel for {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model)
    
    model = model.to(device)
    model.eval()

    # 3. データローダー準備
    dataset = InstagramImageDataset(IMAGE_ROOT, transform=transform, processed_ids=processed_ids)
    if len(dataset) == 0:
        print("No new images to process.")
        return

    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=True # 高速化
    )

    # 4. ループ実行
    results_buffer = []
    print(f"Starting extraction (Batch: {BATCH_SIZE}, Workers: {NUM_WORKERS})...")
    
    with torch.no_grad():
        for images, metadata_list in tqdm(dataloader, desc="Extracting"):
            if images is None: continue
            
            images = images.to(device)
            
            # --- 推論 (GPU) ---
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            # 上位1位のクラスを取得
            top_probs, top_idxs = torch.max(probs, dim=1)
            
            # CPUへ戻す
            top_probs = top_probs.cpu().numpy()
            top_idxs = top_idxs.cpu().numpy()
            
            # --- 結果格納 ---
            for i, meta in enumerate(metadata_list):
                row = {
                    'post_id': str(meta['post_id']),
                    'username': meta['username'],
                    'image_category': meta['category'],
                    
                    # ImageNet認識結果
                    'detected_object': class_names[top_idxs[i]],
                    'detection_confidence': top_probs[i],
                    
                    # ピクセル統計量 (Dataset内で計算済み)
                    'brightness': meta['brightness'],
                    'colorfulness': meta['colorfulness'],
                    'color_temp': meta['color_temp'],
                    
                    'filename': meta['filename']
                }
                results_buffer.append(row)
            
            # --- チャンク書き込み (メモリ溢れ防止) ---
            if len(results_buffer) >= WRITE_CHUNK_SIZE:
                results_buffer = flush_results_to_csv(results_buffer, OUTPUT_FILE)
    
    # 残りを書き込む
    if results_buffer:
        flush_results_to_csv(results_buffer, OUTPUT_FILE)
    
    print(f"\nDone! All data saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
