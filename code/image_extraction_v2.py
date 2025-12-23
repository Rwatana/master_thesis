import os
import re
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageFile
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing

# 画像読み込み時のTruncatedエラーを許容
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 設定: Apple M1 Ultra (128GB RAM) 推奨設定 ---
IMAGE_ROOT = os.path.expanduser('../image')
OUTPUT_FILE = 'image_features_v2_full_fixed_v2.csv'

# ★ M1 Ultraのメモリ(128GB)を活かして巨大バッチにする
# GPU使用率を100%に張り付かせるための設定
BATCH_SIZE = 1024 

# ★ M1 Ultraの高性能コア数(16)に合わせる
# これ以上増やすとコンテキストスイッチのオーバーヘッドで逆に遅くなる可能性があります
NUM_WORKERS = 20

# CSV書き込み頻度を下げる (メモリに余裕があるため)
WRITE_CHUNK_SIZE = 20000

# --- デバイス設定 (Apple Silicon Native) ---
# CUDAではなくMPS (Metal Performance Shaders) を使用
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available. Using CPU (Will be slow)")

# --- ヘルパー関数: 処理済みチェック ---
def get_processed_post_ids(csv_path):
    if not os.path.exists(csv_path):
        return set()
    try:
        print(f"Loading existing IDs from {csv_path} for resume...")
        df = pd.read_csv(csv_path, usecols=['post_id'], dtype={'post_id': str})
        existing_ids = set(df['post_id'].values)
        print(f"-> Found {len(existing_ids)} processed images.")
        return existing_ids
    except Exception as e:
        print(f"Warning: Could not read existing file. Starting fresh. ({e})")
        return set()

# --- ヘルパー関数: 画像統計量 (CPU処理) ---
def calculate_pixel_stats(image_pil):
    """
    CPUパワーを使って計算。M1 Ultraのシングルスレッド性能が高いので高速です。
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

    # 2. 色彩度 (Colorfulness)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(np.std(rg)**2 + np.std(yb)**2)
    mean_root = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    colorfulness = std_root + 0.3 * mean_root

    # 3. 色温度 (Color Temperature)
    mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
    
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
        
        # ディレクトリ走査
        for category in os.listdir(root_dir):
            cat_path = os.path.join(root_dir, category)
            if not os.path.isdir(cat_path) or category.startswith('.'): continue
            
            for username in os.listdir(cat_path):
                user_path = os.path.join(cat_path, username)
                if not os.path.isdir(user_path) or username.startswith('.'): continue
                
                for filename in os.listdir(user_path):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        post_id = self._extract_post_id(filename)
                        
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
        if '-' in base:
            return base.split('-')[-1]
        return base

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        info = self.image_files[idx].copy()
        
        try:
            image = Image.open(info['path']).convert('RGB')
            
            # CPU並列処理 (M1 Ultraの20コアを活用)
            stats = calculate_pixel_stats(image)
            info.update(stats)
            
            if self.transform:
                image_tensor = self.transform(image)
                
            return image_tensor, info
        except Exception:
            return None, info

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if not batch: return None, None
    images = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]
    return images, metadata

def flush_results_to_csv(results_buffer, output_file):
    if not results_buffer: return []
    df = pd.DataFrame(results_buffer)
    df['post_id'] = df['post_id'].astype(str)
    header = not os.path.exists(output_file)
    df.to_csv(output_file, mode='a', index=False, header=header)
    del df
    return []

# --- メイン処理 ---
def main():
    # 1. Resume用チェック
    processed_ids = get_processed_post_ids(OUTPUT_FILE)
    
    # 2. モデル準備
    print("Loading ResNet50 for Metal (MPS)...")
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    transform = weights.transforms()
    class_names = weights.meta["categories"]
    
    # M1 Ultraは単体GPUとして認識されるためDataParallelは不要
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
        pin_memory=True,     # MPSでも有効
        persistent_workers=True, # ★重要: macOSでの再spawnオーバーヘッドを防ぐ
        prefetch_factor=2    # ★重要: 次のバッチを常に準備させてCPUを休ませない
    )

    # 4. ループ実行
    results_buffer = []
    print(f"🚀 Starting extraction on M1 Ultra (Batch: {BATCH_SIZE}, Workers: {NUM_WORKERS})...")
    
    with torch.no_grad():
        for images, metadata_list in tqdm(dataloader, desc="Extracting"):
            if images is None: continue
            
            images = images.to(device)
            
            # --- 推論 (MPS) ---
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_idxs = torch.max(probs, dim=1)
            
            # CPUへ戻す (ここがUnified Memoryで高速)
            top_probs = top_probs.cpu().numpy()
            top_idxs = top_idxs.cpu().numpy()
            
            # --- 結果格納 ---
            for i, meta in enumerate(metadata_list):
                row = {
                    'post_id': str(meta['post_id']),
                    'username': meta['username'],
                    'image_category': meta['category'],
                    'detected_object': class_names[top_idxs[i]],
                    'detection_confidence': top_probs[i],
                    'brightness': meta['brightness'],
                    'colorfulness': meta['colorfulness'],
                    'color_temp': meta['color_temp'],
                    'filename': meta['filename']
                }
                results_buffer.append(row)
            
            # --- チャンク書き込み ---
            if len(results_buffer) >= WRITE_CHUNK_SIZE:
                results_buffer = flush_results_to_csv(results_buffer, OUTPUT_FILE)
    
    if results_buffer:
        flush_results_to_csv(results_buffer, OUTPUT_FILE)
    
    print(f"\nDone! All data saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()