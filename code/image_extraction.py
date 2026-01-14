import os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm
import gc

# ==========================================================
# ★ Mac Studio Ultra 向け設定エリア
# ==========================================================

# Macのユニファイドメモリを使用します。
# メモリが64GB/128GBあるなら 512〜1024 くらいまで増やせますが,
# 画面が固まらないようまずは 256 か 512 で試してください。
BATCH_SIZE = 512

# MacのCPU高性能コア数に合わせて設定 (Ultraなら 16〜20 くらいでもOKですが,まずは 8 で安定動作確認)
NUM_WORKERS = 8

# 出力ファイル名
OUTPUT_FILE = 'image_classification_mac.csv'
# 画像フォルダ
BASE_DIR = '../image'

# ログ設定: エラーのみ表示してバーを見やすくする
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers").setLevel(logging.ERROR)
Image.MAX_IMAGE_PIXELS = None

# ==========================================================
# 1. データセット定義 (CPU処理部分)
# ==========================================================
class InfluencerImageDataset(Dataset):
    def __init__(self, task_list, processor):
        self.task_list = task_list
        self.processor = processor

    def __len__(self):
        return len(self.task_list)

    def calculate_stats(self, img_bgr):
        """OpenCVを使用して,輝度・色彩度・色温度を計算"""
        try:
            if img_bgr is None: return 0.0, 0.0, 0.0
            
            # 1. 輝度
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            brightness = float(np.mean(img_hsv[:, :, 2]))
            
            # 2. 色温度傾向
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
            color_temp = float(np.mean(img_lab[:, :, 2]))
            
            # 3. 色彩度
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float")
            R, G, B = cv2.split(img_rgb)
            rg = R - G
            yb = 0.5 * (R + G) - B
            std_root = np.sqrt((np.std(rg) ** 2) + (np.std(yb) ** 2))
            mean_root = np.sqrt((np.mean(rg) ** 2) + (np.mean(yb) ** 2))
            colorfulness = float(std_root + (0.3 * mean_root))
            
            return brightness, colorfulness, color_temp
        except:
            return 0.0, 0.0, 0.0

    def __getitem__(self, idx):
        file_path, category, username, post_id = self.task_list[idx]
        
        try:
            # 画像読み込み
            img_bgr = cv2.imread(file_path)
            if img_bgr is None:
                raise ValueError("Image decode failed")
            
            # 統計量計算
            brightness, colorfulness, color_temp = self.calculate_stats(img_bgr)
            
            # モデル入力用
            img_rgb_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            pixel_values = self.processor(images=img_rgb_pil, return_tensors="pt").pixel_values.squeeze(0)
            
            valid = True
        except Exception:
            pixel_values = torch.zeros((3, 224, 224))
            brightness, colorfulness, color_temp = 0.0, 0.0, 0.0
            valid = False

        return {
            'pixel_values': pixel_values,
            'username': username,
            'post_id': post_id,
            'category': category,
            'brightness': brightness,
            'colorfulness': colorfulness,
            'color_temp': color_temp,
            'valid': valid
        }

# ==========================================================
# 2. ヘルパー関数
# ==========================================================
def parse_filename(file_path):
    try:
        fname = os.path.basename(file_path)
        fname_no_ext = os.path.splitext(fname)[0]
        dir_username = os.path.basename(os.path.dirname(file_path))
        if fname_no_ext.startswith(dir_username + '-'):
            post_id = fname_no_ext[len(dir_username) + 1:]
            return dir_username, post_id
        else:
            return dir_username, fname_no_ext
    except:
        return None, None

def get_file_list(base_dir):
    tasks = []
    print(f"Scanning directory: {base_dir} ...")
    for category in os.listdir(base_dir):
        c_path = os.path.join(base_dir, category)
        if not os.path.isdir(c_path): continue
        for u_dir in os.listdir(c_path):
            u_path = os.path.join(c_path, u_dir)
            if not os.path.isdir(u_path): continue
            for fname in os.listdir(u_path):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    f_path = os.path.join(u_path, fname)
                    username, post_id = parse_filename(f_path)
                    if username:
                        tasks.append((f_path, category, username, post_id))
    return tasks

# ==========================================================
# 3. メイン処理 (Mac Optimized)
# ==========================================================
def main():
    # ★ Mac (MPS) 検知ロジック
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Mac GPU Detected: Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✅ NVIDIA GPU Detected: Using CUDA")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU Not Detected: Using CPU (Slow)")

    # 既存ファイル削除
    if os.path.exists(OUTPUT_FILE):
        print(f"Removing old file: {OUTPUT_FILE}")
        os.remove(OUTPUT_FILE)

    tasks = get_file_list(BASE_DIR)
    if not tasks:
        print("No images found.")
        return
    print(f"Total Images: {len(tasks)}")

    print("Loading ImageNet Model...")
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device)
    model.eval()

    # MacではFP16 (Half) は動作が不安定な場合があるため,安全のためFloat32で実行します。
    # UltraならFloat32でも十分高速です。

    id2label = model.config.id2label

    # DataLoader
    # Macの場合,num_workersが多すぎると "Too many open files" エラーが出ることがあるので注意
    dataset = InfluencerImageDataset(tasks, processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True, # MPSでも有効
        prefetch_factor=2
    )

    cols = [
        'username', 'post_id', 'user_category', 
        'top1_label', 'top1_score', 'top2_label', 'top2_score', 'top3_label', 'top3_score',
        'brightness', 'colorfulness', 'color_temp'
    ]
    pd.DataFrame(columns=cols).to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"Starting Processing on {device}. Batch: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    
    buffer = []
    save_interval = 20

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, unit="batch")):
            
            # データをデバイスへ転送
            pixel_values = batch['pixel_values'].to(device, non_blocking=True)
            valid_mask = batch['valid']

            # --- 推論 ---
            outputs = model(pixel_values)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Top 3
            top3_probs, top3_indices = torch.topk(probs, 3, dim=1)
            
            # CPUへ戻す
            top3_probs = top3_probs.cpu().numpy()
            top3_indices = top3_indices.cpu().numpy()
            
            current_batch_size = len(pixel_values)
            for i in range(current_batch_size):
                if not valid_mask[i]: continue

                buffer.append({
                    'username': batch['username'][i],
                    'post_id': batch['post_id'][i],
                    'user_category': batch['category'][i],
                    'brightness': batch['brightness'][i].item(),
                    'colorfulness': batch['colorfulness'][i].item(),
                    'color_temp': batch['color_temp'][i].item(),
                    
                    'top1_label': id2label[top3_indices[i][0]],
                    'top1_score': top3_probs[i][0],
                    'top2_label': id2label[top3_indices[i][1]],
                    'top2_score': top3_probs[i][1],
                    'top3_label': id2label[top3_indices[i][2]],
                    'top3_score': top3_probs[i][2],
                })

            # 保存 & メモリ解放
            if (batch_idx + 1) % save_interval == 0 and buffer:
                df = pd.DataFrame(buffer)
                df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
                buffer = [] 
                gc.collect()

    if buffer:
        df = pd.DataFrame(buffer)
        df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

    print("DONE! All data processed and saved.")

if __name__ == "__main__":
    # Macでは 'spawn' ではなく 'fork' がデフォルトの場合がありますが,
    # PyTorchでは 'spawn' が推奨されることが多いため明示します
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    main()