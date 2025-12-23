import polars as pl
import os
from datetime import datetime

# --- 設定 ---
INPUT_FILE = 'preprocessed_posts_detailed.csv'

# 出力ファイル名
FILE_A = 'dataset_A_active_all.csv'
FILE_B = 'dataset_B_medium_rich.csv'
FILE_C = 'dataset_C_rich_only.csv'

def generate_three_datasets_fast():
    print(f"🚀 [M1 Ultra Optimized] Starting Multi-Dataset Generation...")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return

    start_time = datetime.now()

    # ---------------------------------------------------------
    # Step 1: ユーザーランクの判定 (Lazyモードで並列スキャン)
    # ---------------------------------------------------------
    print("   Step 1: Analyzing user activity levels (Jan-Nov 2017)...")
    
    # 学習期間の設定
    hist_start = datetime(2017, 1, 1)
    hist_end = datetime(2017, 11, 30, 23, 59, 59)

    # LazyFrameを使用してメタデータだけをまずスキャン
    q = (
        pl.scan_csv(INPUT_FILE, low_memory=False, rechunk=False)
        .with_columns(pl.col("datetime").str.to_datetime())
        .filter((pl.col("datetime") >= hist_start) & (pl.col("datetime") <= hist_end))
        .group_by("username")
        .agg(
            pl.col("datetime").dt.month().n_unique().alias("active_months")
        )
    )
    
    # 判定実行 (ここで全コアが回ります)
    user_stats = q.collect(streaming=True)

    # 各グループのユーザーリストを取得
    rich_users = user_stats.filter(pl.col("active_months") >= 9).get_column("username").unique()
    medium_users = user_stats.filter((pl.col("active_months") >= 4) & (pl.col("active_months") < 9)).get_column("username").unique()
    sparse_users = user_stats.filter((pl.col("active_months") >= 1) & (pl.col("active_months") < 4)).get_column("username").unique()

    # セットの構築
    # Polarsのis_inは非常に高速なので、集合演算を使わずそのまま渡せます
    list_A = pl.concat([rich_users, medium_users, sparse_users])
    list_B = pl.concat([rich_users, medium_users])
    list_C = rich_users

    print(f"   📊 User Stats:")
    print(f"      - Rich   : {len(rich_users):,} users")
    print(f"      - Medium : {len(medium_users):,} users")
    print(f"      - Sparse : {len(sparse_users):,} users")

    # ---------------------------------------------------------
    # Step 2: データセットの抽出と書き出し
    # ---------------------------------------------------------
    print("\n   Step 2: Processing and writing datasets...")

    # メインデータの読み込み (128GB RAMを活かしてメモリに載せる)
    # もしCSVが100GBを超える場合は streaming=True を維持します
    df_full = pl.read_csv(INPUT_FILE, low_memory=False)

    # Dataset A (Empty以外全て)
    print(f"      -> Writing {FILE_A}...")
    df_A = df_full.filter(pl.col("username").is_in(list_A))
    df_A.write_csv(FILE_A)
    print(f"         Total rows: {len(df_A):,}")

    # Dataset B (Medium + Rich)
    print(f"      -> Writing {FILE_B}...")
    df_B = df_A.filter(pl.col("username").is_in(list_B)) # df_Aから絞り込むことで高速化
    df_B.write_csv(FILE_B)
    print(f"         Total rows: {len(df_B):,}")

    # Dataset C (Rich only)
    print(f"      -> Writing {FILE_C}...")
    df_C = df_B.filter(pl.col("username").is_in(list_C)) # df_Bから絞り込むことで高速化
    df_C.write_csv(FILE_C)
    print(f"         Total rows: {len(df_C):,}")

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\n✅ All Done! Process took: {duration}")

if __name__ == "__main__":
    generate_three_datasets_fast()