import streamlit as st
import pandas as pd
import os
import json

st.set_page_config(page_title="データビューア", layout="wide")
st.title("📊 Data Viewer")
st.write("各データファイルの先頭部分をプレビュー表示します。")
st.markdown("---")

# --- 1. influencers.txt の表示 ---
st.header("Raw Data: influencers.txt")
st.write("インフルエンサーの基本情報ファイル。")
try:
    df_influencers = pd.read_csv('influencers.txt', sep='\t', skiprows=[1])
    st.dataframe(df_influencers.head(10))
except FileNotFoundError:
    st.error("`influencers.txt` が見つかりません。")
except Exception as e:
    st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# --- 2. JSON-Image_files_mapping.txt の表示 ---
st.header("Raw Data: JSON-Image_files_mapping.txt")
st.write("投稿メタデータ(JSON)と画像ファイルのマッピング情報。")
try:
    df_mapping = pd.read_csv('JSON-Image_files_mapping.txt', sep='\t', header=None, names=["influencer_name", "JSON_PostMetadata_file_name", "Image_file_name"])
    st.dataframe(df_mapping.head(10))
except FileNotFoundError:
    st.error("`JSON-Image_files_mapping.txt` が見つかりません。")
except Exception as e:
    st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# --- 3. 投稿メタデータ (JSONファイル) のサンプル表示 ---
st.header("投稿メタデータ (JSONファイルのサンプル)")
st.write("指定されたサンプルファイル `00_rocketgirl-1188140434601337485.info` の中身を表示します。")

info_dir = 'posts_info/unzipped_data_7z/info/'
# ▼▼▼ 修正点: ファイル名を直接指定 ▼▼▼
sample_file_name = '00_rocketgirl-1188140434601337485.info'
sample_file_path = os.path.join(info_dir, sample_file_name)
# ▲▲▲ 修正点 ▲▲▲

try:
    st.write(f"サンプルファイル: **{sample_file_name}**")
    
    with open(sample_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        st.json(json_data, expanded=False)

except FileNotFoundError:
    st.error(f"指定されたサンプルファイル `{sample_file_path}` が見つかりません。")
except Exception as e:
    st.error(f"JSONファイルの読み込み中にエラーが発生しました: {e}")

# --- 集計済みファイルプレビュー ---
st.markdown("---")
st.header("📂 集計済みファイルプレビュー (Processed Data Preview)")
st.write("各種スクリプトによって生成されたCSVファイルの先頭10行を表示します。")

files_to_preview = [
    'preprocessed_posts_with_metadata.csv',
    'output_beauty_category.csv',
    'output_hashtags_all_parallel.csv',
    'output_mentions_all_parallel.csv'
]

for filepath in files_to_preview:
    with st.expander(f"ファイル: `{filepath}`"):
        try:
            df_preview = pd.read_csv(filepath, nrows=10)
            st.dataframe(df_preview)
        except FileNotFoundError:
            st.warning(f"ファイル `{filepath}` が見つかりませんでした。")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
