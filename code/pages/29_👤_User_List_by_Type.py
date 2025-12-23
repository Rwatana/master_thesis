import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ユーザーリスト", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_growth_data(filepath):
    """事前に計算されたgrowth_rates_normalized.csvを読み込む"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。先に `growth_analyzer.py` を実行してください。")
        return None

@st.cache_data
def load_influencer_data(filepath):
    """influencers.txtを読み込む"""
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=[1])
        # 列名を統一
        df.columns = ['Username', 'Category', '#Followers', '#Followees', '#Posts']
        return df
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def calculate_posting_period(filepath):
    """preprocessed_posts_with_metadata.csvから各ユーザーの投稿期間を計算する"""
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'], usecols=['username', 'datetime'])
        if df.empty:
            return pd.DataFrame(columns=['username', 'posting_period_days'])
            
        period_df = df.groupby('username')['datetime'].agg(['min', 'max'])
        period_df['posting_period_days'] = (period_df['max'] - period_df['min']).dt.days
        return period_df[['posting_period_days']].reset_index()
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。先に `preprocess_data.py` を実行してください。")
        return None

# --- UI描画 ---
st.title("👤 ユーザーリスト（規模別）")
st.info("インフルエンサーの規模、カテゴリ、活動期間で絞り込みを行い、各ユーザーの成長率や活動期間を比較します。")

# --- データの準備 ---
df_growth = load_growth_data('growth_rates_normalized.csv')
df_period = calculate_posting_period('preprocessed_posts_with_metadata.csv')
df_influencers = load_influencer_data('influencers.txt')

if any(df is None for df in [df_growth, df_period, df_influencers]):
    st.stop()

df_temp = pd.merge(df_growth, df_period, on='username', how='left')
df_analysis = pd.merge(df_temp, df_influencers[['Username', 'Category']], left_on='username', right_on='Username', how='left')

# --- サイドバー ---
st.sidebar.header("フィルター設定")

# 1. インフルエンサータイプによる絞り込み
type_order = ['Nano', 'Micro', 'Macro', 'Mega']
selected_types = st.sidebar.multiselect(
    'インフルエンサー規模で絞り込む (任意):',
    options=type_order,
    default=[]
)

# ▼▼▼ 新機能: カテゴリによる絞り込み ▼▼▼
all_categories = sorted(df_analysis['Category'].dropna().unique())
selected_categories = st.sidebar.multiselect(
    'カテゴリで絞り込む (任意):',
    options=all_categories,
    default=[]
)
# ▲▲▲ 新機能 ▲▲▲


# ▼▼▼ 新機能: 活動期間による絞り込み ▼▼▼
min_period = int(df_analysis['posting_period_days'].min())
max_period = int(df_analysis['posting_period_days'].max())
selected_period = st.sidebar.slider(
    '活動期間（日数）で絞り込む (任意):',
    min_value=min_period,
    max_value=max_period,
    value=(min_period, max_period) # デフォルトは全範囲
)
# ▲▲▲ 新機能 ▲▲▲

# 並び替え指標の選択
sort_metric = st.sidebar.radio(
    "ランキングの並び替え基準:",
    ('正規化いいね成長率 (%)', '絶対いいね成長率 (Slope)')
)
growth_column = 'normalized_likes_growth_pct' if sort_metric == '正規化いいね成長率 (%)' else 'likes_growth_rate'

# --- メイン画面 ---
# フィルタリング
df_filtered = df_analysis.copy()
header_filters = []

if selected_types:
    df_filtered = df_filtered[df_filtered['influencer_type'].isin(selected_types)]
    header_filters.append(f"規模: {', '.join(selected_types)}")
if selected_categories:
    df_filtered = df_filtered[df_filtered['Category'].isin(selected_categories)]
    header_filters.append(f"カテゴリ: {', '.join(selected_categories)}")
# スライダーがデフォルト値から変更された場合のみフィルターを適用
if selected_period != (min_period, max_period):
    df_filtered = df_filtered[
        (df_filtered['posting_period_days'] >= selected_period[0]) &
        (df_filtered['posting_period_days'] <= selected_period[1])
    ]
    header_filters.append(f"活動期間: {selected_period[0]}~{selected_period[1]}日")

if header_filters:
    st.header(f"📈 [{ ' | '.join(header_filters) }] のユーザーリスト")
else:
    st.header("📈 全てのユーザーリスト")

st.write(f"絞り込まれた **{len(df_filtered)}** 人のインフルエンサーを「{sort_metric}」の高い順に表示しています。")

# ランキング表示
if not df_filtered.empty:
    df_sorted = df_filtered.sort_values(growth_column, ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index + 1
    
    columns_to_display = [
        'rank', 'username', 'Category', 'followers', 'influencer_type',
        'posting_period_days', 'normalized_likes_growth_pct', 'likes_growth_rate', 
        'average_likes'
    ]
    st.dataframe(df_sorted[columns_to_display], use_container_width=True)
else:
    st.warning("選択された条件に合致するインフルエンサーがいません。")

