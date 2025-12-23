import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="成長率ランキング", layout="wide")

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
        return pd.read_csv(filepath, sep='\t', skiprows=[1])
    except FileNotFoundError:
        return None

def load_user_post_data(username):
    """ユーザーごとに分割された投稿データを読み込む"""
    filepath = f"user_data/{username}.csv"
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.warning(f"ユーザー '{username}' の個別データファイルが見つかりませんでした。")
        return None

# --- UI描画 ---
st.title("🏆 成長率ランキング分析")
st.info("事前に計算された成長率スコアに基づき、インフルエンサーをランキング形式で表示・分析します。")

# --- データの準備 ---
df_growth = load_growth_data('growth_rates_normalized.csv')
df_influencers = load_influencer_data('influencers.txt')

if df_growth is None or df_influencers is None:
    st.stop()

# 成長率データにカテゴリ情報を結合
df_ranked = pd.merge(df_growth, df_influencers[['Username', 'Category']], left_on='username', right_on='Username', how='left')

# --- サイドバー ---
st.sidebar.header("フィルター設定")

# 1. カテゴリによる絞り込み（任意）
all_categories = sorted(df_ranked['Category'].dropna().unique())
selected_categories = st.sidebar.multiselect(
    'カテゴリで絞り込む (任意):',
    options=all_categories,
    default=[]
)

# ▼▼▼ 新機能: インフルエンサータイプでの絞り込み ▼▼▼
# 2. インフルエンサータイプによる絞り込み（任意）
all_types = sorted(df_ranked['influencer_type'].dropna().unique())
selected_types = st.sidebar.multiselect(
    'インフルエンサータイプで絞り込む (任意):',
    options=all_types,
    default=[]
)
# ▲▲▲ 新機能 ▲▲▲

# 3. ランキング指標の選択
metric_to_rank = st.sidebar.radio(
    "ランキング指標:",
    ('正規化いいね成長率 (%)', '正規化コメント成長率 (%)', '絶対いいね成長率', '絶対コメント成長率')
)
if metric_to_rank == '正規化いいね成長率 (%)':
    growth_column = 'normalized_likes_growth_pct'
    metric_column_for_graph = 'likes'
elif metric_to_rank == '正規化コメント成長率 (%)':
    growth_column = 'normalized_comments_growth_pct'
    metric_column_for_graph = 'comments'
elif metric_to_rank == '絶対いいね成長率':
    growth_column = 'likes_growth_rate'
    metric_column_for_graph = 'likes'
else: # 絶対コメント成長率
    growth_column = 'comments_growth_rate'
    metric_column_for_graph = 'comments'


# --- メイン画面 ---
# フィルタリング
df_filtered = df_ranked.copy()
header_filters = []
if selected_categories:
    df_filtered = df_filtered[df_filtered['Category'].isin(selected_categories)]
    header_filters.append(f"カテゴリ: {', '.join(selected_categories)}")
if selected_types:
    df_filtered = df_filtered[df_filtered['influencer_type'].isin(selected_types)]
    header_filters.append(f"タイプ: {', '.join(selected_types)}")

if header_filters:
    st.header(f"📈 [{ ' | '.join(header_filters) }] 内での成長率ランキング")
else:
    st.header("📈 全体での成長率ランキング")

# ランキングの計算と表示
df_sorted = df_filtered.sort_values(growth_column, ascending=False).reset_index(drop=True)
df_sorted['rank'] = df_sorted.index + 1

# ▼▼▼ 表示する列を更新 ▼▼▼
st.dataframe(df_sorted[[
    'rank', 'username', 'Category', 'influencer_type', 
    'normalized_likes_growth_pct', 'likes_growth_rate', 
    'normalized_comments_growth_pct', 'comments_growth_rate', 'average_likes'
]], use_container_width=True)
# ▲▲▲ 表示する列を更新 ▲▲▲


# --- 個別ユーザーの詳細分析 ---
st.markdown("---")
st.header("👤 個別ユーザーの成長トレンド分析")
st.write("上のランキングから詳細を見たいユーザーを一人選択してください。")

user_options = df_sorted['username'].tolist()
selected_user_detail = st.selectbox(
    "ユーザーを選択 (任意):",
    options=['（選択しない）'] + user_options
)

if selected_user_detail != '（選択しない）':
    with st.spinner(f"'{selected_user_detail}' の投稿データを読み込んでいます..."):
        df_detail = load_user_post_data(selected_user_detail)

    if df_detail is not None:
        fig = px.scatter(
            df_detail,
            x='datetime',
            y=metric_column_for_graph,
            title=f'{selected_user_detail} の「{metric_column_for_graph}」数の推移と近似曲線',
            labels={'datetime': '投稿日時', metric_column_for_graph: f'{metric_column_for_graph}数'},
            trendline="ols",
            trendline_color_override="red"
        )
        st.plotly_chart(fig, use_container_width=True)

