import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="ダッシュボード", layout="wide")

# ▼▼▼ 修正点: 正しいカテゴリのリストを定義 ▼▼▼
VALID_CATEGORIES = [
    'beauty', 'family', 'fashion', 'fitness', 'food', 
    'interior', 'pet', 'travel', 'Other'
]
# ▲▲▲ 修正点 ▲▲▲

# --- データ読み込み（キャッシュを利用） ---
@st.cache_data
def load_influencer_data(filepath):
    """influencers.txtを正しく読み込み、不正なカテゴリを除外する関数"""
    try:
        df = pd.read_csv(filepath, sep='\t')
        # ▼▼▼ 修正点: 不正なデータをクリーニング ▼▼▼
        # 正しいカテゴリリストに含まれる行のみを保持
        df = df[df['Category'].isin(VALID_CATEGORIES)]
        return df
        # ▲▲▲ 修正点 ▲▲▲
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def load_posts_data(filepath):
    """output_beauty_category.csvを読み込む"""
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def create_summary_table(df_influencers, df_posts):
    """カテゴリ別の要約統計量テーブルを作成する"""
    posts_agg = df_posts.groupby('username')[['likes', 'comments']].mean().reset_index()
    df_merged = pd.merge(df_influencers, posts_agg, left_on='Username', right_on='username', how='left')
    df_merged['Follower/Followee Ratio'] = df_merged['#Followers'] / df_merged['#Followees'].replace(0, np.nan)
    df_merged['Engagement Rate (%)'] = ((df_merged['likes'] + df_merged['comments']) / df_merged['#Followers'].replace(0, np.nan)) * 100
    summary = df_merged.groupby('Category').agg(
        インフルエンサー数=('Username', 'size'),
        フォロワー数_中央値=('#Followers', 'median'),
        エンゲージメント率_中央値=('Engagement Rate (%)', 'median'),
        投稿数_中央値=('#Posts', 'median'),
        フォロワー_フォロー比率_中央値=('Follower/Followee Ratio', 'median')
    ).reset_index()
    summary.rename(columns={
        'Category': 'カテゴリ', 'フォロワー数_中央値': 'フォロワー数（中央値）',
        'エンゲージメント率_中央値': 'エンゲージメント率（中央値）', '投稿数_中央値': '投稿数（中央値）',
        'フォロワー_フォロー比率_中央値': 'フォロワー/フォロー比率（中央値）'
    }, inplace=True)
    return summary

# --- UI描画 ---
st.title("📊 データセット概要ダッシュボード")
st.write("インフルエンサーデータセット全体の傾向を分析します。")

df_influencers = load_influencer_data('influencers.txt')
df_posts = load_posts_data('output_beauty_category.csv')

if df_influencers is None or df_posts is None:
    st.warning("必要なデータファイルが読み込めませんでした。")
    st.stop()

# ▼▼▼ 修正点: 統一されたカテゴリ順序を定義 ▼▼▼
CATEGORY_ORDER = sorted(df_influencers['Category'].unique())
# ▲▲▲ 修正点 ▲▲▲

# --- 1. 主要指標 (KPI) の表示 ---
st.markdown("---")
st.header("主要指標")
col1, col2, col3, col4 = st.columns(4)
col1.metric("インフルエンサー総数", f"{len(df_influencers):,} 人")
col2.metric("総フォロワー数", f"{int(df_influencers['#Followers'].sum()):,} 人")
col3.metric("総投稿数", f"{int(df_influencers['#Posts'].sum()):,} 件")
col4.metric("カテゴリ数", f"{df_influencers['Category'].nunique()} 種類")

# --- 2. カテゴリ別 要約統計量 ---
st.markdown("---")
st.header("カテゴリ別 要約統計量")
st.write("各カテゴリのインフルエンサーの特性を中央値で比較します。")
summary_df = create_summary_table(df_influencers, df_posts)
# ▼▼▼ 修正点: 表の順序を統一 ▼▼▼
summary_df['カテゴリ'] = pd.Categorical(summary_df['カテゴリ'], categories=CATEGORY_ORDER, ordered=True)
summary_df = summary_df.sort_values('カテゴリ')
# ▲▲▲ 修正点 ▲▲▲
st.dataframe(
    summary_df.set_index('カテゴリ'),
    use_container_width=True,
    column_config={
        "インフルエンサー数": st.column_config.NumberColumn(format="%d 人"),
        "フォロワー数（中央値）": st.column_config.NumberColumn(format="%d"),
        "エンゲージメント率（中央値）": st.column_config.NumberColumn(format="%.2f %%"),
        "投稿数（中央値）": st.column_config.NumberColumn(format="%d"),
        "フォロワー/フォロー比率（中央値）": st.column_config.NumberColumn(format="%.2f"),
    }
)

# --- 3. 上位・下位ランキング ---
st.markdown("---")
st.header("フォロワー数ランキング")
# ... (このセクションは変更なし) ...

# --- 4. カテゴリ分析 ---
st.markdown("---")
st.header("カテゴリ分析")
category_counts = df_influencers['Category'].value_counts().reset_index()
category_counts.columns = ['Category', 'Count']
# ▼▼▼ 修正点: グラフの順序を統一 ▼▼▼
fig_cat = px.bar(category_counts, x='Category', y='Count', title='カテゴリごとのインフルエンサー数',
                 category_orders={'Category': CATEGORY_ORDER})
# ▲▲▲ 修正点 ▲▲▲
st.plotly_chart(fig_cat, use_container_width=True)

# --- 5. 分布分析 ---
# ... (このセクションは変更なし) ...

# --- 6. 相関分析 (フィルター機能付き) ---
st.markdown("---")
st.header("相関分析")
st.write("フォロワー数と投稿数の関係性を分析します。")

selected_categories = st.multiselect(
    '表示するカテゴリを選択 (複数可):',
    options=CATEGORY_ORDER, # 選択肢も統一された順序に
    default=CATEGORY_ORDER
)

if not selected_categories:
    st.warning("少なくとも1つのカテゴリを選択してください。")
else:
    filtered_df = df_influencers[df_influencers['Category'].isin(selected_categories)]
    # ▼▼▼ 修正点: グラフの凡例順序を統一 ▼▼▼
    fig_scatter = px.scatter(
        filtered_df, x='#Followers', y='#Posts', color='Category',
        hover_name='Username', title='フォロワー数 vs 投稿数', log_x=True,
        category_orders={'Category': CATEGORY_ORDER}
    )
    # ▲▲▲ 修正点 ▲▲▲
    st.plotly_chart(fig_scatter, use_container_width=True)
