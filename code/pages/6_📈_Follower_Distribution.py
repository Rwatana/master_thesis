import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="傾向分析", layout="wide")

# ▼▼▼ 修正点: 正しいカテゴリのリストを定義 ▼▼▼
VALID_CATEGORIES = [
    'beauty', 'family', 'fashion', 'fitness', 'food', 
    'interior', 'pet', 'travel', 'Other'
]
# ▲▲▲ 修正点 ▲▲▲

# --- データ読み込み（キャッシュを利用） ---
@st.cache_data
def load_data(filepath):
    """influencers.txtを正しく読み込み、不正なカテゴリを除外する関数"""
    try:
        df = pd.read_csv(filepath, sep='\t')
        # ▼▼▼ 修正点: 不正なデータをクリーニング ▼▼▼
        df = df[df['Category'].isin(VALID_CATEGORIES)]
        return df
        # ▲▲▲ 修正点 ▲▲▲
    except FileNotFoundError:
        return None

# --- UI描画 ---
st.title("📈 カテゴリ別 傾向分析")
st.info(
    """
    カテゴリごとに「フォロワー数」と「フォロー数」がどのように分布しているかを分析します。\n
    **箱ひげ図の見方**:
    - **箱**: データの中間50%（25パーセンタイル〜75パーセンタイル）の範囲を示します。
    - **箱の中の線**: 中央値（メディアン）です。
    - **上下の線（ひげ）**: 外れ値を除いたデータの最小値と最大値を示します。
    """
)

df_influencers = load_data('influencers.txt')

if df_influencers is None:
    st.error("`influencers.txt` が見つかりません。")
    st.stop()

# ▼▼▼ 修正点: 統一されたカテゴリ順序を定義 ▼▼▼
CATEGORY_ORDER = sorted(df_influencers['Category'].unique())
# ▲▲▲ 修正点 ▲▲▲

# --- 1. フォロワー数のカテゴリ別分布 ---
st.markdown("---")
st.header("フォロワー数のカテゴリ別分布")
st.write("カテゴリによってフォロワー数の分布にどのような違いがあるかを確認できます。")

# ▼▼▼ 修正点: グラフの順序を統一 ▼▼▼
fig_followers = px.box(
    df_influencers,
    x='Category',
    y='#Followers',
    title='カテゴリごとのフォロワー数 分布',
    labels={'#Followers': 'フォロワー数', 'Category': 'カテゴリ'},
    log_y=True,
    category_orders={'Category': CATEGORY_ORDER}
)
# ▲▲▲ 修正点 ▲▲▲
st.plotly_chart(fig_followers, use_container_width=True)

# --- 2. フォロー数のカテゴリ別分布 ---
st.markdown("---")
st.header("フォロー数のカテゴリ別分布")
st.write("フォロー数の傾向をカテゴリ別に比較します。")

# ▼▼▼ 修正点: グラフの順序を統一 ▼▼▼
fig_followees = px.box(
    df_influencers,
    x='Category',
    y='#Followees',
    title='カテゴリごとのフォロー数 分布',
    labels={'#Followees': 'フォロー数', 'Category': 'カテゴリ'},
    category_orders={'Category': CATEGORY_ORDER}
)
# ▲▲▲ 修正点 ▲▲▲
st.plotly_chart(fig_followees, use_container_width=True)
