import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="ユーザーエクスプローラー", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_growth_data(filepath):
    """事前に計算されたgrowth_rates_normalized.csvを読み込む"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。先に `growth_analyzer.py` を実行してください。")
        return None

# ▼▼▼ 修正: influencers.txtを読み込む関数を追加 ▼▼▼
@st.cache_data
def load_influencer_data(filepath):
    """influencers.txtを読み込む"""
    try:
        # skiprows=[1]を削除し、ヘッダーに関わらず列名を直接指定
        df = pd.read_csv(filepath, sep='\t')
        df.columns = ['Username', 'Category', 'followers_info', 'Followees', 'Posts']
        return df
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None
# ▲▲▲ 修正 ▲▲▲


# --- UI描画 ---
st.title("👥 ユーザーエクスプローラー")
st.info("インフルエンサーを「カテゴリ」や「規模」でグループ化し、各グループに属するユーザーを成長率の高い順に表示します。")

# --- データの準備 ---
df_growth = load_growth_data('growth_rates_normalized.csv')
# ▼▼▼ 修正: influencers.txtを読み込み、カテゴリ情報を結合する ▼▼▼
df_influencers = load_influencer_data('influencers.txt')

if df_growth is None or df_influencers is None:
    st.stop()

# 成長率データにカテゴリ情報を結合
df_analysis = pd.merge(df_growth, df_influencers[['Username', 'Category']], left_on='username', right_on='Username', how='left')
# ▲▲▲ 修正 ▲▲▲

# --- サイドバー ---
st.sidebar.header("表示設定")
sort_metric = st.sidebar.radio(
    "並び替えの基準となる指標:",
    ('正規化いいね成長率 (%)', '絶対いいね成長率 (Slope)')
)
growth_column = 'normalized_likes_growth_pct' if sort_metric == '正規化いいね成長率 (%)' else 'likes_growth_rate'


# --- 分析タブ ---
tab1, tab2, tab3 = st.tabs(["カテゴリ別", "インフルエンサー規模別", "カテゴリ × 規模 詳細"])

# 表示する列を定義
columns_to_display = [
    'username', 'Category', 'followers', 'influencer_type', 'normalized_likes_growth_pct', 
    'likes_growth_rate', 'average_likes'
]

with tab1:
    st.header("カテゴリ別 ユーザーリスト")
    st.write("各カテゴリに属するインフルエンサーを、選択した成長指標の高い順に表示します。")

    for category in sorted(df_analysis['Category'].dropna().unique()):
        with st.expander(f"📁 カテゴリ: {category}"):
            category_df = df_analysis[df_analysis['Category'] == category]
            sorted_df = category_df.sort_values(growth_column, ascending=False)
            st.dataframe(sorted_df[columns_to_display], use_container_width=True)

with tab2:
    st.header("インフルエンサー規模別 ユーザーリスト")
    
    st.info("""
    **インフルエンサータイプの定義:**
    - **Mega (メガ)**: 100万人以上
    - **Macro (マクロ)**: 10万～100万人
    - **Micro (マイクロ)**: 1万～10万人
    - **Nano (ナノ)**: 1,000～1万人
    """)
    
    type_order = ['Nano', 'Micro', 'Macro', 'Mega']
    
    for influencer_type in type_order:
        with st.expander(f"👤 規模: {influencer_type}"):
            # influencer_typeがNaNの行を除外してフィルタリング
            type_df = df_analysis.dropna(subset=['influencer_type'])
            type_df = type_df[type_df['influencer_type'] == influencer_type]
            sorted_df = type_df.sort_values(growth_column, ascending=False)
            st.dataframe(sorted_df[columns_to_display], use_container_width=True)

with tab3:
    st.header("カテゴリ × 規模 詳細リスト")
    st.write("カテゴリと規模の両方でグループ化した、より詳細なユーザーリストです。")
    
    for category in sorted(df_analysis['Category'].dropna().unique()):
        with st.expander(f"📁 カテゴリ: {category}"):
            category_df = df_analysis[df_analysis['Category'] == category]
            
            type_order = ['Nano', 'Micro', 'Macro', 'Mega']

            for influencer_type in type_order:
                # influencer_typeがNaNの行を除外してフィルタリング
                type_df = category_df.dropna(subset=['influencer_type'])
                type_df = type_df[type_df['influencer_type'] == influencer_type]
                if not type_df.empty:
                    st.subheader(f"👤 規模: {influencer_type}")
                    sorted_df = type_df.sort_values(growth_column, ascending=False)
                    st.dataframe(sorted_df[columns_to_display], use_container_width=True)

