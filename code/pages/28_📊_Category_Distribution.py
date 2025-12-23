import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="カテゴリ内部分布", layout="wide")

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
        df = pd.read_csv(filepath, sep='\t')
        df.columns = ['Username', 'Category', 'followers_info', 'Followees', 'Posts']
        return df
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None


# --- UI描画 ---
st.title("📊 カテゴリ内のインフルエンサー規模分布")
st.info("各カテゴリ内に、どの規模（Nano, Micro, Macro, Mega）のインフルエンサーが何人存在するかを分析します。")

# --- データの準備 ---
df_growth = load_growth_data('growth_rates_normalized.csv')
df_influencers = load_influencer_data('influencers.txt')

if df_growth is None or df_influencers is None:
    st.stop()

# 成長率データにカテゴリ情報を結合
df_analysis = pd.merge(df_growth, df_influencers[['Username', 'Category']], left_on='username', right_on='Username', how='left')


# --- 分析タブ ---
# ▼▼▼ 修正点: 新しいタブを追加 ▼▼▼
tab0, tab1, tab2 = st.tabs(["データセット全体の分布", "カテゴリ内訳（グラフ）", "カテゴリ内訳（集計表）"])

with tab0:
    st.header("全体の分布")
    st.write("データセットに含まれる全インフルエンサーの規模とカテゴリの分布です。")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # インフルエンサー規模ごとの分布
        st.subheader("インフルエンサー規模の分布")
        type_counts = df_analysis['influencer_type'].value_counts()
        fig_type = px.bar(
            type_counts, 
            x=type_counts.index, 
            y=type_counts.values,
            labels={'x': 'インフルエンサー規模', 'y': '人数'},
            title="全体の規模ごとのインフルエンサー数",
            category_orders={'x': ['Nano', 'Micro', 'Macro', 'Mega']}
        )
        st.plotly_chart(fig_type, use_container_width=True)

    with col2:
        # カテゴリごとの分布
        st.subheader("カテゴリの分布")
        category_counts = df_analysis['Category'].value_counts()
        fig_cat = px.bar(
            category_counts, 
            x=category_counts.index, 
            y=category_counts.values,
            labels={'x': 'カテゴリ', 'y': '人数'},
            title="全体のカテゴリごとのインフルエンサー数"
        )
        st.plotly_chart(fig_cat, use_container_width=True)

# ▲▲▲ 修正点 ▲▲▲


with tab1:
    st.header("カテゴリ内訳（グラフ）")
    st.write("各カテゴリのインフルエンサー規模の構成比をグラフで確認します。")

    # カテゴリごとにループ
    for category in sorted(df_analysis['Category'].dropna().unique()):
        with st.expander(f"📁 カテゴリ: {category}"):
            # 該当カテゴリのデータを抽出
            category_df = df_analysis[df_analysis['Category'] == category]
            
            # インフルエンサータイプごとの人数をカウント
            type_counts = category_df['influencer_type'].value_counts()
            
            # 棒グラフを作成
            fig = px.bar(
                type_counts, 
                x=type_counts.index, 
                y=type_counts.values,
                labels={'x': 'インフルエンサー規模', 'y': '人数'},
                title=f"「{category}」カテゴリ内の規模分布",
                category_orders={'x': ['Nano', 'Micro', 'Macro', 'Mega']} # 順序を固定
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("カテゴリ内訳（集計表）")
    st.write("カテゴリと規模ごとのインフルエンサー数を集計した表です。")

    # ピボットテーブル（クロス集計）を作成
    summary_table = pd.crosstab(
        index=df_analysis['Category'], 
        columns=df_analysis['influencer_type']
    )
    
    # 表示順序を整える
    type_order = [col for col in ['Nano', 'Micro', 'Macro', 'Mega'] if col in summary_table.columns]
    
    st.dataframe(summary_table[type_order], use_container_width=True)

