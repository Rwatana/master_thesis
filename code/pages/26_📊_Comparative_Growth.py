import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Pythonのパスに親ディレクトリを追加
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

st.set_page_config(page_title="成長率 比較分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_growth_data(filepath):
    """事前に計算されたgrowth_rates_normalized.csvを読み込む"""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。先に `growth_analyzer.py` を実行してください。")
        return None

# ▼▼▼ 修正: influencers.txtを読み込み、列名を整形する ▼▼▼
@st.cache_data
def load_influencer_data(filepath):
    """influencers.txtを読み込み、列名を整形する"""
    try:
        # skiprows=[1]を削除し、ヘッダーに関わらず列名を直接指定
        df = pd.read_csv(filepath, sep='\t')
        df.columns = ['Username', 'Category', 'followers', 'Followees', 'Posts']
        return df
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None
# ▲▲▲ 修正 ▲▲▲


# --- UI描画 ---
st.title("📊 成長率 比較分析")
st.info("インフルエンサーの成長率を「カテゴリ」や「規模」でグループ化し、同じ土俵でのパフォーマンスを比較します。")

# --- データの準備 ---
df_growth = load_growth_data('growth_rates_normalized.csv')
df_influencers = load_influencer_data('influencers.txt')

if df_growth is None or df_influencers is None:
    st.stop()

# ▼▼▼ 修正: カテゴリとフォロワー数の両方を結合 ▼▼▼
# 成長率データにカテゴリとフォロワー数情報を結合
# df_growthの'followers'列は分類に使っただけなので、influencers.txtの最新情報で上書きする
df_growth_no_followers = df_growth.drop(columns=['followers'], errors='ignore')
df_analysis = pd.merge(df_growth_no_followers, df_influencers[['Username', 'Category', 'followers']], left_on='username', right_on='Username', how='left')
# ▲▲▲ 修正 ▲▲▲


# --- サイドバー ---
st.sidebar.header("表示設定")
metric_to_analyze = st.sidebar.radio(
    "分析する成長指標:",
    ('正規化いいね成長率 (%)', '絶対いいね成長率 (Slope)'),
    key='compare_metric'
)
growth_column = 'normalized_likes_growth_pct' if metric_to_analyze == '正規化いいね成長率 (%)' else 'likes_growth_rate'


# --- 分析タブ ---
tab1, tab2, tab3 = st.tabs(["カテゴリ別 比較", "規模別 比較", "カテゴリ × 規模 比較"])

with tab1:
    st.header("カテゴリ別の平均成長率")
    st.write("どのカテゴリのインフルエンサーが、全体的に成長しやすい傾向にあるかを示します。")
    
    # カテゴリごとの平均成長率を計算
    category_growth = df_analysis.groupby('Category')[growth_column].median().sort_values(ascending=False)
    
    fig1 = px.bar(
        category_growth,
        title='カテゴリ別 平均成長率',
        labels={'value': f'中央値: {metric_to_analyze}', 'Category': 'カテゴリ'}
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    st.header("インフルエンサー規模別の平均成長率")
    st.write("どの規模のインフルエンサーが、最も成長ポテンシャルが高いかを示します。")
    
    # 規模ごとの平均成長率を計算
    type_order = ['Nano', 'Micro', 'Macro', 'Mega']
    scale_growth = df_analysis.dropna(subset=['influencer_type']).groupby('influencer_type')[growth_column].median().reindex(type_order)

    fig2 = px.bar(
        scale_growth,
        title='インフルエンサー規模別 平均成長率',
        labels={'value': f'中央値: {metric_to_analyze}', 'influencer_type': 'インフルエンサー規模'}
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.header("カテゴリ × 規模 ごとの成長率分布")
    st.write("同じカテゴリ・同じ規模のインフルエンサー同士のパフォーマンスを比較します。箱ひげ図は、グループ内のばらつきや中央値、外れ値を示します。")

    fig3 = px.box(
        df_analysis,
        x='Category',
        y=growth_column,
        color='influencer_type',
        title=f'カテゴリ・規模別 {metric_to_analyze} の分布',
        labels={'Category': 'カテゴリ', growth_column: metric_to_analyze},
        category_orders={'influencer_type': ['Nano', 'Micro', 'Macro', 'Mega']}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("詳細データテーブル")
    st.write("グラフの元となる詳細なデータです。")
    st.dataframe(
        df_analysis.sort_values(growth_column, ascending=False),
        use_container_width=True,
        column_config={
            "normalized_likes_growth_pct": st.column_config.ProgressColumn(
                "正規化いいね成長率 (%)", format="%.2f%%",
                min_value=0, max_value=float(df_analysis['normalized_likes_growth_pct'].max()) if not df_analysis.empty else 1
            ),
        }
    )

