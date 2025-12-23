import streamlit as st
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np
import plotly.express as px
import concurrent.futures
from tqdm import tqdm
from growth_analyzer_worker import calculate_growth_for_user # ヘルパーファイルから関数をインポート

st.set_page_config(page_title="成長分析", layout="wide")

# --- データ読み込みと計算 ---

@st.cache_data
def load_influencer_data(filepath):
    """influencers.txtを読み込む"""
    try:
        return pd.read_csv(filepath, sep='\t', skiprows=[1])
    except FileNotFoundError:
        return None

@st.cache_data
def load_category_posts_data(category):
    """指定されたカテゴリの事前処理済みCSVを読み込む"""
    if not category:
        return pd.DataFrame()
    
    safe_category_name = str(category).lower().replace(' ', '_')
    filepath = f"processed_by_category/processed_{safe_category_name}.csv"
    try:
        return pd.read_csv(filepath, parse_dates=['datetime'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。先に `split_preprocessed_by_category.py` を実行してください。")
        return None

@st.cache_data
def calculate_growth_rates_parallel(df):
    """ProcessPoolExecutorを使って全ユーザーの成長率を並列計算する"""
    if df is None or df.empty:
        return pd.DataFrame()

    user_groups = list(df.groupby('username'))
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # インポートした関数を使用
        results_iterator = executor.map(calculate_growth_for_user, user_groups)
        results = list(tqdm(results_iterator, total=len(user_groups), desc="Calculating growth rates"))

    valid_results = [r for r in results if r is not None]
    return pd.DataFrame(valid_results)

# --- UI描画 ---
st.title("🚀 成長分析（青田買い）")
st.info("エンゲージメント（いいね・コメント）が時間と共に成長しているインフルエンサーを特定します。")

# --- サイドバー ---
st.sidebar.header("分析設定")

df_influencers = load_influencer_data('influencers.txt')
if df_influencers is None:
    st.error("`influencers.txt` が見つかりません。")
    st.stop()

placeholder = '（分析したいカテゴリを選択してください）'

if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False
if 'analyzed_category' not in st.session_state:
    st.session_state.analyzed_category = None

all_categories = sorted(df_influencers['Category'].unique())
options = [placeholder] + all_categories

selected_category = st.sidebar.selectbox(
    "1. 分析したいカテゴリを選択:",
    options=options
)

if st.sidebar.button("分析を開始"):
    if selected_category != placeholder:
        st.session_state.run_analysis = True
        st.session_state.analyzed_category = selected_category
    else:
        st.sidebar.warning("先にカテゴリを選択してください。")

if selected_category != st.session_state.analyzed_category:
    st.session_state.run_analysis = False

if st.session_state.run_analysis:
    category_to_run = st.session_state.analyzed_category
    
    with st.spinner(f"'{category_to_run}' カテゴリのデータを読み込んでいます..."):
        df_posts = load_category_posts_data(category_to_run)

    if df_posts is None or df_posts.empty:
        st.warning(f"'{category_to_run}' カテゴリの投稿データがありません。")
        st.stop()

    with st.spinner(f"'{category_to_run}' カテゴリの成長率を並列計算中..."):
        df_growth = calculate_growth_rates_parallel(df_posts)

    if df_growth.empty:
        st.warning("成長率を計算できるユーザーがいませんでした。")
        st.stop()

    metric_to_analyze = st.sidebar.radio(
        "2. 分析対象の指標を選択:",
        ('いいね数の成長', 'コメント数の成長'),
        key='growth_metric'
    )
    growth_column = 'likes_growth_rate' if metric_to_analyze == 'いいね数の成長' else 'comments_growth_rate'

    min_growth_rate = st.sidebar.slider(
        '3. 表示する最小成長率（傾き）:',
        min_value=0.0,
        max_value=float(df_growth[growth_column].quantile(0.99)),
        value=float(df_growth[growth_column].quantile(0.80)),
        step=0.1,
        key=f"slider_{category_to_run}"
    )

    st.markdown("---")
    high_growth_users = df_growth[df_growth[growth_column] >= min_growth_rate].sort_values(growth_column, ascending=False)
    st.header(f"📈 [{category_to_run}] {metric_to_analyze}が急上昇中のユーザーリスト")
    st.write(f"成長率が **{min_growth_rate:.2f}** 以上のユーザーが **{len(high_growth_users)}** 人見つかりました。")
    st.dataframe(high_growth_users, use_container_width=True)

    st.markdown("---")
    st.header("👤 個別ユーザーのエンゲージメント推移")
    st.write("上のリストから詳細を見たいユーザーを一人選択してください。")

    user_options = high_growth_users['username'].tolist()
    selected_user_detail = st.selectbox(
        "ユーザーを選択 (任意):",
        options=['（選択しない）'] + user_options
    )

    if selected_user_detail != '（選択しない）':
        df_detail = df_posts[df_posts['username'] == selected_user_detail]
        
        fig = px.line(
            df_detail,
            x='datetime',
            y=['likes', 'comments'],
            title=f'{selected_user_detail} の「いいね」と「コメント」の推移',
            labels={'datetime': '投稿日時', 'value': '数', 'variable': '指標'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 サイドバーから分析したいカテゴリを選択し、「分析を開始」ボタンを押してください。")

