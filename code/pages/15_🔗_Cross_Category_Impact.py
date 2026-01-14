import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="越境影響分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_mention_data(filepath):
    """メンションファイルを読み込む"""
    try:
        df = pd.read_csv(filepath, header=0, names=['username', 'mention', 'timestamp'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.drop(columns=['timestamp'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
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
st.title("🔗 越境影響分析")
st.info("個人のいいね数推移と,**他分野のインフルエンサーからメンションされたタイミング**との相関を分析します。")

# --- データの準備 (軽量なファイルのみ先に読み込む) ---
df_mentions = load_mention_data('output_mentions_all_parallel.csv')
df_influencers = load_influencer_data('influencers.txt')

if df_mentions is None or df_influencers is None:
    st.warning("分析に必要な基本データファイルが読み込めませんでした。")
    st.stop()

# --- サイドバー ---
st.sidebar.header("分析対象の選択")

try:
    user_list = sorted([f.replace('.csv', '') for f in os.listdir('user_data') if f.endswith('.csv')])
except FileNotFoundError:
    st.error("`user_data` ディレクトリが見つかりません。先に `aggregate_user_data.py` を実行してください。")
    user_list = []

if not user_list:
    st.warning("分析対象のユーザーがいません。")
    st.stop()

selected_user = st.sidebar.selectbox("1. 分析したいユーザーを選択:", options=user_list)

# --- 分析開始ボタンと状態管理 ---
if 'run_cross_analysis' not in st.session_state:
    st.session_state.run_cross_analysis = False
if 'analyzed_user_cross' not in st.session_state:
    st.session_state.analyzed_user_cross = ""

if st.sidebar.button("分析を開始"):
    st.session_state.run_cross_analysis = True
    st.session_state.analyzed_user_cross = selected_user
# ユーザーが変更されたら分析状態をリセット
elif selected_user != st.session_state.analyzed_user_cross:
    st.session_state.run_cross_analysis = False


# --- メイン画面 ---
if st.session_state.run_cross_analysis:
    user_to_analyze = st.session_state.analyzed_user_cross
    
    with st.spinner(f"'{user_to_analyze}' の投稿データとメンションを分析中..."):
        # ボタンが押されてから,ユーザー固有のファイルを読み込む
        user_posts_df = load_user_post_data(user_to_analyze)

        if user_posts_df is None:
            st.stop() # エラーメッセージは関数内で表示

        user_info = df_influencers[df_influencers['Username'] == user_to_analyze]
        if user_info.empty:
            st.warning(f"{user_to_analyze}のカテゴリ情報が`influencers.txt`に見つかりませんでした。")
            st.stop()
        user_category = user_info['Category'].iloc[0]

        st.header(f"📈 {user_to_analyze} (カテゴリ: {user_category}) の分析結果")

        mentions_to_user = df_mentions[df_mentions['mention'] == user_to_analyze].copy()
        
        if not mentions_to_user.empty:
            mentions_with_category = pd.merge(
                mentions_to_user,
                df_influencers[['Username', 'Category']],
                left_on='username',
                right_on='Username',
                how='left'
            ).rename(columns={'Category': 'mentioner_category'})

            cross_category_mentions = mentions_with_category[
                mentions_with_category['mentioner_category'] != user_category
            ].dropna(subset=['mentioner_category'])
        else:
            cross_category_mentions = pd.DataFrame()

    # --- グラフ描画 ---
    fig = px.line(
        user_posts_df,
        x='datetime',
        y='likes',
        title=f'「いいね数」の推移と他分野からのメンションイベント',
        markers=True,
        labels={'datetime': '日付', 'likes': 'いいね数'}
    )

    if not cross_category_mentions.empty:
        y_position = user_posts_df['likes'].max() * 1.05 if not user_posts_df.empty else 1
        y_values = [y_position] * len(cross_category_mentions)
        
        fig.add_trace(
            go.Scatter(
                x=cross_category_mentions['datetime'],
                y=y_values,
                mode='markers',
                marker=dict(symbol='star', color='red', size=12),
                name='他分野からのメンション',
                hovertext=cross_category_mentions.apply(
                    lambda row: f"<b>メンション元:</b> {row['username']}<br><b>カテゴリ:</b> {row['mentioner_category']}",
                    axis=1
                ),
                hoverinfo='text'
            )
        )
        unique_mentioner_count = cross_category_mentions['username'].nunique()
        st.success(f"期間中に **{unique_mentioner_count}** 人の他分野インフルエンサーから,合計 **{len(cross_category_mentions)}** 回のメンションがありました。")
    else:
        st.info("期間中に他分野からのメンションはありませんでした。")

    st.plotly_chart(fig, use_container_width=True)

    # --- イベント詳細リスト ---
    if not cross_category_mentions.empty:
        st.markdown("---")
        st.subheader("異カテゴリ間メンションのペア詳細")
        st.write("グラフにプロットされた,他分野のインフルエンサーからのメンションの詳細リストです。")

        display_df = cross_category_mentions[['datetime', 'username', 'mentioner_category', 'mention']].copy()
        display_df.rename(columns={
            'datetime': '日時',
            'username': 'メンションしたユーザー',
            'mentioner_category': '相手のカテゴリ',
            'mention': 'メンションされたユーザー'
        }, inplace=True)
        
        st.dataframe(
            display_df.sort_values('日時', ascending=False),
            use_container_width=True
        )

else:
    # 初期表示メッセージ
    st.info("👈 サイドバーで分析したいユーザーを選択し,「分析を開始」ボタンを押してください。")
