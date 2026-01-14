import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="投稿頻度の一貫性分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_processed_data(filepath):
    """事前処理済みのCSVを読み込む"""
    try:
        df = pd.read_csv(filepath, parse_dates=['datetime'])
        return df
    except FileNotFoundError:
        st.error(f"'{filepath}'が見つかりません。先に `preprocess_data.py` を実行してください。")
        return None

# --- UI描画 ---
st.title("📅 投稿頻度の一貫性分析")
st.info("インフルエンサーの投稿頻度が時間と共にどう変化したかを,移動平均を用いて可視化します。")

# --- データの準備 ---
df_posts = load_processed_data('preprocessed_posts_with_metadata.csv')
if df_posts is None:
    st.stop()

# --- サイドバー ---
st.sidebar.header("分析対象の選択")
user_list = sorted(df_posts['username'].unique())
selected_user = st.sidebar.selectbox("分析したいユーザーを選択:", options=user_list)

window_size = st.sidebar.slider("移動平均のウィンドウサイズ（日数）:", min_value=7, max_value=180, value=30)

# --- メイン画面 ---
st.header(f"📈 {selected_user} の投稿頻度推移")

# 1. 選択されたユーザーのデータを抽出
user_posts_df = df_posts[df_posts['username'] == selected_user].copy()

if user_posts_df.empty:
    st.warning("このユーザーの投稿データはありません。")
else:
    # 2. 日ごとの投稿数を集計
    user_posts_df.set_index('datetime', inplace=True)
    # 1日ごとにリサンプリングし,投稿数をカウント
    daily_post_counts = user_posts_df.resample('D').size().rename('daily_posts')

    # 3. 移動平均を計算
    # rolling()で指定した日数分のデータをウィンドウとし,その平均を計算
    rolling_avg_posts = daily_post_counts.rolling(window=f'{window_size}D').mean()
    rolling_avg_posts = rolling_avg_posts.reset_index() # プロットのためにインデックスを列に戻す

    # 4. グラフを描画
    fig = px.line(
        rolling_avg_posts,
        x='datetime',
        y='daily_posts',
        title=f'{selected_user} の投稿頻度 ({window_size}日移動平均)',
        labels={'datetime': '日付', 'daily_posts': '1日あたりの平均投稿数'}
    )
    
    # グラフのY軸の範囲を調整
    fig.update_yaxes(rangemode='tozero')
    
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"上のグラフは,各時点において**過去{window_size}日間**の「1日あたりの平均投稿数」を示しています。")
    st.info("""
    **分析のポイント**:
    - **グラフが安定している**: 一貫したペースで投稿を続けていることを示します。
    - **グラフが急上昇している**: 特定の期間に集中的に投稿している（例: キャンペーン期間など）ことを示します。
    - **グラフが下降している**: 投稿のペースが落ちている,あるいは活動が休止気味であることを示唆します。
    """)
