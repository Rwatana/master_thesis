import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="ハッシュタグライフサイクル分析", layout="wide")

# --- データ読み込み関数 ---
@st.cache_data
def load_hashtag_data(filepath):
    """ハッシュタグファイルをヘッダー付きで正しく読み込む"""
    try:
        df = pd.read_csv(filepath, header=0)
        df.rename(columns={'source': 'username', 'target': 'hashtag'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df.drop(columns=['timestamp'])
    except FileNotFoundError:
        st.error(f"ファイル '{filepath}' が見つかりません。")
        return None

@st.cache_data
def get_hashtag_ranks(_df_hashtags):
    """全ハッシュタグの総使用回数と順位を計算する"""
    if _df_hashtags is None:
        return pd.DataFrame()
    hashtag_total_counts = _df_hashtags['hashtag'].value_counts().reset_index()
    hashtag_total_counts.columns = ['hashtag', 'total_usage']
    hashtag_total_counts['rank'] = hashtag_total_counts.index + 1
    return hashtag_total_counts

@st.cache_data
def precompute_all_ranks(_df_hashtags):
    """
    全ハッシュタグ利用履歴から,各ユーザーの初回利用が「何人目の使用者」で
    「全体で何回目の使用」なのかを事前に計算する。
    """
    # 1. 各ユーザーが各ハッシュタグを「初めて」使った投稿を抽出
    first_usages_df = _df_hashtags.loc[_df_hashtags.groupby(['username', 'hashtag'])['datetime'].idxmin()].copy()
    
    # 2. 「何人目の使用者か」を計算
    #    各ハッシュタグごとに,初回使用日時でランク付け
    first_usages_df['user_adoption_rank'] = first_usages_df.groupby('hashtag')['datetime'].rank(method='min').astype(int)

    # 3. 「全体で何回目の使用か」を計算
    #    まず,全ハッシュタグ利用履歴を日時でソート
    _df_hashtags_sorted = _df_hashtags.sort_values('datetime')
    #    次に,各ハッシュタグごとに累積カウント（これが全体での使用回数の通し番号になる）
    _df_hashtags_sorted['global_usage_rank'] = _df_hashtags_sorted.groupby('hashtag').cumcount() + 1
    
    # 4. 初回利用時の「全体での使用回数」をマージ
    #    ユニークなキー（user, hashtag, datetime）で結合
    first_usages_with_global_rank = pd.merge(
        first_usages_df,
        _df_hashtags_sorted[['username', 'hashtag', 'datetime', 'global_usage_rank']],
        on=['username', 'hashtag', 'datetime'],
        how='left'
    )
    
    return first_usages_with_global_rank[['username', 'hashtag', 'datetime', 'user_adoption_rank', 'global_usage_rank']]


@st.cache_data
def create_user_hashtag_summary(username, _df_hashtags, _hashtag_rank_df, _df_all_ranks):
    """ユーザーが使用したハッシュタグのサマリーテーブルを作成する"""
    # 事前計算されたランク情報から,選択されたユーザーのデータを抽出
    user_ranks_df = _df_all_ranks[_df_all_ranks['username'] == username].copy()
    if user_ranks_df.empty:
        return pd.DataFrame()

    # ユーザー個人のハッシュタグ使用回数を計算
    user_counts = _df_hashtags[_df_hashtags['username'] == username]['hashtag'].value_counts().reset_index()
    user_counts.columns = ['hashtag', 'user_usage_count']
    
    # 必要なデータを全て結合
    summary = pd.merge(user_ranks_df, user_counts, on='hashtag', how='left')
    summary = pd.merge(summary, _hashtag_rank_df, on='hashtag', how='left')
    
    summary.rename(columns={
        'hashtag': 'ハッシュタグ', 
        'user_usage_count': '本人の使用回数',
        'datetime': '初回使用日時', 
        'user_adoption_rank': '何人目の使用者か',
        'global_usage_rank': '全体で何回目の使用か',
        'total_usage': '全体の総使用回数', 
        'rank': '全体の人気順位'
    }, inplace=True)
    
    # 表示する列を選択・並び替え
    final_cols = [
        'ハッシュタグ', '本人の使用回数', '初回使用日時', '何人目の使用者か',
        '全体で何回目の使用か', '全体の総使用回数', '全体の人気順位'
    ]
    return summary[final_cols].sort_values('初回使用日時')

# --- UI描画 ---
st.title("📈 ハッシュタグライフサイクル分析")
st.info("特定のハッシュタグの全体的な流行と,選択したインフルエンサーがそれを使用したタイミングを比較分析します。")

# --- データの読み込み ---
df_hashtags = load_hashtag_data('output_hashtags_all_parallel.csv')
if df_hashtags is None:
    st.stop()

# 事前計算の実行
hashtag_rank_df = get_hashtag_ranks(df_hashtags)
df_all_ranks = precompute_all_ranks(df_hashtags)
user_list = sorted(df_hashtags['username'].unique())

# --- サイドバー ---
st.sidebar.header("分析対象の選択")
selected_user = st.sidebar.selectbox("1. 分析したいユーザーを選択:", options=user_list)

# --- 分析開始ボタンと状態管理 ---
if 'run_hashtag_analysis' not in st.session_state:
    st.session_state.run_hashtag_analysis = False
if 'analyzed_user_hashtag' not in st.session_state:
    st.session_state.analyzed_user_hashtag = ""

if st.sidebar.button("分析を開始"):
    st.session_state.run_hashtag_analysis = True
    st.session_state.analyzed_user_hashtag = selected_user
elif selected_user != st.session_state.analyzed_user_hashtag:
    st.session_state.run_hashtag_analysis = False

# --- メイン画面 ---
if st.session_state.run_hashtag_analysis:
    user = st.session_state.analyzed_user_hashtag
    
    with st.spinner(f"ユーザー '{user}' のハッシュタグ利用履歴を集計中..."):
        user_summary_df = create_user_hashtag_summary(user, df_hashtags, hashtag_rank_df, df_all_ranks)
    
    if user_summary_df.empty:
        st.warning(f"ユーザー '{user}' が使用したハッシュタグのデータがありません。")
        st.stop()

    st.header(f"分析結果: {user}")
    st.subheader("使用ハッシュタグのサマリー")
    st.dataframe(user_summary_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("ハッシュタグのトレンドと比較")
    
    selected_hashtag = st.selectbox(
        "グラフで分析したいハッシュタグを上の表から選択してください:",
        options=user_summary_df['ハッシュタグ'].unique()
    )

    if selected_hashtag:
        with st.spinner(f"ハッシュタグ '{selected_hashtag}' のトレンドを分析中..."):
            hashtag_info = user_summary_df[user_summary_df['ハッシュタグ'] == selected_hashtag].iloc[0]
            col1, col2 = st.columns(2)
            col1.metric("全体の総使用回数", f"{int(hashtag_info['全体の総使用回数']):,} 回")
            col2.metric("全体の人気順位", f"{int(hashtag_info['全体の人気順位'])} 位")

            # 全体的な使用回数の推移（週ごと）
            hashtag_trend = df_hashtags[df_hashtags['hashtag'] == selected_hashtag].set_index('datetime').resample('W').size().rename('weekly_usage')
            
            # ユーザーが使用したタイミング
            user_usage_points = df_hashtags[(df_hashtags['username'] == user) & (df_hashtags['hashtag'] == selected_hashtag)]

            fig = px.line(
                hashtag_trend,
                title=f"ハッシュタグ '{selected_hashtag}' の全体的な流行推移と {user} の使用タイミング",
                labels={'datetime': '日付', 'value': '週間使用回数'}
            )
            
            fig.add_trace(go.Scatter(
                x=user_usage_points['datetime'],
                y=[hashtag_trend.max() * 0.95] * len(user_usage_points),
                mode='markers',
                marker=dict(symbol='star', color='red', size=10),
                name=f'{user} の使用タイミング',
                hovertext=user_usage_points['datetime'].dt.strftime('%Y-%m-%d'),
                hoverinfo='text'
            ))
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 サイドバーで分析したいユーザーを選択し,「分析を開始」ボタンを押してください。")

