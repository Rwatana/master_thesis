import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="投稿分析", layout="wide")

# --- データ読み込み関数（キャッシュを利用） ---

@st.cache_data
def load_influencer_data(filepath):
    """
    influencers.txtを読み込む。
    1行目をヘッダーとし,2行目の区切り線はスキップする。
    """
    try:
        df = pd.read_csv(filepath, sep='\t', skiprows=[1])
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_post_data(filepath):
    """指定されたJSONファイルを読み込む"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def get_posts_for_influencer(username, info_dir):
    """
    指定されたインフルエンサーの全投稿データを収集し,時系列でソートして返す。
    """
    try:
        all_files = os.listdir(info_dir)
        user_post_files = [f for f in all_files if f.startswith(f"{username}-")]
    except FileNotFoundError:
        st.error(f"投稿データディレクトリ `{info_dir}` が見つかりません。")
        return []

    if not user_post_files:
        st.warning(f"{username} の投稿データが見つかりませんでした。")
        return []

    posts_data = []
    for filename in user_post_files:
        data = load_post_data(os.path.join(info_dir, filename))
        if data:
            posts_data.append(data)
    
    posts_data.sort(key=lambda x: x.get('taken_at_timestamp', 0), reverse=True)
    return posts_data

# --- UI描画 ---

st.title("📝 投稿分析")
st.write("サイドバーでインフルエンサーを選択すると,その人の全投稿を時系列で表示します。")

# サイドバーでのインフルエンサー選択
st.sidebar.header("インフルエンサー選択")
df_influencers = load_influencer_data('influencers.txt')

if df_influencers is None:
    st.error("`influencers.txt` が見つかりません。")
    st.stop()

influencer_list = sorted(df_influencers['Username'].unique())
selected_influencer = st.sidebar.selectbox(
    '分析したいインフルエンサーを選択:',
    options=influencer_list
)

# メイン画面での投稿表示
st.markdown("---")
st.header(f"👤 {selected_influencer} の投稿履歴")

info_dir = 'posts_info/unzipped_data_7z/info/'

with st.spinner(f'{selected_influencer}の投稿データを読み込んでいます...'):
    posts = get_posts_for_influencer(selected_influencer, info_dir)

if posts:
    st.write(f"合計 {len(posts)} 件の投稿が見つかりました。")
    # --- 投稿をカード形式で表示 ---
    for post in posts:
        timestamp = post.get('taken_at_timestamp', 0)
        post_date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')
        likes = post.get('edge_media_preview_like', {}).get('count', 0)
        comments = post.get('edge_media_to_parent_comment', {}).get('count', 0)
        caption_edges = post.get('edge_media_to_caption', {}).get('edges', [])
        caption = caption_edges[0]['node']['text'] if caption_edges else "（キャプションなし）"
        display_url = post.get('display_url', '')

        with st.expander(f"📅 **{post_date}** |  👍 {likes:,} いいね  |  💬 {comments:,} コメント"):
            col1, col2 = st.columns([1, 2])
            with col1:
                if display_url:
                    st.image(display_url, use_container_width=True)
            with col2:
                st.markdown("**キャプション:**")
                st.text_area(f"caption_{post.get('id', timestamp)}", caption, height=150, disabled=True, label_visibility="collapsed")
    
    # ▼▼▼ ここからグラフ作成のコードを追加 ▼▼▼
    st.markdown("---")
    st.header("📈 エンゲージメント数の時系列推移")

    # グラフ用にデータを整形
    chart_data = []
    for post in posts:
        timestamp = post.get('taken_at_timestamp', 0)
        likes = post.get('edge_media_preview_like', {}).get('count', 0)
        comments = post.get('edge_media_to_parent_comment', {}).get('count', 0)
        chart_data.append({
            'date': datetime.fromtimestamp(timestamp),
            'Likes': likes,
            'Comments': comments,
            'Total': likes + comments
        })
    
    if chart_data:
        # Pandas DataFrameに変換
        df_chart = pd.DataFrame(chart_data)
        
        # データをプロット
        fig = px.line(
            df_chart,
            x='date',
            y=['Likes', 'Comments', 'Total'],
            title=f'{selected_influencer}のエンゲージメント推移',
            labels={'date': '投稿日', 'value': '数', 'variable': '指標'}
        )
        
        # グラフの線を太くするなどの調整
        fig.update_traces(mode='lines+markers')
        
        st.plotly_chart(fig, use_container_width=True)
    # ▲▲▲ ここまでが追加部分 ▲▲▲