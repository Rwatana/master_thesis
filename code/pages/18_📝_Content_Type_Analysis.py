import streamlit as st
import pandas as pd
import os
import json
from scipy.stats import ttest_ind
import plotly.express as px
import numpy as np

st.set_page_config(page_title="仮説検証", layout="wide")

# --- データ読み込み・分類関数 ---
@st.cache_data
def load_and_classify_posts(info_dir):
    """
    全投稿の.infoファイルを読み込み、キャプション内容に基づいてコンテンツタイプを分類する
    """
    all_post_details = []
    try:
        all_files = [f for f in os.listdir(info_dir) if f.endswith('.info')]
    except FileNotFoundError:
        st.error(f"投稿データディレクトリ '{info_dir}' が見つかりません。")
        return pd.DataFrame()

    progress_bar = st.progress(0, text="全投稿のキャプションを解析中...")
    
    # 情報系コンテンツを判定するためのキーワード
    informative_keywords = [
        'レビュー', 'review', '比較', 'レポ', 'repo', '使い方', 'howto',
        'コスメ紹介', '解説', 'まとめ', 'スウォッチ', 'swatch', 'tips', '方法'
    ]
    regex_pattern = '|'.join(informative_keywords)

    for i, filename in enumerate(all_files):
        try:
            with open(os.path.join(info_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)

            caption_edges = data.get('edge_media_to_caption', {}).get('edges', [])
            caption = caption_edges[0]['node']['text'] if caption_edges else ""
            
            # キャプションにキーワードが含まれているかで分類
            content_type = '情報系コンテンツ' if pd.Series(caption).str.contains(regex_pattern, case=False, na=False).any() else '一般コンテンツ'
            
            likes = data.get('edge_media_preview_like', {}).get('count', 0)
            comments = data.get('edge_media_to_parent_comment', {}).get('count', 0)

            all_post_details.append({
                'username': data.get('owner', {}).get('username', ''),
                'likes': likes,
                'comments': comments,
                'engagement': likes + comments,
                'content_type': content_type
            })
        except (json.JSONDecodeError, KeyError):
            continue
        
        progress_bar.progress((i + 1) / len(all_files), text=f"解析中: {filename}")
    
    progress_bar.empty()
    return pd.DataFrame(all_post_details)

# --- UI描画 ---
st.title("🔬 仮説検証：コンテンツタイプとエンゲージメント")
st.header("論文の仮説：有益なコンテンツはエンゲージメントを高めるか？")
st.info("""
論文のメタ分析では、**「有益で機能的な (Informative and functional)」**コンテンツが、エンゲージメントと強い正の相関 ($\rho = 0.33$) を持つことが示唆されました。

ここでは、投稿の**キャプション**に含まれる単語を手がかりに、コンテンツを以下の2種類に分類し、エンゲージメントに統計的な差があるかを検証します。

- **情報系コンテンツ**: 「#コスメレビュー」「#使い方」など、知識やTIPSを提供する投稿。
- **一般コンテンツ**: 上記以外の投稿。
""")

info_dir = 'posts_info/unzipped_data_7z/info/'
df = load_and_classify_posts(info_dir)

if df.empty:
    st.error("分析対象の投稿データが見つかりませんでした。")
    st.stop()

st.markdown("---")
st.header("分析結果")

# コンテンツタイプごとの投稿数を表示
st.subheader("コンテンツタイプの投稿数")
type_counts = df['content_type'].value_counts()
st.bar_chart(type_counts)

# コンテンツタイプごとのエンゲージメント分布を比較
st.subheader("コンテンツタイプ別のエンゲージメント分布")
fig = px.box(
    df,
    x='content_type',
    y='engagement',
    color='content_type',
    title='情報系コンテンツ vs 一般コンテンツのエンゲージメント',
    labels={'content_type': 'コンテンツタイプ', 'engagement': 'エンゲージメント数（いいね+コメント）'},
    notched=True,
    log_y=True # 外れ値が多いため対数軸で表示
)
st.plotly_chart(fig, use_container_width=True)
st.markdown("※ 箱の中の線が中央値、箱の上下が25/75パーセンタイル、ひげが外れ値を除いた範囲を示します。")

# 統計的な検定
st.subheader("統計的有意差の検定")
informative_engagement = df[df['content_type'] == '情報系コンテンツ']['engagement'].dropna()
general_engagement = df[df['content_type'] == '一般コンテンツ']['engagement'].dropna()

if len(informative_engagement) > 1 and len(general_engagement) > 1:
    # t検定の実施
    t_stat, p_value = ttest_ind(informative_engagement, general_engagement, equal_var=False) # Welch's t-test

    avg_info = informative_engagement.mean()
    avg_general = general_engagement.mean()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("情報系投稿の平均エンゲージメント", f"{avg_info:,.0f}")
    with col2:
        st.metric("一般投稿の平均エンゲージメント", f"{avg_general:,.0f}")
    with col3:
        st.metric("p値", f"{p_value:.4f}", help="p値が0.05未満の場合、2つのグループの平均値の差は統計的に有意である（偶然とは考えにくい）と判断されます。")
else:
    p_value = 1.0 # 比較できない場合はp値を1に設定
    avg_info = 0
    avg_general = 0
    st.warning("両方のコンテンツタイプのデータが不足しているため、統計検定を実行できませんでした。")


st.header("結論と考察")
if p_value < 0.05 and avg_info > avg_general:
    st.success(f"""
    **検証結果: 論文の仮説は強く支持されました。**

    このデータセットにおいて、「情報系コンテンツ」は「一般コンテンツ」よりも**統計的に有意に高いエンゲージメント**を獲得していることが確認できました（平均エンゲージメント差: {avg_info - avg_general:,.0f}）。

    これは、単に美しい写真や日常を投稿するだけでなく、フォロワーにとって**「役に立つ」情報**を提供することが、エンゲージメントを高めるための非常に有効な戦略であることを示唆しています。
    """)
else:
    st.warning(f"""
    **検証結果: 論文の仮説は明確には支持されませんでした。**

    {'平均エンゲージメントに差は見られますが、' if abs(avg_info - avg_general) > 100 else ''}p値が {p_value:.4f} であり、統計的に有意な差があるとまでは言えませんでした。
    このデータセットでは、コンテンツのタイプ（情報系か否か）がエンゲージメントに与える影響は限定的であるか、他の要因がより強く作用している可能性が考えられます。
    """)