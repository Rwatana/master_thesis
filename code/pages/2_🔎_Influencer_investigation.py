import streamlit as st
import pandas as pd

st.set_page_config(page_title="インフルエンサー分析", layout="wide")

st.title("🔎 インフルエンサー分析")
st.write("サイドバーでインフルエンサーを選択してください。カテゴリでの絞り込みも可能です。")

# --- データ読み込み（キャッシュを利用） ---
@st.cache_data
def load_data(filepath):
    """influencers.txtを読み込む関数"""
    try:
        # 1行目がヘッダー、タブ区切りで読み込み
        df = pd.read_csv(filepath, sep='\t')
        return df
    except FileNotFoundError:
        return None

df_influencers = load_data('influencers.txt')

# --- サイドバー ---
st.sidebar.header("絞り込みと選択")

if df_influencers is not None:
    # 1. カテゴリによる絞り込み（任意）
    all_categories = sorted(df_influencers['Category'].unique())
    selected_categories = st.sidebar.multiselect(
        'カテゴリで絞り込む (任意):',
        options=all_categories,
        default=[]  # デフォルトは空（何も選択しない）
    )

    # 絞り込み用のDataFrameを準備
    if selected_categories:
        # カテゴリが選択された場合、データフレームをフィルタリング
        filtered_df = df_influencers[df_influencers['Category'].isin(selected_categories)]
    else:
        # 何も選択されていない場合、全データを対象とする
        filtered_df = df_influencers

    # 2. インフルエンサーの選択
    # 上で準備した(フィルタリング済みまたは全量の)DFから名前のリストを作成
    influencer_list = sorted(filtered_df['Username'].unique())
    selected_influencer = st.sidebar.selectbox(
        'インフルエンサーを選択:',
        options=influencer_list
    )

    # --- メイン画面 ---
    st.markdown("---")

    if selected_influencer:
        st.header(f"👤 {selected_influencer} の詳細情報")
        # 選択されたインフルエンサーのデータを元のDFから取得
        influencer_data = df_influencers[df_influencers['Username'] == selected_influencer].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("フォロワー数", f"{int(influencer_data['#Followers']):,}")
        col2.metric("フォロー数", f"{int(influencer_data['#Followees']):,}")
        col3.metric("投稿数", f"{int(influencer_data['#Posts']):,}")

        st.subheader("カテゴリ")
        st.info(f"**{influencer_data['Category']}**")
        
        st.subheader("プロフィール")
        st.markdown(f"[{selected_influencer}のInstagramプロフィールへ](https://www.instagram.com/{selected_influencer}/)", unsafe_allow_html=True)
    
    else:
        st.info("サイドバーでインフルエンサーを選択してください。")

else:
    st.error("`influencers.txt` が見つかりません。プロジェクトのルートディレクトリに配置してください。")