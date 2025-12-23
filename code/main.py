import streamlit as st

st.set_page_config(
    page_title="Instagramインフルエンサーデータセット概要",
    page_icon="📝",
    layout="wide"
)

st.title("📝 Instagramインフルエンサー データセット概要")
st.markdown("---")

st.write(
    "このアプリは、Instagramインフルエンサーのデータセットを理解し、可視化するためのツールです。"
    "サイドバーから **Data Viewer** を選択すると、実際のデータファイルの中身を確認できます。"
)
st.sidebar.success("ページを選択してください。")


st.header("1. influencers.txt")
st.info(
    """
    各インフルエンサーのInstagramユーザー名、カテゴリ、フォロワー数、フォロー数、投稿数が含まれています。
    - **主要カテゴリ**: ビューティー, ファミリー, ファッション, フィットネス, フード, インテリア, ペット, トラベル
    - 上記8つ以外は **'Other'** として分類されています。
    """
)


st.header("2. 投稿のメタデータ (JSONファイル)")
st.info(
    """
    `posts_info/unzipped_data_7z/info/` ディレクトリ内に格納されています。
    - 各JSONファイルには、キャプション、いいね数、コメント、タイムスタンプ、ユーザータグ等の詳細な投稿データが含まれます。
    - ファイル名は `ユーザー名-投稿ID.info` の形式になっています。
    """
)


st.header("3. JSON-Image_files_mapping.txt")
st.info(
    """
    投稿メタデータ（JSONファイル）と、それに対応する画像ファイル（JPG）を紐付けるためのマッピングファイルです。
    - 1つの投稿に複数の画像が含まれる場合があるため、このファイルで対応関係を確認します。
    """
)

st.header("4. その他のファイル")
st.info(
    """
    - **投稿の画像 (JPGファイル)**: `posts_image` ディレクトリに格納されています。
    - **sample_images.zip**: データセット内のファミリーカテゴリのインフルエンサー1人分のサンプル画像です。
    """
)