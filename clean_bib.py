from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase


import bibtexparser
import re


def normalize_title(title):
    """
    タイトルの重複判定を正確にするための正規化。
    小文字化,記号の除去,余分なスペースの削除を行います。
    """
    if not title:
        return ""
    # {} などのBibTeX特有の記号を削除
    title = title.replace("{", "").replace("}", "")
    # 英数字以外をスペースに変換し,小文字化
    title = re.sub(r'[^a-zA-Z0-9]', ' ', title).lower()
    # 連続するスペースを一つにまとめ,前後の空白を削除
    return " ".join(title.split())


def remove_duplicates(input_file, output_file):
    # 1. BibTeXファイルの読み込み
    with open(input_file, 'r', encoding='utf-8') as bibfile:
        parser = bibtexparser.bparser.BibTexParser(common_strings=True)
        bib_database = bibtexparser.load(bibfile, parser=parser)

    entries = bib_database.entries
    unique_entries = {}
    duplicate_count = 0

    for entry in entries:
        # タイトルとDOIを取得（存在しない場合はNone）
        title = entry.get('title')
        doi = entry.get('doi')
        
        # 判定用のキーを作成（DOIがあればDOI,なければ正規化タイトル）
        norm_title = normalize_title(title)
        identity_key = doi if doi else norm_title

        if identity_key in unique_entries:
            # すでに存在するエントリと比較し,情報量が多い方（フィールド数が多い方）を残す
            existing_entry = unique_entries[identity_key]
            if len(entry) > len(existing_entry):
                unique_entries[identity_key] = entry
            duplicate_count += 1
            print(f"Duplicate found: {title[:50]}...")
        else:
            unique_entries[identity_key] = entry

    # 2. 新しいBibDatabaseの作成
    new_db = BibDatabase()
    new_db.entries = list(unique_entries.values())

    # 3. ファイルへの書き出し
    writer = BibTexWriter()
    # 出力のフォーマット（インデントなど）を整える設定
    writer.indent = '  '
    with open(output_file, 'w', encoding='utf-8') as bibfile:
        bibfile.write(writer.write(new_db))

    print("-" * 30)
    print(f"Done! Processed {len(entries)} entries.")
    print(f"Removed {duplicate_count} duplicates.")
    print(f"Cleaned file saved as: {output_file}")


if __name__ == "__main__":
    # ファイル名は環境に合わせて変更してください
    INPUT_FILENAME = 'papers.bib'
    OUTPUT_FILENAME = 'papers.bib'
    
    try:
        remove_duplicates(INPUT_FILENAME, OUTPUT_FILENAME)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILENAME} not found.")