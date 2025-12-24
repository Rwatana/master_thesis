# paperize_xai_outputs.py の使い方

## 1) 目的
- `xai_export_out/` にある MaskOpt 出力を集計して、論文に貼れる図（PNG/PDF）と表（CSV/LaTeX）を生成します。
- NeighborName が `nbr_12345` のようなプレースホルダになっている場合、`NeighborGlobalIdx` を使って実名に復元できます。

## 2) 実行コマンド

### (A) 実名復元なし（速い）
```
python paperize_xai_outputs.py --xai-dir xai_export_out --no-recover-names
```

### (B) 実名復元あり（重い）
投稿/hashtags/mentions の元ファイルを参照して node_to_idx を再構築します（Dec 2017 に投稿したユーザでフィルタします）。

```
python paperize_xai_outputs.py \
  --xai-dir xai_export_out \
  --posts-file dataset_A_active_all.csv \
  --hashtags-file hashtags_2017.csv \
  --mentions-file mentions_2017.csv
```

## 3) 出力
`xai_export_out/paper/` に以下が生成されます。

- `agg_features_by_pos_named.csv`（pos 別特徴量集計）
- `agg_neighbors_by_pos_named.csv`（pos 別近傍ノード集計）
- `summary_named.csv`
- `node_idmap_used.csv`（復元した index→name の対応; 復元時のみ）
- `tables/`（LaTeX テーブル）
- `figs/`（PNG/PDF 図）
- `figure_captions_ja.md`（図キャプションの叩き台）

## 4) 図の使い方（論文）
- PNG はスライド向き、PDF は論文向きです。
- `figure_captions_ja.md` のキャプション案を、実験設定に合わせて微修正して使ってください。
