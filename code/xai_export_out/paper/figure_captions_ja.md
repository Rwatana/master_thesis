# 図キャプション案（日本語）

以下は、そのまま論文に貼れるようにしたキャプションの叩き台です。

## Fig.01 Fidelity
- MaskOpt 最適化後の fidelity (maskopt_best_fid) の分布。閾値（例: 1e-6）より小さい実行が多いほど、マスク付き予測が元予測を良く保持している。

## Fig.02 Top features (pos別)
- ExplainPos ごとに、平均 |score impact| が大きい特徴量を上位 K 個可視化。モデル予測に寄与する主要因（プロファイル系 / 投稿間隔 / 投稿量など）を示す。

## Fig.03 Top neighbor nodes (pos別)
- ExplainPos ごとに、近傍ノード（エッジグループ=neighbor）を上位 K 個可視化。構造寄与の代表例（特定ハッシュタグ/メンション/ユーザとの関係など）を示す。

## Fig.04 Feature heatmap
- 上位特徴量の寄与を月位置 [0, 1, 2] に対してヒートマップ化。寄与の時間変化（最近効き始めた/継続的に重要）を視覚化する。

## Fig.05 Scatter (importance vs impact)
- ゲート値（importance）と |score impact| の関係を散布図で示す。両者の相関が高いほど、importance が予測スコア変化を良く反映していると解釈できる（ただし非線形・飽和などで相関が低くなる場合もある）。

## Fig.06 Total feature vs neighbor impact
- ExplainPos ごとに、特徴量側の総 |impact| と近傍ノード側の総 |impact| を比較。予測根拠が『特徴量主導』か『構造主導』かを議論する材料。

## Fig.07 Static vs dynamic share
- 静的特徴（followers/followees/posts_history + cat_/type_）と動的特徴（投稿・コメント・間隔など）で、総寄与の比率を比較。『過去プロフィールが強い』などの考察に対応。