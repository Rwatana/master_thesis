了解。あなたのコードでやっている

- **(1) mask の最適化（feat_gate / edge_gate を学習する部分）**
- **(2) “貢献度スコア”の算出（Importance と Score_Impact と Direction）**

を,コードの実装に沿って “数式＋処理手順” で詳しく整理します。

---

## 1. Mask 最適化で何を学習しているか

### 学習パラメータ（最適化対象）

`maskopt_e2e_explain()` の中で,説明用に **2 種類の logits** を学習します。

- 特徴量マスク：`feat_logits ∈ R^F`
- エッジマスク：`edge_logits ∈ R^E`（incident のときは “対象ノードに接続するエッジ数” だけ）

これを sigmoid で 0〜1 の連続ゲートにします：

- `feat_gate = σ(feat_logits)` （形状 `[F]`）
- `edge_gate = σ(edge_logits)` （形状 `[E]`）

> **重要**：この gate 自体が “説明のマスク” で,学習により「予測を維持できる範囲で最小限に残す」方向に寄ります。

---

## 2. Gate をどうモデルに適用しているか（forward の中身）

説明対象は `explain_pos`（ある月 t\*）だけで,他の月はキャッシュで固定です。

### 2.1 時系列の扱い：explain_pos 以外は固定（cache）

`E2EMaskOptWrapper._prepare_cache()` で,t ≠ t\* の月は

- `cached_proj[t] = projection_layer(x_t)[target_node]`
- `cached_gcn[t]  = gcn(projection_layer(x_t), edge_index_t)[target_node]`

を保存しておき,mask 最適化中は **t\* の月だけ再計算**します。

---

### 2.2 特徴量マスクの適用（feat_mask_scope="target"）

`_apply_feature_gate()` で,`feat_mask_scope="target"` のときは

- 対象ノード行だけ：`x[target, :] ← x[target, :] ⊙ feat_gate`
- 他ノードはそのまま

という “対象ノードの特徴量だけ絞る” 形です。

（実装上は in-place を避けるために `x + sel*x*(gate-1)` の形でやっていますが意味は同じ）

---

### 2.3 エッジマスクの適用（edge_mask_scope="incident"）

`_make_edge_weight()` で edge_weight を作ります。

- エッジ全体は基本 `w = 1`
- incident のとき,対象ノードに接続しているエッジの index だけ

  - `w[incident_edges] = edge_gate`

そして GCN に `edge_weight=w` を渡してメッセージパッシングを弱めます。

> つまりエッジは「残す/消す」ではなく **重みで連続的に弱める（0〜1）** 方式。

---

### 2.4 エンドツーエンド推論（GCN→LSTM→Attn→MLP）

説明月 t\* だけ mask を入れて再計算した埋め込み列を作り,

- `seq_gcn = [gcn_out_t(target)]_{t=1..T}`
- `seq_raw = [proj_out_t(target)]_{t=1..T}`

を `HardResidualInfluencerModel.forward(seq_gcn, seq_raw)` に通し,最終スコア `pred_masked` を得ます。

---

## 3. Mask 最適化の目的関数（loss）の詳細

### 3.1 基本の忠実度（fidelity）

まず “元の予測” を固定目標にします：

- `orig = wrapper.original_pred()`（feat=1, edge=1 の予測）
- `pred = wrapper.predict_with_gates(feat_gate, edge_gate)`

忠実度損失は

[
L_\text{fid} = (pred - orig)^2
]

コードでは `fid_weight` を掛けています：

[
fid_weight \cdot L_\text{fid}
]

> **ここが最重要**で,これが強いほど「予測を変えない説明」を優先します。

---

### 3.2 マスクを小さくする正則化（size）

残す量を減らしたいので平均ゲートを罰します：

[
L_\text{feat-size} = mean(feat_gate)
]
[
L_\text{edge-size} = mean(edge_gate)
]

---

### 3.3 0/1 に寄せる正則化（entropy）

ゲートが 0.3 とか 0.6 の “曖昧” にならないよう二値エントロピーを罰します：

[
H(p) = -(p\log p + (1-p)\log(1-p))
]
[
L_\text{feat-ent} = mean(H(feat_gate))
]
[
L_\text{edge-ent} = mean(H(edge_gate))
]

---

### 3.4 （任意）コントラスト項（use_contrastive=True）

補集合マスク（1-gate）では予測が変わってほしい,という制約。

- `feat_gate_drop = 1 - feat_gate`
- `edge_gate_drop = 1 - edge_gate`
- `pred_drop = predict_with_gates(feat_gate_drop, edge_gate_drop)`
- `delta = |pred_drop - orig|`
- margin 未満ならペナルティ：

[
L_\text{contrast} = \max(0, margin - delta)
]

---

### 3.5 最終 loss（コードと対応）

[
L =
fid*weight \cdot L*\text{fid}

- contrastive*weight \cdot L*\text{contrast}
- c*\text{nf-size}L*\text{feat-size}
- c*\text{nf-ent}L*\text{feat-ent}
- c*\text{e-size}L*\text{edge-size}
- c*\text{e-ent}L*\text{edge-ent}
  ]

ここで `coeffs = {"node_feat_size":..., "node_feat_ent":..., "edge_size":..., "edge_ent":...}`。

---

## 4. “貢献度スコア”の算出方法（Importance / Score_Impact / Direction）

あなたの出力 CSV は基本この 3 つが肝です：

### 4.1 Importance（= 最適化された gate 値）

- Feature の Importance：`feat_gate[j]`
- Edge の Importance：`edge_gate[e]`（incident の一部）または full_edge_imp[e]

> これは **「その要素を残しておく必要度」**。
> SHAP みたいな “寄与の加算分解” ではなく,**最小説明集合のメンバー度**に近いです。

---

### 4.2 Score_Impact(orig-ablated)（= アブレーションで差分を測る）

ここが “符号付きの貢献度” を作っている部分です。

#### Feature の場合（1 特徴ずつ潰す）

まず “マスクなし予測” を基準にします：

- `pred_full_unmasked = predict_with_gates(ones_feat, ones_edge)`
  （feat=1, edge=1,= 元モデルの予測）

次に特徴 j を **baseline に置き換えた x_override** を作り：

- baseline `base[j]` は `baseline_scope` で決める

  - `full_graph_month`: その月の全ノード平均 or 中央値
  - `explain_subgraph`: サブグラフ内平均 or 中央値
  - `target_only`: 0 ベクトル

- `ablate_strength` で置換の強さを決める（1.0 なら完全置換）

[
x'*{target,j} = (1-s)\cdot x*{target,j} + s\cdot base_j \quad (s=ablate_strength)
]

これで

- `pred_abl = predict_with_gates(ones_feat, ones_edge, x_override=x_ab)`

そして差分：

[
diff = pred_full_unmasked - pred_abl
]

これが CSV の `Score_Impact(orig-ablated)`。

- `diff > 0`：その特徴を潰すとスコアが下がる → **押し上げ要因**
- `diff < 0`：その特徴を潰すとスコアが上がる → **押し下げ要因**

---

#### Edge の場合（1 エッジを落とす）

同様に

- `pred_full = predict_with_gates(ones_feat, ones_edge)`

次に対象エッジ (u,v) を “両方向まとめて” 落とす edge_weight_override を作り：

- `edge_w_ab = ones(E)`
- `edge_w_ab[both_dir(u,v)] = 0`

で

- `pred_abl = predict_with_gates(ones_feat, ones_edge, edge_weight_override=edge_w_ab)`

差分：

[
diff = pred_full - pred_abl
]

---

### 4.3 Direction（Zero/+/− の判定）

差分が小さすぎるとノイズなので “ゼロ扱い” をします。あなたの最新版では

[
thr = \max(\epsilon_{abs},\ \epsilon_{rel}\cdot|pred_full|)
]
[
|diff|\le thr \Rightarrow Zero(0)
]

- Feature: `eps_abs_feat`, `eps_rel_feat`
- Edge: `eps_abs_edge`, `eps_rel_edge`

それ以外は符号で `Positive / Negative` を決めます。

---

## 5. まとめ：この XAI のスコアの意味（誤解しやすい点）

- **Importance（gate）**：
  「予測を保つために残す必要があった度合い」
  → _最小説明集合_ の考え方（加法分解ではない）
- **Score_Impact（アブレーション差分）**：
  「その要素を baseline に寄せたら予測がどれだけ動くか」
  → _局所的な因果っぽい_ 指標（ただし baseline 依存）
- **Direction**：
  その差分の符号（しきい値付き）

---

## 6. 実務で効くチェックポイント（短く）

- `fid_weight` が大きすぎると **ゲートが全部 1 寄り**になりやすい（説明が薄い）
- `edge_size/edge_ent` が強すぎると **エッジが全部 0 寄り**になりやすい
- baseline_scope を変えると Score_Impact の符号・大きさが普通に変わる（特に full_graph_month は安定だが “平均との差” になる）

---

もし次を教えてくれたら,あなたのログ（feat_max, edge_max, fid_err 等）を前提に「どこをどういじれば説明が立つか」を具体設定で提案できます：

- いま出ている `feat_max / edge_max / fid` のログ数行（ep=1, 50, 100, 300）
- その月の `E`（incident エッジ数）と `num_hops`,`coeffs`,`fid_weight`
