# DeepLIFT: Learning Important Features Through Propagating Activation Differences

**著者**: Avanti Shrikumar, Peyton Greenside, Anshul Kundaje  
**公開**: PMLR 70 (ICML 2017)  
**DOI**: 10.48550/arXiv.1704.02685  
**リンク**: [arXiv:1704.02685](https://arxiv.org/abs/1704.02685)

DeepLIFT は,参照（baseline）入力 \(x_0\) に対する「差分」をネットワーク全体に逆伝播し,出力差分 \(\Delta f = f(x)-f(x_0)\) を入力差分の貢献度の総和として厳密に分解する手法です。勾配が 0 になる領域でも動作し,単一の逆伝播で計算できます。

---

### 1. 新規性：この論文の貢献と既存研究との差分

- **差分駆動の貢献度分解（Completeness）**: 出力差分を入力差分の寄与に完全分解（保存則）。
- **勾配不要・飽和頑健**: 非線形の飽和で勾配が 0 でも,差分を介して非ゼロ貢献を捕捉。
- **層別ルール（Rescale/RevealCancel）**: ReLU 等の非線形・相殺効果を扱う厳密な分配則を提示。
- **IG/SHAP との接続**: Path-based な IG の離散近似,SHAP（DeepSHAP）への橋渡し。

---

### 2. 理論/手法の核心：数式できっちり定式化

#### 2.1 記法と目的

- 参照（baseline）\(x_0\),実入力 \(x\),差分 \(\Delta x = x - x_0\)。
- 出力差分 \(\Delta f = f(x) - f(x_0)\)。
- 目標：各入力特徴 \(i\) の貢献度 \(C*{\Delta x_i \to \Delta f}\) を定め,
  \[
  \sum*{i} C\_{\Delta x_i \to \Delta f} \;=\; \Delta f
  \]
  を満たす（完全性）。

DeepLIFT は,各ニューロンの出力差分を親ニューロンの差分へ分配する形で,局所保存則を満たしつつ最終的な入力貢献に還元します。

#### 2.2 マルチプライヤと鎖則（DeepLIFT Chain Rule）

ニューロン \(t\) の出力を \(y*t\),参照時を \(y_t^0\),差分を \(\Delta y_t = y_t - y_t^0\) とする。親ニューロン \(p\) に対し,\(\Delta y_t\) への寄与を
\[
C*{\Delta y*p \to \Delta y_t} \,=\, m*{p \to t} \, \Delta y*p
\]
で定義する。ここで \(m*{p\to t}\) はマルチプライヤ。ネットワーク全体では経路上のマルチプライヤを積にして鎖則を満たす：
\[
m*{i \to f} \,=\, \prod*{\text{edges }(u\to v)} m*{u\to v}, \qquad C*{\Delta x*i \to \Delta f} \,=\, m*{i\to f} \, \Delta x_i.
\]
層ごとに以下のルールで \(m\) を定義する。

#### 2.3 線形結合と活性化の分解

- アフィン（前活性）: \(z*t = \sum_p w*{tp} y*p + b_t\)。参照 \(z_t^0 = \sum_p w*{tp} y*p^0 + b_t\)。差分 \(\Delta z_t = \sum_p w*{tp} \, \Delta y_p\)。
  - 線形部のマルチプライヤ: \(m*{p\to z_t} = w*{tp}\)。従って \(C*{\Delta y_p \to \Delta z_t} = w*{tp}\,\Delta y_p\)。
- 活性化: \(y_t = g(z_t)\),\(y_t^0 = g(z_t^0)\),\(\Delta y_t = g(z_t) - g(z_t^0)\)。
  - Rescale ルール（基本）:
    \[
    m\_{z_t\to y_t} \,=\, \begin{cases}
    \dfrac{\Delta y_t}{\Delta z_t}, & \Delta z_t \neq 0 \\
    0, & \Delta z_t = 0
    \end{cases}
    \]
  - したがって,\(C*{\Delta y_p \to \Delta y_t} = m*{z*t\to y_t} \, w*{tp} \, \Delta y_p\)。
- 局所保存則：\(\sum*p C*{\Delta y_p \to \Delta y_t} = \Delta y_t\)。

#### 2.4 RevealCancel ルール（相殺の解消）

非線形部で正負の差分が相殺される問題に対処するため,\(\Delta z_t\) を正負に分解：\(\Delta z_t = \Delta z_t^{+} + \Delta z_t^{-}\)。

- 段階的寄与：
  \[
  m^{+}_{z_t\to y_t} = \frac{g(z_t^0 + \Delta z_t^{+}) - g(z_t^0)}{\Delta z_t^{+}},\quad
  m^{-}_{z_t\to y_t} = \frac{g(z_t^0 + \Delta z_t^{+} + \Delta z_t^{-}) - g(z_t^0 + \Delta z_t^{+})}{\Delta z_t^{-}}.
  \]
- 入力ごとの正負成分 \(\Delta y_p^{+}, \Delta y_p^{-}\) を線形部で分けて伝播し,それぞれに \(m^{+}, m^{-}\) を適用して合成する。これにより打ち消し合いを顕在化できる。

#### 2.5 完全性（Completeness）

上記の局所保存則と鎖則から,入力までの全寄与が出力差分に一致：
\[
\sum*{i} C*{\Delta x_i \to \Delta f} = f(x) - f(x_0).
\]

---

### 3. IG/SHAP との関係

- **Integrated Gradients (IG)**: 直線経路 \(x(\alpha)=x*0+\alpha(x-x_0)\) に沿う経路積分
  \[
  \text{IG}\_i(x) = (x_i-x*{0,i}) \int_0^1 \frac{\partial f(x(\alpha))}{\partial x_i}\, d\alpha
  \]
  は DeepLIFT の差分版を無限小分割で極限化したものと解釈できる。DeepLIFT は 1 ステップ差分で近似するため実装不変性は一般に満たさないが,計算効率と飽和頑健性を得る。
- **SHAP/DeepSHAP**: 背景分布上の期待を組み合わせ,DeepLIFT の層別ルールでシャープレイ値を近似（[arXiv:1705.07874](https://arxiv.org/abs/1705.07874) 参照）。

---

### 4. 実装ノート（重要ポイント）

- **参照（baseline）の選択**: 画像は黒/平均画像,テキストはパディング/UNK,ゲノミクスはニュートラル配列等。複数背景の平均も有効（DeepSHAP）。
- **演算規則**:
  - Max-pooling: 選択インデックスへ寄与を集中（スイッチを使用）。
  - BatchNorm: 学習後はアフィンに吸収可能（線形部に含める）。
  - ドロップアウト等は評価時に無効化。
- **数値安定化**: \(\Delta z\to 0\) での Rescale は \(0\) とするか小さなイプシロンでクリップ。
- **出力選択**: 分類ではロジット（前ソフトマックス）を推奨。

---

### 5. 「キモ」と重要性：本論文の核と影響

- **キモ**: 参照との差分を厳密に保存しながら層ごとに分配することで,飽和や非線形の壁を越えて「何が出力を変えたか」を定量化。
- **重要性/影響**: 単一逆伝播で高速・安定な説明を提供し,ゲノミクス等の高信頼ドメインで実用。SHAP/DeepSHAP への理論的基盤としても機能。

---

### 6. まとめ（要点）

- 完全性：\(\sum*i C*{\Delta x_i\to\Delta f}=f(x)-f(x_0)\)。
- ルール：線形は重み,非線形は Rescale,相殺には RevealCancel。
- 鎖則：マルチプライヤの積で全体寄与を計算。
- IG/SHAP と整合：IG の離散近似,DeepSHAP でシャープレイ近似。

参考: [arXiv:1704.02685](https://arxiv.org/abs/1704.02685)
