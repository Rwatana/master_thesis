import os
import glob
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Metrics
# -------------------------
def _dcg(rels: np.ndarray, k: int, gain_type: str = "linear") -> float:
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    gains = (np.power(2.0, rels) - 1.0) if gain_type == "exp2" else rels
    return float(np.sum(gains * discounts))


def ndcg_at_k(df: pd.DataFrame, rel_col: str, score_col: str, k: int, gain_type: str = "linear") -> float:
    d = df[[rel_col, score_col]].dropna()
    if d.empty:
        return 0.0
    pred = d.sort_values(score_col, ascending=False)[rel_col].to_numpy()
    ideal = d.sort_values(rel_col, ascending=False)[rel_col].to_numpy()
    dcg_val = _dcg(pred, k, gain_type=gain_type)
    idcg_val = _dcg(ideal, k, gain_type=gain_type)
    return 0.0 if idcg_val <= 0 else float(dcg_val / idcg_val)


def rbp_at_k(
    df: pd.DataFrame,
    rel_col: str,
    score_col: str,
    k: int,
    p: float = 0.8,
    rel_norm: str = "max",
) -> float:
    d = df[[rel_col, score_col]].dropna()
    if d.empty:
        return 0.0

    ranked_rel = d.sort_values(score_col, ascending=False)[rel_col].to_numpy()[:k].astype(float)
    if ranked_rel.size == 0:
        return 0.0

    rel_all = d[rel_col].to_numpy().astype(float)
    denom = float(np.percentile(rel_all, 95)) if rel_norm == "p95" else float(np.max(rel_all))
    denom = max(denom, 1e-12)
    rel01 = np.clip(ranked_rel / denom, 0.0, 1.0)

    weights = np.power(p, np.arange(rel01.size))
    return float((1.0 - p) * np.sum(rel01 * weights))


def mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    if y_true.size < 2:
        return {"pearson": 0.0, "spearman": 0.0}
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else 0.0
    rt = pd.Series(y_true).rank(method="average").to_numpy()
    rp = pd.Series(y_pred).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(rt, rp)[0, 1]) if np.std(rt) > 0 and np.std(rp) > 0 else 0.0
    return {"pearson": pearson, "spearman": spearman}


def compute_metrics(
    df: pd.DataFrame,
    rel_col: str,
    score_col: str,
    ks: List[int],
    gain_type: str,
    rbp_p: float,
    rbp_norm: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in ks:
        out[f"ndcg@{k}"] = ndcg_at_k(df, rel_col, score_col, k, gain_type=gain_type)
        out[f"rbp(p={rbp_p:.2f})@{k}"] = rbp_at_k(df, rel_col, score_col, k, p=rbp_p, rel_norm=rbp_norm)
    return out


# -------------------------
# Data loading helpers
# -------------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


@st.cache_data(show_spinner=False)
def load_influencers_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    rename_map = {
        "Username": "username",
        "Category": "category",
        "#Followers": "followers_raw",
        "#Followees": "followees_raw",
        "#Posts": "posts_history_raw",
    }
    df = df.rename(columns=rename_map)
    for c in ["username", "category", "followers_raw"]:
        if c not in df.columns:
            df[c] = np.nan
    df["username"] = df["username"].astype(str).str.strip()
    df["category"] = df["category"].fillna("Unknown").astype(str)
    df["followers_raw"] = pd.to_numeric(df["followers_raw"], errors="coerce").fillna(0.0)
    return df[["username", "category", "followers_raw"]].copy()


def list_local_files(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def ensure_followers_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "followers_log1p" in out.columns and "followers_raw" not in out.columns:
        out["followers_raw"] = np.expm1(pd.to_numeric(out["followers_log1p"], errors="coerce").fillna(0.0))

    if "followers_raw" in out.columns and "followers_log1p" not in out.columns:
        out["followers_log1p"] = np.log1p(pd.to_numeric(out["followers_raw"], errors="coerce").fillna(0.0))

    return out


def parse_int_like(s: str) -> Optional[int]:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    s = s.replace(",", "").replace("_", "").replace(" ", "")
    try:
        return int(float(s))
    except Exception:
        return None


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="InfluencerRank Prediction Analyzer", layout="wide")
st.title("InfluencerRank Prediction Analyzer (CSV + Filters + NDCG/RBP)")

with st.sidebar:
    st.header("1) 予測CSV")

    pred_candidates = list_local_files(
        [
            "../predictions/*.csv",
            "./data/*.csv",
            "./*.csv",
        ]
    )

    load_mode = st.radio("読み込み方法", ["ローカル一覧", "アップロード"], index=0)

    pred_df = None
    pred_path = None

    if load_mode == "ローカル一覧":
        if not pred_candidates:
            st.warning("CSVが見つかりません。./predictions/ に置くか,アップロードしてください。")
        else:
            pred_path = st.selectbox("pred_*.csv を選択", pred_candidates, index=0)
            pred_df = load_csv(pred_path) if pred_path else None
    else:
        up = st.file_uploader("予測CSVをアップロード", type=["csv"])
        if up is not None:
            pred_df = pd.read_csv(up)
            pred_df.columns = [c.strip() for c in pred_df.columns]

if pred_df is None or pred_df.empty:
    st.info("左のサイドバーから予測CSVを読み込んでください。")
    st.stop()

need_cols = ["true_score", "pred_score"]
missing = [c for c in need_cols if c not in pred_df.columns]
if missing:
    st.error(f"予測CSVに必要な列がありません: {missing}\n\n最低限 true_score, pred_score が必要です。")
    st.stop()

df = pred_df.copy()
df.columns = [c.strip() for c in df.columns]

# numeric (initial)
for c in ["true_score", "pred_score", "baseline_score", "followers_log1p", "followers_raw"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = ensure_followers_cols(df)

# optional metadata merge
with st.sidebar:
    st.header("2) メタデータ結合（任意）")

    auto_active = st.checkbox("active_influencers_v8_*.csv を自動検出して結合", value=True)
    active_df = None
    if auto_active:
        act_candidates = list_local_files(["./active_influencers_v8_*.csv", "./data/active_influencers_v8_*.csv"])
        if act_candidates:
            act_path = st.selectbox("active_influencers を選択", act_candidates, index=0)
            active_df = load_csv(act_path)
        else:
            st.caption("active_influencers_v8_*.csv が見つかりませんでした。")

    use_inftxt = st.checkbox("influencers.txt を読み込んで結合", value=False)
    inf_txt_df = None
    if use_inftxt:
        inf_path = st.text_input("influencers.txt のパス", value="../influencers.txt")
        if inf_path and os.path.exists(inf_path):
            inf_txt_df = load_influencers_txt(inf_path)
        else:
            st.caption("influencers.txt が見つかりません（パスを確認）。")

# merge by username if possible
if "username" in df.columns:
    if active_df is not None and (not active_df.empty) and "username" in active_df.columns:
        active_df = active_df.copy()
        active_df.columns = [c.strip() for c in active_df.columns]
        active_df["username"] = active_df["username"].astype(str).str.strip()

        df = df.merge(active_df, on="username", how="left", suffixes=("", "_active"))

        # active file may have "followers" raw
        if "followers" in df.columns and "followers_raw" not in df.columns:
            df["followers_raw"] = pd.to_numeric(df["followers"], errors="coerce")

        df = ensure_followers_cols(df)

    if inf_txt_df is not None and (not inf_txt_df.empty):
        df = df.merge(inf_txt_df, on="username", how="left", suffixes=("", "_txt"))

        if "followers_raw_txt" in df.columns:
            df["followers_raw"] = pd.to_numeric(df.get("followers_raw"), errors="coerce")
            df["followers_raw_txt"] = pd.to_numeric(df["followers_raw_txt"], errors="coerce")
            df["followers_raw"] = df["followers_raw"].fillna(df["followers_raw_txt"])
            df = ensure_followers_cols(df)

        if "category" in df.columns and "category_txt" in df.columns:
            df["category"] = df["category"].fillna(df["category_txt"])
else:
    st.warning("予測CSVに username 列がないため,カテゴリ結合はできません（出力側で username を含めるのが推奨）。")

# basic cleanup (after merges, coerce again)
for c in ["true_score", "pred_score", "baseline_score", "followers_raw", "followers_log1p"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = ensure_followers_cols(df)

df["true_score"] = pd.to_numeric(df["true_score"], errors="coerce")
df["pred_score"] = pd.to_numeric(df["pred_score"], errors="coerce")
df = df.dropna(subset=["true_score", "pred_score"]).copy()

# -------------------------
# Filters
# -------------------------
with st.sidebar:
    st.header("3) フィルタ")

    if "username" in df.columns:
        q_user = st.text_input("username 検索（部分一致）", value="")
    else:
        q_user = ""

    cat_col = "category" if "category" in df.columns else None
    if cat_col:
        cats = sorted(df[cat_col].dropna().astype(str).unique().tolist())
        sel_cats = st.multiselect("カテゴリ", options=cats, default=cats)
    else:
        sel_cats = None
        st.caption("category が無いのでカテゴリフィルタは無効です（active_influencers か influencers.txt を結合すると出ます）。")

    # followers_raw exact range (typed min/max, supports commas)
    if "followers_raw" in df.columns and df["followers_raw"].notna().any():
        lo = int(np.floor(float(df["followers_raw"].min())))
        hi = int(np.ceil(float(df["followers_raw"].max())))
        st.caption(f"followers_raw の範囲: {lo:,} 〜 {hi:,}")

        if "fol_min_s" not in st.session_state:
            st.session_state["fol_min_s"] = ""
        if "fol_max_s" not in st.session_state:
            st.session_state["fol_max_s"] = ""

        fol_min_s = st.text_input("followers_raw 最小（空欄=下限なし）", value=st.session_state["fol_min_s"], key="fol_min_s")
        fol_max_s = st.text_input("followers_raw 最大（空欄=上限なし）", value=st.session_state["fol_max_s"], key="fol_max_s")

        fol_min = parse_int_like(fol_min_s)
        fol_max = parse_int_like(fol_max_s)

        if (fol_min_s.strip() and fol_min is None) or (fol_max_s.strip() and fol_max is None):
            st.warning("followers_raw の入力が数値として解釈できません（例: 10000 / 10,000）。")

        if fol_min is not None:
            fol_min = max(lo, min(hi, fol_min))
        if fol_max is not None:
            fol_max = max(lo, min(hi, fol_max))

        if fol_min is not None and fol_max is not None and fol_min > fol_max:
            st.warning("followers_raw: 最小 > 最大 なので入れ替えました。")
            fol_min, fol_max = fol_max, fol_min
    else:
        fol_min, fol_max = None, None
        st.caption("followers_raw が無いのでフォロワーフィルタは無効です。")

    # score range
    tmin, tmax = float(df["true_score"].min()), float(df["true_score"].max())
    pmin, pmax = float(df["pred_score"].min()), float(df["pred_score"].max())
    true_range = st.slider("true_score 範囲", min_value=tmin, max_value=tmax, value=(tmin, tmax))
    pred_range = st.slider("pred_score 範囲", min_value=pmin, max_value=pmax, value=(pmin, pmax))

f = df.copy()
if q_user and "username" in f.columns:
    f = f[f["username"].astype(str).str.contains(q_user, case=False, na=False)]
if sel_cats is not None and cat_col:
    f = f[f[cat_col].astype(str).isin([str(x) for x in sel_cats])]

# followers filter (typed exact)
if "followers_raw" in f.columns:
    f["followers_raw"] = pd.to_numeric(f["followers_raw"], errors="coerce")
    if fol_min is not None:
        f = f[f["followers_raw"] >= fol_min]
    if fol_max is not None:
        f = f[f["followers_raw"] <= fol_max]

f = f[(f["true_score"] >= true_range[0]) & (f["true_score"] <= true_range[1])]
f = f[(f["pred_score"] >= pred_range[0]) & (f["pred_score"] <= pred_range[1])]

if f.empty:
    st.warning("フィルタ後のデータが空です。条件を緩めてください。")
    st.stop()

# -------------------------
# Evaluation settings
# -------------------------
with st.sidebar:
    st.header("4) 評価設定")
    ks = st.multiselect("K（@K）", options=[1, 5, 10, 20, 50, 100, 200, 500, 1000], default=[10, 50, 100])
    ks = sorted(list(set([int(x) for x in ks]))) if ks else [10, 50, 100]

    gain_type = st.selectbox("NDCG gain", ["linear", "exp2"], index=0)
    rbp_p = st.slider("RBP p", min_value=0.5, max_value=0.95, value=0.8, step=0.01)
    rbp_norm = st.selectbox("RBP relevance 正規化", ["max", "p95"], index=0)

    score_col = st.selectbox(
        "ランキングに使う列",
        options=[c for c in ["pred_score", "baseline_score"] if c in f.columns],
        index=0,
    )
    rel_col = "true_score"

# -------------------------
# Main layout
# -------------------------
c1, c2 = st.columns([1.15, 0.85])

with c1:
    st.subheader("上位ランキング（フィルタ後）")
    topn = st.slider("表示件数", 10, 500, 100, 10)

    cols_show = [c for c in ["node_id", "username", "category", "followers_raw", "true_score", "pred_score", "baseline_score"] if c in f.columns]
    view = f.sort_values(score_col, ascending=False).head(topn).copy()
    view.insert(0, "rank", np.arange(1, len(view) + 1))
    st.dataframe(view[["rank"] + cols_show], use_container_width=True, height=440)

    st.subheader("可視化")
    st.caption("true_score vs pred_score")
    st.scatter_chart(f[["true_score", "pred_score"]], x="true_score", y="pred_score")

    if "followers_raw" in f.columns:
        st.caption("followers_raw vs pred_score")
        st.scatter_chart(f[["followers_raw", "pred_score"]].dropna(), x="followers_raw", y="pred_score")

    dl_cols = st.multiselect("DLする列", options=list(f.columns), default=cols_show if cols_show else list(f.columns))
    st.download_button(
        "フィルタ後CSVをダウンロード",
        data=f[dl_cols].to_csv(index=False).encode("utf-8-sig"),
        file_name="filtered_predictions.csv",
        mime="text/csv",
    )

with c2:
    st.subheader("全体指標（フィルタ後集合で計算）")
    m_rank = compute_metrics(
        f,
        rel_col=rel_col,
        score_col=score_col,
        ks=ks,
        gain_type=gain_type,
        rbp_p=rbp_p,
        rbp_norm=rbp_norm,
    )
    m_err = mae_rmse(f[rel_col].to_numpy(), f[score_col].to_numpy())
    m_cor = corr(f[rel_col].to_numpy(), f[score_col].to_numpy())

    summary = {**m_rank, **m_err, **m_cor, "rows": int(len(f))}
    mdf = pd.DataFrame([summary]).T.reset_index()
    mdf.columns = ["metric", "value"]
    st.dataframe(mdf, use_container_width=True, height=360)

    if "baseline_score" in f.columns and score_col != "baseline_score":
        st.subheader("baseline 比較（delta）")
        base = compute_metrics(
            f,
            rel_col=rel_col,
            score_col="baseline_score",
            ks=ks,
            gain_type=gain_type,
            rbp_p=rbp_p,
            rbp_norm=rbp_norm,
        )
        rows = []
        for k in ks:
            rows.append(
                {
                    "k": k,
                    "ndcg_pred": m_rank[f"ndcg@{k}"],
                    "ndcg_base": base[f"ndcg@{k}"],
                    "ndcg_delta": m_rank[f"ndcg@{k}"] - base[f"ndcg@{k}"],
                    f"rbp_pred(p={rbp_p:.2f})": m_rank[f"rbp(p={rbp_p:.2f})@{k}"],
                    f"rbp_base(p={rbp_p:.2f})": base[f"rbp(p={rbp_p:.2f})@{k}"],
                    f"rbp_delta(p={rbp_p:.2f})": m_rank[f"rbp(p={rbp_p:.2f})@{k}"]
                    - base[f"rbp(p={rbp_p:.2f})@{k}"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=240)

    if "category" in f.columns:
        st.subheader("カテゴリ別（NDCG/RBP）")
        cat_rows = []
        for cat, g in f.groupby("category"):
            if len(g) < 20:
                continue
            mm = compute_metrics(g, rel_col=rel_col, score_col=score_col, ks=ks, gain_type=gain_type, rbp_p=rbp_p, rbp_norm=rbp_norm)
            cat_rows.append({"category": cat, "rows": int(len(g)), **mm})
        if cat_rows:
            st.dataframe(pd.DataFrame(cat_rows).sort_values("rows", ascending=False), use_container_width=True, height=320)
        else:
            st.caption("カテゴリ別集計は 1カテゴリあたり20件以上のデータがある場合のみ表示します。")
