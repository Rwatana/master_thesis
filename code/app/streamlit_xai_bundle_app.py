
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import torch
import matplotlib.pyplot as plt
import networkx as nx

from torch_geometric.utils import k_hop_subgraph

# ============================================================
# Config
# ============================================================
TZ = "Asia/Tokyo"
YEAR_OF_INTEREST = 2017

# ============================================================
# Utils
# ============================================================
def _load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def _resolve_dir(p: str) -> str:
    return str(Path(p).expanduser().resolve())

def _ts_to_dt_jst(ts):
    """
    ts: scalar or Series (unix seconds or milliseconds)
    return: tz-naive datetime (JST表記)
    """
    s = pd.to_numeric(ts, errors="coerce")
    if isinstance(s, (pd.Series, np.ndarray)):
        mx = float(np.nanmax(s.values)) if hasattr(s, "values") else float(np.nanmax(s))
        unit = "s" if (np.isfinite(mx) and mx < 1e11) else "ms"
        dt = pd.to_datetime(s, unit=unit, utc=True).dt.tz_convert(TZ)
        return dt.dt.tz_localize(None)
    else:
        if not np.isfinite(float(s)):
            return pd.NaT
        unit = "s" if float(s) < 1e11 else "ms"
        dt = pd.to_datetime(int(s), unit=unit, utc=True).tz_convert(TZ)
        return dt.tz_localize(None)

def _ensure_posts_time_cols(df_posts: pd.DataFrame) -> pd.DataFrame:
    df = df_posts.copy()

    # username 列名ゆれ吸収
    if "username" not in df.columns:
        for c in ["user", "author", "src_user", "screen_name"]:
            if c in df.columns:
                df = df.rename(columns={c: "username"})
                break

    # post_id 列名ゆれ吸収
    if "post_id" not in df.columns:
        for c in ["id", "postid", "media_id", "pk"]:
            if c in df.columns:
                df = df.rename(columns={c: "post_id"})
                break

    # datetime/month/year を作る（JST）
    if "timestamp" in df.columns:
        df["datetime_jst"] = _ts_to_dt_jst(df["timestamp"])
    elif "datetime" in df.columns:
        df["datetime_jst"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        df["datetime_jst"] = pd.NaT

    df["month"] = df["datetime_jst"].dt.to_period("M").astype(str)   # "2017-01"
    df["year"] = df["datetime_jst"].dt.year.astype("Int64")          # 2017

    return df

def add_rank_within_user_year(d_user_year: pd.DataFrame, metric_col: str, year: int) -> pd.DataFrame:
    """
    選択ユーザーの「その年の投稿だけ」で順位 a/n を作る（他ユーザーは入れない）
    """
    d = d_user_year.copy()
    d[metric_col] = pd.to_numeric(d.get(metric_col, np.nan), errors="coerce")

    if "year" in d.columns:
        d = d[d["year"].astype("Int64") == int(year)].copy()

    n = int(len(d))
    if n == 0:
        d["rank_in_user_year"] = pd.Series(dtype="Int64")
        d["n_posts_in_user_year"] = 0
        d["rank_user_year_str"] = ""
        return d

    d["rank_in_user_year"] = d[metric_col].rank(ascending=False, method="min")
    d["n_posts_in_user_year"] = n
    d["rank_user_year_str"] = d["rank_in_user_year"].astype("Int64").astype(str) + "/" + str(n)
    return d

def add_rank_within_user_month(d_user_month: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    選択ユーザーの「その月の投稿だけ」で順位 a/n を作る（他ユーザーは入れない）
    """
    d = d_user_month.copy()
    d[metric_col] = pd.to_numeric(d.get(metric_col, np.nan), errors="coerce")

    n = int(len(d))
    if n == 0:
        d["rank_in_user_month"] = pd.Series(dtype="Int64")
        d["n_posts_in_user_month"] = 0
        d["rank_user_month_str"] = ""
        return d

    d["rank_in_user_month"] = d[metric_col].rank(ascending=False, method="min")
    d["n_posts_in_user_month"] = n
    d["rank_user_month_str"] = d["rank_in_user_month"].astype("Int64").astype(str) + "/" + str(n)
    return d

# ============================================================
# Bundle loaders (cached)
# ============================================================
@st.cache_resource
def load_bundle_meta(bundle_dir: str):
    """
    メタ情報だけ読む（重い torch.load はしない）
    """
    bundle_dir = _resolve_dir(bundle_dir)
    manifest_path = os.path.join(bundle_dir, "manifest.json")
    manifest = _load_json(manifest_path)

    node_to_idx = _load_json(os.path.join(bundle_dir, manifest["paths"]["node_to_idx"]))
    node_to_idx = {str(k): int(v) for k, v in node_to_idx.items()}
    idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

    months = _load_json(os.path.join(bundle_dir, manifest["paths"]["months"]))
    feature_names = _load_json(os.path.join(bundle_dir, manifest["paths"]["feature_names"]))

    pred_path = manifest["paths"].get("predictions", None)
    evid_path = manifest["paths"].get("evidence_index", None)
    posts_path = manifest["paths"].get("posts", None)

    pred_path = os.path.join(bundle_dir, pred_path) if pred_path else None
    evid_path = os.path.join(bundle_dir, evid_path) if evid_path else None
    posts_path = os.path.join(bundle_dir, posts_path) if posts_path else None

    return manifest, idx_to_node, months, feature_names, pred_path, evid_path, posts_path

@st.cache_data
def load_predictions_csv(pred_path: str):
    if pred_path and os.path.exists(pred_path):
        return pd.read_csv(pred_path)
    return None

@st.cache_data
def load_evidence_table(evid_path: str):
    if evid_path and os.path.exists(evid_path):
        if evid_path.endswith(".parquet"):
            return pd.read_parquet(evid_path)
        return pd.read_csv(evid_path)
    return None

@st.cache_data
def load_posts_table(posts_path: str):
    """
    posts は「2017年だけ」に絞って返す（母数を2017年に固定）
    ※ ただし順位は「他ユーザーを混ぜない」要望なので、ここでは rank を作らない
    """
    if not posts_path or (not os.path.exists(posts_path)):
        return None

    df = pd.read_parquet(posts_path) if posts_path.endswith(".parquet") else pd.read_csv(posts_path)
    df = _ensure_posts_time_cols(df)

    # 2017年のみ
    df = df[df["year"].astype("Int64") == YEAR_OF_INTEREST].copy()

    # username/post_id が空だと join / 表示が壊れるので最低限整える
    if "username" not in df.columns:
        df["username"] = ""
    if "post_id" not in df.columns:
        df["post_id"] = df.index.astype(int)

    return df

@st.cache_data
def load_graph_at_pos(bundle_dir: str, manifest: dict, pos: int):
    """
    pos の月だけ読む
    - 推奨：manifest["paths"]["graph_files"] がある形式（posごとpt）
    - 旧：manifest["paths"]["graphs"] が単一pt（全pos一括）
    """
    bundle_dir = _resolve_dir(bundle_dir)

    graph_files = manifest["paths"].get("graph_files", None)
    if graph_files:
        p = os.path.join(bundle_dir, graph_files[int(pos)])
        return torch.load(p, map_location="cpu")

    graphs_path = os.path.join(bundle_dir, manifest["paths"]["graphs"])
    graphs = torch.load(graphs_path, map_location="cpu", weights_only=False)
    return graphs[int(pos)]

@st.cache_data
def load_edge_importance(bundle_dir: str, target_node_id: int, pos: int):
    bundle_dir = _resolve_dir(bundle_dir)
    p = os.path.join(bundle_dir, "xai", f"node_{int(target_node_id)}", f"pos_{int(pos)}", "edge_importance.csv")
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)

# ============================================================
# Ego graph helpers
# ============================================================
def _scale_width(w, w_min=0.5, w_max=6.0):
    a = np.asarray(w, dtype=float)
    if a.size == 0:
        return a
    m = np.max(np.abs(a))
    if m < 1e-12:
        return np.ones_like(a) * w_min
    s = (np.abs(a) / m) * (w_max - w_min) + w_min
    return s

@st.cache_data
def build_user_year_rank_map(df_posts: pd.DataFrame, username: str, year: int, metric_col: str):
    """
    選択ユーザーの year 全投稿で順位を作って、post_id -> year_rank を返す
    """
    d = df_posts[
        (df_posts["username"].astype(str) == str(username)) &
        (df_posts["year"].astype("Int64") == int(year))
    ].copy()

    if d.empty:
        return pd.DataFrame(columns=["post_id", "rank_in_user_year", "n_posts_in_user_year", "rank_user_year_str"])

    d["post_id"] = d["post_id"].astype(str)
    d[metric_col] = pd.to_numeric(d.get(metric_col, np.nan), errors="coerce")

    # 1が最大
    d["rank_in_user_year"] = d[metric_col].rank(ascending=False, method="min")
    n = int(len(d))
    d["n_posts_in_user_year"] = n
    d["rank_user_year_str"] = d["rank_in_user_year"].astype("Int64").astype(str) + f"/{n}"

    return d[["post_id", "rank_in_user_year", "n_posts_in_user_year", "rank_user_year_str"]]


def build_ego_nx_graph_limited(g, center: int, hops: int, idx_to_node: dict,
                              top_neighbors: list[int], max_nodes: int = 250):
    center = int(center)

    subset, edge_index_sub, _, _ = k_hop_subgraph(
        node_idx=center,
        num_hops=int(hops),
        edge_index=g.edge_index,
        relabel_nodes=False,
        num_nodes=g.num_nodes,
        flow="source_to_target",
    )

    subset = subset.detach().cpu().numpy().astype(int)
    edge_index_sub = edge_index_sub.detach().cpu().numpy().astype(int)

    keep = set([center]) | set(int(x) for x in top_neighbors)
    if len(subset) > int(max_nodes):
        extra = [x for x in subset.tolist() if x not in keep]
        need = int(max_nodes) - len(keep)
        if need > 0:
            keep |= set(extra[:need])
        subset = np.array(sorted(list(keep)), dtype=int)

        s = edge_index_sub[0]
        t = edge_index_sub[1]
        mask = np.isin(s, subset) & np.isin(t, subset)
        edge_index_sub = edge_index_sub[:, mask]

    G = nx.DiGraph()
    for nid in subset:
        G.add_node(int(nid), label=idx_to_node.get(int(nid), str(int(nid))))
    for s, t in zip(edge_index_sub[0], edge_index_sub[1]):
        if int(s) in G.nodes and int(t) in G.nodes:
            G.add_edge(int(s), int(t))
    return G

def draw_ego_graph(G: nx.DiGraph, center: int, edge_imp_df: pd.DataFrame,
                   topk: int, show_all_edges: bool):
    center = int(center)
    imp_map = {}

    if edge_imp_df is not None and not edge_imp_df.empty:
        df = edge_imp_df.copy().sort_values("importance", ascending=False).head(int(topk))
        if "neighbor_id" in df.columns:
            for _, r in df.iterrows():
                imp_map[int(r["neighbor_id"])] = float(r["importance"])

    edges = list(G.edges())
    widths, colors = [], []
    for u, v in edges:
        w, c = 0.2, 0.0
        if u == center and v in imp_map:
            w = abs(imp_map[v]); c = imp_map[v]
        elif v == center and u in imp_map:
            w = abs(imp_map[u]); c = imp_map[u]
        else:
            if not show_all_edges:
                w = 0.0
        widths.append(w); colors.append(c)

    widths = np.asarray(widths, dtype=float)
    keep = widths > 0
    edges_kept = [e for e, k in zip(edges, keep) if k]
    widths_kept = _scale_width(widths[keep])
    colors_kept = np.asarray(colors, dtype=float)[keep]

    pos = nx.spring_layout(G.to_undirected(), seed=0, iterations=60)

    fig = plt.figure(figsize=(9, 7))
    ax = plt.gca()
    ax.set_axis_off()

    node_sizes, node_colors = [], []
    for n in G.nodes():
        if int(n) == center:
            node_sizes.append(900); node_colors.append(1.0)
        else:
            node_sizes.append(280); node_colors.append(0.2)

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, font_size=8)

    if len(edges_kept) > 0:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges_kept,
            width=widths_kept.tolist(),
            edge_color=colors_kept.tolist(),
            edge_cmap=plt.cm.coolwarm,
            alpha=0.8,
            arrows=True,
            arrowsize=12
        )

    plt.tight_layout()
    return fig

# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="XAI Bundle Analyzer", layout="wide")
st.title("XAI Bundle Analyzer — Ego graph & Evidence & Post effectiveness (user-only ranks)")

# -------------------------
# Sidebar: sticky bundle loader
# -------------------------
st.sidebar.header("Bundle")

if "bundle_dir" not in st.session_state:
    st.session_state.bundle_dir = "../streamlit_bundle/run_20260114_000100"
if "bundle_loaded" not in st.session_state:
    st.session_state.bundle_loaded = False
if "loaded_bundle_dir" not in st.session_state:
    st.session_state.loaded_bundle_dir = None
if "bundle_meta" not in st.session_state:
    st.session_state.bundle_meta = None

bundle_dir_in = st.sidebar.text_input(
    "Bundle directory (manifest.json があるフォルダ)",
    value=st.session_state.bundle_dir,
    key="bundle_dir",
)

c1, c2, c3 = st.sidebar.columns(3)
load_clicked = c1.button("Load / Apply", type="primary")
reload_clicked = c2.button("Reload")
clear_clicked = c3.button("Clear cache")

if clear_clicked:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.sidebar.success("cache cleared")

bundle_dir = _resolve_dir(bundle_dir_in)
manifest_path = os.path.join(bundle_dir, "manifest.json")

path_changed = (st.session_state.loaded_bundle_dir is not None) and (bundle_dir != st.session_state.loaded_bundle_dir)

if (not st.session_state.bundle_loaded) and (not load_clicked) and (not reload_clicked):
    st.info("左のサイドバーでパスを確認して **Load / Apply** を押してください（初回のみ）。")
    st.stop()

if path_changed and (not load_clicked) and (not reload_clicked):
    st.sidebar.warning("パスが変更されています（まだ適用されていません）。反映するには **Load / Apply** を押してください。")

need_load = (
    (not st.session_state.bundle_loaded)
    or load_clicked
    or reload_clicked
    or (st.session_state.loaded_bundle_dir != bundle_dir and load_clicked)
)

if need_load:
    if not os.path.exists(manifest_path):
        st.error(f"manifest.json が見つかりません: {manifest_path}")
        st.stop()
    with st.spinner("Loading bundle meta..."):
        meta = load_bundle_meta(bundle_dir)
    st.session_state.bundle_meta = meta
    st.session_state.bundle_loaded = True
    st.session_state.loaded_bundle_dir = bundle_dir
    st.sidebar.success(f"Loaded: {bundle_dir}")

manifest, idx_to_node, months, feature_names, pred_path, evid_path, posts_path = st.session_state.bundle_meta

target_node_id = int(manifest.get("target_node_id", manifest.get("target_node", -1)))
available_pos = manifest.get("xai", {}).get("edge_importance_positions", [])

st.sidebar.markdown("---")
st.sidebar.write("Run:", manifest.get("run_name", "unknown"))
st.sidebar.write("Target node:", target_node_id, idx_to_node.get(target_node_id, str(target_node_id)))
st.sidebar.write("Edge positions:", available_pos)

if not available_pos:
    st.error("manifest の edge_importance_positions が空です（run_experiment 側で edge_importance_by_pos が保存されてない可能性）。")
    st.stop()

pos = st.sidebar.selectbox("Explain pos", options=available_pos, index=0)
hops = st.sidebar.slider("Ego hops", 1, 3, 1)
topk = st.sidebar.slider("Top-K important incident edges", 5, 100, 30, step=5)
min_abs_imp = st.sidebar.slider("Min |importance| filter", 0.0, 1.0, 0.0, step=0.01)
show_all_edges = st.sidebar.checkbox("Show non-important edges too", value=False)
max_nodes = st.sidebar.slider("Max nodes (layout safety)", 80, 600, 250, step=10)

# -------------------------
# Load edge importance
# -------------------------
with st.spinner("Loading edge importance..."):
    edge_imp = load_edge_importance(bundle_dir, target_node_id, int(pos))
if edge_imp is None:
    st.error("edge_importance.csv が見つかりません。bundle の xai/node_x/pos_y/ を確認してください。")
    st.stop()

edge_imp = edge_imp.copy()
if "importance" in edge_imp.columns:
    edge_imp["abs_importance"] = edge_imp["importance"].abs()
    edge_imp = edge_imp[edge_imp["abs_importance"] >= float(min_abs_imp)].reset_index(drop=True)
else:
    edge_imp["abs_importance"] = np.nan

month_label = months[int(pos)] if int(pos) < len(months) else f"pos_{int(pos)}"

# -------------------------
# Layout: Ego graph + Top edges + Edge selector
# -------------------------
colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("1) Ego graph (importance-weighted incident edges)")
    with st.spinner("Loading graph for selected pos..."):
        g = load_graph_at_pos(bundle_dir, manifest, int(pos))

    if "neighbor_id" in edge_imp.columns:
        top_neighbors = edge_imp.sort_values("importance", ascending=False).head(int(topk))["neighbor_id"].astype(int).tolist()
    else:
        top_neighbors = []

    with st.spinner("Building ego graph..."):
        Gnx = build_ego_nx_graph_limited(
            g, center=target_node_id, hops=hops, idx_to_node=idx_to_node,
            top_neighbors=top_neighbors, max_nodes=int(max_nodes)
        )
        fig = draw_ego_graph(Gnx, center=target_node_id, edge_imp_df=edge_imp, topk=topk, show_all_edges=show_all_edges)

    st.pyplot(fig, clear_figure=True)
    st.caption(f"pos={pos} / month={month_label} / nodes={Gnx.number_of_nodes()} / edges={Gnx.number_of_edges()}")

with colB:
    st.subheader("2) Top edges")
    st.caption(f"pos={pos} / month={month_label}")

    show_df = edge_imp.sort_values("importance", ascending=False).head(int(topk)) if "importance" in edge_imp.columns else edge_imp.head(int(topk))
    show_cols = [c for c in ["neighbor_id", "neighbor_name", "edge_type", "importance", "abs_importance"] if c in show_df.columns]
    if show_cols:
        st.dataframe(show_df[show_cols], use_container_width=True)
    else:
        st.dataframe(show_df, use_container_width=True)

    st.markdown("#### Select an edge (for evidence/post check)")
    edge_type, neighbor_name = "", ""

    if ("edge_type" in show_df.columns) and ("neighbor_name" in show_df.columns) and ("importance" in show_df.columns):
        edge_candidates = show_df.copy()
        edge_candidates["label"] = edge_candidates.apply(
            lambda r: f'{r["edge_type"]}: {r["neighbor_name"]} (imp={r["importance"]:.3g})', axis=1
        )
        labels = edge_candidates["label"].tolist()
        if len(labels) > 0:
            sel = st.selectbox("Edge", options=labels, index=0)
            sel_row = edge_candidates[edge_candidates["label"] == sel].iloc[0]
            edge_type = str(sel_row["edge_type"])
            neighbor_name = str(sel_row["neighbor_name"])
        else:
            st.info("edge candidates が空です。")
    else:
        st.warning("edge_importance.csv に edge_type / neighbor_name / importance が揃っていないため、edge選択ができません。")

    st.markdown("---")
    st.subheader("Prediction (optional)")
    df_pred = None
    if pred_path and os.path.exists(pred_path):
        with st.spinner("Loading predictions.csv..."):
            df_pred = load_predictions_csv(pred_path)

    if df_pred is not None and "node_id" in df_pred.columns:
        row = df_pred[df_pred["node_id"] == target_node_id]
        if len(row) > 0:
            r = row.iloc[0]
            out = {}
            for k in ["true_score", "pred_score", "baseline_score"]:
                if k in r.index:
                    out[k] = float(r[k])
            st.write(out)
        else:
            st.info("predictions.csv に target node がいません（nonzero filter などの影響）。")
    else:
        st.info("predictions.csv が bundle に含まれていません。")

st.markdown("---")

# -------------------------
# 3) Evidence view (optional)
# -------------------------
st.header("3) Evidence view (edge reality check)")

df_evid = None
if evid_path and os.path.exists(evid_path):
    with st.spinner("Loading evidence_index..."):
        df_evid = load_evidence_table(evid_path)

if df_evid is None:
    st.warning("evidence_index が bundle にありません（run_experiment 側で生成されていない/pathsに入ってない）。")
else:
    st.caption("evidence_index がある場合のみ表示（example_post_ids があれば posts と join できます）")

    src_user = idx_to_node.get(target_node_id, str(target_node_id))

    dm = df_evid.copy()
    for c in ["month", "src_user", "edge_type", "neighbor_name"]:
        if c in dm.columns:
            dm[c] = dm[c].astype(str)

    hit = dm[
        (dm.get("month", "") == str(month_label)) &
        (dm.get("src_user", "") == str(src_user)) &
        (dm.get("edge_type", "") == str(edge_type)) &
        (dm.get("neighbor_name", "") == str(neighbor_name))
    ].copy()

    if hit.empty:
        st.info("該当 evidence が見つかりませんでした（month/username/neighbor_name/edge_type の一致を要確認）。")
    else:
        r = hit.iloc[0]
        out = {}
        for k in ["month", "src_user", "edge_type", "neighbor_name", "count"]:
            if k in r.index:
                out[k] = r[k]
        st.write(out)

        ex = r.get("example_post_ids", [])
        if isinstance(ex, str):
            exs = ex.strip()
            if exs.startswith("[") and exs.endswith("]"):
                try:
                    ex = json.loads(exs)
                except Exception:
                    ex = [exs]
            elif exs == "" or exs.lower() == "nan":
                ex = []
            else:
                ex = [exs]

        st.write({"example_post_ids": ex})

st.markdown("---")

# -------------------------
# 4) Timestamp convert (debug)
# -------------------------
st.header("4) Timestamp convert (debug)")
ts_in = st.number_input("unix timestamp", value=1484985425, step=1)
dt_jst = pd.to_datetime(int(ts_in), unit="s", utc=True).tz_convert("Asia/Tokyo")
st.write({"timestamp": int(ts_in), "datetime_jst": str(dt_jst)})

st.markdown("---")

# -------------------------
# Load posts once (2017 only)
# -------------------------
df_posts = None
if posts_path and os.path.exists(posts_path):
    with st.spinner("Loading posts table (2017 only)..."):
        df_posts = load_posts_table(posts_path)

if df_posts is None:
    st.warning("posts が bundle にありません。EVID_POSTS_CSV に dataset_A_active_all.csv を指定して bundle に入れてください。")
    st.stop()

if "like_count" not in df_posts.columns:
    st.error("posts に like_count がありません（順位が作れない）。")
    st.stop()

src_user = idx_to_node.get(target_node_id, str(target_node_id))
metric = "like_count"

# ============================================================
# 5) Post effectiveness in selected month (user-only ranks)
# ============================================================
st.header("5) Post effectiveness in selected month (user-only ranks)")

d_user_month = df_posts[
    (df_posts["username"].astype(str) == str(src_user)) &
    (df_posts["month"].astype(str) == str(month_label))
].copy()

if d_user_month.empty:
    st.info(f"{src_user} の {month_label} の投稿が見つかりませんでした。")
else:
    # month rank（その月のユーザー投稿だけ）
    d_user_month = add_rank_within_user_month(d_user_month, metric)

    # year rank（2017年のユーザー全投稿で作って map して join）
    year_map = build_user_year_rank_map(df_posts, src_user, YEAR_OF_INTEREST, metric)

    # join のために型合わせ（post_id が数値/文字列揺れすると全部NaNになる）
    d_user_month["post_id"] = d_user_month["post_id"].astype(str)
    year_map["post_id"] = year_map["post_id"].astype(str)

    d_user_month = d_user_month.merge(year_map, on="post_id", how="left")

    # 表示用サマリ
    n_user_month = int(len(d_user_month))
    n_user_year = int(d_user_month["n_posts_in_user_year"].iloc[0]) if "n_posts_in_user_year" in d_user_month.columns and d_user_month["n_posts_in_user_year"].notna().any() else int(len(year_map))

    best_month = int(pd.to_numeric(d_user_month["rank_in_user_month"], errors="coerce").min())
    best_year  = int(pd.to_numeric(d_user_month["rank_in_user_year"], errors="coerce").min()) if "rank_in_user_year" in d_user_month.columns else -1

    st.write({
        "user": src_user,
        "month": month_label,
        "posts_in_user_month": str(n_user_month),
        "posts_in_user_2017": str(n_user_year),
        "best_rank_in_user_month": f"{best_month}/{n_user_month}",
        "best_rank_in_user_2017": (f"{best_year}/{n_user_year}" if best_year > 0 else "N/A"),
    })

    d_user_month_sorted = d_user_month.sort_values([metric, "datetime_jst"], ascending=[False, True])

    show_cols = [c for c in [
        "datetime_jst", "timestamp", "post_id",
        "like_count",
        "rank_user_month_str",   # 月内 a/n
        "rank_user_year_str",    # 年内 a/n  ← ここが修正の本命
        "caption"
    ] if c in d_user_month_sorted.columns]

    st.dataframe(d_user_month_sorted[show_cols], use_container_width=True)

# ============================================================
# (6) Edge-related posts in this month (user-only ranks)
# ============================================================
st.markdown("---")
st.header("6) Edge-related posts in this month (user-only rank a/n)")

# ここは (5) で作った d_user_month を再利用するのが安全
# d_user_month が無い場合に備えて再構築もする
metric = "like_count"
src_user = idx_to_node.get(target_node_id, str(target_node_id))

d_user_month = df_posts[
    (df_posts["username"].astype(str) == str(src_user)) &
    (df_posts["month"].astype(str) == str(month_label))
].copy()

if d_user_month.empty:
    st.info("その月の投稿が無いので (6) は表示できません。")
else:
    # month rank
    d_user_month = add_rank_within_user_month(d_user_month, metric)

    # year rank map を join（※ここが重要：月DFの中で年順位を計算しない）
    year_map = build_user_year_rank_map(df_posts, src_user, YEAR_OF_INTEREST, metric)
    d_user_month["post_id"] = d_user_month["post_id"].astype(str)
    year_map["post_id"] = year_map["post_id"].astype(str)
    d_user_month = d_user_month.merge(year_map, on="post_id", how="left")

    cap_col = "caption" if "caption" in d_user_month.columns else ("text" if "text" in d_user_month.columns else None)
    if cap_col is None:
        st.info("posts に caption/text が無いので edge 含有判定ができません。")
    elif (edge_type == "") or (neighbor_name == ""):
        st.info("edge が選べていないので、(6) は表示できません。")
    else:
        needle = str(neighbor_name)
        et = str(edge_type).lower()
        if "hash" in et and not needle.startswith("#"):
            needle = "#" + needle
        if "ment" in et and not needle.startswith("@"):
            needle = "@" + needle

        has_edge = d_user_month[cap_col].astype(str).str.contains(re.escape(needle), na=False)
        d_edge_month = d_user_month[has_edge].copy()

        st.write({
            "edge": f"{edge_type}:{neighbor_name}",
            "matched_posts_in_this_month": int(len(d_edge_month)),
        })

        if d_edge_month.empty:
            st.info("その月にこの edge を含む投稿が見つかりませんでした（caption からの推定）。")
        else:
            d_edge_month = d_edge_month.sort_values(metric, ascending=False)
            cols = [c for c in [
                "datetime_jst", "post_id", "like_count",
                "rank_user_month_str",      # 月内順位（ユーザー内）
                "rank_user_year_str",       # 年内順位（ユーザー内）←ここが月と違う値になる
                cap_col
            ] if c in d_edge_month.columns]
            st.dataframe(d_edge_month[cols], use_container_width=True)


# ============================================================
# (7) Edge-related posts in 2017 (user-only year rank a/n)
# ============================================================
st.markdown("---")
st.header("7) Edge-related posts in 2017 (user-only year rank a/n)")

d_user_year = df_posts[
    (df_posts["username"].astype(str) == str(src_user)) &
    (df_posts["year"].astype("Int64") == int(YEAR_OF_INTEREST))
].copy()

if d_user_year.empty:
    st.info("2017年の投稿が無いので (7) は表示できません。")
else:
    # ★年順位は “年全体” で作って join（作り直さない）
    year_map = build_user_year_rank_map(df_posts, src_user, YEAR_OF_INTEREST, metric)
    d_user_year["post_id"] = d_user_year["post_id"].astype(str)
    year_map["post_id"] = year_map["post_id"].astype(str)
    d_user_year = d_user_year.merge(year_map, on="post_id", how="left")

    cap_col_y = "caption" if "caption" in d_user_year.columns else ("text" if "text" in d_user_year.columns else None)
    if cap_col_y is None:
        st.info("posts に caption/text が無いので edge 含有判定ができません。")
    elif (edge_type == "") or (neighbor_name == ""):
        st.info("edge が選べていないので、(7) は表示できません。")
    else:
        needle = str(neighbor_name)
        et = str(edge_type).lower()
        if "hash" in et and not needle.startswith("#"):
            needle = "#" + needle
        if "ment" in et and not needle.startswith("@"):
            needle = "@" + needle

        has_edge_y = d_user_year[cap_col_y].astype(str).str.contains(re.escape(needle), na=False)
        d_edge_year = d_user_year[has_edge_y].copy()

        st.write({
            "edge": f"{edge_type}:{neighbor_name}",
            "matched_posts_in_2017": int(len(d_edge_year)),
            "user_posts_in_2017": int(len(d_user_year)),
        })

        if d_edge_year.empty:
            st.info("2017年にこの edge を含む投稿が見つかりませんでした（caption からの推定）。")
        else:
            d_edge_year = d_edge_year.sort_values(metric, ascending=False)
            cols = [c for c in [
                "month", "datetime_jst", "post_id", "like_count",
                "rank_user_year_str",   # 年内順位（ユーザー内）←(7)の本命
                cap_col_y
            ] if c in d_edge_year.columns]
            st.dataframe(d_edge_year[cols], use_container_width=True)