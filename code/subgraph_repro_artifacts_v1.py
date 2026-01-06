#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
subgraph_repro_artifacts.py

目的:
  pred_full vs pred_sub の "サブグラフ近似の再現性チェック" を、論文/発表にそのまま貼れる形で
  artifact(CSV/PNG/ログ)として出力するためのユーティリティ。

出力 (最低限セット):
  1) pos × hops × target_node のCSV
     columns:
       pos, hops, target_node
       pred_full, pred_sub, abs_diff
       sub_n_nodes, sub_n_edges
       subset_ok, x_diff_inf
       full_repeat_absdiff, sub_repeat_absdiff
  2) abs_diff vs hops の推移プロット（posごとPNG）
  3) mapping整合性ログ（subset_ok / x_diff_inf をCSVに含める）

追加（target_nodesが複数のとき）:
  - pred_full vs pred_sub scatter（hops固定、posごとPNG）
  - 相関(Pearson/Spearman)とTopK overlap（CSVに追記 + JSON summary）

想定:
  - graphs_full: List[torch_geometric.data.Data]  (Tヶ月分)
  - あなたのモデル推論関数 predict_fn(model, graphs, target_node, pos, device=...) が存在する
    ※ここで pos は「説明したい月（または推論対象の月）」のindex（0..T-1）。
      あなたの実装に合わせて、predict_fn 内で使ってください。

使い方（あなたの既存 FULL SCRIPT / xai系 script に組み込む）:

  from subgraph_repro_artifacts import run_subgraph_repro_check

  def predict_fn(model, graphs, target_node, pos, device="cpu"):
      # 例: 既存の推論関数に合わせて実装してください
      # pred = model.infer_one(graphs, target_node=target_node, explain_pos=pos)
      # return float(pred)
      ...

  df = run_subgraph_repro_check(
      graphs_full=input_graphs,
      model=model,
      predict_fn=predict_fn,
      target_nodes=[1238805, 916490],
      positions=[4,1,5],
      hops_list=[0,1,2,3],
      outdir="out_subgraph_check",
      device="cuda:0",
      undirected=True,
      replace_all_months=False,   # まずは "posのみ置換" が推奨
      repeats=2,                  # 同設定2回推論のブレ検査
      seed=123,
      scatter_hops=2,             # scatter用に固定するhop
      topk=20,
      mlflow_log=True,
  )

  print(df.head())

注意:
  - “モデルが悪い” かどうかはこのチェックでは言えません。
    これは「サブグラフ近似がフル推論をどれだけ再現できるか」の検査です。
"""

from __future__ import annotations

import os
import json
import math
import csv
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

import matplotlib.pyplot as plt

try:
    import torch
except Exception as e:  # pragma: no cover
    raise RuntimeError("This utility requires PyTorch.") from e

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import k_hop_subgraph
except Exception as e:  # pragma: no cover
    raise RuntimeError("This utility requires torch-geometric.") from e


# -------------------------
# Determinism helpers
# -------------------------

def set_global_determinism(seed: int = 123) -> None:
    """Best-effort deterministic settings (still depends on ops/device)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic-ish behavior (may raise if unsupported ops are used)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _to_float(x: Any) -> float:
    if isinstance(x, (float, int)):
        return float(x)
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        arr = x.numpy()
        if np.size(arr) == 1:
            return float(arr.reshape(-1)[0])
        # if vector, take first element (explicitly)
        return float(arr.reshape(-1)[0])
    return float(x)


# -------------------------
# MLflow optional logging
# -------------------------

def _mlflow_active_run():
    try:
        import mlflow  # type: ignore
        return mlflow.active_run()
    except Exception:
        return None


def _mlflow_log_artifact(path: str) -> None:
    try:
        import mlflow  # type: ignore
        if mlflow.active_run() is None:
            return
        mlflow.log_artifact(path)
    except Exception:
        return


# -------------------------
# Correlation / ranking helpers (no scipy dependency)
# -------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    # average ranks for ties
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)

    # tie handling
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0 + 1.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    return _pearson(rx, ry)


def _topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if a.size == 0:
        return float("nan")
    k = min(k, a.size)
    top_a = set(np.argsort(-a)[:k].tolist())
    top_b = set(np.argsort(-b)[:k].tolist())
    return float(len(top_a & top_b) / float(k))


# -------------------------
# Subgraph extraction + sanity checks
# -------------------------

def build_khop_subgraph(
    data_full: Data,
    target_node: int,
    num_hops: int,
    undirected: bool = True,
) -> Tuple[Data, torch.Tensor, int, torch.Tensor]:
    """
    Returns:
      data_sub: Data
      subset: Tensor[ n_sub ] (global node ids)
      mapping: int (index in subset for target)
      edge_mask: Tensor[ n_edges_full ] boolean
    """
    if num_hops < 0:
        raise ValueError("num_hops must be >= 0")

    # PyG k_hop_subgraph:
    subset, edge_index_sub, mapping, edge_mask = k_hop_subgraph(
        node_idx=int(target_node),
        num_hops=int(num_hops),
        edge_index=data_full.edge_index,
        relabel_nodes=True,
        num_nodes=data_full.num_nodes,
        flow="source_to_target",
        directed=not undirected,
    )

    # Build Data and carry over attributes best-effort
    data_sub = Data()
    data_sub.edge_index = edge_index_sub

    n_full = int(data_full.num_nodes)
    n_sub = int(subset.numel())
    e_full = int(data_full.edge_index.size(1))
    e_sub = int(edge_index_sub.size(1))

    # node features
    if hasattr(data_full, "x") and data_full.x is not None:
        data_sub.x = data_full.x[subset]

    # edge_attr
    if hasattr(data_full, "edge_attr") and data_full.edge_attr is not None:
        if data_full.edge_attr.size(0) == e_full:
            data_sub.edge_attr = data_full.edge_attr[edge_mask]
        else:
            # unknown layout; keep as-is (but this usually indicates a mismatch)
            data_sub.edge_attr = data_full.edge_attr

    # copy other tensor attributes by shape heuristic
    for key in data_full.keys:
        if key in ("x", "edge_index", "edge_attr"):
            continue
        try:
            val = data_full[key]
        except Exception:
            continue
        if torch.is_tensor(val):
            if val.size(0) == n_full:
                data_sub[key] = val[subset]
            elif val.size(0) == e_full:
                data_sub[key] = val[edge_mask]
            else:
                data_sub[key] = val
        else:
            data_sub[key] = val

    data_sub.num_nodes = n_sub
    return data_sub, subset, int(mapping), edge_mask


def mapping_sanity(
    data_full: Data,
    data_sub: Data,
    subset: torch.Tensor,
    mapping: int,
    target_node: int,
    atol: float = 1e-7,
) -> Tuple[bool, float]:
    """
    Checks:
      - subset[mapping] == target_node
      - max(|x_sub[mapping] - x_full[target]|)  (∞-norm)

    Returns:
      subset_ok, x_diff_inf
    """
    subset_ok = bool(int(subset[mapping].item()) == int(target_node))

    x_diff_inf = float("nan")
    if hasattr(data_full, "x") and data_full.x is not None and hasattr(data_sub, "x") and data_sub.x is not None:
        xf = data_full.x[int(target_node)]
        xs = data_sub.x[int(mapping)]
        # handle float precision
        x_diff_inf = float((xs - xf).abs().max().detach().cpu().item())
        # if subset_ok is false but x_diff is tiny, mapping might still be wrong but coincidentally similar
        # (still treat as failure)
    return subset_ok, x_diff_inf


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------
# Main runner
# -------------------------

def run_subgraph_repro_check(
    *,
    graphs_full: Sequence[Data],
    model: Any,
    predict_fn: Callable[[Any, Sequence[Data], int, int, str], float],
    target_nodes: Sequence[int],
    positions: Sequence[int],
    hops_list: Sequence[int],
    outdir: str,
    device: str = "cpu",
    undirected: bool = True,
    replace_all_months: bool = False,
    repeats: int = 2,
    seed: int = 123,
    scatter_hops: Optional[int] = None,
    topk: int = 20,
    mlflow_log: bool = True,
) -> "pd.DataFrame | List[Dict[str, Any]]":
    """
    Parameters
    ----------
    graphs_full:
      List[T] of PyG Data (full graphs for each month)
    model:
      your trained model object
    predict_fn:
      function(model, graphs, target_node, pos, device) -> float
      NOTE: pos is forwarded as-is; adapt inside predict_fn to your implementation.
    replace_all_months:
      False: only graphs[pos] is replaced by its k-hop subgraph.
      True: each month graph is replaced by its k-hop subgraph (computed per pos).
            (This is stricter / usually diff gets smaller but cost rises.)
    repeats:
      run same inference N times to check non-determinism. Minimum 2 recommended.
    """
    if pd is None:
        raise RuntimeError("pandas is required for this utility (please `pip install pandas`).")

    _ensure_dir(outdir)

    set_global_determinism(seed)

    # Put model in eval mode if possible
    if hasattr(model, "eval"):
        model.eval()

    results: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "positions": list(map(int, positions)),
        "hops_list": list(map(int, hops_list)),
        "target_nodes_count": int(len(target_nodes)),
        "replace_all_months": bool(replace_all_months),
        "undirected": bool(undirected),
        "device": str(device),
        "seed": int(seed),
        "topk": int(topk),
    }

    T = len(graphs_full)

    # Cache full predictions per (pos, target)
    full_cache: Dict[Tuple[int, int], float] = {}

    def _predict_repeat(graphs: Sequence[Data], target_node: int, pos: int) -> Tuple[float, float]:
        """Returns (mean_pred, max_absdiff_across_repeats_vs_first)."""
        preds = []
        for _ in range(max(1, int(repeats))):
            set_global_determinism(seed)  # reset RNG to reduce drift if any stochasticity exists
            p = _to_float(predict_fn(model, graphs, int(target_node), int(pos), str(device)))
            preds.append(p)
        mean_pred = float(np.mean(preds))
        max_abs = float(np.max(np.abs(np.array(preds) - preds[0]))) if len(preds) >= 2 else 0.0
        return mean_pred, max_abs

    for pos in positions:
        pos = int(pos)
        if not (0 <= pos < T):
            raise ValueError(f"pos={pos} is out of range (T={T}).")

        for hop in hops_list:
            hop = int(hop)
            for target in target_nodes:
                target = int(target)

                # full pred (cached)
                key = (pos, target)
                if key not in full_cache:
                    pred_full, full_repeat_absdiff = _predict_repeat(graphs_full, target, pos)
                    full_cache[key] = pred_full
                else:
                    pred_full = full_cache[key]
                    # still evaluate repeat stability once per key (best-effort)
                    _, full_repeat_absdiff = _predict_repeat(graphs_full, target, pos)

                # subgraph build
                data_full_pos = graphs_full[pos]
                data_sub_pos, subset, mapping, edge_mask = build_khop_subgraph(
                    data_full_pos, target, hop, undirected=undirected
                )
                subset_ok, x_diff_inf = mapping_sanity(
                    data_full_pos, data_sub_pos, subset, mapping, target
                )

                sub_n_nodes = int(data_sub_pos.num_nodes) if hasattr(data_sub_pos, "num_nodes") else int(data_sub_pos.x.size(0))
                sub_n_edges = int(data_sub_pos.edge_index.size(1))

                # construct graphs_sub
                if replace_all_months:
                    graphs_sub = []
                    for t in range(T):
                        df = graphs_full[t]
                        ds, _, _, _ = build_khop_subgraph(df, target, hop, undirected=undirected)
                        graphs_sub.append(ds)
                else:
                    graphs_sub = list(graphs_full)
                    graphs_sub[pos] = data_sub_pos

                # IMPORTANT: when using subgraph, the target index MUST be relabeled to `mapping`
                pred_sub, sub_repeat_absdiff = _predict_repeat(graphs_sub, mapping, pos)

                abs_diff = float(abs(pred_full - pred_sub))

                results.append({
                    "pos": pos,
                    "hops": hop,
                    "target_node": target,
                    "pred_full": float(pred_full),
                    "pred_sub": float(pred_sub),
                    "abs_diff": abs_diff,
                    "sub_n_nodes": sub_n_nodes,
                    "sub_n_edges": sub_n_edges,
                    "subset_ok": bool(subset_ok),
                    "x_diff_inf": float(x_diff_inf),
                    "full_repeat_absdiff": float(full_repeat_absdiff),
                    "sub_repeat_absdiff": float(sub_repeat_absdiff),
                })

    df = pd.DataFrame(results)
    csv_path = os.path.join(outdir, "subgraph_repro_pos_hops.csv")
    df.to_csv(csv_path, index=False)

    # Plot abs_diff vs hops per pos
    for pos in sorted(set(map(int, positions))):
        dpos = df[df["pos"] == pos]
        if dpos.empty:
            continue
        # aggregate across nodes: median/mean
        agg = (
            dpos.groupby("hops", as_index=False)["abs_diff"]
            .agg(["mean", "median", "max"])
            .reset_index()
        )
        fig = plt.figure()
        plt.plot(agg["hops"], agg["mean"], marker="o", label="mean")
        plt.plot(agg["hops"], agg["median"], marker="o", label="median")
        plt.plot(agg["hops"], agg["max"], marker="o", label="max")
        plt.xlabel("hops")
        plt.ylabel("abs_diff = |pred_full - pred_sub|")
        plt.title(f"abs_diff vs hops (pos={pos})")
        plt.grid(True, alpha=0.3)
        plt.legend()
        fig_path = os.path.join(outdir, f"abs_diff_vs_hops_pos_{pos}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close(fig)

        if mlflow_log and _mlflow_active_run() is not None:
            _mlflow_log_artifact(fig_path)

    if mlflow_log and _mlflow_active_run() is not None:
        _mlflow_log_artifact(csv_path)

    # Additional: scatter + correlations if multiple nodes
    if len(target_nodes) >= 2:
        if scatter_hops is None:
            scatter_hops = int(sorted(hops_list)[-1])

        scatter_hops = int(scatter_hops)
        corr_rows = []
        for pos in sorted(set(map(int, positions))):
            d = df[(df["pos"] == pos) & (df["hops"] == scatter_hops)].copy()
            if d.empty:
                continue

            x = d["pred_full"].to_numpy(dtype=float)
            y = d["pred_sub"].to_numpy(dtype=float)

            pear = _pearson(x, y)
            spear = _spearman(x, y)
            overlap = _topk_overlap(x, y, int(topk))

            corr_rows.append({
                "pos": int(pos),
                "hops": int(scatter_hops),
                "pearson": float(pear),
                "spearman": float(spear),
                "topk": int(topk),
                "topk_overlap": float(overlap),
                "n": int(len(d)),
            })

            # scatter plot
            fig = plt.figure()
            plt.scatter(x, y, s=18, alpha=0.7)
            # y=x line
            mn = float(min(x.min(), y.min()))
            mx = float(max(x.max(), y.max()))
            plt.plot([mn, mx], [mn, mx], linestyle="--")
            plt.xlabel("pred_full")
            plt.ylabel("pred_sub")
            plt.title(f"pred_full vs pred_sub (pos={pos}, hops={scatter_hops})\n"
                      f"pearson={pear:.3f} spearman={spear:.3f} top{topk} overlap={overlap:.3f}")
            plt.grid(True, alpha=0.3)
            fig_path = os.path.join(outdir, f"scatter_full_vs_sub_pos_{pos}_hops_{scatter_hops}.png")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close(fig)

            if mlflow_log and _mlflow_active_run() is not None:
                _mlflow_log_artifact(fig_path)

        if corr_rows:
            corr_df = pd.DataFrame(corr_rows)
            corr_csv = os.path.join(outdir, "subgraph_repro_scatter_metrics.csv")
            corr_df.to_csv(corr_csv, index=False)
            if mlflow_log and _mlflow_active_run() is not None:
                _mlflow_log_artifact(corr_csv)

            summary["scatter_hops"] = int(scatter_hops)
            summary["scatter_metrics"] = corr_rows

    # Save summary JSON (便利)
    summary_path = os.path.join(outdir, "subgraph_repro_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if mlflow_log and _mlflow_active_run() is not None:
        _mlflow_log_artifact(summary_path)

    return df


# -------------------------
# Example predict_fn template
# -------------------------

def example_predict_fn_template(model: Any, graphs: Sequence[Data], target_node: int, pos: int, device: str) -> float:
    """
    あなたの実装に合わせて、ここを "既存推論" に置換してください。

    重要:
      - model.eval() になっていること
      - device への転送（model / data）
      - subgraph時は target_node が "mapping" に置換済みで渡ってくる

    例（ダミー）:
      model が graphs[pos].x の target_node 行を取り出して MLP するだけのとき:
         x = graphs[pos].x[target_node]
         return float(model(x))
    """
    raise NotImplementedError("Replace example_predict_fn_template with your project-specific predict_fn.")
