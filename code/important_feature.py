# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Aggregate MaskOpt feature importances across many users and build a user×feature heatmap,
# PLUS a global-mean row, and a "personalness" (log2 lift vs global mean) heatmap.

# ✅ user×feature heatmap (L1-normalized abs importance)
# ✅ global mean row appended (GLOBAL_MEAN)
# ✅ personalness heatmap side-by-side (log2((user+eps)/(mean+eps)))
# ✅ uses HEATMAP_USERNAMES (env var) if --usernames is empty
# ✅ can also load TOP-N usernames from a CSV produced by your "consistently_incident_edge_users_ALLMONTHS.csv"
# ✅ saves ONLY ONE final PNG (no CSVs) to avoid disk bloat

# Requirements:
# - Your project provides:
#     prepare_graph_data, load_model_from_ckpt, maskopt_e2e_explain
#   in influencer_rank_full_fixed_xai_paper_no_imagecsv_v20_base.py

# Example:
#   export HEATMAP_USERNAMES="diana.stef,troppaseta"
#   python agg_user_feature_heatmap_v1.py --ckpt ./checkpoints/Run_xxx/model_state.pth --pos_mode latest --show_top_features 40

# Or use TOP-N from detection CSV:
#   python agg_user_feature_heatmap_v1.py --ckpt ./checkpoints/Run_xxx/model_state.pth \
#     --users_csv consistently_incident_edges_out/consistently_incident_edge_users_ALLMONTHS.csv \
#     --top_users 50
# """

# import os
# import argparse
# import datetime
# import inspect
# import numpy as np
# import pandas as pd
# import torch
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# from influencer_rank_full_fixed_xai_paper_no_imagecsv_v20_base import (
#     prepare_graph_data,
#     load_model_from_ckpt,
#     maskopt_e2e_explain,
# )


# # -----------------------------
# # small helpers
# # -----------------------------
# def _parse_list(x):
#     if x is None:
#         return []
#     s = str(x).strip()
#     if not s:
#         return []
#     return [p.strip() for p in s.split(",") if p.strip()]


# def _pick_col(df, cols):
#     for c in cols:
#         if c in df.columns:
#             return c
#     return None


# def _resolve_username_to_node_id(username: str, node_to_idx: dict):
#     """Try multiple variants: original, lower, strip '@'."""
#     if username is None:
#         return None
#     u = str(username).strip()
#     if not u:
#         return None
#     cand = [u, u.lower()]
#     if u.startswith("@"):
#         cand += [u[1:], u[1:].lower()]
#     for c in cand:
#         if c in node_to_idx:
#             return int(node_to_idx[c])
#     return None


# def _filter_kwargs_by_signature(fn, kwargs: dict):
#     """Pass only kwargs accepted by fn(...)."""
#     try:
#         sig = inspect.signature(fn)
#         allowed = set(sig.parameters.keys())
#         return {k: v for k, v in kwargs.items() if k in allowed}
#     except Exception:
#         # if signature fails, just pass through (may error later)
#         return kwargs


# def _plot_dual_heatmap(mat_left, mat_right, x_labels, y_labels,
#                        title_left, title_right, out_png,
#                        vmaxq_left=0.98, vmaxq_right=0.98):
#     """
#     Left: non-negative (0..vmax)
#     Right: symmetric (-vmax..+vmax)
#     """
#     L = np.array(mat_left, dtype=np.float32)
#     R = np.array(mat_right, dtype=np.float32)

#     # left scale (0..quantile)
#     finiteL = L[np.isfinite(L)]
#     if finiteL.size > 0:
#         vmaxL = float(np.quantile(finiteL, vmaxq_left) + 1e-12)
#         vminL = 0.0
#     else:
#         vminL, vmaxL = None, None

#     # right symmetric scale
#     finiteR = R[np.isfinite(R)]
#     if finiteR.size > 0:
#         vmaxR = float(np.quantile(np.abs(finiteR), vmaxq_right) + 1e-12)
#         vminR = -vmaxR
#     else:
#         vminR, vmaxR = None, None

#     H = len(y_labels)
#     W = len(x_labels)

#     fig_w = max(12, 0.33 * W * 2)  # two panels
#     fig_h = max(5, 0.28 * H)

#     fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))

#     im0 = axes[0].imshow(L, aspect="auto", interpolation="nearest", vmin=vminL, vmax=vmaxL)
#     axes[0].set_title(title_left)
#     axes[0].set_xticks(np.arange(W))
#     axes[0].set_xticklabels(x_labels, rotation=90, fontsize=7)
#     axes[0].set_yticks(np.arange(H))
#     axes[0].set_yticklabels(y_labels, fontsize=8)
#     fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

#     im1 = axes[1].imshow(R, aspect="auto", interpolation="nearest", vmin=vminR, vmax=vmaxR)
#     axes[1].set_title(title_right)
#     axes[1].set_xticks(np.arange(W))
#     axes[1].set_xticklabels(x_labels, rotation=90, fontsize=7)
#     axes[1].set_yticks(np.arange(H))
#     axes[1].set_yticklabels(y_labels, fontsize=8)
#     fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)

#     fig.tight_layout()
#     fig.savefig(out_png, dpi=220)
#     plt.close(fig)


# def _sort_users_by_distance_to_mean(Mn: np.ndarray, users: list[str]):
#     """Optional: sort users by L1 distance to global mean (descending = more 'special')."""
#     if Mn.size == 0:
#         return users, Mn
#     mu = Mn.mean(axis=0, keepdims=True)
#     dist = np.abs(Mn - mu).sum(axis=1)
#     order = np.argsort(-dist)
#     return [users[i] for i in order], Mn[order]


# # -----------------------------
# # main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True, help="checkpoint path (.pt/.pth)")
#     ap.add_argument("--end_date", default="2017-12-31")
#     ap.add_argument("--num_months", type=int, default=12)

#     # user selection
#     ap.add_argument("--usernames", default="", help="comma-separated usernames. If empty, uses env HEATMAP_USERNAMES, else users_csv/top_users.")
#     ap.add_argument("--users_csv", default="", help="CSV path containing 'username' column (e.g., consistently_incident_edge_users_ALLMONTHS.csv)")
#     ap.add_argument("--top_users", type=int, default=0, help="if users_csv is used, take top N rows (0 = all)")

#     # position policy
#     ap.add_argument("--pos_mode", choices=["latest", "fixed", "attn_top"], default="latest")
#     ap.add_argument("--fixed_pos", type=int, default=None)
#     ap.add_argument("--topk_pos", type=int, default=1)

#     # MaskOpt controls (feature-only)
#     ap.add_argument("--maskopt_epochs", type=int, default=50)
#     ap.add_argument("--maskopt_lr", type=float, default=0.05)
#     ap.add_argument("--budget_feat", type=int, default=15)
#     ap.add_argument("--topk_feat_per_user", type=int, default=20)
#     ap.add_argument("--num_hops", type=int, default=1)

#     # heatmap controls
#     ap.add_argument("--show_top_features", type=int, default=40)
#     ap.add_argument("--sort_users", choices=["none", "distance"], default="none")
#     ap.add_argument("--outdir", default="agg_feat_results")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # ---- fallback: env HEATMAP_USERNAMES ----
#     if (not str(args.usernames).strip()) and os.getenv("HEATMAP_USERNAMES"):
#         args.usernames = os.getenv("HEATMAP_USERNAMES")

#     # ---------- build graphs ----------
#     target_date = pd.to_datetime(args.end_date)
#     prep = prepare_graph_data(
#         end_date=target_date,
#         num_months=int(args.num_months),
#         metric_numerator="likes_and_comments",
#         metric_denominator="followers",
#     )
#     if prep is None or prep[0] is None:
#         raise SystemExit("prepare_graph_data failed (prep is None).")

#     monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prep
#     feature_names = list(static_cols) + list(dynamic_cols)

#     # For Dec prediction you used input months Jan..Nov => monthly_graphs[:-1]
#     input_graphs = monthly_graphs[:-1]
#     T_input = len(input_graphs)

#     idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

#     # ---------- load model ----------
#     loaded_model, loaded_feature_dim, _ = load_model_from_ckpt(args.ckpt, device=device)
#     model = loaded_model.to(device)
#     model.eval()

#     # ---------- choose usernames ----------
#     usernames = _parse_list(args.usernames)

#     if not usernames:
#         # try CSV
#         if args.users_csv and os.path.exists(args.users_csv):
#             dfu = pd.read_csv(args.users_csv)
#             if "username" not in dfu.columns:
#                 raise SystemExit(f"--users_csv must contain 'username' column. columns={list(dfu.columns)}")
#             if args.top_users and int(args.top_users) > 0:
#                 dfu = dfu.head(int(args.top_users))
#             usernames = [str(x).strip() for x in dfu["username"].tolist() if str(x).strip()]
#         else:
#             raise SystemExit(
#                 "No users provided. Use --usernames or env HEATMAP_USERNAMES, or provide --users_csv."
#             )

#     # resolve to node ids (global ids)
#     target_nodes = []
#     target_usernames = []
#     for u in usernames:
#         nid = _resolve_username_to_node_id(u, node_to_idx)
#         if nid is None:
#             print(f"⚠️ username not found in node_to_idx: {u}")
#             continue
#         target_nodes.append(int(nid))
#         target_usernames.append(idx_to_node.get(int(nid), u))

#     if len(target_nodes) == 0:
#         raise SystemExit("No target nodes resolved (all usernames missing).")

#     print(f"[Targets] resolved users={len(target_nodes)}")

#     # ---------- precompute attention (only if pos_mode=attn_top) ----------
#     # We'll compute embeddings for influencers once, then run attention per user quickly.
#     use_attn = (args.pos_mode == "attn_top")
#     influencer_indices_np = np.asarray(influencer_indices, dtype=np.int64)

#     if use_attn:
#         inf_global = torch.tensor(influencer_indices_np, dtype=torch.long, device=device)
#         N0 = int(input_graphs[0].num_nodes)
#         global2local = torch.full((N0,), -1, dtype=torch.long, device=device)
#         global2local[inf_global] = torch.arange(inf_global.numel(), device=device, dtype=torch.long)

#         with torch.no_grad():
#             seq_emb_l, raw_emb_l = [], []
#             for g in input_graphs:
#                 g = g.to(device)
#                 p_x = model.projection_layer(g.x)
#                 gcn_out = model.gcn_encoder(p_x, g.edge_index)
#                 raw_emb_l.append(p_x.index_select(0, inf_global).cpu())      # [Ninf,P]
#                 seq_emb_l.append(gcn_out.index_select(0, inf_global).cpu())  # [Ninf,D]
#             f_seq_inf = torch.stack(seq_emb_l, dim=0).permute(1, 0, 2).contiguous()  # [Ninf,T,D]
#             f_raw_inf = torch.stack(raw_emb_l, dim=0).permute(1, 0, 2).contiguous()  # [Ninf,T,P]

#     def _select_positions_for_user(node_id: int):
#         if args.pos_mode == "latest":
#             return [T_input - 1]
#         if args.pos_mode == "fixed":
#             if args.fixed_pos is None:
#                 raise ValueError("--fixed_pos is required when pos_mode=fixed")
#             return [int(args.fixed_pos)]
#         if args.pos_mode == "attn_top":
#             # attention for this single user
#             loc = int(global2local[int(node_id)].item())
#             if loc < 0:
#                 return [T_input - 1]
#             b_seq = f_seq_inf[loc:loc+1].to(device)
#             b_raw = f_raw_inf[loc:loc+1].to(device)
#             b_base = torch.zeros((1,), dtype=torch.float32, device=device)  # baseline not needed for pos selection
#             with torch.no_grad():
#                 _, attn = model(b_seq, b_raw, b_base)
#             a = attn.detach().cpu().view(-1).numpy()
#             order = np.argsort(-a)
#             k = max(1, int(args.topk_pos))
#             return [int(i) for i in order[:k]]
#         return [T_input - 1]

#     # MaskOpt coeffs: feature-only
#     coeffs_feat_only = {
#         "edge_size": 0.0,
#         "edge_ent":  0.0,
#         "node_feat_size": 0.02,
#         "node_feat_ent":  0.15,
#     }

#     # ---------- run explanations ----------
#     rows = []  # long format for aggregation: (username, node_id, pos, feature, importance)

#     # Build base kwargs, then filter by maskopt signature (robust to your local implementation)
#     base_kwargs = dict(
#         model=model,
#         input_graphs=input_graphs,
#         feature_names=feature_names,
#         node_to_idx=node_to_idx,
#         device=device,
#         use_subgraph=True,
#         num_hops=int(args.num_hops),

#         # feature-only intent
#         edge_mask_scope="incident",
#         edge_grouping="none",
#         budget_edge=0,
#         coeffs=coeffs_feat_only,

#         epochs=int(args.maskopt_epochs),
#         lr=float(args.maskopt_lr),
#         budget_feat=int(args.budget_feat),

#         mlflow_log=False,
#         tag=None,
#     )

#     base_kwargs = _filter_kwargs_by_signature(maskopt_e2e_explain, base_kwargs)

#     for node_id, uname in zip(target_nodes, target_usernames):
#         pos_list = _select_positions_for_user(int(node_id))

#         for explain_pos in pos_list:
#             call_kwargs = dict(base_kwargs)
#             call_kwargs.update({
#                 "target_node_idx": int(node_id),
#                 "explain_pos": int(explain_pos),
#             })
#             call_kwargs = _filter_kwargs_by_signature(maskopt_e2e_explain, call_kwargs)

#             try:
#                 df_feat, _df_edge, _meta = maskopt_e2e_explain(**call_kwargs)
#             except TypeError as e:
#                 raise SystemExit(
#                     f"maskopt_e2e_explain signature mismatch: {e}\n"
#                     f"Passed keys={sorted(list(call_kwargs.keys()))}"
#                 )
#             except Exception as e:
#                 print(f"⚠️ MaskOpt failed for user={uname} node_id={node_id} pos={explain_pos}: {e}")
#                 continue

#             if df_feat is None or df_feat.empty:
#                 continue

#             idx_col = _pick_col(df_feat, ["feature_idx", "feat_idx", "feature_index", "idx"])
#             imp_col = _pick_col(df_feat, ["importance", "score_impact", "impact", "weight", "mask", "Importance"])
#             if idx_col is None or imp_col is None:
#                 # help debug without saving files
#                 print(f"⚠️ df_feat missing cols for user={uname}. cols={list(df_feat.columns)}")
#                 continue

#             top = df_feat.sort_values(imp_col, ascending=False).head(int(args.topk_feat_per_user))

#             for _, r in top.iterrows():
#                 fi = int(r[idx_col])
#                 if 0 <= fi < len(feature_names):
#                     rows.append({
#                         "username": str(uname),
#                         "node_id": int(node_id),
#                         "pos": int(explain_pos),
#                         "feature": str(feature_names[fi]),
#                         "importance": float(r[imp_col]),
#                     })

#     df_long = pd.DataFrame(rows)
#     if df_long.empty:
#         raise SystemExit("No explanations collected (df_long empty). Try fewer users or check df_feat column names.")

#     # ---------- aggregate to user×feature ----------
#     # pick a single pos for heatmap: if multiple pos exist, we use the most common pos
#     pos_counts = df_long["pos"].value_counts()
#     use_pos = int(pos_counts.index[0])
#     dfp = df_long[df_long["pos"] == use_pos].copy()
#     dfp["imp_abs"] = dfp["importance"].abs()

#     # sum abs importance for each user-feature
#     piv = dfp.groupby(["username", "feature"], as_index=False)["imp_abs"].sum().pivot(
#         index="username", columns="feature", values="imp_abs"
#     ).fillna(0.0)

#     users = list(piv.index)
#     feats = list(piv.columns)

#     # L1 normalize per user (so users comparable)
#     eps = 1e-9
#     M = piv.values.astype(np.float32)
#     Mn = M / (M.sum(axis=1, keepdims=True) + eps)  # [U,F] non-negative

#     # optional sort users (most "special" first)
#     if args.sort_users == "distance":
#         users, Mn = _sort_users_by_distance_to_mean(Mn, users)

#     # global mean vector
#     mu = Mn.mean(axis=0)  # [F]

#     # show top-K features by global mean
#     K = min(int(args.show_top_features), len(feats))
#     feat_order = np.argsort(-mu)
#     top_cols = [feats[i] for i in feat_order[:K]]
#     col_idx = [feats.index(c) for c in top_cols]

#     Mn_show = Mn[:, col_idx]
#     mu_show = mu[col_idx].reshape(1, -1)

#     # personalness = log2 lift vs global mean
#     lift = np.log2((Mn_show + eps) / (mu_show + eps))  # [U,K]
#     lift_mu_row = np.zeros_like(mu_show)               # mean row lift = 0

#     # append GLOBAL_MEAN row
#     Mn_show2 = np.vstack([Mn_show, mu_show])
#     lift_show2 = np.vstack([lift, lift_mu_row])

#     y_labels = users + ["GLOBAL_MEAN"]
#     x_labels = top_cols

#     # ---------- save ONLY ONE PNG ----------
#     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     out_png = os.path.join(args.outdir, f"heat_user_feature_with_global_pos{use_pos}_{ts}.png")

#     _plot_dual_heatmap(
#         Mn_show2, lift_show2,
#         x_labels=x_labels,
#         y_labels=y_labels,
#         title_left=f"User×Feature (L1-normalized |abs importance|)  pos={use_pos}  users={len(users)}",
#         title_right="Personalness = log2 lift vs GLOBAL_MEAN",
#         out_png=out_png,
#         vmaxq_left=0.98,
#         vmaxq_right=0.98
#     )
#     print("[write]", out_png)

#     # ---------- quick console summaries (no files) ----------
#     print("\n[General features TOP10 by GLOBAL_MEAN]")
#     for i in range(min(10, len(x_labels))):
#         print(f"  {x_labels[i]:30s} mean={float(mu_show[0, i]):.4f}")

#     print("\n[Personal features TOP5 per user by lift]")
#     for ui, uname in enumerate(users[:min(len(users), 60)]):  # limit console spam
#         order_u = np.argsort(-lift[ui])
#         top5 = [(x_labels[j], float(lift[ui, j])) for j in order_u[:5]]
#         s = ", ".join([f"{f}({v:+.2f})" for f, v in top5])
#         print(f"  {uname}: {s}")

#     print("\nDone.")


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-pos User×Feature heatmaps (MaskOpt feature-only)
- For each pos in --pos_list:
    1) User×Feature heatmap (L1-normalized |importance|) with GLOBAL_MEAN row
    2) Personalness heatmap = log2 lift vs GLOBAL_MEAN (same grid)
- Save ONLY final PNGs by default (optional CSV outputs via flags)

Usage example:
  python heatmap_user_feature_multi_pos.py \
    --ckpt ./checkpoints/Run_030_20260108_2035/model_state.pth \
    --users_csv consistently_incident_edges_out/consistently_incident_edge_users_ALLMONTHS.csv \
    --top_users 50 \
    --pos_list all \
    --show_top_features 40

Or:
  HEATMAP_USERNAMES="diana.stef,troppaseta" python heatmap_user_feature_multi_pos.py \
    --ckpt ./checkpoints/Run_030_20260108_2035/model_state.pth \
    --pos_list 8,9,10
"""

import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- project imports ----
from influencer_rank_full_fixed_xai_paper_no_imagecsv_v20_base import (
    prepare_graph_data,
    load_model_from_ckpt,
    maskopt_e2e_explain,
)

# -------------------------
# utils
# -------------------------
def _parse_list(x: str):
    if x is None:
        return []
    s = str(x).strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]

def _pick_col(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None

def _resolve_username_to_node_id(u: str, node_to_idx: dict):
    """
    Try a few variants to map username -> global node id
    """
    if u is None:
        return None
    u0 = str(u).strip()
    if not u0:
        return None

    cand = [u0, u0.lower()]
    if u0.startswith("@"):
        cand += [u0[1:], u0[1:].lower()]
    # sometimes your node keys might include "@"
    if not u0.startswith("@"):
        cand += [f"@{u0}", f"@{u0}".lower()]

    for c in cand:
        if c in node_to_idx:
            return int(node_to_idx[c])
    return None

def _parse_pos_list(pos_list_str: str, T_input: int):
    s = str(pos_list_str).strip().lower()
    if s in ["all", "*", "full"]:
        return list(range(T_input))
    out = []
    for p in _parse_list(pos_list_str):
        try:
            pi = int(p)
        except Exception:
            continue
        if 0 <= pi < T_input:
            out.append(pi)
    out = list(dict.fromkeys(out))
    if not out:
        # fallback latest
        out = [T_input - 1]
    return out

def _plot_heatmap(mat, x_labels, y_labels, title, out_png, vmax_quantile=0.98):
    arr = np.asarray(mat, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        vmax = float(np.quantile(np.abs(finite), vmax_quantile) + 1e-9)
        vmin = -vmax
    else:
        vmin, vmax = None, None

    plt.figure(figsize=(max(10, 0.35 * len(x_labels)), max(4, 0.35 * len(y_labels))))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(y_labels)), y_labels, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def _plot_heatmap_separate(mat, x_labels, y_labels, title, out_png, vmax_quantile=0.98):
    arr = np.array(mat, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        vmax = float(np.quantile(np.abs(finite), vmax_quantile) + 1e-9)
        vmin = -vmax
    else:
        vmin, vmax = None, None

    fig = plt.figure(figsize=(max(10, 0.35*len(x_labels)), max(4, 0.35*len(y_labels))))
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def _stable_log2_lift(user_val, global_val, eps=1e-12):
    # log2((u+eps)/(g+eps))
    return np.log2((user_val + eps) / (global_val + eps))


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint path (.pt/.pth)")
    ap.add_argument("--end_date", default="2017-12-31")
    ap.add_argument("--num_months", type=int, default=12)

    # user selection
    ap.add_argument("--usernames", default="", help="comma-separated usernames")
    ap.add_argument("--users_csv", default="", help="CSV containing usernames (e.g., consistently_*_users_ALLMONTHS.csv)")
    ap.add_argument("--top_users", type=int, default=0, help="take top N users from users_csv (0=all in file)")
    ap.add_argument("--sample_n", type=int, default=0, help="if no users given, sample N from influencer_indices (0 disables)")

    # pos selection (shared across all users)
    ap.add_argument("--pos_list", default="all", help="e.g., all or 0,5,10 (0=oldest -> T-1=latest)")

    # MaskOpt controls (feature-only)
    ap.add_argument("--maskopt_epochs", type=int, default=50)
    ap.add_argument("--maskopt_lr", type=float, default=0.05)
    ap.add_argument("--budget_feat", type=int, default=15)
    ap.add_argument("--num_hops", type=int, default=1)
    ap.add_argument("--use_subgraph", action="store_true", help="use k-hop ego subgraph (recommended)")
    ap.add_argument("--topk_feat_per_user", type=int, default=0,
                    help="0=use all features returned by df_feat; >0=keep only top-k by importance (speed/memory)")

    # output
    ap.add_argument("--show_top_features", type=int, default=40)
    ap.add_argument("--outdir", default="agg_feat_results_multi_pos")
    ap.add_argument("--save_long_csv", action="store_true", help="also save long table CSV (otherwise PNG only)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- build graphs ----------
    target_date = pd.to_datetime(args.end_date)
    prep = prepare_graph_data(
        end_date=target_date,
        num_months=int(args.num_months),
        metric_numerator="likes_and_comments",
        metric_denominator="followers",
    )
    if prep[0] is None:
        raise SystemExit("prepare_graph_data failed.")

    # v20_base returns:
    # monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols
    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prep
    feature_names = list(static_cols) + list(dynamic_cols)

    input_graphs = monthly_graphs[:-1]                 # Jan..Nov
    T_input = len(input_graphs)                        # typically 11
    pos_list = _parse_pos_list(args.pos_list, T_input) # shared positions

    idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

    # ---------- load model ----------
    loaded_model, loaded_feature_dim, _ = load_model_from_ckpt(args.ckpt, device=device)
    model = loaded_model.to(device)
    model.eval()

    Fdim = len(feature_names)
    print(f"[Info] T_input={T_input}  Fdim={Fdim}  pos_list={pos_list}")

    # ---------- resolve user list ----------
    usernames = _parse_list(args.usernames)

    # env fallback
    if not usernames:
        env_u = os.environ.get("HEATMAP_USERNAMES", "")
        usernames = _parse_list(env_u)

    # users_csv fallback
    if (not usernames) and args.users_csv:
        if not os.path.exists(args.users_csv):
            raise FileNotFoundError(f"missing users_csv: {args.users_csv}")
        dfu = pd.read_csv(args.users_csv, low_memory=False)

        ucol = _pick_col(dfu, ["username", "user", "influencer", "name"])
        if ucol is None:
            # if the file has node_id, we can map to name later, but better to have usernames
            nidcol = _pick_col(dfu, ["node_id", "id"])
            if nidcol is None:
                raise ValueError(f"users_csv has no username/node_id column. cols={list(dfu.columns)}")
            node_ids_from_csv = dfu[nidcol].dropna().astype(int).tolist()
            if args.top_users and args.top_users > 0:
                node_ids_from_csv = node_ids_from_csv[: int(args.top_users)]
            # convert ids -> names (for printing only)
            usernames = [idx_to_node.get(int(n), str(int(n))) for n in node_ids_from_csv]
            print(f"[Users] loaded by node_id from csv: n={len(usernames)}")
        else:
            # often already sorted by your upstream script; take head top_users if provided
            us = dfu[ucol].dropna().astype(str).tolist()
            if args.top_users and args.top_users > 0:
                us = us[: int(args.top_users)]
            usernames = us
            print(f"[Users] loaded from csv col={ucol}: n={len(usernames)}")

    # final fallback: sample from influencer_indices
    target_nodes = []
    if usernames:
        for u in usernames:
            nid = _resolve_username_to_node_id(u, node_to_idx)
            if nid is None:
                print(f"⚠️ username not found in node_to_idx: {u!r}")
                continue
            target_nodes.append(int(nid))
    else:
        inf = list(map(int, list(influencer_indices)))
        if args.sample_n and args.sample_n > 0:
            rng = np.random.RandomState(0)
            inf = list(rng.choice(np.array(inf), size=min(int(args.sample_n), len(inf)), replace=False))
        target_nodes = inf
        print(f"[Users] sampled from influencer_indices: n={len(target_nodes)}")

    target_nodes = list(dict.fromkeys(target_nodes))
    if len(target_nodes) == 0:
        # show hint
        example_keys = list(node_to_idx.keys())[:30]
        raise SystemExit(f"No users resolved. Example node_to_idx keys: {example_keys}")

    # user labels (stable order)
    user_labels = [idx_to_node.get(int(n), str(int(n))) for n in target_nodes]

    # ---------- MaskOpt settings (feature-only) ----------
    coeffs_feat_only = {
        "edge_size": 0.0,
        "edge_ent":  0.0,
        "node_feat_size": 0.02,
        "node_feat_ent":  0.15,
    }

    # storage: importance_abs[pos_idx, user_idx, feat_idx]
    # keep in RAM (small): users * pos * features
    imp_abs = np.zeros((len(pos_list), len(target_nodes), Fdim), dtype=np.float32)

    # optional long table
    long_rows = []

    # ---------- run explanations ----------
    for ui, node_id in enumerate(target_nodes):
        uname = user_labels[ui]
        print(f"\n[Explain] ({ui+1}/{len(target_nodes)}) node_id={node_id} username={uname}")

        for pi, pos in enumerate(pos_list):
            tag = f"u{node_id}_pos{pos}"
            try:
                df_feat, _df_edge, _meta = maskopt_e2e_explain(
                    model=model,
                    input_graphs=input_graphs,
                    target_node_idx=int(node_id),
                    explain_pos=int(pos),
                    feature_names=feature_names,
                    node_to_idx=node_to_idx,
                    device=device,

                    # subgraph
                    use_subgraph=bool(args.use_subgraph),
                    num_hops=int(args.num_hops),

                    # feature-only
                    edge_mask_scope="incident",
                    edge_grouping="none",
                    budget_edge=0,
                    coeffs=coeffs_feat_only,

                    # optimization
                    epochs=int(args.maskopt_epochs),
                    lr=float(args.maskopt_lr),
                    budget_feat=int(args.budget_feat),

                    mlflow_log=False,
                    tag=None,
                )
            except TypeError:
                # in case your maskopt signature differs (older), retry without epochs/lr
                df_feat, _df_edge, _meta = maskopt_e2e_explain(
                    model=model,
                    input_graphs=input_graphs,
                    target_node_idx=int(node_id),
                    explain_pos=int(pos),
                    feature_names=feature_names,
                    node_to_idx=node_to_idx,
                    device=device,
                    use_subgraph=bool(args.use_subgraph),
                    num_hops=int(args.num_hops),
                    edge_mask_scope="incident",
                    edge_grouping="none",
                    budget_edge=0,
                    coeffs=coeffs_feat_only,
                    budget_feat=int(args.budget_feat),
                    mlflow_log=False,
                    tag=None,
                )

            if df_feat is None or df_feat.empty:
                print(f"  - pos={pos}: df_feat empty (skip)")
                continue

            idx_col = _pick_col(df_feat, ["feature_idx", "feat_idx", "feature_index", "idx"])
            imp_col = _pick_col(df_feat, ["importance", "score_impact", "impact", "weight", "mask", "Importance"])
            if idx_col is None or imp_col is None:
                print(f"  - pos={pos}: df_feat missing idx/importance columns. cols={list(df_feat.columns)}")
                continue

            # optionally keep only top-k per user-pos
            d = df_feat[[idx_col, imp_col]].copy()
            d[idx_col] = pd.to_numeric(d[idx_col], errors="coerce")
            d[imp_col] = pd.to_numeric(d[imp_col], errors="coerce")
            d = d.dropna()

            if args.topk_feat_per_user and int(args.topk_feat_per_user) > 0:
                d = d.reindex(d[imp_col].abs().sort_values(ascending=False).index).head(int(args.topk_feat_per_user))

            v = np.zeros((Fdim,), dtype=np.float32)
            for fi, im in d.values:
                fi = int(fi)
                if 0 <= fi < Fdim:
                    v[fi] += float(abs(im))

            imp_abs[pi, ui, :] = v

            if args.save_long_csv:
                # store per-feature (sparse) rows only for selected features in d
                for fi, im in d.values:
                    fi = int(fi)
                    if 0 <= fi < Fdim:
                        long_rows.append({
                            "node_id": int(node_id),
                            "username": uname,
                            "pos": int(pos),
                            "feature": feature_names[fi],
                            "importance_abs": float(abs(im)),
                        })

            print(f"  - pos={pos}: collected_nonzero={int((v>0).sum())}")

    # ---------- build normalized matrices and feature selection ----------
    # L1-normalized per user (per pos)
    eps = 1e-12
    imp_abs_l1 = np.zeros_like(imp_abs, dtype=np.float32)
    for pi in range(len(pos_list)):
        for ui in range(len(target_nodes)):
            s = float(np.sum(imp_abs[pi, ui, :]))
            if s <= 0:
                continue
            imp_abs_l1[pi, ui, :] = imp_abs[pi, ui, :] / (s + eps)

    # GLOBAL_MEAN per pos (mean over users)
    global_mean = np.mean(imp_abs_l1, axis=1)  # [P,F]

    # choose features to show: mean(global_mean over pos)
    global_mean_over_pos = np.mean(global_mean, axis=0)  # [F]
    feat_order = np.argsort(-global_mean_over_pos)
    topK = min(int(args.show_top_features), Fdim)
    feat_idx_show = feat_order[:topK].tolist()
    feat_names_show = [feature_names[i] for i in feat_idx_show]

    # ---------- save outputs (PNG per pos) ----------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    y_labels = user_labels + ["GLOBAL_MEAN"]

    # for pi, pos in enumerate(pos_list):
    #     # matrix: users + global_mean row
    #     mat_users = imp_abs_l1[pi, :, :][:, feat_idx_show]               # [U,K]
    #     mat_global = global_mean[pi, feat_idx_show][None, :]             # [1,K]
    #     mat1 = np.vstack([mat_users, mat_global])

    #     out_png1 = os.path.join(args.outdir, f"heat_user_feature_absL1_pos{pos}_{ts}.png")
    #     _plot_heatmap(
    #         mat1,
    #         x_labels=feat_names_show,
    #         y_labels=y_labels,
    #         title=f"User×Feature (L1-normalized |abs importance|)  pos={pos}  users={len(target_nodes)}",
    #         out_png=out_png1,
    #         vmax_quantile=0.98
    #     )
    #     print("[write]", out_png1)

    #     # personalness: log2 lift vs GLOBAL_MEAN (last row = 0)
    #     g = mat_global[0, :]
    #     pers_users = _stable_log2_lift(mat_users, g[None, :], eps=1e-12)  # [U,K]
    #     pers = np.vstack([pers_users, np.zeros((1, topK), dtype=np.float32)])

    #     out_png2 = os.path.join(args.outdir, f"heat_personalness_log2lift_pos{pos}_{ts}.png")
    #     _plot_heatmap(
    #         pers,
    #         x_labels=feat_names_show,
    #         y_labels=y_labels,
    #         title=f"Personalness = log2 lift vs GLOBAL_MEAN  pos={pos}  users={len(target_nodes)}",
    #         out_png=out_png2,
    #         vmax_quantile=0.99
    #     )
    #     print("[write]", out_png2)


    for pi, pos in enumerate(pos_list):
        mat_users = imp_abs_l1[pi, :, :][:, feat_idx_show]               # [U,K]
        mat_global = global_mean[pi, feat_idx_show][None, :]             # [1,K]
        mat1 = np.vstack([mat_users, mat_global])

        out_png1 = os.path.join(args.outdir, f"heat_user_feature_absL1_pos{pos}_{ts}.png")
        _plot_heatmap_separate(
            mat1,
            x_labels=feat_names_show,
            y_labels=y_labels,
            title=f"User×Feature (L1-normalized |abs importance|)  pos={pos}  users={len(target_nodes)}",
            out_png=out_png1,
            vmax_quantile=0.98
        )
        print("[write]", out_png1)

        # personalness: log2 lift vs GLOBAL_MEAN (last row = 0)
        g = mat_global[0, :]  # [K]
        pers_users = _stable_log2_lift(mat_users, g[None, :], eps=1e-12)  # [U,K]
        K = mat_users.shape[1]
        pers = np.vstack([pers_users, np.zeros((1, K), dtype=np.float32)])  # [U+1,K]

        out_png2 = os.path.join(args.outdir, f"heat_personalness_log2lift_pos{pos}_{ts}.png")
        _plot_heatmap_separate(
            pers,
            x_labels=feat_names_show,
            y_labels=y_labels,
            title=f"Personalness = log2 lift vs GLOBAL_MEAN  pos={pos}  users={len(target_nodes)}",
            out_png=out_png2,
            vmax_quantile=0.99
        )
        print("[write]", out_png2)


    # optional long csv
    if args.save_long_csv:
        df_long = pd.DataFrame(long_rows)
        out_long = os.path.join(args.outdir, f"long_importance_abs_{ts}.csv")
        df_long.to_csv(out_long, index=False, encoding="utf-8-sig")
        print("[write]", out_long)

    print("\nDone.")


if __name__ == "__main__":
    main()
