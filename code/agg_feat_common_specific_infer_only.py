#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from influencer_rank_full_fixed_xai_paper_no_imagecsv_v20_base import prepare_graph_data, HardResidualInfluencerModel
from influencer_rank_full_fixed_xai_paper_no_imagecsv_v20_base import load_model_from_ckpt, maskopt_e2e_explain
# ---- your project imports (adjust as needed) ----
# from your_project import prepare_graph_data, HardResidualInfluencerModel, load_model_from_ckpt, maskopt_e2e_explain

def _parse_list(x):
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

def _plot_heatmap(mat, x_labels, y_labels, title, out_png, vmax_quantile=0.98):
    arr = np.array(mat, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        vmax = float(np.quantile(np.abs(finite), vmax_quantile) + 1e-9)
        vmin = -vmax
    else:
        vmin, vmax = None, None

    plt.figure(figsize=(max(10, 0.35*len(x_labels)), max(4, 0.35*len(y_labels))))
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.02, pad=0.02)
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90, fontsize=7)
    plt.yticks(np.arange(len(y_labels)), y_labels, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint path (.pt/.pth)")
    ap.add_argument("--end_date", default="2017-12-31")
    ap.add_argument("--num_months", type=int, default=12)

    # target influencers
    ap.add_argument("--usernames", default="", help="comma separated influencer usernames (e.g., diana.stef,troppaseta)")
    ap.add_argument("--sample_n", type=int, default=0, help="if usernames empty, sample N from influencer_indices (0 disables sampling)")

    # pos policy
    ap.add_argument("--pos_mode", choices=["latest", "fixed", "attn_top"], default="latest")
    ap.add_argument("--fixed_pos", type=int, default=None, help="used when pos_mode=fixed")
    ap.add_argument("--topk_pos", type=int, default=1, help="used when pos_mode=attn_top")

    # MaskOpt controls (feature-only)
    ap.add_argument("--maskopt_epochs", type=int, default=50)
    ap.add_argument("--maskopt_lr", type=float, default=0.05)
    ap.add_argument("--budget_feat", type=int, default=15)
    ap.add_argument("--topk_feat_per_user", type=int, default=20)
    ap.add_argument("--num_hops", type=int, default=1)

    # aggregation output
    ap.add_argument("--show_top_features", type=int, default=40)
    ap.add_argument("--outdir", default="agg_feat_results")

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

    monthly_graphs, influencer_indices, node_to_idx, follower_feat_idx, static_cols, dynamic_cols = prep
    feature_names = static_cols + dynamic_cols
    T_input = len(monthly_graphs) - 1  # input months for Dec prediction (Jan..Nov)
    input_graphs = monthly_graphs[:-1]

    # idx->name
    idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

    # ---------- load model ----------
    loaded_model, loaded_feature_dim, _ = load_model_from_ckpt(args.ckpt, device=device)
    model = loaded_model.to(device)
    model.eval()

    # ---------- choose target nodes ----------
    usernames = _parse_list(args.usernames)
    target_nodes = []

    if usernames:
        for u in usernames:
            cand = [u, u.lower()]
            if u.startswith("@"):
                cand += [u[1:], u[1:].lower()]
            hit = None
            for c in cand:
                if c in node_to_idx:
                    hit = c
                    break
            if hit is None:
                print(f"⚠️ username not found in node_to_idx: {u}")
                continue
            target_nodes.append(int(node_to_idx[hit]))
    else:
        # sample from influencer_indices (already global ids)
        inf = list(map(int, list(influencer_indices)))
        if args.sample_n and args.sample_n > 0:
            rng = np.random.RandomState(0)
            inf = list(rng.choice(np.array(inf), size=min(args.sample_n, len(inf)), replace=False))
        target_nodes = inf

    if len(target_nodes) == 0:
        raise SystemExit("No target nodes resolved.")

    # ---------- precompute influencer embeddings + attention (only if needed for pos_mode=attn_top) ----------
    # We need influencer_indices as tensor and a global2local mapping to pick users.
    influencer_indices_np = np.asarray(influencer_indices, dtype=np.int64)
    N0 = int(input_graphs[0].num_nodes)

    inf_global = torch.tensor(influencer_indices_np, dtype=torch.long, device=device)
    global2local = torch.full((N0,), -1, dtype=torch.long, device=device)
    global2local[inf_global] = torch.arange(inf_global.numel(), device=device, dtype=torch.long)

    # Build f_seq_inf/f_raw_inf for ALL influencers once (fast enough, and avoids repeated GCN runs)
    with torch.no_grad():
        seq_emb_l, raw_emb_l = [], []
        for g in input_graphs:
            g = g.to(device)
            p_x = model.projection_layer(g.x)
            gcn_out = model.gcn_encoder(p_x, g.edge_index)
            raw_emb_l.append(p_x.index_select(0, inf_global).cpu())      # [Ninf,P]
            seq_emb_l.append(gcn_out.index_select(0, inf_global).cpu())  # [Ninf,D]
        f_seq_inf = torch.stack(seq_emb_l, dim=0).permute(1, 0, 2).contiguous()  # [Ninf,T,D]
        f_raw_inf = torch.stack(raw_emb_l, dim=0).permute(1, 0, 2).contiguous()  # [Ninf,T,P]

    # ---------- run explanations ----------
    rows = []            # long table for aggregation
    rows_user = []       # for personalness ranking

    def _select_positions_for_user(node_id: int):
        if args.pos_mode == "latest":
            return [T_input - 1]
        if args.pos_mode == "fixed":
            if args.fixed_pos is None:
                raise ValueError("--fixed_pos required when pos_mode=fixed")
            return [int(args.fixed_pos)]
        if args.pos_mode == "attn_top":
            # compute attention for this single user (cheap)
            loc = int(global2local[int(node_id)].item())
            if loc < 0:
                return [T_input - 1]
            b_seq = f_seq_inf[loc:loc+1].to(device)
            b_raw = f_raw_inf[loc:loc+1].to(device)
            b_base = torch.zeros((1,), dtype=torch.float32, device=device)  # baseline not needed for attn selection
            with torch.no_grad():
                _, attn = model(b_seq, b_raw, b_base)
            a = attn.detach().cpu().view(-1).numpy()
            order = np.argsort(-a)
            return [int(i) for i in order[:max(1, int(args.topk_pos))]]
        return [T_input - 1]

    # MaskOpt coeffs: feature-only (edge regularizers = 0)
    coeffs_feat_only = {
        "edge_size": 0.0,
        "edge_ent":  0.0,
        "node_feat_size": 0.02,
        "node_feat_ent":  0.15,
    }

    for node_id in target_nodes:
        node_name = idx_to_node.get(int(node_id), str(int(node_id)))
        pos_list = _select_positions_for_user(int(node_id))

        for explain_pos in pos_list:
            tag = f"u{node_id}_pos{explain_pos}"

            df_feat, _df_edge, _meta = maskopt_e2e_explain(
                model=model,
                input_graphs=input_graphs,
                target_node_idx=int(node_id),
                explain_pos=int(explain_pos),
                feature_names=feature_names,
                node_to_idx=node_to_idx,
                device=device,
                use_subgraph=True,
                num_hops=int(args.num_hops),

                # feature-only
                edge_mask_scope="incident",   # if your function supports "none", set "none"
                edge_grouping="none",
                budget_edge=0,
                coeffs=coeffs_feat_only,

                epochs=int(args.maskopt_epochs),
                lr=float(args.maskopt_lr),
                budget_feat=int(args.budget_feat),

                mlflow_log=False,
                tag=None,
            )

            if df_feat is None or df_feat.empty:
                continue

            idx_col = _pick_col(df_feat, ["feature_idx","feat_idx","feature_index","idx"])
            imp_col = _pick_col(df_feat, ["importance","score_impact","impact","weight","mask","Importance"])
            if idx_col is None or imp_col is None:
                continue

            top = df_feat.sort_values(imp_col, ascending=False).head(int(args.topk_feat_per_user))
            for _, r in top.iterrows():
                fi = int(r[idx_col])
                if 0 <= fi < len(feature_names):
                    imp = float(r[imp_col])
                    rec = {
                        "node_id": int(node_id),
                        "username": node_name,
                        "pos": int(explain_pos),
                        "feature": feature_names[fi],
                        "importance": imp,
                    }
                    rows.append(rec)
                    rows_user.append(rec)

    df_long = pd.DataFrame(rows)
    if df_long.empty:
        raise SystemExit("No explanations collected (df_long empty). Try fewer constraints or check MaskOpt output cols.")

    # ---------- aggregate: strength (median) and prevalence ----------
    # strength
    g_med = df_long.groupby(["pos","feature"], as_index=False)["importance"].median()
    piv_med = g_med.pivot(index="pos", columns="feature", values="importance").reindex(index=list(range(T_input))).fillna(0.0)

    # prevalence
    g_cnt = df_long.drop_duplicates(["node_id","pos","feature"]).groupby(["pos","feature"]).size().reset_index(name="count")
    num_users = df_long["node_id"].nunique()
    g_cnt["prevalence"] = g_cnt["count"] / float(num_users + 1e-9)
    piv_prev = g_cnt.pivot(index="pos", columns="feature", values="prevalence").reindex(index=list(range(T_input))).fillna(0.0)

    # choose features to show (by mean prevalence)
    overall_prev = piv_prev.mean(axis=0).sort_values(ascending=False)
    show_feats = overall_prev.head(int(args.show_top_features)).index.tolist()

    # outputs
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_long = os.path.join(args.outdir, f"agg_long_{ts}.csv")
    df_long.to_csv(out_long, index=False, encoding="utf-8-sig")
    print("[write]", out_long)

    out_med = os.path.join(args.outdir, f"heat_strength_median_{ts}.png")
    _plot_heatmap(
        piv_med[show_feats].values,
        show_feats,
        [f"pos{p}" for p in range(T_input)],
        title=f"Strength (median importance) | users={num_users}",
        out_png=out_med
    )
    print("[write]", out_med)

    out_prev = os.path.join(args.outdir, f"heat_prevalence_{ts}.png")
    _plot_heatmap(
        piv_prev[show_feats].values,
        show_feats,
        [f"pos{p}" for p in range(T_input)],
        title=f"Commonness (prevalence) | users={num_users}",
        out_png=out_prev,
        vmax_quantile=0.99
    )
    print("[write]", out_prev)

    # ---------- personalness per user (optional but useful) ----------
    # personal_score = imp * (1 - prevalence(pos,feature))
    df_u = pd.DataFrame(rows_user)
    # join prevalence
    df_prev_long = piv_prev.stack().reset_index()
    df_prev_long.columns = ["pos","feature","prevalence"]
    df_u = df_u.merge(df_prev_long, on=["pos","feature"], how="left")
    df_u["prevalence"] = df_u["prevalence"].fillna(0.0)
    df_u["personal_score"] = df_u["importance"] * (1.0 - df_u["prevalence"])

    out_personal = os.path.join(args.outdir, f"personalness_top_{ts}.csv")
    df_u.sort_values(["node_id","pos","personal_score"], ascending=[True, True, False]).to_csv(out_personal, index=False, encoding="utf-8-sig")
    print("[write]", out_personal)

    print("Done.")

if __name__ == "__main__":
    main()
