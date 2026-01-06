# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import gc
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_undirected, coalesce, k_hop_subgraph

from .model import gcn_forward_concat

def _binary_entropy(p, eps=1e-12):
    p = torch.clamp(p, eps, 1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))

class _DisableCudnn:
    def __enter__(self):
        self.prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
    def __exit__(self, exc_type, exc, tb):
        torch.backends.cudnn.enabled = self.prev
        return False

class E2EMaskOptWrapper(nn.Module):
    def __init__(
        self,
        model,
        input_graphs,
        target_node_idx,
        explain_pos,
        device,
        use_subgraph=True,
        num_hops=2,
        undirected=True,
        feat_mask_scope="target",
        edge_mask_scope="incident",
        edge_grouping="none",     # "none" | "neighbor"
        idx_to_node=None,         # {global_idx: "username/hashtag/object"}
    ):
        super().__init__()
        self.model = model
        self.input_graphs = input_graphs
        self.T = len(input_graphs)
        self.target_global = int(target_node_idx)
        self.explain_pos = int(explain_pos)
        self.device = device

        self.use_subgraph = use_subgraph
        self.num_hops = int(num_hops)
        self.undirected = bool(undirected)
        self.feat_mask_scope = feat_mask_scope
        self.edge_mask_scope = edge_mask_scope

        self.edge_grouping = edge_grouping
        self.idx_to_node = idx_to_node or {}

        self.edge_group_names = None
        self.edge_group_members = None
        self.edge_group_meta = None

        self.cached_proj = [None] * self.T
        self.cached_gcn  = [None] * self.T
        self._prepare_cache()
        self._prepare_explain_graph()

    def _prepare_cache(self):
        self.model.eval()
        with torch.no_grad():
            for t, g in enumerate(self.input_graphs):
                if t == self.explain_pos:
                    continue
                g = g.to(self.device)
                p = self.model.projection_layer(g.x)
                out = gcn_forward_concat(self.model.gcn_encoder, p, g.edge_index, edge_weight=None)
                self.cached_proj[t] = p[self.target_global].detach()
                self.cached_gcn[t]  = out[self.target_global].detach()

    def _prepare_explain_graph(self):
        g = self.input_graphs[self.explain_pos].to(self.device)
        x_full = g.x
        ei_full = g.edge_index

        if (not self.use_subgraph) or (ei_full.numel() == 0):
            self.x_exp = x_full
            self.ei_exp = ei_full
            self.target_local = self.target_global
            self.local2global = None
        else:
            subset, ei_sub, mapping, _ = k_hop_subgraph(
                self.target_global,
                self.num_hops,
                ei_full,
                relabel_nodes=True,
                num_nodes=x_full.size(0)
            )
            x_sub = x_full[subset]
            if self.undirected and ei_sub.numel() > 0:
                ei_sub = to_undirected(ei_sub, num_nodes=x_sub.size(0))
                ei_sub = coalesce(ei_sub, num_nodes=x_sub.size(0))
            self.x_exp = x_sub
            self.ei_exp = ei_sub
            self.target_local = int(mapping.item())
            self.local2global = subset

        if self.ei_exp.numel() == 0:
            self.incident_edge_idx = torch.empty(0, dtype=torch.long, device=self.device)
        else:
            src, dst = self.ei_exp
            incident = (src == self.target_local) | (dst == self.target_local)
            self.incident_edge_idx = torch.where(incident)[0]

        # neighbor grouping (only meaningful for incident scope)
        if (self.edge_mask_scope == "incident") and (self.edge_grouping == "neighbor") and (self.incident_edge_idx.numel() > 0):
            src, dst = self.ei_exp
            groups = {}  # neighbor_global -> list[edge_pos]
            for epos in self.incident_edge_idx.detach().cpu().tolist():
                s = int(src[epos].item())
                d = int(dst[epos].item())
                nbr_local = d if s == self.target_local else s
                if self.local2global is not None:
                    nbr_global = int(self.local2global[nbr_local].item())
                else:
                    nbr_global = int(nbr_local)
                groups.setdefault(nbr_global, []).append(epos)

            keys = sorted(groups.keys())
            self.edge_group_members = [groups[k] for k in keys]
            self.edge_group_names = [str(self.idx_to_node.get(k, f"node_{k}")) for k in keys]
            self.edge_group_meta = []
            for k in keys:
                self.edge_group_meta.append({
                    "neighbor_global": int(k),
                    "neighbor_name": str(self.idx_to_node.get(k, f"node_{k}")),
                    "num_edges_in_group": int(len(groups[k])),
                })
            self.num_edge_params = int(len(self.edge_group_members))
        else:
            if self.edge_mask_scope == "incident":
                self.num_edge_params = int(self.incident_edge_idx.numel())
            else:
                self.num_edge_params = int(self.ei_exp.size(1))

        self.feature_dim = int(self.x_exp.size(1))

    def num_mask_params(self):
        return self.feature_dim, self.num_edge_params

    def _apply_feature_gate(self, x, feat_gate):
        if self.feat_mask_scope in ("all", "subgraph"):
            return x * feat_gate.view(1, -1)

        n = x.size(0)
        sel = F.one_hot(
            torch.tensor(self.target_local, device=x.device),
            num_classes=n
        ).to(x.dtype).unsqueeze(1)  # [N,1]

        return x + sel * x * (feat_gate.view(1, -1) - 1.0)

    def _make_edge_weight(self, edge_gate):
        E = int(self.ei_exp.size(1))
        w = torch.ones(E, device=self.device)

        if E == 0 or edge_gate is None or edge_gate.numel() == 0:
            return w

        if (self.edge_mask_scope == "incident") and (self.edge_grouping == "neighbor") and (self.edge_group_members is not None):
            w = w.clone()
            for g, epos_list in enumerate(self.edge_group_members):
                if not epos_list:
                    continue
                idx = torch.tensor(epos_list, device=self.device, dtype=torch.long)
                w[idx] = edge_gate[g]
            return w

        if self.edge_mask_scope == "incident":
            w = w.clone()
            if self.incident_edge_idx.numel() > 0:
                w[self.incident_edge_idx] = edge_gate
            return w

        return edge_gate

    def predict_with_gates(self, feat_gate, edge_gate, edge_weight_override=None, x_override=None):
        seq_gcn, seq_raw = [], []
        for t in range(self.T):
            if t != self.explain_pos:
                seq_gcn.append(self.cached_gcn[t])
                seq_raw.append(self.cached_proj[t])
                continue

            x = x_override if x_override is not None else self.x_exp
            ei = self.ei_exp

            x_masked = self._apply_feature_gate(x, feat_gate)
            ew = self._make_edge_weight(edge_gate)
            if edge_weight_override is not None:
                ew = ew * edge_weight_override

            p = self.model.projection_layer(x_masked)
            out = gcn_forward_concat(self.model.gcn_encoder, p, ei, edge_weight=ew)
            seq_gcn.append(out[self.target_local])
            seq_raw.append(p[self.target_local])

        seq_gcn = torch.stack(seq_gcn, dim=0).unsqueeze(0)
        seq_raw = torch.stack(seq_raw, dim=0).unsqueeze(0)
        pred, _ = self.model(seq_gcn, seq_raw, baseline_scores=None)
        return pred.view(())

    @torch.no_grad()
    def original_pred(self):
        feat_gate = torch.ones(self.feature_dim, device=self.device)
        edge_gate = torch.ones(int(self.num_edge_params), device=self.device) if self.num_edge_params > 0 else None
        if edge_gate is not None and edge_gate.numel() == 0:
            edge_gate = None
        return float(self.predict_with_gates(feat_gate, edge_gate).item())

def maskopt_e2e_explain(
    model,
    input_graphs,
    target_node_idx,
    explain_pos,
    feature_names,
    node_to_idx=None,
    device=None,
    use_subgraph=True,
    num_hops=2,
    undirected=True,
    feat_mask_scope="target",
    edge_mask_scope="incident",
    epochs=300,
    lr=0.05,
    coeffs=None,
    print_every=50,
    topk_feat=20,
    topk_edge=30,
    min_show=1e-6,
    disable_cudnn_rnn=True,
    mlflow_log=False,
    fid_weight=100.0,
    use_contrastive=False,
    contrastive_margin=0.002,
    contrastive_weight=1.0,
    tag="pos_0",
    edge_grouping="none",              # "none" | "neighbor"
    impact_reference="masked",         # "masked" | "unmasked" | "both"
    budget_feat=None,
    budget_edge=None,
    budget_weight=0.0,
    eps_abs_feat=1e-9,
    eps_rel_feat=1e-6,
    eps_abs_edge=1e-9,
    eps_rel_edge=1e-6,
):
    assert len(input_graphs) >= 2, "input_graphs length must be >= 2"

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    T = len(input_graphs)
    if explain_pos < 0:
        explain_pos = (explain_pos + T) % T

    if coeffs is None:
        coeffs = {
            "edge_size": 0.05,
            "edge_ent": 0.10,
            "node_feat_size": 0.02,
            "node_feat_ent": 0.10
        }

    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    idx_to_node = None
    if node_to_idx is not None and isinstance(node_to_idx, dict):
        idx_to_node = {int(v): str(k) for k, v in node_to_idx.items()}

    wrapper = E2EMaskOptWrapper(
        model=model,
        input_graphs=input_graphs,
        target_node_idx=target_node_idx,
        explain_pos=explain_pos,
        device=device,
        use_subgraph=use_subgraph,
        num_hops=num_hops,
        undirected=undirected,
        feat_mask_scope=feat_mask_scope,
        edge_mask_scope=edge_mask_scope,
        edge_grouping=edge_grouping,
        idx_to_node=idx_to_node,
    )

    Fdim, Edim = wrapper.num_mask_params()
    feat_logits = nn.Parameter(0.1 * torch.randn(Fdim, device=device))
    edge_logits = nn.Parameter(0.1 * torch.randn(Edim, device=device)) if Edim > 0 else None
    mask_params = [feat_logits] + ([edge_logits] if edge_logits is not None else [])
    opt = torch.optim.Adam(mask_params, lr=lr)

    orig = float(wrapper.original_pred())
    orig_t = torch.tensor(orig, device=device)

    print(f"🧠 [MaskOpt] target_node={int(target_node_idx)} explain_pos={explain_pos}/{T-1} orig={orig:.6f}")
    print(f"   use_subgraph={use_subgraph}, num_hops={num_hops}, undirected={undirected}, feat_dim={Fdim}, edge_params={Edim}")
    print(f"   edge_grouping={edge_grouping}")

    cudnn_ctx = _DisableCudnn() if disable_cudnn_rnn else type("N", (), {"__enter__": lambda s: None, "__exit__": lambda s,a,b,c: False})()
    best = {"loss": float("inf"), "feat": None, "edge": None, "pred": None}

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        feat_gate = torch.sigmoid(feat_logits)
        edge_gate = torch.sigmoid(edge_logits) if (Edim > 0 and edge_logits is not None) else None

        with cudnn_ctx:
            pred = wrapper.predict_with_gates(feat_gate, edge_gate)

        loss_fid = (pred - orig_t) ** 2

        if use_contrastive:
            feat_gate_drop = (1.0 - feat_gate).clamp(0.0, 1.0)
            edge_gate_drop = (1.0 - edge_gate).clamp(0.0, 1.0) if edge_gate is not None else None
            with cudnn_ctx:
                pred_drop = wrapper.predict_with_gates(feat_gate_drop, edge_gate_drop)
            delta = (pred_drop - orig_t).abs()
            margin_t = torch.as_tensor(float(contrastive_margin), device=device, dtype=delta.dtype)
            loss_contrast = F.relu(margin_t - delta)
        else:
            delta = pred.new_zeros(())
            loss_contrast = pred.new_zeros(())

        loss_feat_size = feat_gate.mean()
        loss_feat_ent  = _binary_entropy(feat_gate).mean()

        if edge_gate is not None and edge_gate.numel() > 0:
            loss_edge_size = edge_gate.mean()
            loss_edge_ent  = _binary_entropy(edge_gate).mean()
        else:
            loss_edge_size = pred.new_zeros(())
            loss_edge_ent  = pred.new_zeros(())

        def _budget_loss(gate, budget, denom):
            if (budget is None) or (gate is None) or (gate.numel() == 0):
                return gate.new_zeros(())
            return ((gate.sum() - float(budget)) / float(max(1, denom))) ** 2

        loss_budget = pred.new_zeros(())
        if float(budget_weight) > 0.0:
            loss_budget = (
                _budget_loss(feat_gate, budget_feat, Fdim)
                + _budget_loss(edge_gate, budget_edge, max(1, Edim))
            )

        loss = (
            float(fid_weight) * loss_fid
            + float(contrastive_weight) * loss_contrast
            + float(budget_weight) * loss_budget
            + coeffs["node_feat_size"] * loss_feat_size
            + coeffs["node_feat_ent"]  * loss_feat_ent
            + coeffs["edge_size"]      * loss_edge_size
            + coeffs["edge_ent"]       * loss_edge_ent
        )

        loss.backward()
        opt.step()

        lval = float(loss.item())
        if lval < best["loss"]:
            best["loss"] = lval
            best["feat"] = feat_gate.detach().clone()
            best["edge"] = edge_gate.detach().clone() if edge_gate is not None else None
            best["pred"] = float(pred.detach().item())

        if (ep == 1) or (ep % int(print_every) == 0) or (ep == int(epochs)):
            feat_max = float(feat_gate.max().item()) if feat_gate.numel() > 0 else 0.0
            edge_max = float(edge_gate.max().item()) if (edge_gate is not None and edge_gate.numel() > 0) else 0.0
            print(
                f"  [MaskOpt] ep={ep:4d} loss={lval:.6e} fid={float(loss_fid.item()):.3e} "
                f"drop_d={float(delta.item()):.3e} pred={float(pred.item()):.6f} feat_max={feat_max:.4f} edge_max={edge_max:.4f}"
            )

    feat_gate = best["feat"].clamp(0.0, 1.0) if best["feat"] is not None else None
    edge_gate = best["edge"].clamp(0.0, 1.0) if best["edge"] is not None else None

    ones_feat = torch.ones(Fdim, device=device)
    ones_edge = torch.ones(Edim, device=device) if Edim > 0 else None

    def _thr(pred_base, eps_abs, eps_rel):
        return max(float(eps_abs), float(eps_rel) * abs(float(pred_base)))

    def _direction(diff, pred_base, eps_abs, eps_rel):
        th = _thr(pred_base, eps_abs, eps_rel)
        if abs(diff) <= th:
            return "Zero (0)"
        return "Positive (+)" if diff > 0 else "Negative (-)"

    with torch.no_grad():
        with cudnn_ctx:
            pred_unmasked = float(wrapper.predict_with_gates(ones_feat, ones_edge).item())
            pred_masked   = float(wrapper.predict_with_gates(feat_gate, edge_gate).item()) if (feat_gate is not None) else pred_unmasked

    # Feature table (topk by gate)
    feat_rows = []
    if feat_gate is not None and feat_gate.numel() > 0:
        feat_np = feat_gate.detach().cpu().numpy()
        top_feat_idx = np.argsort(feat_np)[::-1][:int(topk_feat)]
        for j in top_feat_idx:
            imp = float(feat_np[j])
            if imp < float(min_show):
                continue
            refs = []
            if impact_reference in ("unmasked", "both"):
                ab_f = ones_feat.clone(); ab_f[j] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(ab_f, ones_edge).item())
                diff = pred_unmasked - pred_abl
                refs.append(("unmasked", diff, _direction(diff, pred_unmasked, eps_abs_feat, eps_rel_feat)))
            if impact_reference in ("masked", "both"):
                base_f = feat_gate.clone()
                base_e = edge_gate.clone() if edge_gate is not None else None
                ab_f = base_f.clone(); ab_f[j] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(ab_f, base_e).item())
                diff = pred_masked - pred_abl
                refs.append(("masked", diff, _direction(diff, pred_masked, eps_abs_feat, eps_rel_feat)))
            name = feature_names[j] if j < len(feature_names) else f"feat_{j}"
            row = {"Type":"Feature","Name":str(name),"Importance":imp}
            for key, diff, direc in refs:
                row[f"Score_Impact({key})"] = float(diff)
                row[f"Direction({key})"] = direc
            feat_rows.append(row)

    df_feat = pd.DataFrame(feat_rows)
    if not df_feat.empty:
        df_feat.insert(0, "target_node", int(target_node_idx))
        df_feat.insert(1, "explain_pos", int(explain_pos))
        df_feat.insert(2, "tag", str(tag))
        df_feat = df_feat.sort_values("Importance", ascending=False).reset_index(drop=True)

    # Edge/group table
    edge_rows = []
    if edge_gate is not None and edge_gate.numel() > 0:
        edge_np = edge_gate.detach().cpu().numpy()
        top_edge_idx = np.argsort(edge_np)[::-1][:int(topk_edge)]

        group_names = None
        if getattr(wrapper, "edge_group_names", None) is not None and len(wrapper.edge_group_names) == int(Edim):
            group_names = list(wrapper.edge_group_names)
        if group_names is None:
            group_names = [f"edge_group_{i}" for i in range(int(Edim))]

        for gidx in top_edge_idx:
            imp = float(edge_np[gidx])
            if imp < float(min_show):
                continue
            refs = []
            if impact_reference in ("unmasked", "both"):
                ab_e = ones_edge.clone() if ones_edge is not None else None
                if ab_e is not None:
                    ab_e[gidx] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(ones_feat, ab_e).item())
                diff = pred_unmasked - pred_abl
                refs.append(("unmasked", diff, _direction(diff, pred_unmasked, eps_abs_edge, eps_rel_edge)))
            if impact_reference in ("masked", "both"):
                base_f = feat_gate.clone() if feat_gate is not None else ones_feat.clone()
                base_e = edge_gate.clone()
                ab_e = base_e.clone(); ab_e[gidx] = 0.0
                with torch.no_grad():
                    with cudnn_ctx:
                        pred_abl = float(wrapper.predict_with_gates(base_f, ab_e).item())
                diff = pred_masked - pred_abl
                refs.append(("masked", diff, _direction(diff, pred_masked, eps_abs_edge, eps_rel_edge)))
            nm = group_names[gidx] if gidx < len(group_names) else f"edge_group_{gidx}"
            row = {"Type":"Edge","Name":str(nm),"Importance":imp}
            for key, diff, direc in refs:
                row[f"Score_Impact({key})"] = float(diff)
                row[f"Direction({key})"] = direc
            edge_rows.append(row)

    df_edge = pd.DataFrame(edge_rows)
    if not df_edge.empty:
        df_edge.insert(0, "target_node", int(target_node_idx))
        df_edge.insert(1, "explain_pos", int(explain_pos))
        df_edge.insert(2, "tag", str(tag))
        df_edge = df_edge.sort_values("Importance", ascending=False).reset_index(drop=True)

    meta = {
        "orig_pred": float(orig),
        "best_pred": float(best["pred"]),
        "best_loss": float(best["loss"]),
        "target_node": int(target_node_idx),
        "explain_pos": int(explain_pos),
        "T": int(T),
        "feat_dim": int(Fdim),
        "edge_params": int(Edim),
        "edge_grouping": str(edge_grouping),
        "pred_unmasked": float(pred_unmasked),
        "pred_masked": float(pred_masked),
        "impact_reference": str(impact_reference),
        "budget_feat": None if budget_feat is None else float(budget_feat),
        "budget_edge": None if budget_edge is None else float(budget_edge),
        "budget_weight": float(budget_weight),
        "coeffs": dict(coeffs),
        "fid_weight": float(fid_weight),
    }

    if mlflow_log:
        try:
            import mlflow as _ml
            import json

            feat_csv = f"maskopt_feat_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
            edge_csv = f"maskopt_edge_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
            df_feat.to_csv(feat_csv, index=False, float_format="%.8e")
            df_edge.to_csv(edge_csv, index=False, float_format="%.8e")
            _ml.log_artifact(feat_csv, artifact_path="xai")
            _ml.log_artifact(edge_csv, artifact_path="xai")
            os.remove(feat_csv); os.remove(edge_csv)

            gates_npz = f"maskopt_gates_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.npz"
            np.savez_compressed(
                gates_npz,
                feat_gate=feat_gate.detach().cpu().numpy().astype(np.float32) if feat_gate is not None else None,
                edge_gate=edge_gate.detach().cpu().numpy().astype(np.float32) if edge_gate is not None else None,
                meta=meta,
            )
            _ml.log_artifact(gates_npz, artifact_path="xai")
            os.remove(gates_npz)

            if (edge_grouping == "neighbor") and (getattr(wrapper, "edge_group_meta", None) is not None):
                m = pd.DataFrame(wrapper.edge_group_meta)
                m.insert(0, "group_id", np.arange(len(m), dtype=int))
                m.insert(1, "target_node", int(target_node_idx))
                m.insert(2, "explain_pos", int(explain_pos))
                map_csv = f"maskopt_edge_group_map_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.csv"
                m.to_csv(map_csv, index=False)
                _ml.log_artifact(map_csv, artifact_path="xai")
                os.remove(map_csv)

            mpath = f"maskopt_meta_{tag}_node_{int(target_node_idx)}_pos_{int(explain_pos)}.json"
            with open(mpath, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            _ml.log_artifact(mpath, artifact_path="xai")
            os.remove(mpath)

        except Exception as e:
            print(f"⚠️ MLflow log failed: {e}")

    return df_feat, df_edge, meta

def compute_time_step_sensitivity(model, input_graphs, target_node_idx, device, topk=3,
                                  score_mode="alpha_x_delta", min_delta=1e-6):
    model = model.to(device)
    model.eval()
    T = len(input_graphs)
    target_node_idx = int(target_node_idx)

    with torch.no_grad():
        seq_gcn, seq_raw = [], []
        for g in input_graphs:
            g = g.to(device)
            p = model.projection_layer(g.x)
            h = gcn_forward_concat(model.gcn_encoder, p, g.edge_index, edge_weight=None)
            seq_gcn.append(h[target_node_idx])
            seq_raw.append(p[target_node_idx])
        seq_gcn = torch.stack(seq_gcn, dim=0).unsqueeze(0)
        seq_raw = torch.stack(seq_raw, dim=0).unsqueeze(0)

        pred_full_t, alpha_t = model(seq_gcn, seq_raw, baseline_scores=None)
        pred_full = float(pred_full_t.view(()).item())

        if alpha_t is None:
            alpha = np.ones(T, dtype=np.float32) / float(T)
        else:
            alpha = alpha_t.detach().view(-1).cpu().numpy().astype(np.float32)
            if alpha.size != T:
                alpha = np.ones(T, dtype=np.float32) / float(T)

        rows = []
        for t in range(T):
            sg = seq_gcn.clone()
            sr = seq_raw.clone()
            sg[:, t, :] = 0.0
            sr[:, t, :] = 0.0
            pred_drop_t, _ = model(sg, sr, baseline_scores=None)
            pred_drop = float(pred_drop_t.view(()).item())
            delta = abs(pred_full - pred_drop)
            if delta < float(min_delta):
                delta = 0.0

            if score_mode == "delta":
                score = delta
            elif score_mode == "alpha":
                score = float(alpha[t])
            else:
                score = float(alpha[t]) * delta

            rows.append({
                "pos": int(t),
                "alpha": float(alpha[t]),
                "pred_full": float(pred_full),
                "pred_drop": float(pred_drop),
                "delta_total": float(delta),
                "score": float(score),
            })

    sens_df = pd.DataFrame(rows).sort_values(["score","delta_total","alpha"], ascending=False).reset_index(drop=True)
    selected_positions = sens_df.head(int(topk))["pos"].astype(int).tolist()
    return sens_df, selected_positions, pred_full, alpha
