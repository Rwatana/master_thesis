# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import shutil
import datetime
import random
import gc
import itertools

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

import mlflow

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from torch_geometric.utils import degree

from .model import HardResidualInfluencerModel
from .losses import ListMLELoss
from .data import get_dataset_with_baseline
from .mlflow_utils import save_model_checkpoint, maybe_download_ckpt_from_mlflow, load_model_from_ckpt
from .plots import (
    plot_attention_weights,
    generate_enhanced_scatter_plot,
    mlflow_log_pred_scatter,
    mlflow_log_maskopt_plots,
)
from .xai_maskopt import compute_time_step_sensitivity, maskopt_e2e_explain

def _select_positions_by_attention(attn_w, num_positions, topk=3, min_w=0.0):
    import torch
    if topk is None or topk <= 0 or topk >= num_positions:
        return list(range(num_positions))
    if attn_w is None:
        return list(range(min(topk, num_positions)))
    attn_w = attn_w.detach().float().flatten()
    T = min(num_positions, attn_w.numel())
    attn_w = attn_w[:T]
    if min_w > 0:
        keep = torch.nonzero(attn_w >= min_w, as_tuple=False).flatten().tolist()
        if len(keep) == 0:
            keep = torch.topk(attn_w, k=min(topk, T)).indices.tolist()
        w_keep = attn_w[keep]
        order = torch.argsort(w_keep, descending=True).tolist()
        return [keep[i] for i in order[:min(topk, len(keep))]]
    return torch.topk(attn_w, k=min(topk, T)).indices.tolist()

def run_experiment(params: dict, graphs_data: tuple, device: torch.device, experiment_id=None):
    """Returns: (run_id, final_test_metrics_dict)."""
    run_id = None
    final_test_metrics = None

    monthly_graphs, influencer_indices, node_to_idx, feature_dim, follower_feat_idx, static_cols, dynamic_cols = graphs_data
    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    run_name = f"{params.get('name_prefix', 'Run')}_{current_time_str}"

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        run_id = mlflow.active_run().info.run_id
        mlflow.log_params(params)
        print(f"\n🚀 Starting MLflow Run: {run_name}")
        if 'note' in params:
            print(f"Note: {params['note']}")

        model = HardResidualInfluencerModel(
            feature_dim=feature_dim,
            gcn_dim=params['GCN_DIM'],
            rnn_dim=params['RNN_DIM'],
            num_gcn_layers=params['NUM_GCN_LAYERS'],
            dropout_prob=params['DROPOUT_PROB'],
            projection_dim=params['PROJECTION_DIM']
        ).to(device)

        mode_run = str(params.get("MODE", "train")).lower()
        ckpt_path = params.get("CKPT_PATH")
        ckpt_run_id = params.get("CKPT_MLFLOW_RUN_ID")
        ckpt_art = params.get("CKPT_MLFLOW_ARTIFACT", "model/model_state.pt")

        if mode_run == "infer":
            if ckpt_run_id:
                ckpt_path = maybe_download_ckpt_from_mlflow(str(ckpt_run_id), str(ckpt_art))
            if not ckpt_path:
                raise ValueError("infer mode requires --ckpt or --mlflow_run_id")
            loaded_model, loaded_feature_dim, _ = load_model_from_ckpt(str(ckpt_path), device=device)
            if int(loaded_feature_dim) != int(feature_dim):
                raise ValueError(f"feature_dim mismatch: ckpt={loaded_feature_dim} vs current={feature_dim}")
            model.load_state_dict(loaded_model.state_dict(), strict=True)
            model.eval()
            mlflow.log_param("infer_only", 1)
            mlflow.log_param("ckpt_path", str(ckpt_path))
            print(f"[InferOnly] loaded ckpt={ckpt_path}")

        optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'])
        criterion_list = ListMLELoss().to(device)
        criterion_mse = nn.MSELoss().to(device)

        train_dataset = get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-2)

        sampler = None
        if params.get('USE_SAMPLER', False):
            targets_for_weight = train_dataset.tensors[1].cpu().numpy()
            weights = [5.0 if t > 0.01 else 1.0 for t in targets_for_weight]
            sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_input_graphs = monthly_graphs[:-2]
        gpu_graphs = [g.to(device) for g in train_input_graphs]

        inf_global = torch.tensor(influencer_indices, dtype=torch.long, device=device)
        num_nodes_all = int(gpu_graphs[0].num_nodes)
        global2local = torch.full((num_nodes_all,), -1, dtype=torch.long, device=device)
        global2local[inf_global] = torch.arange(inf_global.numel(), device=device, dtype=torch.long)
        if int((global2local[inf_global] < 0).sum().item()) != 0:
            raise RuntimeError("global2local mapping failed for some influencer indices.")

        if mode_run != "infer":
            print("Starting Training...")
            model.train()

            list_size = int(params["LIST_SIZE"])
            if len(train_dataset) < list_size:
                raise RuntimeError(f"train_dataset too small ({len(train_dataset)}) for LIST_SIZE={list_size}")

            safe_bs = min(int(params["BATCH_SIZE"]), len(train_dataset))
            safe_bs = (safe_bs // list_size) * list_size
            safe_bs = max(list_size, safe_bs)

            dataloader = DataLoader(
                train_dataset,
                batch_size=safe_bs,
                sampler=sampler,
                shuffle=(sampler is None),
                drop_last=True
            )

            for epoch in range(int(params["EPOCHS"])):
                model.train()
                total_loss = 0.0
                optimizer.zero_grad(set_to_none=True)

                seq_emb, raw_emb = [], []
                for g in gpu_graphs:
                    p_x = model.projection_layer(g.x)
                    gcn_out = model.gcn_encoder(p_x, g.edge_index)
                    raw_emb.append(p_x.index_select(0, inf_global))
                    seq_emb.append(gcn_out.index_select(0, inf_global))

                full_seq = torch.stack(seq_emb, dim=1)  # [Ninf,T,D]
                full_raw = torch.stack(raw_emb, dim=1)  # [Ninf,T,P]

                loss_sum = None
                num_batches = 0

                for batch in dataloader:
                    b_idx_global, b_target, b_baseline = batch
                    b_idx_global = b_idx_global.to(device)
                    b_target = b_target.to(device)
                    b_baseline = b_baseline.to(device)

                    b_local = global2local[b_idx_global]
                    if int((b_local < 0).sum().item()) != 0:
                        raise RuntimeError("Found indices not in influencer set (global2local == -1).")

                    b_seq = full_seq.index_select(0, b_local)
                    b_raw = full_raw.index_select(0, b_local)

                    preds, _ = model(b_seq, b_raw, baseline_scores=b_baseline)
                    preds = preds.view(-1)

                    log_target = torch.log1p(b_target * 100.0)
                    log_pred = torch.log1p(preds * 100.0)

                    loss_rank = criterion_list(
                        preds.view(-1, list_size),
                        log_target.view(-1, list_size)
                    )
                    loss_mse = criterion_mse(log_pred, log_target)

                    loss = loss_rank + loss_mse * float(params.get("POINTWISE_LOSS_WEIGHT", 1.0))

                    total_loss += float(loss.item())
                    num_batches += 1
                    loss_sum = loss if loss_sum is None else (loss_sum + loss)

                if loss_sum is not None:
                    (loss_sum / float(max(1, num_batches))).backward()
                optimizer.step()

                del full_seq, full_raw, seq_emb, raw_emb
                if (epoch + 1) % 10 == 0:
                    avg_loss = total_loss / max(1, num_batches)
                    mlflow.log_metric("train_loss", avg_loss, step=epoch + 1)
                    print(f"Epoch {epoch+1}/{params['EPOCHS']} Loss: {avg_loss:.4f}")
        else:
            print("[InferOnly] training skipped")

        # checkpoint (only when trained)
        if mode_run != "infer":
            try:
                os.makedirs(os.path.join("checkpoints", run_name), exist_ok=True)
                print("\nSaving Model Checkpoint...")
                ckpt_local = os.path.join("checkpoints", run_name, "model_state.pt")
                ckpt_local, cfg_local = save_model_checkpoint(
                    model, params, feature_dim=feature_dim, out_path=ckpt_local
                )
                ckpt_pth = os.path.splitext(ckpt_local)[0] + ".pth"
                try:
                    shutil.copy2(ckpt_local, ckpt_pth)
                except Exception:
                    torch.save(torch.load(ckpt_local, map_location="cpu"), ckpt_pth)

                try:
                    mlflow.log_artifact(ckpt_local, artifact_path="model")
                    mlflow.log_artifact(ckpt_pth, artifact_path="model")
                    mlflow.log_artifact(cfg_local, artifact_path="model")
                    print("[Checkpoint] logged artifacts: model/model_state.pt, model/model_state.pth, model/model_config.json")
                except Exception as e:
                    print(f"⚠️ [Checkpoint] MLflow log failed (local checkpoint is kept): {e}")

                print(f"[Checkpoint] local saved: {ckpt_local} (+ {ckpt_pth}) and {cfg_local}")
            except Exception as e:
                print(f"⚠️ [Checkpoint] save/log failed: {e}")

        # ----- Inference -----
        print("\nStarting Inference...")
        model.eval()

        test_dataset = get_dataset_with_baseline(monthly_graphs, influencer_indices, target_idx=-1)
        all_indices = test_dataset.tensors[0]
        all_targets = test_dataset.tensors[1]
        all_baselines = test_dataset.tensors[2]

        inf_input_graphs = monthly_graphs[:-1]

        with torch.no_grad():
            seq_emb_l, raw_emb_l = [], []
            for g in inf_input_graphs:
                g = g.to(device)
                p_x = model.projection_layer(g.x)
                gcn_out = model.gcn_encoder(p_x, g.edge_index)
                raw_emb_l.append(p_x.cpu())
                seq_emb_l.append(gcn_out.cpu())

            f_seq = torch.stack(seq_emb_l)[:, influencer_indices].permute(1, 0, 2)
            f_raw = torch.stack(raw_emb_l)[:, influencer_indices].permute(1, 0, 2)

            preds_all, attn_all = [], []
            infer_batch_size = 1024
            for i in range(0, len(all_indices), infer_batch_size):
                end = min(i + infer_batch_size, len(all_indices))
                b_seq = f_seq[i:end].to(device)
                b_raw = f_raw[i:end].to(device)
                b_base = all_baselines[i:end].to(device)
                p, attn = model(b_seq, b_raw, b_base)
                preds_all.append(p.cpu())
                attn_all.append(attn.cpu())

            predicted_scores = torch.cat(preds_all).squeeze().numpy()
            attention_matrix = torch.cat(attn_all).squeeze().cpu().numpy()

            true_scores = all_targets.cpu().numpy()
            baseline_scores = all_baselines.cpu().numpy()

        # ----- Plots -----
        print("\n--- 📊 Generating and Logging Plots ---")
        last_input_graph = inf_input_graphs[-1]
        follower_counts = last_input_graph.x[all_indices, follower_feat_idx].cpu().numpy()
        log_follower_counts = follower_counts  # already log1p

        epsilon = 1e-9
        true_growth = (true_scores - baseline_scores) / (baseline_scores + epsilon)
        pred_growth = (predicted_scores - baseline_scores) / (baseline_scores + epsilon)

        plot_files = []
        plot_files.append(generate_enhanced_scatter_plot(true_scores, predicted_scores,
                                                        "True Engagement Score", "Predicted Score",
                                                        run_name, "score_basic"))

        att_bar_file, att_heat_file, att_csv_file, att_raw_file = plot_attention_weights(attention_matrix, run_name)
        for f in [att_bar_file, att_heat_file, att_csv_file, att_raw_file]:
            if f is None:
                continue
            mlflow.log_artifact(f)
            os.remove(f)

        plot_files.append(generate_enhanced_scatter_plot(true_scores, predicted_scores,
                                                        "True Engagement Score", "Predicted Score",
                                                        run_name, "score_by_followers",
                                                        color_data=log_follower_counts,
                                                        color_label="log1p(Followers)",
                                                        title_suffix="(Colored by Followers)"))

        plot_files.append(generate_enhanced_scatter_plot(true_scores, predicted_scores,
                                                        "True Engagement Score", "Predicted Score",
                                                        run_name, "score_by_growth",
                                                        color_data=true_growth,
                                                        color_label="True Growth Rate",
                                                        title_suffix="(Colored by Growth Rate)"))

        plot_files.append(generate_enhanced_scatter_plot(true_growth, pred_growth,
                                                        "True Growth Rate", "Predicted Growth Rate",
                                                        run_name, "growth_by_followers",
                                                        color_data=log_follower_counts,
                                                        color_label="log1p(Followers)",
                                                        title_suffix="(Colored by Followers)"))

        for f in plot_files:
            if f is not None and os.path.exists(f):
                mlflow.log_artifact(f)
                os.remove(f)

        # ----- Metrics -----
        print("\n--- 📊 Evaluation Metrics ---")
        mae = mean_absolute_error(true_scores, predicted_scores)
        rmse = float(np.sqrt(mean_squared_error(true_scores, predicted_scores)))
        p_corr, _ = pearsonr(true_scores, predicted_scores)
        s_corr, _ = spearmanr(true_scores, predicted_scores)

        mlflow.log_metrics({"mae": float(mae), "rmse": float(rmse), "pearson_corr": float(p_corr), "spearman_corr": float(s_corr)})
        print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, Pearson: {p_corr:.4f}, Spearman: {s_corr:.4f}")

        final_test_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "pearson_corr": float(p_corr),
            "spearman_corr": float(s_corr),
        }

        # ----- Explanation target: hub influencer in Nov graph -----
        feature_names = static_cols + dynamic_cols
        target_graph = monthly_graphs[-2]  # Nov
        edge_index = target_graph.edge_index.to(device)
        d = degree(edge_index[1], num_nodes=target_graph.num_nodes)

        max_degree = -1
        target_node_global_idx = -1
        for idx in influencer_indices:
            deg = d[idx].item()
            if deg > max_degree:
                max_degree = deg
                target_node_global_idx = idx

        print(f"\n🎯 Selected Hub User (Node {target_node_global_idx}) with {int(max_degree)} edges.")

        input_graphs = monthly_graphs[:-1]  # Jan..Nov (T=11)
        T = len(input_graphs)

        sens_df, sens_selected = None, None

        try:
            attn_w = None
            pos_in_all = (all_indices == target_node_global_idx).nonzero(as_tuple=False)
            if pos_in_all.numel() > 0:
                row = int(pos_in_all[0].item())
                attn_w = torch.tensor(attention_matrix[row], dtype=torch.float32)

            if params.get("explain_use_sensitivity", True):
                try:
                    sens_df, sens_selected, _pred_full, _alpha = compute_time_step_sensitivity(
                        model=model,
                        input_graphs=input_graphs,
                        target_node_idx=target_node_global_idx,
                        device=device,
                        topk=int(params.get("xai_topk_pos", 3)),
                        score_mode=str(params.get("sensitivity_score_mode", "alpha_x_delta")),
                        min_delta=float(params.get("sensitivity_min_delta", 1e-4)),
                    )
                except Exception as e:
                    print(f"[Explain] sensitivity computation skipped: {e}")
                    sens_df, sens_selected = None, None

            if attn_w is None:
                positions_attn = list(range(min(3, T)))
            else:
                positions_attn = _select_positions_by_attention(
                    attn_w, T,
                    topk=int(params.get("xai_topk_pos", 3)),
                    min_w=float(params.get("xai_attn_min_w", 0.0)),
                )

            positions_to_explain = positions_attn
            if sens_selected is not None and len(sens_selected) > 0:
                positions_to_explain = sens_selected[: int(params.get("xai_topk_pos", 3))]

            print(f"[Explain] positions_to_explain={positions_to_explain} / T={T}")

            mlflow.log_param("xai_positions", ",".join(map(str, positions_to_explain)))
            mlflow.log_param("xai_target_node", int(target_node_global_idx))

            # run MaskOpt for selected months
            for explain_pos in positions_to_explain:
                tag = f"pos_{int(explain_pos)}"
                df_feat, df_edge, meta = maskopt_e2e_explain(
                    model=model,
                    input_graphs=input_graphs,
                    target_node_idx=target_node_global_idx,
                    explain_pos=int(explain_pos),
                    feature_names=feature_names,
                    node_to_idx=node_to_idx,
                    device=device,
                    use_subgraph=True,
                    num_hops=1,
                    edge_mask_scope="incident",
                    edge_grouping="neighbor",
                    fid_weight=2000.0,
                    coeffs={"edge_size":0.08,"edge_ent":0.15,"node_feat_size":0.02,"node_feat_ent":0.15},
                    budget_feat=10, budget_edge=20, budget_weight=1.0,
                    impact_reference="masked",
                    use_contrastive=False,
                    mlflow_log=True,
                    tag=tag,
                )

                mlflow_log_maskopt_plots(
                    df_feat=df_feat,
                    df_edge=df_edge,
                    meta=meta,
                    tag=tag,
                    topk_feat=15,
                    topk_edge=15,
                    artifact_path="xai",
                    fname_prefix=f"node_{target_node_global_idx}",
                )
        except Exception as e:
            print(f"💥 Explanation Error: {e}")

        mlflow_log_pred_scatter(
            y_true=true_scores,
            y_pred=predicted_scores,
            tag="test_dec2017",
            step=params.get("EPOCHS", None),
            artifact_path="plots",
        )

        # Cleanup
        del model, optimizer, criterion_list, criterion_mse
        if 'gpu_graphs' in locals():
            del gpu_graphs
        if 'f_seq' in locals():
            del f_seq
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🧹 Memory cleared after {run_name}.")

    return run_id, final_test_metrics
