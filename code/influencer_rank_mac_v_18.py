#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InfluencerRank (Mac GPU version)
Version: v18 (MPS-compatible)

Changes:
- Load trained model from MLflow if available (skip training)
- Only explain edges directly connected to the influencer node
- Normalize node/edge importance and impact to [0, 1]
- Log scatter plot (importance vs impact) to MLflow artifacts
"""

import torch
import torch.nn.functional as F
import mlflow
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# =============================
# 1. DEVICE CONFIGURATION
# =============================
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[Device] Using: {DEVICE}")

# =============================
# 2. DUMMY MODEL (for simplicity)
# =============================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# =============================
# 3. LOAD FROM MLFLOW (Skip Training)
# =============================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("InfluencerRankSweep")

runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)

if len(runs) > 0:
    run_id = runs.loc[0, "run_id"]
    model_path = f"mlruns/1/{run_id}/artifacts/model.pt"
    if os.path.exists(model_path):
        print(f"[MLflow] Loading trained model from {model_path}")
        model = torch.load(model_path, map_location=DEVICE)
    else:
        print("[Warning] Model not found. Training required.")
        model = GCN(10, 16, 1).to(DEVICE)
else:
    print("[Warning] No previous runs found. Training new model.")
    model = GCN(10, 16, 1).to(DEVICE)

# =============================
# 4. SIMULATED GRAPH DATA
# =============================
N = 50
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 3], [1, 2, 3, 4, 5, 6, 0, 2]], dtype=torch.long)
x = torch.rand((N, 10))
data = Data(x=x, edge_index=edge_index)

# =============================
# 5. TARGET NODE EXPLANATION
# =============================
TARGET_NODE = 3
neighbors = set(edge_index[1, edge_index[0] == TARGET_NODE].tolist()) | set(edge_index[0, edge_index[1] == TARGET_NODE].tolist())
connected_edges = [(src.item(), dst.item()) for src, dst in zip(edge_index[0], edge_index[1]) if src.item() == TARGET_NODE or dst.item() == TARGET_NODE]
print(f"[XAI] Target node {TARGET_NODE} has {len(connected_edges)} connected edges.")

# =============================
# 6. MOCK IMPORTANCE CALCULATION
# =============================
node_importance = np.random.rand(x.shape[1])
edge_importance = np.random.rand(len(connected_edges))

# Normalize to [0, 1]
node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min())
edge_importance = (edge_importance - edge_importance.min()) / (edge_importance.max() - edge_importance.min())

# =============================
# 7. SCATTER PLOT (Node vs Edge Importance)
# =============================
plt.figure(figsize=(6,4))
plt.scatter(range(len(node_importance)), node_importance, label="Node Features", alpha=0.7)
plt.scatter(np.arange(len(edge_importance)) + len(node_importance), edge_importance, label="Edges", alpha=0.7)
plt.xlabel("Feature / Edge Index")
plt.ylabel("Normalized Importance (0-1)")
plt.legend()
plt.title(f"XAI Comparison for Node {TARGET_NODE}")
plt.tight_layout()
plt.savefig("xai_importance_scatter.png")

# Log to MLflow
with mlflow.start_run(run_id=run_id) as run:
    mlflow.log_artifact("xai_importance_scatter.png")

print("[Done] Logged scatter plot to MLflow.")