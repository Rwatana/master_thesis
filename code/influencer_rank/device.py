# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import torch

def get_device(requested_idx: int) -> torch.device:
    # CUDA
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[Device] torch sees {n} CUDA device(s). CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
        for i in range(n):
            try:
                print(f"  cuda:{i} -> {torch.cuda.get_device_name(i)}")
            except Exception:
                print(f"  cuda:{i} -> (name unavailable)")

        if requested_idx < 0:
            return torch.device("cpu")
        if requested_idx >= n:
            print(f"[Device] WARNING: requested cuda:{requested_idx} but only 0..{n-1} available. Fallback cuda:0")
            requested_idx = 0

        torch.cuda.set_device(requested_idx)
        return torch.device(f"cuda:{requested_idx}")

    # MPS (mac) fallback (best-effort)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[Device] CUDA not available -> try MPS")
        try:
            _ = torch.tensor([1.0], device="mps")
            return torch.device("mps")
        except Exception as e:
            print(f"[Device] MPS not usable ({e}) -> CPU")

    print("[Device] -> CPU")
    return torch.device("cpu")
