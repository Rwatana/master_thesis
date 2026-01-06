# InfluencerRank modular refactor (paper-ready script split)

This is a **file-splitted** version of your monolithic script, with the same main flow:
- MLflow setup
- data -> monthly graphs
- train (optional) -> infer
- XAI MaskOpt (neighbor grouping)

## Files
- `run_influencer_rank.py` : entry point (handles CUDA_VISIBLE_DEVICES before torch import)
- `influencer_rank/cli.py` : CLI parsing (NO torch imports)
- `influencer_rank/app.py` : high-level orchestration (sweep loop)
- `influencer_rank/experiment.py` : training/inference/XAI pipeline for one run
- `influencer_rank/data.py` : data loading + graph construction
- `influencer_rank/model.py` : GCN-LSTM-Attention-MLP model + helper
- `influencer_rank/xai_maskopt.py` : MaskOpt E2E explainer (+ neighbor grouping)
- `influencer_rank/mlflow_utils.py` : MLflow + checkpoint helpers
- `influencer_rank/plots.py` : plotting + MLflow plot logging
- `influencer_rank/losses.py` : ListMLE loss

## Usage
```bash
# train+infer+XAI
python run_influencer_rank.py --mode train --device 0

# infer-only (load checkpoint file)
python run_influencer_rank.py --mode infer --ckpt checkpoints/<run_name>/model_state.pt

# infer-only (download checkpoint from MLflow run)
python run_influencer_rank.py --mode infer --mlflow_run_id <RUN_ID> --mlflow_ckpt_artifact model/model_state.pt
```
