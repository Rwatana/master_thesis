# Streamlit XAI Dashboard (InfluencerRank)

## Files
- `streamlit_xai_app.py` : Streamlit app
- expects your script next to it (default):
  - `influencer_rank_full_fixed_xai_paper_no_imagecsv_v19_lossA_logrank_sampler_v2_log_scale.py`

## Required data files (in your Data directory)
- `dataset_A_active_all.csv`
- `hashtags_2017.csv`
- `mentions_2017.csv`
- `influencers.txt`
- (optional) `image_features_v2_full_fixed.csv`

## Run
```bash
pip install streamlit networkx matplotlib pandas numpy torch torch-geometric
streamlit run streamlit_xai_app.py
```

## Tips (practical)
- If XAI is slow: lower `epochs` (e.g., 80–150) and keep `Use subgraph` enabled.
- Cached XAI results are stored under `./xai_cache/` inside your Data directory.
- Evidence view is “best-effort”: it will show example posts by matching `token` in `caption`.


## Added (Jan 2026)
- "Score impact (pos)" tab: compare score-based impacts across past months (e.g., past 11 months) and track a feature/edge neighbor's score impact over time.
