(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v3.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Total feature dimension: 36
Building monthly graphs: 100%|████████████████████████████████████████████████████████████████████████████| 12/12 [02:05<00:00, 10.42s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0          15552     45.83%
1           6450     19.01%
2           5073     14.95%
3           2892      8.52%
4           2252      6.64%
5           1716      5.06%

--- Strategy: Two-Stage Learning (Fast) ---
GCN Encoding: 100%|███████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:40<00:00,  3.38s/it]
Epoch 1/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  4.64it/s]
Epoch 1/20, Average Batch Loss: 2.3084
Epoch 2/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  4.60it/s]
Epoch 2/20, Average Batch Loss: 2.3044
Epoch 3/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.84it/s]
Epoch 3/20, Average Batch Loss: 2.3034
Epoch 4/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.82it/s]
Epoch 4/20, Average Batch Loss: 2.3029
Epoch 5/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.28it/s]
Epoch 5/20, Average Batch Loss: 2.3026
Epoch 6/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.84it/s]
Epoch 6/20, Average Batch Loss: 2.3026
Epoch 7/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.87it/s]
Epoch 7/20, Average Batch Loss: 2.3025
Epoch 8/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.83it/s]
Epoch 8/20, Average Batch Loss: 2.3025
Epoch 9/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.87it/s]
Epoch 9/20, Average Batch Loss: 2.3025
Epoch 10/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.82it/s]
Epoch 10/20, Average Batch Loss: 2.3025
Epoch 11/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.82it/s]
Epoch 11/20, Average Batch Loss: 2.3025
Epoch 12/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.28it/s]
Epoch 12/20, Average Batch Loss: 2.3025
Epoch 13/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.87it/s]
Epoch 13/20, Average Batch Loss: 2.3025
Epoch 14/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.82it/s]
Epoch 14/20, Average Batch Loss: 2.3025
Epoch 15/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.81it/s]
Epoch 15/20, Average Batch Loss: 2.3025
Epoch 16/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.81it/s]
Epoch 16/20, Average Batch Loss: 2.3025
Epoch 17/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.26it/s]
Epoch 17/20, Average Batch Loss: 2.3025
Epoch 18/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  3.80it/s]
Epoch 18/20, Average Batch Loss: 2.3025
Epoch 19/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  4.13it/s]
Epoch 19/20, Average Batch Loss: 2.3025
Epoch 20/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  4.21it/s]
Epoch 20/20, Average Batch Loss: 2.3025

--- Training Complete ---
✅ Model saved to 'influencer_rank_model_20251019_rich_features_2017.pth'
Total time: 500.76 seconds
--- 📈 Starting Inference Process ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Total feature dimension: 36
Building monthly graphs: 100%|████████████████████████████████████████████████████████████████████████████| 12/12 [02:20<00:00, 11.69s/it]
/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py:334: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
Successfully loaded model from 'influencer_rank_model_20251019_rich_features_2017.pth'

🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆
               Username  Predicted_Score  True_Score
          rebeccazamolo        -0.134286    0.036693
           nala_rinaldo        -0.138705    0.009103
      travelingteenybug        -0.140448    0.052606
             ryanadams9        -0.140624    0.042784
              alexslens        -0.141324    0.000000
           momtalkradio        -0.141663    0.051318
spaceshipsandlaserbeams        -0.142270    0.000000
             ninaegreen        -0.142367    0.045492
             gronkhusky        -0.142367    0.092810
              keinobenz        -0.142367    0.024326
             melina.kes        -0.142367    0.053976
         giallafornelli        -0.142379    0.064835
         elyseneilerman        -0.142391    0.181832
         mvp86hinesward        -0.142769    0.087508
         quirkyinspired        -0.142874    0.006473
            salmadinani        -0.142961    0.034547
   crazylifewithlittles        -0.143362    0.000000
          lancejohnson_        -0.143452    0.023581
             elylaninny        -0.143680    0.012864
             alihallock        -0.143777    0.000000


==================================================
📊 MODEL PERFORMANCE EVALUATION
==================================================

--- 📈 Inference Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0          15552     45.83%
1           6450     19.01%
2           5073     14.95%
3           2892      8.52%
4           2252      6.64%
5           1716      5.06%

--- 🤖 Inference Data Predicted Distribution ---
           Count Percentage
Relevance                  
0          33935    100.00%
1              0      0.00%
2              0      0.00%
3              0      0.00%
4              0      0.00%
5              0      0.00%

🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---
   - **MAE (平均絶対誤差)**: 0.2010
   - **RMSE (二乗平均平方根誤差)**: 0.2041

🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---
   - **NDCG@K (正規化割引累積利得)**:
     - NDCG@1  : 0.4000
     - NDCG@10 : 0.3387
     - NDCG@50 : 0.3243
     - NDCG@100: 0.3144
     - NDCG@200: 0.3173

   - **RBP (ランクバイアス適合率)**: 0.0314

Total inference time: 512.03 seconds
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v3.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Traceback (most recent call last):
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'hashtag'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py", line 371, in <module>
    train_and_save_model()
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py", line 236, in train_and_save_model
    monthly_graphs, influencer_indices, _ = prepare_graph_data(end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py", line 58, in prepare_graph_data
    all_hashtags = set(df_hashtags['hashtag'].astype(str))
                       ~~~~~~~~~~~^^^^^^^^^^^
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/pandas/core/frame.py", line 4107, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: 'hashtag'
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v3.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs:   0%|                                                                                     | 0/12 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "pandas/_libs/index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 7088, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 7096, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'username'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py", line 371, in <module>
    train_and_save_model()
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py", line 236, in train_and_save_model
    monthly_graphs, influencer_indices, _ = prepare_graph_data(end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py", line 107, in prepare_graph_data
    edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
                                                                           ~~~~~~~~~~~~~~~~^^^^^^^^^^^^
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/pandas/core/frame.py", line 4107, in __getitem__
    indexer = self.columns.get_loc(key)
              ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/pandas/core/indexes/base.py", line 3819, in get_loc
    raise KeyError(key) from err
KeyError: 'username'
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v3.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|████████████████████████████████████████████████████████████████████████████| 12/12 [02:03<00:00, 10.32s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- Strategy: Two-Stage Learning (Fast) ---
GCN Encoding: 100%|███████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:40<00:00,  3.39s/it]
Epoch 1/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.19it/s]
Epoch 1/20, Average Batch Loss: 2.3175
Epoch 2/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.47it/s]
Epoch 2/20, Average Batch Loss: 2.3124
Epoch 3/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.49it/s]
Epoch 3/20, Average Batch Loss: 2.3097
Epoch 4/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.51it/s]
Epoch 4/20, Average Batch Loss: 2.3072
Epoch 5/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.53it/s]
Epoch 5/20, Average Batch Loss: 2.3053
Epoch 6/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.52it/s]
Epoch 6/20, Average Batch Loss: 2.3042
Epoch 7/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.38it/s]
Epoch 7/20, Average Batch Loss: 2.3035
Epoch 8/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.40it/s]
Epoch 8/20, Average Batch Loss: 2.3030
Epoch 9/20: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.47it/s]
Epoch 9/20, Average Batch Loss: 2.3029
Epoch 10/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.34it/s]
Epoch 10/20, Average Batch Loss: 2.3027
Epoch 11/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.41it/s]
Epoch 11/20, Average Batch Loss: 2.3027
Epoch 12/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.44it/s]
Epoch 12/20, Average Batch Loss: 2.3027
Epoch 13/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.37it/s]
Epoch 13/20, Average Batch Loss: 2.3026
Epoch 14/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.44it/s]
Epoch 14/20, Average Batch Loss: 2.3026
Epoch 15/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.30it/s]
Epoch 15/20, Average Batch Loss: 2.3026
Epoch 16/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.49it/s]
Epoch 16/20, Average Batch Loss: 2.3026
Epoch 17/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.35it/s]
Epoch 17/20, Average Batch Loss: 2.3026
Epoch 18/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.33it/s]
Epoch 18/20, Average Batch Loss: 2.3026
Epoch 19/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.45it/s]
Epoch 19/20, Average Batch Loss: 2.3026
Epoch 20/20: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.43it/s]
Epoch 20/20, Average Batch Loss: 2.3026

--- Training Complete ---
✅ Model saved to 'influencer_rank_model_20251019_rich_features_2017.pth'
Total time: 473.78 seconds
--- 📈 Starting Inference Process ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|████████████████████████████████████████████████████████████████████████████| 12/12 [02:03<00:00, 10.30s/it]
/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3.py:304: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
Successfully loaded model from 'influencer_rank_model_20251019_rich_features_2017.pth'

🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆
               Username  Predicted_Score  True_Score
         dishitgirldina        -0.056152    0.023143
             ryanadams9        -0.056819    0.042784
             fork_knife        -0.063754    0.033336
          essentially_c        -0.064246    0.050427
        theeveryhostess        -0.064527    0.014908
         paigelorentzen        -0.066200    0.000000
            thebillhorn        -0.066342    0.104123
            dan.bennett        -0.067572    0.019119
             tobruckave        -0.067691    0.023075
    giovannimasiero____        -0.068539    0.022769
            zjrosenberg        -0.069123    0.040086
         official_hogue        -0.069493    0.039285
             patigiemza        -0.069762    0.024245
            jeannedamas        -0.070097    0.027720
              syndicate        -0.070219    0.043610
          beachyogagirl        -0.070505    0.008898
             silvia_mmb        -0.071580    0.091080
                 spenfc        -0.071803    0.031989
             sarahstage        -0.073256    0.036035
vanessawilliamsofficial        -0.073499    0.013116


==================================================
📊 MODEL PERFORMANCE EVALUATION
==================================================

--- 📈 Inference Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- 🤖 Inference Data Predicted Distribution ---
           Count Percentage
Relevance                  
0          20190    100.00%
1              0      0.00%
2              0      0.00%
3              0      0.00%
4              0      0.00%
5              0      0.00%

🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---
   - **MAE (平均絶対誤差)**: 0.1329
   - **RMSE (二乗平均平方根誤差)**: 0.1391

🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---
   - **NDCG@K (正規化割引累積利得)**:
     - NDCG@1  : 0.2000
     - NDCG@10 : 0.3307
     - NDCG@50 : 0.3638
     - NDCG@100: 0.3934
     - NDCG@200: 0.3897

   - **RBP (ランクバイアス適合率)**: 0.0297

Total inference time: 470.65 seconds
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v4.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [02:03<00:00, 10.32s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- Strategy: End-to-End Learning (Slow, High-Memory) ---
Epoch 1/200: Performing GCN forward pass for this epoch...
Training Batches:   0%|                                                                                                                   | 0/1 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4.py", line 417, in <module>
    train_and_save_model()
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4.py", line 309, in train_and_save_model
    loss_listwise = criterion_listwise(predicted_scores_reshaped, batch_true_scores_reshaped)
                    ^^^^^^^^^^^^^^^^^^
NameError: name 'criterion_listwise' is not defined. Did you mean: 'criterion_pointwise'?
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v4.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [02:03<00:00, 10.32s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- Strategy: End-to-End Learning (Slow, High-Memory) ---
Epoch 1/200: Performing GCN forward pass for this epoch...
Training Batches:   0%|                                                                                                                   | 0/1 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4.py", line 417, in <module>
    train_and_save_model()
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4.py", line 315, in train_and_save_model
    loss = loss_listwise + ALPHA * loss_pointwise
                           ^^^^^
NameError: name 'ALPHA' is not defined
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v4.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [02:02<00:00, 10.20s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- Strategy: Two-Stage Learning (Fast) ---
GCN Encoding: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:40<00:00,  3.35s/it]
Epoch 1/200:   0%|                                                                                                                        | 0/1 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4.py", line 417, in <module>
    
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4.py", line 278, in train_and_save_model
    batch_true_scores_reshaped = batch_true_scores.view(LISTS_PER_BATCH, LIST_SIZE)
                    ^^^^^^^^^
NameError: name 'criterion' is not defined
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v4.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [02:04<00:00, 10.34s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- Strategy: Two-Stage Learning (Fast) ---
GCN Encoding: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:55<00:00,  4.64s/it]
Epoch 1/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.33it/s]
Epoch 1/200, Average Batch Loss: 2.5858
Epoch 2/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.05it/s]
Epoch 2/200, Average Batch Loss: 2.4762
Epoch 3/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.80it/s]
Epoch 3/200, Average Batch Loss: 2.4184
Epoch 4/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.28it/s]
Epoch 4/200, Average Batch Loss: 2.3825
Epoch 5/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.86it/s]
Epoch 5/200, Average Batch Loss: 2.3731
Epoch 6/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.08it/s]
Epoch 6/200, Average Batch Loss: 2.3549
Epoch 7/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.01it/s]
Epoch 7/200, Average Batch Loss: 2.3460
Epoch 8/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.06it/s]
Epoch 8/200, Average Batch Loss: 2.3416
Epoch 9/200: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.77it/s]
Epoch 9/200, Average Batch Loss: 2.3381
Epoch 10/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.13it/s]
Epoch 10/200, Average Batch Loss: 2.3342
Epoch 11/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.73it/s]
Epoch 11/200, Average Batch Loss: 2.3317
Epoch 12/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.05it/s]
Epoch 12/200, Average Batch Loss: 2.3312
Epoch 13/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.89it/s]
Epoch 13/200, Average Batch Loss: 2.3304
Epoch 14/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.93it/s]
Epoch 14/200, Average Batch Loss: 2.3292
Epoch 15/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.03it/s]
Epoch 15/200, Average Batch Loss: 2.3271
Epoch 16/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.79it/s]
Epoch 16/200, Average Batch Loss: 2.3269
Epoch 17/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.69it/s]
Epoch 17/200, Average Batch Loss: 2.3256
Epoch 18/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.47it/s]
Epoch 18/200, Average Batch Loss: 2.3259
Epoch 19/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.32it/s]
Epoch 19/200, Average Batch Loss: 2.3255
Epoch 20/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.97it/s]
Epoch 20/200, Average Batch Loss: 2.3251
Epoch 21/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.00it/s]
Epoch 21/200, Average Batch Loss: 2.3242
Epoch 22/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.97it/s]
Epoch 22/200, Average Batch Loss: 2.3245
Epoch 23/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.67it/s]
Epoch 23/200, Average Batch Loss: 2.3222
Epoch 24/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.96it/s]
Epoch 24/200, Average Batch Loss: 2.3233
Epoch 25/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.60it/s]
Epoch 25/200, Average Batch Loss: 2.3224
Epoch 26/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.14it/s]
Epoch 26/200, Average Batch Loss: 2.3227
Epoch 27/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.69it/s]
Epoch 27/200, Average Batch Loss: 2.3234
Epoch 28/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.95it/s]
Epoch 28/200, Average Batch Loss: 2.3227
Epoch 29/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.73it/s]
Epoch 29/200, Average Batch Loss: 2.3210
Epoch 30/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.99it/s]
Epoch 30/200, Average Batch Loss: 2.3225
Epoch 31/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.95it/s]
Epoch 31/200, Average Batch Loss: 2.3214
Epoch 32/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.99it/s]
Epoch 32/200, Average Batch Loss: 2.3226
Epoch 33/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.65it/s]
Epoch 33/200, Average Batch Loss: 2.3212
Epoch 34/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.12it/s]
Epoch 34/200, Average Batch Loss: 2.3230
Epoch 35/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.90it/s]
Epoch 35/200, Average Batch Loss: 2.3211
Epoch 36/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.87it/s]
Epoch 36/200, Average Batch Loss: 2.3212
Epoch 37/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.59it/s]
Epoch 37/200, Average Batch Loss: 2.3223
Epoch 38/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.82it/s]
Epoch 38/200, Average Batch Loss: 2.3202
Epoch 39/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.86it/s]
Epoch 39/200, Average Batch Loss: 2.3224
Epoch 40/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.15it/s]
Epoch 40/200, Average Batch Loss: 2.3213
Epoch 41/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.97it/s]
Epoch 41/200, Average Batch Loss: 2.3192
Epoch 42/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.97it/s]
Epoch 42/200, Average Batch Loss: 2.3217
Epoch 43/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.46it/s]
Epoch 43/200, Average Batch Loss: 2.3198
Epoch 44/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.86it/s]
Epoch 44/200, Average Batch Loss: 2.3210
Epoch 45/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.93it/s]
Epoch 45/200, Average Batch Loss: 2.3206
Epoch 46/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.24it/s]
Epoch 46/200, Average Batch Loss: 2.3201
Epoch 47/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.77it/s]
Epoch 47/200, Average Batch Loss: 2.3204
Epoch 48/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.19it/s]
Epoch 48/200, Average Batch Loss: 2.3199
Epoch 49/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.25it/s]
Epoch 49/200, Average Batch Loss: 2.3203
Epoch 50/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.75it/s]
Epoch 50/200, Average Batch Loss: 2.3201
Epoch 51/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.23it/s]
Epoch 51/200, Average Batch Loss: 2.3185
Epoch 52/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.89it/s]
Epoch 52/200, Average Batch Loss: 2.3199
Epoch 53/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.52it/s]
Epoch 53/200, Average Batch Loss: 2.3213
Epoch 54/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.40it/s]
Epoch 54/200, Average Batch Loss: 2.3192
Epoch 55/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.01it/s]
Epoch 55/200, Average Batch Loss: 2.3190
Epoch 56/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.23it/s]
Epoch 56/200, Average Batch Loss: 2.3196
Epoch 57/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.84it/s]
Epoch 57/200, Average Batch Loss: 2.3196
Epoch 58/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.07it/s]
Epoch 58/200, Average Batch Loss: 2.3175
Epoch 59/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.21it/s]
Epoch 59/200, Average Batch Loss: 2.3206
Epoch 60/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.02it/s]
Epoch 60/200, Average Batch Loss: 2.3187
Epoch 61/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.97it/s]
Epoch 61/200, Average Batch Loss: 2.3188
Epoch 62/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.23it/s]
Epoch 62/200, Average Batch Loss: 2.3192
Epoch 63/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.90it/s]
Epoch 63/200, Average Batch Loss: 2.3182
Epoch 64/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.92it/s]
Epoch 64/200, Average Batch Loss: 2.3188
Epoch 65/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.78it/s]
Epoch 65/200, Average Batch Loss: 2.3199
Epoch 66/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.91it/s]
Epoch 66/200, Average Batch Loss: 2.3201
Epoch 67/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.05it/s]
Epoch 67/200, Average Batch Loss: 2.3203
Epoch 68/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.96it/s]
Epoch 68/200, Average Batch Loss: 2.3193
Epoch 69/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.07it/s]
Epoch 69/200, Average Batch Loss: 2.3195
Epoch 70/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.78it/s]
Epoch 70/200, Average Batch Loss: 2.3185
Epoch 71/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.39it/s]
Epoch 71/200, Average Batch Loss: 2.3191
Epoch 72/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.69it/s]
Epoch 72/200, Average Batch Loss: 2.3190
Epoch 73/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.53it/s]
Epoch 73/200, Average Batch Loss: 2.3206
Epoch 74/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.62it/s]
Epoch 74/200, Average Batch Loss: 2.3198
Epoch 75/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.55it/s]
Epoch 75/200, Average Batch Loss: 2.3202
Epoch 76/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.75it/s]
Epoch 76/200, Average Batch Loss: 2.3176
Epoch 77/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.64it/s]
Epoch 77/200, Average Batch Loss: 2.3181
Epoch 78/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.18it/s]
Epoch 78/200, Average Batch Loss: 2.3203
Epoch 79/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.79it/s]
Epoch 79/200, Average Batch Loss: 2.3180
Epoch 80/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.33it/s]
Epoch 80/200, Average Batch Loss: 2.3179
Epoch 81/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.88it/s]
Epoch 81/200, Average Batch Loss: 2.3176
Epoch 82/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.24it/s]
Epoch 82/200, Average Batch Loss: 2.3190
Epoch 83/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.32it/s]
Epoch 83/200, Average Batch Loss: 2.3182
Epoch 84/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.48it/s]
Epoch 84/200, Average Batch Loss: 2.3177
Epoch 85/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.26it/s]
Epoch 85/200, Average Batch Loss: 2.3189
Epoch 86/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.48it/s]
Epoch 86/200, Average Batch Loss: 2.3200
Epoch 87/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.39it/s]
Epoch 87/200, Average Batch Loss: 2.3177
Epoch 88/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.74it/s]
Epoch 88/200, Average Batch Loss: 2.3193
Epoch 89/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.59it/s]
Epoch 89/200, Average Batch Loss: 2.3178
Epoch 90/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.02it/s]
Epoch 90/200, Average Batch Loss: 2.3190
Epoch 91/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.14it/s]
Epoch 91/200, Average Batch Loss: 2.3191
Epoch 92/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.72it/s]
Epoch 92/200, Average Batch Loss: 2.3202
Epoch 93/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.46it/s]
Epoch 93/200, Average Batch Loss: 2.3187
Epoch 94/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.32it/s]
Epoch 94/200, Average Batch Loss: 2.3176
Epoch 95/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.63it/s]
Epoch 95/200, Average Batch Loss: 2.3193
Epoch 96/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.27it/s]
Epoch 96/200, Average Batch Loss: 2.3192
Epoch 97/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.45it/s]
Epoch 97/200, Average Batch Loss: 2.3197
Epoch 98/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.31it/s]
Epoch 98/200, Average Batch Loss: 2.3178
Epoch 99/200: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.06it/s]
Epoch 99/200, Average Batch Loss: 2.3183
Epoch 100/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.27it/s]
Epoch 100/200, Average Batch Loss: 2.3191
Epoch 101/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.08it/s]
Epoch 101/200, Average Batch Loss: 2.3190
Epoch 102/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.38it/s]
Epoch 102/200, Average Batch Loss: 2.3193
Epoch 103/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.97it/s]
Epoch 103/200, Average Batch Loss: 2.3182
Epoch 104/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.70it/s]
Epoch 104/200, Average Batch Loss: 2.3174
Epoch 105/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.24it/s]
Epoch 105/200, Average Batch Loss: 2.3181
Epoch 106/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.87it/s]
Epoch 106/200, Average Batch Loss: 2.3203
Epoch 107/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.33it/s]
Epoch 107/200, Average Batch Loss: 2.3191
Epoch 108/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.17it/s]
Epoch 108/200, Average Batch Loss: 2.3190
Epoch 109/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.16it/s]
Epoch 109/200, Average Batch Loss: 2.3188
Epoch 110/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.05it/s]
Epoch 110/200, Average Batch Loss: 2.3188
Epoch 111/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.71it/s]
Epoch 111/200, Average Batch Loss: 2.3206
Epoch 112/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.39it/s]
Epoch 112/200, Average Batch Loss: 2.3201
Epoch 113/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.20it/s]
Epoch 113/200, Average Batch Loss: 2.3193
Epoch 114/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.75it/s]
Epoch 114/200, Average Batch Loss: 2.3183
Epoch 115/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.23it/s]
Epoch 115/200, Average Batch Loss: 2.3198
Epoch 116/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.00it/s]
Epoch 116/200, Average Batch Loss: 2.3186
Epoch 117/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.51it/s]
Epoch 117/200, Average Batch Loss: 2.3209
Epoch 118/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.54it/s]
Epoch 118/200, Average Batch Loss: 2.3178
Epoch 119/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.50it/s]
Epoch 119/200, Average Batch Loss: 2.3197
Epoch 120/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.07it/s]
Epoch 120/200, Average Batch Loss: 2.3197
Epoch 121/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.87it/s]
Epoch 121/200, Average Batch Loss: 2.3190
Epoch 122/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.83it/s]
Epoch 122/200, Average Batch Loss: 2.3193
Epoch 123/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.61it/s]
Epoch 123/200, Average Batch Loss: 2.3192
Epoch 124/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.21it/s]
Epoch 124/200, Average Batch Loss: 2.3184
Epoch 125/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.63it/s]
Epoch 125/200, Average Batch Loss: 2.3183
Epoch 126/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.36it/s]
Epoch 126/200, Average Batch Loss: 2.3172
Epoch 127/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.90it/s]
Epoch 127/200, Average Batch Loss: 2.3207
Epoch 128/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.17it/s]
Epoch 128/200, Average Batch Loss: 2.3199
Epoch 129/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.55it/s]
Epoch 129/200, Average Batch Loss: 2.3183
Epoch 130/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.33it/s]
Epoch 130/200, Average Batch Loss: 2.3189
Epoch 131/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.29it/s]
Epoch 131/200, Average Batch Loss: 2.3191
Epoch 132/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.82it/s]
Epoch 132/200, Average Batch Loss: 2.3188
Epoch 133/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.55it/s]
Epoch 133/200, Average Batch Loss: 2.3190
Epoch 134/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.62it/s]
Epoch 134/200, Average Batch Loss: 2.3195
Epoch 135/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.02it/s]
Epoch 135/200, Average Batch Loss: 2.3192
Epoch 136/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.49it/s]
Epoch 136/200, Average Batch Loss: 2.3192
Epoch 137/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.95it/s]
Epoch 137/200, Average Batch Loss: 2.3201
Epoch 138/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.75it/s]
Epoch 138/200, Average Batch Loss: 2.3208
Epoch 139/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.72it/s]
Epoch 139/200, Average Batch Loss: 2.3200
Epoch 140/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.78it/s]
Epoch 140/200, Average Batch Loss: 2.3189
Epoch 141/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.04it/s]
Epoch 141/200, Average Batch Loss: 2.3204
Epoch 142/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.85it/s]
Epoch 142/200, Average Batch Loss: 2.3192
Epoch 143/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.99it/s]
Epoch 143/200, Average Batch Loss: 2.3183
Epoch 144/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.98it/s]
Epoch 144/200, Average Batch Loss: 2.3199
Epoch 145/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.37it/s]
Epoch 145/200, Average Batch Loss: 2.3178
Epoch 146/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.34it/s]
Epoch 146/200, Average Batch Loss: 2.3175
Epoch 147/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.90it/s]
Epoch 147/200, Average Batch Loss: 2.3189
Epoch 148/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.07it/s]
Epoch 148/200, Average Batch Loss: 2.3188
Epoch 149/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.42it/s]
Epoch 149/200, Average Batch Loss: 2.3203
Epoch 150/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.29it/s]
Epoch 150/200, Average Batch Loss: 2.3200
Epoch 151/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.97it/s]
Epoch 151/200, Average Batch Loss: 2.3192
Epoch 152/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.68it/s]
Epoch 152/200, Average Batch Loss: 2.3191
Epoch 153/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.18it/s]
Epoch 153/200, Average Batch Loss: 2.3190
Epoch 154/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.86it/s]
Epoch 154/200, Average Batch Loss: 2.3171
Epoch 155/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.00it/s]
Epoch 155/200, Average Batch Loss: 2.3186
Epoch 156/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.59it/s]
Epoch 156/200, Average Batch Loss: 2.3191
Epoch 157/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.02it/s]
Epoch 157/200, Average Batch Loss: 2.3169
Epoch 158/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.47it/s]
Epoch 158/200, Average Batch Loss: 2.3193
Epoch 159/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.36it/s]
Epoch 159/200, Average Batch Loss: 2.3183
Epoch 160/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.88it/s]
Epoch 160/200, Average Batch Loss: 2.3178
Epoch 161/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.66it/s]
Epoch 161/200, Average Batch Loss: 2.3194
Epoch 162/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.61it/s]
Epoch 162/200, Average Batch Loss: 2.3192
Epoch 163/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.03it/s]
Epoch 163/200, Average Batch Loss: 2.3189
Epoch 164/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.95it/s]
Epoch 164/200, Average Batch Loss: 2.3191
Epoch 165/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.44it/s]
Epoch 165/200, Average Batch Loss: 2.3202
Epoch 166/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.26it/s]
Epoch 166/200, Average Batch Loss: 2.3190
Epoch 167/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.65it/s]
Epoch 167/200, Average Batch Loss: 2.3189
Epoch 168/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.06it/s]
Epoch 168/200, Average Batch Loss: 2.3209
Epoch 169/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.70it/s]
Epoch 169/200, Average Batch Loss: 2.3191
Epoch 170/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.28it/s]
Epoch 170/200, Average Batch Loss: 2.3176
Epoch 171/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.53it/s]
Epoch 171/200, Average Batch Loss: 2.3186
Epoch 172/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.58it/s]
Epoch 172/200, Average Batch Loss: 2.3201
Epoch 173/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.70it/s]
Epoch 173/200, Average Batch Loss: 2.3185
Epoch 174/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.54it/s]
Epoch 174/200, Average Batch Loss: 2.3205
Epoch 175/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.73it/s]
Epoch 175/200, Average Batch Loss: 2.3198
Epoch 176/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.78it/s]
Epoch 176/200, Average Batch Loss: 2.3193
Epoch 177/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.26it/s]
Epoch 177/200, Average Batch Loss: 2.3191
Epoch 178/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.44it/s]
Epoch 178/200, Average Batch Loss: 2.3203
Epoch 179/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.84it/s]
Epoch 179/200, Average Batch Loss: 2.3173
Epoch 180/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.22it/s]
Epoch 180/200, Average Batch Loss: 2.3191
Epoch 181/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.17it/s]
Epoch 181/200, Average Batch Loss: 2.3188
Epoch 182/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.95it/s]
Epoch 182/200, Average Batch Loss: 2.3187
Epoch 183/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.44it/s]
Epoch 183/200, Average Batch Loss: 2.3190
Epoch 184/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.61it/s]
Epoch 184/200, Average Batch Loss: 2.3173
Epoch 185/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.76it/s]
Epoch 185/200, Average Batch Loss: 2.3191
Epoch 186/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.34it/s]
Epoch 186/200, Average Batch Loss: 2.3192
Epoch 187/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.18it/s]
Epoch 187/200, Average Batch Loss: 2.3187
Epoch 188/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.24it/s]
Epoch 188/200, Average Batch Loss: 2.3179
Epoch 189/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.50it/s]
Epoch 189/200, Average Batch Loss: 2.3208
Epoch 190/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.76it/s]
Epoch 190/200, Average Batch Loss: 2.3189
Epoch 191/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.86it/s]
Epoch 191/200, Average Batch Loss: 2.3173
Epoch 192/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.08it/s]
Epoch 192/200, Average Batch Loss: 2.3191
Epoch 193/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.88it/s]
Epoch 193/200, Average Batch Loss: 2.3192
Epoch 194/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.21it/s]
Epoch 194/200, Average Batch Loss: 2.3187
Epoch 195/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.90it/s]
Epoch 195/200, Average Batch Loss: 2.3175
Epoch 196/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.64it/s]
Epoch 196/200, Average Batch Loss: 2.3191
Epoch 197/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.96it/s]
Epoch 197/200, Average Batch Loss: 2.3177
Epoch 198/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.99it/s]
Epoch 198/200, Average Batch Loss: 2.3200
Epoch 199/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.41it/s]
Epoch 199/200, Average Batch Loss: 2.3174
Epoch 200/200: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.59it/s]
Epoch 200/200, Average Batch Loss: 2.3195

--- Training Complete ---
✅ Model saved to 'influencer_rank_model_20251020_rich_features_2017.pth'
Total time: 560.29 seconds
--- 📈 Starting Inference Process ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [02:09<00:00, 10.81s/it]
/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4.py:346: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
Successfully loaded model from 'influencer_rank_model_20251020_rich_features_2017.pth'

🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆
                   Username  Predicted_Score  True_Score
         timjeremiasjanssen         0.055089    0.120767
                dan.bennett         0.054999    0.019119
            victoriajustice         0.054076    0.025778
              amiranbabayev         0.053274    0.083615
                 geowollner         0.051848    0.124012
            alyssekatherine         0.051829    0.050706
      ourlittleredbrickhome         0.051746    0.038801
                  model_roz         0.051596    0.013950
           imjustsammielynn         0.051480    0.044696
                anallievent         0.051450    0.015680
              mrsuperman_03         0.051445    0.006193
          digitalcontentlab         0.051444    0.020483
            ridenowchandler         0.051443    0.068683
                paula_muir_         0.051443    0.072455
                 meheureuse         0.051443    0.018212
              bitsbitesblog         0.051443    0.065007
                    choco_k         0.051443    0.000000
            thehustlingmama         0.051443    0.090760
keeping.up.with.mumma.jones         0.051443    0.118983
           timelessoptimist         0.051441    0.066523


==================================================
📊 MODEL PERFORMANCE EVALUATION
==================================================

--- 📈 Inference Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- 🤖 Inference Data Predicted Distribution ---
           Count Percentage
Relevance                  
0              4      0.02%
1             90      0.45%
2          20074     99.43%
3             22      0.11%
4              0      0.00%
5              0      0.00%

🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---
   - **MAE (平均絶対誤差)**: 0.0277
   - **RMSE (二乗平均平方根誤差)**: 0.0404

🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---
   - **NDCG@K (正規化割引累積利得)**:
     - NDCG@1  : 1.0000
     - NDCG@10 : 0.5603
     - NDCG@50 : 0.5236
     - NDCG@100: 0.5189
     - NDCG@200: 0.5159

   - **RBP (ランクバイアス適合率)**: 0.0455

Total inference time: 486.76 seconds
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ 

