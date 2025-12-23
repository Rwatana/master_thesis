(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v3_scaler.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Total feature dimension: 36
Building monthly graphs: 100%|████████████████████████████████████████████████████████████████████████████| 12/12 [04:24<00:00, 22.02s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0          15552     45.83%
1           6450     19.01%
2           5073     14.95%
3           2892      8.52%
4           2252      6.64%
5           1716      5.06%

--- Training Complete ---
✅ Model saved to 'influencer_rank_model_20251019_rich_features_2017_scailing.pth'
Total time: 571.10 seconds
--- 📈 Starting Inference Process ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Total feature dimension: 36
Building monthly graphs: 100%|████████████████████████████████████████████████████████████████████████████| 12/12 [03:53<00:00, 19.48s/it]
/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3_scaler.py:341: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
Successfully loaded model from 'influencer_rank_model_20251019_rich_features_2017_scailing.pth'

🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆
              Username  Predicted_Score  True_Score
   giovannimasiero____         0.204892    0.022769
      mrbentley_thedog         0.190840    0.000000
    jessicaannewoodley         0.190222    0.031987
          jadelizroper         0.188463    0.000000
             proudlock         0.188459    0.000000
           itsnotsonia         0.181986    0.083750
           elielbuenob         0.180783    0.000000
eleonorabrunaccidivaio         0.178392    0.000000
         limaswardrobe         0.176228    0.000000
            josefinehj         0.174728    0.000000
          amandabatula         0.173043    0.009990
             bradbeal3         0.172381    0.031790
            ryanadams9         0.172083    0.042784
       mytravelaffairs         0.169029    0.066570
              anthonin         0.168782    0.049367
          realmomofsfv         0.168386    0.000000
     raisinglilmuslims         0.168141    0.000000
             2008irene         0.167653    0.000000
           jessicamxmx         0.167649    0.018018
          lynettecenee         0.167529    0.039165


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
0          18367     54.12%
1           6202     18.28%
2           2958      8.72%
3           2870      8.46%
4           2156      6.35%
5           1382      4.07%

🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---
   - **MAE (平均絶対誤差)**: 0.0510
   - **RMSE (二乗平均平方根誤差)**: 0.0641

🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---
   - **NDCG@K (正規化割引累積利得)**:
     - NDCG@1  : 0.2000
     - NDCG@10 : 0.1508
     - NDCG@50 : 0.2043
     - NDCG@100: 0.1830
     - NDCG@200: 0.2115

   - **RBP (ランクバイアス適合率)**: 0.0171

Total inference time: 583.75 seconds
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v3_scaler.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [03:42<00:00, 18.54s/it]

--- 📊 Training Data Ground Truth Distribution ---
           Count Percentage
Relevance                  
0           1807      8.95%
1           6450     31.95%
2           5073     25.13%
3           2892     14.32%
4           2252     11.15%
5           1716      8.50%

--- Training Complete ---
✅ Model saved to 'influencer_rank_model_20251019_rich_features_2017_scailing.pth'
Total time: 524.10 seconds
--- 📈 Starting Inference Process ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [03:40<00:00, 18.39s/it]
/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v3_scaler.py:320: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
Successfully loaded model from 'influencer_rank_model_20251019_rich_features_2017_scailing.pth'

🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆
             Username  Predicted_Score  True_Score
         chloe.pettit         0.442205    0.016921
          aspen.hicks         0.432826    0.035709
        curlyfitvegan         0.425122    0.077206
    hercampus_adelphi         0.415707    0.035237
     mrs.hollynichole         0.407981    0.031433
        amyboylephoto         0.405838    0.033106
          zodiakgirlz         0.398596    0.006894
         wordsbyladyg         0.397136    0.038688
      shelleybethblog         0.396689    0.060287
         applejustine         0.395601    0.057776
         dharrismedia         0.392785    0.067801
        darcicreative         0.383486    0.047438
theintentionalmomblog         0.383476    0.028268
          verdeorigen         0.379087    0.005757
alligoldenphotography         0.377115    0.033747
      ericabmccleskey         0.375268    0.056509
        elizadarlings         0.374808    0.096927
      prettylilmudder         0.373808    0.050068
             halitrax         0.373776    0.049298
    theefunnyaquarius         0.373692    0.038599


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
0           2266     11.22%
1           1040      5.15%
2           1210      5.99%
3           1246      6.17%
4           2772     13.73%
5          11656     57.73%

🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---
   - **MAE (平均絶対誤差)**: 0.0944
   - **RMSE (二乗平均平方根誤差)**: 0.1210

🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---
   - **NDCG@K (正規化割引累積利得)**:
     - NDCG@1  : 0.2000
     - NDCG@10 : 0.3966
     - NDCG@50 : 0.4285
     - NDCG@100: 0.4194
     - NDCG@200: 0.4054

   - **RBP (ランクバイアス適合率)**: 0.0362

Total inference time: 573.49 seconds
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v4_simple.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs:  25%|████████████████████████▊                                                                          | 3/12 [00:23<01:12,  8.01s/it]Building monthly graphs:  25%|████████████████████████▊                                                                          | 3/12 [00:24<01:14,  8.25s/it]
Traceback (most recent call last):
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4_simple.py", line 418, in <module>
    train_and_save_model()
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4_simple.py", line 244, in train_and_save_model
    monthly_graphs, influencer_indices, _ = prepare_graph_data(end_date=latest_date, num_months=12, metric_numerator=METRIC_NUMERATOR, metric_denominator=METRIC_DENOMINATOR)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4_simple.py", line 113, in prepare_graph_data
    edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4_simple.py", line 113, in <listcomp>
    edges_ht = [(node_to_idx[str(u)], node_to_idx[str(h)]) for u, h in zip(current_hashtags['username'], current_hashtags['hashtag']) if str(u) in node_to_idx and str(h) in node_to_idx]
                                                  ^^^^^^
KeyboardInterrupt
^CException ignored in atexit callback: <function dump_compile_times at 0x74d141d18540>
Traceback (most recent call last):
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/torch/_dynamo/utils.py", line 399, in dump_compile_times
    log.info(compile_times(repr="str", aggregate=True))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/torch/_dynamo/utils.py", line 385, in compile_times
    out += tabulate(rows, headers=("Function", "Runtimes (s)"))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/torch/_dynamo/utils.py", line 148, in tabulate
    import tabulate
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/tabulate/__init__.py", line 15, in <module>
    import wcwidth  # optional wide-character (CJK) support
    ^^^^^^^^^^^^^^
  File "/home/lab/ryoma/master/envs/pems-metra/lib/python3.11/site-packages/wcwidth/__init__.py", line 12, in <module>
    from .wcwidth import ZERO_WIDTH  # noqa
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1176, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1147, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 690, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 936, in exec_module
  File "<frozen importlib._bootstrap_external>", line 1032, in get_code
  File "<frozen importlib._bootstrap_external>", line 1130, in get_data
KeyboardInterrupt: 
^C^C^C^C^C
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ ^C
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ ^C
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ python influencer_rank_v4_simple.py 
--- Starting Training ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [02:05<00:00, 10.43s/it]

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
GCN Encoding: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:40<00:00,  3.38s/it]
Epoch 1/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.05it/s]
Epoch 1/20, Average Batch Loss: 2.6181
Epoch 2/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.03it/s]
Epoch 2/20, Average Batch Loss: 2.4264
Epoch 3/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.77it/s]
Epoch 3/20, Average Batch Loss: 2.4125
Epoch 4/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.07it/s]
Epoch 4/20, Average Batch Loss: 2.4155
Epoch 5/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.83it/s]
Epoch 5/20, Average Batch Loss: 2.4067
Epoch 6/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  5.13it/s]
Epoch 6/20, Average Batch Loss: 2.4021
Epoch 7/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.82it/s]
Epoch 7/20, Average Batch Loss: 2.3964
Epoch 8/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.40it/s]
Epoch 8/20, Average Batch Loss: 2.3959
Epoch 9/20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.17it/s]
Epoch 9/20, Average Batch Loss: 2.3944
Epoch 10/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.42it/s]
Epoch 10/20, Average Batch Loss: 2.3927
Epoch 11/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  3.50it/s]
Epoch 11/20, Average Batch Loss: 2.3872
Epoch 12/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.30it/s]
Epoch 12/20, Average Batch Loss: 2.3894
Epoch 13/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.98it/s]
Epoch 13/20, Average Batch Loss: 2.3862
Epoch 14/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.94it/s]
Epoch 14/20, Average Batch Loss: 2.3812
Epoch 15/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.63it/s]
Epoch 15/20, Average Batch Loss: 2.3806
Epoch 16/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.22it/s]
Epoch 16/20, Average Batch Loss: 2.3767
Epoch 17/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.86it/s]
Epoch 17/20, Average Batch Loss: 2.3776
Epoch 18/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.54it/s]
Epoch 18/20, Average Batch Loss: 2.3757
Epoch 19/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  2.04it/s]
Epoch 19/20, Average Batch Loss: 2.3738
Epoch 20/20: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.46it/s]
Epoch 20/20, Average Batch Loss: 2.3729

--- Training Complete ---
✅ Model saved to 'influencer_rank_model_20251020_rich_features_2017_v4_simple.pth'
Total time: 498.97 seconds
--- 📈 Starting Inference Process ---

Building graph sequence for 12 months ending on 2017-12...
Using Engagement Metric: likes_and_comments / followers
Found 20,588 active influencers in posts_2017.csv.
Total feature dimension: 36
Building monthly graphs: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [02:17<00:00, 11.47s/it]
/home/lab/ryoma/graph_dataset/instagram_influencer/influencer_rank_v4_simple.py:346: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(MODEL_SAVE_PATH))
Successfully loaded model from 'influencer_rank_model_20251020_rich_features_2017_v4_simple.pth'

🏆 --- Top 20 Predicted Influencers (with True Scores) --- 🏆
       Username  Predicted_Score  True_Score
      kaigreene         0.115746    0.021749
       ddlovato         0.113051    0.020001
      amberrose         0.112574    0.026498
 missjuliakelly         0.111693    0.037080
     hijabhills         0.106426    0.018018
     wwerollins         0.100855    0.018843
         mrkate         0.099604    0.038947
  rebeccazamolo         0.096738    0.036693
  uravgconsumer         0.092904    0.077630
thekolors_stash         0.092696    0.028400
           zane         0.092313    0.114548
   nala_rinaldo         0.091573    0.009103
claudiasulewski         0.090894    0.061024
   ninahouston_         0.090693    0.142988
       lalicler         0.090685    0.025221
   jessie_maya_         0.090125    0.126693
      conangray         0.089696    0.154236
   brunafurrlan         0.089406    0.058472
       kikalite         0.089282    0.018820
   ginnyscloset         0.089105    0.018952


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
0           8691     43.05%
1           8226     40.74%
2           2286     11.32%
3            693      3.43%
4            288      1.43%
5              6      0.03%

🎯 --- A. Prediction Accuracy Metrics (値の正確さ) ---
   - **MAE (平均絶対誤差)**: 0.0386
   - **RMSE (二乗平均平方根誤差)**: 0.0557

🏅 --- B. Ranking Quality Metrics (順序の正しさ) ---
   - **NDCG@K (正規化割引累積利得)**:
     - NDCG@1  : 0.2000
     - NDCG@10 : 0.2873
     - NDCG@50 : 0.4043
     - NDCG@100: 0.3958
     - NDCG@200: 0.3948

   - **RBP (ランクバイアス適合率)**: 0.0375

Total inference time: 518.12 seconds
(pems-metra) ryoma@aurora:~/graph_dataset/instagram_influencer$ 