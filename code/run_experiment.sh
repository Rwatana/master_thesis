#!/bin/bash

# --- 設定 ---
NUM_RUNS=100  # 実行回数
RESULTS_FILE="experiment_results.csv" # 結果を保存するファイル名

# --- 前準備 ---
# 既存の結果ファイルを削除 (まっさらな状態から始める場合)
# rm -f $RESULTS_FILE 
echo "Starting $NUM_RUNS experiment runs..."
echo "Results will be saved to $RESULTS_FILE"

# --- 実行ループ ---
for i in $(seq 1 $NUM_RUNS)
do
  # 実行IDを定義 (例: run_1, run_2, ...)
  RUN_ID="run_${i}"
  
  echo ""
  echo "----------------------------------------------------"
  echo "--- Starting Run ${i}/${NUM_RUNS} (ID: ${RUN_ID}) ---"
  echo "----------------------------------------------------"
  
  # Pythonスクリプトを実行。引数として実行IDと結果ファイル名を渡す
  python influencer_rank_v4_refactored.py --run_id "$RUN_ID" --results_file "$RESULTS_FILE"
  
  # エラーが発生したらスクリプトを停止
  if [ $? -ne 0 ]; then
    echo "Error during run ${RUN_ID}. Stopping script."
    exit 1
  fi
done

echo ""
echo "----------------------------------------------------"
echo "🎉 All $NUM_RUNS runs completed successfully! 🎉"
echo "----------------------------------------------------"