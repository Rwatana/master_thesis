#!/bin/bash

# --- ⚙️ 設定 ---
NUM_RUNS=100           # 実行回数の合計
MAX_PARALLEL_JOBS=8  # 同時に実行する最大プロセス数。マシンのCPUコア数に合わせる
RESULTS_DIR="experiment_results_v5" # 個別結果を保存するディレクトリ
FINAL_RESULTS_FILE="experiment_results_v5_combined.csv" # 最終的な結合ファイル
PYTHON_SCRIPT="influencer_rank_v5.py"

# --- 1. 前準備 ---
mkdir -p $RESULTS_DIR # 結果ディレクトリを作成
rm -f $RESULTS_DIR/run_*.csv # 既存の個別結果を削除
rm -f $FINAL_RESULTS_FILE # 既存の結合ファイルを削除

echo "Starting $NUM_RUNS experiment runs in parallel (max ${MAX_PARALLEL_JOBS} jobs)..."
echo "Individual results will be saved in $RESULTS_DIR"

# --- 2. 並列実行 (xargsを使用) ---

# 実行する関数を定義
run_experiment() {
  i=$1
  RUN_ID="run_${i}"
  RESULT_FILE="${RESULTS_DIR}/${RUN_ID}.csv"
  
  echo "--- Starting Run ${i}/${NUM_RUNS} (ID: ${RUN_ID}) ---"
  
  # Pythonスクリプトを実行。結果は *個別のファイル* に保存
  # (あなたのPythonスクリプトが --results_file 引数を取る前提です)
  python $PYTHON_SCRIPT --run_id "$RUN_ID" --results_file "$RESULT_FILE"
  
  if [ $? -ne 0 ]; then
    echo "🚨 Error during run ${RUN_ID}. See output above."
    # xargsは他のプロセスを止められないが、エラーとして記録
    exit 255 
  else
    echo "--- Finished Run ${i}/${NUM_RUNS} (ID: ${RUN_ID}) ---"
  fi
}

# bash -c で関数を呼び出せるようにエクスポート
export -f run_experiment
export RESULTS_DIR
export PYTHON_SCRIPT
export NUM_RUNS

# seq で 1 から NUM_RUNS までの数字を生成し、xargs に渡す
# -n 1: 一度に1つの引数（数字）を
# -P $MAX_PARALLEL_JOBS: 最大並列数
# bash -c 'run_experiment "$@"' _: 各引数（数字）に対してrun_experiment関数を実行
seq 1 $NUM_RUNS | xargs -n 1 -P $MAX_PARALLEL_JOBS bash -c 'run_experiment "$@"' _

echo ""
echo "----------------------------------------------------"
echo "🎉 All $NUM_RUNS parallel runs completed!"
echo "----------------------------------------------------"


# --- 3. 結果の結合 ---
echo "Combining results into $FINAL_RESULTS_FILE ..."

# ヘッダー処理: 
# 1. 最初のファイル (run_1.csv) を見つけて、そのヘッダーを最終ファイルにコピー
FIRST_FILE="${RESULTS_DIR}/run_1.csv"
if [ -f "$FIRST_FILE" ]; then
    head -n 1 "$FIRST_FILE" > $FINAL_RESULTS_FILE
else
    echo "Error: ${FIRST_FILE} not found. Cannot create combined file header."
    exit 1
fi

# 2. 全てのファイルの2行目以降（データ本体）を最終ファイルに追記
for f in $RESULTS_DIR/run_*.csv
do
  tail -n +2 "$f" >> $FINAL_RESULTS_FILE
done

echo "✅ Results combined successfully!"
echo "Individual files are in $RESULTS_DIR"