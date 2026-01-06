#!/usr/bin/env bash
set -u  # 未定義変数だけ検知（set -e は使わない：失敗しても続行するため）
set -o pipefail

ts="$(date +%Y%m%d_%H%M%S)"
logdir="logs_run_v18_4_continue_${ts}"
mkdir -p "${logdir}"

CKPT_PTH="./checkpoints/Run_000_20251226_1547/model_state.pth"
CKPT_PT="./checkpoints/Run_000_20251226_1547/model_state.pt"

CKPT_FOR_EXP4="${CKPT_PT}"
if [[ ! -f "${CKPT_FOR_EXP4}" && -f "${CKPT_PTH}" ]]; then
  echo "[run] WARN: ${CKPT_PT} not found, using ${CKPT_PTH} for exp4."
  CKPT_FOR_EXP4="${CKPT_PTH}"
fi

# 結果格納
declare -a names
declare -a cmds
declare -a statuses
declare -a logs

run_cmd () {
  local name="$1"; shift
  local log="${logdir}/${name}.log"

  echo "===================================================="
  echo "[run] ${name}"
  echo "[run] $*"
  echo "[run] log => ${log}"
  echo "===================================================="

  # 実行して exit code を取る（tee しても pipefail 有効なので rc が取れる）
  "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}

  names+=("${name}")
  logs+=("${log}")
  cmds+=("$*")
  statuses+=("${rc}")

  if [[ "${rc}" -eq 0 ]]; then
    echo "[run] RESULT: SUCCESS (${name})"
  else
    echo "[run] RESULT: FAIL(${rc}) (${name})"
  fi
  echo
}

# 1) exp3 stability
run_cmd "01_exp3_stability" \
  python ./influencer_rank_full_fixed_xai_paper_no_imagecsv_v18_exp3.py \
    --mode infer \
    --ckpt "${CKPT_PTH}" \
    --stability_test \
    --stability_positions 0,1,2 \
    --stability_runs 20 \
    --stability_topk 20 \
    --stability_kind both \
    --stability_base_seed 1234 \
    --stability_epochs 300 \
    --stability_lr 0.05

# 2) exp2 infer
run_cmd "02_exp2_infer" \
  python ./influencer_rank_full_fixed_xai_paper_no_imagecsv_v18_exp2.py \
    --mode infer \
    --ckpt "${CKPT_PTH}"

# 3) exp4 frontier
run_cmd "03_exp4_frontier" \
  python ./influencer_rank_full_fixed_xai_paper_no_imagecsv_v18_exp4.py \
    --mode frontier \
    --target_node 1238805 \
    --ckpt "${CKPT_FOR_EXP4}" \
    --end_date 2017-12-31 \
    --num_months 12 \
    --metric_numerator likes_and_comments \
    --metric_denominator followers \
    --explain_pos 4 \
    --lambdas 0,0.001,0.003,0.01,0.03,0.1 \
    --epochs 300 \
    --lr 0.05 \
    --use_subgraph \
    --num_hops 1 \
    --undirected \
    --edge_grouping neighbor \
    --mlflow

# 4) v18 infer
run_cmd "04_v18_infer" \
  python ./influencer_rank_full_fixed_xai_paper_no_imagecsv_v18.py \
    --mode infer \
    --ckpt "${CKPT_PTH}"

# ---------------------------
# サマリ出力
# ---------------------------
echo "================ SUMMARY ================"
ok=0
ng=0
for i in "${!names[@]}"; do
  name="${names[$i]}"
  rc="${statuses[$i]}"
  log="${logs[$i]}"
  if [[ "${rc}" -eq 0 ]]; then
    printf "✅  %-18s  rc=%-3s  log=%s\n" "${name}" "${rc}" "${log}"
    ok=$((ok+1))
  else
    printf "❌  %-18s  rc=%-3s  log=%s\n" "${name}" "${rc}" "${log}"
    ng=$((ng+1))
  fi
done
echo "----------------------------------------"
echo "SUCCESS: ${ok} / FAIL: ${ng}"
echo "logs dir: ${logdir}"
echo

# ---------------------------
# 再実行用コマンド（失敗したやつだけ）
# ---------------------------
echo "=========== RE-RUN COMMANDS (FAILED ONLY) ==========="
for i in "${!names[@]}"; do
  if [[ "${statuses[$i]}" -ne 0 ]]; then
    echo "# ${names[$i]} (rc=${statuses[$i]})"
    echo "${cmds[$i]}"
    echo
  fi
done

# 全部のコマンドも出す（コピペ用）
echo "=========== ALL COMMANDS (COPY/PASTE) ==========="
for i in "${!names[@]}"; do
  echo "# ${names[$i]}"
  echo "${cmds[$i]}"
  echo
done

# 終了コード：失敗が1つでもあれば 1 を返す（CI等でも使える）
if [[ "${ng}" -gt 0 ]]; then
  exit 1
fi
exit 0
