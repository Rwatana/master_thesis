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

# 2) exp2 infer
run_cmd "01_lossA_logrank_calib_train" \
  python ./influencer_rank_full_fixed_xai_paper_no_imagecsv_v19_lossA_logrank_calib.py \
    --mode train

run_cmd "02_lossA_logrank_sampler_train" \
  python ./influencer_rank_full_fixed_xai_paper_no_imagecsv_v19_lossA_logrank_sampler.py \
    --mode train

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
