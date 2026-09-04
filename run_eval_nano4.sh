#!/bin/bash
#SBATCH --job-name=eval10
#SBATCH --account=acd109125
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval10_%j.out
#SBATCH --error=logs/eval10_%j.err
#
# 算十項 MMD/KS，五個模型一次算完。
#
#   sbatch run_eval_nano4.sh 0            seed 0
#   sbatch run_eval_nano4.sh 1            seed 1
#
# 產出目錄不在預設位置時用環境變數指定：
#   RESULTS=~/tg MODIFF=~/mg GM=~/gg sbatch --export=ALL run_eval_nano4.sh 0
#
# 只補算某個模型（其他四個不重算，省下數小時）：
#   MODELS=GraphMaker sbatch --export=ALL run_eval_nano4.sh 0
#
# MMD 是 O(n²) 且每次比對要解 EMD，一個 seed 要數小時，所以送運算節點。
# 只有缺席的模型會被跳過，不會中斷。

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SEED=${1:?用法：sbatch run_eval_nano4.sh <seed>}
RESULTS=${RESULTS:-$HOME/test_and_generated_graphs}
MODIFF=${MODIFF:-$HOME/modiff_generated}
GM=${GM:-$HOME/gm_generated}
MODELS=${MODELS:-}
# 只算部分模型時輸出到另一個檔，不要蓋掉完整那份。
_suffix=""
[ -n "$MODELS" ] && _suffix="_$(echo "$MODELS" | tr ',' '-')"
OUT=${OUT:-$HOME/eval_seed${SEED}${_suffix}.csv}

# 工具會 import MoDiff 的 evaluation.stats，所以要用那個環境的 python。
PY=${EVAL_PY:-$HOME/miniconda3/envs/modiff/bin/python}
[ -x "$PY" ] || { echo "[ERROR] 找不到 $PY，用 EVAL_PY 指定"; exit 1; }

mkdir -p logs

echo "=================================================="
echo " seed       : ${SEED}"
echo " results    : ${RESULTS}"
echo " modiff     : ${MODIFF}"
echo " graphmaker : ${GM}"
echo " 模型       : ${MODELS:-全部}"
echo " 輸出       : ${OUT}"
echo "=================================================="

ARGS=(--results "$RESULTS" --skip-seed 1 --seed "$SEED" --csv "$OUT")
[ -n "$MODELS" ] && ARGS+=(--models "$MODELS")
[ -d "$MODIFF" ] && ARGS+=(--modiff "$MODIFF") || echo "[WARN] 沒有 ${MODIFF}，跳過 MoDiff"
[ -d "$GM" ]     && ARGS+=(--graphmaker "$GM")  || echo "[WARN] 沒有 ${GM}，跳過 GraphMaker"

echo
"$PY" -u eval_all_metrics.py "${ARGS[@]}"

echo
echo "===== DONE  ${OUT}  $(wc -l < "$OUT") 列 ====="
