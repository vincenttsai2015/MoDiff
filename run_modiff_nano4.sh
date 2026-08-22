#!/bin/bash
#SBATCH --job-name=modiff
#SBATCH --account=ACD109125
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#
# 用法：
#   sbatch run_modiff_nano4.sh <dataset> <seed> [mode]
#
# 與 run_modiff.sh 的差別只在 SBATCH 標頭與 python 的取得方式：
# Nano4 用 --gres=gpu:1（不是 --gpus-per-node），partition 是 8gpus（48 小時），
# 每張 GPU 可配 12 cores / 200 GB。短測用 dev（4 小時）：
#
#   sbatch --partition=dev --time=04:00:00 run_modiff_nano4.sh macro_wiki_vote_burst_support 0
#
# 環境路徑用 MODIFF_PY 覆寫，預設找 ~/miniconda3 或 ~/.conda 底下的 modiff。
#
#   sbatch run_modiff.sh wiki-vote 42          訓練 + 取樣（預設）
#   sbatch run_modiff.sh twitter   42 eval     只取樣，沿用既有的 checkpoint
#
# 注入巨觀動態的資料用 macro_ 開頭，名稱對應 config/Macro/Macro_<名稱>.yaml：
#   sbatch run_modiff.sh macro_wiki_vote_burst_support 42
#
# mode=eval 用在訓練成功但取樣失敗的情況，省下重訓的時間。
#
# 模型與訓練參數一律沿用 config 內既有的設定，不覆寫。
# 只有 max_node_num 與 spec_dim 依實測資料調整（見 set_data_dims.py）。
#
# 產出命名一律 MoDiff_<dataset>_<seed>，多個 run 可以同時跑不會互相覆蓋。

set -eo pipefail
module purge

# batch shell 沒有 conda 這個函式，直接叫 env 裡的 python。
# 位置依 conda 怎麼裝的而不同，兩個常見的都試，都沒有就用 MODIFF_PY 指定。
PY=${MODIFF_PY:-}
if [ -z "$PY" ]; then
    for c in ~/miniconda3/envs/modiff/bin/python ~/.conda/envs/modiff/bin/python; do
        [ -x "$c" ] && { PY=$c; break; }
    done
fi
[ -n "$PY" ] && [ -x "$PY" ]     || { echo "[ERROR] 找不到 modiff 環境的 python，用 MODIFF_PY 指定"; exit 1; }
echo "python: $PY"
export PATH="$(dirname "$PY"):$PATH"
export PYTHONNOUSERSITE=1
export PIP_USER=0
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DS=${1:-wiki-vote}
SEED=${2:-42}
MODE=${3:-all}

case "$MODE" in
    all|eval) ;;
    *) echo "[ERROR] mode 只能是 all 或 eval，收到: $MODE"; exit 1 ;;
esac

# macro_ 開頭的是注入巨觀動態的資料，config 在 config/Macro/，資料已經轉好，
# 不需要跑 data_preprocess.py。
IS_MACRO=0
case "$DS" in
    macro_*)   IS_MACRO=1; FOLDER=Macro; SCALE=${DS#macro_} ;;
    wiki-vote) FOLDER=Wiki-vote; SCALE=25600  ;;
    twitter)   FOLDER=Twitter;   SCALE=204800 ;;
    superuser) FOLDER=Superuser; SCALE=320000 ;;
    *) echo "[ERROR] 未知的資料集: $DS"; exit 1 ;;
esac

RUNNAME="MoDiff_${DS}_${SEED}"
BASE_CFG="config/${FOLDER}/${FOLDER}_${SCALE}.yaml"
PREFIX="${FOLDER}_${SEED}"
CFG="config/${FOLDER}/${PREFIX}_${SCALE}.yaml"

export MODIFF_RUN_TAG="${RUNNAME}"

echo "=================================================="
echo " ${RUNNAME}   scale=${SCALE}"
echo " base config: ${BASE_CFG}"
echo " run  config: ${CFG}"
echo "=================================================="

[ -f "$BASE_CFG" ] || { echo "[ERROR] 找不到 $BASE_CFG"; exit 1; }
mkdir -p logs/slurm

# ---------- 產生本次的 config ----------
cp "$BASE_CFG" "$CFG"
sed -i "s/^  seed: .*/  seed: ${SEED}/"              "$CFG"
sed -i "s/^  name: .*/  name: ${RUNNAME}/"           "$CFG"

# repo 內的 config 是 CRLF，解析前先去掉 \r，否則路徑尾端會多一個字元
DATA_NAME=$(grep -E "^  data:"  "$CFG" | head -1 | tr -d '\r' | awk '{print $2}')
DATA_DIR=$(grep  -E "^  dir:"   "$CFG" | head -1 | tr -d '\r' | sed "s/.*'\(.*\)'.*/\1/")
FILE1=$(grep     -E "^  file1:" "$CFG" | head -1 | tr -d '\r' | awk '{print $2}')

echo "--- 資料位置 ---"
echo " data.data = ${DATA_NAME}"
echo " data.dir  = ${DATA_DIR}"
echo " data.file1= ${FILE1}"

# ---------- 前處理（缺才做） ----------
if [ "$IS_MACRO" = "1" ]; then
    # 讀的是 config.data.file1 加上 R/V/T，短檔名 R.pkl 那組是重複的，不一定會傳
    [ -f "${DATA_DIR}/${FILE1}R.pkl" ]         || { echo "[ERROR] 找不到 ${DATA_DIR}/${FILE1}R.pkl"; exit 1; }
    echo "===== 資料已轉好，略過 PREPROCESS ====="
elif [ ! -f "${DATA_DIR}/R.pkl" ]; then
    echo
    echo "===== PREPROCESS ====="
    $PY data_preprocess.py \
        --dataset-path "./data/${DS}_raw/actions.csv" \
        --output-dir   "${DATA_DIR}" \
        --num-bins     ${SCALE} \
        --dataset-name ${DS}
else
    echo "===== PREPROCESS 已存在，略過 ====="
fi

# ---------- 依實測資料設定 max_node_num / spec_dim ----------
echo
echo "===== 資料形狀 ====="
$PY set_data_dims.py "$CFG" "$DATA_DIR" "$FILE1"

echo
echo "--- config 生效值 ---"
grep -nE '^  (seed|name|num_epochs|use_ema|max_node_num|spec_dim|batch_size|test_split):' "$CFG"

# ---------- 訓練 ----------
CKPT="${RUNNAME}_comp"
if [ "$MODE" = "eval" ]; then
    echo
    echo "===== TRAIN 略過（mode=eval）====="
    [ -f "checkpoints/${DATA_NAME}/${CKPT}.pth" ] \
        || { echo "[ERROR] mode=eval 但找不到 checkpoints/${DATA_NAME}/${CKPT}.pth"; exit 1; }
else
    echo
    echo "===== TRAIN ====="
    $PY main.py --type train_comp --scale ${SCALE} \
        --config_folder "${FOLDER}" --config_prefix "${PREFIX}" --seed "${SEED}"

    [ -f "checkpoints/${DATA_NAME}/${CKPT}.pth" ] \
        || { echo "[ERROR] 找不到 checkpoints/${DATA_NAME}/${CKPT}.pth"; exit 1; }
fi
echo "===== CKPT = ${CKPT} ====="

# ---------- 取樣 + 評估 ----------
echo
echo "===== SAMPLE + EVAL ====="
$PY main.py --type eval_comp --scale ${SCALE} \
    --config_folder "${FOLDER}" --config_prefix "${PREFIX}" --ckpt_name "${CKPT}"

echo
echo "===== DONE  ${RUNNAME} ====="
# grep 找不到時回傳 1，配上 set -e + pipefail 會讓成功的 job 被記成 FAILED
grep -h "Final Metrics" "logs_sample/${DATA_NAME}/${RUNNAME}/"*.log 2>/dev/null | tail -1 || true
exit 0
