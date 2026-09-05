#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --account=acd109125
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#
# 四類指標，五個模型。
#
#   sbatch run_eval_nano4.sh 0                    seed 0，能算的都算
#   KIND=lp sbatch --export=ALL run_eval_nano4.sh 0        只算連結預測
#   MODELS=GraphMaker sbatch --export=ALL run_eval_nano4.sh 0   只補這個模型
#
# KIND 可選 mmd / topo / attr / lp，逗號分隔，預設四類都跑。
# MODELS 讓已跑完的模型先算，之後補其他的不必把全部重算一遍。
#
#   RESULTS / MODIFF / GM  指定產出目錄
#   NPZ_ROOT               原始序列（topo 與 attr 需要），預設 ~/npz
#
# MMD 是 O(n²) 且每次比對要解 EMD，一個 seed 要數小時，所以走 sbatch。
# 產出目錄不存在的模型會被跳過，不會中斷。

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SEED=${1:?用法：sbatch run_eval_nano4.sh <seed>}
RESULTS=${RESULTS:-$HOME/test_and_generated_graphs}
MODIFF=${MODIFF:-$HOME/modiff_generated}
GM=${GM:-$HOME/gm_generated}
KIND=${KIND:-mmd,topo,attr,lp}
MODELS=${MODELS:-}
export NPZ_ROOT=${NPZ_ROOT:-$HOME/npz}

# 只算部分模型時輸出到另一組檔名，不要蓋掉完整那份。
_sfx=""
[ -n "$MODELS" ] && _sfx="_$(echo "$MODELS" | tr ',' '-')"
OUTDIR=${OUTDIR:-$HOME/eval_seed${SEED}}
mkdir -p "$OUTDIR" logs

PY=${EVAL_PY:-$HOME/miniconda3/envs/modiff/bin/python}
[ -x "$PY" ] || { echo "[ERROR] 找不到 $PY，用 EVAL_PY 指定"; exit 1; }

echo "=================================================="
echo " seed       : ${SEED}"
echo " 類別       : ${KIND}"
echo " 模型       : ${MODELS:-全部}"
echo " results    : ${RESULTS}"
echo " modiff     : ${MODIFF}"
echo " graphmaker : ${GM}"
echo " npz        : ${NPZ_ROOT}"
echo " 輸出       : ${OUTDIR}"
echo "=================================================="

has() { echo ",$KIND," | grep -q ",$1,"; }
opt_models() { [ -n "$MODELS" ] && echo "--models $MODELS"; }

if has mmd; then
    echo; echo "===== MMD 與 KS ====="
    A=(--results "$RESULTS" --skip-seed 1 --seed "$SEED"
       --csv "$OUTDIR/mmd_ks${_sfx}.csv")
    [ -d "$MODIFF" ] && A+=(--modiff "$MODIFF")
    [ -d "$GM" ]     && A+=(--graphmaker "$GM")
    # shellcheck disable=SC2046
    "$PY" -u eval_all_metrics.py "${A[@]}" $(opt_models)
fi

if has topo; then
    echo; echo "===== 拓撲指標 ====="
    if [ -d "$NPZ_ROOT/data_processed" ]; then
        # shellcheck disable=SC2046
        "$PY" -u macro_topo_eval.py "$SEED" "$RESULTS" "$MODIFF" "$GM" \
            $(opt_models) > "$OUTDIR/topo${_sfx}.csv"
        echo "寫出 $OUTDIR/topo${_sfx}.csv（$(wc -l < "$OUTDIR/topo${_sfx}.csv") 列）"
    else
        echo "[WARN] 找不到 ${NPZ_ROOT}/data_processed，跳過拓撲指標"
    fi
fi

if has attr; then
    echo; echo "===== 屬性指標 ====="
    if [ -d "$NPZ_ROOT/data_processed" ]; then
        A=(--results "$RESULTS" --seed-tag "$SEED"
           --csv "$OUTDIR/attr${_sfx}.csv")
        [ -d "$MODIFF" ] && A+=(--modiff "$MODIFF")
        [ -d "$GM" ]     && A+=(--graphmaker "$GM")
        # shellcheck disable=SC2046
        "$PY" -u attribute_decoder.py "${A[@]}" $(opt_models)
    else
        echo "[WARN] 找不到 ${NPZ_ROOT}/data_processed，跳過屬性指標"
    fi
fi

if has lp; then
    echo; echo "===== 連結預測 ====="
    A=(--results "$RESULTS" --seed-tag "$SEED" --csv "$OUTDIR/linkpred${_sfx}.csv")
    [ -d "$MODIFF" ] && A+=(--modiff "$MODIFF")
    [ -d "$GM" ]     && A+=(--graphmaker "$GM")
    # shellcheck disable=SC2046
    "$PY" -u temporal_link_pred.py "${A[@]}" $(opt_models)
fi

echo
echo "===== DONE ====="
ls -la "$OUTDIR"
