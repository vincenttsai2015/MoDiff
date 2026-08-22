#!/bin/bash
# 在登入節點執行（不是 sbatch）。一次把指定資料集 x 指定 seed 的所有 run 送出去。
#
#   bash submit_modiff.sh                                  三個資料集 x 三個 seed = 9 個 job
#   bash submit_modiff.sh "wiki-vote" "42 123 2024"        指定資料集與 seed
#   bash submit_modiff.sh "wiki-vote twitter" "42"
#   bash submit_modiff.sh "twitter" "42 123 2024" eval     只重跑取樣，沿用既有 checkpoint
#
# TWCC 的 queue 上限是 20 個 job（pending + running 都算），送之前先看一下 squeue。

set -eo pipefail
cd ~/MoDiff

DATASETS=${1:-"wiki-vote twitter superuser"}
SEEDS=${2:-"42 123 2024"}
MODE=${3:-all}   # all = 訓練+取樣；eval = 只取樣，沿用既有 checkpoint

N_EXIST=$(squeue -u "$USER" -h | wc -l)
N_NEW=$(( $(echo $DATASETS | wc -w) * $(echo $SEEDS | wc -w) ))
echo "queue 現有 ${N_EXIST} 個 job，本次要送 ${N_NEW} 個（上限 20）"
if [ $(( N_EXIST + N_NEW )) -gt 20 ]; then
    echo "[ERROR] 會超過上限，請分批送或等前面清掉"
    exit 1
fi
echo

for DS in $DATASETS; do
    for S in $SEEDS; do
        JID=$(sbatch --parsable --job-name="modiff_${DS}_${S}" run_modiff.sh "$DS" "$S" "$MODE")
        printf "  %-28s job %s  (%s)\n" "MoDiff_${DS}_${S}" "$JID" "$MODE"
    done
done

echo
echo "追蹤： squeue -u \$USER"
