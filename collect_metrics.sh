#!/bin/bash
# 收集所有 run 的 sampling metrics。
#
#   bash collect_metrics.sh            每個 run 逐項列出
#   bash collect_metrics.sh csv        逗號分隔，方便貼進試算表
#
# 這支不用 set -e：grep 找不到東西時回傳 1 是正常情形，不是錯誤。

cd ~/MoDiff || { echo "[ERROR] 找不到 ~/MoDiff"; exit 1; }
shopt -s nullglob

MODE=${1:-table}
KEYS="degree cluster spectral node_behavior_ks random_walk_ks pagerank_ks node_degree_behavior_ks degree_centrality_behavior_ks betweenness_centrality_behavior_ks closeness_centrality_behavior_ks"

DIRS=(logs_sample/*/*/)
if [ ${#DIRS[@]} -eq 0 ]; then
    echo "[WARN] logs_sample/ 底下沒有任何 run 目錄" >&2
    echo "       代表取樣階段沒有跑完。先看 logs/slurm/*.err" >&2
    exit 0
fi

if [ "$MODE" = "csv" ]; then
    printf "run"
    for k in $KEYS; do printf ",%s" "$k"; done
    printf "\n"
fi

FOUND=0
for d in "${DIRS[@]}"; do
    run=$(basename "$d")
    line=$(grep -h "Final Metrics" "$d"*.log 2>/dev/null | tail -1)
    if [ -z "$line" ]; then
        echo "[WARN] ${run}: 沒有 Final Metrics（取樣未完成）" >&2
        continue
    fi
    FOUND=$((FOUND + 1))

    if [ "$MODE" = "csv" ]; then
        printf "%s" "$run"
        for k in $KEYS; do
            # 取到逗號或右大括號為止，這樣 nan / inf 也抓得到
            v=$(printf "%s" "$line" | sed -n "s/.*'${k}': \([^,}]*\).*/\1/p")
            printf ",%s" "$v"
        done
        printf "\n"
    else
        echo "=== $run ==="
        printf "%s\n" "$line" | sed "s/Final Metrics: //" | tr ',' '\n' | sed "s/[{}]//g;s/^ *//"
        echo
    fi
done

echo "[INFO] 共 ${#DIRS[@]} 個 run 目錄，其中 ${FOUND} 個有指標" >&2
