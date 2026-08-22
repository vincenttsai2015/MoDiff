#!/bin/bash
# 掃出哪些組合還沒跑完，並印出補跑指令。
#
#   bash check_modiff.sh                 列出缺漏
#   bash check_modiff.sh --submit        把缺的送出去（受 queue 名額限制）
#   ONLY=wiki_vote bash check_modiff.sh
#   SEEDS="0 1 2" bash check_modiff.sh
#   LIMIT=10 bash check_modiff.sh --submit
#
# 完成的判準是兩個：checkpoint 存在（訓練完），以及取樣 log 裡有 Final Metrics
# （取樣加評估完）。只有 checkpoint 的用 mode=eval 補，省下重訓。
#
# 執行中的組合從 scontrol 的 Command= 讀參數判斷，不論用哪種方式送出都認得出來。

set -eo pipefail
cd "$(dirname "$0")"

RUNNER=${RUNNER:-run_modiff_nano4.sh}
SEEDS=${SEEDS:-0}
LIMIT=${LIMIT:-20}
SUBMIT=0
[ "$1" = "--submit" ] && SUBMIT=1

[ -f "$RUNNER" ] || { echo "[ERROR] 找不到 $RUNNER，用 RUNNER= 指定"; exit 1; }

RUNNING=""
N_RUNNING=$( { squeue -u "$USER" -h 2>/dev/null || true; } | wc -l )
if [ "$N_RUNNING" -gt 0 ]; then
    echo "queue 裡有 ${N_RUNNING} 個 job。"
    # 送出時帶的 job name 就是 MoDiff_<組合>_<seed>，直接讀。
    # -o "%j" 給的是完整名稱（預設欄位才會截斷成 MoDiff_m）。
    for nm in $(squeue -u "$USER" -h -o "%j" 2>/dev/null); do
        case "$nm" in
            MoDiff_*) RUNNING="${RUNNING} ${nm#MoDiff_}" ;;
        esac
    done

    # 舊的 job 可能沒有 job name，再從 scontrol 的參數補一次
    for j in $(squeue -u "$USER" -h -o "%i" 2>/dev/null); do
        cmd=$(scontrol show job "$j" 2>/dev/null | grep -o 'Command=.*' | head -1)
        cmd=${cmd#Command=}
        set -- $cmd
        case "$1" in
            *run_modiff*) [ -n "$2" ] && RUNNING="${RUNNING} ${2}_${3}" ;;
        esac
    done

    echo "辨識出執行中的組合：$(echo $RUNNING | wc -w) 個"
fi

N_OK=0; N_MISS=0; N_RUN=0; N_TRAIN=0
MISSING=""

CFGS=$(find config/Macro -maxdepth 1 -name 'Macro_*.yaml' 2>/dev/null | sort)
[ -n "$CFGS" ] || { echo "[ERROR] config/Macro 底下沒有 Macro_*.yaml"; exit 1; }

for cfg in $CFGS; do
    base=$(basename "$cfg" .yaml)          # Macro_wiki_vote_burst_support
    name=${base#Macro_}                    # wiki_vote_burst_support
    # run_modiff.sh 每次執行會產生 Macro_<seed>_<組合>.yaml。那不是組合，
    # 掃進來會被當成新組合送出去，再產生一份，一路長下去。
    case "$name" in
        [0-9]*) continue ;;
    esac
    if [ -n "$ONLY" ] && [ "${name#*$ONLY}" = "$name" ]; then continue; fi

    for seed in $SEEDS; do
        ds="macro_${name}"
        key="${ds}_${seed}"
        case " $RUNNING " in
            *" $key "*) N_RUN=$((N_RUN + 1)); continue ;;
        esac

        run="MoDiff_${ds}_${seed}"
        ckpt="checkpoints/${name}/${run}_comp.pth"
        done_mark=0
        if ls "logs_sample/${name}/${run}/"*.log >/dev/null 2>&1; then
            grep -lq "Final Metrics" "logs_sample/${name}/${run}/"*.log 2>/dev/null \
                && done_mark=1
        fi

        if [ "$done_mark" = "1" ]; then
            N_OK=$((N_OK + 1))
        elif [ -f "$ckpt" ]; then
            # 訓練完了，只缺取樣。用 eval 省下重訓
            N_TRAIN=$((N_TRAIN + 1)); N_MISS=$((N_MISS + 1))
            printf '%-52s 缺: 取樣（有 checkpoint）\n' "$key"
            MISSING="${MISSING}sbatch --job-name=${run} ${RUNNER} ${ds} ${seed} eval
"
        else
            N_MISS=$((N_MISS + 1))
            printf '%-52s 缺: 訓練 + 取樣\n' "$key"
            MISSING="${MISSING}sbatch --job-name=${run} ${RUNNER} ${ds} ${seed} all
"
        fi
    done
done

N_SCANNED=$((N_OK + N_MISS))
echo
echo "掃描 ${N_SCANNED} 組（另有 ${N_RUN} 組執行中不列入）"
echo "  完成 ${N_OK} 組，有缺 ${N_MISS} 組（其中 ${N_TRAIN} 組已有 checkpoint，只補取樣）"
[ "$N_MISS" -eq 0 ] && exit 0

echo
echo "--- 補跑指令 ---"
printf '%s' "$MISSING"

if [ "$SUBMIT" = "1" ]; then
    SLOTS=$((LIMIT - N_RUNNING))
    N_CMD=$(printf '%s' "$MISSING" | grep -c . || true)
    echo
    echo "queue 現有 ${N_RUNNING} 個，上限 ${LIMIT}，這次可送 ${SLOTS} 個（待補 ${N_CMD} 個）"
    if [ "$SLOTS" -le 0 ]; then
        echo "沒有名額，等 queue 空出來再執行一次"
        exit 0
    fi

    echo
    echo "--- 送出 ---"
    n=0
    printf '%s' "$MISSING" | while read -r cmd; do
        [ -n "$cmd" ] || continue
        n=$((n + 1))
        [ "$n" -gt "$SLOTS" ] && break
        echo "$cmd"
        eval "$cmd"
    done

    if [ "$N_CMD" -gt "$SLOTS" ]; then
        echo
        echo "還有 $((N_CMD - SLOTS)) 個沒送，等 queue 空出來再執行一次"
    fi
fi
