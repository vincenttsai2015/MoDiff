#!/bin/bash
# 在 TWCC 執行。把 MoDiff 的結果打包成一個檔，方便一次抓回本機。
#
#   bash pack_results.sh              指標 + log + 生成圖
#   bash pack_results.sh --with-ckpt  另外含 checkpoint
#
# 輸出： ~/modiff_results.tar.gz

cd ~/MoDiff || { echo "[ERROR] 找不到 ~/MoDiff"; exit 1; }
shopt -s nullglob
OUT=~/modiff_results.tar.gz

echo "===== 目前有什麼 ====="
for d in logs_sample logs_train samples/fig logs/slurm checkpoints; do
    if [ -d "$d" ]; then
        printf "  %-16s %s，%s\n" "$d" \
            "$(find "$d" -type f | wc -l) 個檔" "$(du -sh "$d" | cut -f1)"
    else
        printf "  %-16s 不存在\n" "$d"
    fi
done

echo
echo "===== 產生指標彙整 ====="
bash collect_metrics.sh csv > modiff_metrics.csv
echo "--- modiff_metrics.csv ---"
if command -v column >/dev/null 2>&1; then
    column -s, -t modiff_metrics.csv
else
    cat modiff_metrics.csv
fi

ITEMS=(modiff_metrics.csv)
for d in logs_sample logs_train samples/fig logs/slurm; do
    [ -d "$d" ] && ITEMS+=("$d")
done
if [ "$1" = "--with-ckpt" ] && [ -d checkpoints ]; then
    ITEMS+=(checkpoints)
fi

echo
echo "===== 打包 ====="
echo "內容: ${ITEMS[*]}"
rm -f "$OUT"
tar czf "$OUT" "${ITEMS[@]}" || { echo "[ERROR] 打包失敗"; exit 1; }

echo
ls -lh "$OUT"
echo
echo "本機（WSL）抓回："
echo "  scp roy12358@ln01.twcc.ai:~/modiff_results.tar.gz /mnt/c/Users/User/Desktop/intern/refs/results/"
