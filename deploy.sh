#!/bin/bash
# 安全地套用部署包，即使有 job 正在跑也不會影響它們。
#
#   bash deploy.sh [tarball]        預設 ~/modiff_deploy.tar.gz
#
# 為什麼不直接 tar xzf：
#   tar 是就地截斷寫入，檔案的 inode 不變。而 bash 執行腳本是邊讀邊執行、
#   記著位元組位移，執行中的腳本被覆寫的話，bash 下一次讀取會落進新內容的
#   中間，導致整段被跳過而且還正常結束（exit 0），非常難察覺。
#
#   這裡先解到暫存目錄再用 mv 搬過去。mv 在同一個檔案系統內是 rename，
#   會建立新的 inode，執行中的行程繼續讀舊的那份，互不干擾。

set -eo pipefail

TARBALL=${1:-~/modiff_deploy.tar.gz}
DEST=~/MoDiff

[ -f "$TARBALL" ] || { echo "[ERROR] 找不到 $TARBALL"; exit 1; }
[ -d "$DEST" ]    || { echo "[ERROR] 找不到 $DEST"; exit 1; }

N_RUNNING=$(squeue -u "$USER" -h 2>/dev/null | wc -l)
if [ "$N_RUNNING" -gt 0 ]; then
    echo "[注意] 目前有 ${N_RUNNING} 個 job 在 queue 裡。"
    echo "       這支腳本用 mv 套用，執行中的 job 不會被影響，"
    echo "       但它們用的仍是舊版本的腳本與程式碼。"
    echo
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "解開到暫存目錄..."
tar xzf "$TARBALL" -C "$TMP"

echo "搬移..."
COUNT=0
while IFS= read -r -d '' f; do
    rel=${f#"$TMP"/}
    mkdir -p "$DEST/$(dirname "$rel")"
    mv -f "$f" "$DEST/$rel"
    COUNT=$((COUNT + 1))
done < <(find "$TMP" -type f -print0)

echo "完成，共 ${COUNT} 個檔案。"
echo
echo "--- 重點檔案 ---"
ls -l --time-style=+%H:%M "$DEST/main.py" "$DEST/solver.py" "$DEST/trainer.py" \
                          "$DEST/run_modiff.sh" "$DEST/submit_modiff.sh" 2>/dev/null || true
