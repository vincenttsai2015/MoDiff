"""依實際資料設定 config 的 max_node_num 與 spec_dim。

這兩個是資料形狀參數，不是模型超參數：

  max_node_num  鄰接矩陣的 padding 大小。設得比實際最大節點數小，
                pad_adjs()（utils/graph_utils.py:340）會直接 raise；
                設得過大則記憶體以平方成長，且 eigh 的成本同樣以平方成長。
                各資料集的 config 都寫 1100，是沿用預設值，與實際差距很大。

  spec_dim      取前 k 個特徵值。必須 <= max_node_num，否則 top_k_eigen 的
                shape 對不上。只有在超過時才下修，並印出訊息。

其餘欄位一律不動。

用法：
    python set_data_dims.py <config.yaml> <data_dir> <file1_prefix>
"""
import pickle
import os
import re
import sys


def max_nodes_in(path):
    with open(path, "rb") as f:
        graphs = pickle.load(f)
    return max((g.number_of_nodes() for g in graphs), default=0), len(graphs)


def main():
    cfg_path, data_dir, file1 = sys.argv[1], sys.argv[2], sys.argv[3]

    overall = 0
    for suffix in ("R", "V", "T"):
        p = os.path.join(data_dir, file1 + suffix + ".pkl")
        if not os.path.exists(p):
            print("[ERROR] 找不到 {}".format(p))
            sys.exit(1)
        mx, n = max_nodes_in(p)
        overall = max(overall, mx)
        print("  {}{}.pkl  count={:<8d} max_nodes={}".format(file1, suffix, n, mx))

    text = open(cfg_path, encoding="utf-8").read()

    cur_max = int(re.search(r"^  max_node_num:\s*(\d+)", text, re.M).group(1))
    cur_spec = int(re.search(r"^  spec_dim:\s*(\d+)", text, re.M).group(1))

    new_max = overall
    new_spec = cur_spec
    if cur_spec > new_max:
        new_spec = new_max
        print("  [注意] spec_dim {} 超過實測最大節點數 {}，下修為 {}".format(
            cur_spec, new_max, new_spec))

    text = re.sub(r"^  max_node_num:.*$", "  max_node_num: {}".format(new_max),
                  text, flags=re.M)
    text = re.sub(r"^  spec_dim:.*$", "  spec_dim: {}".format(new_spec),
                  text, flags=re.M)
    with open(cfg_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    print("  max_node_num: {} -> {}".format(cur_max, new_max))
    print("  spec_dim    : {} -> {}".format(cur_spec, new_spec))


if __name__ == "__main__":
    main()
