import re

max_nodes = 0

with open("output.txt", "r", encoding="utf-8") as f:
    for line in f:
        # 尋找 "Graph created: X nodes, Y edges"
        match = re.search(r"Graph created: (\d+) nodes,", line)
        if match:
            nodes = int(match.group(1))
            if nodes > max_nodes:
                max_nodes = nodes

print("最大的 node 數:", max_nodes)

