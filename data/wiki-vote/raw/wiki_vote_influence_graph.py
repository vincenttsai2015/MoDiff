import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime, timedelta

print('Load social graph...')
with open('social_graph.pkl','rb') as f:
    G_social = pickle.load(f) # undirected social graph

# user_actions[user][behavior][target] = {timestamps}
user_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

print('Load action files...')
action_df = pd.read_csv('actions.csv')
behavior_types = action_df['interaction'].unique().tolist()

for behavior_type in behavior_types:  # 多個不同行為
    sub_df = action_df[action_df['interaction']==behavior_type]

    for _, row in sub_df.iterrows():
        user = row["source"]
        target = row["target"]        
        timestamp = str(row["datetime"])
        user_actions[user][behavior_type][target].add(timestamp)

# print('Save action dict...')
# with open('user_action_dict.pkl','wb') as f:
#     pickle.dump(user_actions,f)

print('Start influence analysis')
T = timedelta(days=1)  # 影響窗口
influence_counts = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # influence_counts[behavior][(A,B)] = [影響次數, 總次數]

for A in G_social.nodes():
    for B in G_social.neighbors(A):  # 只考慮好友
        print(f'Finding social influences of {A} on {B}...')
        for behavior, A_targets in user_actions[A].items():
            if behavior not in user_actions[B]:  
                continue  # B 沒有這個行為，跳過
            
            for target, A_times in A_targets.items():
                if target not in user_actions[B][behavior]:  
                    continue  # B 沒對相同 target 採取相同行為，跳過
                
                B_times = user_actions[B][behavior][target]

                for t_A in A_times:
                    t_A = datetime.strptime(t_A, "%Y-%m-%d %H:%M:%S")
                    # 檢查 B 是否在 (t_A, t_A + T) 內對相同 target 採取相同行為
                    print('Check influence...')
                    influenced = any(datetime.strptime(t_B, "%Y-%m-%d %H:%M:%S") <= t_A + T for t_B in B_times)

                    influence_counts[behavior][(A, B)][1] += 1  # A 做了 X 的次數
                    if influenced:
                        influence_counts[behavior][(A, B)][0] += 1  # A 影響了 B
                
# 計算機率
print('Influence probability...')
influence_prob = {behavior: {pair: count[0] / count[1] if count[1] > 0 else 0 for pair, count in pairs.items()} for behavior, pairs in influence_counts.items()}
with open('influence_prob.pkl','wb') as f:
    pickle.dump(influence_prob,f)

avg_influence = {behavior: sum(p.values()) / len(p) for behavior, p in influence_prob.items()}
print(avg_influence)

influential_users = {behavior: max(p, key=p.get) for behavior, p in influence_prob.items()}
print(influential_users)

print('Visualization')
# 設定顏色對應
behavior_colors = {"support": "red", "oppose": "green", "neutral": "blue", "nominate": "purple"}

# 創建畫布
plt.figure(figsize=(8, 6))

# 計算佈局 (確保兩個網絡的節點位置相同)
pos = nx.spring_layout(G_social, seed=42)  # 固定佈局，確保兩圖一致

# 1. 畫 G_social（黑色粗線）
nx.draw(G_social, pos, with_labels=False, edge_color="black", width=2, alpha=0.5, node_size=150)

# 2. 疊加畫 G_influence（不同行為不同顏色）
for behavior, pairs in influence_prob.items():
    edges = [(A, B) for (A, B), prob in pairs.items() if prob > 0.1]  # 過濾影響力較高的邊
    weights = [pairs[(A, B)] * 5 for (A, B) in edges]  # 影響力高的邊較粗
    nx.draw_networkx_edges(G_social, pos, edgelist=edges, edge_color=behavior_colors[behavior], width=weights, alpha=0.7)

# 3. 加上圖例
from matplotlib.patches import Patch
legend_handles = [Patch(color=color, label=behavior) for behavior, color in behavior_colors.items()]
plt.legend(handles=legend_handles, loc="best")

# 顯示圖像
plt.title("Social Network & Influence Propagation")
plt.savefig('influence.jpg')