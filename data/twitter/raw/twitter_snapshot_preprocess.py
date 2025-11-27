import pickle
import networkx as nx
import pandas as pd

# 加載社交網絡數據
print('Loading social network')
social_network = pd.read_csv('higgs-social_network.edgelist', sep=' ', header=None, names=['userA', 'userB'])
G_social = nx.from_pandas_edgelist(social_network, 'userA', 'userB', create_using=nx.DiGraph()) # 456626 nodes, 14855842 edges
G_social_undir = G_social.to_undirected() # 456626 nodes, 12508436 edges 

largest_cc = max(nx.connected_components(G_social_undir), key=len)
G_social_induced = nx.induced_subgraph(G_social_undir, largest_cc) # 456290 nodes, 12508244 edges 

with open('twitter_social_graph.pkl','wb') as f:
    pickle.dump(G_social_induced,f)

# 加載活動數據
print('Loading user interactions')
activity_data = pd.read_csv('higgs-activity_time.txt', sep=' ', header=None, names=['userA', 'userB', 'timestamp', 'interaction'])

# 將 timestamp 轉換為 datetime 格式
print('Converting timestamp to datetime')
activity_data['datetime'] = pd.to_datetime(activity_data['timestamp'], unit='s')
# 設定時間間隔（例如，每天）
print('Setting time intervals')
activity_data.set_index('datetime', inplace=True)
time_intervals = pd.date_range(start=activity_data.index.min(), end=activity_data.index.max(), freq='D')

# 定義互動類型
interaction_types = ['RT', 'MT', 'RE']

# 初始化存儲圖的字典
layer_graphs = {interaction: [] for interaction in interaction_types}

# 遍歷每個時間間隔
print('Creating multi-layer graph snapshots')
for start_time in time_intervals[:-1]:
    print(f'Time {start_time}')
    end_time = start_time + pd.Timedelta(days=1)
    
    # 篩選該時間間隔內的活動
    interval_data = activity_data[start_time:end_time]
    
    # 遍歷每種類型的互動
    for interaction in interaction_types:
        # 篩選特定互動類型的數據
        interaction_data = interval_data[interval_data['interaction'] == interaction]
        
        # 構建圖
        G_interaction = nx.from_pandas_edgelist(interaction_data, 'userA', 'userB', create_using=nx.DiGraph())
        
        # 添加邊屬性
        print('Edge attribute assignment')
        for u, v in G_interaction.edges():
            if G_social.has_edge(u, v):
                G_interaction[u][v]['edge_attr'] = [0, 0, 1]  # 有好友關係且有互動
            else:
                G_interaction[u][v]['edge_attr'] = [0, 1, 0]  # 無好友關係但有互動
        
        # 添加節點屬性
        print('Node attribute assignment')
        for node in G_interaction.nodes():
            if G_interaction.degree(node) > 0:
                G_interaction.nodes[node]['x'] = [0, 1]  # 有互動
            else:
                G_interaction.nodes[node]['x'] = [1, 0]  # 無互動
        
        # 將圖添加到對應的層列表中
        layer_graphs[interaction].append(G_interaction)

# 初始化跨層連接圖
print('Creating the cross-layer graph')
G_cross = nx.Graph()

# 遍歷每個時間間隔的索引
print('Processing inter-layer edges')
for idx in range(len(time_intervals) - 1):
    print(f'Time {idx}')
    # 初始化該時間間隔的用戶層集合
    user_layers = {}
    
    # 遍歷每種類型的互動
    for interaction in interaction_types:
        G = layer_graphs[interaction][idx]
        
        # 遍歷圖中的節點
        for node in G.nodes():
            if node not in user_layers:
                user_layers[node] = []
            user_layers[node].append(interaction)
    
    # 為在多個層中出現的用戶添加跨層連接
    for user, layers in user_layers.items():
        if len(layers) > 1:
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    layer_i = layers[i]
                    layer_j = layers[j]
                    G_cross.add_edge((user, layer_i), (user, layer_j), edge_attr=[0, 1])