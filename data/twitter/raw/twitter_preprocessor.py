import pickle
import pandas as pd
import networkx as nx

df_activity = pd.read_csv('higgs-activity_time.txt', delimiter=' ', header=None)
df_activity.columns = ['source','target','timestamp','interaction']

df_activity['datetime'] = pd.to_datetime(df_activity['timestamp'], unit='s')
df_activity = df_activity.sort_values(by='datetime')
df_activity['timeslot'] = df_activity.datetime.dt.floor('1D')
df_activity = df_activity[['source','target','datetime','interaction']]
df_activity = df_activity.rename(columns={'datetime': 'timestamp'})
df_activity.to_csv('actions.csv', index=False)
# df_reply = df_activity[df_activity['action']=='RE']
# df_reply.to_csv('reply.csv', index=False)
# df_mention = df_activity[df_activity['action']=='MT']
# df_mention.to_csv('mention.csv', index=False)
# df_retweet = df_activity[df_activity['action']=='RT']
# df_retweet.to_csv('retweet.csv', index=False)

df_social = pd.read_csv('higgs-social_network.edgelist', delimiter=' ', header=None)
df_social.columns = ['user_id_1','user_id_2']
G_social = nx.from_pandas_edgelist(df_social, source='user_id_1', target='user_id_2', create_using=nx.Graph)
with open('social_graph.pkl','wb') as f:
    pickle.dump(G_social,f)

# MLG = {}
# MLG['RE'] = [] # reply
# MLG['MT'] = [] # mention
# MLG['RT'] = [] # retweet

# for t in timeslots:
#     df_activity_snapshot = df_activity[df_activity['timeslot'] == t]
#     df_mention_snapshot = df_activity_snapshot[df_activity_snapshot['action'] == 'MT']
#     df_reply_snapshot = df_activity_snapshot[df_activity_snapshot['action'] == 'RE']
#     df_retweet_snapshot = df_activity_snapshot[df_activity_snapshot['action'] == 'RT']
#     G_mention_snapshot = nx.from_pandas_edgelist(df_mention_snapshot, source='user_id_1', target='user_id_2', create_using=nx.Graph)
#     print(f'# Nodes in G_mention_snapshot at time {t}: {len(G_mention_snapshot.nodes())}')
#     print(f'# Edges in G_mention_snapshot at time {t}: {len(G_mention_snapshot.edges())}')
#     G_reply_snapshot = nx.from_pandas_edgelist(df_reply_snapshot, source='user_id_1', target='user_id_2', create_using=nx.Graph)
#     print(f'# Nodes in G_reply_snapshot at time {t}: {len(G_reply_snapshot.nodes())}')
#     print(f'# Edges in G_reply_snapshot at time {t}: {len(G_reply_snapshot.edges())}')
#     G_retweet_snapshot = nx.from_pandas_edgelist(df_retweet_snapshot, source='user_id_1', target='user_id_2', create_using=nx.Graph)
#     print(f'# Nodes in G_retweet_snapshot at time {t}: {len(G_retweet_snapshot.nodes())}')
#     print(f'# Edges in G_retweet_snapshot at time {t}: {len(G_retweet_snapshot.edges())}')