import pickle
import pandas as pd
import networkx as nx

df_social = pd.read_csv('sx-superuser.txt', delimiter=' ', header=None)
df_social.columns = ['source','target','timestamp']
G_social = nx.from_pandas_edgelist(df_social, source='source', target='target', create_using=nx.Graph)
with open('social_graph.pkl','wb') as f:
    pickle.dump(G_social,f)

df_c2a = pd.read_csv('sx-superuser-c2a.txt', delimiter=' ', header=None)
df_c2q = pd.read_csv('sx-superuser-c2q.txt', delimiter=' ', header=None)
df_a2q = pd.read_csv('sx-superuser-a2q.txt', delimiter=' ', header=None)

df_c2a.columns = ['source','target','timestamp']
df_c2q.columns = ['source','target','timestamp']
df_a2q.columns = ['source','target','timestamp']

df_c2a['datetime'] = pd.to_datetime(df_c2a['timestamp'], unit='s')
df_c2a = df_c2a.sort_values(by='datetime')
df_c2a['timeslot'] = df_c2a.datetime.dt.floor('1D')
df_c2a['interaction'] = 'c2a'

df_c2q['datetime'] = pd.to_datetime(df_c2q['timestamp'], unit='s')
df_c2q = df_c2q.sort_values(by='datetime')
df_c2q['timeslot'] = df_c2q.datetime.dt.floor('1D')
df_c2q['interaction'] = 'c2q'

df_a2q['datetime'] = pd.to_datetime(df_a2q['timestamp'], unit='s')
df_a2q = df_a2q.sort_values(by='datetime')
df_a2q['timeslot'] = df_a2q.datetime.dt.floor('1D')
df_a2q['interaction'] = 'a2q'

df_activity = pd.concat([df_c2a, df_c2q, df_a2q], axis=0, ignore_index=True)
# df_activity = df_activity[['source','target','datetime','interaction']]
# df_activity = df_activity.rename(columns={'datetime': 'timestamp'})
df_activity.to_csv('actions.csv', index=False)