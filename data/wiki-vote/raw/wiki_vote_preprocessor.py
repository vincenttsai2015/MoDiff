import pandas as pd
import networkx as nx
import pickle

def parse_wiki_vote(file_path):
    """
    Parses the Wiki-Vote.txt file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Wiki-Vote.txt file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the directed edges (votes) with columns ['FromNodeId', 'ToNodeId'].
    """
    df = pd.read_csv(file_path, sep='\t', comment='#', header=None, names=['FromNodeId', 'ToNodeId'])
    return df

def parse_wiki_elec(file_path):
    """
    Parses the Wiki-Elec.txt file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the Wiki-Elec.txt file.
    
    Returns:
        pd.DataFrame: A DataFrame containing election results with columns:
        ['success', 'close_time', 'promoted_user_id', 'nominator_user_id', 'vote', 'voting_user_id', 'vote_time']
    """
    elections = []
    with open(file_path, 'r', encoding='latin-1') as f:
        election = {}
        for line in f:
            line = line.strip()
            if not line:
                if election and 'votes' in election:
                    elections.extend(election['votes'])
                election = {}
                continue
            
            parts = line.split('\t')
            if parts[0] == 'E':
                election['success'] = int(parts[1])
            elif parts[0] == 'T':
                election['close_time'] = parts[1]
            elif parts[0] == 'U':
                election['promoted_user_id'] = int(parts[1])
            elif parts[0] == 'N':
                election['nominator_user_id'] = int(parts[1])
                election['votes'] = []
            elif parts[0] == 'V':
                vote = int(parts[1])
                voting_user_id = int(parts[2])
                vote_time = parts[3]
                election['votes'].append([
                    election['success'],
                    election['close_time'],
                    election['promoted_user_id'],
                    election['nominator_user_id'],
                    vote,
                    voting_user_id,
                    vote_time
                ])
    
    df = pd.DataFrame(elections, columns=['success', 'close_time', 'promoted_user_id', 'nominator_user_id', 'vote', 'voting_user_id', 'vote_time'])

    # Convert time columns to datetime format
    df['close_time'] = pd.to_datetime(df['close_time'])
    df['vote_time'] = pd.to_datetime(df['vote_time'])
    
    # Sort by vote_time and close_time
    df = df.sort_values(by=['close_time','vote_time']).reset_index(drop=True)

    return df

def action_collector(df_elec):
    """
    Collects the actions ready for influence computation.
    
    Args:
        df_elec (pd.DataFrame): The election DataFrame.
    
    Returns:
        pd.DataFrame: DataFrame with actions.
    """
    action_data = []
    
    for _, group in df_elec.groupby(['promoted_user_id', 'close_time']):
        first_vote_time = group['vote_time'].min()
        
        # Nominations
        for _, row in group.iterrows():
            if row['nominator_user_id'] != -1:
                action_data.append([row['nominator_user_id'], row['promoted_user_id'], 'nominate', first_vote_time])
            
            # Votes
            if row['vote'] == 1:
                action_data.append([row['voting_user_id'], row['promoted_user_id'], 'support', row['vote_time']])
            elif row['vote'] == 0:
                action_data.append([row['voting_user_id'], row['promoted_user_id'], 'neutral', row['vote_time']])
            elif row['vote'] == -1:
                action_data.append([row['voting_user_id'], row['promoted_user_id'], 'oppose', row['vote_time']])
    
    action_df = pd.DataFrame(action_data, columns=['source', 'target', 'interaction', 'datetime'])
    action_df = action_df.drop_duplicates()
    
    # Compute probabilities based on user A influencing user B for the same action on the same target
    # grouped = action_df.groupby(['target', 'interaction'])
    # influence_prob_data = []
    
    # for (target, interaction), sub_df in grouped:
    #     pairs = sub_df.groupby('source')['time'].count().reset_index(name='count')
    #     total_counts = pairs['count'].sum()
    #     for _, row in pairs.iterrows():
    #         influence_prob_data.append([row['source'], target, interaction, row['count'] / total_counts])
    
    # influence_prob_df = pd.DataFrame(influence_prob_data, columns=['source', 'target', 'interaction', 'probability'])
    
    return action_df

if __name__ == '__main__':
    print('Parsing network data...')
    network_file_path = 'wiki-Vote.txt'
    df_network = parse_wiki_vote(network_file_path)
    df_network.to_csv('wiki-vote.csv', index=False)
    G_network_dir = nx.from_pandas_edgelist(df_network, source='FromNodeId', target='ToNodeId', create_using=nx.DiGraph)
    G_network_undir = G_network_dir.to_undirected()
    with open('wiki-vote_social_graph.pkl','wb') as f:
        pickle.dump(G_network_undir,f)
    print(f'#Nodes in wiki-vote: {len(G_network_undir.nodes())}')
    print(f'#Undirected Edges in wiki-vote: {len(G_network_undir.edges())}')
    print(f'#Directed Edges in wiki-vote: {len(G_network_dir.edges())}')

    print('Parsing election data...')
    election_file_path = 'wikiElec.ElecBs3.txt'
    df_election = parse_wiki_elec(election_file_path)
    df_election.to_csv('wiki-elec.csv', index=False)

    print('Collecting valid actions and calculating influence prob...')
    action_df = action_collector(df_election)
    
    action_df.to_csv('actions.csv', index=False)
    # influence_prob_df.to_csv('wiki-influence.csv', index=False)
