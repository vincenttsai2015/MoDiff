import math
import networkx as nx
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pickle
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.VisibleDeprecationWarning)


options = {
    'node_size': 2,
    'edge_color' : 'black',
    'linewidths': 1,
    'width': 0.5
}

def plot_graphs_list(graphs, title='title', max_num=16, save_dir=None, N=0):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        idx = i + max_num*N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)

def plot_graphs_list_DiT(graphs, title='title', max_num=16, save_dir=None, N=0):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure()

    for i in range(max_num):
        idx = i + max_num*N
        if not isinstance(graphs[idx], nx.DiGraph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.DiGraph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)

    save_fig(save_dir=save_dir, title=title)


def plot_graphs_list_huge(graphs, title='title', max_num=16, save_dir=None, N=0):
    batch_size = len(graphs)
    max_num = min(batch_size, max_num)
    img_c = int(math.ceil(np.sqrt(max_num)))
    figure = plt.figure(figsize=(12, 12))  # Increase figure size for better clarity

    for i in range(max_num):
        idx = i + max_num * N
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()

        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))  # Remove isolated nodes

        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        
        # Set up the subplot
        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        
        # Use a layout for better visualization
        pos = nx.spring_layout(G, k=0.1, seed=42)  # Reduce k to make edges less crowded

        # Edge attributes for clarity (thin edges, transparent arrows for directed graphs)
        edge_options = {
            'width': 0.5,  # Reduce edge width for clarity
            'edge_color': 'gray',  # Lighter color for edges
            'alpha': 0.7,  # Make edges semi-transparent
            'arrowsize': 10,  # Adjust size of arrows in directed graphs
        }
        
        # Handle directed graphs and drawing with arrows
        if G.is_directed():
            nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', **edge_options)
        else:
            nx.draw_networkx_edges(G, pos, edgelist=G.edges(), **edge_options)
        
        # Node drawing attributes (with node color and size)
        node_options = {
            'node_size': 50,  # Reduce node size for large graphs
            'node_color': 'skyblue',  # Node color
             # 'with_labels': False,  # Don't display node labels
        }
        
        # Draw the nodes
        nx.draw_networkx_nodes(G, pos, **node_options)

        # Add node labels if needed, controlling font size
        label_options = {
            'font_size': 8,  # Font size for labels
            'font_weight': 'bold',  # Bold labels for emphasis (optional)
        }
        if not G.is_directed():
            nx.draw_networkx_labels(G, pos, **label_options)
        
        ax.title.set_text(title_str)

    figure.suptitle(title, fontsize=16)  # Add title for the whole figure
    plt.tight_layout()

    # Save figure
    if save_dir:
        figure.savefig(f"{save_dir}/{title}.png", bbox_inches='tight', dpi=300)

    plt.show()



def save_fig(save_dir=None, title='fig', dpi=300):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(*['samples', 'fig', save_dir])
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=False)
        plt.close()
    return


def save_graph_list(log_folder_name, exp_name, gen_graph_list):

    if not(os.path.isdir('./samples/pkl/{}'.format(log_folder_name))):
        os.makedirs(os.path.join('./samples/pkl/{}'.format(log_folder_name)))
    with open('./samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name), 'wb') as f:
            pickle.dump(obj=gen_graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
    save_dir = './samples/pkl/{}/{}.pkl'.format(log_folder_name, exp_name)
