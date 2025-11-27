import concurrent.futures
import os
import subprocess as sp
from datetime import datetime
import random
from tqdm import trange

from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
from scipy.stats import ks_2samp
from evaluation.mmd import process_tensor, compute_mmd, compute_mmd_WOnorm, gaussian, gaussian_emd, compute_nspdk_mmd
from utils.graph_utils import adjs_to_graphs

PRINT_TIME = False 
# -------- the relative path to the orca dir --------
ORCA_DIR = 'evaluation/orca'  


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    x, y = process_tensor(x, y)
    return x + y

# -------- Compute degree MMD --------
def degree_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian_emd, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    # -------- in case an empty graph is generated --------
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in trange(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in trange(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL)
    # mmd_dist = compute_mmd_WOnorm(sample_ref, sample_pred, kernel=KERNEL)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def spectral_worker(G):
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


# -------- Compute spectral MMD --------
def spectral_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian_emd, is_parallel=True):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)

    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    # print("clustering_coeffs_list:",clustering_coeffs_list)
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


# -------- Compute clustering coefficients MMD --------
# def clustering_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian, bins=100, is_parallel=True):
def clustering_stats(graph_ref_list, graph_pred_list, KERNEL=gaussian, bins=100, is_parallel=True):
    print("bins number:", bins)
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
    # print("graph_ref_list:",graph_ref_list)
    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                # print("clustering_hist:",clustering_hist)
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_pred_list_remove_empty]):
                # print("clustering_hist2:", clustering_hist)
                sample_pred.append(clustering_hist)
    else:
        for i in trange(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            print("clustering_coeffs_list:",clustering_coeffs_list)
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            # print("hist:", hist)
            sample_ref.append(hist)

        for i in trange(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            # print("hist2:", hist)
            sample_pred.append(hist)
    try:
        # print("sample_ref:",sample_ref)
        # print("sample_pred:", sample_pred)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, 
                            sigma=1.0 / 10, distance_scaling=bins)
    except:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist


# -------- maps motif/orbit name string to its corresponding list of indices from orca output --------
motif_to_indices = {
    '3path': [1, 2],
    '4cycle': [8],
}
COUNT_START_STR = 'orbit counts: \n'


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    tmp_input_file_path = os.path.join(ORCA_DIR, f'tmp-{random.random():.4f}.txt')
    f = open(tmp_input_file_path, 'w')
    f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
    for (u, v) in edge_list_reindexed(graph):
        f.write(str(u) + ' ' + str(v) + '\n')
    f.close()

    # output = sp.check_output([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_file_path, 'std'])
    # # print("output:", output)
    # output = output.decode('utf8').strip()

    tmp_output_file_path = os.path.join(ORCA_DIR, f'tmp-output-{random.random():.4f}.txt')
    sp.check_call([os.path.join(ORCA_DIR, 'orca'), 'node', '4', tmp_input_file_path, tmp_output_file_path])
    with open(tmp_output_file_path, 'r') as f:
        output = f.read()

    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR)
    output = output[idx:]
    node_orbit_counts = np.array([list(map(int, node_cnts.strip().split(' ')))
                                  for node_cnts in output.strip('\n').split('\n')])

    try:
        # os.remove(tmp_file_path)
        os.remove(tmp_input_file_path)
        os.remove(tmp_output_file_path)
    except OSError:
        pass

    return node_orbit_counts

def orbit_stats_all(graph_ref_list, graph_pred_list, KERNEL=gaussian):
    # print("in orbit_stats_all")
    total_counts_ref = []
    total_counts_pred = []

    prev = datetime.now()

    for G in graph_ref_list:
        try:
            # print("try to count orca(G)")
            orbit_counts = orca(G)
        except Exception as e:
            print(e)
            continue
        # print("Label  number_of_nodes:", G.number_of_nodes(), np.sum(orbit_counts, axis=0))
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G)
        except:
            print('orca failed')
            continue
        # print("Predict  number_of_nodes:", G.number_of_nodes(), np.sum(orbit_counts, axis=0))
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=KERNEL,
                           is_hist=False, sigma=30.0)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing orbit mmd: ', elapsed)
    return mmd_dist


def total_count_orbit(G):
    A = nx.adjacency_matrix(G).todense()
    n = G.number_of_nodes()

    A_undirected = np.maximum(A, A.T)
    A2 = np.matmul(A_undirected, A_undirected)
    A3 = np.matmul(A2, A_undirected)
    A3_mask = A3 > 0

    motif_counts = np.zeros((n, 1))
    for i in range(n):
        for j in range(n):
            if A3_mask[i, j] and A_undirected[j, i] > 0:
                motif_counts[i, 0] += 1
    motif_counts[:, 0] = motif_counts[:, 0] / 2
    return motif_counts.astype(int)
    
def orbit_stats_4DiG(graph_ref_list, graph_pred_list, KERNEL=gaussian):
    total_counts_ref = []
    total_counts_pred = []

    prev = datetime.now()

    for G in graph_ref_list:
        try:
            orbit_counts = total_count_orbit(G)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = total_count_orbit(G)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=KERNEL,
                           is_hist=False, sigma=30.0)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing Orbit mmd: ', elapsed)
    return mmd_dist

##### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/stats.py
def nspdk_stats(graph_ref_list, graph_pred_list):
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    mmd_dist = compute_nspdk_mmd(graph_ref_list, graph_pred_list_remove_empty, metric='nspdk', is_hist=False, n_jobs=20)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

# -------- Node Behavior Metric (2D KS using Q1 and Q3 of degree) -------- #
def extract_q1_q3_points(graph_seq):
    q1_q3_points = []
    for g in graph_seq:
        degrees = np.array([d for _, d in g.degree()])
        if len(degrees) < 4:
            continue
        q1 = np.percentile(degrees, 25)
        q3 = np.percentile(degrees, 75)
        q1_q3_points.append([q1, q3])
    return np.array(q1_q3_points)

def compute_2d_ks(test_points, gen_points):
    ks_x = ks_2samp(test_points[:, 0], gen_points[:, 0]).statistic
    ks_y = ks_2samp(test_points[:, 1], gen_points[:, 1]).statistic
    return (ks_x + ks_y) / 2

def run_node_behavior_eval(test_graphs, gen_graphs):
    test_points = extract_q1_q3_points(test_graphs)
    gen_points = extract_q1_q3_points(gen_graphs)
    
    ks_score = compute_2d_ks(test_points, gen_points)
    return ks_score

# -------- Dynamical Similarity via Random Walk Coverage -------- #
def random_walk_coverage(graph_seq, walk_times=10):
    T = len(graph_seq)
    cover_counts = []
    for _ in range(walk_times):
        visited = set()
        g0 = graph_seq[0]
        if len(g0.nodes) == 0:
            continue
        current = random.choice(list(g0.nodes))
        visited.add(current)
        for t in range(1, T):
            g = graph_seq[t]
            if current not in g:
                break
            neighbors = list(g.neighbors(current))
            if neighbors:
                current = random.choice(neighbors)
            visited.add(current)
        cover_counts.append(len(visited))
    return cover_counts

def run_dynamic_sim_eval(test_graphs, gen_graphs):
    test_cover = random_walk_coverage(test_graphs)
    gen_cover = random_walk_coverage(gen_graphs)

    return ks_2samp(test_cover, gen_cover).statistic

# -------- PageRank Behavior Metric (2D KS using Q1 and Q3 of degree) -------- #
def extract_pagerank_q1_q3_points(graph_seq):
    q1_q3_points = []
    for g in graph_seq:
        pagerank_scores = np.array([d for _, d in nx.pagerank(g).items()])
        if len(pagerank_scores) < 4:
            continue
        q1 = np.percentile(pagerank_scores, 25)
        q3 = np.percentile(pagerank_scores, 75)
        q1_q3_points.append([q1, q3])
    return np.array(q1_q3_points)

def run_pagerank_eval(test_graph_seq, gen_graph_seq):
    test_points = extract_pagerank_q1_q3_points(test_graph_seq)
    gen_points = extract_pagerank_q1_q3_points(gen_graph_seq)
    ks = compute_2d_ks(test_points, gen_points)
    return ks

# -------- Node Degree Behavior Metric (2D KS using Q1 and Q3 of degree) -------- #
def extract_degree_q1_q3_points(graph_seq):
    q1_q3_points = []
    for g in graph_seq:
        degrees = np.array([d for _, d in g.degree()])
        if len(degrees) < 4:
            continue
        q1 = np.percentile(degrees, 25)
        q3 = np.percentile(degrees, 75)
        q1_q3_points.append([q1, q3])
    return np.array(q1_q3_points)

def run_node_degree_behavior_eval(test_graph_seq, gen_graph_seq):
    test_points = extract_degree_q1_q3_points(test_graph_seq)
    gen_points = extract_degree_q1_q3_points(gen_graph_seq)
    ks = compute_2d_ks(test_points, gen_points)
    return ks

def extract_centrality_q1_q3_points(graph_seq, type):
    q1_q3_points = []
    for g in graph_seq:
        if type == 'degree':
            centralities = np.array([d for _, d in nx.degree_centrality(g).items()])
        elif type == 'betweenness':
            centralities = np.array([d for _, d in nx.betweenness_centrality(g).items()])
        elif type == 'closeness':
            centralities = np.array([d for _, d in nx.closeness_centrality(g).items()])
        elif type == 'eigenvector':
            centralities = np.array([d for _, d in nx.eigenvector_centrality(g).items()])
        elif type == 'information':
            centralities = np.array([d for _, d in nx.information_centrality(g).items()])
        
        if len(centralities) < 4:
            continue
        q1 = np.percentile(centralities, 25)
        q3 = np.percentile(centralities, 75)
        q1_q3_points.append([q1, q3])
    return np.array(q1_q3_points)

def run_centrality_behavior_eval(test_graph_seq, gen_graph_seq, type):
    test_points = extract_centrality_q1_q3_points(test_graph_seq, type=type)
    gen_points = extract_centrality_q1_q3_points(gen_graph_seq, type=type)
    ks = compute_2d_ks(test_points, gen_points)
    return ks

METHOD_NAME_TO_FUNC = {
    'degree': degree_stats,
    'cluster': clustering_stats,
    'orbit': orbit_stats_4DiG,
    'spectral': spectral_stats,
    'node_behavior_ks': run_node_behavior_eval,
    'random_walk_ks': run_dynamic_sim_eval,
    'pagerank_ks': run_pagerank_eval,
    'node_degree_behavior_ks': run_node_degree_behavior_eval,
    'degree_centrality_behavior_ks': run_centrality_behavior_eval,
    'betweenness_centrality_behavior_ks': run_centrality_behavior_eval,
    'closeness_centrality_behavior_ks': run_centrality_behavior_eval,
    'nspdk': nspdk_stats
}


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    graph_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, graph_pred_list, methods=methods)
    return results


# -------- Evaluate generated generic graphs --------
def eval_graph_list(graph_ref_list, graph_pred_list, methods=None, kernels=None):
    if methods is None:
        methods = ['degree', 'cluster', 'spectral', 'node_behavior_ks', 'random_walk_ks','pagerank_ks','node_degree_behavior_ks',
                   'degree_centrality_behavior_ks', 'betweenness_centrality_behavior_ks', 'closeness_centrality_behavior_ks'] #
    results = {}
    for method in methods:
        if method == 'nspdk':
            results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list)
        elif method in ['node_behavior_ks', 'random_walk_ks', 'pagerank_ks', 'node_degree_behavior_ks']:
            results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list))
        elif method == 'degree_centrality_behavior_ks':
            results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, type='degree'))
        elif method == 'betweenness_centrality_behavior_ks':
            results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, type='betweenness'))
        elif method == 'closeness_centrality_behavior_ks':
            results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, type='closeness'))
        else:
            results[method] = round(METHOD_NAME_TO_FUNC[method](graph_ref_list, graph_pred_list, kernels[method]), 6)
        print('\033[91m' + f'{method:9s}' + '\033[0m' + ' : ' + '\033[94m' +  f'{results[method]:.6f}' + '\033[0m')
    return results
