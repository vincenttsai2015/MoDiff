import torch
import torch.nn.functional as F
import networkx as nx
import numpy as np
import datetime
import json

# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

    if flags is None:
        flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
    flags = flags[:,:,None]
    return x * flags

def compute_overall_mean_degree(graph_list):
    mean_degrees = []
    for graph in graph_list:
        # print(len(graph.edges()), len(graph.nodes()))
        mean_degrees.append(len(graph.edges())/len(graph.nodes()))
        # if mean_degrees[-1] > 4: print(len(mean_degrees), mean_degrees[-1])
    overall_mean_degree = np.mean(mean_degrees)
    return overall_mean_degree

# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
    """
    :param adjs:  B x N x N or B x C x N x N
    :param flags: B x N
    :return:
    """
    # print("in mask_adjs")
    # print("flags:", flags.shape)
    # print("adjs:", adjs.shape)
    # for i in range(flags.shape[0]):
    #     print("flags[i]:",flags[i])
    if flags is None:
        flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

    if len(adjs.shape) == 4:
        flags = flags.unsqueeze(1)  # B x 1 x N
    adjs = adjs * flags.unsqueeze(-1)
    adjs = adjs * flags.unsqueeze(-2)
    return adjs


# -------- Create flags tensor from graph dataset --------
def node_flags(adj, eps=1e-5):

    flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

    if len(flags.shape)==3:
        flags = flags[:,0,:]
    return flags


# -------- Create initial node features --------
def init_features(init, adjs=None, nfeat=10, degree_map =None):
    #degrees
    if init=='zeros':
        feature = torch.zeros((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='ones':
        feature = torch.ones((adjs.size(0), adjs.size(1), nfeat), dtype=torch.float32, device=adjs.device)
    elif init=='deg':
        feature = adjs.sum(dim=-1).to(torch.long)
        feature = degree_map[feature]
        num_classes = nfeat
        try:
            feature = F.one_hot(feature, num_classes=num_classes).to(torch.float32)
        except:
            print("feature.max()", feature.max())
            raise NotImplementedError(f'max_feat_num mismatch')
    else:
        raise NotImplementedError(f'{init} not implemented')

    flags = node_flags(adjs)

    return mask_x(feature, flags)


# -------- Sample initial flags tensor from the training graph set --------
def init_flags(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)

    flags = node_flags(graph_tensor[idx])

    return flags

def init_flags2(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])
    selected_trains = graph_tensor[idx]

    return flags, selected_trains

def init_flags2_wnodes(graph_list, config, round, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num

    start_item = min(round*batch_size, len(graph_list)-batch_size)
    end_item = min((round+1)*batch_size, len(graph_list))
    adj_tensor, nodes_tensor = graphs_to_adjWnodes(graph_list[start_item: end_item], max_node_num)
    flags = node_flags(adj_tensor)
    selected_trains_adj = adj_tensor
    selected_trains_nodes = nodes_tensor
    return flags, selected_trains_adj, selected_trains_nodes

def init_flags2_wnodes_4Comp(graph_lists, config, round, batch_size=None):
    if batch_size is None:   batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num

    start_item = min(round*batch_size, len(graph_lists[0])-batch_size)
    end_item = min((round+1)*batch_size, len(graph_lists[0]))
    adj_tensor, nodes_tensor = graphs_to_adjWnodes_4Comp([graph_list[start_item: end_item] for graph_list in graph_lists], max_node_num)

    flags = node_flags(adj_tensor)
    selected_trains_adj = adj_tensor
    selected_trains_nodes = nodes_tensor
    return flags, selected_trains_adj, selected_trains_nodes


def init_flags3(graph_list, config, batch_size=None):
    if batch_size is None:
        batch_size = config.data.batch_size
    max_node_num = config.data.max_node_num
    graph_tensor = graphs_to_tensor(graph_list, max_node_num)
    idx = np.random.choice(list(range(len(graph_list))), size=batch_size, replace=False, p=None)
    # idx = np.random.randint(0, len(graph_list), batch_size)
    flags = node_flags(graph_tensor[idx])
    selected_trains = graph_tensor[idx]

    return flags, selected_trains


# -------- Generate noise --------
def gen_noise(x, flags, sym=True):
    z = torch.randn_like(x)
    if sym:
        z = z.triu(1)
        z = z + z.transpose(-1,-2)
        z = mask_adjs(z, flags)
    else:
        z = mask_x(z, flags)
    return z

def gen_spec_noise(adj, flags, u, la):
    z = torch.randn_like(adj, device=adj.device)
    eye = torch.eye(adj.shape[-1], device=adj.device)
    eye = eye.unsqueeze(0)
    eye = eye.repeat(adj.shape[0], 1, 1)
    z = z * eye
    u_T = torch.transpose(u, -1,-2)
    z = torch.bmm(torch.bmm(u, z),u_T )*np.sqrt(z.shape[-1])
    z = mask_adjs(z, flags)

    return z

def gen_spec_noise2(adj, flags, u, la):
    z = torch.randn_like(la, device=adj.device)
    return z


# -------- Quantize generated graphs --------
def quantize(adjs, thr=0.5):
    adjs_ = torch.where(adjs < thr, torch.zeros_like(adjs), torch.ones_like(adjs))
    return adjs_


def quantize_DegreeBound_4Comp(Gen_Adj_list, train_nodes_array , thres1=0.2, thres2=0.2, lower_bound=2, upper_bound=4):
    adj_f = []

    prev_values = []  # Store previous threshold values
    max_repeats = 50 

    while True:
        mean_degree = 0
        for i in range(len(train_nodes_array)):
            num_nodes = len(train_nodes_array[i])
            adj = np.asarray(Gen_Adj_list[i], dtype=np.complex64)

            result_matrix = np.zeros_like(adj, dtype=np.float32)

            real_H = np.real(adj)
            imag_H = np.imag(adj)

            bidirectional_mask = (np.abs(real_H - 0.5) <= thres1)  # | (np.abs(real_H + 1) <= thres1)
            result_matrix[bidirectional_mask] = 1
            bidirectional_mask = (np.abs(real_H - 1) <= thres1) # | (np.abs(real_H + 2) <= thres1)
            result_matrix[bidirectional_mask] = 1
            result_matrix = np.maximum(result_matrix, result_matrix.T)  # Ensure symmetry for bidirectional edges
            
            imag_condition = (np.abs(imag_H - 1) <= thres2)
            result_matrix[imag_condition] = 1

            imag_condition2 = (np.abs(imag_H - 0.5) <= thres2) 
            result_matrix[imag_condition2] = 1

            mean_degree += np.sum(result_matrix) / num_nodes

        mean_degree /= len(train_nodes_array)
    
        if lower_bound <= mean_degree <= upper_bound:
            break
        elif thres1 > 0.6:
            break
        elif mean_degree < lower_bound:
            thres1 += 0.002
            thres2 += 0.002
        else:
            thres1 -= 0.002
            thres2 -= 0.002

        prev_values.append((thres1, mean_degree))
        if len(prev_values) > max_repeats:
            prev_values.pop(0)

            # Detect oscillation
            if len(set(thres for thres, _ in prev_values)) < max_repeats:
                print("Detected oscillation, selecting best threshold...")

                best_thres1 = min(prev_values, key=lambda x: min(abs(x[1] - lower_bound), abs(x[1] - upper_bound)))[0]
                thres1 = best_thres1
                break
    mean_degree = 0
    for i in range(len(train_nodes_array)):
        num_nodes = len(train_nodes_array[i])
        adj = np.asarray(Gen_Adj_list[i], dtype=np.complex64)

        result_matrix = np.zeros_like(adj, dtype=np.float32)

        real_H = np.real(adj)
        imag_H = np.imag(adj)

        bidirectional_mask = (np.abs(real_H - 0.5) <= thres1)  # | (np.abs(real_H + 1) <= thres1)
        result_matrix[bidirectional_mask] = 1
        bidirectional_mask = (np.abs(real_H - 1) <= thres1) # | (np.abs(real_H + 2) <= thres1)
        result_matrix[bidirectional_mask] = 1
        result_matrix = np.maximum(result_matrix, result_matrix.T)  # Ensure symmetry for bidirectional edges
        
        imag_condition = (np.abs(imag_H - 1) <= thres2) 
        result_matrix[imag_condition] = 1

        imag_condition2 = (np.abs(imag_H - 0.5) <= thres2) 
        result_matrix[imag_condition2] = 1
            
        adj_f.append(torch.from_numpy(result_matrix))
        mean_degree += np.sum(result_matrix) / num_nodes

    mean_degree /= len(train_nodes_array)
    adj_f = torch.stack(adj_f)

    return adj_f, thres1

def adjs_to_graphs(adjs, is_cuda=False):
    graph_list = []
    for adj in adjs:
        if is_cuda:
            adj = adj.detach().cpu().numpy()
        # G = nx.from_numpy_matrix(adj)
        # print(np.array_equal(adj, adj.T), np.any(adj))
        G = nx.DiGraph(adj)
        G.remove_edges_from(nx.selfloop_edges(G))
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list

def adjsWnodes_to_graphs(adjs, nodes, is_cuda=False):
    graph_list = []
    for adj, node_list in zip(adjs, nodes):
        # Ensure the adjacency matrix is square and matches the length of the node list
        nonzero_indices = np.argwhere(adj.cpu() != 0)
        if len(nonzero_indices[0])>0: max_row_index = torch.max(nonzero_indices[0]).item() 
        else: max_row_index =0
        if len(nonzero_indices[1])>0: max_col_index = torch.max(nonzero_indices[1]).item()
        else: max_col_index =0
        assert max(max_row_index, max_col_index) <= len(node_list), "Adjacency Matrix is higher than Nodes"
        sub_adj = adj[:len(node_list), :len(node_list)]
        if is_cuda:
            sub_adj = sub_adj.detach().cpu().numpy()

        # N = sub_adj.shape[0]
        # directed_matrix = np.zeros((N, N), dtype=np.float32)
        # mask_upper = np.triu(np.ones((N, N), dtype=bool), k=0)
        # directed_matrix[mask_upper] = (sub_adj[mask_upper] == 1).astype(np.float32)
        # directed_matrix.T[mask_upper] = (sub_adj[mask_upper] == 2).astype(np.float32)

        G = nx.DiGraph(sub_adj)
        mapping = {i: node for i, node in enumerate(node_list)}
        G = nx.relabel_nodes(G, mapping)
        G.remove_edges_from(nx.selfloop_edges(G))
        # G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list



# -------- Check if the adjacency matrices are symmetric --------
def check_sym(adjs, print_val=False):
    sym_error = (adjs-adjs.transpose(-1,-2)).abs().sum([0,1,2])
    if not sym_error < 1e-2:
        raise ValueError(f'Not symmetric: {sym_error:.4e}')
    if print_val:
        print(f'{sym_error:.4e}')


# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
    # x : B x N x N
    x_ = x.clone()
    xc = [x.unsqueeze(1)]
    for _ in range(cnum-1):
        x_ = torch.bmm(x_, x)
        xc.append(x_.unsqueeze(1))
    xc = torch.cat(xc, dim=1)

    return xc


# -------- Create padded adjacency matrices --------
def pad_adjs(ori_adj, node_number):
    a = ori_adj
    ori_len = a.shape[-1]
    if ori_len == node_number:
        return a
    if ori_len > node_number:
        raise ValueError(f'ori_len {ori_len} > node_number {node_number}')
    a = np.concatenate([a, np.zeros([ori_len, node_number - ori_len])], axis=-1)
    a = np.concatenate([a, np.zeros([node_number - ori_len, node_number])], axis=0)
    return a

    
def graphs_to_tensor(graph_list, max_node_num):
    adjs_list = []
    max_node_num = max_node_num

    for g in graph_list:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)

        # adj = nx.to_numpy_matrix(g, nodelist=node_list)
        adj = nx.to_numpy_array(g, nodelist=node_list)
        padded_adj = pad_adjs(adj, node_number=max_node_num)
        adjs_list.append(padded_adj)

    del graph_list

    adjs_np = np.asarray(adjs_list)
    del adjs_list

    adjs_tensor = torch.tensor(adjs_np, dtype=torch.float32)
    del adjs_np

    return adjs_tensor 

def remove_small_parts(matrix, threshold=1e-8):
    real_parts = np.real(matrix)
    imag_parts = np.imag(matrix)
    real_parts[np.abs(real_parts) < threshold] = 0
    imag_parts[np.abs(imag_parts) < threshold] = 0
    return real_parts + 1j * imag_parts

# Compute the final matrix H
def compute_H(A):
    A_s =  0.5 * (A + A.T)
    exp_Theta = np.exp(1j * 0.5 * np.pi * (A - A.T))
    # Compute H
    H = A_s * exp_Theta
    H = remove_small_parts(H)
    return H

def compute_H_rotate(A, angle):
    A_s = 0.5 * (A + A.T)
    exp_Theta = np.exp(1j * 0.5 * np.pi * (A - A.T))
    H = A_s * exp_Theta
    rotation_factor = np.exp(1j * angle)
    H_rotated = H * rotation_factor

    H_rotated = remove_small_parts(H_rotated)
    
    return H_rotated

def compute_H_transformation(A, transformation_type):
    A_s =  0.5 * (A + A.T)
    exp_Theta = np.exp(1j * 0.5 * np.pi * (A - A.T))
    H = A_s * exp_Theta
    H = remove_small_parts(H)

    H = np.asarray(H, dtype=np.complex64)
    transformed_H = np.zeros_like(H)

    if transformation_type == 1:  transformed_H = H

    elif transformation_type == 2:
        # Transform bidirectional edges to 0.5
        unidirectional_edges = (np.imag(H) == 0.5)
        bidirectional_edges = (np.abs(np.real(H)) == 1)

        transformed_H[unidirectional_edges] = 0.25j
        transformed_H[unidirectional_edges.T] = -0.25j  # Ensuring Hermitian property
        transformed_H[bidirectional_edges] = 0.25
        transformed_H = transformed_H + transformed_H.T.conj()

    elif transformation_type == 3:
        # Transform unidirectional edges to -1j and bidirectional edges to -1
        unidirectional_edges = (np.imag(H) == 0.5)
        bidirectional_edges = (np.abs(np.real(H)) == 1)

        transformed_H[unidirectional_edges] = 0.5j
        transformed_H[unidirectional_edges.T] = -0.5j  # Ensuring Hermitian property
        transformed_H[bidirectional_edges] = 0.5
        transformed_H = transformed_H + transformed_H.T.conj()

    return transformed_H

def graphs_to_MultiD_tensor(graph_list, max_node_num):
    adjs_list = []
    max_node_num = max_node_num
    
    for g in graph_list:
        assert isinstance(g, nx.DiGraph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)

        # adj = nx.to_numpy_matrix(g, nodelist=node_list)
        adj = nx.to_numpy_array(g, nodelist=node_list)
    
        padded_adj = pad_adjs(adj, node_number=max_node_num)
        H = compute_H(padded_adj)
        H = torch.tensor(H, dtype=torch.complex64)


        # here because we need a fixed symmetric matrix, so padding first
        # symmetric_matrix = torch.zeros((max_node_num, max_node_num), dtype=torch.float32)
        # mask_upper = torch.triu(torch.ones(max_node_num, max_node_num), diagonal=1).bool()
        # mask_lower = torch.tril(torch.ones(max_node_num, max_node_num), diagonal=-1).bool()
        # symmetric_matrix[mask_upper] = (padded_adj.T[mask_upper] * 2) + padded_adj[mask_upper]
        # symmetric_matrix[mask_lower] = symmetric_matrix.T[mask_lower]

        adjs_list.append(H)
        if(len(adjs_list)%50 == 0): print(f"Graph To Magnetic Tensor: {len(adjs_list)} / {len(graph_list)}")

    del graph_list

    # adjs_np = np.asarray(adjs_list)
    # del adjs_list

    #adjs_tensor = torch.tensor(adjs_list, dtype=torch.float32)
    adjs_tensor = torch.stack(adjs_list)
    del adjs_list

    return adjs_tensor 


def graphs_to_MultiD_tensor_rotate(graph_lists, max_node_num):
    transform_type = [1, 2, 3]
    combined_adjs_list = []
    max_node_num = max_node_num

    assert len(graph_lists) == len(transform_type), "Number of angles must match number of graph lists"
    
    for i in range(len(graph_lists[0])):
        combined_H = np.zeros((max_node_num, max_node_num), dtype=np.complex64)
        
        for graph_list, type in zip(graph_lists, transform_type):
            g = graph_list[i]
            assert isinstance(g, nx.DiGraph)

            node_list = []
            for v, feature in g.nodes.data('feature'):
                node_list.append(v)

            adj = nx.to_numpy_array(g, nodelist=node_list)
            padded_adj = pad_adjs(adj, node_number=max_node_num)
            H = compute_H_transformation(padded_adj, type)
            #combined_H += H
            combined_H[H != 0] = H[H != 0]

        combined_H = torch.tensor(combined_H, dtype=torch.complex64)
        combined_adjs_list.append(combined_H)

        if(i%100 == 0):
            print(f"Graph To Magnetic Tensor: {i} / {len(graph_list)}")

    del graph_list
    adjs_tensor = torch.stack(combined_adjs_list)
    del combined_adjs_list

    return adjs_tensor 


def graphs_to_adjWnodes(graph_list, max_node_num):
    adjs_list = []
    nodes_list = []
    max_node_num = max_node_num

    for g in graph_list:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)
        node_list.sort()
        # adj = nx.to_numpy_matrix(g, nodelist=node_list)
        adj = nx.to_numpy_array(g, nodelist=node_list)
        nodes_list.append(node_list)
        if(adj.shape[0] > len(node_list)): print(f"adj.shape[0] {adj.shape[0]} > len(node_list){len(node_list)}")
        padded_adj = pad_adjs(adj, node_number=max_node_num)
        
        # padded_adj = torch.tensor(padded_adj, dtype=torch.float32)
        # symmetric_matrix = torch.zeros((max_node_num, max_node_num), dtype=torch.float32)
        # mask_upper = torch.triu(torch.ones(max_node_num, max_node_num), diagonal=1).bool()
        # mask_lower = torch.tril(torch.ones(max_node_num, max_node_num), diagonal=-1).bool()
        # symmetric_matrix[mask_upper] = (padded_adj.T[mask_upper] * 2) + padded_adj[mask_upper]
        # symmetric_matrix[mask_lower] = symmetric_matrix.T[mask_lower]

        padded_adj = pad_adjs(adj, node_number=max_node_num)
        H = compute_H(padded_adj)
        H = torch.tensor(H, dtype=torch.complex64)

        adjs_list.append(H)

    del graph_list

    # adjs_np = np.asarray(adjs_list)
    nodes_np = np.asarray(nodes_list)
    # del adjs_list, nodes_list

    #adjs_tensor = torch.tensor(adjs_list, dtype=torch.float32)
    adjs_tensor = torch.stack(adjs_list)
    del adjs_list, nodes_list
    #del adjs_np

    return adjs_tensor, nodes_np


def graphs_to_adjWnodes_4Comp(graph_lists, max_node_num):
    hermitian_lists = []
    nodes_lists = []
    max_node_num = max_node_num
    transform_type = [1, 2, 3]

    for g in graph_lists[0]:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)
        node_list.sort()
        nodes_lists.append(node_list)
                
           
    pointer = 0
    for i in range(len(graph_lists[0])):
        combined_H = np.zeros((max_node_num, max_node_num), dtype=np.complex64)
        
        for graph_list, type in zip(graph_lists, transform_type):
            g = graph_list[i]
            node_list = nodes_lists[i]
            for node in node_list:
                if not g.has_node(node):
                    g.add_node(node)
            adj = nx.to_numpy_array(g, nodelist=node_list)

            if(adj.shape[0] > len(node_list)):
                print(f"adj.shape[0] {adj.shape[0]} > len(node_list){len(node_list)}")

            padded_adj = pad_adjs(adj, node_number=max_node_num)
            H = compute_H_transformation(padded_adj, type)
            combined_H[H != 0] = H[H != 0]

        combined_H = torch.tensor(combined_H, dtype=torch.complex64)
        hermitian_lists.append(combined_H)

    del graph_lists
    
    # [Fix] Padding
    max_len = max([len(x) for x in nodes_lists])
    padded_nodes_lists = []
    for nl in nodes_lists:
        pad_len = max_len - len(nl)
        padded_nodes_lists.append(list(nl) + [-1] * pad_len)
    nodes_np = np.asarray(padded_nodes_lists)
    
    # nodes_np = np.asarray(nodes_lists)

    hermitian_tensor = torch.stack(hermitian_lists)
    del hermitian_lists, nodes_lists

    return hermitian_tensor, nodes_np

def graphs_to_adjWnodes_woMotif(graph_lists, max_node_num):
    hermitian_lists = []
    nodes_lists = []
    max_node_num = max_node_num
    transform_type = [1, 1, 1]

    for g in graph_lists[0]:
        assert isinstance(g, nx.Graph)
        node_list = []
        for v, feature in g.nodes.data('feature'):
            node_list.append(v)
        node_list.sort()
        nodes_lists.append(node_list)
                
           
    pointer = 0
    for i in range(len(graph_lists[0])):
        combined_H = np.zeros((max_node_num, max_node_num), dtype=np.complex64)
        
        for graph_list, type in zip(graph_lists, transform_type):
            g = graph_list[i]
            node_list = nodes_lists[i]
            adj = nx.to_numpy_array(g, nodelist=node_list)

            if(adj.shape[0] > len(node_list)):
                print(f"adj.shape[0] {adj.shape[0]} > len(node_list){len(node_list)}")

            padded_adj = pad_adjs(adj, node_number=max_node_num)
            H = compute_H_transformation(padded_adj, type)
            combined_H[H != 0] = H[H != 0]

        combined_H = torch.tensor(combined_H, dtype=torch.complex64)
        hermitian_lists.append(combined_H)

    del graph_lists

    nodes_np = np.asarray(nodes_lists)

    hermitian_tensor = torch.stack(hermitian_lists)
    del hermitian_lists, nodes_lists

    return hermitian_tensor, nodes_np


def graphs_to_adj(graph, max_node_num):
    max_node_num = max_node_num

    assert isinstance(graph, nx.Graph)
    node_list = []
    for v, feature in graph.nodes.data('feature'):
        node_list.append(v)

    # adj = nx.to_numpy_matrix(graph, nodelist=node_list)
    adj = nx.to_numpy_array(graph, nodelist=node_list)
    padded_adj = pad_adjs(adj, node_number=max_node_num)

    adj = torch.tensor(padded_adj, dtype=torch.float32)
    del padded_adj

    return adj


def node_feature_to_matrix(x):
    """
    :param x:  BS x N x F
    :return:
    x_pair: BS x N x N x 2F
    """
    x_b = x.unsqueeze(-2).expand(x.size(0), x.size(1), x.size(1), -1)  # BS x N x N x F
    x_pair = torch.cat([x_b, x_b.transpose(1, 2)], dim=-1)  # BS x N x N x 2F

    return x_pair


def combine_graphs(graph_list):
    """
    Combine multiple DiGraphs into a single DiGraph.

    Parameters:
    graph_list (list): List of NetworkX DiGraphs.

    Returns:
    DiGraph: A single combined DiGraph.
    """
    # Create an empty DiGraph to hold the combined graph
    combined_graph = nx.DiGraph()

    for g in graph_list:
        # Ensure each item in the list is a DiGraph
        assert isinstance(g, nx.DiGraph), "Each item in the list must be a DiGraph"
        # Compose the graphs
        combined_graph = nx.compose(combined_graph, g)

    return combined_graph

from itertools import combinations
from tqdm import tqdm
def count_motifs(G):
    motifs_count = {
        "single_directed_edge": 0,
        "feedforward_loop": 0,
        "cyclic_triad": 0,
        "two_path": 0,
        "three_node_feedback_loop": 0
    }

    nodes = list(G.nodes())

    # Single directed edge
    motifs_count["single_directed_edge"] = G.number_of_edges()
    total_combinations = sum(1 for _ in combinations(nodes, 3))
    # Check all combinations of 3 nodes for other motifs
    with tqdm(total=total_combinations, desc="Counting motifs") as pbar:

        for node_comb in combinations(nodes, 3):
            subgraph = G.subgraph(node_comb)
            num_edges = subgraph.number_of_edges()
            edges = list(subgraph.edges())

            if num_edges >= 6:
                # Three-node feedback loop: A <-> B <-> C <-> A
                all_bidirectional = True
                for u, v in edges:
                    if not subgraph.has_edge(v, u):
                        all_bidirectional = False
                        break
                if all_bidirectional:
                    motifs_count["three_node_feedback_loop"] += 1

            elif num_edges >= 3:
                # Cyclic triad: A -> B -> C -> A
                if (subgraph.has_edge(node_comb[0], node_comb[1]) and 
                    subgraph.has_edge(node_comb[1], node_comb[2]) and 
                    subgraph.has_edge(node_comb[2], node_comb[0])):
                    motifs_count["cyclic_triad"] += 1
                    continue

                # Feedforward loop: A -> B -> C and A -> C
                if (subgraph.has_edge(node_comb[0], node_comb[1]) and 
                    subgraph.has_edge(node_comb[1], node_comb[2]) and 
                    subgraph.has_edge(node_comb[0], node_comb[2])):
                    motifs_count["feedforward_loop"] += 1
                    continue

            elif num_edges == 2:
                # Two-path: A -> B -> C (A -> C should not exist)
                if ((subgraph.has_edge(node_comb[0], node_comb[1]) and 
                    subgraph.has_edge(node_comb[1], node_comb[2]) and 
                    not subgraph.has_edge(node_comb[0], node_comb[2])) or
                    (subgraph.has_edge(node_comb[1], node_comb[0]) and 
                    subgraph.has_edge(node_comb[2], node_comb[1]) and 
                    not subgraph.has_edge(node_comb[2], node_comb[0]))):
                    motifs_count["two_path"] += 1

            pbar.update(1)

        
    return motifs_count



def calculate_degree_distribution(graphs):
    degree_count = {}
    
    for G in graphs:
        degrees = [G.degree(n) for n in G.nodes]
        for degree in degrees:
            degree_count[degree] = degree_count.get(degree, 0) + 1

    if 0 in degree_count.keys(): del degree_count[0]
    total_nodes = sum(degree_count.values())
    degree_prob = {k: v / total_nodes for k, v in degree_count.items()}
    
    return degree_prob

def map_degree2class(degree_prob, N):
    max_degree = max(degree_prob.keys())
    N_list = [(i+1)*(1/(N-1)) for i in range(N-1)]
    de_sum = []
    sum_de = 0
    for i in range(max_degree):
        if i in degree_prob.keys(): sum_de+=degree_prob[i]
        de_sum.append(sum_de)

    mapped_degree = [0]
    mapped_index = []
    for n_value in N_list:
        closest_idx = np.argmin([abs(k_value - n_value) for k_value in de_sum])
        mapped_index.extend([closest_idx])
    mapped_index.insert(0,0)
    for i in range(N-1):
        mapped_degree.extend([i+1]*(mapped_index[i+1]-mapped_index[i]))
    return mapped_degree


def encode_to_reversed_binary(feature, num_bits):
    max_value = (1 << num_bits) - 1 

    # Clip feature values at the max_value to avoid overflow
    clipped_feature = torch.clamp(feature, max=max_value)

    binary_encoded = []
    for row in clipped_feature:
        # Convert each number in the row to binary, reversed, and padded to `num_bits`
        row_encoded = [
            [int(bit) for bit in format(int(min(x, max_value)), f'0{num_bits}b')[::-1]] for x in row ]
        binary_encoded.append(row_encoded)
    
    binary_encoded = np.array(binary_encoded, dtype=np.float32)
    binary_tensor = torch.tensor(binary_encoded, dtype=torch.float32)
    return binary_tensor

def upsert_dense_value(graphlists, key, json_file = 'utils/predensity.json' ):
    densevalue =  sum([compute_overall_mean_degree(graphlists[i]) for i in range(len(graphlists))])
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data[key] = round(densevalue - 0.1, 1)

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


start_date = datetime.datetime(2013, 12, 31)
def reformat_Reddit_timestamp(timestamp):
    return (timestamp.year - start_date.year) * 12 + (timestamp.month - start_date.month) + 1
