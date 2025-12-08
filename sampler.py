import os
import time
import pickle, json
import math
import torch
import random

from utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from utils.loader import load_ckpt,load_graph_list, load_data_TD_test, load_seed, load_device, load_model_from_ckpt, \
    load_ema_from_ckpt, load_eval_settings, load_sampling_fn4Di_spec
from utils.graph_utils import adjsWnodes_to_graphs, mask_adjs, init_flags2_wnodes_4Comp, quantize_DegreeBound_4Comp,  compute_overall_mean_degree
from utils.plot import save_graph_list, plot_graphs_list
from evaluation.stats import eval_graph_list
import numpy as np
import networkx as nx
from evaluation.mmd import gaussian, gaussian_emd
from evaluation.stats import eval_graph_list

def rewire_for_rw(G, p=0.3):
    H = G.copy()
    edges = list(H.edges())
    nodes = list(H.nodes())
    m = len(edges)
    num_ops = int(p * m)

    for _ in range(num_ops):
        if not edges:
            break
        # 隨機挑一條邊拆掉
        u, v = random.choice(edges)
        if H.has_edge(u, v):
            H.remove_edge(u, v)
        # 隨機加一條新邊
        a = random.choice(nodes)
        b = random.choice(nodes)
        # 避免 self-loop & duplicate
        tries = 0
        while (a == b or H.has_edge(a, b)) and tries < 10:
            a = random.choice(nodes)
            b = random.choice(nodes)
            tries += 1
        if a != b and not H.has_edge(a, b):
            H.add_edge(a, b)
        edges = list(H.edges())
    return H

def rewire_hubs_for_rw(G, frac_hubs=0.01, rewire_ratio=0.7):
    H = G.copy()
    n = H.number_of_nodes()
    m = H.number_of_edges()
    nodes = list(H.nodes())
    
    # 1. 找最高度數的一小部分節點（也可以用 PageRank）
    degs = dict(H.degree())
    k = max(1, int(frac_hubs * n))
    hubs = sorted(degs, key=degs.get, reverse=True)[:k]

    # 2. 對這些 hub 的 incident edges 做大規模 rewiring
    for h in hubs:
        nbrs = list(H.neighbors(h))
        num_rewire = int(rewire_ratio * len(nbrs))
        for _ in range(num_rewire):
            if not nbrs:
                break
            v = random.choice(nbrs)
            if H.has_edge(h, v):
                H.remove_edge(h, v)
                nbrs.remove(v)
            # 換一條亂邊
            u = h
            w = random.choice(nodes)
            tries = 0
            while (u == w or H.has_edge(u, w)) and tries < 10:
                w = random.choice(nodes)
                tries += 1
            if u != w and not H.has_edge(u, w):
                H.add_edge(u, w)
    return H

class Sampler_G_DiT(object):
    def __init__(self, config):
        super(Sampler_G_DiT, self).__init__()

        self.config = config
        self.device = [0]

    def evaluation_ByCompound(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']
        if not "type" in self.configt:
            print("self.configt has not type setting")
            self.configt.type = self.config.type
        load_seed(self.configt.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(self.model_x, self.ckpt_dict['ema_x'], self.configt.train.ema)
            self.ema_adj = load_ema_from_ckpt(self.model_adj, self.ckpt_dict['ema_adj'], self.configt.train.ema)

            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn2 = load_sampling_fn4Di_spec(self.configt, self.config.sampler, self.config.sample, self.device)

        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)
        
        train_graph_lists, test_graph_lists = [], []
        # file_name_first = f'sampled_{self.config.scale}/motif/G0_mot'
        file_name_first = ''
        for file_name_last in ['V','R','T']:
            self.configt.data.file1 = file_name_first + file_name_last
            train_graph_list, test_graph_list = load_data_TD_test(self.configt, get_graph_list=True)
            train_graph_lists.append(train_graph_list)
            test_graph_lists.append(test_graph_list)

        num_sampling_rounds = math.ceil(len(test_graph_list) / self.configt.data.batch_size)
        gen_adj_list = []
        train_node_list = []
        # -------- Generate samples --------
        for r in range(num_sampling_rounds):
            t_start = time.time()
            self.init_flags, train_hermitian_tensor, train_nodes_array = init_flags2_wnodes_4Comp(train_graph_lists, self.configt, r)
            self.init_flags = self.init_flags.to(self.device[0])
            train_hermitian_tensor = train_hermitian_tensor.to(self.device[0])
            x, adj, _ = self.sampling_fn2(self.model_x, self.model_adj, self.init_flags, train_hermitian_tensor, self.configt.data.spec_dim, 200) #

            adj = mask_adjs(adj, self.init_flags)
            logger.log(f"Round {r} : {time.time() - t_start:.2f}s")
            print(f"Round {r} : {time.time() - t_start:.2f}s")
            adj_np = adj.cpu().numpy()
            
            if r == num_sampling_rounds-1:
                adj_np = adj_np[num_sampling_rounds*self.configt.data.batch_size - len(test_graph_list) : ]
                train_nodes_array = train_nodes_array[num_sampling_rounds*self.configt.data.batch_size - len(test_graph_list) : ]

            gen_adj_list.extend([adj_np[i] for i in range(adj_np.shape[0])])
            train_node_list.extend([set(train_nodes_array[i]) for i in range(train_nodes_array.shape[0])])
            
        assert len(gen_adj_list)==len(train_node_list)
        
        pre_dense_true = json.load(open('utils/predensity.json'))[self.config.data.data + self.config.scale]
        density_scale = getattr(self.config.sampler, "density_scale", 1.0) # 加一個 config 超參數，讓你可以在 YAML 裡指定 density_scale
        pre_dense = density_scale * pre_dense_true

        lower_lim = pre_dense + 0.15
        upper_lim = pre_dense - 0.15
        thres1 = self.config.sampler.threshold1
        thres2 = self.config.sampler.threshold2
        samples_int, final_thres = quantize_DegreeBound_4Comp(gen_adj_list, train_node_list, thres1, thres2, lower_lim, upper_lim) # lambda = final_thres
        
        # lam = self.config.sampler.lambda_fixed  # 例如從 YAML 讀進來
        # samples_int = []
        # for A, nodes in zip(gen_adj_list, train_node_list):
        #     A_bin = (A > lam).astype(int)
        #     samples_int.append(A_bin)
        # p = self.config.sampler.rw_rewire_p if hasattr(self.config.sampler, "rw_rewire_p") else 0.0
        frac = self.config.sampler.frac_hubs if hasattr(self.config.sampler, "frac_hubs") else 0.0
        rewire_p = self.config.sampler.rewire_ratio if hasattr(self.config.sampler, "rewire_ratio") else 0.0
        gen_graph_list = adjsWnodes_to_graphs(samples_int, train_node_list, True)

        if rewire_p > 0.0:
            gen_graph_list_rewired = []
            for G in gen_graph_list:
                G_rewired = rewire_hubs_for_rw(G, frac_hubs=frac, rewire_ratio=rewire_p) # for superuser
                # G_rewired = rewire_for_rw(G, p=p)  # original
                gen_graph_list_rewired.append(G_rewired)
            gen_graph_list = gen_graph_list_rewired

        assert len(gen_graph_list)==len(test_graph_list)
        # save_graph_list(os.path.join(*['Sensity', self.config.data.data]) , f"{self.config.ckpt[:]}_{pre_dense}_{str('%.2f'%final_thres)[-2:]}", gen_graph_list)
        print("Evaluation By Compund Finish")

        methods = ['degree','cluster', 'spectral', 'node_behavior_ks', 'random_walk_ks','pagerank_ks','node_degree_behavior_ks',
                   'degree_centrality_behavior_ks', 'betweenness_centrality_behavior_ks', 'closeness_centrality_behavior_ks']
        kernels = {'degree':gaussian_emd,
                    'cluster':gaussian_emd,
                    'spectral':gaussian_emd}
        whole_mot_graph_list = load_graph_list(os.path.join(self.config.data.dir, file_name_first[:-6], f'{self.config.data.file1}.pkl'))[:len(gen_graph_list)] 
        result_dict = eval_graph_list(whole_mot_graph_list, gen_graph_list, methods=methods, kernels=kernels)
        
        # Display Metric values and output images

        # 1. Output MMD and KS values
        # result_dict contains the MMD results for degree, cluster, and orbit
        print("\n" + "="*40)
        print(f"Evaluation Results (MMD) for {self.config.ckpt}:")
        print(result_dict)
        print("="*40 + "\n")

        # If the logger is still active, it can also write to the log file
        try:
            logger.log(f"Final Metrics: {result_dict}") 
        except:
            pass

        # 2. Plot and save images
        # Set the directory name for saving files (usually using the dataset name, e.g., qm9)
        save_folder_name = self.config.data.data 

        # (A) Plot Ground Truth (Real data)
        plot_graphs_list(
            whole_mot_graph_list, 
            title=f"{self.config.ckpt}_GroundTruth", 
            save_dir=save_folder_name,
            max_num=16  # Default to plot 16 images, can be adjusted
        )

        # (B) Plot Generated (Generated data)
        plot_graphs_list(
            gen_graph_list, 
            title=f"{self.config.ckpt}_Generated", 
            save_dir=save_folder_name,
            max_num=16
        )

        print(f"Graphs saved to directory: ./samples/fig/{save_folder_name}/")
