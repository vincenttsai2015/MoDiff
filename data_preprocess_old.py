import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
from datetime import datetime
from collections import defaultdict
import argparse
import time
import torch
from torch_geometric.utils import to_networkx

def load_csv(csv_path):
    """
    Load actions.csv (original logic preserved)
    """
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = {'source', 'target', 'timestamp'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}, current columns: {df.columns.tolist()}")
    
    # Convert timestamp to numeric
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
    
    # Ensure source/target are integers
    df['source'] = df['source'].astype(int)
    df['target'] = df['target'].astype(int)

    # Filtering logic
    valid_max_timestamp = 1800000000  # Approximately year 2027
    
    initial_len = len(df)
    df = df[df['timestamp'] < valid_max_timestamp]
    filtered_len = len(df)
    
    if initial_len != filtered_len:
        print(f"[WARNING] Filtered out {initial_len - filtered_len} rows with future timestamps (> 2027).")

    # Sort by time
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"[INFO] Loaded: {len(df)} edges, {df['source'].nunique()} unique sources, {df['target'].nunique()} unique targets")
    return df


def split_time_windows(df, num_windows=10):
    """
    Split data into K time windows
    """
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    window_size = (max_time - min_time) / num_windows
    
    df['time_window'] = ((df['timestamp'] - min_time) // window_size).astype(int)
    df['time_window'] = df['time_window'].clip(0, num_windows - 1)
    
    print(f"[INFO] Time window split: {num_windows} windows, range [{min_time}, {max_time}]")
    return df


def create_snapshot_graph(edges_df):
    """
    Build a directed graph for a single time window.
    Reference: MulDyDiff but simplified to single-layer
    """
    G = nx.DiGraph()
    
    # Add edges
    for _, row in edges_df.iterrows():
        src, tgt = int(row['source']), int(row['target'])
        
        if G.has_edge(src, tgt):
            # If edge exists, accumulate weight
            G[src][tgt]['weight'] = G[src][tgt].get('weight', 0) + 1
        else:
            G.add_edge(src, tgt, weight=1)
    
    # Add node features (simple degree-based)
    for node in G.nodes():
        G.nodes[node]['feature'] = np.array([G.degree(node)], dtype=np.float32)
    
    return G


def load_muldydiff_processed(dataset_path, split='train'):
    """
    Load data already processed by MulDyDiff
    """
    processed_dir = os.path.join(dataset_path, 'processed')
    
    if not os.path.exists(processed_dir):
        print(f"[WARNING] Processed directory not found: {processed_dir}")
        return None
    
    # Find corresponding .pt file (support various naming formats)
    pt_files = [f for f in os.listdir(processed_dir) 
                if f.endswith(f'_{split}.pt') and 'temporal' in f]
    
    if not pt_files:
        print(f"[WARNING] No processed files found for split '{split}' in {processed_dir}")
        return None
    
    # Use the first found file
    pt_file = sorted(pt_files)[0]
    pt_path = os.path.join(processed_dir, pt_file)
    
    print(f"[INFO] Loading MulDyDiff processed data: {pt_path}")
    
    try:
        # PyTorch 2.6+ requires weights_only=False
        pyg_sequences = torch.load(pt_path, weights_only=False)
    except Exception as e:
        print(f"[ERROR] Failed to load {pt_path}: {e}")
        return None
    
    # Convert PyG sequences into NetworkX graphs
    graph_list = []
    for seq_idx, seq in enumerate(pyg_sequences):
        for snap_idx, pyg_data in enumerate(seq):
            G = pyg_to_networkx_directed(pyg_data)
            if G.number_of_nodes() > 0:
                graph_list.append(G)
        
        if (seq_idx + 1) % 10 == 0:
            print(f'  Processed {seq_idx + 1}/{len(pyg_sequences)} sequences')
    
    print(f"[INFO] Loaded {len(graph_list)} graphs from {split} split")
    return graph_list


def pyg_to_networkx_directed(data):
    """
    Convert PyG Data to a NetworkX DiGraph.
    Support both HeteroData and regular Data
    """
    # Check if HeteroData
    if hasattr(data, 'metadata'):
        # HeteroData - needs special handling
        G = nx.DiGraph()
        
        # Iterate over all node types
        node_offset = 0
        node_mapping = {}
        
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            for i in range(num_nodes):
                global_id = node_offset + i
                node_mapping[(node_type, i)] = global_id
                
                # Add node features
                if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                    if data[node_type].x.dim() > 1:
                        G.add_node(global_id, feature=data[node_type].x[i].numpy())
                    else:
                        G.add_node(global_id, feature=np.array([data[node_type].x[i].item()], dtype=np.float32))
                else:
                    G.add_node(global_id, feature=np.zeros(1, dtype=np.float32))
            
            node_offset += num_nodes
        
        # Iterate over all edge types
        for edge_type in data.edge_types:
            src_type, rel, dst_type = edge_type
            edge_index = data[edge_type].edge_index.numpy()
            
            for i in range(edge_index.shape[1]):
                src_local = int(edge_index[0, i])
                dst_local = int(edge_index[1, i])
                
                src = node_mapping.get((src_type, src_local))
                dst = node_mapping.get((dst_type, dst_local))
                
                if src is not None and dst is not None:
                    if hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None:
                        weight = float(data[edge_type].edge_attr[i].sum().item())
                    else:
                        weight = 1.0
                    
                    if G.has_edge(src, dst):
                        G[src][dst]['weight'] += weight
                    else:
                        G.add_edge(src, dst, weight=weight)
        
        # Update degree features
        for node in G.nodes():
            if np.sum(G.nodes[node]['feature']) == 0:
                G.nodes[node]['feature'] = np.array([G.degree(node)], dtype=np.float32)
        
        return G
    
    else:
        # Regular Data
        G = nx.DiGraph()
        
        # Get number of nodes
        if hasattr(data, 'x') and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = data.edge_index.max().item() + 1
        
        # Add nodes and features
        for i in range(num_nodes):
            node_attrs = {}
            if hasattr(data, 'x') and data.x is not None:
                if data.x.dim() > 1:
                    node_attrs['feature'] = data.x[i].numpy()
                else:
                    node_attrs['feature'] = np.array([data.x[i].item()], dtype=np.float32)
            else:
                node_attrs['feature'] = np.zeros(1, dtype=np.float32)
            
            G.add_node(i, **node_attrs)
        
        # Add edges
        if hasattr(data, 'edge_index'):
            edge_index = data.edge_index.numpy()
            
            for i in range(edge_index.shape[1]):
                src, tgt = int(edge_index[0, i]), int(edge_index[1, i])
                edge_attrs = {}
                
                # Edge attributes
                if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                    if data.edge_attr.dim() > 1:
                        edge_attrs['weight'] = float(data.edge_attr[i].sum().item())
                    else:
                        edge_attrs['weight'] = float(data.edge_attr[i].item())
                else:
                    edge_attrs['weight'] = 1.0
                
                # Accumulate weight for duplicate edges
                if G.has_edge(src, tgt):
                    G[src][tgt]['weight'] += edge_attrs['weight']
                else:
                    G.add_edge(src, tgt, **edge_attrs)
        
        # Update node features to degree if no meaningful feature exists
        for node in G.nodes():
            if 'feature' not in G.nodes[node] or np.sum(G.nodes[node]['feature']) == 0:
                G.nodes[node]['feature'] = np.array([G.degree(node)], dtype=np.float32)
        
        return G


def print_statistics(graph_list, output_file):
    """
    Print statistics
    """
    avg_nodes = np.mean([g.number_of_nodes() for g in graph_list])
    avg_edges = np.mean([g.number_of_edges() for g in graph_list])
    avg_degree = np.mean([2 * g.number_of_edges() / g.number_of_nodes() 
                          if g.number_of_nodes() > 0 else 0 
                          for g in graph_list])
    avg_density = np.mean([nx.density(g) for g in graph_list])
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Saved {len(graph_list)} graphs to: {output_file}")
    print(f"{'='*60}")
    print("Statistics:")
    print(f"  Total graphs: {len(graph_list)}")
    print(f"  Avg nodes: {avg_nodes:.1f}")
    print(f"  Avg edges: {avg_edges:.1f}")
    print(f"  Avg degree: {avg_degree:.2f}")
    print(f"  Avg density: {avg_density:.6f}")


def process_dataset_from_csv(csv_path, output_dir, num_windows=10, dataset_name=None):
    """
    Process directly from CSV (for datasets not processed by MulDyDiff)
    """
    df = load_csv(csv_path)
    df = split_time_windows(df, num_windows)
    
    # Determine dataset name
    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        # Remove 'actions' if present
        if dataset_name == 'actions':
            parent_dir = os.path.basename(os.path.dirname(csv_path))
            if parent_dir != 'raw':
                dataset_name = parent_dir
    
    graph_list = []
    stats = {
        'total_windows': len(df['time_window'].unique()),
        'empty_graphs': 0,
        'passed': 0
    }
    
    for window_id in sorted(df['time_window'].unique()):
        print(f"\n--- Window {window_id} ---")
        window_df = df[df['time_window'] == window_id]
        print(f"  Raw data: {len(window_df)} edges")
        
        G = create_snapshot_graph(window_df)
        print(f"  Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        if G.number_of_nodes() == 0:
            stats['empty_graphs'] += 1
            print(f"  [Skipped] Empty graph")
            continue
        
        graph_list.append(G)
        stats['passed'] += 1
    
    # Save results with naming: {dataset_name}_{num_bins}bins_modiff.pkl
    if len(graph_list) > 0:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{dataset_name}_{num_windows}bins_modiff.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(graph_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compute statistics
        avg_nodes = np.mean([g.number_of_nodes() for g in graph_list])
        avg_edges = np.mean([g.number_of_edges() for g in graph_list])
        avg_degree = np.mean([2 * g.number_of_edges() / g.number_of_nodes() for g in graph_list])
        avg_density = np.mean([nx.density(g) for g in graph_list])
        
        print(f"\n{'='*60}")
        print(f"[SUCCESS] Saved {len(graph_list)} graphs to: {output_file}")
        print(f"{'='*60}")
        print("Statistics:")
        print(f"  Total time windows: {stats['total_windows']}")
        print(f"  Passed: {stats['passed']} ({stats['passed']/stats['total_windows']*100:.1f}%)")
        print(f"  Empty graphs: {stats['empty_graphs']}")
        print("  ---")
        print(f"  Avg nodes: {avg_nodes:.1f}")
        print(f"  Avg edges: {avg_edges:.1f}")
        print(f"  Avg degree: {avg_degree:.2f}")
        print(f"  Avg density: {avg_density:.4f}")
        
        return graph_list
    else:
        print("\n[WARNING] No graphs generated")
        return None


def process_dataset_from_muldydiff(dataset_path, output_dir, num_bins=None):
    """
    Convert datasets processed by MulDyDiff
    """
    dataset_name = os.path.basename(dataset_path.rstrip('/'))
    
    # Try to extract num_bins from processed file name if not provided
    if num_bins is None:
        processed_dir = os.path.join(dataset_path, 'processed')
        if os.path.exists(processed_dir):
            pt_files = [f for f in os.listdir(processed_dir) if f.endswith('_train.pt')]
            if pt_files:
                # Extract num_bins from filename like "pyg_temporal_200000_bins_..."
                import re
                match = re.search(r'(\d+)_bins', pt_files[0])
                if match:
                    num_bins = int(match.group(1))
                    print(f"[INFO] Detected num_bins from filename: {num_bins}")
    
    # Process train, val, test
    all_graphs = []
    for split in ['train', 'val', 'test']:
        graphs = load_muldydiff_processed(dataset_path, split=split)
        if graphs:
            all_graphs.extend(graphs)
            print(f"[INFO] {split}: {len(graphs)} graphs")
    
    if not all_graphs:
        print(f"[ERROR] No graphs loaded from {dataset_path}")
        return None
    
    # Save results with naming: {dataset_name}_{num_bins}bins_modiff.pkl
    os.makedirs(output_dir, exist_ok=True)
    if num_bins:
        output_file = os.path.join(output_dir, f"{dataset_name}_{num_bins}bins_modiff.pkl")
    else:
        output_file = os.path.join(output_dir, f"{dataset_name}_modiff.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Statistics
    print_statistics(all_graphs, output_file)
    return all_graphs


def main():
    parser = argparse.ArgumentParser(description='Convert temporal graphs to MoDiff format')
    parser.add_argument('--dataset-path', required=True, 
                        help='Path to dataset (CSV file or MulDyDiff data directory)')
    parser.add_argument('--output-dir', default='./data/modiff_processed', 
                        help='Output directory')
    parser.add_argument('--num-bins', type=int, default=None, 
                        help='Number of time bins (for CSV processing or override MulDyDiff detection)')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Dataset name (optional, will be inferred if not provided)')
    parser.add_argument('--from-muldydiff', action='store_true',
                        help='Load from MulDyDiff processed data instead of CSV')
    
    args = parser.parse_args()
    
    if args.from_muldydiff:
        # Convert from MulDyDiff processed data
        print(f"[INFO] Processing from MulDyDiff data: {args.dataset_path}")
        process_dataset_from_muldydiff(args.dataset_path, args.output_dir, args.num_bins)
    else:
        # Process directly from CSV
        print(f"[INFO] Processing from CSV: {args.dataset_path}")
        # Use default num_bins=10 if not specified
        num_bins = args.num_bins if args.num_bins else 10
        process_dataset_from_csv(args.dataset_path, args.output_dir, num_bins, args.dataset_name)
    
    print("\n[DONE] Data preparation complete!")


if __name__ == '__main__':
    main()
