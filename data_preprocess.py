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
    Load actions.csv
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

    # Sort by time (Crucial for temporal split)
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"[INFO] Loaded: {len(df)} edges, {df['source'].nunique()} unique sources, {df['target'].nunique()} unique targets")
    return df


def split_time_windows(df, num_windows=10):
    """
    Split data into K time windows
    """
    min_time = df['timestamp'].min()
    max_time = df['timestamp'].max()
    # Add a small epsilon to ensure the last timestamp falls in the last bin
    window_size = (max_time - min_time + 1e-5) / num_windows

    df['time_window'] = ((df['timestamp'] - min_time) // window_size).astype(int)
    df['time_window'] = df['time_window'].clip(0, num_windows - 1)

    print(f"[INFO] Time window split: {num_windows} windows, range [{min_time}, {max_time}]")
    return df


def create_snapshot_graph(edges_df):
    """
    Build a directed graph for a single time window.
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

def save_single_split(graph_list, file_path):
    if not graph_list:
        return
    with open(file_path, 'wb') as f:
        pickle.dump(graph_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  -> Saved {len(graph_list)} graphs to: {file_path}")

def print_statistics(graph_list, name="Graph List"):
    """
    Print statistics (Updated with Max Nodes)
    """
    if not graph_list:
        return
    
    # Calculate node counts once
    node_counts = [g.number_of_nodes() for g in graph_list]

    avg_nodes = np.mean(node_counts)
    max_nodes = np.max(node_counts) # <--- New Logic: Get Max Nodes

    avg_edges = np.mean([g.number_of_edges() for g in graph_list])
    avg_degree = np.mean([2 * g.number_of_edges() / g.number_of_nodes()
                          if g.number_of_nodes() > 0 else 0
                          for g in graph_list])
    avg_density = np.mean([nx.density(g) for g in graph_list])

    print(f"\n[Statistics] {name}:")
    print(f"  Count: {len(graph_list)}")
    print(f"  Avg nodes: {avg_nodes:.1f}")
    print(f"  Max nodes: {max_nodes}") # <--- Print Max Nodes
    print(f"  Avg edges: {avg_edges:.1f}")
    print(f"  Avg degree: {avg_degree:.2f}")
    print(f"  Avg density: {avg_density:.6f}")

def process_dataset_from_csv(csv_path, output_dir, num_windows=10, dataset_name=None):
    """
    Process directly from CSV.
    Splits data into Train (80%), Val (10%), Test (10%) based on time windows order.
    """
    df = load_csv(csv_path)
    df = split_time_windows(df, num_windows)

    if dataset_name is None:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        if dataset_name == 'actions':
            parent_dir = os.path.basename(os.path.dirname(csv_path))
            if parent_dir != 'raw':
                dataset_name = parent_dir

    graph_list = []
    stats = {'total_windows': len(df['time_window'].unique()), 'empty_graphs': 0, 'passed': 0}

    # Generate graphs in temporal order
    for window_id in sorted(df['time_window'].unique()):
        window_df = df[df['time_window'] == window_id]
        G = create_snapshot_graph(window_df)
        
        if G.number_of_nodes() == 0:
            stats['empty_graphs'] += 1
            continue

        graph_list.append(G)
        stats['passed'] += 1

    if not graph_list:
        print("\n[WARNING] No graphs generated")
        return None

    # --- Save the FULL dataset before splitting (No Suffix) ---
    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, f"{dataset_name}_{num_windows}bins_modiff.pkl")
    print(f"\n[INFO] Saving FULL dataset (before split)...")
    save_single_split(graph_list, full_path)
    print_statistics(graph_list, "Full Dataset (All Windows)")
    # -------------------------------------------------------------

    # --- Splitting Logic (80% Train, 10% Val, 10% Test) ---
    total_graphs = len(graph_list)
    val_size = int(total_graphs * 0.1)
    test_size = int(total_graphs * 0.1)
    train_size = total_graphs - val_size - test_size # Use remainder for train

    train_graphs = graph_list[:train_size]
    val_graphs = graph_list[train_size:train_size + val_size]
    test_graphs = graph_list[train_size + val_size:]

    # --- Save Split Files ---
    
    # 1. Train set
    train_path = os.path.join(output_dir, f"{dataset_name}_{num_windows}bins_modiffR.pkl")
    save_single_split(train_graphs, train_path)
    # Extra save: R.pkl
    save_single_split(train_graphs, os.path.join(output_dir, "R.pkl"))
    
    # 2. Validation set
    val_path = os.path.join(output_dir, f"{dataset_name}_{num_windows}bins_modiffV.pkl")
    save_single_split(val_graphs, val_path)
    # Extra save: V.pkl
    save_single_split(val_graphs, os.path.join(output_dir, "V.pkl"))
    
    # 3. Test set
    test_path = os.path.join(output_dir, f"{dataset_name}_{num_windows}bins_modiffT.pkl")
    save_single_split(test_graphs, test_path)
    # Extra save: T.pkl
    save_single_split(test_graphs, os.path.join(output_dir, "T.pkl"))

    print(f"\n{'='*60}")
    print(f"[SUCCESS] Data processing and splitting complete.")
    print_statistics(train_graphs, "Train Set (R / R.pkl)")
    print_statistics(val_graphs, "Validation Set (V / V.pkl)")
    print_statistics(test_graphs, "Test Set (T / T.pkl)")
    print(f"{'='*60}")
    
    return graph_list


def process_dataset_from_muldydiff(dataset_path, output_dir, num_bins=None):
    """
    Convert datasets processed by MulDyDiff.
    Directly maps 'train' -> .pkl, 'val' -> V.pkl, 'test' -> T.pkl
    """
    dataset_name = os.path.basename(dataset_path.rstrip('/'))

    if num_bins is None:
        processed_dir = os.path.join(dataset_path, 'processed')
        if os.path.exists(processed_dir):
            pt_files = [f for f in os.listdir(processed_dir) if f.endswith('_train.pt')]
            if pt_files:
                import re
                match = re.search(r'(\d+)_bins', pt_files[0])
                if match:
                    num_bins = int(match.group(1))
                    print(f"[INFO] Detected num_bins from filename: {num_bins}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = f"{dataset_name}_{num_bins}bins_modiff" if num_bins else f"{dataset_name}_modiff"

    # Process and Save each split independently
    
    # 1. Train
    train_graphs = load_muldydiff_processed(dataset_path, split='train')
    if train_graphs:
        save_single_split(train_graphs, os.path.join(output_dir, f"{base_name}.pkl"))
        print_statistics(train_graphs, "Train Set")

    # 2. Validation
    val_graphs = load_muldydiff_processed(dataset_path, split='val')
    if val_graphs:
        save_single_split(val_graphs, os.path.join(output_dir, f"{base_name}V.pkl"))
        print_statistics(val_graphs, "Validation Set")

    # 3. Test
    test_graphs = load_muldydiff_processed(dataset_path, split='test')
    if test_graphs:
        save_single_split(test_graphs, os.path.join(output_dir, f"{base_name}T.pkl"))
        print_statistics(test_graphs, "Test Set")

    print("\n[DONE] MulDyDiff data conversion complete.")

def main():
    parser = argparse.ArgumentParser(description='Convert temporal graphs to MoDiff format')
    parser.add_argument('--dataset-path', required=True,
                        help='Path to dataset (CSV file or MulDyDiff data directory)')
    parser.add_argument('--output-dir', default='./data/Twitter',
                        help='Output directory')
    parser.add_argument('--num-bins', type=int, default=None,
                        help='Number of time bins (for CSV processing or override MulDyDiff detection)')
    parser.add_argument('--dataset-name', type=str, default=None,
                        help='Dataset name (optional, will be inferred if not provided)')
    parser.add_argument('--from-muldydiff', action='store_true',
                        help='Load from MulDyDiff processed data instead of CSV')

    args = parser.parse_args()

    if args.from_muldydiff:
        print(f"[INFO] Processing from MulDyDiff data: {args.dataset_path}")
        process_dataset_from_muldydiff(args.dataset_path, args.output_dir, args.num_bins)
    else:
        print(f"[INFO] Processing from CSV: {args.dataset_path}")
        num_bins = args.num_bins if args.num_bins else 10
        process_dataset_from_csv(args.dataset_path, args.output_dir, num_bins, args.dataset_name)

if __name__ == '__main__':
    main()
