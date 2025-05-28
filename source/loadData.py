import gzip
import json
import torch
from torch_geometric.data import Dataset, Data
import os
from tqdm import tqdm 
from torch_geometric.loader import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, filename, transform=None, pre_transform=None):
        self.raw = filename
        self.graphs = self.loadGraphs(self.raw)
        super().__init__(None, transform, pre_transform)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

    @staticmethod
    def loadGraphs(path):
        print(f"Loading graphs from {path}...")
        print("This may take a few minutes, please wait...")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            graphs_dicts = json.load(f)
        graphs = []
        for graph_dict in tqdm(graphs_dicts, desc="Processing graphs", unit="graph"):
            graphs.append(dictToGraphObject(graph_dict))
        return graphs
    
    def analyze(self):
        num_nodes_list = []
        num_edges_list = []
        label_counter = Counter()

        for graph in self.graphs:
            num_nodes_list.append(graph.num_nodes)
            num_edges_list.append(graph.edge_index.size(1))
            if graph.y is not None:
                label_counter[int(graph.y)] += 1

        print("=== Dataset Analysis ===")
        print(f"Total graphs: {len(self.graphs)}")
        print(f"Average number of nodes per graph: {np.mean(num_nodes_list):.2f}")
        print(f"Average number of edges per graph: {np.mean(num_edges_list):.2f}")
        print(f"Label distribution: {label_counter}")
        print(f"Unique labels: {sorted(label_counter.keys())}")

        return {
            "num_graphs": len(self.graphs),
            "avg_nodes": np.mean(num_nodes_list),
            "avg_edges": np.mean(num_edges_list),
            "label_distribution": dict(label_counter),
            "unique_labels": list(sorted(label_counter.keys()))
        }

    def visualize_graph(self, idx=0):
        data = self.get(idx)
        G = nx.Graph()
        edge_index = data.edge_index.numpy()

        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            G.add_edge(src, dst)

        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, node_color="skyblue", edge_color="gray")
        plt.title(f"Graph #{idx} - Label: {data.y} - Node: {data.num_nodes}, Edge: {data.edge_index.shape[1]}")
        plt.show()



def dictToGraphObject(graph_dict):
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float) if graph_dict["edge_attr"] else None
    num_nodes = graph_dict["num_nodes"]
    y = torch.tensor(graph_dict["y"][0], dtype=torch.long) if graph_dict["y"] is not None else None
    return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes, y=y)
