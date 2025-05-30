import argparse
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

import sys
sys.path.append('source')

from loadData import GraphDataset
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm, global_max_pool, global_add_pool
from torch.utils.data import random_split
import gc

import conv
import logging
import kaggle_models as km

from tqdm import tqdm

def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)  
    return data

class NoisyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, p_noisy=0.2):
        super().__init__()
        self.p = p_noisy
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets):
        losses = self.ce(logits, targets)
        weights = (1 - self.p) + self.p * (1 - torch.nn.functional.one_hot(targets, num_classes=logits.size(1)).float().sum(dim=1))
        return (losses * weights).mean()

class SimpleGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.embedding = torch.nn.Embedding(1, input_dim) 
        self.conv1 = GCNConv(input_dim, hidden_dim,  add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.conv3 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.conv4 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.conv5 = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        self.global_pool = global_mean_pool  
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)  
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.global_pool(x, batch)  
        out = self.fc(x)  
        return out


def train(data_loader, save_checkpoints, checkpoint_path, current_epoch, dataset='A'):
    model.train()
    total_loss = 0
    for data in tqdm(data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Save checkpoints if required
    if save_checkpoints:
        checkpoint_file = f"{checkpoint_path}_epoch_{current_epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_file)
        print(f"Checkpoint saved at {checkpoint_file}")
        
    return total_loss / len(data_loader)


def evaluate(data_loader, calculate_accuracy=False):
    model.eval()
    correct = 0
    total = 0
    predictions = []
    with torch.no_grad():
        for data in tqdm(data_loader):
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.cpu().numpy())
            if calculate_accuracy:
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
    if calculate_accuracy:
        accuracy = correct / total
        return accuracy, predictions
    return predictions


def main(args):
    global model, optimizer, criterion, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    script_dir = os.getcwd() # r"C:\Users\fede6\Desktop\DeepHW"

    # Parameters for the GCN model
    input_dim = 300  # Example input feature dimension (you can adjust this)
    hidden_dim = 150
    output_dim = 6  # Number of classes

    dataset = args.test_path.split("\\")[-2]
    
    if dataset == 'A':
        print(f"Loading models for ensemble on dataset {dataset}...", end=" ")
        # Initialize the model, optimizer, and loss criterion
        model1 = km.GNN(gnn_type='gin', num_class=6, num_layer=5, emb_dim=150, drop_ratio=0.5, virtual_node=True, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_A_best_1.pth")
        checkpoint = torch.load(checkpoint_fn)
        model1.load_state_dict(checkpoint['model_state_dict'])
        
        model2 = km.GNN(gnn_type='gin', num_class=6, num_layer=5, emb_dim=150, drop_ratio=0.5, virtual_node=True, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_A_best_2.pth")
        checkpoint = torch.load(checkpoint_fn)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        model = km.GNNEnsemble([model1, model2])
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "model_A_best.pth")
        checkpoint = torch.load(checkpoint_fn)
        model.load_state_dict(checkpoint['model_state_dict']) 
        print("Done") 
    elif dataset == 'B':
        print(f"Loading models for ensemble on dataset {dataset}...", end=" ")
        # Initialize the model, optimizer, and loss criterion
        model1 = km.GNN(gnn_type='gin', num_class=6, num_layer=5, emb_dim=150, drop_ratio=0.5, virtual_node=True, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_B_best_1.pth")
        checkpoint = torch.load(checkpoint_fn)
        model1.load_state_dict(checkpoint['model_state_dict'])
        
        model2 = conv.GNN(gnn_type='gine', num_class=6, num_layer=5, emb_dim=128, drop_ratio=0.5, virtual_node=False, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_B_best_2.pth")
        checkpoint = torch.load(checkpoint_fn)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        model = km.GNNEnsemble([model1, model2])
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "model_B_best.pth")
        checkpoint = torch.load(checkpoint_fn)
        model.load_state_dict(checkpoint['model_state_dict']) 
        print("Done")       
    elif dataset == 'C':
        print(f"Loading models for ensemble on dataset {dataset}...", end=" ")
        # Initialize the model, optimizer, and loss criterion
        model1 = km.GNN(gnn_type='gin', num_class=6, num_layer=5, emb_dim=150, drop_ratio=0.5, virtual_node=True, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_C_best_1.pth")
        checkpoint = torch.load(checkpoint_fn)
        model1.load_state_dict(checkpoint['model_state_dict'])
        
        model2 = conv.GNN(gnn_type='gine', num_class=6, num_layer=5, emb_dim=128, drop_ratio=0.5, virtual_node=False, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_C_best_2.pth")
        checkpoint = torch.load(checkpoint_fn)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        model = km.GNNEnsemble([model1, model2])
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "model_C_best.pth")
        checkpoint = torch.load(checkpoint_fn)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Done")
    elif dataset == 'D':
        print(f"Loading models for ensemble on dataset {dataset}...", end=" ")
        # Initialize the model, optimizer, and loss criterion
        model1 = km.GNN(gnn_type='gin', num_class=6, num_layer=5, emb_dim=150, drop_ratio=0.5, virtual_node=True, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_D_best_1.pth")
        checkpoint = torch.load(checkpoint_fn)
        model1.load_state_dict(checkpoint['model_state_dict'])
        
        model2 = conv.GNN(gnn_type='gine', num_class=6, num_layer=5, emb_dim=128, drop_ratio=0.5, virtual_node=False, residual=True, graph_pooling='attention').to(device)
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "ensembled_models", "model_D_best_2.pth")
        checkpoint = torch.load(checkpoint_fn)
        model2.load_state_dict(checkpoint['model_state_dict'])
        
        model = km.GNNEnsemble([model1, model2])
        checkpoint_fn = os.path.join(script_dir, "checkpoints", "model_D_best.pth")
        checkpoint = torch.load(checkpoint_fn)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Done")
    else:
        raise ValueError("Invalid dataset!")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-10)
    criterion = NoisyCrossEntropyLoss() # torch.nn.CrossEntropyLoss()

    # Prepare test dataset and loader
    test_dataset = GraphDataset(args.test_path, transform=add_zeros)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train dataset and loader (if train_path is provided)
    if args.train_path:
        train_dataset = GraphDataset(args.train_path, transform=add_zeros)
        
        val_size = int(0.2 * len(train_dataset))
        train_size = len(train_dataset) - val_size

        generator = torch.Generator().manual_seed(12)
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=generator)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        script_dir = os.getcwd()
        test_dir_name = os.path.basename(os.path.dirname(args.test_path))
        logs_folder = os.path.join(script_dir, "logs", test_dir_name)
        log_file = os.path.join(logs_folder, "training.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
        # logging.getLogger().addHandler(logging.StreamHandler())

        checkpoint_path = os.path.join(script_dir, "checkpoints", f"A\model_{test_dir_name}")
        checkpoints_folder = os.path.join(script_dir, "checkpoints", test_dir_name)
        os.makedirs(checkpoints_folder, exist_ok=True)

        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device, save_checkpoints=True, checkpoint_path=checkpoint_path, current_epoch=epoch)
            val_loss, val_acc, val_f1 = evaluate(val_loader, calculate_accuracy=False)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}| Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
            gc.collect()

    # Evaluate and save test predictions
    predictions = evaluate(test_loader, calculate_accuracy=False)
    test_graph_ids = list(range(len(predictions)))  # Generate IDs for graphs

    # Save predictions to CSV
    test_dir_name = os.path.dirname(args.test_path).split(os.sep)[-1]
    output_csv_path = os.path.join(f"testset_{test_dir_name}.csv")
    output_df = pd.DataFrame({
        "id": test_graph_ids,
        "pred": predictions
    })
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test predictions saved to {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a GCN model on graph datasets.")
    parser.add_argument("--train_path", type=str, default=None, help="Path to the training dataset (optional).")
    parser.add_argument("--test_path", type=str, required=True, help="Path to the test dataset.")
    args = parser.parse_args()
    main(args)
