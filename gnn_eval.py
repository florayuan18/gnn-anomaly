#!/usr/bin/env python3
import os
import argparse
from typing import List

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

from gnn_data import load_background_and_signals, H5GraphDataset, load_h5_matrix
from gnn_model import GCNAnomaly


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate GNN anomaly detector and plot ROC")
    p.add_argument("--model", type=str, required=True, help="Path to .pt weights")
    p.add_argument("--background", type=str, required=True, help="Path to background H5")
    p.add_argument("--signals", type=str, nargs="+", required=True, help="Paths to signal H5 files")
    p.add_argument("--signal-labels", type=str, nargs="+", default=None,
                   help="Labels for signal files (default: derived from filenames)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--max-background", type=int, default=200000)
    p.add_argument("--max-signal-per-file", type=int, default=None)
    p.add_argument("--out", type=str, default="outputs/roc_gnn.png")
    # architecture options
    p.add_argument("--encoder-type", type=str, default="GCN", 
                   choices=["GCN", "GraphSAGE", "GIN", "GAT"],
                   help="Type of encoder: GCN, GraphSAGE, GIN, or GAT")
    p.add_argument("--sage-agg", type=str, default="mean", choices=["mean", "max", "lstm"],
                   help="GraphSAGE aggregation method")
    p.add_argument("--variational", action="store_true", 
                   help="Use Variational Graph Autoencoder (VGAE)")
    # edge construction options
    p.add_argument("--edge-method", type=str, default="chain",
                   choices=["chain", "deltaR", "energy", "knn"],
                   help="Method for constructing edges")
    p.add_argument("--deltaR-threshold", type=float, default=0.5,
                   help="ΔR threshold for deltaR edge method")
    p.add_argument("--energy-threshold", type=float, default=0.1,
                   help="Energy correlation threshold for energy edge method")
    p.add_argument("--knn-k", type=int, default=5,
                   help="Number of neighbors for knn edge method")
    return p.parse_args()


def collect_scores(model: GCNAnomaly, loader: DataLoader, device: torch.device) -> np.ndarray:
    """collect reconstruction error scores (unsupervised anomaly detection)."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # handle both standard and VGAE models
            if model.variational:
                x_recon, _, _, _ = model(batch)
            else:
                x_recon, _ = model(batch)
            
            # reconstruction error per node: MSE between original and reconstructed
            recon_error = torch.nn.functional.mse_loss(x_recon, batch.x, reduction='none')
            # recon_error shape: [num_nodes, 1]
            recon_error = recon_error.squeeze(-1) # [num_nodes]
            
            # compute mean error per graph
            if hasattr(batch, 'batch') and batch.batch is not None:
                # Batched graphs: group by graph index
                num_graphs = batch.num_graphs
                graph_scores = []
                for i in range(num_graphs):
                    mask = (batch.batch == i)
                    graph_error = recon_error[mask].mean().item()
                    graph_scores.append(graph_error)
                scores.append(np.array(graph_scores))
            else:
                # Single graph: mean over all nodes
                graph_error = recon_error.mean().item()
                scores.append(np.array([graph_error]))
    return np.concatenate(scores, axis=0)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # generate signal labels from filenames if not provided
    if args.signal_labels is None:
        signal_labels = []
        for sig_path in args.signals:
            filename = os.path.basename(sig_path)
            label = os.path.splitext(filename)[0]
            # some clean up
            label = label.replace("_13TeV_filtered", "").replace("_lepFilter", "").replace("_PU20", "")
            signal_labels.append(label)
    else:
        signal_labels = list(args.signal_labels)
    
    if len(signal_labels) != len(args.signals):
        raise ValueError(f"Number of signal labels ({len(signal_labels)}) must match number of signal files ({len(args.signals)})")

    # model
    print(f"loading model: {args.model}...")
    print(f"  encoder type: {args.encoder_type}")
    print(f"  variational: {args.variational}")
    model = GCNAnomaly(
        in_channels=1, 
        hidden_channels=args.hidden_dim, 
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        encoder_type=args.encoder_type,
        sage_agg=args.sage_agg,
        variational=args.variational
    ).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    
    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model loaded successfully!")
    print(f"  total parameters: {total_params:,}")
    print(f"  trainable parameters: {trainable_params:,}")

    # load background data
    print(f"\nLoading background data...")
    bg_mat, _ = load_background_and_signals(
        args.background,
        [],
        background_key="Particles",
        signal_key="Particles",
        max_background=args.max_background,
        max_signal_per_file=None
    )
    
    # edge kwargs
    edge_kwargs = {}
    if args.edge_method == "deltaR":
        edge_kwargs["threshold"] = args.deltaR_threshold
    elif args.edge_method == "energy":
        edge_kwargs["threshold"] = args.energy_threshold
    elif args.edge_method == "knn":
        edge_kwargs["k"] = args.knn_k
    
    # compute background scores
    bg_dataset = H5GraphDataset([bg_mat], [0], 
                                edge_method=args.edge_method,
                                edge_kwargs=edge_kwargs)
    bg_loader = DataLoader(bg_dataset, batch_size=args.batch_size, shuffle=False)
    bg_scores = collect_scores(model, bg_loader, device)
    print(f"background scores: {len(bg_scores)} samples")

    # signal data and compute scores
    print(f"\nloading signal data...")
    signal_scores_dict = {}
    
    for signal_file, signal_label in zip(args.signals, signal_labels):
        print(f"\nprocessing {signal_label}...")
        # load signal data, not background
        sig_mat = load_h5_matrix(
            signal_file,
            dataset_key="Particles",
            max_samples=args.max_signal_per_file
        )
        sig_dataset = H5GraphDataset([sig_mat], [1],
                                    edge_method=args.edge_method,
                                    edge_kwargs=edge_kwargs)
        sig_loader = DataLoader(sig_dataset, batch_size=args.batch_size, shuffle=False)
        sig_scores = collect_scores(model, sig_loader, device)
        signal_scores_dict[signal_label] = sig_scores
        print(f"{signal_label} scores: {len(sig_scores)} samples")

    # compute roc for each signal separately
    print("ROC AUC Results:")
    print(f"{'Signal':<30} {'AUC':<10} {'TPR@FPR=1e-5':<15} {'Samples':<10}")
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.figure(figsize=(8, 8))
    
    target_fpr = 1e-5
    
    # plot roc curve for each signal
    for label in signal_labels:
        sig_scores = signal_scores_dict[label]
        
        # combine background and this signal
        y_true = np.concatenate([np.zeros_like(bg_scores), np.ones_like(sig_scores)], axis=0)
        y_scores = np.concatenate([bg_scores, sig_scores], axis=0)
        
        # roc
        auc = roc_auc_score(y_true, y_scores)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        
        # compute tpr at fpr = 1e-5
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_target = tpr[idx] if idx < len(tpr) else 0.0
        
        plt.plot(fpr, tpr, label=f'{label} (AUC = {auc:.4f})', linewidth=2)
        
        print(f"{label:<30} {auc:<10.5f} {tpr_at_target*100:<15.6f}% {len(sig_scores):<10}")
    
    # plot diagonal line
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random")
    
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("GNN Anomaly Detection ROC Curves", fontsize=14)
    
    # add log scale and vertical line at FPR = 1e-5
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-7, 1)
    plt.ylim(1e-7, 1)
    plt.axvline(x=target_fpr, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'FPR = {target_fpr}')
    
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    
    print(f"\nTPR at FPR = {target_fpr} for each signal:")
    for label in signal_labels:
        sig_scores = signal_scores_dict[label]
        y_true = np.concatenate([np.zeros_like(bg_scores), np.ones_like(sig_scores)], axis=0)
        y_scores = np.concatenate([bg_scores, sig_scores], axis=0)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_target = tpr[idx] if idx < len(tpr) else 0.0
        print(f"{label}: {tpr_at_target*100:.6f}%")
    print(f"\nROC curves saved to {args.out}")


if __name__ == "__main__":
    main()


