#!/usr/bin/env python3
import os
import argparse
from typing import List

import numpy as np
import torch
from torch.utils.data import random_split, Subset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from gnn_data import load_background_and_signals, H5GraphDataset
from gnn_model import GCNAnomaly


def parse_args() -> argparse.Namespace:
    """parse command line args for training."""
    p = argparse.ArgumentParser("Train a GNN autoencoder (unsupervised) on H5 tabular data")
    p.add_argument("--background", type=str, required=True, help="Path to background H5")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--latent-dim", type=int, default=16, help="Latent dimension for autoencoder")
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-background", type=int, default=4000000, 
                   help="Total background samples to load (first 3.8M for train/val, last 200K for test)")
    p.add_argument("--output-dir", type=str, default="outputs/models")
    p.add_argument("--model-name", type=str, default="gnn_anomaly.pt")
    # architecture options
    p.add_argument("--encoder-type", type=str, default="GCN", 
                   choices=["GCN", "GraphSAGE", "GIN", "GAT"],
                   help="Type of encoder: GCN, GraphSAGE, GIN, or GAT")
    p.add_argument("--sage-agg", type=str, default="mean", choices=["mean", "max", "lstm"],
                   help="GraphSAGE aggregation method")
    p.add_argument("--variational", action="store_true", 
                   help="Use Variational Graph Autoencoder (VGAE)")
    # edge options
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


def main():
    """train gnn autoencoder on background data, save best model."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load background only
    print("\nloading background data (unsupervised training)...")
    bg_mat, _ = load_background_and_signals(
        args.background,
        [],
        background_key="Particles",
        signal_key="Particles",
        max_background=args.max_background,
        max_signal_per_file=None
    )

    # dataset: background only
    print("\ncreating graph dataset...")
    print(f"creating dataset from background data only...")
    print(f"edge method: {args.edge_method}")
    
    # edge kwargs
    edge_kwargs = {}
    if args.edge_method == "deltaR":
        edge_kwargs["threshold"] = args.deltaR_threshold
    elif args.edge_method == "energy":
        edge_kwargs["threshold"] = args.energy_threshold
    elif args.edge_method == "knn":
        edge_kwargs["k"] = args.knn_k
    
    dataset = H5GraphDataset([bg_mat], [0], 
                             edge_method=args.edge_method,
                             edge_kwargs=edge_kwargs)
    print(f"dataset created: {len(dataset)} total graphs")

    # split train/val/test
    print("\nsplitting into train/val/test sets...")
    n_total = len(dataset)
    
    # first 3.8M train/val (90/10 train/val), last 200K test
    n_train_val = 3800000
    n_test = 200000
    
    if n_total < n_train_val + n_test:
        print(f"warning: dataset has {n_total} samples, but need {n_train_val + n_test} for train/val/test split")
        print(f"using all available samples. adjusting split...")
        n_train_val = n_total - n_test if n_total > n_test else int(n_total * 0.95)
        n_test = n_total - n_train_val
    
    # split 90/10 train/val
    n_train = int(n_train_val * 0.9)
    n_val = n_train_val - n_train
    
    # create split indices
    train_indices = list(range(n_train))
    val_indices = list(range(n_train, n_train_val))
    test_indices = list(range(n_total - n_test, n_total))
    
    print(f"  total samples: {n_total}")
    print(f"  train samples: {len(train_indices)} (first {n_train} from training set)")
    print(f"  val samples: {len(val_indices)} (next {n_val} from training set)")
    print(f"  test samples: {len(test_indices)} (last {n_test} samples)")
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    print("\ncreating data loaders...")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    print(f"  train batches: {len(train_loader)}")
    print(f"  val batches: {len(val_loader)}")
    print(f"  test batches: {len(test_loader)}")

    # model
    print("\ninitializing model...")
    print(f"  device: {device}")
    print(f"  encoder type: {args.encoder_type}")
    print(f"  variational: {args.variational}")
    print(f"  hidden dim: {args.hidden_dim}")
    print(f"  latent dim: {args.latent_dim}")
    print(f"  layers: {args.layers}")
    print(f"  learning rate: {args.lr}")
    if args.encoder_type == "GraphSAGE":
        print(f"  graphsage aggregation: {args.sage_agg}")
    
    model = GCNAnomaly(
        in_channels=1, 
        hidden_channels=args.hidden_dim, 
        latent_dim=args.latent_dim,
        num_layers=args.layers,
        encoder_type=args.encoder_type,
        sage_agg=args.sage_agg,
        variational=args.variational
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # loss: mse for standard, elbo for vgae
    criterion_recon = torch.nn.MSELoss() # used for both
    if not args.variational:
        criterion = criterion_recon # alias

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    
    # auto-generate model name
    if args.model_name == "gnn_anomaly.pt": # default name
        name_parts = []
        name_parts.append(args.encoder_type.lower())
        if args.encoder_type == "GraphSAGE":
            name_parts.append(args.sage_agg)
        name_parts.append(args.edge_method)
        if args.edge_method == "deltaR":
            name_parts.append(f"th{args.deltaR_threshold}")
        elif args.edge_method == "energy":
            name_parts.append(f"th{args.energy_threshold}")
        elif args.edge_method == "knn":
            name_parts.append(f"k{args.knn_k}")
        if args.variational:
            name_parts.append("vgae")
        name_parts.append(f"h{args.hidden_dim}_l{args.latent_dim}")
        auto_name = "_".join(name_parts) + ".pt"
        best_path = os.path.join(args.output_dir, auto_name)
        print(f"  auto-generated model name: {auto_name}")
    else:
        best_path = os.path.join(args.output_dir, args.model_name)
    
    print("\nstarting training...")

    # epoch loop
    epoch_pbar = tqdm(range(1, args.epochs + 1), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        # train
        model.train()
        total_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", 
                         leave=False, position=1)
        for batch in train_pbar:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            if args.variational:
                x_recon, _, z_mean, z_logvar = model(batch)
                # vgae loss: recon + kl
                recon_loss = criterion_recon(x_recon, batch.x)
                # kl divergence
                kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
                kl_loss = kl_loss.mean()
                loss = recon_loss + 0.001 * kl_loss  # weight kl
            else:
                x_recon, _ = model(batch)
                # recon loss
                loss = criterion(x_recon, batch.x)
            
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch.num_graphs
            
            # update progress bar
            current_loss = loss.item()
            train_pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        train_loss = total_loss / max(1, len(train_loader.dataset))

        # validate
        model.eval()
        total_v = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", 
                       leave=False, position=1)
        with torch.no_grad():
            for batch in val_pbar:
                batch = batch.to(device)
                
                if args.variational:
                    x_recon, _, z_mean, z_logvar = model(batch)
                    recon_loss = criterion_recon(x_recon, batch.x)
                    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
                    kl_loss = kl_loss.mean()
                    loss = recon_loss + 0.001 * kl_loss
                else:
                    x_recon, _ = model(batch)
                    # recon loss
                    loss = criterion(x_recon, batch.x)
                
                total_v += float(loss.item()) * batch.num_graphs
                
                # update progress bar
                current_loss = loss.item()
                val_pbar.set_postfix({"loss": f"{current_loss:.4f}"})
        
        val_loss = total_v / max(1, len(val_loader.dataset))

        # test eval
        model.eval()
        total_test = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                
                if args.variational:
                    x_recon, _, z_mean, z_logvar = model(batch)
                    recon_loss = criterion_recon(x_recon, batch.x)
                    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
                    kl_loss = kl_loss.mean()
                    loss = recon_loss + 0.001 * kl_loss
                else:
                    x_recon, _ = model(batch)
                    loss = criterion(x_recon, batch.x)
                
                total_test += float(loss.item()) * batch.num_graphs
        
        test_loss = total_test / max(1, len(test_loader.dataset))

        # update epoch bar
        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "test_loss": f"{test_loss:.4f}",
            "best_val": f"{best_val:.4f}"
        })

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            epoch_pbar.write(f"✓ Saved best model (val_loss={val_loss:.4f}, test_loss={test_loss:.4f}) to {best_path}")

    # final save
    if args.model_name == "gnn_anomaly.pt":
        final_path = best_path.replace(".pt", "_final.pt")
    else:
        final_path = os.path.join(args.output_dir, args.model_name.replace(".pt", "_final.pt"))
    torch.save(model.state_dict(), final_path)
    print(f"saved final model to {final_path}")


if __name__ == "__main__":
    main()


