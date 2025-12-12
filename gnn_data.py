#!/usr/bin/env python3
"""
Convert H5 tabular data to PyTorch Geometric graphs such that we have one graph per event, 56 nodes per graph. 
Edge methods: chain, deltaR, energy, knn.
"""
import os
import h5py
import numpy as np
from typing import Iterable, List, Tuple, Optional, Literal
from sklearn.neighbors import kneighbors_graph

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def process_particles_3d_to_2d(data_chunk: np.ndarray) -> np.ndarray:
    """
    transform the 3d particles data [N, 19, 4] to 2D [N, 56] format.
    same transformation as LazyH5Array._process_chunk.
    """
    n_samples = data_chunk.shape[0]
    print(f"  Converting 3d to 2d: {data_chunk.shape} -> ({n_samples}, 56)...")
    output = np.zeros((n_samples, 56), dtype=np.float32)

    # met features (0-1)
    output[:, 0] = data_chunk[:, 0, 0] # pt
    output[:, 1] = data_chunk[:, 0, 2] # phi

    # electron features (2-13)
    for e in range(4):
        output[:, 2+e] = data_chunk[:, 1+e, 0] # e pt
        output[:, 6+e] = data_chunk[:, 1+e, 1] # e eta
        output[:, 10+e] = data_chunk[:, 1+e, 2] # e phi

    # muon features (14-25)
    for m in range(4):
        output[:, 14+m] = data_chunk[:, 5+m, 0] # mu pt
        output[:, 18+m] = data_chunk[:, 5+m, 1] # mu eta
        output[:, 22+m] = data_chunk[:, 5+m, 2] # mu phi

    # jet features (26-55)
    for j in range(10):
        output[:, 26+j] = data_chunk[:, 9+j, 0] # jet pt
        output[:, 36+j] = data_chunk[:, 9+j, 1] # jet eta
        output[:, 46+j] = data_chunk[:, 9+j, 2] # jet phi

    print(f"  conversion has been complete")
    return output


def load_h5_matrix(path: str, dataset_key: Optional[str] = None, max_samples: Optional[int] = None) -> np.ndarray:
    """
    load a 2D dataset from an downloaded H5 file from ml4jets2021.
    - If dataset_key is None, defaults to "particles" (3D) -> converted to 2D.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"H5 file not found: {path}")
    
    # this defaults to "Particles" if not specified
    if dataset_key is None:
        dataset_key = "Particles"
    
    print(f"loading H5 file: {path}")
    print(f"  dataset key: {dataset_key}")
    
    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"dataset key '{dataset_key}' not found in {path}")
        
        dataset = f[dataset_key]
        print(f"  Raw shape: {dataset.shape}")
        print(f"  loading!")
        candidate = dataset[:]
        print(f"  loaded shape: {candidate.shape}")
        
        # if 3D Particles, convert this to 2D
        if candidate.ndim == 3 and candidate.shape[1] == 19 and candidate.shape[2] == 4:
            candidate = process_particles_3d_to_2d(candidate)
        elif candidate.ndim != 2:
            raise ValueError(f"data shape is not supported: {candidate.shape}.")
    
    if max_samples is not None:
        candidate = candidate[:max_samples]
    
    print(f"  final shape: {candidate.shape}")
    return candidate


def extract_particle_coords(feature_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    get (eta, phi) from the 56-feature vector. returns arrays of length 56, NaN where not available.
    """
    eta = np.full(56, np.nan, dtype=np.float32)
    phi = np.full(56, np.nan, dtype=np.float32)
    
    # met: only phi (1)
    phi[0] = feature_vec[1] if feature_vec[1] != 0 else np.nan
    
    # electrons -> 6-9 (eta), 10-13 (phi)
    for i in range(4):
        eta[2+i] = feature_vec[6+i] if feature_vec[6+i] != 0 else np.nan
        phi[2+i] = feature_vec[10+i] if feature_vec[10+i] != 0 else np.nan
    
    # muons: 18-21 (eta), 22-25 (phi)
    for i in range(4):
        eta[14+i] = feature_vec[18+i] if feature_vec[18+i] != 0 else np.nan
        phi[14+i] = feature_vec[22+i] if feature_vec[22+i] != 0 else np.nan
    
    # jets: 36-45 (eta), 46-55 (phi)
    for i in range(10):
        eta[26+i] = feature_vec[36+i] if feature_vec[36+i] != 0 else np.nan
        phi[26+i] = feature_vec[46+i] if feature_vec[46+i] != 0 else np.nan
    
    return eta, phi


def build_chain_edges(num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    simple chain: connect node i to i+1, undirected. returns edge_index, edge_attr is None.
    """
    if num_nodes < 2:
        return np.zeros((2, 0), dtype=np.int64), None
    src = np.arange(0, num_nodes - 1, dtype=np.int64)
    dst = np.arange(1, num_nodes, dtype=np.int64)
    # undirected: add both directions
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    return edge_index, None


def build_deltaR_edges(feature_vec: np.ndarray, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    connect nodes if angular distance ΔR = sqrt(Δη² + Δφ²) < threshold. edge_attr is the ΔR value.
    """
    eta, phi = extract_particle_coords(feature_vec)
    
    # find valid particles (non-NaN coordinates)
    valid_mask = ~(np.isnan(eta) | np.isnan(phi))
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        return np.zeros((2, 0), dtype=np.int64), None
    
    # compute ΔR for all pairs
    src_list = []
    dst_list = []
    edge_attr_list = []
    
    for i, idx_i in enumerate(valid_indices):
        for j, idx_j in enumerate(valid_indices):
            if i < j:
                deta = eta[idx_i] - eta[idx_j]
                dphi = phi[idx_i] - phi[idx_j]
                # handle phi wraparound
                if dphi > np.pi:
                    dphi -= 2 * np.pi
                elif dphi < -np.pi:
                    dphi += 2 * np.pi
                deltaR = np.sqrt(deta**2 + dphi**2)
                
                if deltaR < threshold:
                    src_list.extend([idx_i, idx_j])
                    dst_list.extend([idx_j, idx_i])
                    edge_attr_list.extend([deltaR, deltaR])
    
    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64), None
    
    edge_index = np.stack([np.array(src_list, dtype=np.int64), 
                          np.array(dst_list, dtype=np.int64)], axis=0)
    edge_attr = np.array(edge_attr_list, dtype=np.float32)
    
    return edge_index, edge_attr


def build_energy_edges(feature_vec: np.ndarray, threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    connect nodes if normalized pt correlation (pt_i * pt_j) > threshold. edge_attr is the correlation value.
    """
    # extract pt values
    pt_values = np.zeros(56, dtype=np.float32)
    pt_values[0] = feature_vec[0] # met pt
    pt_values[2:6] = feature_vec[2:6] # electron pt
    pt_values[14:18] = feature_vec[14:18] # muon pt
    pt_values[26:36] = feature_vec[26:36] # jet pt
    
    # normalization
    pt_max = np.max(pt_values)
    if pt_max > 0:
        pt_normalized = pt_values / pt_max
    else:
        pt_normalized = pt_values
    
    # find valid particles
    valid_mask = pt_normalized > 0
    valid_indices = np.where(valid_mask)[0]
    
    if len(valid_indices) < 2:
        return np.zeros((2, 0), dtype=np.int64), None
    
    # compute correlation
    src_list = []
    dst_list = []
    edge_attr_list = []
    
    for i, idx_i in enumerate(valid_indices):
        for j, idx_j in enumerate(valid_indices):
            if i < j:
                correlation = pt_normalized[idx_i] * pt_normalized[idx_j]
                if correlation > threshold:
                    src_list.extend([idx_i, idx_j])
                    dst_list.extend([idx_j, idx_i])
                    edge_attr_list.extend([correlation, correlation])
    
    if len(src_list) == 0:
        return np.zeros((2, 0), dtype=np.int64), None
    
    edge_index = np.stack([np.array(src_list, dtype=np.int64), 
                          np.array(dst_list, dtype=np.int64)], axis=0)
    edge_attr = np.array(edge_attr_list, dtype=np.float32)
    
    return edge_index, edge_attr


def build_knn_edges(feature_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-nearest neighbors in feature space. edge_attr is the distance between features.
    """
    # reshape to [56, 1] for kNN
    X = feature_vec.reshape(-1, 1)
    
    # build kNN graph
    knn_graph = kneighbors_graph(X, n_neighbors=k, mode='connectivity', include_self=False)
    
    # convert to edge_index format
    coo = knn_graph.tocoo()
    src = coo.row.astype(np.int64)
    dst = coo.col.astype(np.int64)
    
    # undirected
    edge_index = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0)
    
    # compute distances as edge attributes
    distances = np.abs(feature_vec[src] - feature_vec[dst])
    edge_attr = np.concatenate([distances, distances])
    
    return edge_index, edge_attr


class H5GraphDataset(Dataset):
    """
    one graph per event: x=[56,1], edge_index, edge_attr (optional), y=label (0/1).
    """
    def __init__(
        self, 
        matrices: List[np.ndarray], 
        labels: List[int],
        edge_method: Literal["chain", "deltaR", "energy", "knn"] = "chain",
        edge_kwargs: Optional[dict] = None
    ):
        """
        matrices: list of [N_i, 56] arrays. labels: 0 or 1 per matrix.
        edge_method: chain, deltaR, energy, or knn.
        edge_kwargs: e.g. {"threshold": 0.5} for deltaR/energy, {"k": 5} for knn.
        """
        assert len(matrices) == len(labels), "matrices and labels must be same length"
        self.samples: List[Tuple[np.ndarray, int]] = []
        for mat, lab in zip(matrices, labels):
            for i in range(mat.shape[0]):
                self.samples.append((mat[i], lab))
        self.num_nodes = self.samples[0][0].shape[0]
        self.edge_method = edge_method
        self.edge_kwargs = edge_kwargs or {}
        
        # pre-compute edges if method is chain (same for all graphs)
        if edge_method == "chain":
            self.edge_index_np, _ = build_chain_edges(self.num_nodes)
            self.edge_index = torch.from_numpy(self.edge_index_np).long()
            self.use_fixed_edges = True
        else:
            self.use_fixed_edges = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        vec, lab = self.samples[idx]
        x = torch.from_numpy(vec.astype(np.float32)).view(-1, 1) # [56, 1]
        y = torch.tensor([lab], dtype=torch.float32) # shape [1]
        
        # build edges based on method
        if self.use_fixed_edges:
            edge_index = self.edge_index
            edge_attr = None
        else:
            if self.edge_method == "deltaR":
                threshold = self.edge_kwargs.get("threshold", 0.5)
                edge_index_np, edge_attr_np = build_deltaR_edges(vec, threshold)
            elif self.edge_method == "energy":
                threshold = self.edge_kwargs.get("threshold", 0.1)
                edge_index_np, edge_attr_np = build_energy_edges(vec, threshold)
            elif self.edge_method == "knn":
                k = self.edge_kwargs.get("k", 5)
                edge_index_np, edge_attr_np = build_knn_edges(vec, k)
            else:
                raise ValueError(f"Unknown edge_method: {self.edge_method}")
            
            # handle empty edge_index if no edges found
            if edge_index_np.shape[1] == 0:
                # create empty edge_index with correct shape
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                # create empty tensor for consistency
                edge_attr = torch.zeros((0,), dtype=torch.float32)
            else:
                edge_index = torch.from_numpy(edge_index_np).long()
                if edge_attr_np is not None and len(edge_attr_np) > 0:
                    edge_attr = torch.from_numpy(edge_attr_np).float()
                else:
                    # Create empty tensor for consistency
                    edge_attr = torch.zeros((edge_index.shape[1],), dtype=torch.float32)
        
        # for edge methods that support edge_attr, always include it even if empty
        if self.use_fixed_edges:
            return Data(x=x, edge_index=edge_index, y=y)
        else:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def load_background_and_signals(
    background_path: str,
    signal_paths: Iterable[str],
    background_key: Optional[str] = None,
    signal_key: Optional[str] = None,
    max_background: Optional[int] = None,
    max_signal_per_file: Optional[int] = None
) -> Tuple[np.ndarray, List[np.ndarray]]:
    print("loading background data...")
    bg = load_h5_matrix(background_path, dataset_key=background_key, max_samples=max_background)
    print(f"Background loaded: {bg.shape}\n")
    
    sigs = []
    for i, sp in enumerate(signal_paths, 1):
        print(f"Loading signal file {i}/{len(signal_paths)}: {os.path.basename(sp)}")
        sig = load_h5_matrix(sp, dataset_key=signal_key, max_samples=max_signal_per_file)
        print(f"Signal {i} loaded: {sig.shape}\n")
        sigs.append(sig)
    
    print("Data loading complete!")
    print(f"  Background: {bg.shape}")
    for i, sig in enumerate(sigs, 1):
        print(f"  Signal {i}: {sig.shape}")
    return bg, sigs