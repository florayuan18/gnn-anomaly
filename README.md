# GNN Anomaly Detection for Particle Physics

Unsupervised graph neural network autoencoder for anomaly detection on high-energy physics events. Converts tabular particle data (56 features per event) into graphs and trains autoencoders to detect signal events via reconstruction error.

## Setup

```bash
pip install -r requirements.txt
```

Requires: PyTorch, torch-geometric, h5py, numpy, scikit-learn, matplotlib

## Data

- Background: `background_for_training.h5` (unsupervised training)
- Signals: `Ato4l_lepFilter_13TeV_filtered.h5`, `hToTauTau_13TeV_PU20_filtered.h5`, `hChToTauNu_13TeV_PU20_filtered.h5`, `leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5`
- Given that the signal files are too large, we did not upload the .h5 files. They can be found and downloaded here: https://mpp-hep.github.io/ADC2021/

Each event → graph with 56 nodes (one per feature), edges constructed via chosen method.

## Training

```bash
python gnn_training.py --background <bg_file.h5> [options]
```

**Options:**
- `--encoder-type {GCN,GraphSAGE,GIN,GAT}` (default: GCN)
- `--edge-method {chain,deltaR,energy,knn}` (default: chain)
- `--sage-agg {mean,max,lstm}` (GraphSAGE only, default: mean)
- `--variational` (use VGAE)
- `--deltaR-threshold <float>` (deltaR edges, default: 0.5)
- `--energy-threshold <float>` (energy edges, default: 0.1)
- `--knn-k <int>` (kNN edges, default: 5)
- `--hidden-dim <int>` (default: 32)
- `--latent-dim <int>` (default: 16)
- `--layers <int>` (default: 3)
- `--epochs <int>` (default: 10)
- `--batch-size <int>` (default: 64)
- `--lr <float>` (default: 1e-3)
- `--max-background <int>` (default: 4000000)
- `--model-name <str>` (default: auto-generated)
- `--output-dir <str>` (default: outputs/models)

**Examples:**
```bash
# GCN with chain edges
python gnn_training.py --background background_for_training.h5 --encoder-type GCN --edge-method chain

# GraphSAGE with deltaR edges
python gnn_training.py --background background_for_training.h5 --encoder-type GraphSAGE --sage-agg mean --edge-method deltaR --deltaR-threshold 0.5

# GAT with kNN edges
python gnn_training.py --background background_for_training.h5 --encoder-type GAT --edge-method knn --knn-k 5

# VGAE variant
python gnn_training.py --background background_for_training.h5 --encoder-type GCN --variational --edge-method chain
```

Models auto-save as `{encoder}_{agg}_{edge}_{params}_h{hidden}_l{latent}.pt` (e.g., `gcn_chain_h32_l16.pt`, `sage_mean_deltaR_th0.5_h32_l16.pt`)

## Evaluation

```bash
python gnn_eval.py --model <model.pt> --background <bg_file.h5> --signals <sig1.h5> <sig2.h5> ... [options] --out <roc.png>
```

**Options:** Same as training (must match training config exactly)
- `--encoder-type`, `--edge-method`, `--sage-agg`, `--variational`, edge params, `--hidden-dim`, `--latent-dim`, `--layers` must match training
- `--batch-size <int>` (default: 128)
- `--max-background <int>` (default: 200000)
- `--max-signal-per-file <int>` (default: None)
- `--out <path>` (default: outputs/roc_gnn.png)

**Examples:**
```bash
# Evaluate GCN chain model
python gnn_eval.py --model outputs/models/gcn_chain_h32_l16.pt \
  --background background_for_training.h5 \
  --signals Ato4l_lepFilter_13TeV_filtered.h5 hToTauTau_13TeV_PU20_filtered.h5 hChToTauNu_13TeV_PU20_filtered.h5 leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5 \
  --encoder-type GCN --edge-method chain --hidden-dim 32 --latent-dim 16 --layers 3 \
  --out outputs/roc_gcn.png

# Evaluate GraphSAGE deltaR model
python gnn_eval.py --model outputs/models/sage_mean_deltaR_th0.5_h32_l16.pt \
  --background background_for_training.h5 \
  --signals Ato4l_lepFilter_13TeV_filtered.h5 hToTauTau_13TeV_PU20_filtered.h5 hChToTauNu_13TeV_PU20_filtered.h5 leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5 \
  --encoder-type GraphSAGE --sage-agg mean --edge-method deltaR --deltaR-threshold 0.5 \
  --hidden-dim 32 --latent-dim 16 --layers 3 --out outputs/roc_sage.png
```

## Architecture

- **Encoder**: GCN/GraphSAGE/GIN/GAT layers → graph-level latent representation
- **Decoder**: Reconstructs node features from latent
- **Anomaly score**: Reconstruction error (MSE)
- **Edge methods**: Chain (sequential), deltaR (angular distance), energy (pt correlation), kNN (feature space)

## All Variants

**Train:** `python gnn_training.py --background <bg_file.h5> [options below]`  
**Eval:** `python gnn_eval.py --model <model.pt> --background <bg_file.h5> --signals <sig1.h5> <sig2.h5> ... [same options as training] --out <roc.png>`

| Variant | Training/Evaluation Options |
|---------|---------------------------|
| **GCN + chain** | `--encoder-type GCN --edge-method chain` |
| **GraphSAGE + chain** | `--encoder-type GraphSAGE --sage-agg mean --edge-method chain` |
| **GIN + chain** | `--encoder-type GIN --edge-method chain` |
| **GAT + chain** | `--encoder-type GAT --edge-method chain` |
| **GCN + deltaR** | `--encoder-type GCN --edge-method deltaR --deltaR-threshold 0.5` |
| **GCN + energy** | `--encoder-type GCN --edge-method energy --energy-threshold 0.1` |
| **GCN + kNN** | `--encoder-type GCN --edge-method knn --knn-k 5` |
| **GraphSAGE + deltaR** | `--encoder-type GraphSAGE --sage-agg mean --edge-method deltaR --deltaR-threshold 0.5` |
| **GraphSAGE + energy** | `--encoder-type GraphSAGE --sage-agg mean --edge-method energy --energy-threshold 0.1` |
| **GraphSAGE + kNN** | `--encoder-type GraphSAGE --sage-agg mean --edge-method knn --knn-k 5` |
| **GIN + deltaR** | `--encoder-type GIN --edge-method deltaR --deltaR-threshold 0.5` |
| **GAT + deltaR** | `--encoder-type GAT --edge-method deltaR --deltaR-threshold 0.5` |
| **GAT + kNN** | `--encoder-type GAT --edge-method knn --knn-k 5` |
| **GCN + VGAE + chain** | `--encoder-type GCN --variational --edge-method chain` |
| **GCN + VGAE + deltaR** | `--encoder-type GCN --variational --edge-method deltaR --deltaR-threshold 0.5` |

All variants use default: `--hidden-dim 32 --latent-dim 16 --layers 3` (specify if different). For GraphSAGE, can use `--sage-agg {mean,max,lstm}`.
