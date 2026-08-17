# 🧬 AlloGNN: Multi-Modal Graph Neural Network with Residue-Level Gated Fusion

Official PyTorch implementation of **"ALLOGNN: A Multi-Modal Graph Neural Network with Residue-Level Gated Fusion for Allosteric Binding Site Prediction in Human Protein Kinases"**.

AlloGNN is a multi-modal deep learning framework designed to predict both active and highly elusive allosteric binding sites. By employing a learned residue-level gating mechanism, AlloGNN overcomes the "allosteric blind spot" of sequence-only models caused by neutral mutational frustration, achieving an **11-fold improvement** in detecting distal allosteric pockets.



## 🧬 Architecture Overview

<p align="center">
  <img src="figures/architecture.png" width="800" alt="AlloGNN Architecture Diagram">
</p>

- **Sequence Stream (Evolutionary Context):** Extracts deep, coevolutionary patterns from the frozen **ESM2-650M** protein language model.
- **Structure Stream (Spatial Topology):** Propagates sequence embeddings over an $SE(3)$-invariant C$\alpha$ contact graph via **Graph Attention Convolution (GATConv)**, identifying distinct topological neighborhoods.
- **Surface Stream (Local Biophysical Profiling):** Encodes physical constraints—including relative SASA, crystallographic B-factor, and secondary structure—into a localized physical proxy.
- **Residue-Level Gated Fusion:** A dynamic softmax gating network that independently weights the three modalities per residue. It natively learns to rely on sequence and surface features when local structural topologies are highly plastic (neutrally frustrated).

## ⚙️ Installation

We recommend using Conda to reproduce the exact software environment.

```bash
# Clone the repository
git clone https://github.com/srajalcodes/AlloGNN.git
cd AlloGNN

# Create and activate the environment
conda env create -f environment.yml
conda activate allognn
```

## 📂 Download Data & Pre-Trained Weights

To ensure complete reproducibility and avoid computational overhead, our fully precomputed PyTorch Geometric graph dataset (**9,217 chains**), pre-extracted ESM2 embeddings, and model checkpoints are hosted on Zenodo.

Run the automated download script:

```bash
python download_data.py
```

This script will automatically download the 2GB archive and initialize the `data/`, `output/`, and `results/` directories required by the model.

## 🚀 Training & Evaluation

### 1. Evaluate the Pre-trained Model

To evaluate the model on the unseen gene-cold test split (**812 chains**) and reproduce Table II and Table III from the paper:

```bash
python evaluation.py
```

### 2. Reproduce the Ablation Study

To run the targeted ablations on the development subset (Table V), demonstrating the necessity of the multi-modal architecture:

```bash
python run_ablations.py
```

### 3. Train the Full Model

To run the full end-to-end pipeline (data preprocessing → graph construction → model training):

```bash
python train.py
```

## 📊 Main Results

### Head-to-Head Comparative Performance (Unseen Gene-Cold Test Set)

| Target Site Class | LBS-pLM AUPR | P2Rank AUPR | AlloGNN AUPR (Ours) |
|---|---:|---:|---:|
| Kinase Type I (Orthosteric) | 0.629 | 0.485 | **0.754** |
| Kinase Type I.5 (Back-pocket) | 0.749 | 0.578 | **0.878** |
| Kinase Type II (DFG-out) | 0.680 | 0.613 | **0.856** |
| Kinase Type III (Proximal Allosteric) | 0.363 | 0.559 | **0.940** |
| Kinase Type IV (Distal Allosteric) | 0.077 | 0.375 | **0.853** |

> **Note:** AlloGNN achieves an **11.08-fold relative AUPR improvement** over the sequence-only baseline on Type IV distal allosteric sites.

## 📜 Citation

If you find this repository or our dataset splits useful, please cite:

```bibtex
@inproceedings{allognn2026,
  title={ALLOGNN: A Multi-Modal Graph Neural Network with Residue-Level Gated Fusion for Allosteric Binding Site Prediction in Human Protein Kinases},
  author={Anonymous Authors},
  booktitle={Proceedings of the 26th IEEE International Conference on Bioinformatics and Bioengineering (BIBE)},
  year={2026},
  organization={IEEE}
}
```

## 📄 License

This project is licensed under the MIT License.