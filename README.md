# Deep-MM: REVISITING OPTIMAL MIMO SECURE BEAMFORMING:A LOW-COMPLEXITY GNN-ENABLED DEEP UNFOLDING METHOD

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository provides the release implementation and a pretrained checkpoint
for the paper:

> **REVISITING OPTIMAL MIMO SECURE BEAMFORMING:A LOW-COMPLEXITY GNN-ENABLED DEEP UNFOLDING METHOD**  
> Miao Jiang, Ruijie Huang, and Yiqing Li

## Overview

Deep-MM is a GNN-guided fixed-depth unfolding method for transmit-covariance
design in the fixed-antenna, ideal-CSI MIMO wiretap channel. Each unfolded
layer predicts a positive control parameter, while the covariance update and
power projection are computed deterministically inside the unfolded model.

The release also contains the original iterative MM method as a reference
implementation.

## Methods

| Component | Description | Location |
| --- | --- | --- |
| Deep-MM | GNN-guided fixed-depth unfolding with deterministic covariance updates | [`deep_mm/models/deep_mm.py`](deep_mm/models/deep_mm.py) |
| Original MM | Original iterative MM reference method | [`deep_mm/solvers/mm.py`](deep_mm/solvers/mm.py) |

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── checkpoints/
│   └── deep_mm_nt4_nr4_ne2_L06.pt
├── deep_mm/
│   ├── common/
│   │   ├── channel.py       # Channel generation and validation
│   │   ├── config.py        # Fixed-channel experiment configuration
│   │   ├── data.py          # Reproducible channel-corpus utilities
│   │   └── metrics.py       # Secrecy-rate and covariance metrics
│   ├── models/
│   │   ├── checkpoints.py   # Checkpoint serialization and validation
│   │   └── deep_mm.py       # Deep-MM model and unfolded update
│   └── solvers/
│       └── mm.py            # Original MM method
└── scripts/
    ├── infer.py             # Checkpoint inference example
    └── train_deep_mm.py     # Deep-MM training entry point
```

## Requirements

The release has been validated with Python 3.8.20, NumPy 1.23.5, and PyTorch
2.4.0. Install the pinned runtime dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Quick start

Run inference with the released checkpoint from the repository root:

```bash
python scripts/infer.py --checkpoint checkpoints/deep_mm_nt4_nr4_ne2_L06.pt
```

The script generates one seeded channel realization and reports the secrecy
rate, covariance feasibility diagnostics, and the layerwise control values.

## Pretrained checkpoint

`checkpoints/deep_mm_nt4_nr4_ne2_L06.pt` contains the Deep-MM model trained
for the `(N_t, N_r, N_e) = (4, 4, 2)` setting. The model is trained with a
16-layer unfolding depth and uses 6 layers as the default inference depth;
the inference depth can be selected when calling the model directly.

## Citation

If you use this repository, please cite:

```text
M. Jiang, R. Huang, and Y. Li,
"GNN-Enabled Deep Unfolding for Optimal MIMO Secure Beamforming,"
submitted to ICASSP 2027.
```

## License

This project is released under the [MIT License](LICENSE).
