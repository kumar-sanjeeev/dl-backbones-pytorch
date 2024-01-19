# DL-Backbones-Pytorch

This repository contains a PyTorch implementation of widely used backbones in deep learning literature.

## Table of Contents
- [Project Structure Overview](#project-structure-overview)
- [Overview](#overview)

## Project Structure Overview
```
.
├── configs                                           <-- Hydra Configs
│   ├── network                                            <- network module configs
│   │   └── upsample_resnet.yaml
│   └── resnet                                             <- resnet module configs
│       ├── resnet101.yaml
│       ├── resnet152.yaml
│       ├── resnet18.yaml
│       ├── resnet34.yaml
│       └── resnet50.yaml
├── examples                                          <-- Examples to showcase how to use `backbone` and `network` module
│   ├── demo_resnet18.py
│   └── demo_upsample_resnet18.py
└── src                                           
|    ├── backbones                                    <-- Module: backbone 
|    │   └── resnet                                        <- resnet implementation
|    └── networks                                     <-- Module: networks
|        ├── __init__.py
|        └── upsample_resnet.py
├── README.md
├── requirements.txt

```

### Overview:

- **Configs:** Hydra configuration files are organized for both the `network` and `resnet` modules.

- **Examples:** Demonstrative scripts (for e.g. `demo_resnet18.py` and `demo_upsample_resnet18.py`) to guide users on how to utilize the `backbone` and `network` modules.

- **Source Code:**
  - `backbones` module contains the implementation of the different backbones
  - `networks` module contains network implementation using different backbones.

