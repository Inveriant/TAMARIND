# TAMARIND

A modular Python toolkit for estimating quantum resource requirements for cryptographic attacks, with a focus on optimized windowed modular arithmetic circuits for RSA factoring.

## Overview

This project provides comprehensive cost estimation tools for quantum algorithms attacking RSA encryption, featuring optimized windowing techniques that slightly reduce resource requirements compared to previous approaches.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   # On macOS and Linux:
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Or with pip:
   pip install uv
   ```

2. **Clone and setup the project**:
   ```bash
   git clone https://github.com/Inveriant/TAMARIND.git
   cd TAMARIND
   uv sync
   ```

This will automatically create a virtual environment and install all dependencies specified in `pyproject.toml`.

## Usage

### Basic Usage

All the functionality you need is in the `main.py` file. This script provides comprehensive plotting capabilities to analyze quantum resource requirements for RSA factoring attacks.

#### Quick Start Examples
```
key_sizes = [1024, 2048, 3072, 4096]
plot_surviving_estimates(key_sizes, gate_error_rate=1e-3)
plt.show()
```


Generate optimal space time minizing plots for small key sizes
```
plot(key_size=256)
```

Generate optimal space time minizing plots for larger key sizes
```
plot()
```

#### Understanding the Outputs

The plots will show:
- **Physical Qubits vs Key Size**: How quantum hardware requirements scale
- **Runtime vs Key Size**: How execution time grows with problem size
- **Parameter Landscape**: Different algorithm configurations and their relative performance across the cost estimate space

All plots are automatically saved as high-quality PDF files in the output directory.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{luongo2025optimizedcircuitswindowedmodular,
      title={Optimized circuits for windowed modular arithmetic with applications to quantum attacks against RSA}, 
      author={Alessandro Luongo and Varun Narasimhachar and Adithya Sireesh},
      year={2025},
      eprint={2502.17325},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2502.17325}, 
}
```

## Acknowledgments

This work builds on the code and research by **Craig Gidney** work on efficient quantum factoring. The original implementation can be found at: https://github.com/Strilanc/efficient-quantum-factoring-2019

**Authors**: Alessandro Luongo, Varun Narasimhachar, and Adithya Sireesh

**Development History**: Initial development began while Adithya Sireesh was at Invariant Labs.

