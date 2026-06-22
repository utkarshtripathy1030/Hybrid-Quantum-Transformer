# Hybrid Quantum Transformer

A research-oriented implementation of a **Hybrid Quantum Self-Attention Layer** for small-scale language modeling. This project combines classical Transformer architectures with quantum circuits to explore whether quantum-enhanced attention mechanisms can provide meaningful representations for sequence modeling tasks.

## Overview

Traditional Transformers rely entirely on classical computation. This project replaces part of the self-attention mechanism with a parameterized quantum circuit implemented using PennyLane, creating a hybrid quantum-classical architecture.

The goal is to investigate:

- Quantum-enhanced feature representations
- Hybrid quantum-classical attention mechanisms
- Feasibility of quantum Transformers on NISQ devices
- Small-scale language modeling with sequence lengths of 64–256 tokens

---

## Features

- Hybrid Quantum Self-Attention Layer
- PennyLane + PyTorch integration
- Parameterized quantum circuits (4-qubit)
- Classical embedding and feed-forward layers
- Transformer-inspired architecture
- Simulation on classical hardware
- Research-oriented implementation

---

## Architecture

```
Input Tokens
      ↓
Embedding Layer
      ↓
Hybrid Quantum Self-Attention
      ↓
Feed Forward Network
      ↓
Output Layer
```

Quantum Layer:

```
Classical Features
        ↓
Angle Encoding
        ↓
Parameterized Quantum Circuit
        ↓
Measurement
        ↓
Attention Representation
```

---

## Technologies Used

- Python
- PyTorch
- PennyLane
- NumPy
- Matplotlib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer.git
cd Hybrid-Quantum-Transformer
```

Install dependencies:

```bash
pip install torch pennylane numpy matplotlib
```

---

## Usage

Run:

```bash
python main.py
```

or

```bash
python train.py
```

(depending on the project structure)

---

## Project Structure

```
Hybrid-Quantum-Transformer/
│
├── data/
├── models/
├── quantum_layers/
├── notebooks/
├── train.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Research Objective

Design and simulate a hybrid quantum self-attention layer for small-scale language modeling with sequence lengths ranging from **64 to 256 tokens**, and evaluate its effectiveness compared to classical attention mechanisms.

---

## Future Work

- Multi-head quantum attention
- Quantum positional encodings
- Larger sequence lengths
- Benchmarking against classical Transformers
- Execution on real quantum hardware
- Latency and scalability analysis

---

## Author

**Utkarsh Tripathy**

- GitHub: https://github.com/utkarshtripathy1030

---

## License

MIT License

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{tripathy2026hybridquantumtransformer,
  author = {Utkarsh Tripathy},
  title = {Hybrid Quantum Transformer},
  year = {2026},
  url = {https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer}
}
