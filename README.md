# Hybrid Quantum Transformer

A research-oriented implementation of a **Hybrid Quantum Self-Attention Layer** for small-scale language modeling. This project integrates classical Transformer architectures with parameterized quantum circuits to investigate whether quantum-enhanced attention mechanisms can produce meaningful representations for sequence modeling tasks.

---

## Overview

Traditional Transformers rely entirely on classical computation. This project replaces a component of the self-attention mechanism with a **Parameterized Quantum Circuit (PQC)** implemented using PennyLane, resulting in a hybrid quantum-classical architecture.

**Research objectives:**
- Explore quantum-enhanced feature representations
- Develop hybrid quantum-classical attention mechanisms
- Assess the feasibility of quantum Transformers on NISQ devices
- Evaluate performance on small-scale language modeling with sequence lengths of 64–256 tokens

---

## Architecture

```
Input Tokens
      ↓
Embedding Layer
      ↓
Hybrid Quantum Self-Attention
  ├── Angle Encoding
  ├── Parameterized Quantum Circuit
  └── Measurement → Attention Representation
      ↓
Feed-Forward Network
      ↓
Output Layer
```

---

## Key Features

- **Hybrid Quantum Self-Attention Layer** — PennyLane integrated with PyTorch
- **Parameterized 4-qubit quantum circuits** for attention computation
- **Classical embedding and feed-forward layers** preserving Transformer structure
- **Classical hardware simulation** for rapid prototyping and experimentation
- Modular, research-oriented codebase designed for extensibility and benchmarking

---

## Technology Stack

| Component            | Library    |
|----------------------|------------|
| Deep Learning        | PyTorch    |
| Quantum ML           | PennyLane  |
| Numerical Computing  | NumPy      |
| Visualization        | Matplotlib |

---

## Installation

**Clone the repository:**

```bash
git clone https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer.git
cd Hybrid-Quantum-Transformer
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch torchvision torchaudio
pip install pennylane
pip install numpy matplotlib
```

**Verify PennyLane is working:**

```python
import pennylane as qml
dev = qml.device("default.qubit", wires=4)
print(dev)
```

Expected output:
```
<default.qubit device (wires=4)>
```

---

## Usage

```bash
python train_and_benchmark.py
```

This script automatically:

1. Tests the quantum circuit implementation
2. Tests the PyTorch-wrapped quantum layer
3. Trains the classical Transformer baseline
4. Trains the Hybrid Quantum Transformer
5. Computes validation loss and perplexity for both models
6. Evaluates noise robustness
7. Generates comparison plots → saved as `results_comparison.png`

---

## Project Structure

```
Hybrid-Quantum-Transformer/
│
├── classical_transformer.py    # Baseline classical implementation
├── quantum_attention.py        # Quantum circuit and attention layer
├── hybrid_transformer.py       # Hybrid model combining both
├── train_and_benchmark.py      # Training and evaluation pipeline
├── data/
│   └── input.txt               # Sample dataset
├── requirements.txt            # Dependencies
└── results_comparison.png      # Output visualization
```

---

## Research Objective

Design and simulate a hybrid quantum self-attention layer for small-scale language modeling with sequence lengths ranging from 64 to 256 tokens, and evaluate its effectiveness relative to classical attention mechanisms.

---

## Future Directions

- Multi-head quantum attention mechanisms
- Quantum positional encodings
- Scaling to larger sequence lengths
- Comprehensive benchmarking against classical Transformers
- Deployment on real quantum hardware
- Latency and scalability analysis

---

## Author

**Utkarsh Tripathy** — [github.com/utkarshtripathy1030](https://github.com/utkarshtripathy1030)

---

## License

MIT License — see [LICENSE](./LICENSE) for details.

---

## Citation

```bibtex
@misc{tripathy2026hybridquantumtransformer,
  author = {Utkarsh Tripathy},
  title  = {Hybrid Quantum Transformer},
  year   = {2026},
  url    = {https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer}
}
```
