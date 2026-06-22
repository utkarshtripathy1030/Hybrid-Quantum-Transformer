Hybrid Quantum Transformer
A research-oriented implementation of a Hybrid Quantum Self-Attention Layer for small-scale language modeling. This project integrates classical Transformer architectures with parameterized quantum circuits to investigate whether quantum-enhanced attention mechanisms can produce meaningful representations in sequence modeling tasks.

Overview
Standard Transformers rely entirely on classical computation. This project replaces a component of the self-attention mechanism with a Parameterized Quantum Circuit (PQC) implemented via PennyLane, yielding a hybrid quantum-classical architecture that can be prototyped and benchmarked on classical hardware.
Research objectives:

Design quantum-enhanced feature representations for attention computation
Develop and evaluate hybrid quantum-classical attention mechanisms
Assess feasibility of quantum Transformers on NISQ-era hardware constraints
Benchmark against classical baselines on sequence lengths of 64–256 tokens


Architecture
Input Tokens
     ↓
Embedding Layer
     ↓
Hybrid Quantum Self-Attention
  ├── Angle Encoding
  ├── Parameterized 4-Qubit Quantum Circuit
  └── Measurement → Attention Representation
     ↓
Feed-Forward Network
     ↓
Output Layer

Key Features

Hybrid Quantum Self-Attention — PennyLane quantum circuits integrated with PyTorch autograd
4-qubit PQC for attention computation via angle encoding
Classical embedding and feed-forward layers preserving standard Transformer structure
Classical simulation for rapid prototyping without quantum hardware
Modular, research-oriented codebase designed for extensibility and benchmarking


Technology Stack
ComponentLibraryDeep LearningPyTorchQuantum MLPennyLaneNumerical ComputingNumPyVisualizationMatplotlib

Installation
bashgit clone https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer.git
cd Hybrid-Quantum-Transformer
pip install -r requirements.txt
Or install manually:
bashpip install torch pennylane numpy matplotlib
Verify PennyLane:
pythonimport pennylane as qml
dev = qml.device("default.qubit", wires=4)
print(dev)  # <default.qubit device (wires=4)>

Usage
bashpython train_and_benchmark.py
This runs the full benchmark pipeline:

Validates the quantum circuit and PyTorch-wrapped quantum layer
Trains the classical Transformer baseline
Trains the Hybrid Quantum Transformer
Computes validation loss and perplexity for both models
Evaluates noise robustness
Generates side-by-side comparison plots (results_comparison.png)


Project Structure
Hybrid-Quantum-Transformer/
├── classical_transformer.py    # Classical baseline implementation
├── quantum_attention.py        # PQC definition and quantum attention layer
├── hybrid_transformer.py       # Hybrid model combining both components
├── train_and_benchmark.py      # Training, evaluation, and benchmarking pipeline
├── data/
│   └── input.txt               # Sample dataset
├── requirements.txt
└── results_comparison.png      # Output visualization

Future Directions

Multi-head quantum attention mechanisms
Quantum positional encodings
Scaling to longer sequence lengths
Deployment and benchmarking on real quantum hardware
Latency and scalability analysis against classical Transformers


Author
Utkarsh Tripathy — github.com/utkarshtripathy1030

License
MIT License — see LICENSE for details.

Citation
bibtex@misc{tripathy2026hybridquantumtransformer,
  author = {Utkarsh Tripathy},
  title  = {Hybrid Quantum Transformer},
  year   = {2026},
  url    = {https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer}
}
