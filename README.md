Hybrid Quantum Transformer
A research-oriented implementation of a Hybrid Quantum Self-Attention Layer for small-scale language modeling. This project integrates classical Transformer architectures with quantum circuits to investigate whether quantum-enhanced attention mechanisms can produce meaningful representations for sequence modeling tasks.

Overview
Traditional Transformers rely entirely on classical computation. This project replaces a component of the self-attention mechanism with a parameterized quantum circuit implemented using PennyLane, resulting in a hybrid quantum-classical architecture.

The primary research objectives are:

Exploring quantum-enhanced feature representations

Developing hybrid quantum-classical attention mechanisms

Assessing the feasibility of quantum Transformers on NISQ devices

Evaluating performance on small-scale language modeling with sequence lengths of 64–256 tokens

Key Features
Hybrid Quantum Self-Attention Layer integrating PennyLane with PyTorch

Parameterized 4-qubit quantum circuits for attention computation

Classical embedding and feed-forward layers preserving Transformer structure

Simulation on classical hardware for rapid prototyping and experimentation

Research-oriented implementation designed for extensibility and benchmarking

Architecture
text
Input Tokens
      ↓
Embedding Layer
      ↓
Hybrid Quantum Self-Attention
      ↓
Feed Forward Network
      ↓
Output Layer
Quantum Layer Details
text
Classical Features
        ↓
Angle Encoding
        ↓
Parameterized Quantum Circuit
        ↓
Measurement
        ↓
Attention Representation
Technology Stack
Python — Core programming language

PyTorch — Deep learning framework

PennyLane — Quantum machine learning library

NumPy — Numerical computing

Matplotlib — Visualization and plotting

Installation
Clone the repository:

bash
git clone https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer.git
cd Hybrid-Quantum-Transformer
Install dependencies:

bash
pip install torch pennylane numpy matplotlib
Usage
Run the benchmark pipeline:

bash
python train_and_benchmark.py
This will:

Test the quantum circuit implementation

Train both classical and hybrid Transformer models

Compute validation loss and perplexity

Evaluate noise robustness

Generate comparison plots

Project Structure
text
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
Research Objective
Design and simulate a hybrid quantum self-attention layer for small-scale language modeling with sequence lengths ranging from 64 to 256 tokens, and evaluate its effectiveness relative to classical attention mechanisms.

Future Directions
Multi-head quantum attention mechanisms

Quantum positional encodings

Scaling to larger sequence lengths

Comprehensive benchmarking against classical Transformers

Deployment on real quantum hardware

Latency and scalability analysis

Author
Utkarsh Tripathy

GitHub: https://github.com/utkarshtripathy1030

License
MIT License — see the LICENSE file for details.

Citation
If you use this project in your research, please cite:

bibtex
@misc{tripathy2026hybridquantumtransformer,
  author = {Utkarsh Tripathy},
  title = {Hybrid Quantum Transformer},
  year = {2026},
  url = {https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer}
}
Reproducing the Results
Clone the Repository
bash
git clone https://github.com/utkarshtripathy1030/Hybrid-Quantum-Transformer.git
cd Hybrid-Quantum-Transformer
Create a Virtual Environment (Optional)
Windows
bash
python -m venv venv
venv\Scripts\activate
Linux / macOS
bash
python3 -m venv venv
source venv/bin/activate
Install Dependencies
bash
pip install torch torchvision torchaudio
pip install pennylane
pip install numpy matplotlib
Or install everything at once:

bash
pip install -r requirements.txt
Verify PennyLane Installation
Run:

python
import pennylane as qml
dev = qml.device("default.qubit", wires=4)
print(dev)
Expected output:

text
<default.qubit device (wires=4)>
Project Structure
text
Hybrid-Quantum-Transformer/
│
├── classical_transformer.py
├── quantum_attention.py
├── hybrid_transformer.py
├── train_and_benchmark.py
├── data/
│   └── input.txt
├── requirements.txt
└── results_comparison.png
Run the Benchmark Pipeline
Execute:

bash
python train_and_benchmark.py
This script automatically:

Tests the quantum circuit

Tests the PyTorch wrapped quantum layer

Trains the Classical Transformer baseline

Trains the Hybrid Quantum Transformer

Computes validation loss and perplexity

Evaluates noise robustness

Generates comparison plots

