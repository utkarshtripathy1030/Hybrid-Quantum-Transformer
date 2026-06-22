import time
import math
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Import our custom modules
from data_prep import get_data
from classical_transformer import ClassicalTransformer
from quantum_attention import quantum_layer, n_qubits
from hybrid_transformer import HybridTransformer

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def run_stage3_stage4_tests():
    print("=" * 50)
    print("STAGE 3: Testing Quantum Circuit")
    print("=" * 50)
    weights = torch.randn((2, n_qubits, 3))
    inputs = torch.randn(n_qubits)
    q_out = quantum_layer(inputs, weights)
    print(f"Inputs: {inputs}")
    print(f"Weights shape: {weights.shape}")
    print(f"Quantum Circuit Output: {q_out}")
    
    print("\n" + "=" * 50)
    print("STAGE 4: Testing PyTorch Wrapped Quantum Layer")
    print("=" * 50)
    import pennylane as qml
    weight_shapes = {"weights": (2, n_qubits, 3)}
    qlayer = qml.qnn.TorchLayer(quantum_layer, weight_shapes)
    x = torch.randn(4)
    output = qlayer(x)
    print(f"Input: {x}")
    print(f"TorchLayer Output: {output}")
    print(f"TorchLayer Output shape: {output.shape} (acts like nn.Linear)")
    print("=" * 50 + "\n")

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    steps = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / steps if steps > 0 else 0.0

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    steps = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            total_loss += loss.item()
            steps += 1
    return total_loss / steps if steps > 0 else 0.0

def main():
    # Run preliminary circuit tests
    run_stage3_stage4_tests()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load dataset (same data for both models)
    seq_len = 64
    batch_size = 16
    train_loader, val_loader, tokenizer = get_data(seq_len=seq_len, batch_size=batch_size, max_chars=12000)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    
    # Track Classical model parameters
    classical_model = ClassicalTransformer(vocab_size=vocab_size, seq_len=seq_len, embed_dim=32, hidden_size=128)
    classical_params = sum(p.numel() for p in classical_model.parameters())
    print(f"\nClassical Transformer Parameters: {classical_params}")
    
    # Train Classical Model
    classical_train_losses = []
    classical_val_losses = []
    classical_val_perplexities = []
    
    print("\nTraining Classical Transformer...")
    classical_model.to(device)
    classical_opt = torch.optim.Adam(classical_model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        train_loss = train_epoch(classical_model, train_loader, classical_opt, criterion, device)
        val_loss = evaluate_model(classical_model, val_loader, criterion, device)
        val_ppl = math.exp(val_loss)
        
        classical_train_losses.append(train_loss)
        classical_val_losses.append(val_loss)
        classical_val_perplexities.append(val_ppl)
        
        dt = time.perf_counter() - t0
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.4f} | Time: {dt:.2f}s")
        
    # Track Hybrid model parameters
    hybrid_model = HybridTransformer(vocab_size=vocab_size, seq_len=seq_len, embed_dim=32, hidden_size=128, n_qubits=4, n_layers=2, mode='vectorized')
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"\nHybrid Transformer Parameters: {hybrid_params}")
    
    # Train Hybrid Model
    hybrid_train_losses = []
    hybrid_val_losses = []
    hybrid_val_perplexities = []
    
    print("\nTraining Hybrid Transformer...")
    hybrid_model.to(device)
    hybrid_opt = torch.optim.Adam(hybrid_model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        t0 = time.perf_counter()
        train_loss = train_epoch(hybrid_model, train_loader, hybrid_opt, criterion, device)
        val_loss = evaluate_model(hybrid_model, val_loader, criterion, device)
        val_ppl = math.exp(val_loss)
        
        hybrid_train_losses.append(train_loss)
        hybrid_val_losses.append(val_loss)
        hybrid_val_perplexities.append(val_ppl)
        
        dt = time.perf_counter() - t0
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.4f} | Time: {dt:.2f}s")
        
    # 6. Noise Robustness evaluation
    print("\nEvaluating Noise Robustness for Hybrid Model...")
    
    # Noise-free evaluation
    hybrid_model.attn.add_noise = False
    val_loss_clean = evaluate_model(hybrid_model, val_loader, criterion, device)
    val_ppl_clean = math.exp(val_loss_clean)
    
    # Noisy evaluation (+ torch.randn_like(output)*0.01)
    hybrid_model.attn.add_noise = True
    val_loss_noisy = evaluate_model(hybrid_model, val_loader, criterion, device)
    val_ppl_noisy = math.exp(val_loss_noisy)
    
    print("-" * 50)
    print("Noise Robustness Comparison:")
    print(f"Noise-free Hybrid -> Loss: {val_loss_clean:.4f} | Perplexity: {val_ppl_clean:.4f}")
    print(f"Noisy Hybrid      -> Loss: {val_loss_noisy:.4f} | Perplexity: {val_ppl_noisy:.4f}")
    print("-" * 50)
    
    # Retrieve attention matrices for heatmap generation
    x_sample, _ = next(iter(val_loader))
    x_sample = x_sample[:1].to(device) # Shape (1, seq_len)
    
    # Get Classical attention matrix
    classical_model.eval()
    with torch.no_grad():
        _ = classical_model(x_sample)
    classical_attn_matrix = classical_model.attn.attention_weights[0].cpu().numpy() # shape (seq_len, seq_len)
    
    # Get Hybrid attention matrix
    hybrid_model.eval()
    hybrid_model.attn.add_noise = False
    with torch.no_grad():
        _ = hybrid_model(x_sample)
    hybrid_attn_matrix = hybrid_model.attn.attention_weights[0].cpu().numpy() # shape (seq_len, seq_len)
    
    # 7. Plotting Comparison Figure
    print("\nPlotting results and saving comparison figure...")
    epochs_range = np.arange(1, num_epochs + 1)
    
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Loss curves
    plt.subplot(3, 2, 1)
    plt.plot(epochs_range, classical_train_losses, label="Classical Train", color="#1f77b4", marker='o')
    plt.plot(epochs_range, classical_val_losses, label="Classical Val", color="#aec7e8", linestyle='--', marker='s')
    plt.plot(epochs_range, hybrid_train_losses, label="Hybrid Train", color="#ff7f0e", marker='o')
    plt.plot(epochs_range, hybrid_val_losses, label="Hybrid Val", color="#ffbb78", linestyle='--', marker='s')
    plt.title("Loss Curve Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    # Plot 2: Perplexity Comparison
    plt.subplot(3, 2, 2)
    models_list = ["Classical", "Hybrid (Noise-free)", "Hybrid (Noisy)"]
    perplexities_list = [classical_val_perplexities[-1], val_ppl_clean, val_ppl_noisy]
    bars = plt.bar(models_list, perplexities_list, color=["#1f77b4", "#ff7f0e", "#d62728"], alpha=0.8, width=0.4)
    plt.title("Final Validation Perplexity Comparison")
    plt.ylabel("Perplexity")
    plt.grid(True, linestyle="--", alpha=0.5, axis='y')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.1, f"{height:.2f}", ha='center', va='bottom', fontweight='bold')
        
    # Plot 3: Parameter counts
    plt.subplot(3, 2, 3)
    param_models = ["Classical", "Hybrid"]
    params = [classical_params, hybrid_params]
    param_bars = plt.bar(param_models, params, color=["#1f77b4", "#ff7f0e"], alpha=0.8, width=0.4)
    plt.title("Model Parameter Counts")
    plt.ylabel("Number of Parameters")
    plt.grid(True, linestyle="--", alpha=0.5, axis='y')
    for bar in param_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, height + 100, f"{height:,}", ha='center', va='bottom', fontweight='bold')
        
    # Plot 4: Classical Attention Heatmap
    plt.subplot(3, 2, 4)
    sns.heatmap(classical_attn_matrix, cmap="viridis", cbar=True)
    plt.title("Classical Attention Heatmap")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    # Plot 5: Hybrid Attention Heatmap
    plt.subplot(3, 2, 5)
    sns.heatmap(hybrid_attn_matrix, cmap="viridis", cbar=True)
    plt.title("Quantum Attention Heatmap")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    # Save the consolidated figure
    plt.tight_layout()
    plt.savefig("results_comparison.png", dpi=300)
    plt.close()
    
    print("Consolidated comparison figure saved as 'results_comparison.png'")

if __name__ == "__main__":
    main()
