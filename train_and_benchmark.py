import time
import math
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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

def train_model(model, train_loader, device, num_steps=50):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    losses = []
    perplexities = []
    step_times = []
    memory_usages = []
    
    model.train()
    step = 0
    
    # Warm up step to ignore compilation/initialization overhead in time measurements
    print(f"Warming up {model.__class__.__name__}...")
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        break

    print(f"Training {model.__class__.__name__} for {num_steps} steps...")
    
    start_time = time.perf_counter()
    
    while step < num_steps:
        for x, y in train_loader:
            if step >= num_steps:
                break
                
            x, y = x.to(device), y.to(device)
            
            # Start timing and memory profiling
            step_start = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
                
            # Forward pass
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out.view(-1, out.size(-1)), y.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Record step metrics
            step_end = time.perf_counter()
            dt = step_end - step_start
            
            loss_val = loss.item()
            perplexity = math.exp(loss_val) if loss_val < 50 else float('inf')
            
            if device.type == 'cuda':
                mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024) # MB
            else:
                mem = 0.0 # Will track CPU memory separately or list as N/A
                
            losses.append(loss_val)
            perplexities.append(perplexity)
            step_times.append(dt)
            memory_usages.append(mem)
            
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Step {step+1}/{num_steps} | Loss: {loss_val:.4f} | Perplexity: {perplexity:.4f} | Time: {dt*1000:.1f}ms")
                
            step += 1
            
    total_time = time.perf_counter() - start_time
    
    return {
        "losses": losses,
        "perplexities": perplexities,
        "step_times": step_times,
        "memory_usages": memory_usages,
        "total_time": total_time,
        "avg_step_time": np.mean(step_times),
        "peak_memory": np.max(memory_usages) if device.type == 'cuda' else 0.0
    }

def main():
    # Run preliminary tests
    run_stage3_stage4_tests()
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Tiny Shakespeare
    seq_len = 64
    batch_size = 16
    train_loader, val_loader, tokenizer = get_data(seq_len=seq_len, batch_size=batch_size)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    num_steps = 50 # Configuration for benchmark length
    
    # 2. Train Classical Transformer
    print("\nInitializing Classical Transformer...")
    classical_model = ClassicalTransformer(vocab_size=vocab_size, seq_len=seq_len, embed_dim=32, hidden_size=128)
    classical_results = train_model(classical_model, train_loader, device, num_steps=num_steps)
    
    # 3. Train Hybrid Transformer
    print("\nInitializing Hybrid Transformer (QuantumAttention)...")
    hybrid_model = HybridTransformer(vocab_size=vocab_size, seq_len=seq_len, embed_dim=32, hidden_size=128, n_qubits=4, n_layers=2, mode='vectorized')
    hybrid_results = train_model(hybrid_model, train_loader, device, num_steps=num_steps)
    
    # 4. Generate Text
    print("\n" + "=" * 50)
    print("Text Generation Comparison")
    print("=" * 50)
    prompt = "JULIET:"
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    print("--- Classical Transformer Generation ---")
    classical_gen = classical_model.generate(context, max_new_tokens=100, temperature=0.8)
    print(tokenizer.decode(classical_gen[0].tolist()))
    print("-" * 50)
    
    print("--- Hybrid Transformer Generation ---")
    hybrid_gen = hybrid_model.generate(context, max_new_tokens=100, temperature=0.8)
    print(tokenizer.decode(hybrid_gen[0].tolist()))
    print("=" * 50 + "\n")
    
    # 5. Benchmarking & Report
    df_results = pd.DataFrame({
        "Metric": ["Average Step Time (s)", "Total Training Time (s)", "Final Loss", "Final Perplexity", "Peak GPU Memory (MB)"],
        "Classical MHA": [
            classical_results["avg_step_time"],
            classical_results["total_time"],
            classical_results["losses"][-1],
            classical_results["perplexities"][-1],
            classical_results["peak_memory"] if device.type == 'cuda' else "N/A"
        ],
        "Quantum Attention": [
            hybrid_results["avg_step_time"],
            hybrid_results["total_time"],
            hybrid_results["losses"][-1],
            hybrid_results["perplexities"][-1],
            hybrid_results["peak_memory"] if device.type == 'cuda' else "N/A"
        ]
    })
    
    print(df_results.to_markdown(index=False))
    
    # Save results to a CSV
    df_results.to_csv("benchmark_results.csv", index=False)
    
    # Save training metrics to CSVs
    df_metrics = pd.DataFrame({
        "step": list(range(1, num_steps + 1)),
        "classical_loss": classical_results["losses"],
        "classical_perplexity": classical_results["perplexities"],
        "classical_step_time": classical_results["step_times"],
        "hybrid_loss": hybrid_results["losses"],
        "hybrid_perplexity": hybrid_results["perplexities"],
        "hybrid_step_time": hybrid_results["step_times"]
    })
    df_metrics.to_csv("training_metrics.csv", index=False)
    
    # 6. Plotting
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(classical_results["losses"], label="Classical (MHA)", color="#2b5c8f", lw=2)
    plt.plot(hybrid_results["losses"], label="Hybrid (Quantum)", color="#d95f02", lw=2)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Perplexity plot
    plt.subplot(2, 2, 2)
    plt.plot(classical_results["perplexities"], label="Classical (MHA)", color="#2b5c8f", lw=2)
    plt.plot(hybrid_results["perplexities"], label="Hybrid (Quantum)", color="#d95f02", lw=2)
    plt.title("Perplexity")
    plt.xlabel("Step")
    plt.ylabel("Perplexity")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Step Time comparison
    plt.subplot(2, 2, 3)
    categories = ["Classical MHA", "Quantum Attention"]
    times = [classical_results["avg_step_time"] * 1000, hybrid_results["avg_step_time"] * 1000]
    plt.bar(categories, times, color=["#2b5c8f", "#d95f02"], alpha=0.8, width=0.5)
    plt.ylabel("Average Step Time (ms)")
    plt.title("Computation Time per Step")
    plt.grid(True, linestyle="--", alpha=0.6, axis='y')
    for i, v in enumerate(times):
        plt.text(i, v + 2, f"{v:.1f} ms", ha='center', fontweight='bold')
        
    # Final Loss Comparison
    plt.subplot(2, 2, 4)
    losses_comp = [classical_results["losses"][-1], hybrid_results["losses"][-1] ]
    plt.bar(categories, losses_comp, color=["#2b5c8f", "#d95f02"], alpha=0.8, width=0.5)
    plt.ylabel("Final Loss")
    plt.title("Loss after 50 Steps")
    plt.grid(True, linestyle="--", alpha=0.6, axis='y')
    for i, v in enumerate(losses_comp):
        plt.text(i, v + 0.05, f"{v:.4f}", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("benchmark_comparison.png", dpi=300)
    plt.close()
    
    print("\nBenchmark comparison plot saved as 'benchmark_comparison.png'")

if __name__ == "__main__":
    main()
