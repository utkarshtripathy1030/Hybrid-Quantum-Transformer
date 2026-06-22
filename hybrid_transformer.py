import torch
import torch.nn as nn
from quantum_attention import QuantumAttention

class HybridTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len=64, embed_dim=32, hidden_size=128, n_qubits=4, n_layers=2, mode='vectorized'):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = QuantumAttention(embed_dim=embed_dim, n_qubits=n_qubits, n_layers=n_layers, mode=mode)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embed_dim)
        )
        
        self.ln_out = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        self.seq_len = seq_len
        
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos)  # (B, T, C)
        
        # Causal mask for autoregressive generation
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        
        # Hybrid block with Pre-LN Residuals
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        x = self.ln_out(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # Generate new characters autoregressively
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits = self(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
