import torch
import torch.nn as nn

class ClassicalAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        
    def forward(self, x, mask=None):
        # mask shape for nn.MultiheadAttention is (T, T) with -inf or True
        # If mask is None, we generate it on the fly
        if mask is None:
            T = x.shape[1]
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        # nn.MultiheadAttention returns (attn_output, attn_output_weights)
        out, _ = self.mha(x, x, x, attn_mask=mask, need_weights=False)
        return out

class ClassicalTransformer(nn.Module):
    def __init__(self, vocab_size, seq_len=64, embed_dim=32, hidden_size=128):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = ClassicalAttention(embed_dim)
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
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # (1, T)
        x = self.tok_emb(idx) + self.pos_emb(pos) # (B, T, C)
        
        # pre-LN architecture
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = x + self.attn(self.ln1(x), mask=mask)
        x = x + self.mlp(self.ln2(x))
        x = self.ln_out(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx is (B, T)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
