import torch
import torch.nn as nn
import pennylane as qml

n_qubits = 4

# Define the PennyLane device
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface='torch')
def quantum_layer(inputs, weights):
    # RY rotations for inputs
    for i in range(n_qubits):
        qml.RY(inputs[..., i], wires=i)

    # Entangling CNOT gates
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

    # Strongly entangling layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Return PauliZ expectation values for all qubits
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumAttention(nn.Module):
    def __init__(self, embed_dim=32, n_qubits=4, n_layers=2, mode='vectorized'):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.mode = mode
        
        # Dimensions of Q and K project to n_qubits // 2 so their concat matches n_qubits
        self.q_proj = nn.Linear(embed_dim, n_qubits // 2, bias=False)
        self.k_proj = nn.Linear(embed_dim, n_qubits // 2, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Wrap PennyLane QNode in TorchLayer
        self.qlayer = qml.qnn.TorchLayer(quantum_layer, weight_shapes)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        Q = self.q_proj(x)  # (B, T, d_q) where d_q = n_qubits // 2
        K = self.k_proj(x)  # (B, T, d_q)
        V = self.v_proj(x)  # (B, T, C)
        
        if self.mode == 'vectorized':
            # Vectorized / batched implementation
            Q_expanded = Q.unsqueeze(2).expand(-1, -1, T, -1)  # (B, T, T, d_q)
            K_expanded = K.unsqueeze(1).expand(-1, T, -1, -1)  # (B, T, T, d_q)
            
            # Concatenate Q and K
            combined = torch.cat([Q_expanded, K_expanded], dim=-1)  # (B, T, T, 2 * d_q)
            
            # Flatten to (B * T * T, n_qubits)
            combined_flat = combined.reshape(B * T * T, self.n_qubits)
            
            # Run quantum layer in a single batch
            qlayer_out = self.qlayer(combined_flat)  # (B * T * T, n_qubits)
            
            # Take the mean of the expectation values
            scores = qlayer_out.mean(dim=-1)  # (B * T * T,)
            
            # Reshape back to (B, T, T)
            scores = scores.view(B, T, T)
        else:
            # Sequential loop-based implementation (adapted for batch size B)
            scores_list = []
            for b in range(B):
                row_list = []
                for i in range(T):
                    col_list = []
                    for j in range(T):
                        # Combine Q[b, i] and K[b, j]
                        inp = torch.cat([Q[b, i], K[b, j]], dim=-1)  # (n_qubits,)
                        score = self.qlayer(inp).mean()  # scalar
                        col_list.append(score)
                    row_list.append(torch.stack(col_list))
                scores_list.append(torch.stack(row_list))
            scores = torch.stack(scores_list)  # (B, T, T)
            
        # Apply causal mask if provided
        if mask is not None:
            # Mask should have shape (T, T) or (B, T, T)
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask == False, float('-inf'))
            elif (mask == float('-inf')).any():
                # PyTorch style additive mask
                # Check dimensional compatibility
                if mask.dim() == 2:
                    scores = scores + mask.unsqueeze(0)
                else:
                    scores = scores + mask
            else:
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
        weights = torch.softmax(scores, dim=-1)  # (B, T, T)
        out = torch.matmul(weights, V)  # (B, T, C)
        return self.out_proj(out)
