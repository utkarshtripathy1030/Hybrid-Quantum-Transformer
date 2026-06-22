import os
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.char2idx[c] for c in s]

    def decode(self, l):
        return ''.join([self.idx2char[i] for i in l])

class ShakespeareDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_data(seq_len=64, batch_size=16, val_split=0.1):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, "input.txt")
    
    if not os.path.exists(file_path):
        print("Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    tokenizer = CharTokenizer(text)
    encoded_data = tokenizer.encode(text)
    
    n = len(encoded_data)
    train_data = encoded_data[:int(n * (1 - val_split))]
    val_data = encoded_data[int(n * (1 - val_split)):]
    
    train_dataset = ShakespeareDataset(train_data, seq_len)
    val_dataset = ShakespeareDataset(val_data, seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    return train_loader, val_loader, tokenizer
