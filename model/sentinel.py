import torch
import torch.nn as nn
from .layers import TransformerBlock, RMSNorm

class GardecloudSentinel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # The 'Lookup Table' for your vocab
        self.tok_embeddings = nn.Embedding(config['vocab_size'], config['n_embd'])
        
        # The 'Stack' of Transformer Blocks (The Brain Layers)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config['n_layers'])])
        
        # The Final Output Norm and Head
        self.norm = RMSNorm(config['n_embd'])
        self.output = nn.Linear(config['n_embd'], config['vocab_size'], bias=False)

    def forward(self, tokens):
        # 1. Turn IDs into math vectors
        h = self.tok_embeddings(tokens)
        
        # 2. Process through every layer in the stack
        for layer in self.layers:
            h = layer(h)
            
        # 3. Clean up the signal
        h = self.norm(h)
        
        # 4. Predict the next word in the vocab
        logits = self.output(h)
        
        return logits
