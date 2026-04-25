import torch
import torch.optim as optim
import numpy as np
from model.sentinel import GardecloudSentinel

# 1. Load the "Founder's Config"
config = {
    "vocab_size": 12000,
    "n_layers": 8,
    "n_heads": 8,
    "n_embd": 512,
    "max_seq_len": 1024
}

def train():
    # Set device: iPad M2 uses 'mps', Cloudflare/Servers use 'cuda'
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Gardecloud-sentinel training on: {device}")

    # 2. Initialize your sovereign model
    model = GardecloudSentinel(config).to(device)
    
    # 3. Load that Scansion data we prepared
    data = np.fromfile('data/train_data.bin', dtype=np.uint16)
    data = torch.tensor(data.astype(np.int64))

    # 4. The Optimizer (The 'Learning' algorithm)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    
    # Simplified training loop
    for step in range(1000): # Let's start with 1000 steps
        # Grab a random 'chunk' of text to learn from
        ix = torch.randint(len(data) - config['max_seq_len'], (4,)) # Batch size 4
        x = torch.stack([data[i:i+config['max_seq_len']] for i in ix]).to(device)
        y = torch.stack([data[i+1:i+config['max_seq_len']+1] for i in ix]).to(device)

        # Forward pass
        logits = model(x)
        loss = criterion(logits.view(-1, config['vocab_size']), y.view(-1))

        # Backward pass (The actual 'learning')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Loss {loss.item():.4f}")

    # 5. Save the 'Brain' weights
    torch.save(model.state_dict(), "model/weights/sentinel_v1.pth")
    print("Training complete. Weights saved to model/weights/sentinel_v1.pth")

if __name__ == "__main__":
    train()
