import os
import json
import numpy as np

def prepare_data():
    print("🔄 Converting text to Gardecloud IDs...")
    
    # 1. Load your newly forged vocab
    with open('data/vocab.json', 'r') as f:
        vocab = json.load(f)
    
    # Create the encoder (Text -> ID)
    # We use a simple character-level mapping for this custom build
    encoder = {v: int(k) for k, v in vocab.items()}
    
    # 2. Load your raw text
    with open('data/corpus/raw_data.txt', 'r', encoding='utf-8') as f:
        data = f.read()
    
    # 3. Encode the text
    # This turns 'Garde' into [256]
    ids = []
    # Simplified encoding for the founder build
    for char in data:
        if char in encoder:
            ids.append(encoder[char])
        else:
            ids.append(0) # Unknown token
            
    # 4. Save as binary (for high-speed training)
    train_data = np.array(ids, dtype=np.uint16)
    train_data.tofile('data/train_data.bin')
    
    print(f"✅ Success! Created train_data.bin with {len(train_data)} tokens.")

if __name__ == "__main__":
    prepare_data()
