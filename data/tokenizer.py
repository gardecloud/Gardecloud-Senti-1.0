import json
import os

class GardecloudTokenizer:
    def __init__(self, vocab_size=12000):
        self.vocab_size = vocab_size
        # Start with the basic 256 ASCII/Byte characters
        self.vocab = {i: bytes([i]) for i in range(256)} 
        self.merges = {}

    def get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, corpus_path):
        if not os.path.exists(corpus_path):
            print(f"❌ Error: Could not find {corpus_path}")
            return
            
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        print(f"📖 Reading {len(text)} characters from {corpus_path}...")
        ids = list(text.encode("utf-8"))
        
        num_merges = self.vocab_size - 256
        print(f"⚙️ Forging {num_merges} new tokens for Gardecloud...")

        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if (i + 1) % 500 == 0:
                print(f"🏗️  Progress: {i+1}/{num_merges} tokens merged...")

        self.save_vocab()
        print("✅ Tokenization Engine: Training Complete.")

    def save_vocab(self):
        readable_vocab = {}
        for idx, token_bytes in self.vocab.items():
            try:
                readable_vocab[idx] = token_bytes.decode('utf-8')
            except:
                readable_vocab[idx] = str(token_bytes)
        
        # Ensure the data folder exists
        os.makedirs("data", exist_ok=True)
        
        with open("data/vocab.json", "w") as f:
            json.dump(readable_vocab, f, indent=2)
        print("💾 Saved dictionary to data/vocab.json")

# --- EXECUTION BLOCK ---
if __name__ == "__main__":
    print("--- 🚀 GARDECLOUD SENTINEL: DATA ENGINE STARTING ---")
    
    # Initialize
    t = GardecloudTokenizer(vocab_size=12000)
    
    # Define the path to your Scansion text
    # Make sure this file exists in data/corpus/raw_data.txt!
    path = "data/corpus/raw_data.txt"
    
    t.train(path)
    print("--- 🏁 ENGINE SHUTDOWN: VOCAB IS READY ---")
