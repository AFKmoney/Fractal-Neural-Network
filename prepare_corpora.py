"""Encode Shakespeare + Gutenberg in gematria for GPU training."""
import os, pickle
import nltk
nltk.download('shakespeare', quiet=True)
nltk.download('gutenberg', quiet=True)
from nltk.corpus import shakespeare, gutenberg

GEM_SHIFT = 256
GEM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:\"-()[]{}<>/@#%^&*_~+=|\t\n0123456789"

def gematria_encode(text):
    ids = [1]  # BOS
    for ch in text:
        idx = GEM_CHARS.find(ch)
        ids.append(idx + GEM_SHIFT if idx >= 0 else GEM_SHIFT)
    ids.append(2)  # EOS
    return ids

print("Loading Shakespeare...")
shakes_raw = ""
for f in shakespeare.fileids():
    shakes_raw += shakespeare.raw(f)
shakes_tokens = gematria_encode(shakes_raw)
print(f"Shakespeare: {len(shakes_raw):,} chars -> {len(shakes_tokens):,} tokens")

print("Loading Gutenberg...")
gut_raw = ""
for f in gutenberg.fileids():
    gut_raw += gutenberg.raw(f)
gut_tokens = gematria_encode(gut_raw)
print(f"Gutenberg: {len(gut_raw):,} chars -> {len(gut_tokens):,} tokens")

# Save
os.makedirs("data", exist_ok=True)
with open("data/shakespeare_gematria.pkl", "wb") as f:
    pickle.dump(shakes_tokens, f)
with open("data/gutenberg_gematria.pkl", "wb") as f:
    pickle.dump(gut_tokens, f)

print(f"\nSaved: data/shakespeare_gematria.pkl ({os.path.getsize('data/shakespeare_gematria.pkl')/1e6:.1f}MB)")
print(f"Saved: data/gutenberg_gematria.pkl ({os.path.getsize('data/gutenberg_gematria.pkl')/1e6:.1f}MB)")
print("Done.")
