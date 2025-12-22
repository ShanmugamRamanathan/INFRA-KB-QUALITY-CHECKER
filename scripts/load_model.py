from sentence_transformers import SentenceTransformer
import torch

# Load BGE model
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Move model to GPU
if torch.cuda.is_available():
    model = model.to('cuda')
    print("Model moved to GPU")
else:
    print("GPU not available, using CPU")

# Test BGE model
sentences = ["How to fix SCOM heartbeat failure?", "Restart the Health Service."]
embeddings = model.encode(sentences)
print(embeddings.shape)  # Should print (2, 384)