mport torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
torch.manual_seed(42)
 
WORDS = ["PDF", "Ingestion", "Engine", "Indexes", "Vector", "Embeddings"]
D_MODEL = 8
 
def scaled_dot_product_attention(Q, K, V, causal_mask=False):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)
    if causal_mask:
        seq_len = scores.shape[-1]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    output = weights @ V
    return output, weights
 
def get_sample_embeddings():
    return torch.randn(len(WORDS), D_MODEL)
 
def plot_heatmap(weights, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(weights.detach().numpy(), annot=True, fmt=".2f", cmap="viridis", xticklabels=WORDS, yticklabels=WORDS, cbar=True)
    plt.title(title)
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
 
if __name__ == "__main__":
    embeddings = get_sample_embeddings()
    W_q = torch.randn(D_MODEL, D_MODEL)
    W_k = torch.randn(D_MODEL, D_MODEL)
    W_v = torch.randn(D_MODEL, D_MODEL)
 
    Q = embeddings @ W_q
    K = embeddings @ W_k
    V = embeddings @ W_v
 
    output_full, weights_full = scaled_dot_product_attention(Q, K, V, causal_mask=False)
    plot_heatmap(weights_full, "Self-Attention (No Causal Mask)", "attention_no_mask.png")
 
    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, causal_mask=True)
    plot_heatmap(weights_causal, "Self-Attention (Causal Mask)", "attention_causal_mask.png")