import matplotlib.pyplot as plt
import seaborn as sns
from encoder import *

# --- Assume these classes are defined in separate files or above ---
# from multiheadAttention import multiheadAttention # Needs the version returning weights
# from feedforward import feedforward             # Use placeholder or actual
# from layerNorm import layerNorm                 # Use placeholder or actual
# from encoder import encoder                   # Needs the version returning weights

# --- Reuse the visualization function from the previous example ---
def visualize_attention_weights(attention_weights: jnp.ndarray,
                                batch_index: int = 0,
                                head_index: int = 0,
                                sequence_labels: list = None):
    """
    Visualizes the attention weights for a specific head and batch item using a heatmap.
    (Same function as provided in the multiheadAttention test response)
    """
    # Ensure indices are valid
    batch_size, num_heads, seq_length, _ = attention_weights.shape
    if batch_index >= batch_size:
        print(f"Warning: batch_index {batch_index} out of bounds (size {batch_size}). Using index 0.")
        batch_index = 0
    if head_index >= num_heads:
        print(f"Warning: head_index {head_index} out of bounds (size {num_heads}). Using index 0.")
        head_index = 0

    # Extract the weights for the specified batch item and head
    weights_to_plot = attention_weights[batch_index, head_index, :, :]
    # Ensure data is on CPU for plotting
    weights_to_plot = jax.device_get(weights_to_plot)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights_to_plot, cmap='viridis', annot=False, fmt=".2f")
    plt.xlabel("Key Positions (Attended To)")
    plt.ylabel("Query Positions (Attending From)")
    plt.title(f"Encoder Internal Attention Weights (Batch {batch_index}, Head {head_index})")
    if sequence_labels:
        if len(sequence_labels) == seq_length:
            plt.xticks(ticks=jnp.arange(seq_length) + 0.5, labels=sequence_labels, rotation=45, ha="right")
            plt.yticks(ticks=jnp.arange(seq_length) + 0.5, labels=sequence_labels, rotation=0)
        else:
            print(f"Warning: Mismatch between sequence_labels length ({len(sequence_labels)}) and sequence length ({seq_length}). Using default ticks.")
            plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# --- Test function for the Encoder ---
def test_encoder(batch_size: int = 4,
                 seq_length: int = 12,
                 embedding_dim: int = 128, # d_model
                 num_heads: int = 8,       # h
                 hidden_dim: int = 256,    # d_hidden for FeedForward
                 seed: int = 42):
    """
    Tests the encoder class, evaluates output shape, and visualizes internal MHA weights.

    Args:
        batch_size (int): Number of sequences in the batch.
        seq_length (int): Length of each sequence.
        embedding_dim (int): Dimension of the input embeddings (d_model).
        num_heads (int): Number of attention heads (h). Must divide embedding_dim.
        hidden_dim (int): Hidden dimension for the internal feedforward layer.
        seed (int): Random seed for reproducibility.
    """
    print("--- Starting Encoder Block Test ---")

    # --- Step 1: Setup ---
    main_key = jax.random.PRNGKey(seed)
    data_key, layer_key = jax.random.split(main_key)

    print(f"Parameters: Batch Size={batch_size}, Seq Length={seq_length}, Embedding Dim={embedding_dim}, Heads={num_heads}, Hidden Dim={hidden_dim}")

    # Check divisibility constraint for MHA
    if embedding_dim % num_heads != 0:
        print(f"Error: embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads}).")
        return

    # --- Step 2: Generate Dummy Input Data ---
    input_shape = (batch_size, seq_length, embedding_dim)
    dummy_embedding_matrix = jax.random.normal(data_key, shape=input_shape)
    print(f"Generated Input Data Shape: {dummy_embedding_matrix.shape}")

    # --- Step 3: Initialize the Encoder Layer ---
    # Pass input shape tuple, hidden dim, num heads, and key
    encoder_layer = encoder(dimEmbeddingMatrix=input_shape,
                            dimHidden=hidden_dim,
                            h=num_heads,
                            PRNGKey=layer_key)
    print("Encoder Layer Initialized.")
    print(f"  - dE: {encoder_layer.dE}")
    print(f"  - Heads (h): {encoder_layer.h}")

    # --- Step 4: Apply the Encoder Layer (Forward Pass) ---
    # The modified __call__ returns both final output and internal attention weights
    output_tensor, attention_weights = encoder_layer(dummy_embedding_matrix)
    print("Forward Pass Completed.")

    # --- Step 5: Evaluation Metric - Shape Check ---
    # The output shape of an encoder block should match its input shape.
    expected_output_shape = input_shape
    print(f"Expected Output Shape: {expected_output_shape}")
    print(f"Actual Output Shape:   {output_tensor.shape}")

    if output_tensor.shape == expected_output_shape:
        print("[Metric Passed] Encoder output shape is correct.")
    else:
        print("[Metric Failed] Encoder output shape is incorrect!")

    # Check the attention weights shape (comes from the internal MHA)
    expected_weights_shape = (batch_size, num_heads, seq_length, seq_length)
    print(f"Expected Attention Weights Shape: {expected_weights_shape}")
    print(f"Actual Attention Weights Shape:   {attention_weights.shape}")
    if attention_weights.shape == expected_weights_shape:
        print("[Metric Passed] Attention weights shape is correct.")
    else:
        print("[Metric Failed] Attention weights shape is incorrect!")

    # --- Step 6: Visualization ---
    # Visualize the attention weights from the MHA sub-layer
    print("\nVisualizing Internal Attention Weights for Batch 0, Head 0...")
    seq_labels = [f"Pos_{i}" for i in range(seq_length)]
    visualize_attention_weights(attention_weights,
                                batch_index=0,
                                head_index=0,
                                sequence_labels=seq_labels)

    print("--- Encoder Test Finished ---")

# --- Run the Test ---
if __name__ == '__main__':
    # You might need to adjust paths if classes are in separate files
    # Example: from your_module import multiheadAttention, feedforward, layerNorm, encoder
    test_encoder(batch_size=2,
                 seq_length=10,
                 embedding_dim=64,  # d_model
                 num_heads=4,       # h
                 hidden_dim=128)    # Feedforward hidden dim