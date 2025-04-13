from multiheadAttention import *
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention_weights(attention_weights: jnp.ndarray,
                                batch_index: int = 0,
                                head_index: int = 0,
                                sequence_labels: list = None):
    """
    Visualizes the attention weights for a specific head and batch item using a heatmap.

    Args:
        attention_weights (jnp.ndarray): The attention weights tensor of shape
                                         (batch_size, num_heads, seq_length, seq_length).
        batch_index (int): The index of the batch item to visualize. Defaults to 0.
        head_index (int): The index of the attention head to visualize. Defaults to 0.
        sequence_labels (list, optional): Labels for the sequence positions (ticks on axes).
                                          Defaults to None (uses numerical indices).
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
    # Shape: (seq_length, seq_length)
    weights_to_plot = attention_weights[batch_index, head_index, :, :]

    # Ensure data is on CPU for plotting
    weights_to_plot = jax.device_get(weights_to_plot)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.heatmap(weights_to_plot, cmap='viridis', annot=False, fmt=".2f") # annot=True can be slow for large sequences

    plt.xlabel("Key Positions (Attended To)")
    plt.ylabel("Query Positions (Attending From)")
    plt.title(f"Attention Weights (Batch {batch_index}, Head {head_index})")

    # Set ticks and labels if provided
    if sequence_labels:
        if len(sequence_labels) == seq_length:
            plt.xticks(ticks=jnp.arange(seq_length) + 0.5, labels=sequence_labels, rotation=45, ha="right")
            plt.yticks(ticks=jnp.arange(seq_length) + 0.5, labels=sequence_labels, rotation=0)
        else:
            print(f"Warning: Mismatch between sequence_labels length ({len(sequence_labels)}) and sequence length ({seq_length}). Using default ticks.")
            plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

def test_multihead_attention(batch_size: int = 4,
                             seq_length: int = 10,
                             embedding_dim: int = 128, # d_model
                             num_heads: int = 8,       # h
                             seed: int = 42):
    """
    Tests the multiheadAttention class, evaluates output shape, and visualizes weights.

    Args:
        batch_size (int): Number of sequences in the batch.
        seq_length (int): Length of each sequence.
        embedding_dim (int): Dimension of the input embeddings (d_model).
        num_heads (int): Number of attention heads (h). Must divide embedding_dim.
        seed (int): Random seed for reproducibility.
    """
    print("--- Starting Multi-Head Attention Test ---")

    # --- Step 1: Setup ---
    # Create a main PRNG key
    main_key = jax.random.PRNGKey(seed)
    # Split the key for data generation and layer initialization
    data_key, layer_key = jax.random.split(main_key)

    print(f"Parameters: Batch Size={batch_size}, Seq Length={seq_length}, Embedding Dim={embedding_dim}, Heads={num_heads}")

    # Check divisibility constraint
    if embedding_dim % num_heads != 0:
        print(f"Error: embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads}).")
        return

    # --- Step 2: Generate Dummy Input Data ---
    # Create a random input tensor (simulating token embeddings)
    input_shape = (batch_size, seq_length, embedding_dim)
    # Use jax.random for consistency
    dummy_embedding_matrix = jax.random.normal(data_key, shape=input_shape)
    print(f"Generated Input Data Shape: {dummy_embedding_matrix.shape}")

    # --- Step 3: Initialize the Attention Layer ---
    # Pass the expected input shape (or at least d_model), number of heads, and a PRNG key
    attention_layer = multiheadAttention(dimEmbeddingMatrix=input_shape,
                                         h=num_heads,
                                         PRNGKey=layer_key)
    print("Multi-Head Attention Layer Initialized.")
    print(f"  - dModel: {attention_layer.dModel}")
    print(f"  - Heads (h): {attention_layer.h}")
    print(f"  - dK (Key/Query Dim per Head): {attention_layer.dK}")
    print(f"  - dV (Value Dim per Head): {attention_layer.dV}")

    # --- Step 4: Apply the Attention Layer (Forward Pass) ---
    # Call the layer instance with the dummy data
    # The modified __call__ returns both output and attention weights
    output_tensor, attention_weights = attention_layer(dummy_embedding_matrix)
    print("Forward Pass Completed.")

    # --- Step 5: Evaluation Metric - Shape Check ---
    # The primary check is whether the output tensor has the expected shape.
    # It should match the input shape: (batch_size, seq_length, embedding_dim)
    expected_output_shape = input_shape
    print(f"Expected Output Shape: {expected_output_shape}")
    print(f"Actual Output Shape:   {output_tensor.shape}")

    if output_tensor.shape == expected_output_shape:
        print("[Metric Passed] Output shape is correct.")
    else:
        print("[Metric Failed] Output shape is incorrect!")

    # Also check the attention weights shape
    expected_weights_shape = (batch_size, num_heads, seq_length, seq_length)
    print(f"Expected Attention Weights Shape: {expected_weights_shape}")
    print(f"Actual Attention Weights Shape:   {attention_weights.shape}")
    if attention_weights.shape == expected_weights_shape:
        print("[Metric Passed] Attention weights shape is correct.")
    else:
        print("[Metric Failed] Attention weights shape is incorrect!")


    # --- Step 6: Visualization ---
    # Visualize the attention weights for the first head of the first batch item
    print("\nVisualizing Attention Weights for Batch 0, Head 0...")
    # Optional: Create labels for sequence positions
    seq_labels = [f"Pos_{i}" for i in range(seq_length)]
    visualize_attention_weights(attention_weights,
                                batch_index=0,
                                head_index=0,
                                sequence_labels=seq_labels) # Pass labels here

    print("--- Test Finished ---")

# --- Run the Test ---
if __name__ == '__main__': # Ensures this runs only when script is executed directly
    test_multihead_attention(batch_size=2,      # Smaller batch for testing
                             seq_length=8,      # Shorter sequence for visualization
                             embedding_dim=64,  # d_model
                             num_heads=4)       # h