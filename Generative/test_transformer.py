from transformer import *

# --- Test ---
key = jax.random.PRNGKey(PRNG_KEY_JAX)

# Test single-head attention
dim_model_single = 128
dim_k_single = 64
dim_v_single = 64
single_head_attention = attention(dimModel=dim_model_single,
                                 dimK=dim_k_single,
                                 dimV=dim_v_single,
                                 PRNGKey=key)

batch_size = 4
seq_len = 10
Q_single = jnp.array(np.random.rand(batch_size, seq_len, dim_model_single))
K_single = jnp.array(np.random.rand(batch_size, seq_len, dim_model_single))
V_single = jnp.array(np.random.rand(batch_size, seq_len, dim_model_single))

output_single = single_head_attention.attend(Q_single, K_single, V_single)
print("Single-head attention output shape:", output_single.shape)

# Test multi-head attention
dim_model_multi = 128
num_heads = 8
multi_head_attention = multiheadAttention(dimModel=dim_model_multi,
                                           h=num_heads,
                                           PRNGKey=key)

Q_multi = jnp.array(np.random.rand(batch_size, seq_len, dim_model_multi))
K_multi = jnp.array(np.random.rand(batch_size, seq_len, dim_model_multi))
V_multi = jnp.array(np.random.rand(batch_size, seq_len, dim_model_multi))

output_multi = multi_head_attention.multiheadAttend(Q_multi, K_multi, V_multi)
print("Multi-head attention output shape:", output_multi.shape)

# Basic shape assertions
assert output_single.shape == (batch_size, seq_len, dim_v_single)
assert output_multi.shape == (batch_size, seq_len, dim_model_multi)

print("\nTests completed without errors.")