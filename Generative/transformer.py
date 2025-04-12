import jax
import jax.numpy as jnp
import numpy as np

PRNG_KEY_JAX = 0

class attention():

    def __init__(self,
                 dimModel   : int,
                 dimK       : int,
                 dimV       : int,
                 PRNGKey    : jax.Array):
        
        # PRNGKey splitting
        PRNGKeyQuery, PRNGKeyKey, PRNGKeyValue = jax.random.split(PRNGKey, 3)
        # Weights Initialization for Query, Key and Values
        self.wQuery     = jax.random.uniform(PRNGKeyQuery, (dimModel, dimK))
        self.wKey       = jax.random.uniform(PRNGKeyKey, (dimModel, dimK))
        self.wValue     = jax.random.uniform(PRNGKeyValue, (dimModel, dimV))
        # Storing Dimensions
        self.dK         = dimK
        self.dModel     = dimModel
        self.dV         = dimV
    
    def attend(self, 
               Q: jnp.ndarray, 
               K: jnp.ndarray, 
               V: jnp.ndarray) -> jnp.ndarray:
        
        _attention = (Q @ self.wQuery) @ jnp.swapaxes((K @ self.wKey), 1, 2) \
            / jnp.sqrt(self.dK)
        _headOut = jax.nn.softmax(_attention) @ (V @ self.wValue)
        return _headOut
        
class multiheadAttention():

    def __init__(self,
                 dimModel   : int,
                 h          : int,
                 PRNGKey    : jax.Array):

        assert dimModel % h == 0, \
            "Unsymmetrical Division for Multi-Head Attention!"

        PRNGKeyQuery, PRNGKeyKey, PRNGKeyValue, PRNGKeyOut = \
            jax.random.split(PRNGKey, 4)
        
        self.wQuery     = jax.random.uniform(PRNGKeyQuery, (dimModel, dimModel))
        self.wKey       = jax.random.uniform(PRNGKeyKey, (dimModel, dimModel))
        self.wValue     = jax.random.uniform(PRNGKeyValue, (dimModel, dimModel))
        self.wOut       = jax.random.uniform(PRNGKeyOut, (dimModel, dimModel))
        self.dModel     = dimModel
        self.dK         = dimModel // h
        self.dV         = self.dK
        self.h          = h
        self.PRNGKeys   = jax.random.split(PRNGKey, h)
    
    def multiheadAttend(self, 
                        Q: jnp.ndarray, 
                        K: jnp.ndarray, 
                        V: jnp.ndarray) -> jnp.ndarray:
        
        assert len(Q.shape) >= 3, \
            "Query matrix should have at least 3 Dimensions!"
        
        batchSize, sequenceLength, _ = Q.shape

        headQ = jnp.transpose((Q @ self.wQuery).reshape(batchSize,
                                          sequenceLength,
                                          self.h,
                                          self.dK), (0, 2, 1, 3))
        headK = jnp.transpose((K @ self.wKey).reshape(batchSize,
                                          sequenceLength,
                                          self.h,
                                          self.dK), (0, 2, 1, 3))
        headV = jnp.transpose((V @ self.wValue).reshape(batchSize,
                                          sequenceLength,
                                          self.h,
                                          self.dV), (0, 2, 1, 3))

        headsOut = []

        for i in range(self.h):

            headOuti = attention(self.dK,
                                    self.dK,
                                    self.dV,
                                    self.PRNGKeys[i])
            headsOut.append(headOuti.attend(headQ[:, i], 
                                            headK[:, i], 
                                            headV[:, i]))
        
        _multiheadAttention = \
            jnp.concatenate(headsOut, axis = -1).reshape(batchSize,
                                                         sequenceLength,
                                                         self.dModel)
        
        return _multiheadAttention @ self.wOut


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
