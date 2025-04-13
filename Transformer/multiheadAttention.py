import jax
import jax.numpy as jnp

class multiheadAttention:

    def __init__(self,
                 dimEmbeddingMatrix     : tuple,
                 h                      : int,
                 PRNGKey                : jax.Array):

        assert dimEmbeddingMatrix[2] % h == 0, \
            "Unsymmetrical head count for Multi-Head Attention!"

        # PRNGKey splitting
        PRNGKeyQuery, PRNGKeyKey, PRNGKeyValue, PRNGKeyOut = \
            jax.random.split(PRNGKey, 4)
        
        # Saving Dimensions
        self.dE             = dimEmbeddingMatrix
        self.dModel         = dimEmbeddingMatrix[2]
        self.dK             = self.dModel // h
        self.dV             = self.dModel // h
        self.h              = h

        # Initialization of Query, Key, Value and Output Weight's
        self.wQuery     = jax.random.normal(PRNGKeyQuery,  (self.dModel, self.h, self.dK))
        self.wKey       = jax.random.normal(PRNGKeyKey,    (self.dModel, self.h, self.dK))
        self.wValue     = jax.random.normal(PRNGKeyValue,  (self.dModel, self.h, self.dV))
        self.wOut       = jax.random.normal(PRNGKeyOut,    (self.h * self.dV, self.dModel))
        self.PRNGKeys   = jax.random.split(PRNGKey, h)
    
    def __call__(self, 
                embeddingMatrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        
        batchSize, sequenceLength, _ = embeddingMatrix.shape

        # Calculate Q, K and V
        Q = jnp.einsum("bse,ehq->bshq", embeddingMatrix, self.wQuery)
        K = jnp.einsum("bse,ehk->bshk", embeddingMatrix, self.wKey)
        V = jnp.einsum("bse,ehv->bshv", embeddingMatrix, self.wValue)
        
        # Reshape them into (batch_size, h, seq_length, dim per head) (bhsq), (bhsv)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))        

        # Calculate attenion score
        attentionScores     = jnp.einsum("bhsq,bhtk->bhst", Q, K) / jnp.sqrt(self.dK)
        # Calculate attention weights
        attentionWeights    = jax.nn.softmax(attentionScores, axis=-1)
        # Calculate head output
        headOutputs         = attentionWeights @ V 
        # Concatenate heads
        concatenatedHeads   = jnp.transpose(headOutputs, (0, 2, 1, 3))
        # Reshape Concatenated heads
        concatenatedHeads   = concatenatedHeads.reshape(batchSize, sequenceLength, self.h * self.dV)
        # Calculate output
        _multiheadAttention = jnp.einsum("bsd,dm->bsm", concatenatedHeads, self.wOut)
    
        return _multiheadAttention, attentionWeights

        