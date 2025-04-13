import jax 
import jax.numpy as jnp

PRNG_KEY_JAX = 0

class attention():

    def __init__(self,
                 dimEmbeddingMatrix : tuple,
                 PRNGKey            : jax.Array):
        
        assert len(dimEmbeddingMatrix) == 3, \
            "Embedding matrix Dimensions should be (batch_size, sequence_length, embedding_dim)"
        
        # PRNGKey splitting
        PRNGKeyQuery, PRNGKeyKey, PRNGKeyValue = jax.random.split(PRNGKey, 3)

        # Saving dimensions
        self.batchSize      = dimEmbeddingMatrix[0]
        self.sequenceLength = dimEmbeddingMatrix[1]
        self.dModel         = dimEmbeddingMatrix[2]
        self.dK             = self.dModel
        self.dV             = self.dModel

        # Weights Initialization for Query, Key and Values
        self.wQuery     = jax.random.uniform(PRNGKeyQuery, (self.dModel, self.dK))
        self.wKey       = jax.random.uniform(PRNGKeyKey, (self.dModel, self.dK))
        self.wValue     = jax.random.uniform(PRNGKeyValue, (self.dModel, self.dV))
    
    def __call__(self, embeddingMatrix : jnp.ndarray) -> jnp.ndarray:
        
        Q = embeddingMatrix @ self.wQuery
        K = embeddingMatrix @ self.wKey
        V = embeddingMatrix @ self.wValue

        _attentionPartial   = (Q) @ jnp.swapaxes(K, 1, 2) / jnp.sqrt(self.dK)
        _attention          = jax.nn.softmax(_attentionPartial) @ V
        return _attention