import jax
import jax.numpy as jnp

from multiheadAttention import multiheadAttention
from feedforward import feedforward
from layerNorm import layerNorm

class encoder():

    def __init__(self,
                 dimEmbeddingMatrix     : tuple,
                 dimHidden              : int,
                 h                      : int,
                 PRNGKey                : jax.Array):
        
        self.dE         = dimEmbeddingMatrix
        self.h          = h
        
        PRNGKeyMHA, PRNGKeyFF = jax.random.split(PRNGKey)
        self.mha        = multiheadAttention(dimEmbeddingMatrix, h, PRNGKeyMHA)
        self.ff         = feedforward(dimEmbeddingMatrix[2], dimHidden, PRNGKeyFF)
        self.norm1      = layerNorm(dimEmbeddingMatrix[2])
        self.norm2      = layerNorm(dimEmbeddingMatrix[2])
    
    def __call__(self, embeddingMatrix: jnp.ndarray) -> jnp.ndarray:

        mhaOut, attentionWeights = self.mha(embeddingMatrix)
        ffOut   = self.norm2(self.ff(self.norm1(mhaOut + embeddingMatrix)) + embeddingMatrix)
        return ffOut, attentionWeights
    
        


