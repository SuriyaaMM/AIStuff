import jax
import jax.numpy as jnp

class feedforward():

    def __init__(self,
                  inputDim      : int,
                  hiddenDim     : int,
                  PRNGKey       : jax.Array):
        
        PRNGKey1, PRNGKey2 = jax.random.split(PRNGKey)
        self.dW1        = jax.random.normal(PRNGKey1, (inputDim, hiddenDim))
        self.dW2        = jax.random.normal(PRNGKey2, (hiddenDim, inputDim))

    def __call__(self,
                 x: jnp.ndarray) -> jnp.ndarray:
        
        return jnp.einsum('bsh,hi->bsi', jax.nn.gelu(jnp.einsum('bsi,ih->bsh', x, self.dW1)), self.dW2)
