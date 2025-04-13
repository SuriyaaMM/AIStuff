import jax
import jax.numpy as jnp

class LogisticRegression():

    def __BinaryCrossEntropy(self, yhat: jnp.ndarray, y: jnp.ndarray):
        eps = jnp.finfo(jnp.float32).eps
        return -jnp.mean(y * jnp.log(yhat + eps) + (1 - y) * jnp.log(1 - yhat + eps))

    def __LossFunction(self, w : jnp.ndarray, x : jnp.ndarray, y: jnp.ndarray):
        yhat = self.predict(w, x)
        return self.__BinaryCrossEntropy(yhat, y)

    def __init__(self,
                 inputShape : tuple,
                 PRNGKey    : jax.Array):

        self.numFeatures    = inputShape[1]
        self.numRows        = inputShape[0]
        self.w              = jax.random.normal(PRNGKey, (self.numFeatures + 1, 1))
        self.forward        = self.__LossFunction


    def predict(self, w: jnp.ndarray, x : jnp.ndarray) -> jnp.ndarray:

        interceptAxis   = jnp.ones((1, self.numRows))
        xcomplete       = jnp.hstack((interceptAxis.T, x))

        yHat            = jax.nn.sigmoid(xcomplete @ w)
        return yHat

#------------ TESTING (random garbage of values)------------
numFeatures = 3
numRows     = 5
lr          = 1e-4
PRNGKey     = jax.random.PRNGKey(69)

x = jax.random.normal(PRNGKey, (numRows, numFeatures)) * numRows

y = jnp.array([[1.0], [0.8722506], [0.9998621], [1.0], [0.9602329]])

model = LogisticRegression((numRows, numFeatures), PRNGKey)

for epoch in range(800):
    grads = jax.grad(model.forward)(model.w, x, y)
    model.w = model.w - lr * grads

    if epoch % 100 == 0:
        loss = model.forward(model.w, x, y)
        print(f"Epoch {epoch}, Loss: {loss}")
