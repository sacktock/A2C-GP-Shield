import jax.numpy as jnp
import jax.random as jr
import jax
from jax import jit
from functools import partial
import equinox as eqx
import gpjax as gpx
import optax as ox
import paramax
from jax import lax
import cola
import numpy as np

def _freeze_obs_noise(likelihood):
    """Hold the observation noise fixed while the kernel hyperparameters train."""
    return eqx.tree_at(
        lambda lh: lh.obs_stddev,
        likelihood,
        replace_fn=paramax.non_trainable,
    )

@jit
def matrix_diagonal(matrix):
    assert len(matrix.shape) == 2
    batch_size = matrix.shape[0]
    matrix_size = matrix.shape[1]

    diagonal_matrix = jnp.empty((batch_size, matrix_size, matrix_size))

    def loop_body(args):
        i, diagonal_matrix = args
        return i + 1, diagonal_matrix.at[i].set(jnp.diag(matrix[i]))

    _, diagonal_matrix = lax.while_loop(lambda args: args[0] < batch_size, loop_body, (0, diagonal_matrix))

    return diagonal_matrix

@jit
def tensor_diagonal(tensor):
    assert len(tensor.shape) == 3
    dim_0 = tensor.shape[0]
    dim_1 = tensor.shape[1]
    matrix_size = tensor.shape[2]

    diagonal_tensor = jnp.empty((dim_0, dim_1, matrix_size, matrix_size))

    def loop_body_inner(args):
        i, j, diagonal_matrix = args
        return i, j + 1, diagonal_matrix.at[j].set(jnp.diag(tensor[i, j, :]))

    def loop_body_outer(args):
        i, diagonal_tensor = args
        diagonal_matrix = jnp.empty((dim_1, matrix_size, matrix_size))
        _, _, diagonal_matrix  = lax.while_loop(lambda args: args[1] < dim_1, loop_body_inner, (i, 0, diagonal_matrix))
        return i + 1, diagonal_tensor.at[i].set(diagonal_matrix) 

    _, diagonal_tensor = lax.while_loop(lambda args: args[0] < dim_0, loop_body_outer, (0, diagonal_tensor))

    return diagonal_tensor

@jit
def matrix_transpose(batched_matrix):
    assert len(batched_matrix.shape) == 3
    return jnp.transpose(batched_matrix, axes=(0, 2, 1))

@jit
def tensor_transpose(batched_tensor):
    assert len(batched_tensor.shape) == 4
    return jnp.transpose(batched_tensor, axes=(0, 1, 3, 2))

class PsuedoGPR():

    def __init__(self, x_initial, y_initial, kernel):
        self.D = gpx.Dataset(X=x_initial, y=y_initial)
        self.kernel = kernel

    @property
    def X(self):
        return self.D.X

    @property
    def y(self):
        return self.D.y

class GPR():

    def __init__(self, 
            x_initial, 
            y_initial, 
            kernel=gpx.kernels.RBF(), 
            mean=gpx.mean_functions.Zero(), 
            obs_noise=1e-6, 
            learning_rate=0.01, 
            num_iters=100, 
            batch_size=64,
            **kw):
            
        self.D = gpx.Dataset(X=x_initial, y=y_initial)
        self.noise = jnp.array(max(obs_noise, 1e-6))
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.prior = gpx.gps.Prior(mean_function=mean, kernel=kernel, jitter=1e-6)
        self.posterior = None
        self.steps = 0

    def optimize(self, key):
        likelihood = _freeze_obs_noise(
            gpx.likelihoods.Gaussian(obs_stddev=self.noise)
        )

        if self.posterior is not None:
            posterior = self.posterior.prior * likelihood
        else:
            posterior = self.prior * likelihood

        negative_mll = jit(lambda model, data: -gpx.objectives.conjugate_mll(model, data))

        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=negative_mll,
            train_data=self.D,
            optim=ox.adam(learning_rate=self.learning_rate),
            num_iters=self.num_iters,
            batch_size=self.batch_size,
            safe=True,
            key=key,
            verbose=False,
        )

        self.posterior = opt_posterior

        self.steps += 1

        return negative_mll(opt_posterior, train_data=self.D)

    def predict_at_point(self, x):
        latent_dist = self.posterior.predict(x, train_data=self.D)
        predictive_dist = self.posterior.likelihood(latent_dist)
        return predictive_dist.mean, predictive_dist.variance

    def sample_approx(self, key, num_samples=1):
        return self.posterior.sample_approx(
            num_samples=num_samples, train_data=self.D, key=key, num_features=500
        )

    @property
    def X(self):
        return self.D.X

    @property
    def y(self):
        return self.D.y

    @property
    def lengthscale(self):
        return gpx.parameters.val(self.posterior.prior.kernel.lengthscale)

    @property
    def variance(self):
        return gpx.parameters.val(self.posterior.prior.kernel.variance)

    def kernel(self, X1, X2):
        if X2 is None:
            return self.posterior.prior.kernel.gram(X1).as_matrix()
        else:
            return self.posterior.prior.kernel.cross_covariance(X1, X2)


class MGPR():


    def __init__(self, key, x_initial, y_initial, **gp_kwargs):

        assert x_initial.shape[0] == y_initial.shape[0]

        self.n_outs = y_initial.shape[1]
        self.n_ins = x_initial.shape[1]

        self.gp_kwargs = gp_kwargs

        self.models = []
        self.create_models(key, x_initial, y_initial)

    def create_models(self, key, X, Y):

        for i in range(self.n_outs):
            kernel = gpx.kernels.RBF(lengthscale=jnp.ones((self.n_ins,)))
            mean = gpx.mean_functions.Zero()
            y = jnp.expand_dims(Y[:, i], axis=1)
            model = GPR(X, y, kernel=kernel, mean=mean, **self.gp_kwargs)
            self.models.append(model)

    def add_XY(self, X_new, Y_new):
        for i in range(len(self.models)):
            y_new = jnp.expand_dims(Y_new[:, i], axis=1)
            self.models[i].D += gpx.Dataset(X=X_new, y=y_new)

    def set_XY(self, X_new, Y_new):
        for i in range(len(self.models)):
            y_new = jnp.expand_dims(Y_new[:, i], axis=1)
            self.models[i].D = gpx.Dataset(X=X_new, y=y_new)

    def optimize(self, key):
        losses = []
        for model in self.models:
            key, subkey = jr.split(key)
            loss = model.optimize(subkey)
            losses.append(loss)

        return np.mean(losses)

    def predict_at_point(self, x):
        # can probably optimize this with matrix computations rather than for loop
        mu = jnp.empty((self.n_outs,))
        cov = jnp.empty((self.n_outs,))

        for i in range(len(self.models)):
            ith_mean, ith_cov = self.models[i].predict_at_point(x)
            mu = mu.at[i].set(jnp.squeeze(ith_mean))
            cov = cov.at[i].set(jnp.squeeze(ith_cov))

        return mu, jnp.eye(self.n_outs) * cov

    def calculate_factorization(self):
        K = self.K(self.X, X2=None)
        return self._calculate_factorization(K, self.X, self.Y, self.n_outs, self.noise)

    def predict_at_noisy_input(self, mu, cov):
        # reshape mu
        mu = mu.reshape(1, -1)
        iK, beta = self.calculate_factorization()
        return self._predict_at_noisy_input(mu, cov, self.centralized_input(mu), iK, beta, self.n_outs, self.n_ins, self.lengthscales, self.variance)

    @staticmethod
    @partial(jit, static_argnames=["n_outs"])
    def _calculate_factorization(K, X, Y, n_outs, noise):
        I = jnp.tile(jnp.eye(X.shape[0]), (n_outs, 1, 1))
        (c, lower) = jax.scipy.linalg.cho_factor(K + noise[:, None, None] * I)
        iK = jax.scipy.linalg.cho_solve((c, lower), I)
        Y_ = Y.T[:, :, None]
        beta = jax.scipy.linalg.cho_solve((c, lower), Y_)[:, :, 0]
        return iK, beta

    @staticmethod
    @partial(jit, static_argnames=["n_outs", "n_ins"])
    def _predict_at_noisy_input(mu, cov, centralized_input, iK, beta, n_outs, n_ins, lengthscales, variance):

        # might need to double check the beta shape ?
        cov = jnp.tile(cov[None, None, :, :], (n_outs, n_outs, 1, 1))
        inp = jnp.tile(centralized_input[None, :, :], (n_outs, 1, 1))
        
        # calculate the delta mean and input/output covariance
        # lengthscales need to have dim (n_outs, n_ins)
        # iL needs to have dim (n_outs, n_ins, n_ins)
        iL = matrix_diagonal(1/lengthscales)
        iN = jnp.matmul(inp, iL)

        B = jnp.matmul(iL, cov[0, ...]) + jnp.eye(n_ins)

        t = matrix_transpose(jnp.linalg.solve(B, matrix_transpose(iN)))
        
        lb = jnp.exp(- jnp.sum(iN * t, -1)/2) * beta

        tiL = jnp.matmul(t, iL)

        c = variance / jnp.sqrt(jnp.linalg.det(B))

        delta = (jnp.sum(lb, -1) * c)[:, None]
        cross_cov = jnp.matmul(matrix_transpose(tiL), lb[:, :, None])[..., 0] * c[:, None]

        # calculate the predictive covariance

        R = jnp.matmul(cov, tensor_diagonal( \
            1/jnp.square(lengthscales[None, :, :]) + \
            1/jnp.square(lengthscales[:, None, :]))) + \
            jnp.eye(n_ins)

        X = inp[None, :, :, :] / jnp.square(lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :] / jnp.square(lengthscales[None, :, None, :])
        Q = jnp.linalg.solve(R, cov) / 2
        Xs = jnp.sum(jnp.matmul(X, Q) * X, -1)
        X2s = jnp.sum(jnp.matmul(X2, Q) * X2, -1)

        maha = -2 * jnp.matmul(jnp.matmul(X, Q), tensor_transpose(X2)) + \
            Xs[:, :, :, None] + X2s[:, :, None, :]

        k = jnp.log(variance)[:, None] -\
            jnp.sum(jnp.square(iN), -1)/2

        L = jnp.exp(k[:, None, :, None] + k[None, :, None, :] + maha)

        cov_delta = \
            jnp.matmul(
                jnp.matmul(
                    jnp.tile(beta[:, None, None, :], [1, n_outs, 1, 1]), L
                    ),
                jnp.tile(beta[None, :, :, None], [n_outs, 1, 1, 1])
                )[:, :, 0, 0]

        diagL = jnp.transpose(jnp.diagonal(jnp.transpose(L), axis1=-2, axis2=-1))
        
        cov_delta = cov_delta - jnp.diag(jnp.sum(jnp.multiply(iK, diagL), (1, 2)))
        cov_delta = cov_delta / jnp.sqrt(jnp.linalg.det(R))
        cov_delta = cov_delta + jnp.diag(variance)
        cov_delta = cov_delta - jnp.matmul(delta, jnp.transpose(delta))

        return delta.reshape(-1), cov_delta, jnp.transpose(cross_cov)

    def centralized_input(self, m):
        return self.X - m

    def sample_approx(self, key, num_samples=1, jitted=True):
        fns = []
        for model in self.models:
            key, subkey = jr.split(key)
            fns.append(model.sample_approx(subkey, num_samples=num_samples))

        def transition_fn(x_in):
            return jnp.concatenate([fn(x_in)[:, :, None] for fn in fns], axis=-1)

        if jitted:
            return jit(transition_fn)
        else:
            return transition_fn

    @property
    def D(self):
        return self.models[0].D

    def K(self, X1, X2=None):
        return jnp.stack([model.kernel(X1, X2) for model in self.models])

    @property
    def X(self):
        return self.models[0].X

    @property
    def lengthscales(self):
        return jnp.stack([model.lengthscale for model in self.models])

    @property
    def variance(self):
        return jnp.stack([model.variance for model in self.models])

    @property
    def Y(self):
        return jnp.concatenate([model.y for model in self.models], axis=1)

    @property
    def noise(self):
        return jnp.stack([model.noise for model in self.models])

    @property
    def N(self):
        return self.X.shape[0]

