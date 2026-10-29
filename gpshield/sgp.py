import jax.numpy as jnp
import jax.random as jr
import jax
from jax import jit
import gpjax as gpx
import optax as ox
from jax import lax
import cola
from gpshield.gp import GPR, MGPR, _freeze_obs_noise, matrix_diagonal, tensor_diagonal, matrix_transpose, tensor_transpose
from functools import partial

class VGPR(GPR):

    """
    Variational gaussian process regression 
    """

    def __init__(self, 
            x_initial, 
            y_initial, 
            z_initial, 
            kernel=gpx.kernels.RBF(), 
            mean=gpx.mean_functions.Zero(), 
            obs_noise=1e-6, 
            learning_rate=0.01, 
            num_iters=100, 
            batch_size=64,
            **kw):

        
        super().__init__(x_initial, y_initial, kernel, mean, obs_noise, learning_rate, num_iters, **kw)
        self.z = z_initial
        self.prior = gpx.gps.Prior(mean_function=mean, kernel=kernel, jitter=1e-5)
        self.q = None
        

    def optimize(self, key):
        raise NotImplementedError

    def predict_at_point(self, x):
        raise NotImplementedError

    def sample_approx(self, key, num_samples=1):
        return self.q.model.sample_approx(
            num_samples=num_samples, train_data=self.D, key=key, num_features=500
        )

    @property
    def lengthscale(self):
        return gpx.parameters.val(self.q.model.prior.kernel.lengthscale)

    @property
    def variance(self):
        return gpx.parameters.val(self.q.model.prior.kernel.variance)

    def kernel(self, X1, X2):
        if X2 is None:
            return self.q.model.prior.kernel.gram(X1).as_matrix()
        else:
            return self.q.model.prior.kernel.cross_covariance(X1, X2)

    @property
    def Z(self):
        return self.z


class SGPR(VGPR):

    """
    (Collpased) stochastic sparse gaussian processes regression
    """

    def __init__(self, 
            x_initial, 
            y_initial,
            z_initial, 
            kernel=gpx.kernels.RBF(),
            mean=gpx.mean_functions.Zero(), 
            obs_noise=1e-6, 
            learning_rate=0.001, 
            num_iters=500, 
            batch_size=64,
            **kw):

        super().__init__(x_initial, y_initial, z_initial, kernel, mean, obs_noise, learning_rate, num_iters, batch_size, **kw)

    def optimize(self, key):
        likelihood = _freeze_obs_noise(
            gpx.likelihoods.Gaussian(obs_stddev=self.noise)
        )

        if self.q is not None:
            posterior = self.q.model.prior * likelihood
        else:
            posterior = self.prior * likelihood

        q = gpx.variational_families.CollapsedVariationalGaussian(
            model=posterior, inducing_inputs=self.z
        )

        elbo = jit(lambda model, data: -gpx.objectives.collapsed_elbo(model, data))

        opt_q, _ = gpx.fit(
            model=q,
            objective=elbo,
            train_data=self.D,
            optim=ox.adamw(learning_rate=self.learning_rate),
            num_iters=self.num_iters,
            batch_size=self.batch_size,
            safe=True,
            key=key,
            verbose=False,
        )

        self.q = opt_q
        self.z = gpx.parameters.val(opt_q.inducing_inputs)

        self.steps += 1

        return elbo(opt_q, self.D)

    def predict_at_point(self, x):
        latent_dist = self.q(x, train_data=self.D)
        predictive_dist = self.q.model.likelihood(latent_dist)
        return predictive_dist.mean, predictive_dist.variance

class SVGP(VGPR):
    
    """
    (Uncollapsed) sparse variational gaussian process regression
    """

    def __init__(self, 
            x_initial, 
            y_initial,
            z_initial,
            kernel=gpx.kernels.RBF(),
            mean=gpx.mean_functions.Zero(),
            obs_noise=1e-6,
            learning_rate=0.001,
            num_iters=3000,
            batch_size=64,
            **kw):

        super().__init__(x_initial, y_initial, z_initial, kernel, mean, obs_noise, learning_rate, num_iters, batch_size, **kw)

    def optimize(self, key):
        likelihood = _freeze_obs_noise(
            gpx.likelihoods.Gaussian(obs_stddev=self.noise)
        )

        if self.q is not None:
            posterior = self.q.model.prior * likelihood
        else:
            posterior = self.prior * likelihood

        q = gpx.variational_families.VariationalGaussian(model=posterior, inducing_inputs=self.z)

        negative_elbo = jit(lambda model, data: -gpx.objectives.elbo(model, data))

        schedule = ox.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=75,
            decay_steps=2000,
            end_value=0.0001,
        )

        opt_q, _ = gpx.fit(
            model=q,
            objective=negative_elbo,
            train_data=self.D,
            optim=ox.adam(learning_rate=schedule),
            num_iters=self.num_iters,
            batch_size=self.batch_size,
            safe=True,
            key=key,
            verbose=False,
        )

        self.q = opt_q
        self.z = gpx.parameters.val(opt_q.inducing_inputs)

        self.steps += 1

        return negative_elbo(opt_q, self.D)

    def predict_at_point(self, x):
        latent_dist = self.q(x, train_data=self.D)
        predictive_dist = self.q.model.likelihood(latent_dist)
        return predictive_dist.mean, predictive_dist.variance

class SMGPR(MGPR):

    def __init__(self, key, x_initial, y_initial, z_initial=None, collapsed=False, num_inducing_inputs=500, **gp_kwargs):

        self.z_initial = z_initial
        self.collapsed = collapsed
        self.num_inducing_inputs = num_inducing_inputs
        super().__init__(key, x_initial, y_initial, **gp_kwargs)

    def create_models(self, key, X, Y):

        for i in range(self.n_outs):
            kernel = gpx.kernels.RBF(lengthscale=jnp.ones((self.n_ins,)))
            mean = gpx.mean_functions.Zero()
            y = jnp.expand_dims(Y[:, i], axis=1)
            key, subkey = jr.split(key)
            if self.z_initial is None:
                z = jr.uniform(subkey, shape=(self.num_inducing_inputs, self.n_ins))
            else: 
                z = jnp.array(self.z_initial)
            if self.collapsed:
                model = SGPR(X, y, z, kernel=kernel, mean=mean, **self.gp_kwargs)
            else:
                model = SVGP(X, y, z, kernel=kernel, mean=mean, **self.gp_kwargs)
            self.models.append(model)

    def calculate_factorization(self):        
        Kmm = self.K(self.Z)
        Kmn = self.K(self.Z, self.X)
        return self._calculate_factorization(self.num_inducing_inputs, self.n_outs, Kmm, Kmn, self.Y, self.noise, self.variance, lower=True)

    @staticmethod
    @partial(jit, static_argnames=["num_inducing_inputs", "n_outs", "lower"])
    def _calculate_factorization(num_inducing_inputs, n_outs, Kmm, Kmn, Y, noise, variance, lower=True):
        I = jnp.tile(jnp.eye(num_inducing_inputs), (n_outs, 1, 1))

        # TODO change to jitter value
        KmmI = Kmm + 1e-6 * I

        L = jax.scipy.linalg.cholesky(KmmI, lower=lower)

        V = jax.scipy.linalg.solve_triangular(L, Kmn, lower=lower)

        G = variance[:, None] - jnp.sum(jnp.square(V), axis=1)
        G = jnp.sqrt(1.0 + G / noise[:, None])

        V = V / G[:, None]

        Am = jax.scipy.linalg.cholesky(jnp.matmul(V, matrix_transpose(V)) + noise[:, None, None] * I, lower=lower)

        At = jnp.matmul(L, Am)

        iAt = jax.scipy.linalg.solve_triangular(At, I, lower=lower)

        Y_ = jnp.transpose(Y)[:, :, None]

        beta = jax.scipy.linalg.solve_triangular(L, jax.scipy.linalg.cho_solve((Am, lower), jnp.matmul(V / G[:, None], Y_)), lower=lower, trans=1)[:, :, 0]
        
        iB = jnp.matmul(matrix_transpose(iAt), iAt) * noise[:, None, None]
        iK = jax.scipy.linalg.cho_solve((L, lower), I) - iB

        return iK, beta

    def centralized_input(self, m):
        return self.Z - m

    @property
    def Z(self):
        return self.models[0].Z