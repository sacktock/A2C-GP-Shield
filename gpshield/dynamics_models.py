from gpshield.gp import MGPR
from gpshield.sgp import SMGPR

import jax.numpy as jnp
import jax.random as jr
import jax

from jax import jit
from functools import partial
from jax import lax
from scipy.optimize import minimize

import numpy as np
from scipy.spatial import ConvexHull

import gymnasium
from gpjax.distributions import GaussianDistribution

from tqdm import tqdm

import os

import time

from jax.scipy.stats import chi2, norm


def mardia_skewness_kurtosis(X, alpha=0.05):
    """
    X: jnp.ndarray of shape (M, D)
    Returns:
        skewness: scalar (Mardia's skewness)
        kurtosis: scalar (Mardia's kurtosis)
    """
    M, D = X.shape

    # Compute sample mean and covariance
    mean = jnp.mean(X, axis=0)
    X_centered = X - mean
    cov = jnp.cov(X_centered, rowvar=False)

    # Inverse covariance matrix
    cov_inv = jnp.linalg.inv(cov)

    # Mahalanobis distances
    delta = X_centered @ cov_inv @ X_centered.T

    # Mardia's kurtosis
    diags = jnp.diag(delta)
    b2_d = jnp.mean(diags**2)

    # Mardia's skewness
    b1_d = jnp.mean(delta**3)

    # Skewness test statistic
    skew_stat = M * b1_d / 6
    skew_df = (D * (D + 1) * (D + 2)) // 6
    p_skew = 1 - chi2.cdf(skew_stat, df=skew_df)
    skew_pass = p_skew > alpha

    # Kurtosis test statistic
    expected_kurt = D * (D + 2)
    kurt_std = jnp.sqrt((8 * D * (D + 2)) / M)
    kurt_stat = (b2_d - expected_kurt) / kurt_std
    p_kurt = 2 * (1 - norm.cdf(jnp.abs(kurt_stat)))
    kurt_pass = p_kurt > alpha

    # Overall normality pass if both tests pass
    overall_pass = skew_pass and kurt_pass

    return {
        "mardia_skewness": float(b1_d),
        "mardia_kurtosis": float(b2_d),
        "skew_test_stat": float(skew_stat),
        "skew_df": int(skew_df),
        "skew_p_value": float(p_skew),
        "skew_pass": bool(skew_pass),
        "kurt_test_stat": float(kurt_stat),
        "kurt_p_value": float(p_kurt),
        "kurt_pass": bool(kurt_pass),
        "overall_pass": bool(overall_pass),
    }


@jit
def check_ellipsoid_within_limits(loc, cov, limits, sigma_coefficient):
    extent = sigma_coefficient * jnp.sqrt(jnp.diag(cov))
    return jnp.all(
        (loc - extent >= limits[:, 0]) * (loc + extent <= limits[:, 1])
    ).astype(jnp.float32)

    """for j in range(d):
        extent = sigma_coefficient * jnp.sqrt(cov[j, j])
        if not ((loc[j] - extent >= limits[j, 0]) and (loc[j] + extent <= limits[j, 1])):
            return jnp.array(0.0) 

    return jnp.array(1.0)"""


@jit
def check_ellipsoid_outside_limits(loc, cov, limits, sigma_coefficient):
    extent = sigma_coefficient * jnp.sqrt(jnp.diag(cov))
    return jnp.any(
        (loc + extent <= limits[:, 0]) + (loc - extent >= limits[:, 1])
    ).astype(jnp.float32)


@jit
def check_ellipsoids_within_limits(means, covs, limits, sigma_coef=1.96, axes=[0, 2]):

    t = means.shape[0]
    axes = jnp.array(axes)
    limits = jnp.array(limits)

    assert limits.shape[0] == axes.shape[0]

    means = means[:, axes]
    covs = covs[:, axes][:, :, axes]

    def loop_body(args):
        i, within_limits = args
        return i + 1, within_limits.at[i].set(
            check_ellipsoid_within_limits(means[i], covs[i], limits, sigma_coef)
        )

    _, within_limits = lax.while_loop(
        lambda args: args[0] < t, loop_body, (0, jnp.ones(t))
    )

    return within_limits


@jit
def check_ellipsoids_outside_limits(means, covs, limits, sigma_coef=1.96, axes=[0, 2]):

    t = means.shape[0]
    axes = jnp.array(axes)
    limits = jnp.array(limits)

    assert limits.shape[0] == axes.shape[0]

    means = means[:, axes]
    covs = covs[:, axes][:, :, axes]

    def loop_body(args):
        i, outside_limits = args
        return i + 1, outside_limits.at[i].set(
            check_ellipsoid_outside_limits(means[i], covs[i], limits, sigma_coef)
        )

    _, outside_limits = lax.while_loop(
        lambda args: args[0] < t, loop_body, (0, jnp.ones(t))
    )

    return outside_limits


@jit
def check_points_is_inside_hull(A, b, X, axes):
    return jnp.all((A @ X[:, axes].T) <= -b.reshape(-1, 1), axis=0).astype(jnp.float32)


@jit
def check_points_within_limits(limits, X, axes):
    return jnp.all(
        (X[:, axes] >= limits[:, 0]) * (X[:, axes] <= limits[:, 1]), axis=1
    ).astype(jnp.float32)


@jit
def check_points_outside_limits(limits, X, axes):
    return jnp.all(
        (X[:, axes] <= limits[:, 0]) + (X[:, axes] >= limits[:, 1]), axis=1
    ).astype(jnp.float32)


@partial(jit, static_argnames=["obstacle"])
def check_all_points(X, safety_constraint, safety_axes, obstacle=False):

    M, N = X.shape[0], X.shape[1]
    safety_constraint = jnp.array(safety_constraint)
    safety_axes = jnp.array(safety_axes)

    assert safety_constraint.shape[0] == safety_axes.shape[0]

    if obstacle:
        check_points = check_points_outside_limits
    else:
        check_points = check_points_within_limits

    def loop_body(args):
        i, counts = args
        return i + 1, counts.at[:, i].set(
            1.0
            - check_points(safety_constraint, X[:, i, :], safety_axes).astype(
                jnp.float32
            )
        )

    _, counts = lax.while_loop(
        lambda args: args[0] < N, loop_body, (0, jnp.zeros((M, N), dtype=jnp.float32))
    )

    return counts


def is_inside(hull, x_new, axes):
    A = hull.equations[:, :-1]
    b = hull.equations[:, -1]
    return np.all(A @ x_new[axes] <= -b)


def rejection_sample_convex_hull(hull, min_bounds, max_bounds, num_samples):
    """Perform rejection sampling within the convex hull."""
    assert min_bounds.shape == max_bounds.shape
    dim = len(min_bounds)
    samples = []

    while len(samples) < num_samples:
        # Sample uniformly within the bounding box
        sample = np.random.uniform(
            low=min_bounds, high=max_bounds, size=max_bounds.shape
        )

        # Check if the sample is inside the convex hull
        if is_inside(hull, sample, np.arange(len(sample), like=sample)):
            samples.append(sample)

    return np.array(samples)


'''def is_ellipsoid_fully_contained(loc, cov, hull, sigma_coef=1.96):
    """
    Check if a 2-sigma ellipsoid is fully contained within a convex hull
    without sampling.

    Parameters:
    - mu: Center of the ellipsoid (1D array of size d).
    - sigma: Covariance matrix of the ellipsoid (2D array of shape (d, d)).
    - hull_points: Points defining the convex hull (2D array of shape (n_points, d)).

    Returns:
    - True if the ellipsoid is fully contained in the convex hull, False otherwise.
    """
    # Compute the inverse of the covariance matrix
    cov_inv = np.linalg.inv(cov)

    # For each facet of the convex hull, check if the ellipsoid exceeds the boundary
    for eq in hull.equations:
        normal = eq[:-1]
        offset = eq[-1]

        # Define the optimization problem to maximize A_i x - b_i
        def objective(x):
            return -np.dot(normal, x)  # Negative for maximization

        # Define the constraint  0 <= 4 - (x - mu)^T Sigma^-1 (x - mu)
        def constraint(x):
            diff = x - loc
            return sigma_coef**2 - np.dot(diff.T, np.dot(cov_inv, diff))

        cons = {'type': 'ineq', 'fun': constraint}

        # Initial guess (start at the center of the ellipsoid)
        x0 = loc

        # Solve the constrained optimization problem
        res = minimize(objective, x0, constraints=cons)

        # If the maximum value of A_i x - b_i > 0, the ellipsoid exceeds the boundary
        if res.success and -res.fun > -offset:
            return False

    return True'''


def is_ellipsoid_fully_outside(loc, cov, hull, sigma_coef=1.96, tol=1e-12, strict=True):
    """
    Return True if the sigma_coef-ellipsoid centered at `loc` with covariance `cov`
    lies fully outside the convex hull `hull` (scipy.spatial.ConvexHull).
    """

    cov = np.asarray(cov)
    loc = np.asarray(loc)

    for eq in hull.equations:
        normal = eq[:-1]
        offset = eq[-1]

        spread = np.sqrt(np.dot(normal, cov @ normal))
        min_lhs = np.dot(normal, loc) + offset - sigma_coef * spread

        if strict:
            if min_lhs > tol:
                return True
        else:
            if min_lhs >= -tol:
                return True

    return False


def is_ellipsoid_fully_inside(loc, cov, hull, sigma_coef=1.96, tol=1e-12, strict=True):
    """
    Return True if the sigma_coef-ellipsoid is fully contained in the convex hull.
    """
    cov = np.asarray(cov)
    loc = np.asarray(loc)

    for eq in hull.equations:
        normal = eq[:-1]
        offset = eq[-1]

        spread = np.sqrt(np.dot(normal, cov @ normal))
        max_lhs = np.dot(normal, loc) + offset + sigma_coef * spread

        if strict:
            if max_lhs > tol:
                return False
        else:
            if max_lhs > 0 + tol:
                return False
    return True


class GP_Dynamics:
    def __init__(
        self,
        key,
        env,
        s_initial,
        a_initial,
        ns_initial,
        sparse=False,
        obs_noise=1e-6,
        **gp_kwargs,
    ):

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        assert isinstance(self.action_space, gymnasium.spaces.Discrete) or isinstance(
            self.action_space, gymnasium.spaces.Box
        )
        assert isinstance(self.observation_space, gymnasium.spaces.Box)

        s_initial, a_initial, ns_initial = (
            jnp.array(s_initial),
            jnp.array(a_initial),
            jnp.array(ns_initial),
        )

        self.sparse = sparse

        self.obs_noise = obs_noise
        self.jitter = 1e-6

        self.mgpr_models = []

        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            for i in range(self.action_space.n):
                x_initial = s_initial[a_initial == i]
                # model deltas
                y_initial = ns_initial[a_initial == i] - x_initial
                key, subkey = jr.split(key)
                if self.sparse:
                    assert "num_inducing_inputs" in gp_kwargs
                    num_inducing_inputs = gp_kwargs["num_inducing_inputs"]
                    z_initial = np.array(
                        [
                            self.observation_space.sample()
                            for _ in range(num_inducing_inputs)
                        ]
                    )
                    mgpr_model = SMGPR(
                        subkey,
                        x_initial,
                        y_initial,
                        z_initial=z_initial,
                        obs_noise=self.obs_noise,
                        **gp_kwargs,
                    )
                else:
                    mgpr_model = MGPR(
                        subkey,
                        x_initial,
                        y_initial,
                        obs_noise=self.obs_noise,
                        **gp_kwargs,
                    )
                self.mgpr_models.append(mgpr_model)

        if isinstance(self.action_space, gymnasium.spaces.Box):
            x_initial = jnp.concatenate([s_initial, a_initial], axis=1)
            # model deltas
            y_initial = ns_initial - s_initial
            if self.sparse:
                assert "num_inducing_inputs" in gp_kwargs
                num_inducing_inputs = gp_kwargs["num_inducing_inputs"]
                z_initial = np.array(
                    [
                        np.concatenate(
                            [
                                self.observation_space.sample(),
                                self.action_space.sample(),
                            ]
                        )
                        for _ in range(num_inducing_inputs)
                    ]
                )
                mgpr_model = SMGPR(
                    key,
                    x_initial,
                    y_initial,
                    z_initial=z_initial,
                    obs_noise=self.obs_noise,
                    **gp_kwargs,
                )
            else:
                mgpr_model = MGPR(
                    key, x_initial, y_initial, obs_noise=self.obs_noise, **gp_kwargs
                )
            self.mgpr_models.append(mgpr_model)

    def add_experience(self, s_new, a_new, ns_new):
        s_new, a_new, ns_new = jnp.array(s_new), jnp.array(a_new), jnp.array(ns_new)
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            assert len(self.mgpr_models) == self.action_space.n
            for i in range(self.action_space.n):
                x_new = s_new[a_new == i]
                # model deltas
                y_new = ns_new[a_new == i] - x_new
                self.mgpr_models[i].add_XY(x_new, y_new)

        if isinstance(self.action_space, gymnasium.spaces.Box):
            assert len(self.mgpr_models) == 1
            x_new = jnp.concatenate([s_new, a_new], axis=1)
            # model deltas
            y_new = ns_new - s_new
            self.mgpr_models[0].add_XY(x_new, y_new)

    def set_experience(self, s_new, a_new, ns_new):
        s_new, a_new, ns_new = jnp.array(s_new), jnp.array(a_new), jnp.array(ns_new)

        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            assert len(self.mgpr_models) == self.action_space.n
            for i in range(self.action_space.n):
                x_new = s_new[a_new == i]
                # model deltas
                y_new = ns_new[a_new == i] - x_new
                self.mgpr_models[i].set_XY(x_new, y_new)

        if isinstance(self.action_space, gymnasium.spaces.Box):
            assert len(self.mgpr_models) == 1
            x_new = jnp.concatenate([s_new, a_new], axis=1)
            # model deltas
            y_new = ns_new - s_new
            self.mgpr_models[0].set_XY(x_new, y_new)

    def optimize(self, key):
        losses = []
        for mgpr_model in self.mgpr_models:
            key, subkey = jr.split(key)
            loss = mgpr_model.optimize(subkey)
            losses.append(loss)

        return {"gp_loss": np.mean(losses)}

    def predict_delta_at_point(self, state, action):
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            assert len(self.mgpr_models) == self.action_space.n
            assert action in self.action_space
            i = action
            x = state.reshape(1, -1)
            delta, cov = self.mgpr_models[i].predict_at_point(x)
            return delta, cov
        if isinstance(self.action_space, gymnasium.spaces.Box):
            assert len(self.mgpr_models) == 1
            x = jnp.concatenate([state, action], axis=0).reshape(1, -1)
            delta, cov = self.mgpr_models[0].predict_at_point(x)
            return delta, cov

    def sample_delta_at_point(self, key, state, action):
        delta, cov = self.predict_delta_at_point(state, action)
        sample = jr.multivariate_normal(key, delta, cov)
        return sample

    def sample_trans(self, key, state, agent):
        mu = state
        key, subkey = jr.split(key)
        action = agent.act(key, mu)
        delta = self.sample_delta_at_point(subkey, mu, action)
        return mu + delta

    def sample_traj(self, key, state, agent, N, M):
        mus = jnp.empty(
            (
                M,
                N + 1,
            )
            + state.shape
        )
        mus = mus.at[:, 0, :].set(jnp.tile(state.reshape(1, -1), (M, 1)))
        for i in range(0, M):
            mu = state
            for j in range(0, N):
                key, subkey = jr.split(key)
                mu = self.sample_trans(subkey, mu, agent)
                mus = mus.at[i, j + 1].set(mu)
        return mus

    def predict_naive(self, key, state, agent, N):
        mu = state
        mus = jnp.empty((N + 1,) + state.shape)
        mus = mus.at[0].set(state)
        covs = jnp.empty((N + 1,) + (state.shape[0], state.shape[0]))
        covs = covs.at[0].set(jnp.eye(state.shape[0]) * self.obs_noise)
        for i in range(0, N):
            key, subkey = jr.split(key)
            mu, cov = self.propagate_naive(subkey, mu, agent)
            mus = mus.at[i + 1].set(mu)
            covs = covs.at[i + 1].set(cov)
        return mus, covs

    def propagate_naive(self, key, state, agent):
        mu = state
        action = agent.act(key, mu)
        delta, cov = self.predict_delta_at_point(mu, action)
        mu = mu + delta
        return mu, cov

    def predict(self, mu, cov, agent, N):
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            assert NotImplementedError
        if isinstance(self.action_space, gymnasium.spaces.Box):
            assert len(self.mgpr_models) == 1
            mus = jnp.empty((N + 1,) + mu.shape)
            covs = jnp.empty((N + 1,) + cov.shape)
            mus = mus.at[0].set(mu)
            covs = covs.at[0].set(cov)

            for i in range(0, N):
                mu, cov = self.propagate(mu, cov, agent)
                mus = mus.at[i + 1].set(mu)
                covs = covs.at[i + 1].set(cov)

            return mus, covs

    def propagate(self, mu, cov, agent):
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            assert NotImplementedError
        if isinstance(self.action_space, gymnasium.spaces.Box):
            mu_act, cov_act, cross_cov_act = agent.propagate(mu, cov)
            x = jnp.concatenate([mu, mu_act], axis=0)
            cov_1 = jnp.concatenate([cov, jnp.matmul(cov, cross_cov_act)], axis=1)
            cov_2 = jnp.concatenate(
                [jnp.transpose(jnp.matmul(cov, cross_cov_act)), cov_act], axis=1
            )
            cov_x = jnp.concatenate([cov_1, cov_2], axis=0)
            # expecting output shapes delta: (n_outs,), cov_delta: (n_outs, n_outs), cov_delta: (n_ins, n_outs)
            delta, cov_delta, cross_cov_delta = self.mgpr_models[
                0
            ].predict_at_noisy_input(x, cov_x)

            mu = mu + delta
            cov = self._compute_full_covariance(
                cov, cov_delta, cross_cov_delta, cov_1, self.jitter
            )

            return mu, cov

    @staticmethod
    @jit
    def _compute_full_covariance(cov, cov_delta, cross_cov_delta, cov_1, jitter):
        cov = cov_delta + cov + cov_1 @ cross_cov_delta + cross_cov_delta.T @ cov_1.T
        cov_diagonal = jnp.diag(cov)
        modified_diagonal = jnp.where(cov_diagonal < 0, jitter, cov_diagonal)
        return cov.at[jnp.diag_indices_from(cov)].set(modified_diagonal)

    def sample_approx(self, key, num_samples=1, jitted=True):
        if isinstance(self.action_space, gymnasium.spaces.Discrete):
            assert NotImplementedError
        if isinstance(self.action_space, gymnasium.spaces.Box):
            assert len(self.mgpr_models) == 1
            transition_fn = self.mgpr_models[0].sample_approx(
                key, num_samples=num_samples, jitted=jitted
            )
        return transition_fn

    def shield(
        self,
        key,
        state,
        agent,
        safe_agent,
        N,
        safety_constraints,
        safety_axes,
        stable_set,
        stable_axes,
        sigma_coef=2.58,
        method="gaussian_approx",
        num_samples=10,
    ):
        if method == "gaussian_approx":
            # mu, cov = self.propagate(state, jnp.eye(state.shape[0]) * self.obs_noise, agent)
            key, subkey = jr.split(key)
            # mu, cov = self.propagate_naive(subkey, state, agent)
            mu = state
            action = agent.act(subkey, state)
            delta, cov = self.predict_delta_at_point(mu, action)
            mu = mu + delta

            mus, covs = self.predict(mu, cov, safe_agent, N)
            mus = jnp.concatenate([mu.reshape(1, -1), mus], axis=0)
            covs = jnp.concatenate([jnp.expand_dims(cov, axis=0), covs], axis=0)

            constraint_satisfaction = True

            # check constraint satisfaction for the safety constraints
            if "obstacles" in safety_constraints.keys():
                for constraint in safety_constraints["obstacles"]:
                    if isinstance(constraint, np.ndarray):
                        assert constraint.shape == (safety_axes.shape[0], 2)
                        outside_safety_constraint = check_ellipsoids_outside_limits(
                            mus,
                            covs,
                            constraint,
                            axes=safety_axes,
                            sigma_coef=sigma_coef,
                        )
                    elif isinstance(constraint, ConvexHull):
                        outside_safety_constraint = np.array(
                            [
                                is_ellipsoid_fully_outside(
                                    mus[i, stable_axes],
                                    covs[i, stable_axes][:, stable_axes],
                                    stable_set,
                                    sigma_coef=sigma_coef,
                                )
                                for i in range(N + 1)
                            ],
                            dtype=float64,
                        )
                    else:
                        raise NotImplementedError
                    constraint_satisfaction = (
                        constraint_satisfaction
                        and jnp.all(outside_safety_constraint == 1.0).item()
                    )
            if "boundaries" in safety_constraints.keys():
                for constraint in safety_constraints["boundaries"]:
                    if isinstance(constraint, np.ndarray):
                        assert constraint.shape == (safety_axes.shape[0], 2)
                        within_safety_constraint = check_ellipsoids_within_limits(
                            mus,
                            covs,
                            constraint,
                            axes=safety_axes,
                            sigma_coef=sigma_coef,
                        )
                    elif isinstance(constraint, ConvexHull):
                        within_safety_constraint = np.array(
                            [
                                is_ellipsoid_fully_inside(
                                    mus[i, stable_axes],
                                    covs[i, stable_axes][:, stable_axes],
                                    stable_set,
                                    sigma_coef=sigma_coef,
                                )
                                for i in range(N + 1)
                            ],
                            dtype=float64,
                        )
                    else:
                        raise NotImplementedError
                    constraint_satisfaction = (
                        constraint_satisfaction
                        and jnp.all(within_safety_constraint == 1.0).item()
                    )

            # check final ellipsoid lies in the control invariant
            if isinstance(stable_set, np.ndarray):
                assert stable_set.shape == (stable_axes.shape[0], 2)
                stable_axes = jnp.array(stable_axes)
                stable_set = jnp.array(stable_set)
                loc = mus[-1, stable_axes]
                cov = covs[-1, stable_axes][:, stable_axes]
                within_stable_set = (
                    check_ellipsoid_within_limits(
                        loc, cov, stable_set, sigma_coef
                    ).item()
                    == 1.0
                )
            elif isinstance(stable_set, ConvexHull):
                within_stable_set = is_ellipsoid_fully_inside(
                    mus[-1, stable_axes],
                    covs[-1, stable_axes][:, stable_axes],
                    stable_set,
                    sigma_coef=sigma_coef,
                )
            else:
                raise TypeError(
                    f"Expected a numpy.ndarray or scipy.spatial.ConvexHull, but got {type(stable_set)} instead."
                )

            override = (not constraint_satisfaction) or (not within_stable_set)

            """if override:
                outfile = f"shield_ellipses_override_{t}.pdf"
            else:
                outfile = f"shield_ellipses_{t}.pdf"
                
            plot_ellipses_and_hull(mus, covs, stable_set, axes=stable_axes, x_limits=[-1.2, 0.6], y_limits=[-0.07, 0.07], fit_to_limits=True, x_name="car pos", y_name="car vel", sigma_coef=sigma_coef, outfile=outfile)"""

            if override:
                return safe_agent.policy(None, state), True
            else:
                return action, False
        elif method == "posterior_sample_approx":
            M = num_samples
            mus = jnp.empty(
                (
                    M,
                    N + 1,
                )
                + state.shape
            )

            epsilon_t = 0.995

            key, subkey = jr.split(key)
            transition_fn = self.sample_approx(subkey, num_samples=M, jitted=False)

            mu = jnp.tile(state.reshape(1, -1), (M, 1))
            action = agent.select_action(agent.actor_state, mu)
            action = jnp.clip(action, self.action_space.low, self.action_space.high)

            x = jnp.concatenate([mu, action], axis=-1)
            delta = jnp.squeeze(transition_fn(x))[jnp.arange(M), jnp.arange(M), :]

            mu = jnp.tile(state.reshape(1, -1), (M, 1)) + delta
            mus = mus.at[:, 0].set(mu)

            for j in tqdm(range(N)):
                key, subkey = jr.split(key)
                transition_fn = self.sample_approx(subkey, num_samples=M, jitted=False)
                act = safe_agent.act_dim(mu)
                x = jnp.concatenate([mu, act], axis=-1)
                delta = jnp.squeeze(transition_fn(x))[jnp.arange(M), jnp.arange(M), :]
                mu = mu + delta
                mus = mus.at[:, j + 1].set(mu)

            safe_counts = np.ones_like(mus)

            if "obstacles" in safety_constraints.keys():
                for constraint in safety_constraints["obstacles"]:
                    if isinstance(constraint, np.ndarray):
                        safe_counts *= np.asarray(
                            check_all_points(
                                mus, constraint, safety_axes, obstacle=True
                            )
                        )
                    else:
                        raise NotImplementedError
                    # sampling based approximation for convex hull checks is not currently implemented
            if "boundaries" in safety_constraints.keys():
                for constraint in safety_constraints["boundaries"]:
                    if isinstance(constraint, np.ndarray):
                        safe_counts *= np.asarray(
                            check_all_points(
                                mus, constraint, safety_axes, obstacle=False
                            )
                        )
                    else:
                        raise NotImplementedError
                    # sampling based approximation for convex hull checks is not currently implemented

            safe_probs = np.mean(safe_counts, axis=0)

            if isinstance(stable_set, ConvexHull):
                A = stable_set.equations[:, :-1]
                b = stable_set.equations[:, -1]
                stable_axes = jnp.array(stable_axes)
                A = jnp.array(A)
                b = jnp.array(b)
                assert A.shape[0] == b.shape[0]
                assert A.shape[1] == stable_axes.shape[0]
                stable_probs = jnp.mean(
                    check_points_is_inside_hull(A, b, mus[:, -1, :], stable_axes)
                )
            else:
                assert stable_set.shape == (stable_axes.shape[0], 2)
                stable_limits = jnp.array(stable_set)
                stable_axes = jnp.array(stable_axes)
                assert stable_limits.shape[0] == stable_axes.shape[0]
                stable_probs = jnp.mean(
                    check_points_within_limits(
                        stable_limits, mus[:, -1, :], stable_axes
                    ).astype(jnp.float32)
                )

            override = (stable_probs > 1.0 - epsilon_t) and not jnp.all(
                safe_probs <= 1.0 - epsilon_t
            )

            if override:
                return safe_agent.policy(None, state), True
            else:
                return action, False

        elif method == "posterior_sample_approx_skew_and_kurt":
            M = num_samples
            mus = jnp.empty(
                (
                    M,
                    N + 1,
                )
                + state.shape
            )

            epsilon_t = 0.995

            key, subkey = jr.split(key)
            transition_fn = self.sample_approx(subkey, num_samples=M, jitted=False)

            mu = jnp.tile(state.reshape(1, -1), (M, 1))
            action = agent.select_action(agent.actor_state, mu)
            action = jnp.clip(action, self.action_space.low, self.action_space.high)

            x = jnp.concatenate([mu, action], axis=-1)
            delta = jnp.squeeze(transition_fn(x))[jnp.arange(M), jnp.arange(M), :]

            mu = jnp.tile(state.reshape(1, -1), (M, 1)) + delta
            mus = mus.at[:, 0].set(mu)

            for j in tqdm(range(N)):
                key, subkey = jr.split(key)
                transition_fn = self.sample_approx(subkey, num_samples=M, jitted=False)
                act = safe_agent.act_dim(mu)
                x = jnp.concatenate([mu, act], axis=-1)
                delta = jnp.squeeze(transition_fn(x))[jnp.arange(M), jnp.arange(M), :]
                mu = mu + delta
                mus = mus.at[:, j + 1].set(mu)

            for j in range(N + 1):
                X = mus[:, j, :]
                results = mardia_skewness_kurtosis(X)

                print(
                    "$t={}$ & {} & {:.4f} & {:.4f} & {} & {:.4f} & {:.4f} & {} \\\\".format(
                        j,
                        "Pass" if results["overall_pass"] else "Fail",
                        results["mardia_skewness"],
                        results["skew_p_value"],
                        "Pass" if results["skew_pass"] else "Fail",
                        results["mardia_kurtosis"],
                        results["kurt_p_value"],
                        "Pass" if results["kurt_pass"] else "Fail",
                    )
                )

            return action, False

        else:
            raise NotImplementedError
