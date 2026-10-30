"""JAX-native distributions used by the actor.

This is the subset of ``masa.common.distributions`` that the actor needs:
``Categorical``, ``MultivariateNormalDiag``, and ``Independent``. Each exposes
``sample(seed=...)``, ``log_prob``, ``entropy``, and ``mode``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_LOG2PI = jnp.log(2.0 * jnp.pi)


class Distribution:
    """Base class so distributions share a common type for annotations."""


class Categorical(Distribution):
    """Categorical distribution over the last axis of ``logits``."""

    def __init__(self, logits: jnp.ndarray):
        self.log_probs = jax.nn.log_softmax(logits, axis=-1)

    def sample(self, seed) -> jnp.ndarray:
        return jax.random.categorical(seed, self.log_probs, axis=-1)

    def log_prob(self, value) -> jnp.ndarray:
        value = value.astype(jnp.int32)
        gathered = jnp.take_along_axis(self.log_probs, value[..., None], axis=-1)
        return gathered[..., 0]

    def entropy(self) -> jnp.ndarray:
        probs = jnp.exp(self.log_probs)
        terms = jnp.where(probs > 0, probs * self.log_probs, 0.0)
        return -jnp.sum(terms, axis=-1)

    def mode(self) -> jnp.ndarray:
        return jnp.argmax(self.log_probs, axis=-1)


class MultivariateNormalDiag(Distribution):
    """Diagonal-covariance multivariate normal. The event axis is the last axis."""

    def __init__(self, loc, scale_diag):
        self.loc = loc
        self.scale_diag = scale_diag

    def sample(self, seed) -> jnp.ndarray:
        noise = jax.random.normal(seed, self.loc.shape, self.loc.dtype)
        return self.loc + self.scale_diag * noise

    def log_prob(self, value) -> jnp.ndarray:
        z = (value - self.loc) / self.scale_diag
        per_dim = -0.5 * (z**2 + _LOG2PI) - jnp.log(self.scale_diag)
        return jnp.sum(per_dim, axis=-1)

    def entropy(self) -> jnp.ndarray:
        per_dim = 0.5 * (_LOG2PI + 1.0) + jnp.log(self.scale_diag)
        return jnp.sum(per_dim, axis=-1)

    def mode(self) -> jnp.ndarray:
        return self.loc


class Independent(Distribution):
    """Reinterpret the last ``reinterpreted_batch_ndims`` batch axes as the event."""

    def __init__(self, distribution: Distribution, reinterpreted_batch_ndims: int = 1):
        self.distribution = distribution
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def _reduce(self, value) -> jnp.ndarray:
        for _ in range(self.reinterpreted_batch_ndims):
            value = jnp.sum(value, axis=-1)
        return value

    def sample(self, seed) -> jnp.ndarray:
        return self.distribution.sample(seed=seed)

    def log_prob(self, value) -> jnp.ndarray:
        return self._reduce(self.distribution.log_prob(value))

    def entropy(self) -> jnp.ndarray:
        return self._reduce(self.distribution.entropy())

    def mode(self) -> jnp.ndarray:
        return self.distribution.mode()
