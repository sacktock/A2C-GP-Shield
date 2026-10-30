import numpy as np

COMPUTE_DTYPE = np.float64


class Moments:
    def __init__(
        self, impl="mean_std", decay=0.01, max=1e8, perclo=5, perchi=95, eps=0.0
    ):
        self.impl = impl
        self.decay = decay
        self.max = max
        self.eps = eps
        self.perclo = perclo
        self.perchi = perchi

        self.step = np.array(0, dtype=COMPUTE_DTYPE)

        if self.impl == "none":
            pass
        elif self.impl == "mean_std":
            self.mean = np.array(0.0, dtype=COMPUTE_DTYPE)
            self.sqrs = np.array(0.0, dtype=COMPUTE_DTYPE)
        elif self.impl == "perc_ema":
            self.low = np.array(0.0, dtype=COMPUTE_DTYPE)
            self.high = np.array(0.0, dtype=COMPUTE_DTYPE)
        else:
            raise NotImplementedError(self.impl)

    def __call__(self, x):
        self.update(x)
        return self.norm()

    def update(self, x):
        self.step += 1
        x = np.array(x, dtype=COMPUTE_DTYPE)
        m = np.maximum(self.decay, 1 / self.step)
        if self.impl == "none":
            pass
        elif self.impl == "mean_std":
            self.mean = m * np.mean(x) + (1 - m) * self.mean
            self.sqrs = m * np.mean(x * x) + (1 - m) * self.sqrs
        elif self.impl == "perc_ema":
            low, high = np.percentile(x, self.perclo), np.percentile(x, self.perchi)
            self.low = m * low + (1 - m) * self.low
            self.high = m * high + (1 - m) * self.high
        else:
            raise NotImplementedError(self.impl)

    def norm(self):
        if self.impl == "none":
            return 0.0, 1.0
        elif self.impl == "mean_std":
            corr = 1 - (1 - self.decay) ** self.step.astype(COMPUTE_DTYPE)
            mean = self.mean / corr
            var = self.sqrs / corr - self.mean**2
            std = np.sqrt(np.maximum(var, 1 / self.max**2) + self.eps)
            return mean, std
        elif self.impl == "perc_ema":
            offset = self.low
            invscale = np.maximum(1 / self.max, self.high - self.low)
            return offset, invscale
        else:
            raise NotImplementedError(self.impl)
