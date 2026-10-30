import numpy as np


class Summary_Stats:
    def __init__(self, prefix=""):
        self.n = 0
        self.prefix = prefix
        self.stats = None

    def update(self, values):
        if self.stats is None:
            self.stats = {
                "mean": np.mean(values),
                "mean_squares": np.mean(values**2),
                "max": np.max(values),
                "min": np.min(values),
                "mag": np.max(np.abs(values)),
            }
            self.n = 1
        else:
            self.stats = {
                "mean": np.mean(values) * (1 / self.n)
                + self.stats["mean"] * ((self.n - 1) / self.n),
                "mean_squares": np.mean(values**2) * (1 / self.n)
                + self.stats["mean_squares"] * ((self.n - 1) / self.n),
                "max": max(np.max(values), self.stats["max"]),
                "min": min(np.min(values), self.stats["min"]),
                "mag": max(np.max(np.abs(values)), self.stats["mag"]),
            }
            self.n += 1

    def get(self):
        stats = {
            "mean": self.stats["mean"],
            "std": np.sqrt(self.stats["mean_squares"] - self.stats["mean"] ** 2),
            "max": self.stats["max"],
            "min": self.stats["min"],
            "mag": self.stats["mag"],
        }
        if self.prefix:
            stats = {f"{self.prefix}_{k}": v for k, v in stats.items()}
        return stats

    def __add__(self, other):
        assert isinstance(other, Summary_Stats)

        assert self.prefix == other.prefix, (
            "can't add two summary stats objects with different prefixes"
        )

        new = Summary_Stats(prefix=self.prefix)
        new.n = self.n + other.n

        new.stats = {
            "mean": self.stats["mean"] * (self.n / new.n)
            + other.stats["mean"] * (other.n / new.n),
            "mean_squares": self.stats["mean_squares"] * (self.n / new.n)
            + other.stats["mean_squares"] * (other.n / new.n),
            "max": max(self.stats["max"], other.stats["max"]),
            "min": min(self.stats["min"], other.stats["min"]),
            "mag": max(self.stats["mag"], other.stats["mag"]),
        }

        return new
