import math
import random
import statistics


def wilson_ci(k, n, z=1.96):
    """95% Wilson interval for k successes out of n."""
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def bootstrap_ci(values, n_boot=10_000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean, resampling items with replacement."""
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choices(values, k=len(values))
        means.append(statistics.mean(sample))  # mean of this sample
    means.sort()
    low = means[int(alpha / 2 * n_boot)]
    high = means[int((1 - alpha / 2) * n_boot)]  # mirror of low, from the top end
    return (low, high)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value: b and c are the two kinds of flips."""
    n = b + c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def confusion(pairs):
    """pairs: list of (label, judged) booleans. Returns tp, fp, fn, tn, tpr, tnr."""
    tp = sum(1 for label, judged in pairs if label and judged)
    fp = sum(1 for label, judged in pairs if not label and judged)
    fn = sum(1 for label, judged in pairs if label and not judged)
    tn = sum(1 for label, judged in pairs if not label and not judged)
    # None when the set has no positives (or no negatives) to measure against.
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    return tp, fp, fn, tn, tpr, tnr

