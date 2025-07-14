import numpy as np

def calc_hpdi(samples, prob=0.95):
    """Calculate the highest posterior density interval (HPDI) for a set of samples."""
    n = len(samples)
    interval_size = int(np.floor(prob * n))
    sorted_samples = np.sort(samples)
    intervals = np.array([sorted_samples[i:i + interval_size] for i in range(n - interval_size + 1)])
    interval_widths = intervals[:, -1] - intervals[:, 0]
    min_width_index = np.argmin(interval_widths)
    hpdi = intervals[min_width_index]
    return hpdi