import numpy as np

def add_gaussian_noise(data, mean, std_dev):
    """Adds Gaussian noise to a NumPy array."""
    noise = np.random.normal(mean, std_dev, data.shape)
    return data + noise