import numpy as np


def generate_mass_spring_data(k, m, x0, v0, dt, num_steps):
    
    w = np.sqrt(k/m)
    t = np.arange(num_steps) * dt
    x = x0 * np.cos(w * t) + v0 / w * np.sin(w * t)
    v = -x0 * w * np.sin(w * t) + v0 * np.cos(w * t)
    return x, v
