import numpy as np



def courant_beltrami(vs, v_lower=0.95, v_upper=1.05):
    def _courant_beltrami(v):
        return np.square(max(0, v - v_upper)) + np.square(max(0, v_lower - v))
    return np.array([_courant_beltrami(v) for v in vs])