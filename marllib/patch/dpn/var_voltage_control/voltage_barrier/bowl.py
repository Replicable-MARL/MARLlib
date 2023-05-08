import numpy as np



def bowl(vs, v_ref=1.0, scale=.1):
    def normal(v, loc, scale):
        return 1 / np.sqrt(2 * np.pi * scale**2) * np.exp( - 0.5 * np.square(v - loc) / scale**2 )
    def _bowl(v):
        if np.abs(v-v_ref) > 0.05:
            return 2 * np.abs(v-v_ref) - 0.095
        else:
            return - 0.01 * normal(v, v_ref, scale) + 0.04
    return np.array([_bowl(v) for v in vs])