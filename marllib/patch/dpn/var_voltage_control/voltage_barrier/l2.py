import numpy as np



def l2(vs, v_ref=1.0):
    def _l2(v):
        return 2 * np.square(v - v_ref)
    return np.array([_l2(v) for v in vs])