import numpy as np



def l1(vs, v_ref=1.0):
    def _l1(v):
        return np.abs( v - v_ref )
    return np.array([_l1(v) for v in vs])