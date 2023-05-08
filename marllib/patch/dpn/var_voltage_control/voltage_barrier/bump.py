import numpy as np



def bump(vs):
    def _bump(v):
        if np.abs(v) < 1:
            return np.exp( - 1 / (1 - v**4) )
        elif 1 < v < 3:
            return np.exp( - 1 / (1 - ( v - 2 )**4 ) )
        else:
            return 0.0
    return np.array([_bump(v) for v in vs])