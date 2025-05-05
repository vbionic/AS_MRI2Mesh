#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import polar
#----------------------------------------------------------------------------
def H4x4_decomposition(H):
    # Translation
    T4x4 = np.eye(4)
    T4x4[:3,3] = H[:3,3]
    T = H[:3,3]

    # L = RK
    L = H.copy()
    L[:3,3] = 0

    R4x4, K4x4 = polar(L)
    if np.linalg.det(R4x4) < 0:
        R4x4[:3,:3] = -R4x4[:3,:3]
        K4x4[:3,:3] = -K4x4[:3,:3]
    if K4x4[0,0] < 0:
        R4x4[:3,:3] = -R4x4[:3,:3]
        K4x4[:3,:3] = -K4x4[:3,:3]
    R3x3 = R4x4[:3,:3]
    K3x3 = K4x4[:3,:3]
    
    f, X = np.linalg.eig(K4x4)
    S = []
    for factor, axis in zip(f, X.T):
        if not np.isclose(factor, 1):
            scale = np.eye(4) + np.outer(axis, axis) * (factor-1)
            S.append(scale)

    return {"T4x4":T4x4,
            "T3":T,
            "R4x4":R4x4,
            "R3x3":R3x3,
            "K4x4":K4x4,
            "K3x3":K3x3,
            "S":S,
            "_L":L}

def H4x4_composition(T4x4=None, T3=None, R4x4=None, R3x3=None, K4x4=None, K3x3=None):
    if(T4x4 is None):
        T4x4 = np.eye(4)
    if not T3 is None:
        T4x4[:3,3] = T3
        
    if(R4x4 is None):
        R4x4 = np.eye(4)
    if not R3x3 is None:
        R4x4[:3,:3] = R3x3
        
    if(K4x4 is None):
        K4x4 = np.eye(4)
    if not K3x3 is None:
        K4x4[:3,:3] = K3x3
        
    return np.array(T4x4) @ R4x4 @ K4x4 