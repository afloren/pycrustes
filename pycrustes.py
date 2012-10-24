
import numpy as np

def solve(A, B):
    S = np.dot(A.T, B)
    W, D, Vt = np.linalg.svd(S)
    T = np.dot(W, Vt)
    Bp = np.dot(A, T)
    E = B - Bp
    return (T,Bp,E)
