from typing import Callable

import scipy as sp
import numpy as np
import control


def policy_transform(sys, xformA=None, xformB=None, ctrl_law=None):
    A, B, K = sys.A, sys.B, ctrl_law
    F_A = np.eye(len(A)) if xformA is None else xformA
    F_B = np.eye(len(A)) if xformB is None else xformB
    I = np.eye(len(A))
    F_B_B_ = np.linalg.pinv(F_B @ B)
    
    state_xform = (F_B_B_@(F_A-I)@A)
    action_xform = (F_B_B_@B)
    if ctrl_law is not None:
        # return K_F such that u = -K_F x
        K_F = state_xform + action_xform @ ctrl_law
        return K_F
    else:
        # return state, action transforms such that
        # u = K_x @ x + K_u @ u
        # u = (-state_xform) @ x + action_xform @ u
        return -state_xform, action_xform



def pseudo_matrix(sys, dt=1e-2):
    """
    The matrix [A.dt + I, B.dt] to be multiplied with [x;u] to give the next state
    """
    assert sys.A.shape[0]==sys.B.shape[0], 'State size of A=/=B matrix'
    nstates, nactions = sys.A.shape[0], sys.B.shape[1]
    shape = (nstates, nstates + nactions)
    m = np.zeros(shape, dtype=np.float32)
    m[:, :nstates] = sys.A * dt + np.eye(nstates)
    m[:, nstates:] = sys.B * dt
    return m



def ab_xform_from_pseudo_matrix(P_s, P_t, dt=1e-2):
    nstates = P_s.shape[0]
    nactions = P_s.shape[1] - nstates
    # A -> P = [I + dt.A, dt.B] -> A = (P - I) / dt
    A_s = (P_s[:, :nstates] - np.eye(nstates)) / dt
    A_t = (P_t[:, :nstates] - np.eye(nstates)) / dt
    # A_t = F_A A_s
    # F_A = A_t @ np.linalg.pinv(A_s)
    F_A = (A_t @ A_s.T) @ np.linalg.pinv(A_s @ A_s.T)
    # B = P / dt
    B_s = P_s[:, nstates:] / dt
    B_t = P_t[:, nstates:] / dt
    # B_t = F_B B_s
    F_B = (B_t @ B_s.T) @ np.linalg.pinv(B_s @ B_s.T)
    return A_s, B_s, A_t, B_t, F_A, F_B



def basis_vectors(mat):
    # https://stackoverflow.com/a/42868363/4591810
    _, _, upper = sp.linalg.lu(mat)
    idx = [np.nonzero(upper[a])[0][0] for a in range(len(upper))]
    basis = mat[:, list(set(idx))]
    norms = np.linalg.norm(basis, axis=0)
    return basis / norms



def is_controllable(sys):
    #https://www.mathworks.com/help/control/ref/ss.ctrb.html
    return np.linalg.matrix_rank(control.ctrb(sys.A, sys.B)) - \
           len(sys.A) == 0



def get_env_samples(env, n, control_law):
    if isinstance(control_law, np.ndarray):
        policy = lambda x: -control_law @ x
    else:
        policy = lambda x: control_law.predict(x)[0]
    xu, x = [], []
    for e in range(n):
        state = env.reset()
        done = False
        while not done:
            action = policy(state)
            nstate, _, done, info = env.step(action)
            xu.append(np.concatenate((state, info.get('u', action))))
            x.append(nstate)
            state = nstate
            if done: break
    return (np.asarray(xu, dtype=np.float32).T,
           np.asarray(x, dtype=np.float32).T)



class InvertibleFunction:
    """Use as InvertibleFunction() @ and InvertibleFunction.invert() @"""
    def __init__(self, func: Callable, inverse: Callable):
        self.func, self.inverse = func, inverse
    def __call__(self, *args):
        return self.func(*args)
    def __matmul__(self, *args):
        return self(*args)
    def invert(self):
        return InvertibleFunction(self.inverse, self.func)