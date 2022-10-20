"""
Operations primarily on matrices. 
"""
import numpy as np
import scipy as sp



def ab_xform_from_pseudo_matrix(P_s: np.ndarray, P_t: np.ndarray, dt: float=1e-2):
    """
    Infer A,B matrices from pseudo matrices for source and target tasks.
    Also inver the least-squares transformation F_A, F_B between those
    matrices.

    Parameters
    ----------
    P_s : _type_
        Pseudo matrix for source task.
    P_t : np.ndarray
        Pseudo matrix for target task
    dt : _type_, optional
        Simulation time step, by default 1e-2

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A_s, B_s, A_t, B_t, F_A, F_B
    """
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