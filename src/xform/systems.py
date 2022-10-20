"""
Operations primarily on systems and environments.
"""

from typing import Union

import numpy as np
import control

from .matrices import ab_xform_from_pseudo_matrix



def policy_transform(sys: control.LinearIOSystem, xformA=None, xformB=None, ctrl_law=None):
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
        # which means
        # u = (-state_xform) @ x + action_xform @ (-law @ x)
        return -state_xform, action_xform



def pseudo_matrix(sys: control.LinearIOSystem, dt=1e-2):
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



def get_env_samples(
    env, n, control_law=None, n_episodes_or_steps='episodes'
):
    if isinstance(control_law, np.ndarray):
        policy = lambda x: -control_law @ x
    elif control_law is None:
        policy = lambda x: env.action_space.sample()
    else:
        policy = lambda x: control_law.predict(x, deterministic=True)[0]
    xu, x = [], []
    i = 0
    while i < n:
        state = env.reset()[0]
        done = False
        while not done and i < n:
            action = policy(state)
            nstate, _, done, _, info = env.step(action)
            if n_episodes_or_steps == 'steps':
                i += 1
            xu.append(np.concatenate((state, info.get('u', action))))
            x.append(nstate)
            state = nstate
            if done:
                if n_episodes_or_steps == 'episodes':
                    i += 1
                break
    return (np.asarray(xu, dtype=np.float32).T,
           np.asarray(x, dtype=np.float32).T)



def get_transforms(
    agent, env, env_,
    buffer_episodes=5,
    xformA=None, xformB=None,
    data_driven_source=True,
    x0=None, u0=None,
):
    # linearize a non-linear system to get A,B,C,D representation,
    # and the resulting pseudo matrix P_s of the source task
    if hasattr(env, 'system'):
        if isinstance(env.system, control.LinearIOSystem):
            _sys_linear = env.system
        elif isinstance(env.system, control.NonlinearIOSystem):
            if x0 is None:
                x0 = env.observation_space.sample() * 0
            if u0 is None:
                u0 = env.action_space.sample() * 0
            _sys_linear = env.system.linearize(x0, u0)
    else:
        # Learn environment model from data
        data_driven_source = True
    # get the pseudo matrix representing source system dynamics
    if data_driven_source:
        xu, x = get_env_samples(env, buffer_episodes, agent)
        P_s = (x @ xu.T) @ np.linalg.pinv(xu @ xu.T)
    else:
        P_s = pseudo_matrix(_sys_linear, env.dt)
    # get pseudo matrix representing target system dynamics
    xu, x = get_env_samples(env_, buffer_episodes, agent)
    P_t = (x @ xu.T) @ np.linalg.pinv(xu @ xu.T)
    # get the relationship between source and target systems
    A_s, B_s, A_t, B_t, F_A, F_B = ab_xform_from_pseudo_matrix(P_s, P_t, env.dt)
    C_s, D_s = np.eye(len(A_s)), np.zeros_like(B_s)
    if xformA is not None:
        F_A = xformA
    if xformB is not None:
        F_B = xformB
    # generate policy transforms from the source system,
    # and its relationship to the target system
    if data_driven_source:
        source_system = control.ss(A_s, B_s, C_s, D_s)
    else:
        source_system = _sys_linear
    state_xform, action_xform = policy_transform(source_system, F_A, F_B)
    return state_xform, action_xform, F_A, F_B



def is_controllable(sys: control.LinearIOSystem):
    #https://www.mathworks.com/help/control/ref/ss.ctrb.html
    return np.linalg.matrix_rank(control.ctrb(sys.A, sys.B)) - \
           len(sys.A) == 0