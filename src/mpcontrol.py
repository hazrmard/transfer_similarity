from typing import Callable, Tuple
import time

import numpy as np
import torch
from tqdm.autonotebook import trange
from systems.base import SystemEnv
from mpc.mpc import GradMethods, QuadCost, MPC

# TODO: MPCAgent class with a predict() method like RL Agent



def get_dynamics_func(env: SystemEnv) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Convert an env object into a function that returns next state, given current
    state and action arrays.

    Parameters
    ----------
    env : SystemEnv
        A an environment instance which has a step method which is able to
        be called multiple times without persisting effects (i.e. a pure
        function)
    
    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        A `pure` function that returns next state from current state and action.

    """
    def F(x, u):
        x_new, _, _, _, info = env.step(u.data, from_x=x.data, persist=False)
        env.dxdt = info['dxdt']
        return torch.tensor(x_new)
    return F



def learn_mpc(
    env: SystemEnv, horizon: int, q: np.ndarray=None,
    x0: np.ndarray=None, xT: np.ndarray=None,
    state_xform: np.ndarray=None, action_xform: np.ndarray=None,
    model_env: SystemEnv=None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Use Model Predictive Control on an environment.

    Parameters
    ----------
    env : SystemEnv
        The evaluation environment. If `model_env` is None, this is also used
        for planning.
    horizon : int
        The receding horizon for planning.
    q : np.ndarray, optional
        An array of weights (or a diagonal matrix) for each element in the
        state, by default None (taken from env.q)
    x0 : np.ndarray, optional
        The initial state, by default None (taken from env.reset())
    xT : np.ndarray, optional
        The terminal/goal state, by default None (assumed to be zero vector)
    state_xform : np.ndarray, optional
        The state transformation if the policy is to be transformed, by default None
    action_xform : np.ndarray, optional
        The action transformation if the policy is to be transformed, by default None
    model_env : SystemEnv, optional
        The environment for MPC to use for planning, by default None (`env` is used)

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float]
        The states, actions encountered, and mean time of running an episode.
    """
    model_env = env if model_env is None else model_env
    n_batch=1
    n_state=model_env.observation_space.shape[0]
    n_ctrl=model_env.action_space.shape[0]
    if q is None:
        q = getattr(model_env, 'q', np.zeros(n_state, dtype=model_env.dtype))
    q = np.asarray(q, dtype=model_env.dtype)
    if q.ndim==2:
        q = np.asarray(q.diagonal())
    
    goal_weights = torch.tensor(q)
    goal_state = torch.tensor(np.zeros(n_state, dtype=env.dtype) if xT is None else xT)
    ctrl_penalty = 0.001

    q = torch.cat((
        goal_weights,
        ctrl_penalty*torch.ones(n_ctrl)
    ))
    px = -torch.sqrt(goal_weights)*goal_state
    p = torch.cat((px, torch.zeros(n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        horizon, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(horizon, n_batch, 1)

    m = MPC(
        n_state=n_state,
        n_ctrl=n_ctrl,
        T=horizon,
        u_lower=-torch.ones(horizon, n_batch, n_ctrl),
        u_upper=torch.ones(horizon, n_batch, n_ctrl),
        grad_method=GradMethods.FINITE_DIFF,
        verbose=-1,
        exit_unconverged=False,
        backprop=False
    )
    cost = QuadCost(Q,p)
    F = get_dynamics_func(model_env)
    states = [env.reset(x=x0)]
    model_env.reset(x=x0)
    actions = []
    reward = 0
    done = False
    while not done:
        init_state = torch.tensor(np.atleast_2d(states[-1]))
        nominal_states, nominal_actions, costs = \
            m(init_state, cost, F)
        for action in nominal_actions.squeeze(1):
            action = action.detach().numpy()
            if not (state_xform is None or action_xform is None):
                action = state_xform @ states[-1] + action_xform @ action
            state, r, done, _, _ = env.step(action)
            actions.append(action)
            states.append(state)
            reward += r
    states = np.asarray(states)
    actions = np.asarray(actions)
    return states, actions, reward



def evaluate_mpc(
    env: SystemEnv, horizon: int, q: np.ndarray=None, xT=None,
    state_xform: np.ndarray=None, action_xform: np.ndarray=None,
    model_env: SystemEnv=None, n_eval_episodes=10
) -> Tuple[float, float, float]:
    """
    Use Model Predictive Control on an environment.

    Parameters
    ----------
    env : SystemEnv
        The evaluation environment. If `model_env` is None, this is also used
        for planning.
    horizon : int
        The receding horizon for planning.
    q : np.ndarray, optional
        An array of weights (or a diagonal matrix) for each element in the
        state, by default None (taken from env.q)
    xT : np.ndarray, optional
        The terminal/goal state, by default None (assumed to be zero vector)
    state_xform : np.ndarray, optional
        The state transformation if the policy is to be transformed, by default None
    action_xform : np.ndarray, optional
        The action transformation if the policy is to be transformed, by default None
    model_env : SystemEnv, optional
        The environment for MPC to use for planning, by default None (`env` is used)
    n_eval_episodes: int, optional
        The number of trials to run with random initial states.

    Returns
    -------
    Tuple[float, float, float]
        The mean rewards, standard deviation, and mean runtime of episodes
    """
    rewards = []
    start = time.time()
    for ep in trange(n_eval_episodes, leave=False):
        rewards.append(learn_mpc(
            env=env, x0=None, horizon=horizon, q=q, xT=xT,
            state_xform=state_xform, action_xform=action_xform, model_env=model_env
        )[2])
    end = time.time()
    runtime = (end - start ) / n_eval_episodes
    return np.mean(rewards), np.std(rewards), runtime
