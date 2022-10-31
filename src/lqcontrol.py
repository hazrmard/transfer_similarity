"""
Linear Quadratic Regulator control.
"""
import time
from typing import Tuple, Union

import numpy as np
import control

from systems.base import SystemEnv

DEFAULT_EPS = 5e-1
"""Time interval to use to linearize a system."""



class LQRAgent:


    def __init__(
        self, law: np.ndarray, env: SystemEnv, linearize: bool=True, eps: float=DEFAULT_EPS,
        learn_interval: int=np.inf, clip: bool=True
    ) -> None:
        self.law = law
        self.env = env
        self.linearize = linearize
        self.eps = eps
        self.learn_interval = learn_interval
        self.i = 0
        self.clip = clip
        if linearize:
            self.system = linearize_system(env, eps=self.eps)
        else:
            self.system = env.system


    def learn(self, env: SystemEnv=None, q=None, r=None, x0=None):
        if env is None and self.env is None:
            raise ValueError('No env specified.')
        env = self.env if env is None else env
        q = env.q if q is None else q
        r = env.r if r is None else r
        agent = learn_lqr(env, q, r, x0=x0)
        self.law = agent.law
        self.env = env
        if self.linearize:
            self.system = linearize_system(env, x0=x0)
        else:
            self.system = env.system


    def predict(self, state: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, float]:
        if self.i % self.learn_interval == 0:
            self.learn(self.env, x0=state)
        self.i += 1
        action = -self.law @ state
        if self.clip:
            return (np.clip(action,
                            a_min=self.env.action_space.low,
                            a_max=self.env.action_space.high),
                    1.)
        return (action, 1.)



def evaluate_lqr(law: Union[np.ndarray, LQRAgent], env, n_eval_episodes=10,
                 clip_actions=True):
    if isinstance(law, LQRAgent):
        law = law.law
    R = []
    start = time.time()
    for e in range(n_eval_episodes):
        x, u, r = [], [], []
        x.append(env.reset()[0])
        done = False
        while not done:
            action = -law @ x[-1]
            if clip_actions:
                action = np.clip(action,
                    a_min=env.action_space.low,
                    a_max=env.action_space.high
                )
            u.append(action)
            state, reward, done, _, _ = env.step(action)
            r.append(reward)
            if done: break
            x.append(state)
        R.append(sum(r))
    end = time.time()
    runtime = (end - start) / n_eval_episodes
    return np.mean(R), np.std(R), runtime



def linearize_system(sys: Union[SystemEnv, control.InputOutputSystem], x0=None, u0=None, eps=DEFAULT_EPS):
    if isinstance(sys, SystemEnv):
        sys = sys.system
    if isinstance(sys, control.LinearIOSystem):
        sys = sys
    elif isinstance(sys, control.NonlinearIOSystem):
        if x0 is None:
            x0 = np.zeros(sys.nstates)
        if u0 is None:
            u0 = np.zeros(sys.ninputs)
        sys = sys.linearize(x0, u0, eps=eps)
    return sys



def learn_lqr(env: SystemEnv, q=None, r=None, x0=None, u0=None, eps=1e-1):
    sys = linearize_system(env, x0=x0, u0=u0, eps=eps)
    q = env.q if q is None else q
    r = env.r if r is None else r
    k, *_ = control.lqr(sys, q, r)
    return LQRAgent(k, env)



def transform_lqr_policy(
    agent: LQRAgent, state_xform, action_xform, copy=True
) -> LQRAgent:
    new_law = state_xform + action_xform @ agent.law
    return LQRAgent(new_law, agent.env, agent.linearize)
