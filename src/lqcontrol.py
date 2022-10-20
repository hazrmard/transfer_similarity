"""
Linear Quadratic Regulator control.
"""
import time
from typing import Tuple, Union

import numpy as np
import control

from systems.base import SystemEnv



class LQRAgent:


    def __init__(self, law: np.ndarray, env, linearize: bool=False) -> None:
        self.law = law
        self.env = env
        self.linearize = linearize
        if linearize:
            self.system = linearize_system(env)
        else:
            self.system = env.system


    def learn(self, env: SystemEnv, q, r):
        agent = learn_lqr(env, q, r)
        self.law = agent.law
        self.env = env
        if self.linearize:
            self.system = linearize_system(env)
        else:
            self.system = env.system


    def predict(self, state: np.ndarray, *args, **kwargs) -> Tuple[np.ndarray, float]:
        return (np.clip(-self.law @ state,
                        a_min=self.env.action_space.low,
                        a_max=self.env.action_space.high),
                1.)



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



def linearize_system(sys: Union[SystemEnv, control.InputOutputSystem], x0=None, u0=None):
    if isinstance(sys, SystemEnv):
        sys = sys.system
    if isinstance(sys, control.LinearIOSystem):
        sys = sys
    elif isinstance(sys, control.NonlinearIOSystem):
        if x0 is None:
            x0 = np.zeros(sys.nstates)
        if u0 is None:
            u0 = np.zeros(sys.ninputs)
        sys = sys.linearize(x0, u0)
    return sys



def learn_lqr(env: SystemEnv, q=None, r=None):
    sys = linearize_system(env)
    q = env.q if q is None else q
    r = env.r if r is None else r
    k, *_ = control.lqr(sys, q, r)
    return LQRAgent(k, env)



def transform_lqr_policy(
    agent: LQRAgent, state_xform, action_xform, copy=True
) -> LQRAgent:
    new_law = state_xform + action_xform @ agent.law
    return LQRAgent(new_law, agent.env, agent.linearize)
