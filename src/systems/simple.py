"""
\begin{align}
\dot{y_1} &= a \cdot x + b \cdot u(t) \\
&= [a]y + [b] u(t)
\end{align}
"""
import numpy as np
import control
import gym

from .base import SystemEnv



def create_simple(a, b, name='simple'):
    A = np.asarray([[a]], dtype=np.float32)
    B = np.asarray([[b]], dtype=np.float32)
    C = np.eye(1, dtype=np.float32)
    D = np.zeros((1,1), dtype=np.float32)
    return control.ss(A, B, C, D, name=name)



class SimpleEnv(SystemEnv):
    def __init__(self, a, b, q=1, r=0, dt=0.01, seed=None):
        self.a = a
        self.b = b
        system = create_simple(a, b)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.period = int((1/np.abs(a)) / dt)
        super().__init__(system, q, r, dt, seed)
    def reset(self, x=None):
        super().reset(x)
        self.x = (np.asarray(x) or \
             self.random.uniform(
            -1/(2*np.abs(self.a)),
             1/(2*np.abs(self.a)),
             size=(1,)
            )).astype(np.float32)
        return self.x, {}
    def step(self, u: np.ndarray, from_x=None, persist=True):
        x, r, d, _, i = super().step(u, from_x=from_x, persist=persist)
        return x, r, self.n > self.period, False, i