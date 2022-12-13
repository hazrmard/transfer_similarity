"""
$$
m\ddot{x} = -kx -k_f\dot{x} + u(t)
$$

This is a 2nd order ODE. Linearizing it, by letting $y_1 = x, y_2 = \dot{x}$. Then:

\begin{align}
\dot{y}_1 &= 0y_1 + y_2 \\
\dot{y}_2 &= -(k/m)y_1 - (k_f/m)y_2  + u(t)/m
\end{align}

In state space form:

$$
\begin{bmatrix}\dot{y}_1 \\ \dot{y}_2 \end{bmatrix} = 
\begin{bmatrix}0 & 1 \\ -k/m & -k_f/m \end{bmatrix}
\begin{bmatrix}y_1 \\ y_2 \end{bmatrix} + 
\begin{bmatrix}0 \\ 1/m\end{bmatrix}
\begin{bmatrix}u(t)\end{bmatrix}
$$
"""
import numpy as np
import control
import gym

from .base import SystemEnv



def create_springmass(k, m, df=0.02, name='spring'):
    A = np.asarray([[0, 1], [-k/m, -df/m]], dtype=np.float32)
    B = np.asarray([[0], [1/m]], dtype=np.float32)
    C = np.eye(2, dtype=np.float32)
    D = np.zeros((2,1), dtype=np.float32)
    return control.ss(A, B, C, D, name=name)



class SpringMassEnv(SystemEnv):
    def __init__(self, k, m, df=0.02, q=((1,0),(0,1)), r=0, dt=0.01, seed=None):
        system = create_springmass(k, m, df=df)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.period = int(np.pi * 2 * np.sqrt(m / k) / dt)
        super().__init__(system, q, r, dt, seed)
    def reset(self, x=None):
        super().reset(x)
        self.x = (np.asarray(x) if x is not None else \
                 self.random.uniform([-1, -0.1], [1, 0.1])
                 ).astype(np.float32)
        return self.x
    def step(self, u: np.ndarray, from_x=None, persist=True):
        x, r, d, *_, i = super().step(u, from_x=from_x, persist=persist)
        return x, r, self.n > self.period, *_, i
