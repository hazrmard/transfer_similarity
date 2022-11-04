"""
The state of a pendulum is defined by the angle $\theta$, where the action $u(t)$ is the torque being applied to the pendulum:

\begin{align}
\dot{y_1} &= \dot{\theta} =& y_2 \\
\dot{y_2} &= \ddot{\theta} =& -\frac{g}{l}\sin\theta + \frac{u(t) -k_f\dot{\theta}}{ml^2}
\end{align}

\begin{align}
\begin{bmatrix}\dot{y}_1 \\ \dot{y}_2 \end{bmatrix} = 
\begin{bmatrix}0 & 1 \\ -g\sin(\cdot)/l & -k_f/ml^2 \end{bmatrix}
\begin{bmatrix}y_1 \\ y_2 \end{bmatrix} + 
\begin{bmatrix}0 \\ 1/ml^2\end{bmatrix}
\begin{bmatrix}u(t)\end{bmatrix}
\end{align}
"""
import numpy as np
import control
import gym

from .base import SystemEnv



def create_pendulum(m, l, g, df=0.02, name='pendulum',
                    xformA=np.eye(2, dtype=np.float32),
                    xformB=np.eye(2, dtype=np.float32)):
    g_l = -g / l
    ml2_inv = 1 / (m * l**2)
    def updfcn(t, x, u, params):
        if x.ndim==2:
            x = x.squeeze()
            u = np.atleast_1d(u.squeeze())
        Ax = np.asarray(
            [[0, x[1]],
             [g_l * np.sin(x[0]), -df * x[1] * ml2_inv]],
            dtype=np.float32).sum(axis=1)
        Bu = np.asarray(
            [[0],
             [u[0] * ml2_inv]],
            dtype=np.float32).sum(axis=1)
        return xformA @ Ax + xformB @ Bu
    def outfcn(t, x, u, params):
        return x
    sys = control.NonlinearIOSystem(updfcn, outfcn, name=name,
                                    inputs=1, outputs=2, states=2)
    return sys



class PendulumEnv(SystemEnv):
    def __init__(
        self, m, l, g, df=0.02, q=((10,0),(0,0.1)), r=0, dt=0.01,
        seed=None, xformA=np.eye(2), xformB=np.eye(2)
    ):
        system = create_pendulum(m, l, g, df=df, xformA=xformA, xformB=xformB)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.period = int(np.pi * 2 * np.sqrt(l / g) / dt)
        super().__init__(system, q, r, dt, seed)
    def reset(self, x=None):
        super().reset(x)
        self.x = (np.asarray(x) if x is not None else \
                 self.random.uniform([-np.pi/6, -0.05], [np.pi/6, 0.05])
                 ).astype(np.float32)
        return self.x, {}
    def step(self, u: np.ndarray, from_x=None, persist=True):
        x, r, d, _, i = super().step(u, from_x=from_x, persist=persist)
        if x.ndim==1:
            x[0] = np.sign(x[0]) * (np.abs(x[0]) % (2 * np.pi))
            if persist:
                self.x[0] = x[0]
        # the mpc library treats state as 2d vectors (1 row)
        elif x.ndim==2:
            x[0,0] = np.sign(x[0,0]) * (np.abs(x[0,0]) % (2 * np.pi))
            if persist:
                self.x[0,0] = x[0,0]
        return x, r, self.n > self.period, False, i