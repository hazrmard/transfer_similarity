"""
The state of an inverted pendulum system is defined by the linear and angular accelerations $\ddot{x}, \ddot{\theta}$. $x$ is positive right, $\theta$ is positive counter-clockwise.

Then, linearizing it by assuming $y_1=\theta, y_2=\dot{\theta}, y_3=x, y_4=\dot{x}$

\begin{align}
\dot{y_1} &= \dot{\theta} =& y_2 \\
\dot{y_2} &= \ddot{\theta} =& \frac{g}{l}\sin y_1 - \frac{k_f y_2}{ml^2} + \frac{\dot{y_4}\cos y_1}{l} \\
\dot{y_3} &= \dot{x} =& y_4 \\
\dot{y_4} &= \ddot{x} =& \frac{m_p \sin y_1\left(g\cos y_1-l\dot{y_1}^2\right) + u(t)}{m_c+m_p-m_p\cos^2 y_1}
\end{align}
"""
import numpy as np
import control
import gym

from .base import SystemEnv



def create_cartpole(mc, mp, l, g, df=0.01, name='cartpole',
                   xformA=np.eye(4), xformB=np.eye(4)):
    g_l = g / l
    ml2_inv = 1 / (mp * l**2)
    def updfcn(t, x, u, params):
        if x.ndim==2:
            # for MPC simulations
            u = np.atleast_1d(u.squeeze())
            x1,x2,x3,x4=x.squeeze() # theta, theta_dot, x, x_dot
        elif x.ndim==1:
            x1,x2,x3,x4=x # theta, theta_dot, x, x_dot
        cx1, sx1 = np.cos(x1), np.sin(x1)
        x4_denom = mc + mp - mp * cx1**2
        # delta x due to state
        x1_ = x2
        x2_ = (g_l * sx1) - (df * ml2_inv * x2) + \
              ((cx1 / l) * ((mp * sx1 * (g*cx1 - l * x2**2)) / x4_denom))
        x3_ = x4
        x4_ = (mp * sx1 * (g*cx1 - l * x2**2)) / x4_denom
        dx_x = np.asarray([x1_, x2_, x3_, x4_], dtype=np.float32)
        # delta x due to action. Factoring out parts in u
        x1_ = 0
        x2_ = (cx1 / l) * (u[0]) / x4_denom
        x3_ = 0
        x4_ = u[0] / x4_denom
        dx_u = np.asarray([x1_, x2_, x3_, x4_], dtype=np.float32)
        return xformA @ dx_x + xformB @ dx_u
    def outfcn(t, x, u, params):
        return x
    sys = control.NonlinearIOSystem(updfcn, outfcn, name=name,
                                    inputs=1, outputs=4, states=4)
    return sys



class CartpoleEnv(SystemEnv):
    def __init__(self, mc, mp, l, g, df=0.01,
                 q=((1,0,0,0),(0,0.01,0,0),(0,0,0.1,0),(0,0,0,0.01)),
                 r=0,
                 dt=0.01, seed=None, xformA=np.eye(4), xformB=np.eye(4)):
        system = create_cartpole(mc, mp, l, g, df=df,
                                 xformA=xformA, xformB=xformB)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.period = int(np.pi * 2 * np.sqrt(l / g) / dt)
        super().__init__(system, q, r, dt, seed)
    def reset(self, x=None):
        super().reset(x)
        self.x = (np.asarray(x) if x is not None else \
                 self.random.uniform(-0.05, 0.05, size=4)
                 ).astype(np.float32)
        return self.x, {}
    def reward(self, xold, u, x):
        return 1.
    def step(self, u: np.ndarray, from_x=None, persist=True):
        x, r, d, _, i = super().step(u, from_x=from_x, persist=persist)
        if x.ndim==1:
            x[0] = np.sign(x[0]) * (np.abs(x[0]) % (2 * np.pi))
            if persist:
                self.x[0] = x[0]
            done = (np.abs(x[0]) > 0.2095) or \
                   (self.n >= 500) or \
                   (np.abs(x[2]) > 2.4)
        elif x.ndim==2:
            assert not persist, '2 dim array only during MPC simulation'
            x[0,0] = np.sign(x[0,0]) * (np.abs(x[0,0]) % (2 * np.pi))
            done = (np.abs(x[0,0]) > 0.2095) or \
                   (self.n >= 500) or \
                   (np.abs(x[0,2]) > 2.4)
        return x, r, done, False, i