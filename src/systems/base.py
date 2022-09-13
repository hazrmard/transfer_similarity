import numpy as np
import gym


class SystemEnv(gym.Env):


    def __init__(
        self, system, q=1, dt=1e-2, seed=None, constrained_actions=True,
        dtype=np.float32
    ):
        super().__init__()
        self.system = system
        self.dtype = dtype
        self.q = np.atleast_2d(q)
        self.dt = dt
        self.x = None
        self.dxdt = None
        self.n = None
        self.random = np.random.RandomState(seed)


    def reset(self, x=None):
        self.x = (np.asarray(x) if x is not None else \
                  self.observation_space.sample()
                 ).astype(self.dtype)
        self.n = 0
        self.dxdt = 0
        return self.x


    def reward(self, xold, u, x):
        return -(x.T @ self.q @ x).item()


    def step(self, u: np.ndarray):
        u = np.asarray(u, dtype=self.dtype)
        dxdt = self.system.dynamics(None, self.x, u)
        x = self.x
        self.x = (x + 0.5 * (self.dxdt + dxdt) * self.dt).astype(self.dtype)
        self.dxdt = dxdt
        self.n += 1
        return self.x, self.reward(x, u, self.x), False, {'u': u}
