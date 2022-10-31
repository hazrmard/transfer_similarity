from typing import Callable, Tuple
import gym
import numpy as np



class TrajEnv(gym.Env):

    
    def __init__(
        self,
        disturbance: Callable[[float, Tuple], np.ndarray]=lambda t, x: 0.,
        dt=1e-1
    ):
        # x,y position error (reference - measurement)
        # x,y velocity
        self.bounds = np.asarray((10,10,1,1)).astype(np.float32)
        self.observation_space = gym.spaces.Box(
           low=-self.bounds, high=self.bounds, dtype=np.float32 
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.disturbance = disturbance
        # this is how fast the underlying controller is allwoed to change velocity
        # and therefore position.
        self.max_acc = 2.
        self.dt = dt
        # time taken to traverse longest diagonal distance using min velocity
        self.period = 2 * np.linalg.norm(self.bounds[:2]) / (min(self.bounds[2:]) * self.dt)
        self.t = None
        self._start_pos = None
        self._start_pos_unit = None
        super().__init__()


    def reset(self, x=None):
        if x is None:
            self.state = np.random.rand(self.observation_space.shape[0]).astype(np.float32)
            self.state = (self.state - 0.5) * 2 * self.bounds
        else:
            self.state = np.asarray(x, dtype=np.float32)
        self._start_pos = -self.state[:2]
        self._start_pos_unit = self._start_pos / np.linalg.norm(self._start_pos)
        self.t = 0.
        return self.state


    def reward(self, old_x, action, x):
        old_pos = -old_x[:2] # position error to position, negative sign
        new_pos = -x[:2]
        delta_pos = old_pos - new_pos
        advance = np.dot(delta_pos, self._start_pos_unit)
        diverge = np.linalg.norm(np.cross(new_pos, self._start_pos_unit))
        return advance - diverge


    def step(self, u: np.ndarray):
        # controller is a function handling trajectory following and PID control
        # and vehicle dynamics. It returns how the velocity changes in response
        # to waypoint modification.
        waypoint = u # reference waypoint=0, waypoint is changed by action u
        d_state = controller(self.t, waypoint, self.state, self.disturbance, dt=self.dt)
        new_state = self.state + d_state * self.dt
        reward = self.reward(self.state, u, new_state)
        self.state = new_state
        self.t += self.dt
        pos_err = np.linalg.norm(self.state[:2])
        reached = pos_err <= 0.1
        timeout = self.t >= self.period
        if timeout:
            reward -= pos_err
        return self.state, reward, reached or timeout, {}



def controller(
    t, waypoint, state, disturbance,
    k_p=1, max_acc=2., max_vel=5., dt=1e-2
):
    pos = -state[:2]
    err = waypoint - pos # err = reference - position

    # requested dynamics
    ref_velocity = np.clip(k_p * err, -max_vel, max_vel)
    velocity = state[2:]
    acceleration = np.clip((ref_velocity - velocity) / dt, -max_acc, max_acc)

    # actual dynamics
    real_acceleration = acceleration + disturbance(t, pos)
    new_velocity = state[2:] + real_acceleration * dt
    d_err = -new_velocity
    dstate = np.asarray([*d_err, *real_acceleration], np.float32)
    return dstate



class LowerLevelSystem:


    def __init__(self, disturbance):
        self.disturbance = disturbance


    def dxdt(self, t, x, u: np.ndarray) -> np.ndarray:
        pass

