from typing import Callable, Tuple
import gym
import numpy as np
from multirotor.simulation import Multirotor



class TrajEnv(gym.Env):

    
    def __init__(
        self,
        disturbance: Callable[[float, Tuple], np.ndarray]=lambda t, x: 0.,
        dt=1e-1, ndim: int=2, magnitude: float=1, seed=0
    ):
        # x,y position error (reference - measurement)
        # x,y velocity
        self.ndim = ndim
        self.magnitude = magnitude
        max_dist, max_init_vel = 5, 1
        self.bounds = np.concatenate(([max_dist] * ndim, [max_init_vel] * ndim)).astype(np.float32)
        self.observation_space = gym.spaces.Box(
           low=-self.bounds, high=self.bounds, dtype=np.float32 
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(ndim,), dtype=np.float32
        )
        self.disturbance = disturbance
        # this is how fast the underlying controller is allwoed to change velocity
        # and therefore position.
        self.max_acc = 2.
        self.dt = dt
        # time taken to traverse longest diagonal distance using min velocity
        self.period = 2 * np.linalg.norm(self.bounds[:ndim]) / (min(self.bounds[ndim:]) * self.dt)
        self.t = None
        self._start_pos = None
        self._start_pos_unit = None
        self._seed = seed
        self.random = np.random.RandomState(seed)
        super().__init__()


    def reset(self, x=None):
        if x is None:
            self.state = self.random.rand(self.observation_space.shape[0]).astype(np.float32)
            self.state = (self.state - 0.5) * 2 * self.bounds
        else:
            self.state = np.asarray(x, dtype=np.float32)
        self._start_pos = -self.state[:self.ndim]
        self._start_pos_unit = self._start_pos / np.linalg.norm(self._start_pos)
        self.t = 0.
        return self.state


    def reward(self, old_x, action, x):
        old_pos = -old_x[:self.ndim] # position error to position, negative sign
        new_pos = -x[:self.ndim]
        # return -np.linalg.norm(new_pos)
        delta_pos = old_pos - new_pos
        advance = np.dot(delta_pos, self._start_pos_unit)
        diverge = 0 if self.ndim==1 else np.linalg.norm(np.cross(new_pos, self._start_pos_unit))
        return advance - diverge - self.dt


    def step(self, u: np.ndarray):
        # controller is a function handling trajectory following and PID control
        # and vehicle dynamics. It returns how the velocity changes in response
        # to waypoint modification.
        waypoint = self.magnitude # reference waypoint=0, waypoint is changed by action u
        d_state = controller(self.t, waypoint, self.state, self.disturbance, dt=self.dt, ndim=self.ndim)
        new_state = self.state + d_state * self.dt
        reward = self.reward(self.state, u, new_state)
        self.state = new_state
        self.t += self.dt
        pos_err = np.linalg.norm(self.state[:self.ndim])
        reached = pos_err <= 0.2
        timeout = self.t >= self.period
        if timeout:
            reward -= pos_err
        return self.state, reward, reached or timeout, {}



def controller(
    t, waypoint, state, disturbance,
    k_p=1, max_acc=2., max_vel=5., dt=1e-2, ndim: int=2
):
    pos = -state[:ndim]
    err = waypoint - pos # err = reference - position

    # requested dynamics
    ref_vel = k_p * err
    ref_vel_mag = np.linalg.norm(ref_vel)
    ref_vel_unit = ref_vel / (ref_vel_mag + 1e-6)
    ref_velocity = ref_vel_unit * min(ref_vel_mag, max_vel)
    velocity = state[ndim:]

    acc = (ref_velocity - velocity) / dt
    acc_mag = np.linalg.norm(acc)
    acc_unit = (acc / (acc_mag + 1e-6))
    acceleration = acc_unit * min(acc_mag, max_acc)

    # actual dynamics
    real_acceleration = acceleration + disturbance(t, pos)
    new_velocity = state[ndim:] + real_acceleration * dt
    d_err = -new_velocity
    dstate = np.asarray([*d_err, *real_acceleration], np.float32)
    return dstate



class LowerLevelSystem:


    def __init__(self, disturbance, vehicle: Multirotor, controller):
        self.disturbance = disturbance
        self.vehicle = vehicle
        self.controller = controller


    def dxdt(self, t, x, u: np.ndarray) -> np.ndarray:
        dynamics = self.controller.step()
        dxdt = self.vehicle.dxdt()
        self.vehicle.step()
        return dxdt

