from typing import Literal, Callable

from collections import deque
import numpy as np
import control
import gym
from multirotor.simulation import Multirotor
from multirotor.vehicle import BatteryParams, MotorParams, PropellerParams, VehicleParams, SimulationParams
from multirotor.controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller
)

from .base import SystemEnv


BP = BatteryParams(max_voltage=22.2)
MP = MotorParams(
    moment_of_inertia=5e-5,
    # resistance=0.27,
    resistance=0.081,
    k_emf=0.0265,
    k_motor=0.0932,
    speed_voltage_scaling=0.0347,
    max_current=38.
)
PP = PropellerParams(
    moment_of_inertia=1.86e-6,
    use_thrust_constant=True,
    k_thrust=9.8419e-05, # 18-inch propeller
    # k_thrust=5.28847e-05, # 15 inch propeller
    k_drag=1.8503e-06, # 18-inch propeller
    # k_drag=1.34545e-06, # 15-inch propeller
    # motor=MP
    motor=None
)
VP = VehicleParams(
    propellers=[PP] * 8,
    battery=BP,
    # angles in 45 deg increments, rotated to align with
    # model setup in gazebo sim (not part of this repo)
    angles=np.linspace(0, -2*np.pi, num=8, endpoint=False) + 0.375 * np.pi,
    distances=np.ones(8) * 0.635,
    clockwise=[-1,1,-1,1,-1,1,-1,1],
    mass=10.66,
    inertia_matrix=np.asarray([
        [0.2206, 0, 0],
        [0, 0.2206, 0.],
        [0, 0, 0.4238]
    ])
)
SP = SimulationParams(dt=0.01, g=9.81)



def get_controller(m: Multirotor, max_velocity=1., max_acceleration=3.) -> Controller:
    assert m.simulation.dt <= 0.1, 'Simulation time step too large.'
    pos = PosController(
        1.0, 0., 0., 1., vehicle=m,
        max_velocity=max_velocity, max_acceleration=max_acceleration    
    )
    vel = VelController(
        2.0, 1.0, 0.5, 1000., vehicle=m)
    att = AttController(
        [2.6875, 4.5, 4.5],
        0, 0.,
        1., vehicle=m)
    rat = RateController(
        [4., 4., 4.],
        0, 0, # purely P control
        # [0.1655, 0.1655, 0.5],
        # [0.135, 0.135, 0.018],
        # [0.01234, 0.01234, 0.],
        [0.5,0.5,0.5],
        vehicle=m)

    alt = AltController(
        1, 0, 0,
        1, vehicle=m)
    alt_rate = AltRateController(
        5, 0, 0,
        1, vehicle=m)
    ctrl = Controller(
        pos, vel, att, rat, alt, alt_rate,
        interval_p=0.1, interval_a=0.01, interval_z=0.1
    )
    return ctrl



def create_multirotor(
    vp=VP, sp=SP, name='multirotor', xformA=np.eye(12), xformB=np.eye(4),
    return_mult_ctrl=False,
    kind: Literal['speeds', 'dynamics', 'waypoints']='dynamics',
    max_rads: float=700,
    get_controller_fn: Callable[[Multirotor], Controller]=get_controller,
    disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda _: np.zeros(3, np.float32)
):
    m = Multirotor(vp, sp)
    ctrl = get_controller_fn(m)

    if kind=='dynamics':
        inputs=['fz','tx','ty','tz']
        def update_fn(t, x, u, params):
            speeds = m.allocate_control(u[0], u[1:4])
            speeds = np.clip(speeds, a_min=-max_rads, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
            disturb_forces=disturbance_fn(m))
            return dxdt
    elif kind=='speeds':
        raise NotImplementedError # TODO
    elif kind=='waypoints':
        inputs=['x','y','z','yaw']
        def update_fn(t, x, u, params):
            dynamics = ctrl.step(u, ref_is_error=False)
            speeds = m.allocate_control(dynamics[0], dynamics[1:4])
            speeds = np.clip(speeds, a_min=-max_rads, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
            disturb_forces=disturbance_fn(m))
            return dxdt

    sys = control.NonlinearIOSystem(
        updfcn=update_fn,
        inputs=inputs,
        states=['x','y','z',
                'vx','vy','vz',
                'roll','pitch','yaw',
                'xrate', 'yrate', 'zrate']
    )
    if return_mult_ctrl:
        return sys, dict(multirotor=m, ctrl=ctrl)
    return sys


class MultirotorAllocEnv(SystemEnv):


    def __init__(
        self, vp=VP, sp=SP,
        q=np.diagflat([1,1,1,0.25,0.25,0.25,0.5,0.5,0.5,0.1,0.1,0.1]),
        r = np.diagflat([1,1,1,1]) * 1e-4,
        dt=None, seed=None,
        xformA=np.eye(12), xformB=np.eye(4),
        state_buffer_size: int=0,
        normalize: bool=True, # actions/states are [-1,1]
        clip: bool=True,
        get_controller_fn: Callable[[Multirotor], Controller]=None,
        additive_factor: float=0.1,
        steps_u: int=1,
    ):
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True,
            get_controller_fn=get_controller_fn,
            kind='dynamics'
        )
        self.vehicle = extra['multirotor']
        self.ctrl = extra['ctrl']
        super().__init__(system=system, q=q, r=r, dt=sp.dt, seed=seed, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            # pos err, vel err, angle error, ang rate err, prescribed dynamics
            low=-np.inf, high=np.inf,
            shape=(12+4,), dtype=self.dtype
        )
        self.action_space = gym.spaces.Box(
            # thrust, torque_x, torque_y, torque_z
            low=-1, high=1, shape=(4,), dtype=self.dtype
        )
        self.period = int(20 / sp.dt)
        self.max_angular_acc = self.ctrl.ctrl_r.max_acceleration
        self.max_torque = min(
            np.abs(vp.inertia_matrix.diagonal()) * self.max_angular_acc
        )
        _min_dyn = -np.asarray([
            0,
            self.max_torque, self.max_torque, 0
        ]).astype(self.dtype)
        _max_dyn = np.asarray([
            2 * self.vehicle.weight,
            self.max_torque, self.max_torque, 0
        ]).astype(self.dtype)
        self.action_range = _max_dyn - _min_dyn
        self.state_range = np.asarray([5,5,5,5,5,5,0.62,0.62,0.62,0.62,0.62,0.62], self.dtype)
        self.state_range = np.concatenate((self.state_range, self.action_range), dtype=self.dtype)
        self.steps_u = steps_u

        self.additive_factor = additive_factor
        self.fail_penalty = self.dt * self.period
        self.time_penalty = self.dt * self.steps_u


    @property
    def state(self) -> np.ndarray:
        return self.normalize_state(
            np.concatenate((self.x, self.ctrl.action), dtype=self.dtype)
        )


    def normalize_state(self, state):
        state[12] -= self.vehicle.weight
        return state * 2 / (self.state_range+1e-6)
    def unnormalize_state(self, state):
        state *= self.state_range / 2
        state[12] += self.vehicle.weight
        return state
    def normalize_action(self, u):
        u[0] -= self.vehicle.weight
        return u * 2 / (self.action_range+1e-6)
    def unnormalize_action(self, u):
        u *= self.action_range / 2
        u[0] += self.vehicle.weight
        return u


    def reset(self, x=None):
        super().reset(x)
        self.ctrl.reset()
        pos = np.asarray(x[0:3]) if x is not None else \
              self.random.uniform(-2.5, 2.5, size=3)
        vel = np.asarray(x[3:6]) if x is not None else \
              self.random.uniform(-0.00, 0.00, size=3)
        ori = np.asarray(x[6:9]) if x is not None else \
              self.random.uniform(-0., 0., size=3)
        rat = np.asarray(x[9:12]) if x is not None else \
              self.random.uniform(-0.0, 0.0, size=3)
        self.x = np.concatenate((pos, vel, ori, rat), dtype=self.dtype)
        self.vehicle.state = self.x
        return self.state


    def step(self, u: np.ndarray, **kwargs):
        u = np.clip(u, a_min=-1., a_max=1.)
        u = self.unnormalize_action(u)
        u = (self.additive_factor * u) + (1-self.additive_factor) * (self.ctrl.action)
        olddist = np.linalg.norm(self.x[:3])
        x, r, d, *_, i = super().step(u)
        # The controller action is updated based on the current state of the vehicle,
        # after the action has been applied
        self.ctrl.step(reference=np.zeros(12), measurement=self.x)
        # since SystemEnv is only using Multirotor.dxdt methods, the mult
        # object is not integrating the change in state. Therefore manually
        # set the state on the self.vehicle object to reflect the integration
        # that SystemEnv.step() does
        self.vehicle.state = self.x
        # reward/termination logic
        dist = np.linalg.norm(self.x[:3])
        change = olddist - dist
        reached = dist <= 0.1
        outofbounds = (dist >= self.state_range[0])
        outoftime = self.n >= self.period
        tipped = any(self.x[6:9] > self.ctrl.ctrl_v.max_tilt)
        done = outoftime or outofbounds or reached or tipped
        reward = -self.dt
        if reached:
            reward += 10.
        elif outofbounds or tipped:
            reward -= self.fail_penalty
        reward += change * 10
        return self.state, reward, done, *_, i



class MultirotorTrajEnv(SystemEnv):


    def __init__(
        self, vp=VP, sp=SP,
        q=np.diagflat([1,1,1,0.25,0.25,0.25,0.5,0.5,0.5,0.1,0.1,0.1]),
        r = np.diagflat([1,1,1,0]) * 1e-4,
        dt=None, seed=None,
        xformA=np.eye(12), xformB=np.eye(4),
        get_controller_fn: Callable[[Multirotor], Controller]=None,
        disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda t: np.zeros(3, np.float32),
        scaling_factor: float=1.,
        steps_u: int=1,
        bounding_box: float=5.
    ):
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True,
            get_controller_fn=get_controller_fn,
            disturbance_fn=disturbance_fn,
            kind='waypoints'
        )
        self.vehicle = extra['multirotor']
        self.ctrl = extra['ctrl']
        super().__init__(system=system, q=q, r=r, dt=sp.dt, seed=seed, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            # pos err, vel err, angle error, ang rate err, prescribed dynamics
            low=-1, high=1,
            shape=(12,), dtype=self.dtype
        )
        self.action_space = gym.spaces.Box(
            # x,y,z
            low=-1, high=1, shape=(3,), dtype=self.dtype
        )
        self.scaling_factor = scaling_factor
        self.bounding_box = bounding_box
        self.state_range = np.empty(self.observation_space.shape, self.dtype)
        self.action_range = np.empty(self.action_space.shape, self.dtype)
        self.steps_u = steps_u

        self.period = 20 # seconds
        self.fail_penalty = 10
        self.time_penalty = self.dt * self.steps_u
        self._init_pos = None


    @property
    def state(self) -> np.ndarray:
        return self.normalize_state(self.x)


    def normalize_state(self, state):
        return state * 2 / (self.state_range+1e-6)
    def unnormalize_state(self, state):
        state *= self.state_range / 2
        return state
    def normalize_action(self, u):
        return u * 2 / (self.action_range+1e-6)
    def unnormalize_action(self, u):
        u *= self.action_range / 2
        return u


    def reset(self, x=None):
        super().reset(x)
        self.ctrl.reset()
        self.state_range[0:3] = self.bounding_box
        self.state_range[3:6] = self.ctrl.ctrl_p.max_velocity
        self.state_range[6:9] = 2 * self.ctrl.ctrl_v.max_tilt
        # Max velocity equals the max angle err * k_p to convert to rate
        self.state_range[9:12] = 2 * self.ctrl.ctrl_v.max_tilt * self.ctrl.ctrl_a.k_p
        self.action_range = self.state_range[:self.action_space.shape[0]]

        pos = np.asarray(x[0:3]) if x is not None else \
              self.random.uniform(-self.bounding_box, self.bounding_box, size=3) / 2
        vel = np.asarray(x[3:6]) if x is not None else \
              self.random.uniform(-self.ctrl.ctrl_p.max_velocity, self.ctrl.ctrl_p.max_velocity, size=3) / 2
        ori = np.asarray(x[6:9]) if x is not None else \
              self.random.uniform(-0., 0., size=3)
        rat = np.asarray(x[9:12]) if x is not None else \
              self.random.uniform(-0.0, 0.0, size=3)
        self.x = np.concatenate((pos, vel, ori, rat), dtype=self.dtype)
        self.vehicle.state = self.x
        # The desired direction of trajectory. Equal to straight line from initial
        # position to origin.
        self._des_vec = (0 - self.x[:self.action_space.shape[0]]) / np.linalg.norm(self.x[:self.action_space.shape[0]])
        return self.state


    def step(self, u: np.ndarray, **kwargs):
        u = np.clip(u, a_min=-1., a_max=1.)
        u = self.unnormalize_action(u)
        u *= self.scaling_factor
        oldpos = self.x[:3]
        for _ in range(self.steps_u):
            x, r, d, *_, i = super().step(np.concatenate((u,[0])))
            # since SystemEnv is only using Multirotor.dxdt methods, the mult
            # object is not integrating the change in state. Therefore manually
            # set the state on the self.vehicle object to reflect the integration
            # that SystemEnv.step() does
            self.vehicle.state = self.x
            self.vehicle.t = self.t
        pos = self.x[:3]

        dist = np.linalg.norm(self.x[:3])
        reached = dist <= 0.1
        outofbounds = np.any(self.x[:3] > self.bounding_box / 2)
        outoftime = self.t >= self.period
        tipped = np.any(self.x[6:9] > self.ctrl.ctrl_v.max_tilt)
        done = outoftime or outofbounds or reached or tipped

        dpos = pos - oldpos
        advance = np.dot(dpos, self._des_vec)
        cross = np.linalg.norm(np.cross(dpos, self._des_vec))
        reward = -self.time_penalty
        if reached:
            reward += self.fail_penalty
        elif outofbounds or tipped:
            reward -= self.fail_penalty
        reward += (advance - cross) * 10
        return self.state, reward, done, *_, i
