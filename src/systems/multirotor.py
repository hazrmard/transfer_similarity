from typing import Literal, Callable, Type, Union, Iterable
from argparse import Namespace

from copy import deepcopy
import numpy as np
from stable_baselines3.ppo import PPO
import control
import gym
from tqdm.autonotebook import tqdm
from multirotor.simulation import Multirotor
from multirotor.helpers import DataLog
from multirotor.trajectories import Trajectory
from multirotor.vehicle import BatteryParams, MotorParams, PropellerParams, VehicleParams, SimulationParams
from multirotor.controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller
)

from .base import SystemEnv

DEFAULTS = Namespace(
    bounding_box = 20.,
    max_velocity = 3.,
    max_acceleration = 2.5,
    max_tilt = np.pi / 12,
    max_rads = 700
)


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
    propellers=[deepcopy(PP) for _ in range(8)],
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
SP = SimulationParams(dt=0.01, g=9.81, dtype=np.float32)



def get_controller(
        m: Multirotor, max_velocity=DEFAULTS.max_velocity,
        max_acceleration=DEFAULTS.max_acceleration,
        max_tilt=DEFAULTS.max_tilt
    ) -> Controller:
    assert m.simulation.dt <= 0.1, 'Simulation time step too large.'
    pos = PosController(
        1.0, 0., 0.,
        max_err_i=DEFAULTS.max_velocity, vehicle=m,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration  ,
        square_root_scaling=False  
    )
    vel = VelController(
        2.0, 1.0, 0.5,
        max_err_i=DEFAULTS.max_acceleration, vehicle=m, max_tilt=max_tilt)
    att = AttController(
        [2.6875, 4.5, 4.5],
        0, 0.,
        max_err_i=1., vehicle=m)
    rat = RateController(
        [4., 4., 4.],
        0, 0, # purely P control
        # [0.1655, 0.1655, 0.5],
        # [0.135, 0.135, 0.018],
        # [0.01234, 0.01234, 0.],
        max_err_i=0.5,
        vehicle=m)

    alt = AltController(
        1, 0, 0,
        max_err_i=1, vehicle=m, max_velocity=max_velocity)
    alt_rate = AltRateController(
        10, 0, 0,
        max_err_i=1, vehicle=m)
    ctrl = Controller(
        pos, vel, att, rat, alt, alt_rate,
        period_p=0.1, period_a=0.01, period_z=0.1
    )
    return ctrl



def create_multirotor(
    vp=VP, sp=SP, name='multirotor', xformA=np.eye(12), xformB=np.eye(4),
    return_mult_ctrl=False,
    kind: Literal['speeds', 'dynamics', 'waypoints']='dynamics',
    max_rads: float=DEFAULTS.max_rads,
    get_controller_fn: Callable[[Multirotor], Controller]=get_controller,
    disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda _: np.zeros(3, SP.dtype),
    multirotor_class: Type[Multirotor]=Multirotor,
    multirotor_kwargs: dict={}
):
    m = multirotor_class(vp, sp, **multirotor_kwargs)
    ctrl = get_controller_fn(m)

    if kind=='dynamics':
        inputs=['fz','tx','ty','tz']
        def update_fn(t, x, u, params):
            speeds = m.allocate_control(u[0], u[1:4])
            speeds = np.clip(speeds, a_min=0, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
            disturb_forces=disturbance_fn(m))
            for prop, speed in zip(m.propellers, speeds):
                prop.step(speed, max_voltage=m.battery.voltage)
            return dxdt
    elif kind=='speeds':
        inputs = [('w%02d' % n) for n in range(len(m.propellers))]
        def update_fn(t, x, u, params):
            # speeds = np.clip(u, a_min=0, a_max=max_rads)
            # m.step_speeds(speeds, disturb_forces=disturbance_fn(m))
            # # here, waypoint supervision can be added
            # old_dynamics = ctrl.action
            # new_dynamics = ctrl.step(np.zeros(4, m.dtype), ref_is_error=False)
            # return (new_dynamics - old_dynamics) / m.simulation.dt
            return None # integration of dynamics is done directly in MultirotorAllocEnv.step()
    elif kind=='waypoints':
        inputs=['x','y','z','yaw']
        def update_fn(t, x, u, params):
            dynamics = ctrl.step(u, ref_is_error=False)
            speeds = m.allocate_control(dynamics[0], dynamics[1:4])
            speeds = np.clip(speeds, a_min=0, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
                disturb_forces=disturbance_fn(m))
            # for prop, speed in zip(m.propellers, speeds):
            #     prop.step(speed, max_voltage=m.battery.voltage)
            # dxdt[8] = 0. # no yaw change in lateral x/y motion problem
            # dxdt[11] = 0.
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


class MultirotorDynamicsEnv(SystemEnv):


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
        self.x = np.zeros(self.observation_space.shape, self.dtype)
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
        get_controller_fn: Callable[[Multirotor], Controller]=get_controller,
        disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda m: np.zeros(3, np.float32),
        scaling_factor: float=1.,
        steps_u: int=1,
        # length of cube centered at origin within which position is initialized,
        # half length of cube centered at origin within which vehicle may move
        bounding_box: float=DEFAULTS.bounding_box,
        multirotor_class=Multirotor, multirotor_kwargs={}
    ):
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True,
            get_controller_fn=get_controller_fn,
            disturbance_fn=disturbance_fn,
            kind='waypoints',
            multirotor_class=multirotor_class,
            multirotor_kwargs=multirotor_kwargs
        )
        self.vehicle: Multirotor = extra['multirotor']
        self.ctrl: Controller = extra['ctrl']
        super().__init__(system=system, q=q, r=r, dt=sp.dt, seed=seed, dtype=sp.dtype)
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
        self.overshoot_factor = 0.5
        self.state_range = np.empty(self.observation_space.shape, self.dtype)
        self.action_range = np.empty(self.action_space.shape, self.dtype)
        self.x = np.zeros(self.observation_space.shape, self.dtype)
        self.steps_u = steps_u

        self.period = 20 # seconds
        self.max_time_penalty = self.period
        self.motion_reward_scaling = 10
        self.fail_penalty = self.pass_reward = self.bounding_box * self.motion_reward_scaling * 2
        self._proximity = max(self.vehicle.params.distances)


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


    def reset(self, uav_x=None):
        super().reset(uav_x)
        self.ctrl.reset()
        self.vehicle.reset()
        # Nominal range of state, not accounting for overshoot due to process dynamics
        self.state_range[0:3] = 2 * self.bounding_box
        self.state_range[3:6] = 2 * self.ctrl.ctrl_p.max_velocity
        self.state_range[6:9] = 2 * self.ctrl.ctrl_v.max_tilt
        self.state_range[9:12] = 2 * self.ctrl.ctrl_v.max_tilt * self.ctrl.ctrl_a.k_p
        self.action_range = self.state_range[:self.action_space.shape[0]] * self.scaling_factor
        # Max overshoot allowed, which will cause episode to terminate
        self._max_pos = self.bounding_box * (1 + self.overshoot_factor) / 2
        self._max_angle = self.ctrl.ctrl_v.max_tilt * (1 + self.overshoot_factor)
        self.time_penalty = self.dt * self.steps_u

        pos = np.asarray(uav_x[0:3]) if uav_x is not None else \
              self.random.uniform(-self.bounding_box/2, self.bounding_box/2, size=3)
        vel = np.asarray(uav_x[3:6]) if uav_x is not None else \
              self.random.uniform(-self.ctrl.ctrl_p.max_velocity/2, self.ctrl.ctrl_p.max_velocity/2, size=3)
        ori = np.asarray(uav_x[6:9]) if uav_x is not None else \
              self.random.uniform(-0., 0., size=3)
        rat = np.asarray(uav_x[9:12]) if uav_x is not None else \
              self.random.uniform(-0.0, 0.0, size=3)
        self.x = np.concatenate((pos, vel, ori, rat), dtype=self.dtype)
        # Manually set underlying vehicle's state
        self.vehicle.state = self.x
        # The desired direction of trajectory. Equal to straight line from initial
        # position to origin.
        self._des_unit_vec = (0 - pos) / (np.linalg.norm(pos) + 1e-6)
        return self.state


    def step(self, u: np.ndarray, **kwargs):
        u = np.clip(u, a_min=-1., a_max=1.)
        u = self.unnormalize_action(u)
        oldpos = self.x[:3]
        old_turn = np.abs(self.x[8])
        for _ in range(self.steps_u):
            x, r, d, *_, i = super().step(np.concatenate((u,[0])))
            # since SystemEnv is only using Multirotor.dxdt methods, the mult
            # object is not integrating the change in state. Therefore manually
            # set the state on the self.vehicle object to reflect the integration
            # that SystemEnv.step() does
            self.vehicle.state = self.x
            self.vehicle.t = self.t
            dist = np.linalg.norm(self.x[:3])
            reached = dist <= self._proximity
            outofbounds = np.any(np.abs(self.x[:3]) > self._max_pos)
            outoftime = self.t >= self.period
            tipped = np.any(np.abs(self.x[6:9]) > self._max_angle)
            done = outoftime or outofbounds or reached or tipped
            if done:
                i.update(dict(reached=reached, outofbounds=outofbounds, outoftime=outoftime, tipped=tipped))
                break


        # advance = olddist - dist
        delta_pos = (self.x[:3] - oldpos)
        advance = np.linalg.norm(delta_pos)
        cross = np.linalg.norm(np.cross(delta_pos, self._des_unit_vec))
        delta_turn = np.abs(self.x[8]) - old_turn
        # cross = 0.
        reward = ((advance - cross - delta_turn) * self.motion_reward_scaling) - self.time_penalty
        if reached:
            reward += self.pass_reward
        elif tipped or outofbounds:
            reward -= self.fail_penalty
        elif outoftime:
            reward -= dist * self.motion_reward_scaling
        return self.state, reward, done, *_, i


    def ctrl_fn(self, x):
        return np.zeros(3, self.dtype)



class MultirotorAllocEnv(SystemEnv):


    def __init__(
        self, vp=VP, sp=SP,
        q=np.diagflat([1,1,1,1]),
        r = np.diagflat([1,1,1,1,1,1,1,1]) * 1e-4,
        dt=None, seed=None,
        xformA=np.eye(4), xformB=np.eye(8),
        get_controller_fn: Callable[[Multirotor], Controller]=get_controller,
        disturbance_fn: Callable[[Multirotor], np.ndarray]=lambda m: np.zeros(3, SP.dtype),
        scaling_factor: float=1.,
        steps_u: int=1,
        # length of cube centered at origin within which position is initialized,
        # half length of cube centered at origin within which vehicle may move
        bounding_box: float=DEFAULTS.bounding_box
    ):
        self.disturbance_fn = disturbance_fn
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True,
            get_controller_fn=get_controller_fn,
            disturbance_fn=self.disturbance_fn,
            kind='speeds'
        )
        self.vehicle: Multirotor = extra['multirotor']
        self.ctrl: Controller = extra['ctrl']
        super().__init__(system=system, q=q, r=r, dt=sp.dt, seed=seed, dtype=sp.dtype)

        self.observation_space = gym.spaces.Box(
            # prescribed dynamics
            low=-1, high=1,
            shape=(4+12,), dtype=self.dtype
        )
        self.action_space = gym.spaces.Box(
            # x,y,z
            low=-1, high=1, shape=(8,), dtype=self.dtype
        )

        max_acc = self.ctrl.ctrl_p.max_acceleration
        mass = self.vehicle.params.mass
        weight = self.vehicle.weight
        k_th = self.vehicle.params.propellers[0].k_thrust
        n = len(self.vehicle.propellers)
        self.rads_n = np.sqrt(weight / (n*k_th)) # nominal rad/s
        self.delta_rads = np.sqrt((weight + max_acc * mass) / (n*k_th)) - self.rads_n

        self.max_angular_acc = self.ctrl.ctrl_r.max_acceleration
        self.max_torque = min(
            np.abs(vp.inertia_matrix.diagonal()) * self.max_angular_acc
        )
        self.bounding_box = bounding_box
        self.overshoot_factor = 0.5
        self.state_range = np.empty(self.observation_space.shape, self.dtype)
        self.action_range = np.empty(self.action_space.shape, self.dtype)
        self.x = np.zeros(self.observation_space.shape, self.dtype)
        self.scaling_factor = scaling_factor
        self.steps_u = 1

        self.period = 20 # seconds
        self.max_time_penalty = self.period
        self.motion_reward_scaling = 10
        self.fail_penalty = self.pass_reward = self.bounding_box * self.motion_reward_scaling * 2
        self._proximity = max(self.vehicle.params.distances)


    @property
    def state(self) -> np.ndarray:
        return self.normalize_state(self.x)


    def normalize_state(self, state: np.ndarray):
        state = state * 2 / (self.state_range+1e-6)
        state[12] -= 1 # F_z
        return state
    def unnormalize_state(self, state: np.ndarray):
        state = state.copy()
        state[12] += 1 # F_z
        state *= self.state_range / 2
        return state
    def normalize_action(self, u: np.ndarray):
        return (u) / self.delta_rads
    def unnormalize_action(self, u: np.ndarray):
        return u * self.delta_rads


    def reset(self, uav_x=None, x=None):
        super().reset(x)
        self.ctrl.reset()
        self.vehicle.reset()
        # Nominal range of state, not accounting for overshoot due to process dynamics
        self.state_range[0:3] = 2 * self.bounding_box
        self.state_range[3:6] = 2 * self.ctrl.ctrl_p.max_velocity
        self.state_range[6:9] = 2 * self.ctrl.ctrl_v.max_tilt
        self.state_range[9:12] = 2 * self.ctrl.ctrl_v.max_tilt * self.ctrl.ctrl_a.k_p
        self.state_range[12] = 2 * self.vehicle.weight
        self.state_range[13:16] = 2 * self.max_torque
        self.action_range[:] = 2 * self.delta_rads * (1 + self.overshoot_factor)
        # Max overshoot allowed, which will cause episode to terminate
        self._max_pos = self.bounding_box * (1 + self.overshoot_factor) / 2
        self._max_angle = self.ctrl.ctrl_v.max_tilt * (1 + self.overshoot_factor)
        self.time_penalty = self.dt * self.steps_u

        pos = np.asarray(uav_x[0:3]) if uav_x is not None else \
              self.random.uniform(-self.bounding_box/2, self.bounding_box/2, size=3)
        vel = np.asarray(uav_x[3:6]) if uav_x is not None else \
              self.random.uniform(-self.ctrl.ctrl_p.max_velocity/2, self.ctrl.ctrl_p.max_velocity/2, size=3)
        ori = np.asarray(uav_x[6:9]) if uav_x is not None else \
              self.random.uniform(-0., 0., size=3)
        rat = np.asarray(uav_x[9:12]) if uav_x is not None else \
              self.random.uniform(-0.0, 0.0, size=3)
        self.uav_x = np.concatenate((pos, vel, ori, rat), dtype=self.dtype)
        # Manually set underlying vehicle's state
        self.vehicle.state = self.uav_x
        self.x = np.concatenate((
            self.uav_x,
            self.ctrl.step(np.zeros(4, self.dtype), ref_is_error=False) # get initial prescribed dynamics from uav state,
        ))
        # The desired direction of trajectory. Equal to straight line from initial
        # position to origin.
        self._des_unit_vec = (0 - pos) / (np.linalg.norm(pos) + 1e-6)
        return self.state


    def step(self, u: np.ndarray, **kwargs):
        # u = np.clip(u, a_min=-1., a_max=1.)
        u = self.unnormalize_action(u) * self.scaling_factor
        speeds_nominal = self.vehicle.allocate_control(self.x[12], self.x[13:16])
        speeds = np.clip(u + speeds_nominal, a_min=0, a_max=DEFAULTS.max_rads)
        oldpos = self.uav_x[:3]
        old_turn = np.abs(self.uav_x[8])
        persist = kwargs.get('persist', True)
        for _ in range(self.steps_u):
            # we SystemEnv.step() takes integrates a dxdt() function. Instead,
            # we directly get the next state
            # x, r, d, *_, i = super().step(u)
            # this part is replacing super().step(u):
            # Here we are using step_speeds which calls odeint function.
            # THis is different from SystemEnv.step(), used by TrajEnv, which uses trapezoid rule to integrate
            # dxdt_speeds(). The difference in integration causes minor changes in dynamics between
            # TrajEnv and AllocEnv
            self.vehicle.step_speeds(speeds, disturb_forces=self.disturbance_fn(self.vehicle)) # TODO: persist
            old_dynamics = self.x[12:16]
            new_dynamics = self.ctrl.step(np.zeros(4, self.dtype), ref_is_error=False, persist=persist)
            # print('D_old', old_dynamics, 'D_new', new_dynamics)
            dxdt = 2 * (new_dynamics - old_dynamics) / self.dt - self.dxdt
            i = dict(u=speeds, dxdt=dxdt)
            if persist:
                self.n += 1
                self.t += self.dt
                self.vehicle.t = self.t
                self.vehicle.state[8] = 0 # Forcing yaw dynamics to be 0
                self.vehicle.state[11] = 0
                self.dxdt = dxdt # only for dynamics
                self.uav_x = self.vehicle.state
                self.x = np.concatenate((self.uav_x, new_dynamics))

            dist = np.linalg.norm(self.uav_x[:3])
            reached = dist <= self._proximity
            outofbounds = np.any(np.abs(self.uav_x[:3]) > self._max_pos)
            outoftime = self.t >= self.period
            tipped = np.any(np.abs(self.uav_x[6:9]) > self._max_angle)
            done = outoftime or outofbounds or reached or tipped
            if done:
                i.update(dict(reached=reached, outofbounds=outofbounds, outoftime=outoftime, tipped=tipped))
                break


        # advance = olddist - dist
        delta_pos = (self.uav_x[:3] - oldpos)
        advance = np.linalg.norm(delta_pos)
        cross = np.linalg.norm(np.cross(delta_pos, self._des_unit_vec))
        delta_turn = np.abs(self.uav_x[8]) - old_turn
        # cross = 0.
        reward = ((advance - cross - delta_turn) * self.motion_reward_scaling) - self.time_penalty
        if reached:
            reward += self.pass_reward
        elif tipped or outofbounds:
            reward -= self.fail_penalty
        elif outoftime:
            reward -= dist * self.motion_reward_scaling
        return self.state, reward, done, i


    def ctrl_fn(self, x: np.ndarray):
        # dynamics = self.unnormalize_state(x)
        # speeds = self.vehicle.allocate_control(dynamics[0], dynamics[1:4])
        # return self.normalize_action(speeds)
        return self.normalize_action(np.zeros(self.vehicle.params.nprops, self.dtype))



def run_sim(
    env: MultirotorTrajEnv, traj: Trajectory,
    ctrl: Union[Controller, PPO, None],
    max_steps=2_000, relative=None, verbose=False
) -> DataLog:
    if not isinstance(traj, Trajectory):
        traj = Trajectory(env.vehicle, traj, proximity=env._proximity, resolution=env.bounding_box/2)
    log = DataLog(env.vehicle, ctrl if isinstance(ctrl, Controller) else env.ctrl,
                  other_vars=('reward', 'speeds'))
    if isinstance(ctrl, PPO):
        pidctrl = env.ctrl
        predict_fn = lambda x: ctrl.predict(x, deterministic=True)[0]
    elif ctrl is None or isinstance(ctrl, Controller):
        ctrl = pidctrl = env.ctrl
        predict_fn = lambda x: env.normalize_action(np.zeros(env.action_space.shape, env.vehicle.dtype))
    
    if verbose:
        iterator = tqdm(enumerate(traj), leave=False, total=max_steps // env.steps_u)
    else:
        iterator = enumerate(traj)
    for i, (pos, feed_forward_vel) in iterator:
        # Get prescribed normalized action for system as thrust and torques
        action = predict_fn(env.state)
        # Send speeds to environment
        state, r, done, *_, i = env.step(action)
        log.log(reward=r, speeds=env.vehicle.speeds)
        if done:
            if verbose:
                print(i)
            break

    log.done_logging()
    return log



def run_trajectory(
    env: MultirotorTrajEnv, traj: Union[Trajectory, Iterable], ctrl=None, verbose=False, reset_zero=True, log=None
) -> DataLog:
    if reset_zero:
        env.reset(uav_x=np.zeros(12)) # resets controller too
    if log is None:
        old_wp = np.zeros(3)
        pos_global = env.vehicle.position
    else:
        old_wp = log.target.position[-1]
        pos_global = log.position[-1]
        env.vehicle.x = log.states[-1]
        env.x = env.vehicle.x
    if not isinstance(traj, Trajectory):
        traj = Trajectory(env.vehicle, traj, proximity=2, resolution=env.bounding_box/2)
    params = traj.get_params()
    points = traj.generate_trajectory(curr_pos=pos_global)[1:]
    for wp in points:
        wp_rel = wp - old_wp
        pos = env.vehicle.position - wp_rel
        state = np.concatenate((pos, env.vehicle.state[3:]))
        env.reset(state)
        t = Trajectory(env.vehicle, [[0,0,0]], **params)
        l = run_sim(env, t, env.ctrl if ctrl is None else ctrl, verbose=verbose)
        if len(l):
            l.position[:] += old_wp + wp_rel
            l.target.position[:, :3] += old_wp + wp_rel
        else:
            continue
        if log is None:
            log = l
        else:
            log.append(l, relative=True)
        old_wp = wp
    return log