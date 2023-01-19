from typing import Literal

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
    motor=MP
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
SP = SimulationParams(dt=0.001, g=9.81)



def get_controller(m: Multirotor, max_velocity=5., max_acceleration=3.):
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
    kind: Literal['speeds', 'dynamics']='dynamics',
    max_rads: float=700
):
    m = Multirotor(vp, sp)
    ctrl = None
    if kind=='dynamics':
        def update_fn(t, x, u, params):
            speeds = m.allocate_control(u[0], u[1:4])
            speeds = np.clip(speeds, a_min=-max_rads, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds)
            return dxdt
    elif kind=='speeds':
        raise NotImplementedError # TODO
    elif kind=='waypoints':
        ctrl = get_controller(m)
        def update_fn(t, x, u, params):
            dynamics = ctrl.step(x, ref_is_error=True)
            speeds = m.allocate_control(dynamics[0], dynamics[1:4])
            speeds = np.clip(speeds, a_min=-max_rads, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds)
            return dxdt
        raise NotImplementedError # TODO

    sys = control.NonlinearIOSystem(
        updfcn=update_fn,
        inputs=['fz','tx','ty','tz'],
        states=['x','y','z',
                'vx','vy','vz',
                'roll','pitch','yaw',
                'xrate', 'yrate', 'zrate']
    )
    if return_mult_ctrl:
        return sys, dict(multirotor=m, ctrl=ctrl)
    return sys



class MultirotorEnv(SystemEnv):


    def __init__(
        self, vp=VP, sp=SP,
        q=np.diagflat([1,1,1,1,1,1,0.1,0.1,1,0.1,0.1,1]),
        r = np.diagflat([1,1,1,100]) * 1e-4,
        dt=None, seed=None,
        max_angular_acc=1.0,
        xformA=np.eye(12), xformB=np.eye(4),
        state_buffer_size: int=0,
        unnormalize: bool=True,
        clip: bool=True
    ):
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True
        )
        self.mult = extra['multirotor']
        super().__init__(system=system, q=q, r=r, dt=sp.dt, seed=seed, dtype=np.float32)
        self.state_buffer_size = state_buffer_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(12 * (1 + state_buffer_size),), dtype=self.dtype
        )
        self.state_buffer = deque(maxlen=self.state_buffer_size+1)
        self.action_space = gym.spaces.Box(
            # thrust, torque_x, torque_y, torque_z
            low=-1, high=1, shape=(4,), dtype=self.dtype
        )
        self.period = int(5 / sp.dt)
        self.nominal_thrust = vp.mass * sp.g
        self.max_angular_acc = max_angular_acc
        self.max_torque = min(
            np.abs(vp.inertia_matrix.diagonal()) * self.max_angular_acc
        )
        self._min = np.asarray([
            -1,
            0, 0, 0
        ]).astype(self.dtype)
        self._range = np.asarray([
            self.nominal_thrust,
            self.max_torque, self.max_torque, self.max_torque
        ]).astype(self.dtype)
        self._min_dyn = -np.asarray([
            0,
            self.max_torque, self.max_torque, self.max_torque
        ]).astype(self.dtype)
        self._max_dyn = np.asarray([
            (1-self._min[0]) * self.mult.weight,
            self.max_torque, self.max_torque, self.max_torque
        ]).astype(self.dtype)
        self.unnormalize = unnormalize
        self.clip = clip


    @property
    def state(self) -> np.ndarray:
        return np.concatenate(self.state_buffer, dtype=self.dtype)
    @state.setter
    def state(self, x: np.ndarray):
        x = np.asarray(x, dtype=self.dtype)
        for i in range(self.state_buffer_size + 1):
            self.state_buffer.append(x)
        self.x = np.asarray(x, dtype=self.dtype)


    def reset(self, x=None):
        super().reset(x)
        pos = np.asarray(x[0:3]) if x is not None else \
              self.random.uniform(-0.2, 0.2, size=3)
        vel = np.asarray(x[3:6]) if x is not None else \
              self.random.uniform(-0.05, 0.05, size=3)
        ori = np.asarray(x[6:9]) if x is not None else \
              self.random.uniform(-0.2, 0.2, size=3)
        rat = np.asarray(x[9:12]) if x is not None else \
              self.random.uniform(-0.05, 0.05, size=3)
        self.x = np.concatenate((pos, vel, ori, rat),
                                dtype=self.dtype)
        # fill state buffer with current state, ejecting last states
        self.state = self.x
        return self.x


    def step(self, u: np.ndarray):
        # convert [-1,1] to range of actions in dynamics space, and clip optionally
        if self.unnormalize:
            # action is in [-1,1] space and needs to be converted to dynamics
            if self.clip:
                u = np.clip(u, a_min=-1, a_max=1)
            u = (u - self._min) * self._range
        elif self.clip:
            # action is already as dynamics space, clip to max torque
            u = np.clip(u, a_min=self._min_dyn, a_max=self._max_dyn)
        x, r, d, *_, i = super().step(u)
        self.state_buffer.append(x)
        # since SystemEnv is only using Multirotor.dxdt methods, the mult
        # object is not integrating the change in state. Therefore manually
        # set the state on the self.mult object to reflect the integration
        # that SystemEnv.step() does
        self.mult.state = x
        done = (self.n >= self.period)
        return x, r, done, *_, i



class ControlledMultirotorEnv(MultirotorEnv):

    
    def __init__(
        self, vp, sp, q=np.eye(12), dt=None, seed=None,
        max_angular_acc=1.0,
        xformA=np.eye(12), xformB=np.eye(4),
        state_buffer_size: int=0,
        ctrl_state: bool=False

    ):
        super().__init__(
            vp, sp, q=q, dt=dt, seed=seed,
            max_angular_acc=max_angular_acc,
            xformA=xformA, xformB=xformB,
            state_buffer_size= state_buffer_size
        )
        self.ctrl = get_controller(self.mult)
        self.ctrl_state = ctrl_state
        self.observation_space = gym.spaces.Box(
            # pos, vel, ori, rat, pid errors...
            low=-np.inf, high=np.inf,
            shape=((1+state_buffer_size) * 12 + (18 if ctrl_state else 0),),
            dtype=self.dtype
        )
        self.action_space = gym.spaces.Box(
            # x,y,z,yaw
            low=-1, high=1, shape=(4,), dtype=self.dtype
        )


    @property
    def state(self):
        if self.ctrl_state:
            return np.concatenate([
                super().state, self.ctrl.state
            ], dtype=self.dtype)
        return super().state
    @state.setter
    def state(self, x: np.ndarray):
        x = np.asarray(x, dtype=self.dtype)
        for _ in range(self.state_buffer_size + 1):
            self.state_buffer.append(x[:12])
        self.x = x[:12]


    def reward(self, xold, u, x):
        return super().reward(None, u, x[:12])


    def reset(self, x=None):
        self.ctrl.reset()
        self.x = super().reset(x)
        self.x[1:] = 0
        # the exported state is a combination of the multirotor and controller
        return self.state, {}


    def step(self, u: np.ndarray):
        dynamics = self.ctrl.step(u, ref_is_error=True)
        # action is already in dynamics space, but needs to be clipped to the
        # environment's max torque/thrust bounds
        x, r, d, *_, i = super().step(dynamics, unnormalize=False, clip=True)
        return self.state, r, d, False, {'u': u}




class TwoDimWrapper(MultirotorEnv):
    
    def __init__(self, *args, condition=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.condition = condition
        if self.condition=='pitch':
            pitch = self.q[7,7]
            self.q[:] = 0
            self.q[7,7] = pitch
        elif self.condition=='pitchvel':
            pitch = self.q[7,7]
            pitchvel = self.q[10,10]
            self.q[:] = 0
            self.q[7,7] = pitch
            self.q[7,7] = pitchvel
        elif self.condition=='rollpitch':
            roll = self.q[6,6]
            pitch = self.q[7,7]
            self.q[:] = 0
            self.q[6,6] = roll
            self.q[7,7] = pitch
        else:
            self.q[1,1] = 0 # keep x,z position
            self.q[4,4] = 0 # keep x,z velocity
            self.q[6,6] = 0 # only keep pitch
            self.q[8,8] = 0 # only keep pitch
            self.q[9,9] = 0 # only keep pitch rate
            self.q[11,11] = 0 # only keep pitch rate
    
    def step(self, u):
        # only allow thrust and pitch
        if self.condition=='pitch' or self.condition == 'pitchvel':
            u[0] = 0.
            u[1] = 0.
            u[3] = 0.
        elif self.condition=='rollpitch':
            u[0] = 0
            u[3] = 0
        return super().step(u)

    def reset(self, x=None):
        x = super().reset(x)
        if self.condition=='zero':
            x[:] = 0
        elif self.condition=='pitch':
            for i in range(x.shape[0]):
                if i!=7: x[i]=0 # pitch=7
        elif self.condition=='pitchvel':
            for i in range(x.shape[0]):
                if i!=7 or i!=10: x[i]=0 # pitch=7, pitchvel=10
        elif self.condition=='rollpitch':
            for i in range(x.shape[0]):
                if i!=6 and i!=7:
                    x[i]=0 # pitch=7, roll=6
        else:
            x[1] = 0
            x[4] = 0
            x[6] = 0
            x[8] = 0
            x[9] = 0
            x[11] = 0
        self.x = x
        return self.x, {}