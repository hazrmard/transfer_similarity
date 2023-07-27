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
        # NOTE: This is not a pure function due to setting m.speeds
        def update_fn(t, x, u, params):
            speeds = m.allocate_control(u[0], u[1:4])
            speeds = np.clip(speeds, a_min=0, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
            disturb_forces=disturbance_fn(m))
            m.speeds = speeds
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
    # NOTE: This is not a pure function due to setting m.speeds
    elif kind=='waypoints':
        inputs=['x','y','z','yaw']
        def update_fn(t, x, u, params):
            dynamics = ctrl.step(u, ref_is_error=False)
            speeds = m.allocate_control(dynamics[0], dynamics[1:4])
            speeds = np.clip(speeds, a_min=0, a_max=max_rads)
            dxdt = m.dxdt_speeds(t, x.astype(m.dtype), speeds,
                disturb_forces=disturbance_fn(m))
            m.speeds = speeds
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
        max_rads: float=DEFAULTS.max_rads,
        # length of cube centered at origin within which position is initialized,
        # half length of cube centered at origin within which vehicle may move
        bounding_box: float=DEFAULTS.bounding_box,
        random_disturbance_direction=False,
        multirotor_class=Multirotor, multirotor_kwargs={}
    ):
        system, extra = create_multirotor(
            vp, sp, xformA=xformA, xformB=xformB,
            return_mult_ctrl=True,
            get_controller_fn=get_controller_fn,
            disturbance_fn=disturbance_fn,
            kind='waypoints',
            max_rads=max_rads,
            multirotor_class=multirotor_class,
            multirotor_kwargs=multirotor_kwargs
        )
        self.vehicle: Multirotor = extra['multirotor']
        self.ctrl: Controller = extra['ctrl']
        self.disturbance_fn = disturbance_fn
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
        self.ekf = getattr(self.vehicle, 'ekf', False)
        self.random_disturbance_direction = random_disturbance_direction
        self.max_rads = max_rads
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
        if self.ekf:
            x = np.asarray(self.ekf.x, self.dtype)
        else:
            x = self.x
        return self.normalize_state(x)


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
        self._des_unit_vec = (0 - pos) / (np.linalg.norm(pos) + 1e-6)
        if self.random_disturbance_direction:
            pass
            # self._orig_dist_fn = self.vehicle.
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



def run_sim(
    env: MultirotorTrajEnv, traj: Trajectory,
    ctrl: Union[Controller, PPO, None],
    max_steps=2_000, relative=None, verbose=False,
    other_vars=()
) -> DataLog:
    if not isinstance(traj, Trajectory):
        traj = Trajectory(env.vehicle, traj, proximity=env._proximity, resolution=env.bounding_box/2)
    log = DataLog(env.vehicle, ctrl if isinstance(ctrl, Controller) else env.ctrl,
                  other_vars=('reward', 'speeds', *other_vars))
    if getattr(env, 'ekf', False):
        ekflog = DataLog(env.ekf)
        setattr(log, 'ekf', ekflog)
    else:
        ekflog = False

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
        # v = dict(reward=r, speeds=env.vehicle.speeds)
        # v.update({k})
        log.log(reward=r, speeds=env.vehicle.speeds)
        if ekflog:
            ekflog.log()
        if done:
            if verbose:
                print(i)
            break
    log.done_logging()
    if ekflog:
        ekflog.done_logging()
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
    if env.ekf:
        env.ekf.state = env.vehicle.state.copy()
    if not isinstance(traj, Trajectory):
        traj = Trajectory(env.vehicle, traj, proximity=2, resolution=env.bounding_box/2)
    params = traj.get_params()
    points = traj.generate_trajectory(curr_pos=pos_global)[1:]
    for wp in tqdm(points, leave=False):
        wp_rel = wp - old_wp
        pos = env.vehicle.position - wp_rel
        state = np.concatenate((pos, env.vehicle.state[3:]))
        env.reset(state)
        if env.ekf:
            state = env.ekf.x.copy()    # get curent state
            env.ekf.vehicle.reset()     # reset vehicle
            env.ekf.x[:3] = env.ekf.x[:3] - wp_rel # set new relative position
            env.ekf.x[3:] = state[3:]   # restore other variables (vel, angle)
            env.ekf.vehicle.state[:] = env.ekf.x # replicate state in vehicle model
        t = Trajectory(env.vehicle, [[0,0,0]], **params)
        l = run_sim(env, t, env.ctrl if ctrl is None else ctrl, verbose=verbose)
        if len(l):
            l.position[:] += old_wp + wp_rel
            l.target.position[:, :3] += old_wp + wp_rel
            if env.ekf:
                l.ekf.position[:] += old_wp + wp_rel
        else:
            continue
        if log is None:
            log = l
        else:
            log.append(l, relative=True)
            if env.ekf:
                log.ekf.append(l.ekf, relative=True)
        old_wp = wp
    return log