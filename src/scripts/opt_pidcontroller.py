import os
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from typing import Callable

import numpy as np
import optuna
from systems.multirotor import Multirotor, MultirotorTrajEnv, VP
from multirotor.trajectories import Trajectory
from multirotor.helpers import DataLog
from multirotor.controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller
)

from .setup import local_path

DEFAULTS = Namespace(
    ntrials = 100,
    nprocs = 5,
    bounding_box = 20.,
    max_velocity = 5.,
    max_acceleration = 10.,
    max_tilt = np.pi / 12,
    max_angular_acc = 5.,
    num_sims = 5,
    study_name = 'MultirotorPIDController',
)

def get_study(study_name: str):
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, direction='minimize',
        storage=storage_name,
        load_if_exists=True
    )
    return study



def run_sim(
    env, traj: 'multirotor.trajectories.Trajectory',
    ctrl_fn: Callable[[np.ndarray], np.ndarray],
    max_steps=2_000
) -> DataLog:
    log = DataLog(env.vehicle, env.ctrl,
                  other_vars=())
        
    for i, (pos, feed_forward_vel) in enumerate(traj):
        # Get prescribed normalized action for system as thrust and torques
        action = ctrl_fn(env.state)
        # Send speeds to environment
        state, r, done, *_ = env.step(action)
        log.log()
        if done or i >=max_steps-1:
            break

    log.done_logging()
    return log



def get_controller(m: Multirotor, args: Namespace=DEFAULTS):
    assert m.simulation.dt <= 0.1, 'Simulation time step too large.'
    pos = PosController( # PD
        0.8, 0., 3.75, 100., vehicle=m,
        max_velocity=args.max_velocity,
        max_acceleration=args.max_acceleration,
        square_root_scaling=False
    )
    vel = VelController( # P
        1, 0., 0, 10.,
        vehicle=m,
        max_tilt=args.max_tilt)
    att = AttController( # P
        1., 0, 0.,
        1., vehicle=m)
    rat = RateController( # PD
        4, 0, 40,
        0.5,
        max_acceleration=args.max_angular_acc,
        vehicle=m)

    alt = AltController(
        1, 0, 0,
        1, vehicle=m,
        max_velocity=args.max_velocity)
    alt_rate = AltRateController(
        10, 0, 0,
        1, vehicle=m)

    ctrl = Controller(
        pos, vel, att, rat, alt, alt_rate,
        interval_p=0.1, interval_a=0.01, interval_z=0.1
    )
    return ctrl


def make_objective(args: Namespace=DEFAULTS):
    env_kwargs = dict(
        bounding_box=args.bounding_box,
        seed=0,
        get_controller_fn=lambda m: get_controller(m, args)
    )
    def objective(trial: optuna.Trial):
        pos_p = trial.suggest_float('pos_p', 0.1, 50)
        pos_d = trial.suggest_float('pos_d', 1, 250)
        rat_p = trial.suggest_float('rat_p', 0.1, 50)
        rat_d = trial.suggest_float('rat_d', 1, 250)
        
        env = MultirotorTrajEnv(**env_kwargs)
        env.ctrl.ctrl_p.k_p[:] = pos_p
        env.ctrl.ctrl_p.k_d[:] = pos_d
        env.ctrl.ctrl_r.k_p[:] = rat_p
        env.ctrl.ctrl_r.k_d[:] = rat_d
        errs = []
        for i in range(args.num_sims):
            env.reset()
            waypoints = np.asarray([[0,0,0]])
            traj = Trajectory(env.vehicle, waypoints, proximity=0.1)
            log = run_sim(
                env, traj,
                lambda _: np.zeros(3, env.vehicle.dtype),
                max_steps=env.period//env.dt
            )
            errs.append(np.linalg.norm(log.position[-1]))
        return np.mean(errs)
    return objective



def optimize(study: optuna.Study, args: Namespace=DEFAULTS):
    study.optimize(make_objective(args), n_trials=args.ntrials//args.nprocs)
    return study



def get_controller_from_params(**params):
    pass


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('study_name', help='Name of study', default=DEFAULTS.study_name, type=str, nargs='?')
    parser.add_argument('--nprocs', help='Number of processes.', default=DEFAULTS.nprocs, type=int)
    parser.add_argument('--ntrials', help='Number of trials.', default=DEFAULTS.ntrials, type=int)
    parser.add_argument('--max_velocity', default=DEFAULTS.max_velocity, type=float)
    parser.add_argument('--max_acceleration', default=DEFAULTS.max_acceleration, type=float)
    parser.add_argument('--max_angular_acc', default=DEFAULTS.max_angular_acc, type=float)
    parser.add_argument('--max_tilt', default=DEFAULTS.max_tilt, type=float)
    parser.add_argument('--bounding_box', default=DEFAULTS.bounding_box, type=float)
    parser.add_argument('--num_sims', default=DEFAULTS.num_sims, type=int)
    parser.add_argument('--append', action='store_true', default=False)
    args = parser.parse_args()

    if not args.append:
        try:
            os.remove(local_path / (args.study_name + '.db'))
        except OSError:
            pass

    study = get_study(args.study_name)

    for key in vars(args):
        study.set_user_attr(key, getattr(args, key))

    with mp.Pool(args.nprocs) as pool:
        pool.starmap(optimize, [(study, args) for _ in range(args.nprocs)])
