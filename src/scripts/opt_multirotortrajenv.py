import sys
import os
import multiprocessing as mp
from argparse import ArgumentParser, Namespace

import optuna
import numpy as np
from rl import learn_rl, evaluate_rl
from systems.multirotor import Multirotor, MultirotorTrajEnv, VP
from multirotor.controller import (
    AltController, AltRateController,
    PosController, AttController,
    VelController, RateController,
    Controller
)

from .opt_pidcontroller import get_study, get_controller as get_controller_base
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
    study_name = 'MultirotorTrajEnv',
)



def get_controller(m: Multirotor, args: Namespace=DEFAULTS):
    ctrl = get_controller_base(m, args=args)
    # Taken from PID optimization script, given the default controller params
    ctrl.ctrl_p.k_p[:] = 18.28
    ctrl.ctrl_p.k_d[:] = 75
    ctrl.ctrl_r.k_p[:] = 22.97
    ctrl.ctrl_r.k_d[:] = 1.81
    return ctrl


def make_objective(args: Namespace=DEFAULTS):
    max_steps = 50_000
    def objective(trial: optuna.Trial):
        env_kwargs = dict(
            steps_u = trial.suggest_int('steps_u', 1, 50, step=5),
            scaling_factor = trial.suggest_float('scaling_factor', 0.05, 1., step=0.01),
            seed=0,
            get_controller_fn=lambda m: get_controller(m, args),
            vp = VP
        )
        env = MultirotorTrajEnv(**env_kwargs)
        ep_len = env.period//(env.dt * env.steps_u)
        steps = max_steps
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)
        n_steps = trial.suggest_int('n_steps', ep_len, 5 * ep_len, step=100)
        n_epochs = trial.suggest_int('n_epochs', 1, 5)
        batch_size = trial.suggest_int('batch_size', 32, min(256, n_steps), step=32)
        learn_kwargs = dict(
            steps = steps,
            n_steps = n_steps,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            batch_size = batch_size,
            seed=0,
            log_env_params = ('steps_u', 'scaling_factor'),
            tensorboard_log = env.name + ('/optstudy/%03d' % trial.number),
            policy_kwargs=dict(squash_output=False,
                                net_arch=[dict(pi=[128,128], vf=[128,128])])
        )
        agent = learn_rl(env, **learn_kwargs)
        agent.save(agent.logger.dir + '/agent')
        rew, std, time = evaluate_rl(agent, MultirotorTrajEnv(**env_kwargs), args.num_sims)
        return rew
    return objective



def optimize(study: optuna.Study, args: Namespace=DEFAULTS):
    study.optimize(make_objective(args), n_trials=args.ntrials//args.nprocs)
    return study



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
        import shutil
        path = local_path / 'tensorboard/MultirotorTrajEnv/optstudy'
        shutil.rmtree(path=path, ignore_errors=True)
        try:
            os.remove(local_path / (args.study_name + '.db'))
        except OSError:
            pass
        
    study = get_study(args.study_name)

    for key in vars(args):
        study.set_user_attr(key, getattr(args, key))

    mp.set_start_method('spawn')
    with mp.Pool(args.nprocs) as pool:
        pool.starmap(optimize, [(study, args) for _ in range(args.nprocs)])