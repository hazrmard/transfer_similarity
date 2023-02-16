import sys
import multiprocessing as mp

import numpy as np
import optuna
from systems.multirotor import MultirotorTrajEnv
from multirotor.trajectories import Trajectory
from multirotor.helpers import run_sim

study_name = 'MultirotorPIDController'
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name=study_name, direction='minimize',
    storage=storage_name,
    load_if_exists=True
)


def objective(trial: optuna.Trial):
    env_kwargs = dict(
        scaling_factor = trial.suggest_float('scaling_factor', 0.01, 1.),
        seed=0
    )
    pos_p = trial.suggest_float('pos_p', 0.1, 10)
    pos_d = trial.suggest_float('pos_d', 1, 50)
    rat_p = trial.suggest_float('rat_p', 0.1, 10)
    rat_d = trial.suggest_float('rat_d', 1, 50)
    
    env = MultirotorTrajEnv(**env_kwargs)
    env.ctrl.ctrl_p.k_p[:] = pos_p
    env.ctrl.ctrl_p.k_d[:] = pos_d
    env.ctrl.ctrl_r.k_p[:] = rat_p
    env.ctrl.ctrl_r.k_d[:] = rat_d
    env.reset(np.asarray([1,0,0,0,0,0,0,0,0,0,0,0]))
    waypoints = np.asarray([[0,0,0]])
    traj = Trajectory(env.vehicle, waypoints, proximity=0.1)
    log = run_sim(env, traj, lambda x: np.zeros_like(env.action_space), max_steps=1000)
    return np.mean(
        np.linalg.norm(log.position, axis=1)
    )


def optimize(n_trials):
    study.optimize(objective, n_trials=n_trials)


if __name__=='__main__':
    nprocs = 1
    if len(sys.argv) > 1:
        nprocs = int(sys.argv[1])
    ntrials = 100
    if len(sys.argv) > 2:
        ntrials = int(sys.argv[2])
    with mp.Pool(nprocs) as pool:
        pool.map(optimize, [ntrials // nprocs for _ in range(nprocs)])
