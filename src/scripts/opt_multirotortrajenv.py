import sys
import os
import multiprocessing as mp

import optuna
from rl import learn_rl, evaluate_rl
from systems.multirotor import MultirotorTrajEnv


study_name = 'MultirotorTrajEnv'
storage_name = "sqlite:///{}.db".format(study_name)
study = optuna.create_study(
    study_name=study_name, direction='maximize',
    storage=storage_name,
    load_if_exists=True
)

def objective(trial: optuna.Trial):
    max_steps = 200_000
    env_kwargs = dict(
        steps_u = trial.suggest_int('steps_u', 1, 50),
        scaling_factor = trial.suggest_float('scaling_factor', 0.01, 1.),
        seed=0
    )
    env = MultirotorTrajEnv(**env_kwargs)
    ep_len = env.period//(env.dt * env.steps_u)
    learn_kwargs = dict(
        steps = max_steps // env.steps_u,
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
        n_steps = trial.suggest_int('n_steps', ep_len, 5 * ep_len, step=100),
        n_epochs = trial.suggest_int('n_epochs', 1, 10),
        batch_size = trial.suggest_int('batch_size', 32, 256, step=32),
        policy_kwargs=dict(squash_output=True,
                           net_arch=[dict(pi=[128,128], vf=[128,128])]),
        seed=0,
        log_env_params = ('steps_u', 'scaling_factor'),
        tensorboard_log = env.name + '/optstudy'
    )
    agent = learn_rl(env, **learn_kwargs)
    rew, std, time = evaluate_rl(agent, MultirotorTrajEnv(**env_kwargs), 10)
    return rew



def optimize(n_trials):
    study.optimize(objective, n_trials=n_trials)



if __name__=='__main__':
    nprocs = 1
    if len(sys.argv) > 1:
        nprocs = int(sys.argv[1])
    ntrials = 100
    if len(sys.argv) > 2:
        ntrials = int(sys.argv[2])
    mp.set_start_method('spawn')
    with mp.Pool(nprocs) as pool:
        pool.map(optimize, [ntrials // nprocs for _ in range(nprocs)])