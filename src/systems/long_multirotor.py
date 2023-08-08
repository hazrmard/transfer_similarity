from typing import Iterable

import numpy as np

from .multirotor import MultirotorTrajEnv



class LongTrajEnv:

    def __init__(self, waypoints: Iterable[np.ndarray], base_env: MultirotorTrajEnv):
        self.waypoints = waypoints
        self.base_env = base_env
        self.current_waypoint = None


    def reset(self):
        self.base_env.reset(uav_x=np.zeros(12, np.float32))
        self.current_waypoint_idx = 0
        normed_wp = self.waypoints[0] * 2 / (self.state_range[:3]+1e-6)
        return np.concatenate((self.base_env.state, normed_wp))


    def step(self, u: np.ndarray):
        # coming from a tanh NN policy function
        u = np.clip(u, a_min=-1., a_max=1.)
        u = self.base_env.unnormalize_action(u)
        u = u + self.waypoints[self.current_waypoint_idx]
        u = self.base_env.normalize_action(u)

        done = False
        reward = 0
        # TODO: return info dict with done flags
        s, r, _, info = self.base_env.step(u)
        if info.get('reached'):
            self.base_env.reset(uav_x=self.base_env.x)
            self.current_waypoint_idx += 1
            # if full traj is finished
            if self.current_waypoint_idx == len(self.waypoints):
                done = True
                normed_wp = self.waypoints[-1] * 2 / (self.state_range[:3]+1e-6)
            else:
                self.base_env._des_unit_vec = np.linalg.norm(self.waypoints[self.current_waypoint_idx] - 
                                               self.waypoints[self.current_waypoint_idx-1])
                normed_wp = self.waypoints[self.current_waypoint_idx] * 2 / (self.state_range[:3]+1e-6)
        s = np.concatenate((s, normed_wp))

        # reward calculation

        # done calculations
        if not done:
            # tipped, out of bounds from the info dict
            pass

        # state is a 12+3 element vector, where last 3
        # elements are the normalized next waypoint
        return s, reward, done, {}
    


def test():
    env = LongTrajEnv(
        waypoints=[[20,0,0], [40,0,0], [60,0,0]],
        MultirotorTrajEnv(**kwargs)
    )
    done = False
    while not done:
        state, reward, done, _ = env.step(env.waypoints[-1])
    # plotting logic
        