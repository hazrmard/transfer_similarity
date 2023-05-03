import math

from gym.envs.box2d.lunar_lander import (
    LunarLander, LunarLanderContinuous,
    MAIN_ENGINE_POWER, SIDE_ENGINE_POWER,
    SIDE_ENGINE_AWAY, SIDE_ENGINE_HEIGHT,
    SCALE, VIEWPORT_H, VIEWPORT_W, FPS, LEG_DOWN
)
import numpy as np
import gym
import control

from .base import SystemEnv

FPS = 25



def create_lunarlander(
    mass = 1,
    width=1, height=2,
    sheight = 1.,
    relative_power = np.asarray([1., 1.], dtype=np.float32),
    off_thresh = 0.5,
    side_power = 1.,
    main_power = 15,
    use_torch: bool=False
  ):
    gravity = 10.
    inertia = (1/12) * mass * (width**2 + height**2)
    weight = mass * gravity

    def dxdt(t, x, u, *params):
        c,s = np.cos(x[4]), np.sin(x[4])
        main, side = u[0], u[1]
        if abs(side) < off_thresh:
            side = 0
        direction = np.sign(u[1])
        side = np.clip(side, -1, 1) * side_power * relative_power[0 if direction==-1 else 1]
        main = (np.clip(main, 0, 1) + 1) * 0.5 * main_power
        
        dx = np.zeros_like(x)
        # position (2)
        # x',y'= = velocity
        dx[:2] = x[2:4]
        # velocity (2)
        # x''=(thrust*cos(angle)+(-rside*cos(angle))+(lside*cos(angle))/mass
        # y''=thrust*sin(angle)+(rside*sin(angle))+(-lside*sin(angle))) / mass
        dx[2] = ((main * -s) + (-side*c)) / mass
        dx[3] = ((main * c) - weight + (-side*s)) / mass
        # angle (1)
        # w' = angular vel
        dx[4] = x[5]
        # angular vel (1)
        # w'' = (rside-lside)*sheight) / inertia
        dx[5] = side * sheight / inertia
        # legs (2)
        return dx

    sys = control.NonlinearIOSystem(dxdt, None, name='LanderSys',
                                   inputs=2, outputs=8, states=8)
    return sys


class LanderEnv(SystemEnv):


    def __init__(self, *args, system=None, dt=0.1, seed=0, **kwargs):
        system = create_lunarlander(**kwargs) if system is None else system
        # x, y, vx, vy, angle, angle rate, leg1 contact, leg2 contact
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32, seed=seed)
        # main thrust, side thrust
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32, seed=seed)
        self.init_y = 50
        self.max_land_tilt = np.arctan(0.5)
        self.period = 5 * np.sqrt(2 * self.init_y / 2.5) / dt
        super().__init__(system, dt=dt, seed=seed)


    def reset(self, x=None):
        super().reset(x)
        self.prev_shaping = None
        if x is None:
            x = np.zeros(8, np.float32)
            x[1] = self.init_y
            x[2:4] = self.random.uniform(-10, 10, size=2)
        else:
            x = np.asarray(x, np.float32)
        self.x = x
        return self.state


    @property
    def state(self):
        x = self.x
        return np.asarray([
            (x[0] - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (x[1] - (0 + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            x[2] * (VIEWPORT_W / SCALE / 2) * self.dt,
            x[3] * (VIEWPORT_H / SCALE / 2) * self.dt,
            x[4],
            20.0 * x[5] * self.dt,
            0.0,
            0.0,
        ], self.dtype)


    def reward(self, xold, u, x):
        shaping = (
            -1 * np.sqrt(x[0] * x[0] + x[1] * x[1])
            - 1 * np.sqrt(x[2] * x[2] + x[3] * x[3])
            - 1 * abs(x[4])
            + 0.10 * x[6]
            + 0.10 * x[7]
        )
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        else:
            reward = 0
        self.prev_shaping = shaping
        return reward

    def step(self, u, from_x=None, persist=True):
        x, r, d, *_, i = super().step(u, from_x=from_x, persist=persist)
        done = False
        if x[1] <=0:
            done = True
            if abs(x[0]) > 1 or abs(x[4]) > self.max_land_tilt:
                r -= 100
            else:
                r += 100
        elif self.n >= self.period:
            done = True
            r -= 100
        return self.state, r, done, *_, i



class LunarLanderEnv(LunarLanderContinuous):

    def __init__(self, seed=0):
        self.MAIN_ENGINE_POWER = MAIN_ENGINE_POWER
        self.SIDE_ENGINE_POWER = SIDE_ENGINE_POWER
        self.relative_power = np.asarray([1., 1.], dtype=np.float32)
        self.state = None
        self.name = 'LunarLanderEnv'
        self.dt = 1 / FPS
        super().__init__()
        self.seed(seed=seed)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)


    def reset(self):
        self.state = super().reset()
        return self.state


    def step(self, action):
        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float32)
        else:
            assert self.action_space.contains(action), "%r (%s) invalid " % (
                action,
                type(action),
            )

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (
            not self.continuous and action == 2
        ):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert m_power >= 0.5 and m_power <= 1.0
            else:
                m_power = 1.0
            ox = (
                tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            )  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(
                3.5,  # 3.5 is here to make particle speed adequate
                impulse_pos[0],
                impulse_pos[1],
                m_power,
            )  # particles are just a decoration
            p.ApplyLinearImpulse(
                (ox * self.MAIN_ENGINE_POWER * m_power, oy * self.MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * self.MAIN_ENGINE_POWER * m_power, -oy * self.MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (
            not self.continuous and action in [1, 3]
        ):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                SIDE_POWER = self.SIDE_ENGINE_POWER * self.relative_power[0 if direction==-1 else 1]
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert s_power >= 0.5 and s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            oy = -tip[1] * dispersion[0] - side[1] * (
                3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
            )
            impulse_pos = (
                self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
            )
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_POWER * s_power, oy * SIDE_POWER * s_power),
                impulse_pos,
                True,
            )
            self.lander.ApplyLinearImpulse(
                (-ox * SIDE_POWER * s_power, -oy * SIDE_POWER * s_power),
                impulse_pos,
                True,
            )

        self.world.Step(self.dt, 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) * self.dt,
            vel.y * (VIEWPORT_H / SCALE / 2) * self.dt,
            self.lander.angle,
            20.0 * self.lander.angularVelocity * self.dt,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ]
        assert len(state) == 8

        reward = 0
        shaping = (
            -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
            - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
            - 100 * abs(state[4])
            + 10 * state[6]
            + 10 * state[7]
        )  # And ten points for legs contact, the idea is if you
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Commenting out. Reward is a function of state only.
        # reward -= (
        #     m_power * 0.30
        # )  # less fuel spent is better, about -30 for heuristic landing
        # reward -= s_power * 0.03

        done = False
        if self.game_over or abs(state[0]) >= 1.0:
            done = True
            reward = -100
        if not self.lander.awake:
            done = True
            reward = +100
        self.state = np.array(state, dtype=np.float32)
        return self.state, reward, done, {}