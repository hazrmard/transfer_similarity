import math

from gym.envs.box2d.lunar_lander import (
    LunarLander, LunarLanderContinuous,
    MAIN_ENGINE_POWER, SIDE_ENGINE_POWER,
    SIDE_ENGINE_AWAY, SIDE_ENGINE_HEIGHT,
    SCALE, VIEWPORT_H, VIEWPORT_W, FPS, LEG_DOWN
)
import numpy as np

FPS = 25


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