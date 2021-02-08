# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Wrapper for creating the ant environment in gym_mujoco."""

import math
import numpy as np
import mujoco_py
from gym import utils
from gym.envs.mujoco import mujoco_env
from util.rl import discounted_reward

# maze_id = 'Maze'


def q_inv(a):
    return [a[0], -a[1], -a[2], -a[3]]


def q_mult(a, b):  # multiply two quaternion
    w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    i = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    j = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    k = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return [w, i, j, k]


class AntEnvOrigin(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, goal_region):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        self._goal_region = goal_region

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def cum_reward(self, sarss):
        return (max([self._goal_region.reward(s) for s, _, _, _ in sarss])
                + discounted_reward(sarss, gamma=0.99))


class AntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    FILE = "ant.xml"
    ORI_IND = 3

    def __init__(self, start_region, maze_id, start_state=None, file_path=None,
                 expose_all_qpos=True, expose_body_coms=None, expose_body_comvels=None):
        self._start_region = start_region
        self._maze_id = maze_id
        self._start_state = start_state
        self._expose_all_qpos = expose_all_qpos
        self._expose_body_coms = expose_body_coms
        self._expose_body_comvels = expose_body_comvels
        self._body_com_indices = {}
        self._body_comvel_indices = {}

        mujoco_env.MujocoEnv.__init__(self, file_path, 5)
        utils.EzPickle.__init__(self)

    @property
    def physics(self):
        # check mujoco version is greater than version 1.50 to call correct physics
        # model containing PyMjData object for getting and setting position/velocity
        # check https://github.com/openai/mujoco-py/issues/80 for updates to api
        if mujoco_py.get_version() >= '1.50':
            return self.sim
        else:
            return self.model

    def get_state(self):
        if self._maze_id == 'Maze':
            return self.state_vector()[:15]
        elif self._maze_id == 'Push' or self._maze_id == 'Fall':
            return self.state_vector()[:17]

    def _step(self, a):
        return self.step(a)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)

        # stop condition is handled in the wrapper
        return self._get_obs(), 0, False, None

    def _get_obs(self):
        # No cfrc observation
        if self._maze_id == 'Maze':
            obs = np.concatenate([
                self.physics.data.qpos.flat[:15],  # Ensures only ant obs.
                self.physics.data.qvel.flat[:14],
            ])
        elif self._maze_id == 'Push' or self._maze_id == 'Fall':
            obs = np.concatenate([
                self.physics.data.qpos.flat[:17],
                self.physics.data.qvel.flat[:16],
            ])
        else:
            raise Exception('Check Maze ID!')

        return obs

    def reset(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1

        # Set everything other than ant to original position and 0 velocity.
        qpos[15:] = self.init_qpos[15:]
        qvel[14:] = 0.

        # sample from start_state
        if self._start_state is None:
            start_state = self._start_region.sample()
        else:
            start_state = self._start_state

        if self._maze_id == 'Maze':
            if start_state.size >= 29:
                qpos[:15] = start_state[:15]
                qvel[:14] = start_state[15:29]
            else:
                qpos[:2] = start_state[:2]
        if self._maze_id == 'Push' or self._maze_id == 'Fall':
            if start_state.size >= 33:
                qpos[:17] = start_state[:17]
                qvel[:16] = start_state[17:33]
            else:
                qpos[:2] = start_state[:2]
                qpos[15:17] = start_state[2:4]
                if len(start_state) >= 5:
                    qpos[2] = start_state[4]

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def get_ori(self):
        ori = [0, 1, 0, 0]
        # take the quaternion
        rot = self.physics.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[
            1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    def set_xy(self, xy):
        qpos = np.copy(self.physics.data.qpos)
        qpos[0] = xy[0]
        qpos[1] = xy[1]

        qvel = self.physics.data.qvel
        self.set_state(qpos, qvel)

    def get_xy(self):
        return self.physics.data.qpos[:2]

    def set_start_region(self, start_region):
        self._start_region = start_region

    def set_start_state(self, start_state):
        self._start_state = start_state
