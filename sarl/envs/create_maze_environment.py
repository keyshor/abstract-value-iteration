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
from sarl.envs.ant import AntEnvOrigin
from sarl.envs.ant_maze_env import AntMazeEnv


def create_ant_env(goal_region):
    return AntEnvOrigin(goal_region)


def create_maze_env(start_region, final_region, goal_region, env_name,
                    top_down_view=False, max_timesteps=500, start_state=None,
                    use_shaped_reward=False):
    n_bins = 0
    manual_collision = False
    if env_name.startswith('Ant'):
        cls = AntMazeEnv
        env_name = env_name[3:]
        maze_size_scaling = 8
    else:
        assert False, 'unknown env %s' % env_name

    maze_id = None
    observe_blocks = False
    put_spin_near_agent = False
    if env_name == 'Maze':
        maze_id = 'Maze'
    elif env_name == 'Push':
        maze_id = 'Push'
    elif env_name == 'Fall':
        maze_id = 'Fall'
    elif env_name == 'Block':
        maze_id = 'Block'
        put_spin_near_agent = True
        observe_blocks = True
    elif env_name == 'BlockMaze':
        maze_id = 'BlockMaze'
        put_spin_near_agent = True
        observe_blocks = True
    else:
        raise ValueError('Unknown maze environment %s' % env_name)

    gym_mujoco_kwargs = {
        'maze_id': maze_id,
        'n_bins': n_bins,
        'observe_blocks': observe_blocks,
        'put_spin_near_agent': put_spin_near_agent,
        'top_down_view': top_down_view,
        'manual_collision': manual_collision,
        'maze_size_scaling': maze_size_scaling,
        'start_region': start_region,
        'final_region': final_region,
        'goal_region': goal_region,
        'max_timesteps': max_timesteps,
        'start_state': start_state,
        'use_shaped_reward': use_shaped_reward
    }
    gym_env = cls(**gym_mujoco_kwargs)
    gym_env.reset()

    return gym_env
