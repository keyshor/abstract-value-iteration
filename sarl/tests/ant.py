from sarl.envs.create_maze_environment import create_maze_env
from sarl.envs.maze_env_utils import AntMazeAbstractState
from sarl.envs.maze_env_utils import AntPushAbstractState
from sarl.envs.maze_env_utils import AntFallAbstractState
from util.rl import RandomPolicy, get_rollout

import numpy as np
import unittest


class TestAntMazeEnvironment(unittest.TestCase):

    def test_push_abstract_states(self):
        for i in range(5):
            s = AntPushAbstractState(i)
            maze_env = create_maze_env(s, s, s, 'AntPush')
            random_policy = RandomPolicy(
                8, np.array(maze_env.action_space.high))
            get_rollout(maze_env, random_policy, True, max_timesteps=200)

    def test_fall_abstract_states(self):
        for i in range(5):
            s = AntFallAbstractState(i)
            maze_env = create_maze_env(s, s, s, 'AntFall')
            random_policy = RandomPolicy(
                8, np.array(maze_env.action_space.high))
            get_rollout(maze_env, random_policy, True, max_timesteps=200)

    def test_maze_abstract_states(self):
        for i in range(4):
            s = AntMazeAbstractState(i)
            maze_env = create_maze_env(s, s, s, 'AntMaze')
            random_policy = RandomPolicy(
                8, np.array(maze_env.action_space.high))
            get_rollout(maze_env, random_policy, True, max_timesteps=200)
