from util.io import parse_command_line_options
from sarl.envs.create_maze_environment import create_maze_env
from sarl.envs.maze_env_utils import AbstractMap
from sarl.envs.maze_env_utils import build_abstract_graph

import numpy as np


GAMMA = 0.99
TRAINING_MAX_STEPS = 100

flags = parse_command_line_options()

# environment name:
# "AntMaze", "AntPush", or "AntFall"
if flags['env_num'] == 0:
    env_name = 'AntMaze'
elif flags['env_num'] == 1:
    env_name = 'AntPush'
elif flags['env_num'] == 2:
    env_name = 'AntFall'
else:
    raise Exception('Invalid environment number (using option -e)')
top_down_view = False

env_id = env_name[3:]
maze_size_scaling = 8

abstract_graph = build_abstract_graph(env_id, sample_center_only=False)
abstract_map = AbstractMap(abstract_graph.abstract_states)
start_state = abstract_graph.abstract_states[0]
goal_state = abstract_graph.abstract_states[-1]
abstract_graph.pretty_print()

# create and store environemnts for all edges
training_envs = []
for q1 in range(len(abstract_graph.graph)):
    q1_region = abstract_graph.abstract_states[q1]
    training_envs.append([None for _ in range(len(abstract_graph.graph))])
    for q2 in abstract_graph.graph[q1]:
        q2_region = abstract_graph.abstract_states[q2]
        if flags['alg'] == 'ars':
            training_envs[q1][q2] = create_maze_env(q1_region, abstract_map, q2_region, env_name,
                                                    top_down_view=top_down_view,
                                                    max_timesteps=TRAINING_MAX_STEPS)
        else:
            training_envs[q1][q2] = create_maze_env(q1_region, abstract_map, q2_region, env_name,
                                                    top_down_view=top_down_view,
                                                    max_timesteps=TRAINING_MAX_STEPS,
                                                    use_shaped_reward=True)

test_env = create_maze_env(start_state, abstract_map, goal_state, env_name,
                           top_down_view=False, max_timesteps=TRAINING_MAX_STEPS)


# environment generator used for training options (small max timestep)
def train_env_generator(q_start, q_end):
    qs = abstract_map.get_abstract_state(q_start.sample())
    qe = abstract_map.get_abstract_state(q_end.sample())
    training_envs[qs][qe].set_start_region(q_start)
    training_envs[qs][qe].set_goal_region(q_end)
    return training_envs[qs][qe]


# environment generator used for estimating rewards (small max timestep)
def test_env_generator(state):
    q_start = abstract_map.get_abstract_state(state)
    start_region = abstract_graph.abstract_states[q_start]
    test_env.set_start_region(start_region)
    test_env.set_start_state(state)
    return test_env


# actual environment (big max timestep)
env = create_maze_env(start_state, goal_state, goal_state, env_name,
                      top_down_view=False, max_timesteps=500)
obs_dim = env.observation_space.high.size
action_dim = env.action_space.high.size
action_bound = np.array(env.action_space.high)
