import numpy as np

from sarl.envs.rooms import AbstractMap, RoomsEnv, AbstractRoomMap, RandomAbstractMap, AbstractState
from sarl.examples.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, FINAL_ROOM, OBSTACLES
from sarl.examples.rooms_envs import RANDOM_STATES, RANDOM_EDGES
from util.io import parse_command_line_options


flags = parse_command_line_options()
grid_params = GRID_PARAMS_LIST[flags['env_num']]
if flags['obstacle']:
    EVAL_START = 0
    EVAL_END = -1
    obstacle = AbstractState(OBSTACLES[flags['env_num']])
else:
    EVAL_START = -2
    EVAL_END = 2
    obstacle = None
PLANNING_ONLY = True

if flags['baseline_num'] == 0:
    abstract_graph = grid_params.build_abstract_graph(
        every_state_is_start=flags['not_baseline'])
    abstract_map = AbstractMap(grid_params)
elif flags['baseline_num'] == 1:
    abstract_graph = grid_params.build_abstract_room_graph(
        final_room=FINAL_ROOM[flags['env_num']], every_state_is_start=flags['not_baseline'])
    abstract_map = AbstractRoomMap(grid_params,
                                   abstract_graph.abstract_states[0],
                                   abstract_graph.abstract_states[-1])
elif flags['baseline_num'] == 2:
    abstract_graph = grid_params.build_abstract_room_graph(
        final_room=FINAL_ROOM[flags['env_num']],
        every_state_is_start=flags['not_baseline'],
        full_room=False)
    abstract_map = AbstractRoomMap(grid_params,
                                   abstract_graph.abstract_states[0],
                                   abstract_graph.abstract_states[-1],
                                   full_room=False)
elif flags['baseline_num'] == 3:
    abstract_graph = grid_params.build_random_abstract_graph(
        RANDOM_STATES[flags['env_num']], RANDOM_EDGES[flags['env_num']],
        every_state_is_start=flags['not_baseline'])
    abstract_map = RandomAbstractMap(abstract_graph)

abstract_graph.pretty_print()

if not flags['eval_mode']:
    start_state = abstract_graph.abstract_states[0]
    goal_state = abstract_graph.abstract_states[-1]
else:
    start_state = abstract_graph.abstract_states[EVAL_START]
    goal_state = abstract_graph.abstract_states[EVAL_END]


# environment generator used for training options
def train_env_generator(q_start, q_end):
    if flags['alg'] == 'ars':
        return RoomsEnv(grid_params, q_start, abstract_map, q_end,
                        obstacle=obstacle)
    else:
        return RoomsEnv(grid_params, q_start, abstract_map, q_end,
                        use_shaped_rewards=True, obstacle=obstacle)


# environment generator used for estimating rewards
def test_env_generator(state):
    q_start = abstract_map.get_abstract_state(state)
    start_region = abstract_graph.abstract_states[q_start]
    return RoomsEnv(grid_params, start_region, abstract_map, goal_state,
                    start_state=state, obstacle=obstacle)


# actual environment
env = RoomsEnv(grid_params, start_state, goal_state, goal_state,
               max_timesteps=MAX_TIMESTEPS[flags['env_num']], obstacle=obstacle)
obs_dim = env.observation_space.high.size
action_dim = env.action_space.high.size
action_bound = np.array(env.action_space.high)
