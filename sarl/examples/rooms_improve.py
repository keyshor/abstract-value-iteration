from sarl.examples.rooms_common import flags, env, obs_dim, action_dim, action_bound
from sarl.examples.rooms_common import abstract_graph, abstract_map
from sarl.examples.rooms_common import train_env_generator, test_env_generator
from sarl.examples.rooms_envs import GAMMA
from sarl.main.value_iteration import value_iteration, print_values, compute_high_level_policy

from sarl.rl.ars import NNParams, ARSParams
from sarl.rl.option_ars import option_ars
from sarl.main.options import learn_options, OptionPolicy
from sarl.main.smdp import SMDP

from util.io import load_object, save_object, save_log_info
from util.rl import print_performance, get_rollout

import numpy as np


# train options for all edges
options = None
num_transitions = 0
if flags['eval_mode']:
    option_policy = load_object('policy', flags['itno'], flags['folder'])
    print_performance(env, option_policy, GAMMA)
    get_rollout(env, option_policy, True)
    exit(0)

nn_params = NNParams(obs_dim, action_dim, action_bound, 30)
ars_params = ARSParams(2000, 30, 15, 0.05, 0.3)
options, num_transitions = learn_options(
    abstract_graph, train_env_generator, nn_params, ars_params, True)
save_object('options', options, flags['itno'], flags['folder'])


# create smdp
smdp = SMDP(abstract_map, options, test_env_generator, GAMMA)
smdp.pretty_print()
num_transitions += smdp.steps_taken

# do value iteration
values = value_iteration(smdp, 500, V_MIN=-1)
print_values(values)

# create policy based on values
abstract_policy = compute_high_level_policy(
    values, smdp.rewards, smdp.probs, V_MIN=-1)
print('\n**** High Level Policy ****')
print(abstract_policy)
option_policy = OptionPolicy(options, abstract_policy, abstract_map)

# estimate the current option policy
reward, reach_prob = print_performance(
    env, option_policy, GAMMA, n_rollouts=50)

# improve option policy using ars
print('\n**** Improving Option Policy ****')
ars_params = ARSParams(30000, 40, 10, 0.005, 0.1)
log_info = option_ars(env, option_policy, ars_params,
                      gamma=GAMMA, num_steps=num_transitions)
log_info = np.append([[num_transitions, reward, reach_prob]], log_info, axis=0)
save_object('policy', option_policy, flags['itno'], flags['folder'])
save_log_info(log_info, flags['itno'], flags['folder'])

# estimate performance
print_performance(env, option_policy, GAMMA)
