from sarl.examples.rooms_common import flags, env, obs_dim, action_dim, action_bound
from sarl.examples.rooms_common import abstract_graph, abstract_map, EVAL_END, EVAL_START
from sarl.examples.rooms_common import train_env_generator, test_env_generator, PLANNING_ONLY
from sarl.examples.rooms_envs import GAMMA

from sarl.rl.ars import NNParams, ARSParams
from sarl.rl.agents.td3 import TD3Params

from sarl.main.options import learn_options_with_distributions, DistParams, OptionPolicy
from sarl.main.smdp_bounds import SMDP
from sarl.main.value_iteration_bounds import value_iteration, compute_high_level_policy

from util.io import load_object, save_log_info
from util.rl import print_performance

import numpy as np

nn_params = None
rl_params = None
dist_params = None
dist_updates = None
if flags['not_baseline']:
    dist_updates = 'policy'

if flags['alg'] == 'ars':
    nn_params = NNParams(obs_dim, action_dim, action_bound, 30)
    if flags['eval_mode']:
        rl_params = ARSParams(50, 30, 15, 0.05, 0.3)
    elif flags['parallel'] <= 1:
        rl_params = ARSParams(200, 30, 15, 0.05, 0.3)
    elif flags['parallel'] == 2:
        rl_params = ARSParams(2000, 40, 15, 0.03, 0.03)
    dist_params = DistParams(200, 30)
elif flags['alg'] == 'td3' and flags['parallel'] <= 1:
    rl_params = TD3Params(actor_fc_layers=(40, 40),
                          critic_fc_layers=(40, 40),
                          noise_stddev=0.1,
                          num_iter=10000)
    dist_params = DistParams(500, 40)
elif flags['alg'] == 'td3' and flags['parallel'] == 2:
    rl_params = TD3Params(actor_fc_layers=(40, 40),
                          critic_fc_layers=(40, 40),
                          noise_stddev=0.1,
                          num_iter=200000,
                          eval_interval=30)
    dist_params = DistParams(500, 40)

# TRANSFER LEARNING
if flags['eval_mode']:
    options = load_object('options', flags['itno'], flags['load_folder'])
    options[EVAL_END] = []

    # RE-PLANNING
    if PLANNING_ONLY:
        smdp = SMDP(abstract_map, options, test_env_generator, GAMMA)
        v_min, v_max = value_iteration(smdp, 500, V_MIN=-1)
        print('Min Values: {}'.format(v_min))
        print('Max Values: {}'.format(v_max))
        abstract_policy = compute_high_level_policy(v_min, smdp.reward_min, smdp.prob_min, V_MIN=-1)
        print('Abstract Policy: {}'.format(abstract_policy))
        option_policy = OptionPolicy(options, abstract_policy, abstract_map)
        reward, reach_prob = print_performance(env, option_policy, GAMMA)
        num_steps = smdp.num_steps
        log_info = np.array([[num_steps, reward, reach_prob]])
        save_log_info(log_info, flags['itno'], flags['folder'])

    # RE-LEARNING
    else:
        abstract_graph.graph[EVAL_END] = []
        learn_options_with_distributions(abstract_graph, abstract_map,
                                         train_env_generator, test_env_generator,
                                         env, nn_params, rl_params, dist_params,
                                         GAMMA, flags['itno'], flags['folder'], print_rollouts=True,
                                         dist_updates=dist_updates, use_gpu=flags['gpu_flag'],
                                         parallel_training=flags['parallel'],
                                         start_abstract_state=EVAL_START,
                                         pre_trained_options=options)

# LEARN FROM SCRATCH
else:
    learn_options_with_distributions(abstract_graph, abstract_map,
                                     train_env_generator, test_env_generator,
                                     env, nn_params, rl_params, dist_params,
                                     GAMMA, flags['itno'], flags['folder'], print_rollouts=True,
                                     dist_updates=dist_updates,
                                     use_gpu=flags['gpu_flag'], parallel_training=flags['parallel'])
