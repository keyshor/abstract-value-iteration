from sarl.examples.ant_common import flags, env, obs_dim, action_dim, action_bound
from sarl.examples.ant_common import abstract_graph, abstract_map, GAMMA
from sarl.examples.ant_common import train_env_generator, test_env_generator

from sarl.rl.ars import NNParams, ARSParams
from sarl.rl.ddpg.ddpg import DDPGParams
from sarl.rl.agents.td3 import TD3Params
from sarl.rl.sb.util import SB_TD3Params
from sarl.main.options import learn_options_with_distributions, DistParams

from util.io import load_object
from util.rl import print_performance, get_rollout

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if flags['eval_mode']:
    if flags['alg'] != 'td3' and flags['alg'] != 'sbtd3':
        option_policy = load_object(
            'option_policy', flags['itno'], flags['folder'])
        print_performance(env, option_policy, GAMMA)
        print('\nRendering...')
        get_rollout(env, option_policy, True)
    exit(0)


nn_params = None
rl_params = None
dist_params = None
dist_updates = None
if flags['not_baseline']:
    dist_updates = 'policy'

if flags['alg'] == 'ars':
    nn_params = NNParams(obs_dim, action_dim, action_bound, 300)
    rl_params = ARSParams(600, 60, 20, 0.015, 0.5)
    dist_params = DistParams(300, 20)
elif flags['alg'] == 'ddpg':
    rl_params = DDPGParams(obs_dim, action_dim, action_bound,
                           num_episodes=1000,
                           actor_hidden_dim=300,
                           critic_hidden_dim=300,
                           minibatch_size=100,
                           warmup=10000,
                           noise='normal',
                           buffer_size=200000)
    dist_params = DistParams(300, 25)
elif flags['alg'] == 'td3' and flags['parallel'] <= 1:
    rl_params = TD3Params(noise_decay=5e8, gamma=0.95,
                          num_iter=100000)
    dist_params = DistParams(500, 50)
elif flags['alg'] == 'td3' and flags['parallel'] == 2:
    rl_params = TD3Params(num_iter=60000,
                          noise_decay=5e8,
                          noise_stddev=0.1,
                          eval_interval=30)
    dist_params = DistParams(500, 200)
elif flags['alg'] == 'sbtd3' and flags['parallel'] <= 1:
    rl_params = SB_TD3Params(noise_stddev=0.1, batch_size=128, num_iter=100000, gamma=GAMMA)
    dist_params = DistParams(500, 40)
elif flags['alg'] == 'sbtd3' and flags['parallel'] == 2:
    rl_params = SB_TD3Params(noise_stddev=0.1, batch_size=128, num_iter=50000, gamma=GAMMA)
    dist_params = DistParams(500, 200)

learn_options_with_distributions(abstract_graph, abstract_map,
                                 train_env_generator, test_env_generator,
                                 env, nn_params, rl_params, dist_params,
                                 GAMMA, flags['itno'], flags['folder'], print_rollouts=False,
                                 dist_updates=dist_updates, save_video=False,
                                 use_gpu=flags['gpu_flag'], parallel_training=flags['parallel'])
