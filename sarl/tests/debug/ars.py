from sarl.examples.ant_common import flags, training_envs, GAMMA
from sarl.examples.ant_common import obs_dim, action_dim, action_bound
from sarl.rl.ars import ARSParams, NNParams, NNPolicy, ars

from util.rl import print_performance, get_rollout
from util.io import load_object, save_object, save_log_info

import numpy as np

env = None
if flags['itno'] == 0:
    env = training_envs[0][1]
elif flags['itno'] == 1:
    env = training_envs[1][2]
else:
    env = training_envs[2][3]

if flags['eval_mode']:
    policy = load_object('policy', 0, flags['folder'])
    print_performance(env, policy, GAMMA)
    get_rollout(env, policy, True)
    exit(0)


nn_params = NNParams(obs_dim, action_dim, action_bound, 300)
ars_params = ARSParams(40000, 60, 15, 0.001, 0.1)

policy = NNPolicy(nn_params, use_gpu=flags['gpu_flag'])
log_info = []
if flags['alg'] == 'ddpg':
    log_info = ars(env, policy, ars_params,
                   gamma=0.99,
                   use_envs_cum_reward=False)
else:
    log_info = ars(env, policy, ars_params)

save_object('policy', policy.set_use_cpu(), 0, flags['folder'])
save_log_info(np.array(log_info), 0, flags['folder'])
