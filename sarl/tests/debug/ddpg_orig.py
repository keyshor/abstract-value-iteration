# from sarl.examples.ant_common import *
from sarl.rl.ddpg.ddpg import DDPG, DDPGParams

from util.rl import print_performance, get_rollout
from util.io import load_object, save_object, save_log_info, parse_command_line_options

import numpy as np
import gym

flags = parse_command_line_options()
itno = flags['itno']
folder = flags['folder']
eval_mode = flags['eval_mode']
parallel = flags['parallel']
gpu_flag = flags['gpu_flag']

GAMMA = 0.95
MAX_TIMESTEPS = 200
TEST_TIMESTEPS = 1000

if itno == 0:
    env = gym.make('Pendulum-v0')
elif itno == 1:
    env = gym.make('HalfCheetah-v2')
else:
    env = gym.make('Ant-v2')

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

if eval_mode:
    policy = load_object('policy', 0, folder)
    print_performance(env, policy, GAMMA, n_rollouts=20)
    get_rollout(env, policy, True)
    exit(0)

ddpg_params = DDPGParams(obs_dim, action_dim, action_bound,
                         num_episodes=10000,
                         buffer_size=200000,
                         minibatch_size=128,
                         actor_hidden_dim=300,
                         critic_hidden_dim=300,
                         discount=GAMMA,
                         warmup=250,
                         sigma=0.1,
                         tau=0.005,
                         actor_lr=0.0001,
                         critic_lr=0.001,
                         epsilon_decay=1e-6,
                         epsilon_min=0.5,
                         max_timesteps=MAX_TIMESTEPS,
                         test_max_timesteps=TEST_TIMESTEPS)
ddpg_object = DDPG(ddpg_params, use_gpu=gpu_flag)

ddpg_object.train(env)
policy = ddpg_object.get_policy()

policy.set_use_cpu()
save_object('policy', policy, 0, folder)
save_log_info(np.array(ddpg_object.rewardgraph), 0, folder)
