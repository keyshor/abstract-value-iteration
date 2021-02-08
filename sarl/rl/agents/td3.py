from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from sarl.rl.agents import td3_agent
from tf_agents.policies import py_tf_policy
from tf_agents.replay_buffers.tf_uniform_replay_buffer import TFUniformReplayBuffer
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import TimeLimit

from sarl.rl.agents.util import GymPyPolicy
from util.rl import test_policy
from tf_agents.utils import common

import numpy as np
import tensorflow as tf
tf.compat.v1.enable_resource_variables()


class TD3Params:

    def __init__(self,
                 actor_fc_layers=(300, 300),
                 critic_fc_layers=(300, 300),
                 actor_lr=0.0001,
                 critic_lr=0.001,
                 noise_stddev=3.0,
                 target_update_period=2,
                 target_update_tau=0.005,
                 batch_size=100,
                 buffer_capacity=200000,
                 train_steps_per_iter=1,
                 num_iter=100000,
                 eval_interval=10,
                 noise_decay=2e7,
                 gamma=0.99):
        self.actor_fc_layers = actor_fc_layers
        self.critic_fc_layers = critic_fc_layers
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.noise_stddev = noise_stddev
        self.target_update_period = target_update_period
        self.target_update_tau = target_update_tau
        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.train_steps_per_iter = train_steps_per_iter
        self.num_iter = num_iter
        self.eval_interval = eval_interval
        self.noise_decay = noise_decay
        self.gamma = gamma


class TD3:

    def __init__(self, params, env, sess, max_timesteps=None):
        '''
        params : TD3Params
        env : Gym environment
        sess : Tensorflow session
        '''

        self.params = params
        self.max_timesteps = max_timesteps
        self.sess = sess

        tf_env = TFPyEnvironment(GymWrapper(env))
        self.time_step_spec = tf_env.time_step_spec()
        self.action_spec = tf_env.action_spec()

        self.actor_net = actor_network.ActorNetwork(
            self.time_step_spec.observation,
            self.action_spec,
            fc_layer_params=self.params.actor_fc_layers
        )

        critic_input_tensor_spec = (
            self.time_step_spec.observation, self.action_spec)
        self.critic_net = critic_network.CriticNetwork(
            critic_input_tensor_spec,
            observation_fc_layer_params=(self.params.critic_fc_layers[0],),
            action_fc_layer_params=None,
            joint_fc_layer_params=(self.params.critic_fc_layers[1],)
        )

        self.tf_agent = td3_agent.Td3Agent(
            self.time_step_spec,
            self.action_spec,
            actor_network=self.actor_net,
            critic_network=self.critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.params.actor_lr),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=self.params.critic_lr),
            exploration_noise_std=self.params.noise_stddev,
            target_update_tau=self.params.target_update_tau,
            target_update_period=self.params.target_update_period,
            td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
            gamma=self.params.gamma,
            reward_scale_factor=1.0,
            noise_decay=self.params.noise_decay
        )

        eval_policy = py_tf_policy.PyTFPolicy(self.tf_agent.policy)
        # gym_policy only works if you within the context of self.sess
        # Call sess.__enter__() before using this policy
        self.gym_policy = GymPyPolicy(eval_policy)

        self.replay_buffer = TFUniformReplayBuffer(
            data_spec=self.tf_agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=self.params.buffer_capacity
        )

        dataset = self.replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.params.batch_size,
            num_steps=2).prefetch(3)

        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        trajectories, _ = iterator.get_next()
        train_fn = common.function(self.tf_agent.train)
        self.train_op = train_fn(experience=trajectories)

        init_agent_op = self.tf_agent.initialize()
        self.sess.run(iterator.initializer)
        common.initialize_uninitialized_variables(self.sess)
        self.sess.run(init_agent_op)

    def train(self, env, init_steps=None, option_id=''):
        '''
        env : Gym environment
        init_steps : Intial random steps to populate replay buffer
        '''
        tf_env = TFPyEnvironment(GymWrapper(env))
        if self.max_timesteps is not None:
            tf_env = TimeLimit(tf_env, self.max_timesteps)

        if init_steps != 0:
            initial_collect_op = DynamicStepDriver(
                tf_env,
                self.tf_agent.collect_policy,
                observers=[self.replay_buffer.add_batch],
                num_steps=init_steps).run()
            self.sess.run(initial_collect_op)

        collect_op = DynamicStepDriver(
            tf_env,
            self.tf_agent.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=1).run()

        iter_num = 0
        episode_num = 0
        reward_graph = []

        print('\n ***** Training Option {} *****'.format(option_id))

        while iter_num <= self.params.num_iter:
            while True:
                # Step the environment
                time_step, _ = self.sess.run(collect_op)
                # Train the agent using samples from replay buffer
                _ = self.sess.run(self.train_op)
                # increment iter
                iter_num += 1
                # break of episode ends
                if time_step.is_last():
                    break

            episode_num += 1
            if episode_num % self.params.eval_interval == 0:
                estimated_reward = test_policy(env, self.gym_policy, 20, gamma=self.params.gamma,
                                               max_timesteps=(self.max_timesteps or 10000))
                print('Estimated Reward after {} episodes: {}'.format(
                    episode_num, estimated_reward))
                print('Steps after {} episodes: {}'.format(episode_num, iter_num))
                reward_graph.append([iter_num, estimated_reward])

        return iter_num, reward_graph

    def train_multi_env(self, envs, init_steps=None, option_id=''):
        '''
        env : list of Gym environments
        init_steps : Intial random steps to populate replay buffer
        '''
        tf_envs = [TFPyEnvironment(GymWrapper(env)) for env in envs]
        if self.max_timesteps is not None:
            tf_envs = [TimeLimit(tf_env, self.max_timesteps)
                       for tf_env in tf_envs]

        collect_ops = [DynamicStepDriver(
            tf_env,
            self.tf_agent.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=1).run() for tf_env in tf_envs]

        if init_steps is not None:
            for i in range(len(envs)):
                steps_collected = 0
                while steps_collected <= init_steps:
                    while True:
                        time_step, _ = self.sess.run(collect_ops[i])
                        steps_collected += 1
                        if time_step.is_last():
                            break

        iter_num = 0
        episode_num = 0
        reward_graph = []

        print('\n**** Training all options ****')

        while iter_num <= self.params.num_iter:
            # Choose a random environment from envs
            collect_op = collect_ops[np.random.randint(0, len(envs))]
            while True:
                # Step the environment
                time_step, _ = self.sess.run(collect_op)
                # Train the agent using samples from replay buffer
                _ = self.sess.run(self.train_op)
                # increment iter
                iter_num += 1
                # break of episode ends
                if time_step.is_last():
                    break

            episode_num += 1
            if episode_num % self.params.eval_interval == 0:
                estimated_rewards = [test_policy(env, self.gym_policy, 20, gamma=self.params.gamma,
                                                 max_timesteps=(self.max_timesteps or 10000))
                                     for env in envs]
                print('Estimated Reward after {} episodes: {}'.format(
                    episode_num, estimated_rewards))
                print('Steps after {} episodes: {}'.format(episode_num, iter_num))
                reward_graph.append([iter_num, estimated_rewards])

        return iter_num, reward_graph

    def get_policy(self):
        return self.gym_policy

    def get_tf_policy(self):
        return self.tf_agent.policy

    def get_py_policy(self):
        return py_tf_policy.PyTFPolicy(self.tf_agent.policy)
