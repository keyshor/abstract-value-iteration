from sarl.main.value_iteration import value_iteration, print_values, compute_high_level_policy

from sarl.main.smdp import SMDP
from sarl.rl.ars import ARSParams, NNPolicy, ars, NNParams
from sarl.rl.agents.td3 import TD3, TD3Params
from sarl.rl.sb.td3 import TD3 as SB_TD3
from sarl.rl.sb.util import SB_TD3Params, SBPolicy
from sarl.rl.sb.policy import MlpPolicy

from util.io import save_object, open_log_file, save_log_info, generate_video
from util.rl import print_performance, print_rollout, get_rollout, log_to_file
from sarl.main.util import GoalWrapper, GoalPolicy, MultiRandomEnv
from multiprocessing import Process, Queue

import tensorflow as tf
import numpy as np
from copy import deepcopy
import random
import time


# class representing an option policy
# an option is a NN policy along with a set of states where it can be activated
class Option:

    # policy: NNPolicy
    # q_start: implements a sample() method
    def __init__(self, policy, q_start):
        self.policy = policy
        self.q_start = q_start

    def sample_start_state(self):
        return self.q_start.sample()

    def get_action(self, state):
        return self.policy.get_action(state)

    def cpu(self):
        if isinstance(self.policy, NNPolicy):
            return Option(self.policy.set_use_cpu(), self.q_start)
        else:
            return self


# Policy that uses options based on value function
class OptionPolicy:

    # options: [[Option]] list of options available from each abstract state
    # high_level_policy: [int]
    # abstract_map: implements get_abstract_state(state) which
    #               maps a concrete state to an abstract state
    def __init__(self, options, high_level_policy, abstract_map):

        # store the best option for each abstract state
        self.option_map = []
        for q in range(len(options)):
            if len(options[q]) != 0:
                self.option_map.append(options[q][high_level_policy[q]])
            else:
                self.option_map.append(None)
        self.abstract_map = abstract_map

        # current option the policy is following
        self.current_option = None

    def get_action(self, state):

        # find the abstract state corresponding to the current state
        abstract_state = self.abstract_map.get_abstract_state(state)

        # state is in some abstract state
        if abstract_state != -1 and self.option_map[abstract_state] is not None:
            self.current_option = self.option_map[abstract_state]

        return self.current_option.get_action(state)

    def parameters(self):
        parameters = []
        for option in self.option_map:
            if option is not None:
                parameters.extend(option.policy.parameters())
        return parameters

    def flat_options(self, to_cpu=False):
        options = []
        for option in self.option_map:
            if option is None:
                options.append([])
            else:
                if to_cpu:
                    options.append([option.cpu()])
                else:
                    options.append([option])
        return options

    def cpu(self):
        options = self.flat_options(to_cpu=True)
        high_level_policy = [0] * len(options)
        return OptionPolicy(options, high_level_policy, self.abstract_map)


# class representing a list of abstract states and their connections
class AbstractGraph:

    # abstract_states: [AbstractState] (AbstractState implements contains(state): Bool and
    #                                   sample(): np.array(state_dim))
    # graph: adjacency list of possible directed edges denoting nearby abstract states
    def __init__(self, abstract_states, graph):
        self.abstract_states = abstract_states
        self.graph = graph

    def pretty_print(self):
        print('\n**** Abstract Graph ****')
        print('Adjacency List: {}'.format(self.graph))

    def compute_indegrees(self):
        indegrees = [0] * len(self.abstract_states)
        for out_edges in self.graph:
            for vertex in out_edges:
                indegrees[vertex] += 1
        return indegrees


# learn options for each edge in graph using ARS
# abstract_graph: AbstractGraph
# env_generator: maps pair of abstract states to an MDP with modified rewards for reachability
#                of second abstract state
# nn_params: NNParams
# ars_params: ARSParams
def learn_options(abstract_graph, env_generator, nn_params, ars_params, print_rollouts=False):

    num_transitions = 0

    # initialize list of list of options
    options = []
    for q1 in range(len(abstract_graph.graph)):

        # initialize list of options
        q1_options = []
        for q2 in abstract_graph.graph[q1]:

            # initialize policy
            policy = NNPolicy(nn_params)

            # create training environment
            q_start = abstract_graph.abstract_states[q1]
            q_end = abstract_graph.abstract_states[q2]
            env = env_generator(q_start, q_end)

            # train policy
            print('\n**** Learning policy for edge: {} ****'.format((q1, q2)))
            ars_log = ars(env, policy, ars_params)
            q1_options.append(Option(policy, q_start))
            num_transitions += ars_log[-1][0]

            # print rollout
            if print_rollouts:
                print('**** Rollout for edge: {} ****'.format((q1, q2)))
                print_rollout(env, policy)

        options.append(q1_options)

    return options, num_transitions


# class that wraps an AbstractState object
# sample() method is modified if a learned distribution is present
class AbstractStateWrapper:

    # distribution implements sample() or is None
    def __init__(self, abstract_state, distribution):
        self.abstract_state = abstract_state
        self.distribution = distribution

    def contains(self, state):
        return self.abstract_state.contains(state)

    def sample(self):
        if self.distribution is None or self.distribution.isempty():
            return self.abstract_state.sample()
        else:
            return self.distribution.sample()

    def additional_state(self, s):
        return self.abstract_state.additional_state(s)


# Distribution of finitely many points
class FiniteDistribution:

    # list: array_like
    def __init__(self, elements):
        self.elements = elements
        self.num_elements = len(elements)

    # unsafe when num_elements = 0
    def sample(self):
        return self.elements[random.randint(0, self.num_elements-1)]

    def extend(self, new_elements):
        self.elements.extend(new_elements)
        self.num_elements = len(self.elements)

    def isempty(self):
        return self.num_elements == 0


# params for learning options with distribution
class DistParams:

    # num_samples: number of samples to add per iteration
    # num_iters: number of iterations for outer loop
    def __init__(self, num_samples, num_iters):
        self.num_samples = num_samples
        self.num_iters = num_iters


# learn options for each edge in graph using ARS or DDPG
# also learn a distribution of states for each abstract state
def learn_options_with_distributions(abstract_graph,  # abstract_graph : AbstractGraph
                                     abstract_map,  # maps concrete state to abstract state
                                     train_env_generator,  # maps pair of abstract states to an MDP
                                     # with modified rewards for reachability
                                     # of second abstract state
                                     test_env_generator,  # maps a state to mdp with modified start
                                     final_env,  # environment with original rewards
                                     nn_params,  # nn_params : NNParams
                                     rl_params,  # rl_params : ARSParams or DDPGParams
                                     dist_params,  # dist_params: DistParams
                                     gamma,  # discount factor
                                     itno,  # run number for naming files generates
                                     folder,  # folder to store data
                                     print_rollouts=False,  # print rollout after learning option
                                     extra_sample_rate=0,  # sample points from init distribution
                                     dist_updates='options',  # sample points from
                                     # new 'option', 'policy' or None
                                     use_gpu=False,  # whether to use GPU for training
                                     parallel_training=0,  # train options in parallel
                                     print_dim=2,  # number of state dimensions
                                     # to use for printing rollouts
                                     save_video=False,
                                     start_abstract_state=0,
                                     pre_trained_options=None
                                     ):

    # log to return
    log_info = []
    log_file = open_log_file(itno, folder)
    num_transitions = 0

    # initialize distributions
    N = len(abstract_graph.graph)
    options = []
    distributions = [FiniteDistribution([]) for _ in range(N)]

    # states visited by new option policies
    new_states = [[] for _ in range(N)]

    # policy for each option
    goal_policy = None
    policies = []
    ddpg_objects = []

    # using ARS, slow but works well for rooms environments
    if isinstance(rl_params, ARSParams):
        if parallel_training == 2:
            final_goal_env = GoalWrapper(final_env, abstract_graph.abstract_states[-1])
            nn_params_new = NNParams(nn_params.state_dim + final_goal_env.get_extra_dim(),
                                     nn_params.action_dim, nn_params.action_bound,
                                     nn_params.hidden_dim + 20)
            goal_policy = NNPolicy(nn_params_new, use_gpu=use_gpu)
        for q in range(N):
            if parallel_training == 2:
                policies.append([GoalPolicy(goal_policy, abstract_graph.abstract_states[goal_state])
                                 for goal_state in abstract_graph.graph[q]])
            elif pre_trained_options is not None:
                policies.append([pre_trained_options[q][j].policy
                                 for j in range(len(abstract_graph.graph[q]))])
            else:
                policies.append([NNPolicy(nn_params, use_gpu=use_gpu)
                                 for _ in range(len(abstract_graph.graph[q]))])

    # using StableBaselines TD3 implementation
    elif isinstance(rl_params, SB_TD3Params):
        if parallel_training == 1:
            raise RuntimeError("Parallel training not supported with StableBaselines TD3")
            exit(1)

        policy_kwargs = dict(layers=rl_params.actor_fc_layers)

        # one network for each option
        if parallel_training == 0:
            for q in range(N):
                ddpg_objects.append([SB_TD3('MlpPolicy', final_env,
                                            learning_rate_pi=rl_params.actor_lr,
                                            learning_rate_qf=rl_params.critic_lr,
                                            buffer_size=rl_params.buffer_capacity,
                                            batch_size=rl_params.batch_size,
                                            tau=rl_params.target_update_tau,
                                            target_policy_noise=rl_params.noise_stddev,
                                            gamma=rl_params.gamma,
                                            policy_kwargs=policy_kwargs)
                                     for _ in abstract_graph.graph[q]])
                policies.append([])
                for ddpg_object in ddpg_objects[q]:
                    policies[q].append(SBPolicy(ddpg_object))

        # single network for all options
        elif parallel_training == 2:
            final_goal_env = GoalWrapper(final_env, abstract_graph.abstract_states[-1])
            policy_kwargs['extra_state'] = final_goal_env.get_extra_dim()
            ddpg_objects = SB_TD3(MlpPolicy, final_goal_env,
                                  learning_rate_pi=rl_params.actor_lr,
                                  learning_rate_qf=rl_params.critic_lr,
                                  buffer_size=rl_params.buffer_capacity,
                                  batch_size=rl_params.batch_size,
                                  tau=rl_params.target_update_tau,
                                  target_policy_noise=rl_params.noise_stddev,
                                  gamma=rl_params.gamma,
                                  policy_kwargs=policy_kwargs)
            for q in range(N):
                policies.append([])
                for goal_state in abstract_graph.graph[q]:
                    policies[q].append(GoalPolicy(SBPolicy(ddpg_objects),
                                                  abstract_graph.abstract_states[goal_state]))

    # using TFAgents TD3 implementation
    elif isinstance(rl_params, TD3Params):
        if parallel_training == 1:
            raise RuntimeError("Parallel training not supported with TFAgents TD3")
            exit(1)

        sess = tf.compat.v1.Session()
        sess.__enter__()

        # single network for all options
        if parallel_training == 2:
            final_goal_env = GoalWrapper(final_env, abstract_graph.abstract_states[-1])
            ddpg_objects = TD3(rl_params, final_goal_env, sess)
            for q in range(N):
                policies.append([])
                for goal_state in abstract_graph.graph[q]:
                    policies[q].append(GoalPolicy(ddpg_objects.get_policy(),
                                                  abstract_graph.abstract_states[goal_state]))

        # one network for each option
        elif parallel_training == 0:
            for q in range(N):
                ddpg_objects.append([TD3(rl_params, final_env, sess)
                                     for _ in abstract_graph.graph[q]])
                policies.append([])
                for ddpg_object in ddpg_objects[q]:
                    policies[q].append(ddpg_object.get_policy())

    start_time = time.time()

    # outer loop for learning the distributions
    for iter_num in range(dist_params.num_iters):

        # add new states to the distributions
        for q in range(N):
            distributions[q].extend(new_states[q])

        print('\n**** Outer Loop Iteration Number {} ****'.format(iter_num))
        sizes = [d.num_elements for d in distributions]
        print('Number of sample points for each abstract state: {}'.format(sizes))

        # initialize options
        options = []

        # abstract states wrapped with learned distributions
        wrapped_states = []
        envs = []

        # queues and processes for retrieving learned policies
        ret_queues = []
        req_queues = []
        processes = []

        # reset new states
        new_states = [[] for _ in range(N)]
        flat_envs = []

        # traverse the abstract_graph and form new environments for each option
        for q1 in range(N):

            # initialize wrapped states, list of queues and processes
            wrapped_states.append(AbstractStateWrapper(abstract_graph.abstract_states[q1],
                                                       distributions[q1]))
            envs.append([])

            if parallel_training == 1:
                ret_queues.append([])
                req_queues.append([])
                processes.append([])

            # for each outgoing edge, start a new process to learn the option
            for j in range(len(abstract_graph.graph[q1])):

                # create training environment
                q2 = abstract_graph.graph[q1][j]
                q_end = abstract_graph.abstract_states[q2]
                envs[q1].append(train_env_generator(wrapped_states[q1], q_end))

                # initialize queues and processes
                if parallel_training == 1:
                    ret_queues[q1].append(Queue())
                    req_queues[q1].append(Queue())

                if parallel_training == 2:
                    flat_envs.append(GoalWrapper(envs[q1][j], q_end))

                if isinstance(rl_params, ARSParams):
                    if parallel_training == 1:
                        processes[q1].append(Process(target=ars_multiprocess,
                                                     args=(envs[q1][j], policies[q1][j],
                                                           rl_params, ret_queues[q1][j],
                                                           req_queues[q1][j], (q1, q2))))
                    elif parallel_training == 0:
                        ars_log = ars(envs[q1][j], policies[q1][j], rl_params, process_id=(q1, q2))
                        num_transitions += ars_log[-1][0]

                elif isinstance(rl_params, SB_TD3Params):
                    if parallel_training == 0:
                        ddpg_objects[q1][j].set_env(envs[q1][j])
                        ddpg_objects[q1][j].learn(rl_params.num_iter)
                        num_transitions += rl_params.num_iter

                elif isinstance(rl_params, TD3Params):
                    if parallel_training == 0:
                        init_step_count = 0
                        if iter_num == 0:
                            init_step_count = 500
                        num_steps_td3, _ = ddpg_objects[q1][j].train(envs[q1][j],
                                                                     init_steps=init_step_count,
                                                                     option_id=(q1, q2))
                        num_transitions += num_steps_td3

                # train policy
                if parallel_training == 1:
                    print('Starting process for edge: {} ...'.format((q1, q2)))
                    processes[q1][j].start()

        if parallel_training == 2:
            if isinstance(rl_params, ARSParams):
                ars_log = ars(flat_envs, goal_policy, rl_params, multi_env=True)
                num_transitions += ars_log[-1][0]

            if isinstance(rl_params, SB_TD3Params):
                multi_env = MultiRandomEnv(flat_envs)
                ddpg_objects.set_env(multi_env)
                ddpg_objects.learn(rl_params.num_iter)
                num_transitions += rl_params.num_iter

            elif isinstance(rl_params, TD3Params):
                init_step_count = 0
                if iter_num == 0:
                    init_step_count = 2000
                num_steps_td3, _ = ddpg_objects.train_multi_env(
                    flat_envs, init_steps=init_step_count)
                num_transitions += num_steps_td3

        # get results from processes
        for q1 in range(N):

            # initialize option list
            options.append([])
            j = 0

            # for each outgoing edge, retrieve learned policy for the option
            while j < len(abstract_graph.graph[q1]):

                if parallel_training == 1:
                    # retrieve learned policy and steps taken
                    try:
                        req_queues[q1][j].put(1)
                        policy_and_steps = ret_queues[q1][j].get()
                    except RuntimeError:
                        print(
                            'Runtime Error occured while retrieving policy! Retrying...')
                        continue

                    # stop the learning process and join
                    req_queues[q1][j].put(None)
                    processes[q1][j].join()

                    if isinstance(rl_params, ARSParams):
                        policies[q1][j] = policy_and_steps[0]

                    num_transitions += policy_and_steps[1]

                options[q1].append(Option(policies[q1][j], wrapped_states[q1]))

                # print rollout
                if print_rollouts:
                    q2 = abstract_graph.graph[q1][j]
                    print('**** Rollout for edge: {} ****'.format((q1, q2)))
                    print_rollout(envs[q1][j], policies[q1]
                                  [j], state_dim=print_dim)

                # go to next outgoing edge
                j += 1

        # compute high level policy using value iteration
        smdp = SMDP(abstract_map, options, test_env_generator, gamma, num_rollouts=50)
        num_transitions += smdp.steps_taken
        values = value_iteration(smdp, 500, V_MIN=-1)
        print_values(values)
        abstract_policy = compute_high_level_policy(values, smdp.rewards, smdp.probs, V_MIN=-1)
        print('Abstract Policy: {}'.format(abstract_policy))

        # add states visited by the high level policy to new_states
        # first abstract state is assumed to be the start state
        # last abstract state is assumed to be the goal state
        if dist_updates is not None:
            explored = [False for _ in range(N)]
            for i in range(N):
                if len(abstract_graph.graph[i]) == 0:
                    explored[i] = True
            start_state = start_abstract_state
            while explored[start_state] is False:
                if dist_updates == 'options':
                    q_start = AbstractStateWrapper(abstract_graph.abstract_states[start_state],
                                                   distributions[start_state])
                elif dist_updates == 'policy':
                    q_start = AbstractStateWrapper(abstract_graph.abstract_states[start_state],
                                                   FiniteDistribution(new_states[start_state]))
                option_num = abstract_policy[start_state]
                target_state = abstract_graph.graph[start_state][option_num]
                q_end = abstract_graph.abstract_states[target_state]
                env = train_env_generator(q_start, q_end)
                collected_states, steps_taken = _collect_new_states(
                    env, q_end, options[start_state][option_num], dist_params.num_samples)
                new_states[target_state].extend(collected_states)
                num_transitions += steps_taken
                explored[start_state] = True
                start_state = target_state

        # add samples from pre-defined distribution to new_states
        for q in range(N):
            samples_collected = len(new_states[q])
            for _ in range(samples_collected * extra_sample_rate):
                new_states[q].append(
                    abstract_graph.abstract_states[q].sample())

        # estimate performance of current policy
        option_policy = OptionPolicy(options, abstract_policy, abstract_map)
        reward, reach_prob = print_performance(final_env, option_policy, gamma, n_rollouts=50)
        print_rollout(final_env, option_policy, state_dim=print_dim)

        # logging
        new_time = time.time()
        log_info.append([num_transitions, (new_time-start_time)/60, reward, reach_prob])
        log_to_file(log_file, iter_num, num_transitions, reward, reach_prob,
                    {'Distributions': [d.num_elements for d in distributions],
                     'New States': [len(s) for s in new_states],
                     'Values': values})

        # save the options
        if not isinstance(rl_params, TD3Params) and not isinstance(rl_params, SB_TD3Params):
            cpu_options = []
            for q in range(len(options)):
                cpu_options.append([deepcopy(option).cpu() for option in options[q]])
            save_object('options', cpu_options, itno, folder)
            save_object('option_policy', deepcopy(option_policy).cpu(), itno, folder)

    log_file.close()
    save_log_info(np.array(log_info), itno, folder)
    if save_video:
        generate_video(final_env, option_policy, itno, folder)


# collect new states in abstract_state by following given option
# env: environment with starting state modified to that of given option
# abstract_state: target state of the option
# policy: NNPolicy
# num_rollouts: number of rollouts to try
def _collect_new_states(env, abstract_state, policy, num_rollouts):
    states = []
    num_steps = 0
    for _ in range(num_rollouts):
        rollout = get_rollout(env, policy, False)
        num_steps += len(rollout)
        last_state = rollout[-1][-1]
        if abstract_state.contains(last_state):
            states.append(last_state)
    return states, num_steps


# ars function for multiple processes
# ret_queue: Queue (returns the learned policy via the queue)
# req_queue: Queue (gets requests from parent)
# edge: string to use for appending to console outputs
# returns [policy, num_steps_taken]
def ars_multiprocess(env, policy, ars_params, ret_queue, req_queue, edge):
    ars_log = ars(env, policy, ars_params, process_id=edge)
    while req_queue.get() is not None:
        ret_queue.put([policy, ars_log[-1][0]])


# ddpg function for multiple processes
# ret_queue: Queue (returns the learned policy via the queue)
# req_queue: Queue (gets requests from parent)
# returns [ddpg_object, num_steps_taken]
def ddpg_multiprocess(env, ddpg_object, ret_queue, req_queue):
    ddpg_object.train(env)
    num_steps_taken = ddpg_object.rewardgraph[-1][0]
    ddpg_object.rewardgraph = []
    while req_queue.get() is not None:
        ret_queue.put([ddpg_object, num_steps_taken])
