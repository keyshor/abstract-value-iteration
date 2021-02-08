from util.rl import discounted_reward, get_rollout, test_policy, get_reach_prob

import numpy as np
import torch


# Parameters for training a policy neural net.
#
# state_dim: int (n)
# action_dim: int (p)
# hidden_dim: int
# dir: str
# fname: str
class NNParams:
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.hidden_dim = hidden_dim


# Parameters for augmented random search policy.
#
# n_iters: int (ending condition)
# n_samples: int (N)
# n_top_samples: int (b)
# delta_std (nu)
# lr: float (alpha)
class ARSParams:
    def __init__(self, n_iters, n_samples, n_top_samples, delta_std, lr):
        self.n_iters = n_iters
        self.n_samples = n_samples
        self.n_top_samples = n_top_samples
        self.delta_std = delta_std
        self.lr = lr


# Neural network policy.
class NNPolicy:
    # Initialize the neural network.
    #
    # params: NNParams
    def __init__(self, params, use_gpu=False):
        # Step 1: Parameters
        self.params = params
        self.use_gpu = use_gpu

        # Step 2: Construct neural network

        # Step 2a: Construct the input layer
        self.input_layer = torch.nn.Linear(
            self.params.state_dim, self.params.hidden_dim)

        # Step 2b: Construct the hidden layer
        self.hidden_layer = torch.nn.Linear(
            self.params.hidden_dim, self.params.hidden_dim)

        # Step 2c: Construct the output layer
        self.output_layer = torch.nn.Linear(
            self.params.hidden_dim, self.params.action_dim)

        # Step 2d: GPU settings
        if self.use_gpu:
            self.input_layer = self.input_layer.cuda()
            self.hidden_layer = self.hidden_layer.cuda()
            self.output_layer = self.output_layer.cuda()

        # Step 3: Construct input normalization
        self.mu = np.zeros(self.params.state_dim)
        self.sigma_inv = np.ones(self.params.state_dim)

    # Get the action to take in the current state.
    #
    # state: np.array([state_dim])
    def get_action(self, state):
        # Step 1: Normalize state
        state = (state - self.mu) * self.sigma_inv

        # Step 2: Convert to torch
        state = torch.tensor(state, dtype=torch.float)
        if self.use_gpu:
            state = state.cuda()

        # Step 3: Apply the input layer
        hidden = torch.relu(self.input_layer(state))

        # Step 4: Apply the hidden layer
        hidden = torch.relu(self.hidden_layer(hidden))

        # Step 5: Apply the output layer
        output = torch.tanh(self.output_layer(hidden))

        # Step 7: Convert to numpy
        actions = output.cpu().detach().numpy()

        # Step 6: Scale the outputs
        actions = self.params.action_bound * actions

        return actions

    # Construct the set of parameters for the policy.
    #
    # nn_policy: NNPolicy
    # return: list of torch parameters
    def parameters(self):
        parameters = []
        parameters.extend(self.input_layer.parameters())
        parameters.extend(self.hidden_layer.parameters())
        parameters.extend(self.output_layer.parameters())
        return parameters

    def set_use_cpu(self):
        self.use_gpu = False
        self.input_layer = self.input_layer.cpu()
        self.hidden_layer = self.hidden_layer.cpu()
        self.output_layer = self.output_layer.cpu()
        return self


# Run augmented random search.
#
# env: Environment
# nn_policy: NNPolicy
# params: ARSParams
# gamma: discount factor for computing cumulative rewards
# use_envs_cum_reward: uses cum_reward() function for the environment when set to True
# sparse_rewards: adds a satisfaction probability value to log when set to true
#                 (works only for environments with sparse rewards and
#                  use_envs_cum_reward has to be False)
# process_id: string to distiguish console ouputs for simultanious executions
def ars(env, nn_policy, params, gamma=1, use_envs_cum_reward=True,
        sparse_rewards=False, process_id='', multi_env=False):
    # Step 1: Save original policy
    nn_policy_orig = nn_policy
    log_info = []
    best_reward = -1000000
    # best_policy = nn_policy
    num_transitions = 0

    # Step 2: Initialize state distribution estimates
    mu_sum = np.zeros(nn_policy.params.state_dim)
    sigma_sq_sum = np.zeros(nn_policy.params.state_dim)
    n_states = 0

    if multi_env:
        env_list = env
    else:
        env_list = [env]

    # Step 3: Training iterations
    for i in range(params.n_iters):
        env = env_list[np.random.randint(0, len(env_list))]

        # Step 3a: Sample deltas
        deltas = []
        for _ in range(params.n_samples):
            # i) Sample delta
            delta = _sample_delta(nn_policy)

            # ii) Construct perturbed policies
            nn_policy_plus = _get_delta_policy(nn_policy, delta, params.delta_std)
            nn_policy_minus = _get_delta_policy(nn_policy, delta, -params.delta_std)

            # iii) Get rollouts
            sarss_plus = get_rollout(env, nn_policy_plus, False)
            sarss_minus = get_rollout(env, nn_policy_minus, False)
            num_transitions += (len(sarss_plus) + len(sarss_minus))

            # iv) Estimate cumulative rewards
            r_plus = 0
            r_minus = 0
            if use_envs_cum_reward:
                r_plus = env.cum_reward(sarss_plus)
                r_minus = env.cum_reward(sarss_minus)
            else:
                r_plus = discounted_reward(sarss_plus, gamma)
                r_minus = discounted_reward(sarss_minus, gamma)

            # v) Save delta
            deltas.append((delta, r_plus, r_minus))

            # v) Update estimates of normalization parameters
            states = np.array([state for state, _, _, _ in sarss_plus + sarss_minus])
            mu_sum += np.sum(states)
            sigma_sq_sum += np.sum(np.square(states))
            n_states += len(states)

        # Step 3b: Sort deltas
        deltas.sort(key=lambda delta: -max(delta[1], delta[2]))
        deltas = deltas[:params.n_top_samples]

        # Step 3c: Compute the sum of the deltas weighted by their reward differences
        delta_sum = [torch.zeros(delta_cur.shape)
                     for delta_cur in deltas[0][0]]
        if nn_policy.use_gpu:
            delta_sum = [delta_cur.cuda() for delta_cur in delta_sum]

        for j in range(params.n_top_samples):
            # i) Unpack values
            delta, r_plus, r_minus = deltas[j]

            # ii) Add delta to the sum
            for k in range(len(delta_sum)):
                delta_sum[k] += (r_plus - r_minus) * delta[k]

        # Step 3d: Compute standard deviation of rewards
        sigma_r = np.std([delta[1] for delta in deltas] +
                         [delta[2] for delta in deltas])

        # Step 3e: Compute step length
        delta_step = [(params.lr * params.delta_std / (params.n_top_samples * sigma_r + 1e-8))
                      * delta_sum_cur for delta_sum_cur in delta_sum]

        # Step 3f: Update policy weights
        nn_policy = _get_delta_policy(nn_policy, delta_step, 1.0)

        # Step 3g: Update normalization parameters
        nn_policy.mu = mu_sum / n_states
        nn_policy.sigma_inv = 1.0 / np.sqrt(sigma_sq_sum / n_states)

        # Step 3h: Logging
        if i % len(env_list) == 0:
            cum_rewards = []
            for test_env in env_list:
                cum_rewards.append(test_policy(
                    env, nn_policy, 20, gamma=gamma, use_cum_reward=use_envs_cum_reward))
            cum_reward = np.mean(cum_rewards)
            print('[{}] Expected Reward at iteration {}: {}'.format(process_id, i, cum_reward))

        if not sparse_rewards:
            log_info.append([num_transitions, cum_reward])
        else:
            reach_prob = get_reach_prob(env, nn_policy, 20)
            print('[{}] Expected Reaching Probability at iteration {}: {}'.format(process_id, i,
                                                                                  reach_prob))
            log_info.append([num_transitions, cum_reward, reach_prob])

        # Step 3i: Save best policy
        if best_reward <= cum_reward:
            # best_policy = nn_policy
            best_reward = cum_reward

    # Step 4: Copy new weights and normalization parameters to original policy
    for param, param_orig in zip(nn_policy.parameters(), nn_policy_orig.parameters()):
        param_orig.data.copy_(param.data)
    nn_policy_orig.mu = nn_policy.mu
    nn_policy_orig.sigma_inv = nn_policy.sigma_inv

    return np.array(log_info)


# Construct random perturbations to neural network parameters.
#
# nn_policy: NNPolicy
# return: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
def _sample_delta(nn_policy):
    delta = []
    for param in nn_policy.parameters():
        delta_param = torch.normal(torch.zeros(param.shape, dtype=torch.float))
        if nn_policy.use_gpu:
            delta_param = delta_param.cuda()
        delta.append(delta_param)
    return delta


# Construct the policy perturbed by the given delta
#
# nn_policy: NNPolicy
# delta: [torch.tensor] (list of torch tensors that is the same shape as nn_policy.parameters())
# sign: float (should be 1.0 or -1.0, for convenience)
# return: NNPolicy
def _get_delta_policy(nn_policy, delta, sign):
    # Step 1: Construct the perturbed policy
    nn_policy_delta = NNPolicy(nn_policy.params, use_gpu=nn_policy.use_gpu)

    # Step 2: Set normalization of the perturbed policy
    nn_policy_delta.mu = nn_policy.mu
    nn_policy_delta.sigma_inv = nn_policy.sigma_inv

    # Step 3: Set the weights of the perturbed policy
    for param, param_delta, delta_cur in zip(nn_policy.parameters(), nn_policy_delta.parameters(),
                                             delta):
        param_delta.data.copy_(param.data + sign * delta_cur)

    return nn_policy_delta
