import numpy as np
import torch

from sarl.main.options import OptionPolicy, Option
from sarl.rl.ars import NNPolicy

from util.rl import get_rollout, discounted_reward, test_policy, get_reach_prob


# Run augmented random search.
#
# env: Environment
# nn_policy: OptionPolicy
# params: ARSParams
# gamma: discount factor for computing cumulative rewards
def option_ars(env, policy, params, gamma=1, num_steps=0):
    # Step 1: Save original policy
    policy_orig = policy
    log_info = []
    best_reward = -1000000
    best_policy = policy

    # Step 3: Training iterations
    for i in range(params.n_iters):
        # Step 3a: Sample deltas
        deltas = []
        for _ in range(params.n_samples):
            # i) Sample delta
            delta = _sample_delta(policy)

            # ii) Construct perturbed policies
            policy_plus = _get_delta_policy(policy, delta, params.delta_std)
            policy_minus = _get_delta_policy(policy, delta, -params.delta_std)

            # iii) Get rollouts
            sarss_plus = get_rollout(env, policy_plus, False)
            sarss_minus = get_rollout(env, policy_minus, False)
            num_steps += (len(sarss_minus) + len(sarss_plus))

            # iv) Estimate cumulative rewards
            r_plus = discounted_reward(sarss_plus, gamma)
            r_minus = discounted_reward(sarss_minus, gamma)

            # v) Save delta
            deltas.append((delta, r_plus, r_minus))

        # Step 3b: Sort deltas
        deltas.sort(key=lambda delta: -max(delta[1], delta[2]))
        deltas = deltas[:params.n_top_samples]

        # Step 3c: Compute the sum of the deltas weighted by their reward differences
        delta_sum = [torch.zeros(delta_cur.shape)
                     for delta_cur in deltas[0][0]]
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
        policy = _get_delta_policy(policy, delta_step, 1.0)

        # Step 3h: Logging
        cum_reward = test_policy(env, policy, 20, gamma=gamma)
        print('Expected Reward at iteration {}: {}'.format(i, cum_reward))
        reach_prob = get_reach_prob(env, policy, 20)
        print('Expected Reaching Probability at iteration {}: {}'.format(i, reach_prob))
        log_info.append([num_steps, cum_reward, reach_prob])

        # Step 3i: Save best policy
        if best_reward <= cum_reward:
            best_policy = policy
            best_reward = cum_reward

    # Step 4: Copy new weights and normalization parameters to original policy
    for param, param_orig in zip(best_policy.parameters(), policy_orig.parameters()):
        param_orig.data.copy_(param.data)

    return np.array(log_info)


# Construct random perturbations to neural network parameters.
#
# policy: OptionPolicy
# return: [torch.tensor] (list of torch tensors that is the same shape as policy.parameters())
def _sample_delta(policy):
    delta = []
    for param in policy.parameters():
        delta.append(torch.normal(torch.zeros(param.shape, dtype=torch.float)))
    return delta


# Construct the policy perturbed by the given delta
#
# policy: OptionPolicy
# delta: [torch.tensor] (list of torch tensors that is the same shape as policy.parameters())
# sign: float (should be 1.0 or -1.0, for convenience)
# return: OptionPolicy
def _get_delta_policy(policy, delta, sign):
    # Step 1: Construct the perturbed policy
    options = []
    for option in policy.option_map:
        if option is None:
            options.append([])
            continue
        nn_policy = NNPolicy(option.policy.params)
        nn_policy.mu = option.policy.mu
        nn_policy.sigma_inv = option.policy.sigma_inv
        options.append([Option(nn_policy, option.q_start)])
    policy_delta = OptionPolicy(options, [0]*len(options), policy.abstract_map)

    # Step 2: Set the weights of the perturbed policy
    for param, param_delta, delta_cur in zip(policy.parameters(), policy_delta.parameters(),
                                             delta):
        param_delta.data.copy_(param.data + sign * delta_cur)

    return policy_delta
