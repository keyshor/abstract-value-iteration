from spectrl.main.learning import ProductMDP, learn_policy, HyperParams, FinalStateWrapper
from spectrl.main.spec_compiler import ev, seq, choose
from spectrl.ars.rl import test_policy

from sarl.examples.rooms_common import flags, env, action_bound, action_dim
from util.io import parse_command_line_options, save_log_info, save_object
from util.rl import get_rollout

import numpy as np


# wrap env
env = FinalStateWrapper(FinalStateWrapper(env))


# Define the specification
# Reach predicate
#   goal: np.array(2), err: float
def reach(goal, err=1.0):
    def predicate(sys_state, res_state):
        return (min([sys_state[0] - goal[0],
                     goal[0] - sys_state[0],
                     sys_state[1] - goal[1],
                     goal[1] - sys_state[1]]) + err)
    return predicate


# Specification for different room environments
if flags['env_num'] == 0:
    spec = seq(seq(seq(ev(reach([9.0, 4.0])), ev(reach([14.0, 9.0]))),
                   ev(reach([14.0, 19.0]))),
               ev(reach([19.0, 24.0])))
    params = HyperParams(30, action_bound, 15000, 30, 8, 0.05, 0.2, 0.05)
elif flags['env_num'] == 1:
    spec = seq(ev(reach([9.0, 4.0])), ev(reach([14.0, 9.0])))
    params = HyperParams(30, action_bound, 5000, 20, 8, 0.1, 0.3, 0.1)
elif flags['env_num'] == 2:
    spec = seq(ev(reach([9.0, 4.0])),
               seq(choose(seq(ev(reach([14.0, 9.0])), ev(reach([14.0, 19.0]))),
                          seq(ev(reach([19.0, 4.0])), seq(ev(reach([24.0, 9.0])),
                                                          ev(reach([24.0, 19.0]))))),
                   ev(reach([19.0, 24.0]))))
    params = HyperParams(30, action_bound, 50000, 30, 8, 0.05, 0.2, 0.05)
elif flags['env_num'] == 4:
    spec = seq(ev(reach([9.0, 4.0])),
               seq(choose(seq(ev(reach([19.0, 4.0])), seq(ev(reach([24.0, 9.0])),
                                                          seq(ev(reach([24.0, 19.0])),
                                                              ev(reach([24.0, 29.0]))))),
                          seq(ev(reach([19.0, 4.0])), seq(ev(reach([29.0, 4.0])),
                                                          seq(ev(reach([34.0, 9.0])),
                                                              seq(ev(reach([34.0, 19.0])),
                                                                  ev(reach(34.0, 29.0))))))),
                   ev(reach([29.0, 34.0]))))
    params = HyperParams(30, action_bound, 80000, 30, 8, 0.03, 0.1, 0.05)

lb = 7.0

# Construct Product MDP and learn policy
if __name__ == '__main__':
    flags = parse_command_line_options()
    itno = flags['itno']
    folder = flags['folder']

    # Step 1: construct product MDP
    env = ProductMDP(env, action_dim, spec, 0.0, lb)
    total_action_dim = env.action_dim()
    extra_action_dim = total_action_dim - action_dim
    params.action_bound = np.concatenate(
        [params.action_bound, np.ones(extra_action_dim)])

    # Step 2: Learn policy
    policy, log_info = learn_policy(env, params)

    # Save policy and log information
    save_log_info(log_info, itno, folder)
    save_object('policy', policy, itno, folder)

    # Print rollout and performance
    _, succ_rate = test_policy(env, policy, 100)
    print('Estimated Satisfaction Rate: {}%'.format(succ_rate))
    get_rollout(env, policy, True)
