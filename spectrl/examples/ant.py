from spectrl.main.learning import ProductMDP, learn_policy, HyperParams, FinalStateWrapper
from spectrl.main.spec_compiler import ev, seq
from spectrl.ars.rl import test_policy

from sarl.examples.ant_common import flags, env, action_bound, action_dim
from util.io import parse_command_line_options, save_log_info, save_object, generate_video

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


# Avoid predicate
#    obstacle: np.array(4): [x_min, y_min, x_max, y_max]
def avoid(obstacle):
    def predicate(sys_state, res_state):
        return max([obstacle[0] - sys_state[0],
                    obstacle[1] - sys_state[1],
                    sys_state[0] - obstacle[2],
                    sys_state[1] - obstacle[3]])
    return predicate


# Specification for different room environments
if flags['env_num'] == 0:
    spec = seq(seq(ev(reach([14., 2.])), ev(reach([14., 14.]))),
               ev(reach(0., 16.)))
    params = HyperParams(300, action_bound, 15000, 60, 20, 0.015, 0.4, 0.05)
elif flags['env_num'] == 1:
    spec = seq(seq(ev(reach([-9.5, 9.5], err=0.5)), ev(reach([-2., 9.5], err=0.5))),
               ev(reach([0., 19.], err=0.5)))
    params = HyperParams(300, action_bound, 15000, 60, 20, 0.015, 0.4, 0.05)
elif flags['env_num'] == 2:
    spec = seq(seq(seq(ev(reach([8., 2.5], err=0.5)), ev(reach([8., 12.5], err=0.5))),
                   ev(reach([8., 23.], err=0.5))),
               ev(reach([0., 27.], err=0.5)))
    params = HyperParams(300, action_bound, 15000, 60, 20, 0.015, 0.4, 0.05)

lb = 10.

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
    generate_video(env, policy, itno, folder)
