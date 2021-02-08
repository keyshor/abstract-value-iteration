# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for SARL by: Kishor Jothimurugan

"""A policy that wraps a given policy and adds Gaussian noise."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_agents.trajectories import policy_step
from tf_agents.specs import tensor_spec
from tf_agents.policies import tf_policy
import tensorflow_probability as tfp
import tensorflow as tf

tfd = tfp.distributions


class GaussianPolicy(tf_policy.Base):
    """Actor Policy with Gaussian exploration noise."""

    def __init__(self,
                 wrapped_policy,
                 scale=1,
                 exploration_noise=1,
                 min_noise=0.2,
                 exploration_decay=1e6,
                 clip=True,
                 name=None):
        """Builds an GaussianPolicy wrapping wrapped_policy.
        Args:
          wrapped_policy: A policy to wrap and add OU noise to.
          scale: Stddev of the Gaussian distribution from which noise is drawn.
          clip: Whether to clip actions to spec. Default True.
          name: The name of this policy.
        """

        def _validate_action_spec(action_spec):
            if not tensor_spec.is_continuous(action_spec):
                raise ValueError(
                    'Gaussian Noise is applicable only to continuous actions.')

        tf.nest.map_structure(_validate_action_spec,
                              wrapped_policy.action_spec)

        super(GaussianPolicy, self).__init__(
            wrapped_policy.time_step_spec,
            wrapped_policy.action_spec,
            wrapped_policy.policy_state_spec,
            wrapped_policy.info_spec,
            clip=clip,
            name=name)
        self._wrapped_policy = wrapped_policy

        def _create_normal_distribution(action_spec):
            return tfd.Normal(
                loc=tf.zeros(action_spec.shape, dtype=action_spec.dtype),
                scale=tf.ones(action_spec.shape, dtype=action_spec.dtype) * scale)

        self._noise_distribution = tf.nest.map_structure(
            _create_normal_distribution, self._action_spec)
        self.exploration_noise = tf.Variable(1.)
        self.num_explore_steps = tf.Variable(0.)
        self.min_noise = min_noise
        self.exploration_decay = exploration_decay

    def _variables(self):
        return self._wrapped_policy.variables()

    def _action(self, time_step, policy_state, seed):
        seed_stream = tfp.util.SeedStream(seed=seed, salt='gaussian_noise')

        action_step = self._wrapped_policy.action(time_step, policy_state,
                                                  seed_stream())
        step_increment_op = tf.assign(
            self.num_explore_steps, self.num_explore_steps + 1)

        with tf.control_dependencies([step_increment_op]):
            exploration_decay_op = tf.assign(
                self.exploration_noise,
                tf.maximum(self.min_noise,
                           tf.pow(0.8, self.num_explore_steps / self.exploration_decay)))

        def _add_noise(action, distribution, explore_noise):
            return action + explore_noise * distribution.sample(seed=seed_stream())

        with tf.control_dependencies([exploration_decay_op]):
            actions = tf.nest.map_structure(_add_noise, action_step.action,
                                            self._noise_distribution, self.exploration_noise)
        return policy_step.PolicyStep(actions, action_step.state, action_step.info)

    def _distribution(self, time_step, policy_state):
        raise NotImplementedError('Distributions are not implemented yet.')
