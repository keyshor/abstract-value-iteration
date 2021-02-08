from stable_baselines.td3.policies import TD3Policy

import tensorflow as tf


class MlpPolicy(TD3Policy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor))
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False,
                 layers=None, cnn_extractor=None, feature_extraction="mlp",
                 layer_norm=False, act_fun=tf.nn.relu, extra_state=4, **kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                        reuse=reuse, scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        if layers is None:
            layers = [64, 64]
        self.layers = layers
        self.extra_state = extra_state
        self.obs_dim = ob_space.shape[0]

        assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        state_obs, option_obs = tf.split(
            obs, [self.obs_dim - self.extra_state, self.extra_state], -1)

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h_state = tf.layers.flatten(state_obs)
                pi_h_option = tf.layers.flatten(option_obs)

            pi_h_state = tf.layers.dense(pi_h_state, self.layers[0], activation=self.activ_fn)
            pi_h_option = tf.layers.dense(pi_h_option, self.layers[0], activation=self.activ_fn)
            pi_h = tf.concat([pi_h_state, pi_h_option], axis=-1)
            pi_h = tf.layers.dense(pi_h, self.layers[1], activation=self.activ_fn)

            self.policy = policy = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=tf.tanh)

        return policy

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn"):
        if obs is None:
            obs = self.processed_obs

        state_obs, option_obs = tf.split(
            obs, [self.obs_dim - self.extra_state, self.extra_state], -1)

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                qf_h_state = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                qf_h_state = tf.layers.flatten(state_obs)
                qf_h_option = tf.layers.flatten(option_obs)

            # Concatenate preprocessed state and action
            qf_h_state = tf.concat([qf_h_state, action], axis=-1)

            # Double Q values to reduce overestimation
            with tf.variable_scope('qf1', reuse=reuse):
                qf1_h_state = tf.layers.dense(qf_h_state, self.layers[0], activation=self.activ_fn)
                qf1_h_option = tf.layers.dense(
                    qf_h_option, self.layers[0], activation=self.activ_fn)
                qf1_h = tf.concat([qf1_h_state, qf1_h_option], axis=-1)
                qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

            with tf.variable_scope('qf2', reuse=reuse):
                qf2_h_state = tf.layers.dense(qf_h_state, self.layers[0], activation=self.activ_fn)
                qf2_h_option = tf.layers.dense(
                    qf_h_option, self.layers[0], activation=self.activ_fn)
                qf2_h = tf.concat([qf2_h_state, qf2_h_option], axis=-1)
                qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

            self.qf1 = qf1
            self.qf2 = qf2

        return self.qf1, self.qf2

    def step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy, {self.obs_ph: obs})
