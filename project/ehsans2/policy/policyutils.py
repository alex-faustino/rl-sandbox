from stable_baselines.common.distributions import *
from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.policies import nature_cnn
import gym
import tensorflow as tf

class TransformedDiagGaussianProbabilityDistributionType(DiagGaussianProbabilityDistributionType):
    def __init__(self, *args, mean_trans=lambda x: x, logstd_trans=lambda x:x, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_trans = mean_trans
        self.logstd_trans = logstd_trans
        
    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector, init_scale=1.0, init_bias=0.0):
        mean = self.mean_trans(linear(pi_latent_vector, 'pi', self.size, init_scale=init_scale, init_bias=init_bias))
        #logstd = self.logstd_trans(tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.zeros_initializer())) #input-independent logstd
        logstd = self.logstd_trans(linear(pi_latent_vector, 'pi/logstd', self.size, init_scale=init_scale, init_bias=init_bias)) #input-dependent logstd
        pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        q_values = linear(vf_latent_vector, 'q', self.size, init_scale=init_scale, init_bias=init_bias)
        return self.proba_distribution_from_flat(pdparam), mean, q_values

def custom_make_proba_dist_type(ac_space, mean_trans=lambda x: x, logstd_trans=lambda x:x):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space

    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the approriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return TransformedDiagGaussianProbabilityDistributionType(ac_space.shape[0], mean_trans=mean_trans, logstd_trans=logstd_trans)
    elif isinstance(ac_space, spaces.Discrete):
        return CategoricalProbabilityDistributionType(ac_space.n)
    elif isinstance(ac_space, spaces.MultiDiscrete):
        return MultiCategoricalProbabilityDistributionType(ac_space.nvec)
    elif isinstance(ac_space, spaces.MultiBinary):
        return BernoulliProbabilityDistributionType(ac_space.n)
    else:
        raise NotImplementedError("Error: probability distribution, not implemented for action space of type {}."
                                  .format(type(ac_space)) +
                                  " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary.")
        
class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    My own acpolicy
    Created this for transforming mean and logstd of policy
    """
    def __init__(self, *args, mean_trans=lambda x:x, logstd_trans=lambda x:x, **kwargs):
        super().__init__(*args, **kwargs)
        self.pdtype = custom_make_proba_dist_type(args[2], mean_trans, logstd_trans)
        
        
class CustomFeedForwardPolicy(CustomActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.
    :mean_trans, logstd_trans: (Tensorflow activation functions) Activation functions to limit the mean and logstd of the actor net
    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, 
                 reuse=False, layers=[100],
                 cnn_extractor=nature_cnn, feature_extraction="cnn", 
                 mean_trans=lambda x:x, logstd_trans=lambda x:x, activ = tf.tanh, **kwargs):
        super(CustomFeedForwardPolicy, self).__init__(sess, ob_space, ac_space, 
                                                      n_env, n_steps, n_batch, n_lstm=256,
                                                      reuse=reuse, scale=(feature_extraction == "cnn"), 
                                                      mean_trans = mean_trans, logstd_trans = logstd_trans)

        with tf.variable_scope("model", reuse=reuse):
            if feature_extraction == "cnn":
                extracted_features = cnn_extractor(self.processed_x, **kwargs)
                value_fn = linear(extracted_features, 'vf', 1)
                pi_latent = extracted_features
                vf_latent = extracted_features
            else:
                processed_x = tf.layers.flatten(self.processed_x)
                pi_h = processed_x
                vf_h = processed_x
                for i, layer_size in enumerate(layers):
                    pi_h = activ(linear(pi_h, 'pi_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
                    vf_h = activ(linear(vf_h, 'vf_fc' + str(i), n_hidden=layer_size, init_scale=np.sqrt(2)))
                value_fn = linear(vf_h, 'vf', 1)
                pi_latent = pi_h
                vf_latent = vf_h

            self.proba_distribution, self.policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self.initial_state = None
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp


    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


    def value(self, obs, state=None, mask=None):
        return self.sess.run(self._value, {self.obs_ph: obs})
    
