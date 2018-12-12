import gym
import itertools
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import gym_env
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
import math
from scipy.stats import entropy
from scipy.optimize import minimize
from numdifftools import Jacobian, Hessian
from scipy.special import logsumexp

# LAPACK warning when using numdifftools.Jacobian
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

matplotlib.style.use('ggplot')
class Policy():
    """
    Policy for Discrete environments.
    """
    
    def __init__(self, env):
        """
        The policy is initialized with a uniform distribution.
        
        Args:
            env: openAI environment.
        """
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.policy = np.ones([self.nS, self.nA]) / self.nA
        #self.policy = np.array([[0.9, 0.1],[0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
        
class Policy_REPS(Policy):
    """
    Policy for Discrete environments, updated using REPS Policy Iteration.
    """
    
    def __init__(self, env):
        """
        Args:
            env: openAI environment.
        """
        Policy.__init__(self, env)
    
    def action_probabilities(self, state):
        """
        Returns an array of action probabilities for a given state.
        
        Args:
            state: the state.
        
        Returns:
            An array of action probabilities for the state.
        """
        return self.policy[state]
    def print1(self):
        print(self.policy)
        
    def update(self, bellman_error, eta):
        """
        Updates the policy according to REPS.
        
        Args:
            bellman_error: Bellman Error Function is an array of [states][actions]
            eta: lagrange multiplier.
        """
        # TODO: Use a python library / function to compute the weighted softmax safer.
        for s in range(self.nS):
            y = np.dot(self.policy[s], np.exp(1/eta * (bellman_error[s]-np.max(bellman_error[s]))))
            if y < 0.01:
                print('Yaha hai error', y)
                print('Bellman_error', bellman_error[s])
                print('policy', self.policy[s])
            else:    
                for a in range(self.nA):
                    self.policy[s][a] = self.policy[s][a] * math.exp(1/eta * (bellman_error[s][a] \
                               -np.max(bellman_error[s]))) / y
        

class run_REPS(object):
    def REPS(self, env, R, policy, features, x0, epsilon = 0.9, info_loss=1,
             n_policy_updates=100, n_samples=100, opt_method='Newton-CG'):
        """
        Policy Iteration with REPS for Discrete Environments.
        
        Args:
            env: openAI environment.
            R: return for a (state, action) pair.
            policy: policy object.
            features: array of state features.
            info_loss: maximal information loss.
            n_policy_updates: number of policy updates.
            n_samples: number of collected samples per policy update.
            x0: array with initial values for [theta, eta]. Eta is a scalar located in the last
                position of the array.
        
        Returns:
            ...
        """
        # Take for theta and eta the initial values. 
        theta = x0[0:-1]
        eta = x0[-1]
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[-1] = (0.01, np.inf)
        policy0 = policy
        # Arrays for statistics.
        J = np.zeros((n_policy_updates))
        V = np.zeros((n_policy_updates, env.observation_space.n))
        q_sa = np.zeros((n_policy_updates, env.observation_space.n, env.action_space.n))
        
        for i in range(n_policy_updates):
            print('\n#################################################\n')
            print("\rPolicy update {}/{}".format(i+1, n_policy_updates), end=" - ")
            # Obtain N samples (s, a, s', r) using the current (?) policy.
            samples = []
            state = env.reset()
            # Count the frequency of (state, action) pairs.
            d = np.zeros((env.observation_space.n, env.action_space.n))
            # Precompute the average feature error for each (state, action) pair, using
            # the current samples.
            # feature_difference is a matrix of feature differences with dimensions
            # (n_states, n_actions, n_features).
            if n_policy_updates >= 10:
                n_samples = 250
            
            feature_difference = np.zeros((env.observation_space.n, env.action_space.n, features[0].shape[0]))
            
            for fh in range(n_samples):
                #print('\nSample No. : ', fh)
                if i <= -1:
                    action_probabilities = policy0.action_probabilities(state)
                elif np.random.rand() < epsilon:
                    action_probabilities = policy0.action_probabilities(state)
                else:
                    action_probabilities = policy.action_probabilities(state)
                #print('Sampling dist:',action_probabilities)
                
                action = np.random.choice(np.arange(len(action_probabilities)), p=action_probabilities)
                next_state, reward, done, _ = env.step(action)
                samples.append((state, action, next_state, reward))
                # Update (state, action) counter.
                d[state][action] += 1
                # Update feature_difference matrix.
                diff = features[next_state] - features[state]
                feature_difference[state][action] += diff
                
                if done == True:
                    print('damn')
                    break
                state = next_state
            q_sa[i] = np.copy(d) / n_samples
            
            # Update the average feature vector.
            feature_difference = np.divide(feature_difference, d.reshape(d.shape[0], d.shape[1], 1),
                                           out=np.zeros_like(feature_difference),
                                           where=d.reshape(d.shape[0], d.shape[1], 1)!=0)
                
            def bellman_error(theta):
                """
                Compute the average bellman error for each (state, action) pair, using
                the current samples.
    
                Args:
                    theta: an array of lagrange multiplier.
                
                Returns:
                    A matrix of [states][actions] with the average bellman_error for
                    each (state, action) pair.
                """
                n = np.zeros((env.observation_space.n, env.action_space.n))
                for sample in samples:
                    state, action, next_state, reward = sample
                    error = reward + np.dot(features[next_state], theta.T) - np.dot(features[state], theta.T)
                    n[state][action] += error
                return np.divide(n, d, out=np.zeros_like(n), where=d!=0)
           
            def dual_function(x):
                """
                Evaluates the dual function in x, using the current samples.
                
                Args:
                    x: an array with theta and eta.
                
                Returns:
                    The dual function result in x.
                """
                theta = x[0:-1]   # present in original code
                eta = x[-1]       # present in original code
                
                g1 = math.log(1 / n_samples)
                
                g2 = np.zeros((n_samples))
                for i1, sample in enumerate(samples):
                    state, action, next_state, reward = sample
                    g2[i1] = info_loss + 1 / eta * (bellman_error(theta)[state][action])
                g2 = logsumexp(g2)  # use this function to improve numerical stability.
                
                g = eta * (g1 + g2)  # (1/eta) in the original code
                return g
        
            def dual_function_jac(x):
                """
                Evaluates the dual function jacobian in x, using the current samples.
                
                Args:
                    x: an array with theta and eta.
                
                Returns:
                    An array with the dual function jacobian in x.
                """
                # Derivative with respect to theta is an array, with an entry for each theta[i].
                # Derivative with respect to eta is a scalar.
                theta = x[0:-1]   # was not there before
                eta = x[-1]       # was not there in the original code
                d_theta = np.zeros(features[0].shape[0])
                d_eta = 0
                #print(1/eta)
                den = np.zeros((n_samples))  # store (info_loss * 1/eta * bellman_error) for each sample.
                for i, sample in enumerate(samples):
                    state, action, next_state, reward = sample
                    try:
                        tmp1 = math.exp(1/eta * (bellman_error(theta)[state][action]-\
                                                 np.max(bellman_error(theta))))
                        #print(type(tmp1))
                    except OverflowError:
                        #print('lol')
                        tmp1 = float('inf')
                    d_theta += tmp1 * feature_difference[state][action]
                    d_eta += tmp1 * (1/eta)*(bellman_error(theta)[state][action])    # (1/eta) missing in original code
                    den[i] = 1 / eta * (bellman_error(theta)[state][action]-\
                       np.max(bellman_error(theta)))
                # d_theta *= eta # why?
                #print('\nden is : ',np.sum(np.exp(den)))
                d_theta *= 1/np.sum(np.exp(den))
                #d_eta *= - (1 / eta**2) / np.sum(np.exp(den))
                d_eta *= - 1 / np.sum(np.exp(den))
                d_eta += logsumexp(den/n_samples)
                #print(feature_difference[:,:,-1])
                d_g = np.append(d_theta, d_eta)
                
                return d_g
                
            epsilon *= 0.99
            # Optimization routine for theta and eta.
            opt = minimize(dual_function, x0, method=opt_method, bounds = bounds, jac=dual_function_jac) 
                           #options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-03, 'gtol': 1e-5, 'eps': 1e-8, \
                                    #'maxfun': 100000, 'maxiter': 100000, 'iprint': -1, 'maxls': 40})
            
            theta_new = opt.x[0:-1]
            eta_new = opt.x[-1]
            #policy.print1()
            #print('\n',theta_new-theta)
            eta = eta_new
            theta = theta_new
            x0 = np.append(theta, eta)
            
            print('\ndual value optim : ',dual_function(x0))
            #theta0 = np.dot(10,np.random.random_sample(features[0].shape[0])) 
            #eta0 = 15
            #x = np.append(theta0, eta0)
            #print('dual value optim : ',dual_function(x))
            #theta0 = np.dot(10,np.random.random_sample(features[0].shape[0])) 
            #eta0 = 5
            #x = np.append(theta0, eta0)
            #print('dual value optim : ',dual_function(x0))
            print('dual gradient theta : ', np.linalg.norm(dual_function_jac(x0)[0:-1]))
            print('dual gradient eta : ', np.linalg.norm(dual_function_jac(x0)[-1]))
           
            # print("--- Optimization results ---")
            # print("success:\t{}".format(opt.success))
            # print("status :\t{}".format(opt.status))
            # print("message:\t{}".format(opt.message))
    
            
            
            # Value Function, has shape (n_states).
            #print(len(features))
            #print(len(theta))
            V[i] = np.dot(features, theta.T)
            policy_old = np.copy(policy.policy)
            
            # ACTOR - compute the new policy.
            #policy.print1()
            policy.update(bellman_error(theta), eta)
            #policy.print1()
            policy_new = policy.policy
            
            # computing running KL divergence
            S = 0
            #print(env.observation_space)
            #print(policy_old)
            #print(policy_new.policy)
            '''
            for state in range(env.observation_space.n):
                #action_probabilities_1 = policy_old.action_probabilities(state)
                action_probabilities_1 = policy_old[state]
                action_probabilities_2 = policy_new.action_probabilities(state)
                S += entropy(action_probabilities_1, action_probabilities_2)
            '''
            for state in range(env.observation_space.n):
                for action in range(env.action_space.n):
                    S += policy_new[state][action]*np.log(policy_new[state][action]/policy_old[state][action])
            print('KL Divergence: ', S)
                        
            
            # Expected return.
            
            def expected_return(env, R, policy, n_episodes, n_samples_expected_return):
                """
                Compute the expected return by sampling the environment while
                following the policy.
                J = sum_{s,a} q(s,a) * r(s,a)
                q(s,a) is the distribution of (state, action) following the policy.
                q(s,a) = mu_pi(s) * pi(a|s)
                
                Args:
                    env: openAI environment.
                    R: reward for each (state, action) pair.
                    policy: probability of taking action given the state.
                
                Returns:
                    The expected return.
                """
                # Estimate q(s,a) by sampling.
                q = np.zeros((env.observation_space.n, env.action_space.n))
                for _ in range(n_episodes):
                    state = env.reset()
                    for _ in range(n_samples_expected_return):
                        action_probabilities = policy.action_probabilities(state)
                        #print('Computing reward dist ', action_probabilities)
                        action = np.random.choice(np.arange(len(action_probabilities)),
                                                  p=action_probabilities)
                        next_state, reward, done, _ = env.step(action)
                        q[state][action] += 1
                        if done == True:
                            break
                        state = next_state
                        
                q /= (n_episodes * n_samples_expected_return)
                return np.sum(np.multiply(q, R))
            
            
            # TODO: compute J in a different manner.
            J[i] = expected_return(env, R, policy, 50, 50)
            np.set_printoptions(precision=3, suppress=True)
            # Print statistics.
            print("J: {:.2f}; {}".format(J[i], opt.message))
            
            print("eta is:", eta)
            
            print("policy is : \n", policy.policy)
        
        return J, V, policy, q_sa, theta, eta
    
    