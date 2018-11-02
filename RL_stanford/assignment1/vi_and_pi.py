### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from lake_envs import *

np.set_printoptions(precision=3)

def policy_evaluation(P, nS, nA, policy, gamma=0.9, max_iteration=1000, tol=1e-3):
    """Evaluate the value function from a given policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	policy: np.array
		The policy to evaluate. Maps states to actions.
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns
	-------
	value function: np.ndarray
		The value function from the given policy.
    """
    ############################
    V_old = np.zeros(nS)
    V_new = np.zeros(nS)
    i = 0
    while (i < max_iteration) and (np.linalg.norm(V_old-V_new,np.inf) > tol or i == 0):
        V_old[:] = V_new[:]
        for s in range(nS):
            R = 0
            bellmans = 0
            flag = False
            for (prob,s_prime,reward,terminal) in P[s][policy[s]]:
                R = reward
                if s == s_prime and terminal:
                    flag = True
                    break

                bellmans += prob * V_old[s_prime]
            if flag:
                V_new[s] = V_old[s]
            else:
                bellmans = bellmans * gamma + R
                V_new[s] = bellmans

        i += 1
    return V_new
    ############################
    return np.zeros(nS)


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new policy: np.ndarray
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""
    ############################
    for s in range(nS):
        max_val = -1
        max_a = policy[s]
        R = 0
        flag = False
        for a in range(nA):
            tmp_val = 0
            for (prob, s_prime, reward, terminal) in P[s][a]:
                if s == s_prime and terminal:
                    flag = True
                    break
                R = reward
                tmp_val += prob * value_from_policy[s_prime]
            if flag:
                break
            tmp_val = gamma * tmp_val + R
            if tmp_val > max_val:
                max_val = tmp_val
                max_a = a
        policy[s] = max_a

    return policy
    ############################
    return np.zeros(nS, dtype='int')


def policy_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """Runs policy iteration.

	You should use the policy_evaluation and policy_improvement methods to
	implement this method.

	Parameters
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    V_new  = policy_evaluation(P,nS,nA,policy,gamma,max_iteration,tol)
    i = 0
    while (i < max_iteration) and (np.linalg.norm(V - V_new, np.inf) > tol or i == 0):
        policy = policy_improvement(P, nS, nA, V_new, policy, gamma)
        V[:] = V_new[:]
        V_new = policy_evaluation(P, nS, nA, policy, gamma, max_iteration, tol)

        i += 1
    print('it took {} iterations'.format(i + 1))
    ############################
    return V, policy

def get_V_new(P,nS,nA,V):
    V_new = np.zeros_like(V)
    for s in range(nS):
        max_val = 0
        flag = False
        for a in range(nA):
            for (prob, s_prime, reward, terminal) in P[s][a]:
                if s_prime == s and terminal:
                    flag = True
                    break
                if reward > max_val:
                    max_val = reward
            if flag:
                break
        V_new[s] = max_val
    return V_new

def set_policy(P,nS,nA,V,policy,gamma):
    for s in range(nS):
        max_val = 0
        max_a = 0
        flag = False
        for a in range(nA):
            R = 0
            tmp_val = 0
            for (prob, s_prime, reward, terminal) in P[s][a]:
                R = reward
                if s_prime == s and terminal:
                    flag = True
                    break
                tmp_val += prob * V[s_prime]

            if flag:
                break
            tmp_val = tmp_val * gamma + R
            if tmp_val > max_val:
                max_val = tmp_val
                max_a = a
        policy[s] = max_a
    return policy

def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
    """
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, reward, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
    """
    V = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    V_new = get_V_new(P,nS,nA,V)

    i = 0
    while (i < max_iteration) and (np.linalg.norm(V - V_new, np.inf) > tol or i == 0):
        V[:] = V_new[:]
        for s in range(nS):
            max_val = 0
            flag = False
            for a in range(nA):
                R = 0
                tmp_val = 0
                for (prob, s_prime, reward, terminal) in P[s][a]:
                    R =reward
                    if s_prime == s and terminal:
                        flag = True
                        break
                    tmp_val += prob *  V[s_prime]

                if flag:
                    break
                tmp_val = tmp_val * gamma + R
                if tmp_val > max_val:
                    max_val = tmp_val
            V_new[s] = max_val
        i += 1
    print('it took {} iterations'.format(i+1))
    policy = set_policy(P,nS,nA,V,policy,gamma)

    ############################
    return V, policy

def example(env):
	"""Show an example of gym
	Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
	"""
	env.seed(0);
	from gym.spaces import prng; prng.seed(10) # for print the location
	# Generate the episode
	ob = env.reset()
	for t in range(100):
		env.render()
		a = env.action_space.sample()
		ob, rew, done, _ = env.step(a)
		if done:
			break
	assert done
	env.render();

def render_single(env, policy):
	"""Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

	episode_reward = 0
	ob = env.reset()
	for t in range(100):
		env.render()
		time.sleep(0.5) # Seconds between frames. Modify as you wish.
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	assert done
	env.render();
	print ("Episode reward: %f" % episode_reward)


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    env = gym.make("Stochastic-4x4-FrozenLake-v0")
    #Stochastic
    #Deterministic
    print( env.__doc__)
    print ("Here is an example of state, action, reward, and next state")
    example(env)
    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=40, tol=1e-3)
    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, max_iteration=20, tol=1e-3)
    print('value iteration value function:\n{}\npolicy:\n{}'.format(V_vi,p_vi))
    print('policy iteration value function:\n{}\npolicy:\n{}'.format(V_pi,p_pi))
