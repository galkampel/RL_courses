import numpy as np
import MDP
import RL2


def sampleBernoulli(mean):
    ''' function to obtain a sample from a Bernoulli distribution

    Input:
    mean -- mean of the Bernoulli
    
    Output:
    sample -- sample (0 or 1)
    '''

    if np.random.rand(1) < mean: return 1
    else: return 0


# Multi-arm bandit problems (3 arms with probabilities 0.3, 0.5 and 0.7)
T = np.array([[[1]],[[1]],[[1]]])
R = np.array([[0.3],[0.5],[0.7]])
discount = 0.999
mdp = MDP.MDP(T,R,discount)



banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy

empiricalMeans,cum_eps_greedy_bandit_rewards = banditProblem.epsilonGreedyBandit(nIterations=200)

# print ("\nepsilonGreedyBandit results")
# print ('empirical means = {}\ncumrewards = {}'.format(empiricalMeans,cumRewards))

# Test Thompson sampling strategy

empiricalMeans,cum_thompson_rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=200)

# print ( "\nthompsonSamplingBandit results")
# print( empiricalMeans)

# Test UCB strategy

empiricalMeans,cum_ucb_rewards = banditProblem.UCBbandit(nIterations=200)

# print( "\nUCBbandit results")
# print( empiricalMeans)

''' Construct simple MDP as described in Lecture 2a Slides 13-14'''
T = np.array([[[0.5,0.5,0,0],[0,1,0,0],[0.5,0.5,0,0],[0,1,0,0]],[[1,0,0,0],[0.5,0,0,0.5],[0.5,0,0.5,0],[0,0,0.5,0.5]]])
R = np.array([[0,0,10,10],[0,0,10,10]])
discount = 0.9
mdp = MDP.MDP(T,R,discount)
rlProblem = RL2.RL2(mdp,np.random.normal)

# Test REINFORCE
policy,cumRewards = rlProblem.reinforce(s0=0,initialPolicyParams=np.random.rand(mdp.nActions,mdp.nStates),nEpisodes=1000,nSteps=100)
# print( "\nREINFORCE results")
# print('Policy params:\n{}\ncumRewards = {}'.format(policy,cumRewards))
# print( policy)

# Test model-based RL
[V,policy,cumRewards] = rlProblem.modelBasedRL(s0=0,defaultT=np.ones([mdp.nActions,mdp.nStates,mdp.nStates])/mdp.nStates,initialR=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=100,nSteps=100,epsilon=0.05)
# print( "\nmodel-based RL results")
# print('V = {}\nPolicy:\n{}\ncumRewards = {}'.format(V,policy,cumRewards))
# print(policy)
