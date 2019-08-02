import numpy as np
import MDP
import RL2
import matplotlib.pyplot as plt

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

nTrials = 1000
nIterations = 200

banditProblem = RL2.RL2(mdp,sampleBernoulli)

# Test epsilon greedy strategy
avg_eps_greedy_bandit_rewards = np.zeros(nIterations)
for _ in range(nTrials):
    empiricalMeans,cum_eps_greedy_bandit_rewards = banditProblem.epsilonGreedyBandit(nIterations=nIterations)
    avg_eps_greedy_bandit_rewards += cum_eps_greedy_bandit_rewards
avg_eps_greedy_bandit_rewards /= nTrials
# print ("\nepsilonGreedyBandit results")
# print (empiricalMeans)

# Test Thompson sampling strategy
avg_thompson_bandit_rewards = np.zeros(nIterations)
for _ in range(nTrials):
    empiricalMeans,cum_thompson_rewards = banditProblem.thompsonSamplingBandit(prior=np.ones([mdp.nActions,2]),nIterations=nIterations)
    avg_thompson_bandit_rewards += cum_thompson_rewards
avg_thompson_bandit_rewards /= nTrials
# print ("\nthompsonSamplingBandit results")
# print (empiricalMeans)

# Test UCB strategy
avg_UCB_bandit_rewards = np.zeros(nIterations)
for _ in range(nTrials):
    empiricalMeans,cum_ucb_rewards = banditProblem.UCBbandit(nIterations=nIterations)
    avg_UCB_bandit_rewards += cum_ucb_rewards
avg_UCB_bandit_rewards /= nTrials
# print ("\nUCBbandit results")
# print (empiricalMeans)


plt.figure(figsize=(12,9))
plt.plot(range(nIterations),avg_eps_greedy_bandit_rewards,label='eGreedy')
plt.plot(range(nIterations),avg_thompson_bandit_rewards,label='Thompson Samping')
plt.plot(range(nIterations),avg_UCB_bandit_rewards,label='UCB')
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title("Average Reward vs Iteration")
plt.legend(loc='upper left')
plt.grid()
plt.savefig('banditRewards.png')
plt.close()
