import numpy as np
import MDP

class RL2:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        exp_policyParams = np.exp(policyParams)
        # pi_a_s = exp_policyParams / np.sum(exp_policyParams,axis=0)
        # probs = pi_a_s[:,state]
        # print('probs 1 ={}'.format(probs))
        # action = np.random.choice(range(len(probs)),p=probs)
        exp_policyParams = np.exp(policyParams)
        pi_a_s = exp_policyParams[:, state] / np.sum(exp_policyParams[:, state])
        action = np.random.choice(range(len(pi_a_s)), p=pi_a_s)

        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        gamma = self.mdp.discount
        model_mdp = MDP.MDP(defaultT,initialR,gamma)
        nActions = model_mdp.nActions
        nStates = model_mdp.nStates
        V = np.zeros(nStates)
        policy = np.zeros(nStates,int)
        policy, V, _, _ = model_mdp.modifiedPolicyIteration(policy, V, nIterations=1000)
        n_sa = np.zeros((nStates,nActions))
        n_sa_s_next = np.zeros((nStates,nActions,nStates))
        cumRewards = np.zeros(nEpisodes)
        for epoch in range(nEpisodes):
            s = s0
            a = None
            for t in range(nSteps):
                if np.random.rand() > epsilon:
                    # a = np.argmax(policy)
                    a = policy[s]
                else:
                    a = np.random.choice(nActions)

                n_sa[s, a] += 1
                r, s_next = self.sampleRewardAndNextState(s,a)
                cumRewards[epoch] += r * (gamma ** t)
                n_sa_s_next[s,a,s_next] += 1
                model_mdp.T[a,s,:] = n_sa_s_next[s,a,:] / n_sa[s,a]
                model_mdp.R[a,s] = (r + (n_sa[s,a]-1) * model_mdp.R[a,s]) / n_sa[s,a]
                policy, V, _, _ = model_mdp.modifiedPolicyIteration(policy, V,nIterations=1000)
                s = s_next

        # print('transition function = {}'.format(model_mdp.T))
        # print('reward function = {}'.format(model_mdp.R))
        return [V,policy,cumRewards]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
                are combined in one procedure by sampling a random action with
                probabilty epsilon and performing Boltzmann exploration otherwise.
                When epsilon and temperature are set to 0, there is no exploration.

                Inputs:
                s0 -- initial state
                initialQ -- initial Q function (|A|x|S| array)
                nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
                nSteps -- # of steps per episode
                epsilon -- probability with which an action is chosen at random
                temperature -- parameter that regulates Boltzmann exploration

                Outputs:
                Q -- final Q function (|A|x|S| array)
                policy -- final policy
                '''
        Q = initialQ
        gamma = self.mdp.discount
        temperature = float(temperature)
        cumRewards = np.zeros(nEpisodes)
        nActions,nStates = Q.shape
        n_sa = np.zeros_like(Q)
        for epoch in range(nEpisodes):
            s = s0
            for t in range(nSteps):

                if np.random.rand() < epsilon:
                    a = np.random.choice(nActions)
                else:
                    if temperature == 0:
                        a = np.argmax(Q[:,s])
                    else:
                        Q_exp = np.exp(Q)
                        boltzman_probs = (Q_exp / temperature) / np.sum((Q_exp / temperature)   ,axis=0)
                        a = np.random.choice(nActions,size=1,replace=False,p=boltzman_probs[:,s])
                r,s_next = self.sampleRewardAndNextState(s,a)
                n_sa[a,s] += 1
                alpha = 1 / n_sa[a,s]
                Q[a,s] += alpha * (r + gamma * np.max(Q[:,s_next] - Q[a,s]))
                cumRewards[epoch] += (gamma ** t) * r
                s = s_next

        policy = Q.argmax(axis=0)
        return [Q, policy, cumRewards]

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        nActions = self.mdp.nActions
        empiricalMeans = np.zeros(nActions)
        nRewards = np.zeros(nIterations)
        empirical_sums = np.zeros_like(empiricalMeans)
        freq = np.zeros(nActions)
        # cumActProb = np.cumsum(np.ones(self.mdp.nActions) / self.mdp.nActions)
        a = None
        for t in range(nIterations):
            if np.random.rand() <= 1 / (t+1) :
                a = np.random.choice(nActions)
                # a = np.where(cumActProb >= np.random.rand(1))[0][0]
            else:
                if any(empirical_sums == 0):
                    a = np.argmin(empirical_sums)
                else:
                    a = np.argmax((empirical_sums / freq))

            r,_ = self.sampleRewardAndNextState(0,a)
            # rewards = np.zeros(nActions)
            # rewards[a] = r
            # empiricalMeans = (rewards + t * empiricalMeans) / (t+1)
            empirical_sums[a] += r
            freq[a] += 1
            nRewards[t] = r
        empiricalMeans = empirical_sums / freq

        return empiricalMeans,nRewards

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        #empirical means as thetas (prior - uniform)
        nActions = self.mdp.nActions
        empiricalMeans = np.zeros(nActions)
        empiricalAvgs = np.zeros(nActions)
        cumRewards = np.zeros(nIterations)
        beta_params = np.copy(prior)
        freq = np.zeros(nActions)
        for t in range(nIterations):
            for a in range(nActions):
                empiricalAvgs[a] = np.sum(np.random.beta(beta_params[a,0],beta_params[a,1],size=k)) / k
            a = np.argmax(empiricalAvgs)
            r,_ = self.sampleRewardAndNextState(0,a)
            cumRewards[t] = r
            beta_params[a,:] += np.array([r,1-r]) # because r is either 1 or 0
            empiricalMeans[a] +=  r
            freq[a] += 1

        empiricalMeans /= freq
        return empiricalMeans,cumRewards

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        cumRewards = np.zeros(nIterations)
        nActions = self.mdp.nActions
        empiricalMeans = np.zeros(nActions)
        n_a = np.zeros(nActions)
        for t in range(nIterations):
            if any(n_a == 0):
                a = np.argmin(empiricalMeans)
                r, _ = self.sampleRewardAndNextState(0, a)
                cumRewards[t] = r
                n_a[a] += 1
                empiricalMeans[a] += r

            else:
                # if t < 10:
                    # print('sum = {}\nna = {}'.format(empiricalMeans,n_a))
                a = np.argmax(empiricalMeans + np.sqrt((2 * np.log(t)) / n_a))
                r,_ = self.sampleRewardAndNextState(0,a)
                cumRewards[t] = r
                reward_vec = np.zeros(nActions)
                reward_vec[a] = r
                empiricalMeans = (n_a * empiricalMeans + reward_vec)
                n_a[a] += 1
                empiricalMeans /= n_a

        return empiricalMeans,cumRewards

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded

        # policyParams = np.zeros((self.mdp.nActions, self.mdp.nStates))
        gamma = self.mdp.discount
        policyParams = np.copy(initialPolicyParams)
        alpha = 0.01
        cumRewards = np.zeros(nEpisodes)
        nActions = self.mdp.nActions
        for epoch in range(nEpisodes):
            R = np.zeros(nSteps)
            s_a = []
            s = s0
            for t in range(nSteps):
                a = self.sampleSoftmaxPolicy(policyParams, s)
                r, s_next = self.sampleRewardAndNextState(s, a)
                R[t] = r
                cumRewards[epoch] += r * (gamma** t)
                s_a.append((s,a))
                s = s_next
            #update parameters
            #assuming learning rate = 1 (not given)

            for n in range(nSteps):
                Gn = np.sum(R[n:] * np.power(gamma, np.arange(nSteps-n)))
                s,a = s_a[n]
                rel_state_actions = np.zeros(nActions)
                rel_state_actions[a] = 1
                policyParams[:, s] += alpha * (gamma ** n) * Gn * (rel_state_actions - np.exp(policyParams[:,s]) / (np.sum(np.exp(policyParams[:, s]))))  # / pi_as
                # print('policyParams :\n{}'.format(policyParams))


        return policyParams,cumRewards
