import numpy as np
import MDP




class RL:
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

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = initialQ
        # policy = np.zeros(self.mdp.nStates,int)
        discount = self.mdp.discount
        nActions = self.mdp.nActions
        update_counts = np.zeros_like(self.mdp.R,np.int)
        accum_rewards = np.zeros(nEpisodes)
        i = 0
        for episode in range(nEpisodes):
            s = s0
            a = 0
            acc_rewards = 0
            for step in range(nSteps):
                thres = np.random.rand(1)
                if epsilon > thres:  # epsilon greedy
                    a = np.random.randint(nActions)
                elif  temperature == 0:  # only eploitatin
                    a = np.argmax(Q[:,s])
                else: # boltzmann exploration
                    Q_dist = (np.exp(Q[:,s] / temperature) )
                    Q_dist /= np.sum(Q_dist)
                    a = np.random.choice(nActions, p=Q_dist)
                update_counts[a,s] += 1 #( learning rate = 1/ update_counts[s,a])
                reward, s_prime = self.sampleRewardAndNextState(s, a)
                acc_rewards += reward * discount ** step
                Q[a,s] += (1 / update_counts[a,s]) * (reward + discount * np.max(Q[:,s_prime]) - Q[a,s])

                s = s_prime
            accum_rewards[i] = acc_rewards

            i += 1

        policy = np.argmax(Q,axis = 0)
        return [Q,policy,accum_rewards]


# import numpy as np
# import MDP
#
# class RL:
#     def __init__(self,mdp,sampleReward):
#         '''Constructor for the RL class
#
#         Inputs:
#         mdp -- Markov decision process (T, R, discount)
#         sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
#         This function takes one argument: the mean of the distributon and
#         returns a sample from the distribution.
#         '''
#
#         self.mdp = mdp
#         self.sampleReward = sampleReward
#
#     def sampleRewardAndNextState(self,state,action):
#         '''Procedure to sample a reward and the next state
#         reward ~ Pr(r)
#         nextState ~ Pr(s'|s,a)
#
#         Inputs:
#         state -- current state
#         action -- action to be executed
#
#         Outputs:
#         reward -- sampled reward
#         nextState -- sampled next state
#         '''
#
#         reward = self.sampleReward(self.mdp.R[action,state])
#         cumProb = np.cumsum(self.mdp.T[action,state,:])
#         nextState = np.where(cumProb >= np.random.rand(1))[0][0]
#         return [reward,nextState]
#
#     def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
#         '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
#         are combined in one procedure by sampling a random action with
#         probabilty epsilon and performing Boltzmann exploration otherwise.
#         When epsilon and temperature are set to 0, there is no exploration.
#
#         Inputs:
#         s0 -- initial state
#         initialQ -- initial Q function (|A|x|S| array)
#         nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
#         nSteps -- # of steps per episode
#         epsilon -- probability with which an action is chosen at random
#         temperature -- parameter that regulates Boltzmann exploration
#
#         Outputs:
#         Q -- final Q function (|A|x|S| array)
#         policy -- final policy
#         '''
#
#         temperature = float(temperature)
#         counts = np.zeros([self.mdp.nActions, self.mdp.nStates])
#         Q = np.copy(initialQ)
#         ep_rewards = []
#
#         for episode in range(nEpisodes):
#             s = np.copy(s0)
#             discounted_rewards = 0.
#             for step in range(nSteps):
#                 if np.random.uniform() < epsilon:
#                     a = np.random.randint(self.mdp.nActions)
#                 else:
#                     if temperature == 0.:
#                         a = np.argmax(Q[:, s])
#                     else:
#                         prob_a = np.exp(Q[:, s] / temperature) / np.sum(np.exp(Q[:, s] / temperature))
#                         a = np.argmax(np.random.multinomial(1, prob_a))
#
#                 r, next_s = self.sampleRewardAndNextState(s, a)
#                 discounted_rewards += self.mdp.discount**step * r
#
#                 counts[a, s] += 1.
#                 Q[a, s] += (1. / counts[a, s]) * (r + self.mdp.discount * np.amax(Q[:, next_s]) - Q[a, s])
#                 s = np.copy(next_s)
#             ep_rewards.append(discounted_rewards)
#
#         # temporary values to ensure that the code compiles until this
#         # function is coded
#         policy = np.argmax(Q, axis=0)
#
#         return [Q,policy,np.array(ep_rewards)]
