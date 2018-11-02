import numpy as np
import MDP
import RL

import sys

''' Construct a simple maze MDP

  Grid world layout:

  ---------------------
  |  0 |  1 |  2 |  3 |
  ---------------------
  |  4 |  5 |  6 |  7 |
  ---------------------
  |  8 |  9 | 10 | 11 |
  ---------------------
  | 12 | 13 | 14 | 15 |
  ---------------------

  Goal state: 15
  Bad state: 9
  End state: 16

  The end state is an absorbing state that the agent transitions
  to after visiting the goal state.

  There are 17 states in total (including the end state)
  and 4 actions (up, down, left, right).'''

# Transition function: |A| x |S| x |S'| array
T = np.zeros([4,17,17])
a = 0.8;  # intended move
b = 0.1;  # lateral move

# up (a = 0)

T[0,0,0] = a+b;
T[0,0,1] = b;

T[0,1,0] = b;
T[0,1,1] = a;
T[0,1,2] = b;

T[0,2,1] = b;
T[0,2,2] = a;
T[0,2,3] = b;

T[0,3,2] = b;
T[0,3,3] = a+b;

T[0,4,4] = b;
T[0,4,0] = a;
T[0,4,5] = b;

T[0,5,4] = b;
T[0,5,1] = a;
T[0,5,6] = b;

T[0,6,5] = b;
T[0,6,2] = a;
T[0,6,7] = b;

T[0,7,6] = b;
T[0,7,3] = a;
T[0,7,7] = b;

T[0,8,8] = b;
T[0,8,4] = a;
T[0,8,9] = b;

T[0,9,8] = b;
T[0,9,5] = a;
T[0,9,10] = b;

T[0,10,9] = b;
T[0,10,6] = a;
T[0,10,11] = b;

T[0,11,10] = b;
T[0,11,7] = a;
T[0,11,11] = b;

T[0,12,12] = b;
T[0,12,8] = a;
T[0,12,13] = b;

T[0,13,12] = b;
T[0,13,9] = a;
T[0,13,14] = b;

T[0,14,13] = b;
T[0,14,10] = a;
T[0,14,15] = b;

T[0,15,16] = 1;
T[0,16,16] = 1;

# down (a = 1)

T[1,0,0] = b;
T[1,0,4] = a;
T[1,0,1] = b;

T[1,1,0] = b;
T[1,1,5] = a;
T[1,1,2] = b;

T[1,2,1] = b;
T[1,2,6] = a;
T[1,2,3] = b;

T[1,3,2] = b;
T[1,3,7] = a;
T[1,3,3] = b;

T[1,4,4] = b;
T[1,4,8] = a;
T[1,4,5] = b;

T[1,5,4] = b;
T[1,5,9] = a;
T[1,5,6] = b;

T[1,6,5] = b;
T[1,6,10] = a;
T[1,6,7] = b;

T[1,7,6] = b;
T[1,7,11] = a;
T[1,7,7] = b;

T[1,8,8] = b;
T[1,8,12] = a;
T[1,8,9] = b;

T[1,9,8] = b;
T[1,9,13] = a;
T[1,9,10] = b;

T[1,10,9] = b;
T[1,10,14] = a;
T[1,10,11] = b;

T[1,11,10] = b;
T[1,11,15] = a;
T[1,11,11] = b;

T[1,12,12] = a+b;
T[1,12,13] = b;

T[1,13,12] = b;
T[1,13,13] = a;
T[1,13,14] = b;

T[1,14,13] = b;
T[1,14,14] = a;
T[1,14,15] = b;

T[1,15,16] = 1;
T[1,16,16] = 1;

# left (a = 2)

T[2,0,0] = a+b;
T[2,0,4] = b;

T[2,1,1] = b;
T[2,1,0] = a;
T[2,1,5] = b;

T[2,2,2] = b;
T[2,2,1] = a;
T[2,2,6] = b;

T[2,3,3] = b;
T[2,3,2] = a;
T[2,3,7] = b;

T[2,4,0] = b;
T[2,4,4] = a;
T[2,4,8] = b;

T[2,5,1] = b;
T[2,5,4] = a;
T[2,5,9] = b;

T[2,6,2] = b;
T[2,6,5] = a;
T[2,6,10] = b;

T[2,7,3] = b;
T[2,7,6] = a;
T[2,7,11] = b;

T[2,8,4] = b;
T[2,8,8] = a;
T[2,8,12] = b;

T[2,9,5] = b;
T[2,9,8] = a;
T[2,9,13] = b;

T[2,10,6] = b;
T[2,10,9] = a;
T[2,10,14] = b;

T[2,11,7] = b;
T[2,11,10] = a;
T[2,11,15] = b;

T[2,12,8] = b;
T[2,12,12] = a+b;

T[2,13,9] = b;
T[2,13,12] = a;
T[2,13,13] = b;

T[2,14,10] = b;
T[2,14,13] = a;
T[2,14,14] = b;

T[2,15,16] = 1;
T[2,16,16] = 1;

# right (a = 3)

T[3,0,0] = b;
T[3,0,1] = a;
T[3,0,4] = b;

T[3,1,1] = b;
T[3,1,2] = a;
T[3,1,5] = b;

T[3,2,2] = b;
T[3,2,3] = a;
T[3,2,6] = b;

T[3,3,3] = a+b;
T[3,3,7] = b;

T[3,4,0] = b;
T[3,4,5] = a;
T[3,4,8] = b;

T[3,5,1] = b;
T[3,5,6] = a;
T[3,5,9] = b;

T[3,6,2] = b;
T[3,6,7] = a;
T[3,6,10] = b;

T[3,7,3] = b;
T[3,7,7] = a;
T[3,7,11] = b;

T[3,8,4] = b;
T[3,8,9] = a;
T[3,8,12] = b;

T[3,9,5] = b;
T[3,9,10] = a;
T[3,9,13] = b;

T[3,10,6] = b;
T[3,10,11] = a;
T[3,10,14] = b;

T[3,11,7] = b;
T[3,11,11] = a;
T[3,11,15] = b;

T[3,12,8] = b;
T[3,12,13] = a;
T[3,12,12] = b;

T[3,13,9] = b;
T[3,13,14] = a;
T[3,13,13] = b;

T[3,14,10] = b;
T[3,14,15] = a;
T[3,14,14] = b;

T[3,15,16] = 1;
T[3,16,16] = 1;

# Reward function: |A| x |S| array
R = -1 * np.ones([4,17]);

# set rewards
R[:,15] = 100;  # goal state
R[:,9] = -70;   # bad state
R[:,16] = 0;    # end state

# Discount factor: scalar in [0,1)
discount = 0.95

# MDP object
mdp = MDP.MDP(T,R,discount)

# RL problem
rlProblem = RL.RL(mdp,np.random.normal)

# Test Q-learning
import matplotlib.pyplot as plt
import matplotlib.cm as cm
trials = 100
nEpisodes = 200
nSteps = 100
epsilons = [0.05, 0.1, 0.3, 0.5]
accRewards = np.zeros((len(epsilons),nEpisodes))

for i,epsilon in enumerate(epsilons):
    for trial in range(trials):
        print('epsilon = {}, trial = {}'.format(epsilon,trial))
        [Q, policy, acc_rewards] = rlProblem.qLearning(s0=0, initialQ=np.zeros([mdp.nActions, mdp.nStates]),
                                                        nEpisodes=nEpisodes, nSteps=nSteps, epsilon=epsilon)
        accRewards[i,:] += acc_rewards
    accRewards[i,:] /= trials




# print('epsilon = 0.05:\nQ:\n{}\npolicy = {}'.format(0.05,Q,policy))
# [Q,policy,acc_rewards2] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.1)
# [Q,policy,acc_rewards3] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.3)
# [Q,policy,acc_rewards4] = rlProblem.qLearning(s0=0,initialQ=np.zeros([mdp.nActions,mdp.nStates]),nEpisodes=200,nSteps=100,epsilon=0.5)

labels = ["epsilon = {}".format(epsilon) for epsilon in epsilons]
N = len(epsilons)
colors = cm.Pastel1(list(range(N)))
x = np.arange(nEpisodes)
# fig,ax = plt.subplots()
# ax.set_xticks(list(i for i in range(0,nSteps)))
# plt.xlim(0,nSteps)
for i in range(N):
    plt.plot(x,accRewards[i,:],label = labels[i],color=colors[i])
plt.grid()
plt.title("Cumulative Rewards for different epsilons")
plt.ylabel("Cumulative Discounted Rewards")
plt.xlabel("Episode")
plt.legend(loc = 'lower right')
plt.savefig("plot2.png")
plt.close()

# import numpy as np
# import MDP
# import RL
#
# import sys
#
# ''' Construct a simple maze MDP
#
#   Grid world layout:
#
#   ---------------------
#   |  0 |  1 |  2 |  3 |
#   ---------------------
#   |  4 |  5 |  6 |  7 |
#   ---------------------
#   |  8 |  9 | 10 | 11 |
#   ---------------------
#   | 12 | 13 | 14 | 15 |
#   ---------------------
#
#   Goal state: 15
#   Bad state: 9
#   End state: 16
#
#   The end state is an absorbing state that the agent transitions
#   to after visiting the goal state.
#
#   There are 17 states in total (including the end state)
#   and 4 actions (up, down, left, right).'''
#
# # Transition function: |A| x |S| x |S'| array
# T = np.zeros([4, 17, 17])
# a = 0.8;  # intended move
# b = 0.1;  # lateral move
#
# # up (a = 0)
#
# T[0, 0, 0] = a + b;
# T[0, 0, 1] = b;
#
# T[0, 1, 0] = b;
# T[0, 1, 1] = a;
# T[0, 1, 2] = b;
#
# T[0, 2, 1] = b;
# T[0, 2, 2] = a;
# T[0, 2, 3] = b;
#
# T[0, 3, 2] = b;
# T[0, 3, 3] = a + b;
#
# T[0, 4, 4] = b;
# T[0, 4, 0] = a;
# T[0, 4, 5] = b;
#
# T[0, 5, 4] = b;
# T[0, 5, 1] = a;
# T[0, 5, 6] = b;
#
# T[0, 6, 5] = b;
# T[0, 6, 2] = a;
# T[0, 6, 7] = b;
#
# T[0, 7, 6] = b;
# T[0, 7, 3] = a;
# T[0, 7, 7] = b;
#
# T[0, 8, 8] = b;
# T[0, 8, 4] = a;
# T[0, 8, 9] = b;
#
# T[0, 9, 8] = b;
# T[0, 9, 5] = a;
# T[0, 9, 10] = b;
#
# T[0, 10, 9] = b;
# T[0, 10, 6] = a;
# T[0, 10, 11] = b;
#
# T[0, 11, 10] = b;
# T[0, 11, 7] = a;
# T[0, 11, 11] = b;
#
# T[0, 12, 12] = b;
# T[0, 12, 8] = a;
# T[0, 12, 13] = b;
#
# T[0, 13, 12] = b;
# T[0, 13, 9] = a;
# T[0, 13, 14] = b;
#
# T[0, 14, 13] = b;
# T[0, 14, 10] = a;
# T[0, 14, 15] = b;
#
# T[0, 15, 16] = 1;
# T[0, 16, 16] = 1;
#
# # down (a = 1)
#
# T[1, 0, 0] = b;
# T[1, 0, 4] = a;
# T[1, 0, 1] = b;
#
# T[1, 1, 0] = b;
# T[1, 1, 5] = a;
# T[1, 1, 2] = b;
#
# T[1, 2, 1] = b;
# T[1, 2, 6] = a;
# T[1, 2, 3] = b;
#
# T[1, 3, 2] = b;
# T[1, 3, 7] = a;
# T[1, 3, 3] = b;
#
# T[1, 4, 4] = b;
# T[1, 4, 8] = a;
# T[1, 4, 5] = b;
#
# T[1, 5, 4] = b;
# T[1, 5, 9] = a;
# T[1, 5, 6] = b;
#
# T[1, 6, 5] = b;
# T[1, 6, 10] = a;
# T[1, 6, 7] = b;
#
# T[1, 7, 6] = b;
# T[1, 7, 11] = a;
# T[1, 7, 7] = b;
#
# T[1, 8, 8] = b;
# T[1, 8, 12] = a;
# T[1, 8, 9] = b;
#
# T[1, 9, 8] = b;
# T[1, 9, 13] = a;
# T[1, 9, 10] = b;
#
# T[1, 10, 9] = b;
# T[1, 10, 14] = a;
# T[1, 10, 11] = b;
#
# T[1, 11, 10] = b;
# T[1, 11, 15] = a;
# T[1, 11, 11] = b;
#
# T[1, 12, 12] = a + b;
# T[1, 12, 13] = b;
#
# T[1, 13, 12] = b;
# T[1, 13, 13] = a;
# T[1, 13, 14] = b;
#
# T[1, 14, 13] = b;
# T[1, 14, 14] = a;
# T[1, 14, 15] = b;
#
# T[1, 15, 16] = 1;
# T[1, 16, 16] = 1;
#
# # left (a = 2)
#
# T[2, 0, 0] = a + b;
# T[2, 0, 4] = b;
#
# T[2, 1, 1] = b;
# T[2, 1, 0] = a;
# T[2, 1, 5] = b;
#
# T[2, 2, 2] = b;
# T[2, 2, 1] = a;
# T[2, 2, 6] = b;
#
# T[2, 3, 3] = b;
# T[2, 3, 2] = a;
# T[2, 3, 7] = b;
#
# T[2, 4, 0] = b;
# T[2, 4, 4] = a;
# T[2, 4, 8] = b;
#
# T[2, 5, 1] = b;
# T[2, 5, 4] = a;
# T[2, 5, 9] = b;
#
# T[2, 6, 2] = b;
# T[2, 6, 5] = a;
# T[2, 6, 10] = b;
#
# T[2, 7, 3] = b;
# T[2, 7, 6] = a;
# T[2, 7, 11] = b;
#
# T[2, 8, 4] = b;
# T[2, 8, 8] = a;
# T[2, 8, 12] = b;
#
# T[2, 9, 5] = b;
# T[2, 9, 8] = a;
# T[2, 9, 13] = b;
#
# T[2, 10, 6] = b;
# T[2, 10, 9] = a;
# T[2, 10, 14] = b;
#
# T[2, 11, 7] = b;
# T[2, 11, 10] = a;
# T[2, 11, 15] = b;
#
# T[2, 12, 8] = b;
# T[2, 12, 12] = a + b;
#
# T[2, 13, 9] = b;
# T[2, 13, 12] = a;
# T[2, 13, 13] = b;
#
# T[2, 14, 10] = b;
# T[2, 14, 13] = a;
# T[2, 14, 14] = b;
#
# T[2, 15, 16] = 1;
# T[2, 16, 16] = 1;
#
# # right (a = 3)
#
# T[3, 0, 0] = b;
# T[3, 0, 1] = a;
# T[3, 0, 4] = b;
#
# T[3, 1, 1] = b;
# T[3, 1, 2] = a;
# T[3, 1, 5] = b;
#
# T[3, 2, 2] = b;
# T[3, 2, 3] = a;
# T[3, 2, 6] = b;
#
# T[3, 3, 3] = a + b;
# T[3, 3, 7] = b;
#
# T[3, 4, 0] = b;
# T[3, 4, 5] = a;
# T[3, 4, 8] = b;
#
# T[3, 5, 1] = b;
# T[3, 5, 6] = a;
# T[3, 5, 9] = b;
#
# T[3, 6, 2] = b;
# T[3, 6, 7] = a;
# T[3, 6, 10] = b;
#
# T[3, 7, 3] = b;
# T[3, 7, 7] = a;
# T[3, 7, 11] = b;
#
# T[3, 8, 4] = b;
# T[3, 8, 9] = a;
# T[3, 8, 12] = b;
#
# T[3, 9, 5] = b;
# T[3, 9, 10] = a;
# T[3, 9, 13] = b;
#
# T[3, 10, 6] = b;
# T[3, 10, 11] = a;
# T[3, 10, 14] = b;
#
# T[3, 11, 7] = b;
# T[3, 11, 11] = a;
# T[3, 11, 15] = b;
#
# T[3, 12, 8] = b;
# T[3, 12, 13] = a;
# T[3, 12, 12] = b;
#
# T[3, 13, 9] = b;
# T[3, 13, 14] = a;
# T[3, 13, 13] = b;
#
# T[3, 14, 10] = b;
# T[3, 14, 15] = a;
# T[3, 14, 14] = b;
#
# T[3, 15, 16] = 1;
# T[3, 16, 16] = 1;
#
# # Reward function: |A| x |S| array
# R = -1 * np.ones([4, 17]);
#
# # set rewards
# R[:, 15] = 100;  # goal state
# R[:, 9] = -70;  # bad state
# R[:, 16] = 0;  # end state
#
# # Discount factor: scalar in [0,1)
# discount = 0.95
#
# # MDP object
# mdp = MDP.MDP(T, R, discount)
#
# # RL problem
# rlProblem = RL.RL(mdp, np.random.normal)
#
# import matplotlib.pyplot as plt
#
# trials = 100
# epsilons = [.05, .1, .3, .5]
# handles = []
# for epsilon in epsilons:
#     R = []
#     for trial in range(trials):
#         print('epsilon:{}\ttrial = {}'.format(epsilon,trial))
#         [Q, policy, ep_rewards] = rlProblem.qLearning(s0=0, initialQ=np.zeros([mdp.nActions, mdp.nStates]),
#                                                       nEpisodes=200, nSteps=100, epsilon=epsilon)
#         R.append(ep_rewards)
#     R = np.stack(R, axis=0)
#     R = np.mean(R, axis=0)
#
#     handle, = plt.plot(np.arange(len(R)), R, label='epsilon=' + str(epsilon))
#     handles.append(handle)
#
# plt.grid()
# plt.legend(handles=handles)
# plt.xlabel('Episode')
# plt.ylabel('Cumulative Discounted Rewards')
# # plt.show()
# plt.savefig('part2.eps')
