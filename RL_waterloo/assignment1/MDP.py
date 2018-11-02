import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        V = initialV
        iterId = 0
        V_new = np.zeros_like(V)
        V_new[:] = V[:]
        epsilon = np.linalg.norm(V-V_new,np.inf)
        while iterId < nIterations and (epsilon > tolerance or iterId == 0):
            V[:] = V_new[:]
            V_new = np.max(self.R + self.discount * np.dot(self.T,V),axis=0)
            epsilon = np.linalg.norm(V-V_new,np.inf)
            iterId += 1

        print('Policy:\n{}\nV:\n{}\nnIterations = {}'.format(self.extractPolicy(V),V,iterId))
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        # policy = np.zeros(self.nStates)
        policy = np.argmax(self.R + self.discount * np.dot(self.T,V),axis=0)
        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        V = self.R[policy,np.arange(self.nStates)] + self.discount * np.dot(self.T[policy,np.arange(self.nStates),:], V)
        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        V_new = self.evaluatePolicy(initialPolicy)
        policy = initialPolicy
        while iterId < nIterations and not np.array_equal(V , V_new):
            while not np.array_equal(V , V_new):
                V[:] = V_new[:]
                V_new = self.R[policy, np.arange(self.nStates)] + self.discount * np.dot(
                    self.T[policy, np.arange(self.nStates), :], V)

            policy = self.extractPolicy(V)
            V_new = self.R[policy, np.arange(self.nStates)] + self.discount * np.dot(
                self.T[policy, np.arange(self.nStates), :], V)
            iterId += 1

        print('Policy iteration results:\nPolicy:\n{}\nV:\n{}\nnIterations = {}'.format(policy, V, iterId))
        print('Modified policy iteration results:')
        return [policy,V,iterId]
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = initialV.astype(np.float)
        V_new= self.R[policy, np.arange(self.nStates)] + self.discount * np.dot(self.T[policy, np.arange(self.nStates), :], V)
        iterId = 0
        epsilon = np.linalg.norm(V - V_new, np.inf)
        while iterId < nIterations and (epsilon > tolerance or iterId == 0):
            V[:] = V_new[:]
            V_new = self.R[policy, np.arange(self.nStates)] + self.discount * np.dot(
                    self.T[policy, np.arange(self.nStates), :], V)
            epsilon = np.linalg.norm(V - V_new, np.inf)
            iterId += 1

        return [V,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = initialPolicy
        V = initialV.astype(np.float)
        V_new = np.zeros(self.nStates)
        V_new[:] = V[:]
        iterId = 0
        epsilon = np.linalg.norm(V-V_new,np.inf)
        while iterId < nIterations and (epsilon > tolerance or iterId == 0):
            V[:] = V_new[:]
            V_new = self.evaluatePolicyPartially(policy,V,nEvalIterations,tolerance)[0]
            policy = self.extractPolicy(V_new)
            epsilon = np.linalg.norm(V - V_new, np.inf)
            iterId += 1
        print('nEvalIterations = {}'.format(nEvalIterations))
        print('Policy:\n{}\nV:\n{}\nnIterations = {}\n'.format(policy, V, iterId))
        return [policy,V,iterId,epsilon]
        
