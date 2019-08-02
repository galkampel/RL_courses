import numpy as np
from numpy.linalg import inv

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

        V_new = np.copy(initialV)
        for n in range(nIterations):
            V_old = np.copy(V_new)
            V_new = []
            for a in range(self.nActions):
                V_new.append(self.R[a, :] + self.discount * np.matmul(self.T[a, :, :], V_old))
            V_new = np.stack(V_new, axis=-1)
            V_new = np.amax(V_new, axis=-1)

            epsilon = np.amax(np.absolute(V_old - V_new))
            if epsilon < tolerance:
                break

        V = np.copy(V_new)
        iterId = n + 1
        
        return [V,iterId,epsilon]

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        policy = []
        for a in range(self.nActions):
            policy.append(self.R[a, :] + self.discount * np.matmul(self.T[a, :, :], V))
        policy = np.stack(policy, axis=-1)
        policy = np.argmax(policy, axis=-1)

        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''
        from numpy.linalg import inv

        T = []
        R = []
        for a, s in zip(policy, np.arange(self.nStates)):
            T.append(self.T[a, s, :])
            R.append(self.R[a, s])
        T = np.stack(T, axis=0)
        R = np.array(R)

        V = np.matmul(inv(np.eye(self.nStates) - self.discount * T), R)

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

        policy0 = np.copy(initialPolicy)
        policy1 = np.copy(initialPolicy)
        V = np.zeros(self.nStates)
        iterId = nIterations

        for n in range(nIterations):
            V = self.evaluatePolicy(policy1)
            policy1 = self.extractPolicy(V)

            if np.array_equal(policy1, policy0):
                return [policy1,V,n+1]
            else:
                policy0 = np.copy(policy1)

        return [policy1,V,iterId]

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

        T = []
        R = []
        for a, s in zip(policy, np.arange(self.nStates)):
            T.append(self.T[a, s, :])
            R.append(self.R[a, s])
        T = np.stack(T, axis=0)
        R = np.array(R)

        V_new = np.copy(initialV)
        for n in range(nIterations):
            V_old = np.copy(V_new)
            V_new = R + self.discount * np.matmul(T, V_old)

        V = np.copy(V_new)
        iterId = n + 1
        epsilon = np.amax(np.absolute(V_old - V_new))

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

        policy0 = np.copy(initialPolicy)
        policy1 = np.copy(initialPolicy)

        V_new = np.copy(initialV)
        for n in range(nIterations):
            V_old = np.copy(V_new)
            V_new, _, _ = self.evaluatePolicyPartially(policy1, V_old, nIterations=nEvalIterations)
            policy1 = self.extractPolicy(V_new)

            epsilon = np.amax(np.absolute(V_old - V_new))
            if epsilon < tolerance:
                return [policy1,V_new,n+1,epsilon]
            else:
                policy0 = np.copy(policy1)

        V = np.copy(V_new)
        iterId = n + 1
        epsilon = np.amax(np.absolute(V_old - V_new))

        return [policy1,V,iterId,epsilon]
        
