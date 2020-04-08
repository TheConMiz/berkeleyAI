import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # Variable to store the current states
        currentStates = mdp.getStates()

        # For each iteration, do the following: 
        for i in range(iterations):

            # Initialise a variable with a blank Counter
            tempValues = util.Counter()

            # For each state that is currently available, if the corresponding action is not NULL, get its Q-value and store it in the tempValues variable.
            for state in currentStates:

                if self.getAction(state) != None:

                    tempValues[state] = self.getQValue(state, self.getAction(state))
                
                else:
                    continue

            # Update self.values by giving it the new Q-values
            self.values = tempValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # Initialise a qValue variable to 0
        qValue = 0

        # For each state and probability pair available, do the following
        for newState, prob in self.mdp.getTransitionStatesAndProbs(state, action):

            reward = self.mdp.getReward(state, action, newState)
            discount = self.discount
            nextValue = self.values[newState]

            # Increment the qValue variable by the sum of the reward and the next value multiplied by the discount, which is then multiplied by the corresponding probability.
            qValue += reward + (nextValue * discount) * prob

        # Return the resultant qValue
        return qValue

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        # Initialise variables qValue and finalAction to None
        qValue = None
        finalAction = None

        # If there are no possible actions, return none
        if len(self.mdp.getPossibleActions(state)) == 0:
            return None
        
        else:
            
            # For each available action, if qValue has not been changed since initialisation or is smaller than or equivalent to the computed tempQValue, its corresponding action is set as the value of finalAction. qValue is then updated with the larger (or equivalent) tempQValue
            for action in self.mdp.getPossibleActions(state):

                tempQValue = self.computeQValueFromValues(state, action)

                if qValue == None or qValue <= tempQValue:
                    
                    finalAction = action
                    
                    qValue = tempQValue

        # The returned finalAction corresponds to the largest qValue
        return finalAction

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
