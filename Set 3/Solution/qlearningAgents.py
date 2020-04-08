from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"

        self.qvalues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        # If the given state and action tuple does not exist in the Counter of q-values, return 0. If it does exist, return the Counter's value corresponding to the tuple.
        if (state, action) not in self.qvalues:

          return 0
        
        else:
        
          return self.qvalues[(state, action)]

        # util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # Initialise the value variable to None
        value = None
        
        # If there are no legal actions, return 0
        if len(self.getLegalActions(state)) == 0:
          return 0
        
        # Else, for each action, check whether the obtained q-value is greater than or equal to the value variable, or equal to None. In this case, the obtained q-value is assigned to the value variable
        else:
            
          for action in self.getLegalActions(state):

            tempQValue = self.getQValue(state, action)

            if value == None or value <= tempQValue:

                value = tempQValue

            # If the value variable remains unchanged after the previous step, set it to 0 
            if value == None:
              value = 0

        # Return the value variable
        return value
        # util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # Initialise the value and finalAction variables to None
        value = None
        finalAction = None

        # If there are no legal actions, return 0
        if len(self.getLegalActions(state)) == 0:
          return 0

        # Else, for each action, check whether the obtained q-value is greater than or equal to the value variable, or equal to None. In this case, the obtained q-value is assigned to the value variable, and the corresponding action is assigned to finalAction
        for action in self.getLegalActions(state):

          tempQValue = self.getQValue(state, action)

          if value == None or value <= tempQValue:

              finalAction = action

              value = tempQValue

        return finalAction
    
        # util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        # If there are no available legal actions, return None
        if len(legalActions) == 0:
          return None

        # Set the action to a random choice among the available legal actions if the flipCoin() function returns true
        if util.flipCoin(self.epsilon):
          action = random.choice(legalActions)
        
        # Compute the action by using the computeActionFromQValues() function if the flipCoin() function returns false
        else:
          action = self.computeActionFromQValues(state)

        # util.raiseNotDefined()
        
        # Return the action variable
        return action


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # If the q-values Counter does not have a value corresponding to (state, action), set it to 0
        if (state, action) not in self.qvalues:
          self.qvalues[(state, action)] = 0

        # Stores the current value corresponding to the (state, action) tuple
        currentValue = self.qvalues[(state, action)]
        
        # Stores the upcoming value corresponding to the (state, action) tuple
        nextValue = self.computeValueFromQValues(nextState)

        # Calculate the new q-value corresponding to the tuple
        self.qvalues[(state,action)] = (self.alpha * (reward + (self.discount * nextValue) - currentValue)) + currentValue

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"

        # Initialise the q-value to be 0
        qValue = 0

        # Stores all the available features 
        feats = self.featExtractor.getFeatures(state, action)

        # For each individual feature, increment the q-value by the product of the value of an individual feature and its corresponding weight
        for feature in feats:
          qValue += feats[feature] * self.weights[feature]

        # Return the q-value resultant
        return qValue

        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        
        # Stores all the available features
        feats = self.featExtractor.getFeatures(state, action)

        # Calculate the modifier like so:
        modifier = self.discount * self.getValue(nextState) + reward - self.getQValue(state, action)

        # For each available feature, increment its weight by the product of the alpha, the value of the feature and the modifier
        for feature in feats:
          self.weights[feature] += self.alpha * feats[feature] * modifier

        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:

            "*** YOUR CODE HERE ***"

            pass
