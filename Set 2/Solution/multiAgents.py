from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        prevFood = currentGameState.getFood()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Variable to store the distance to the closest food
        closestFood = 999999

        #  Variable to store the distance
        ghostDistances = 1

        # Run through the list of food coordinates, and select the closest one to the current position of PacMan
        for i in newFood.asList():
          # Calculate the Manhattan Distance between the current position and the food coordinate
          i = manhattanDistance(newPos, i)
          #  Compare i with the current value of closestFood, and assign the smaller of the two to closestFood
          closestFood = min(closestFood, i)

        # Run through the current positions of the ghosts
        for i in successorGameState.getGhostPositions():
          # Calculate the Manhattan Distance between the current position and the ghost coordinates, and accumulate those values into ghostDistances
          i = manhattanDistance(newPos, i)
          # Accumulate the distances for all ghost coordinates
          ghostDistances += i

        # The score modifier is calculated by getting the inverse of the closestFood value, and subtracting the inverse of the ghostDistances value from it. This way, the scoreModifier value is higher when PacMan is closer to the food, and further away from the ghosts.
        scoreModifier = float(1 / closestFood) - float(1 / ghostDistances)

        # Return the sum of getScore() and the scoreModifier
        return successorGameState.getScore() + scoreModifier


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # A combined miniMax function
        def miniMax(gameState, currentDepth, agentID):
          
          # Variable for storing the results of the miniMax function
          tempList = []

          # Base conditions for early termination of function
          if gameState.isWin() or gameState.isLose() or self.depth == currentDepth:

            return self.evaluationFunction(gameState)

          # If the agent is PacMan, run miniMax on all of its available legal moves, and return the largest of the results generated
          if agentID == 0:
            
            # For each legal action, append the miniMax results to tempList
            for i in gameState.getLegalActions(agentID):
              tempList.append(miniMax(gameState.generateSuccessor(agentID, i), currentDepth, agentID + 1))

            # Return the maximum value stored in the tempList list
            return max(tempList)

          # If the agent is a ghost, handle the change of agentID and currentDepth depending on how many ghosts there are 
          else:
            
            # Handle the change from  ghost agents to PacMan: if the agentID is equivalent to one less than the total number of agents, it means the next agent has to be PacMan
            if agentID == gameState.getNumAgents() - 1:
              tempAgentID = 0
              # Increase the current depth by 1
              currentDepth += 1

            # Handle the possible existence of multiple ghost agents: while the next agent is not PacMan, keep incrementing the agentID
            else:
              tempAgentID = agentID + 1

            # For each legal action, append the miniMax results to tempList 
            for i in gameState.getLegalActions(agentID):
              tempList.append(miniMax(gameState.generateSuccessor(agentID, i), currentDepth, tempAgentID))

            # Return the minimum value stored in the tempList list
            return min(tempList)

        # Variable to store the maximum result. Initialised to -999999 so that it can be progressively increased until its value is the true maximum
        maximum = -999999

        # Variable to store the action that will be returned as a result
        resultantAction = None

        # For each of the legal actions available to PacMan, run miniMax on the first ghost
        for i in gameState.getLegalActions(0):

          result = miniMax(gameState.generateSuccessor(0, i), 0, 1)

          # After each run of miniMax, compare the results with the maximum value, and increment the maximum value until the true maximum is reached
          if result > maximum:
            maximum = result

            # The resulting action should correspond to the maximum possible value returned by miniMax
            resultantAction = i

        return resultantAction
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Minimizer function that is called from the alphaBetaPrune function
        def minimizer(gameState, currentDepth, agentID, alpha, beta):

          # Variable to store the minimum value. Initialised to 999999 so that it can be progressively reduced until its value is the true minimum
          minimum = 999999

          # Handle the change from  ghost agents to PacMan: if the agentID is equivalent to one less than the total number of agents, it means the next agent has to be PacMan
          if agentID == gameState.getNumAgents() - 1:
            nextAgentID = 0
            # Increase the current depth by 1
            currentDepth += 1

          # Handle the possible existence of multiple ghost agents: while the next agent is not PacMan, keep incrementing the agentID
          else:
            nextAgentID = agentID + 1

          # For each of the legal actions available to the agent, calculate the minimum value by first comparing the alphaBetaPrune value with the current value stored in minimum
          for i in gameState.getLegalActions(agentID):
            
            temp = alphaBetaPrune(gameState.generateSuccessor(agentID, i), currentDepth, nextAgentID, alpha, beta)
            minimum = min(temp, minimum)

            # If the obtained minimum value is smaller than the alpha value, return the minimum value. If not, set the beta value as the smaller of the alpha and the current minimum
            if minimum < alpha:
              return minimum
            
            # Else, ensure that beta is the minimum value between alpha and the current minimum 
            beta = min(alpha, minimum)

          # Return the minimum value
          return minimum
      
        # Minimizer function that is called from the alphaBetaPrune function
        def maximizer(gameState, currentDepth, agentID, alpha, beta):
          # Variable to store the maximum value. . Initialised to -999999 so that it can be progressively increased until its value is the true maximum
          maximum = -999999

          # For each of the legal actions available to the agent, calculate the maximum value by first comparing the alphaBetaPrune value with the current value stored in maximum
          for i in gameState.getLegalActions(agentID):
            temp = alphaBetaPrune(gameState.generateSuccessor(agentID, i), currentDepth, 1, alpha, beta)
            maximum = max(temp, maximum)
            
            # If the obtained maximum value is larger than the beta value, return the maximum value. If not, set the alpha value as the larger of the beta and the current maximum
            if maximum > beta:
              return maximum
            # Else, ensure that beta is the minimum value between alpha and the current minimum
            beta = max(beta, maximum)

          # Return the maximum value
          return maximum

        # Function that calls the minimizer and maximizer functions. 
        def alphaBetaPrune(gameState, currentDepth, agentID, alpha, beta):
          
          # Base conditions for early termination of function
          if gameState.isWin() or gameState.isLose() or self.depth == currentDepth:
            return self.evaluationFunction(gameState)

          # If the agent is PacMan, run the maximizer
          if agentID == 0:
            return maximizer(gameState, currentDepth, agentID, alpha, beta)

          # Else if the agent is a ghost, run the minimizer
          else:
            return minimizer(gameState, currentDepth, agentID, alpha, beta)

        # Variable for holding a maximum value. Initialised to -999999 so that it can be progressively increased to reflect the true maximum value
        maximum = -999999
        # Variable for holding alpha. Initialised to -999999 so that it can be progressively increased to reflect the true alpha value
        alpha = -999999
        # Variable for holding beta. Initialised to 999999 so that it can be progressively decreased to reflect the true beta value
        beta = 999999
        # Variable to hold the action to be returned as a result
        resultantAction = None

        # For each legal move available to PacMan, obtain the ghostValue by running alphaBetaPrune
        for i in gameState.getLegalActions(0):
          ghostValue = alphaBetaPrune(gameState.generateSuccessor(0, i), 0, 1, alpha, beta)

          # Then, compare the ghostValue with maximum, and if the ghostValue is greater than the maximum, increment the maximum and set the resultantAction to be the one corresponding to the current legalAction
          if ghostValue > maximum:
            maximum = ghostValue

            resultantAction = i
          
          # If the maximum value is greater than beta, return maximum. If not, set alpha to be the maximum between the alpha and the maximum variables
          if maximum > beta:
            return maximum

          alpha = max(alpha, maximum)

        # Return the resultant action
        return resultantAction
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        # Function that performs expectiMax 
        def expectiMax(gameState, agentID, currentDepth):
          
          # Variable for storing the results of the expectiMax function
          tempList = []

          # Base conditions for early termination of function
          if gameState.isWin() or gameState.isLose() or self.depth == currentDepth:
            return self.evaluationFunction(gameState)

          # If the agent is PacMan, run expectMax on all of its available legal moves, and return the largest of the results generated
          if agentID == 0:
            # For each legal action, append the miniMax results to tempList
            for i in gameState.getLegalActions(agentID):
              tempList.append(expectiMax(gameState.generateSuccessor(agentID, i), currentDepth, 1))

            # Return the maximum value stored in the tempList list
            return max(tempList)
          
          # If the agent is a ghost, handle the change of agentID and currentDepth depending on how many ghosts there are
          else:
             # Handle the change from  ghost agents to PacMan: if the agentID is equivalent to one less than the total number of agents, it means the next agent has to be PacMan. currentDepth should also be incremented by 1
            if agentID == gameState.getNumAgents() - 1:
              nextAgentID = 0
              currentDepth += 1

            # Handle the possible existence of multiple ghost agents: while the next agent is not PacMan, keep incrementing the agentID
            else:
              nextAgentID = agentID + 1

            # For each legal action, append the expectiMax results to tempList
            for i in gameState.getLegalActions(agentID):
              tempList.append(expectiMax(gameState.generateSuccessor(agentID, i), nextAgentID, currentDepth))
            
            # Return the sim of all values stored in the tempList list divided by number of legal actions available to the current agent
            return sum(tempList) / len(gameState.getLegalActions(agentID))
          
        # Variable for holding a maximum value. Initialised to -999999 so that it can be progressively increased to reflect the true maximum value
        maximum = -999999
        # Variable to hold the action to be returned as a result
        resultantAction = None

        # For each legal action available to PacMan, run expectiMax and compare the result with maximum. If the result is greater than maximum, set maximum to result, and save the corresponding legal action in resultingAction
        for i in gameState.getLegalActions(0):
          
          result = expectiMax(gameState.generateSuccessor(0, i), 1, 0)

          if result > maximum:
            maximum = result
            resultantAction = i

        # Return the resulting action
        return resultantAction

        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
        """
          Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
          evaluation function (question 5).

          DESCRIPTION: <write something here so we know what you did>
        """
        "*** YOUR CODE HERE ***"
        
        # Variables duplicated from question 1
        newPos = currentGameState.getPacmanPosition()
        newFood = currentGameState.getFood()
        
        # Variable to store the distance to the closest food unit
        closestFood = 999999

        # Variable to store the sum of ghost distances
        ghostDistances = 1

        # Variable that stores the number of capsules available for consumption
        capsuleCount = len(currentGameState.getCapsules())

        # Run through the list of food coordinates, and select the closest one to the current position of PacMan
        for i in newFood.asList():
          # Calculate the Manhattan Distance between the current position and the food coordinate
          i = manhattanDistance(newPos, i)
          #  Compare i with the current value of closestFood, and assign the smaller of the two to closestFood
          closestFood = min(closestFood, i)

        # Run through the current positions of the ghosts
        for i in currentGameState.getGhostPositions():
          # Calculate the Manhattan Distance between the current position and the ghost coordinates, and accumulate those values into ghostDistances
          i = manhattanDistance(newPos, i)
          # Accumulate the distances for all ghost coordinates
          ghostDistances += i
        
        # The score modifier is calculated by getting the inverse of the closestFood value, and subtracting the inverse of the ghostDistances, the number of agents in play, and the number of available capsules.
        scoreModifier = float(1 / closestFood) - float(1 / ghostDistances) - (currentGameState.getNumAgents() - 1) - capsuleCount

        # Return the sum of getScore() and the scoreModifier
        return currentGameState.getScore() + scoreModifier

# Abbreviation
better = betterEvaluationFunction
