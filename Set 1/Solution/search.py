"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Frontier stored in a Stack
    frontier = util.Stack()

    # Visited states stored in a list
    visitedStates = []

    # Format of each element: (current coordinates, [path taken to get there]) 
    frontier.push((problem.getStartState(), []))

    # while there are still states to explore
    while not frontier.isEmpty():
        
        # store the current state and path in separate variables
        currentState, pathTaken = frontier.pop()

        # for skipping states that have already been visited
        if currentState in visitedStates:
            continue

        # for returning the correct path to the goal state upon discovering it
        if problem.isGoalState(currentState):
            return pathTaken

        # count the current state as "visited"
        visitedStates.append(currentState)

        # for each successor state, check whether they have already been visited. if not, add their coordinates to the frontier, and append their respective direction to the path list
        for coordinates, direction, cost in problem.getSuccessors(currentState):

            if coordinates not in visitedStates:
                
                frontier.push((coordinates, pathTaken + [direction]))


    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    "*** YOUR CODE HERE ***"

    # BFS is identical to DFS, save for the data structure used to store the frontier

    # Frontier stored in a Queue
    frontier = util.Queue()

    # Visited states stored in a list
    visitedStates = []

    # Format of each element: (current coordinates, [path taken to get there])
    frontier.push((problem.getStartState(), []))

    # while there are still states to explore
    while not frontier.isEmpty():

        # store the current state and path in separate variables
        currentState, pathTaken = frontier.pop()

        # for skipping states that have already been visited
        if currentState in visitedStates:
            continue

        # for returning the correct path to the goal state upon discovering it
        if problem.isGoalState(currentState):
            return pathTaken

        # count the current state as "visited"
        visitedStates.append(currentState)

        # for each successor state, check whether they have already been visited. if not, add their coordinates to the frontier, and append their respective direction to the path list
        for coordinates, direction, cost in problem.getSuccessors(currentState):

            if coordinates not in visitedStates:

                frontier.push((coordinates, pathTaken + [direction]))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    #UCS is similar to DFS and BFS, save for a few key differences

    # Frontier stored in a Priority Queue
    frontier = util.PriorityQueue()

    # Visited states stored in a list
    visitedStates = []

    # Format of each element: ((current coordinates, [path taken to get there]), cost)
    frontier.push((problem.getStartState(), []), 0)

    # while there are still states to explore
    while not frontier.isEmpty():

        # store the current state and path in separate variables
        currentState, pathTaken = frontier.pop()

        # for skipping states that have already been visited
        if currentState in visitedStates:
            continue

        # for returning the correct path to the goal state upon discovering it
        if problem.isGoalState(currentState):
            return pathTaken

        # count the current state as "visited"
        visitedStates.append(currentState)

        # for each successor state, check whether they have already been visited. 
       
        for coordinates, direction, cost in problem.getSuccessors(currentState):

            if coordinates not in visitedStates:
             # if not, re-calculate the cost to reach the given coordinates, and push the updated information to the frontier
                newCost =  problem.getCostOfActions(pathTaken + [direction])

                frontier.push((coordinates, pathTaken + [direction]), newCost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    "Search the node that has the lowest combined cost and heuristic first."
    "*** YOUR CODE HERE ***"

    # A* is different in that the heuristic argument provided is included in some parts

    # Frontier stored in a Priority Queue
    frontier = util.PriorityQueue()

    # Visited states stored in a list
    visitedStates = []

    # Format of each element: ((current coordinates, [path taken to get there]), heuristic function)
    frontier.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem))

    # while there are still states to explore
    while not frontier.isEmpty():

        # store the current state and path in separate variables
        currentState, pathTaken = frontier.pop()

        # for skipping states that have already been visited
        if currentState in visitedStates:
            continue

        # for returning the correct path to the goal state upon discovering it
        if problem.isGoalState(currentState):
            return pathTaken

        # count the current state as "visited"
        visitedStates.append(currentState)

        # for each successor state, check whether they have already been visited.
        for coordinates, direction, cost in problem.getSuccessors(currentState):

            if coordinates not in visitedStates:
             # if not, re-calculate the cost to reach the given coordinates, and push the updated information to the frontier. Here, unlike UCS, the heuristic function is added to the newCost variable
                newCost = problem.getCostOfActions(pathTaken + [direction]) + heuristic(coordinates, problem)

                frontier.push((coordinates, pathTaken + [direction]), newCost)

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
