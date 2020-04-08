import random
import copy
from optparse import OptionParser
import util

class SolveEightQueens:
    def __init__(self, numberOfRuns, verbose, lectureExample):
        """
        Value 1 indicates the position of queen
        """
        self.numberOfRuns = numberOfRuns
        self.verbose = verbose
        self.lectureCase = [[]]
        if lectureExample:
            self.lectureCase = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            ]
    def solve(self):
        solutionCounter = 0
        for i in range(self.numberOfRuns):
            if self.search(Board(self.lectureCase), self.verbose).getNumberOfAttacks() == 0:
                solutionCounter += 1
        print("Solved: %d/%d" % (solutionCounter, self.numberOfRuns))

    def search(self, board, verbose):
        """
        Hint: Modify the stop criterion in this function
        """

        newBoard = board

        # Initialise variable to store the number of consecutive moves 
        numberOfMoves = 0

        i = 0
        
        while True:

            if verbose:
                
                print("iteration %d" % i)
                print(newBoard.toString())
                print("# attacks: %s" % str(newBoard.getNumberOfAttacks()))
                print(newBoard.getCostBoard().toString(True))

            currentNumberOfAttacks = newBoard.getNumberOfAttacks()
            
            # If there are no possible attacks now or over 100 consecutive moves have been made, break
            if currentNumberOfAttacks == 0 or numberOfMoves >= 100:
                break
            
            (newBoard, newNumberOfAttacks, newRow, newCol) = newBoard.getBetterBoard()
            
            i += 1

            # If 30 or more iterations have occured and the current number of attacks is less than  the new number of attacks, reset the number of iterations to 0, increment the numberOfMoves by 1, and create a new Board for the newBoard variable
            if currentNumberOfAttacks <= newNumberOfAttacks and i >= 30:
                
                i = 0
                
                newBoard = Board()
                
                numberOfMoves += 1
        
        return newBoard

class Board:
    def __init__(self, squareArray = [[]]):
        if squareArray == [[]]:
            self.squareArray = self.initBoardWithRandomQueens()
        else:
            self.squareArray = squareArray

    @staticmethod
    def initBoardWithRandomQueens():
        tmpSquareArray = [[ 0 for i in range(8)] for j in range(8)]
        for i in range(8):
            tmpSquareArray[random.randint(0,7)][i] = 1
        return tmpSquareArray
          
    def toString(self, isCostBoard=False):
        """
        Transform the Array in Board or cost Board to printable string
        """
        s = ""
        for i in range(8):
            for j in range(8):
                if isCostBoard: # Cost board
                    cost = self.squareArray[i][j]
                    s = (s + "%3d" % cost) if cost < 9999 else (s + "  q")
                else: # Board
                    s = (s + ". ") if self.squareArray[i][j] == 0 else (s + "q ")
            s += "\n"
        return s 

    def getCostBoard(self):
        """
        First Initalize all the cost as 9999. 
        After filling, the position with 9999 cost indicating the position of queen.
        """
        costBoard = Board([[ 9999 for i in range(8)] for j in range(8)])
        for r in range(8):
            for c in range(8):
                if self.squareArray[r][c] == 1:
                    for rr in range(8):
                        if rr != r:
                            testboard = copy.deepcopy(self)
                            testboard.squareArray[r][c] = 0
                            testboard.squareArray[rr][c] = 1
                            costBoard.squareArray[rr][c] = testboard.getNumberOfAttacks()
        return costBoard

    def getBetterBoard(self):
        """
        "*** YOUR CODE HERE ***"
        This function should return a tuple containing containing four values
        the new Board object, the new number of attacks, 
        the Column and Row of the new queen  
        For exmaple: 
            return (betterBoard, minNumOfAttack, newRow, newCol)
        The datatype of minNumOfAttack, newRow and newCol should be int
        """

        # For selecting a random entry from the minimumCostPositions list
        import random

        # Obtain the current board
        currentBoard = self.squareArray

        # Obtain the corresponding costs
        costBoard = self.getCostBoard().squareArray

        # List to store all the positions that have the minimum cost 
        minimumCostPositions = []

        # Variable to store the minimum cost of the costBoard
        minimumCost = 10000

        # Cycle through each element of the cost board, and update minimunCost until the absolute minimum cost is found
        for indexI, i in enumerate(costBoard):
            for indexJ, j in enumerate(i):
                if min(i) <= minimumCost:
                    minimumCost = min(i)

        # Collect every position with the minimum cost and put it in the minimumCostPosition list. Include their corresponding coordinates.
        for indexI, i in enumerate(costBoard):
            for indexJ, j in enumerate(i):
                if costBoard[indexI][indexJ] == minimumCost:
                    minimumCost = costBoard[indexI][indexJ]
                    minimumCostPositions.append(((indexI, indexJ), costBoard[indexI][indexJ]))

        # # Select a random element from the minimumCostPositions list, and take its coordinates as newRow and newCol
        newRow, newCol = random.choice(minimumCostPositions)[0]

        # In the current board, ensure that the queen is placed in the coordinates corresponding to (newRow, newCol), and that all of the positions on the board on the same row and column as newRow and newColumn are set to 0 
        for i in currentBoard:
            i[newCol] = 0
        currentBoard[newRow][newCol] = 1

        # Initialise a betterBoard variable for returning. Type cast it as Board(), so that it can employ getNumberOfAttacks without any problems
        betterBoard = Board(currentBoard)
        
        return (betterBoard, betterBoard.getNumberOfAttacks(), newRow, newCol)

        util.raiseNotDefined()

    def getNumberOfAttacks(self):
        """
        "*** YOUR CODE HERE ***"
        This function should return the number of attacks of the current board
        The datatype of the return value should be int
        """

        # Store the current board in a variable
        currentBoard = self.squareArray

        # Store the number of attacks on the current board
        numberOfAttacks = 0

        # Initialise a list for storing the coordinates of the queens on the board
        queens = []

        # Iterate through the current board, find the locations of all queens, and store them in the queens list
        for indexX, x in enumerate(currentBoard):
            for indexY, y in enumerate(x):
                if y == 1:
                    queens.append((indexX, indexY))

        # For each queen in the queens list, compare it with the other queens in the row to see if there are any on the same row, or are diagonal. If so, increment numberOFAttacks by 1
        for indexX, x in enumerate(queens):
            for y in range(indexX + 1, len(queens)):
                if x[0] == queens[y][0] or abs(x[0] - queens[y][0]) == abs(x[1] - queens[y][1]):
                    numberOfAttacks += 1

        return numberOfAttacks

        util.raiseNotDefined()

if __name__ == "__main__":
    #Enable the following line to generate the same random numbers (useful for debugging)
    random.seed(1)
    parser = OptionParser()
    parser.add_option("-q", dest="verbose", action="store_false", default=True)
    parser.add_option("-l", dest="lectureExample", action="store_true", default=False)
    parser.add_option("-n", dest="numberOfRuns", default=1, type="int")
    (options, args) = parser.parse_args()
    EightQueensAgent = SolveEightQueens(verbose=options.verbose, numberOfRuns=options.numberOfRuns, lectureExample=options.lectureExample)
    EightQueensAgent.solve()
