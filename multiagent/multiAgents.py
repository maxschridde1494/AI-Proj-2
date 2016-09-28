# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distances = [util.manhattanDistance(newPos, food) for food in newFood.asList()]
        ghostDistances = [util.manhattanDistance(newPos, ghostPos.getPosition()) for ghostPos in newGhostStates]

        if len(distances) > 0:
          closestFood, furthestFood = min(distances), max(distances)
        else:
          closestFood, furthestFood = 0, 0

        if len(ghostDistances) > 0:
          closestGhost = min(ghostDistances)
        else:
          closestGhost = 0

        if closestGhost == 0 or closestGhost == 1:
          return -9999999999

        evaluationScore = 2*(1.0/(closestFood+1.0)) + successorGameState.getScore() - (1.0/(closestGhost+1.0)) 

        return evaluationScore

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

        def minimaxHelper(gameState, depth, agentIndex):
          if gameState.isWin() or gameState.isLose() or depth == 0: #reached the depth specified or win / lose position
            return self.evaluationFunction(gameState)
          else:
            if agentIndex == 0: #pacman --> maximizing agent
              return maxValue(gameState, depth, agentIndex)
            return minValue(gameState, depth, agentIndex)

        def maxValue(gameState, depth, agentIndex):
          v = float('-inf')
          legalActions = gameState.getLegalActions(agentIndex)
          successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions] #generate all possible successors
          
          for successor in successors:
            v = max(v, (minimaxHelper(successor, depth, agentIndex + 1)))
          return v

        def minValue(gameState, depth, agentIndex):
          check = False
          v = float('inf')
          legalActions = gameState.getLegalActions(agentIndex) #generate all legal actions of agent
          successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions] #generate all possible successors
          
          if agentIndex == gameState.getNumAgents() - 1: #last ghost agent (moving to next ply / on to Pacman)
            check = True
          
          for successor in successors:
            if check:
              v = min(v, (minimaxHelper(successor, depth - 1, 0)))
            else:
              v = min(v, (minimaxHelper(successor, depth, agentIndex + 1)))
          return v

        successorsOut = [(gameState.generateSuccessor(0, action), action) for action in gameState.getLegalActions(0)]
        vFinal, vFinalAction = float("-inf"), None
        for succ, act in successorsOut:
          minimaxVal = minimaxHelper(succ, self.depth, 1)
          if minimaxVal > vFinal:
            vFinal, vFinalAction = minimaxVal, act
        return vFinalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimaxHelper(gameState, depth, agentIndex, a, b):
          if gameState.isWin() or gameState.isLose() or depth == 0: #reached the depth specified or win / lose position
            return self.evaluationFunction(gameState)
          else:
            if agentIndex == 0: #pacman --> maximizing agent
              return maxValue(gameState, depth, agentIndex, a, b)
            return minValue(gameState, depth, agentIndex, a, b)

        def maxValue(gameState, depth, agentIndex, a, b):
          v = float('-inf')
          legalActions = gameState.getLegalActions(agentIndex)
          successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions] #generate all possible successors
          
          for successor in successors:
            v = max(v, (minimaxHelper(successor, depth, agentIndex + 1, a, b)))
            if v > b:
              return v
            a = max(a, v)
          return v

        def minValue(gameState, depth, agentIndex, a, b):
          check = False
          v = float('inf')
          legalActions = gameState.getLegalActions(agentIndex) #generate all legal actions of agent
          successors = [gameState.generateSuccessor(agentIndex, action) for action in legalActions] #generate all possible successors
          
          if agentIndex == gameState.getNumAgents() - 1: #last ghost agent (moving to next ply / on to Pacman)
            check = True
          
          for successor in successors:
            if check:
              v = min(v, (minimaxHelper(successor, depth - 1, 0, a, b)))
              if v < a:
                return v
              b = min(v, b)
            else:
              v = min(v, (minimaxHelper(successor, depth, agentIndex + 1, a, b)))
              if v < a:
                return v
              b = min(v, b)
          return v

        successorsOut = [(gameState.generateSuccessor(0, action), action) for action in gameState.getLegalActions(0)]
        vFinal, vFinalAction = float("-inf"), None
        for succ, act in successorsOut:
          minimaxVal = minimaxHelper(succ, self.depth, 1, float("-inf"), float("inf"))
          if minimaxVal > vFinal:
            vFinal, vFinalAction = minimaxVal, act
        return vFinalAction

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
        def expectimaxHelper(gameState, depth, agentIndex):
          if gameState.isWin() or gameState.isLose() or depth == 0: #reached the depth specified or win / lose position
            return self.evaluationFunction(gameState)
          else:
            if agentIndex == 0: #pacman --> maximizing agent
              return maxValue(gameState, depth, agentIndex)
            return expValue(gameState, depth, agentIndex)

        def maxValue(gameState, depth, agentIndex):
          v = float('-inf')
          legalActions = gameState.getLegalActions(agentIndex)
          successors = [(gameState.generateSuccessor(agentIndex, action), action) for action in legalActions] #generate all possible successors
          
          for successor, action in successors:
            v = max(v, (expectimaxHelper(successor, depth, agentIndex + 1)))
          return v

        def expValue(gameState, depth, agentIndex):
          check = False
          v = 0
          legalActions = gameState.getLegalActions(agentIndex) #generate all legal actions of agent
          successors = [(gameState.generateSuccessor(agentIndex, action), action) for action in legalActions] #generate all possible successors
          
          if agentIndex == gameState.getNumAgents() - 1: #last ghost agent (moving to next ply / on to Pacman)
            check = True
          
          for successor, action in successors:
            if check:
              v += (1.0/float(len(legalActions)))*(expectimaxHelper(successor, depth - 1, 0))
            else:
              v = (1.0/float(len(legalActions)))*(expectimaxHelper(successor, depth, agentIndex + 1))
          return v

        successorsOut = [(gameState.generateSuccessor(0, action), action) for action in gameState.getLegalActions(0)]
        vFinal, vFinalAction = float("-inf"), None
        for succ, act in successorsOut:
          expectimaxVal = expectimaxHelper(succ, self.depth, 1)
          if expectimaxVal > vFinal:
            vFinal, vFinalAction = expectimaxVal, act
        return vFinalAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

