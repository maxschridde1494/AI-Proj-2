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
          closestFood = min(distances)
        else:
          closestFood = 0

        if len(ghostDistances) > 0:
          closestGhost = min(ghostDistances)
        else:
          closestGhost = 0

        if closestGhost == 0 or closestGhost == 1:
          return float("-inf")

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
        """
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
        
        return actionGenerator(gameState, self.depth, minimaxHelper)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def abHelper(gameState, depth, agentIndex, a, b):
          if gameState.isWin() or gameState.isLose() or depth == 0: #reached the depth specified or win / lose position
            return self.evaluationFunction(gameState)
          else:
            if agentIndex == 0: #pacman --> maximizing agent
              return maxValue(gameState, depth, agentIndex, a, b)
            return minValue(gameState, depth, agentIndex, a, b)

        def maxValue(gameState, depth, agentIndex, a, b):
          v = float('-inf')
          legalActions = gameState.getLegalActions(agentIndex)
          for action in legalActions:
            v = max(v, abHelper(gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, a, b))
            if v > b:
              return v
            a = max(a, v)
          return v

        def minValue(gameState, depth, agentIndex, a, b):
          check = False
          v = float('inf')
          legalActions = gameState.getLegalActions(agentIndex) #generate all legal actions of agent          
          if agentIndex == gameState.getNumAgents() - 1: #last ghost agent (moving to next ply / on to Pacman)
            check = True
          
          for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if check:
              v = min(v, (abHelper(successor, depth - 1, 0, a, b)))
              if v < a:
                return v
              b = min(v, b)
            else:
              v = min(v, (abHelper(successor, depth, agentIndex + 1, a, b)))
              if v < a:
                return v
              b = min(v, b)
          return v

        legalActions = gameState.getLegalActions(0)
        v, action = float("-inf"), None
        a = float("-inf")
        b = float("inf")
        for act in legalActions:
          successor = gameState.generateSuccessor(0, act)
          abVal = abHelper(successor, self.depth, 1, a, b)
          if abVal > v:
            v, action = abVal, act
          a = max(a, v)
        return action

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
              v += (1.0/float(len(legalActions)))*(expectimaxHelper(successor, depth, agentIndex + 1))
          return v

        return actionGenerator(gameState, self.depth, expectimaxHelper)

def actionGenerator (gameState, depth, valueFunc):
  successorsOut = [(gameState.generateSuccessor(0, action), action) for action in gameState.getLegalActions(0)]
  v, action = float("-inf"), None
  for successor, act in successorsOut:
    # stratVal = strategy(successor, depth, 1)
    stratVal = valueFunc(successor, depth, 1)
    if stratVal > v:
      v, action = stratVal, act
  return action

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    evaluationScore = currentGameState.getScore()
    pacPosition = currentGameState.getPacmanPosition()
    foodPositions = currentGameState.getFood().asList()
    foodDistances = [util.manhattanDistance(pacPosition, foodPellot) for foodPellot in foodPositions]
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghost.scaredTimer for ghost in ghostStates]
    ghostDistances = [util.manhattanDistance(pacPosition, ghostState.getPosition()) for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    capsuleDistances = [util.manhattanDistance(pacPosition, capsule) for capsule in capsules]
    
    closestFood = float('inf')
    if len(foodDistances) > 0:
      closestFood = min(foodDistances)
    evaluationScore += (1.0/(closestFood + 1.0))

    closestCapsule = float('inf')
    if len(capsuleDistances) > 0:
      closestCapsule = min(capsuleDistances)
    evaluationScore += (1.0 / (closestCapsule + 1.0))
    
    scaredGhostTotalTime = reduce(lambda x, y: x + y, scaredTimes, 0)
    evaluationScore += scaredGhostTotalTime

    closestGhost = float('inf')
    if len(ghostDistances) > 0:
      closestGhost = min(ghostDistances)

    if closestGhost == 0 or closestGhost == 1:
      return float('-inf')

    evaluationScore -= (1.0 / (closestGhost + 1.0))
    
    return evaluationScore

# Abbreviation
better = betterEvaluationFunction

