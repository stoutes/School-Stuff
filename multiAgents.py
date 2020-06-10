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
		
        if action == 'Stop':
            return 0
		
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        pacmanPos = newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        foodList = newFood.asList()
        #newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        sumDistance = 0
        foodFactor = 1
        isGhostNear = 0
        result = 0
        ghostDistance = []
		
        for ghost in newGhostStates:
            ghostPosition = ghost.getPosition()
            if ghost.scaredTimer == 0 and abs(pacmanPos[0] - ghostPosition[0]) <= 3 and abs(pacmanPos[1] - ghostPosition[1]) <= 3:
                isGhostNear = True
                distance = util.manhattanDistance(ghostPosition, pacmanPos)
                ghostDistance.append(distance)
				
        if isGhostNear:
		    result = min(ghostDistance)
        else:
            if len(foodList) > 0:
                distance, closestFood = min([(manhattanDistance(newPos, food), food) for food in foodList])
                if not distance == 0:
				    result += (1.0/distance)
                else:
                    result += 10
					
		return result
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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
        """
        "*** YOUR CODE HERE ***"
		
        def maxvalue(gameState, depth, numghosts):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            temp = -(float("inf"))
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                temp = max(temp, minvalue(gameState.generateSuccessor(0, action), depth - 1, 1, numghosts))
            return temp
    
        def minvalue(gameState, depth, agentindex, numghosts):
            "numghosts = len(gameState.getGhostStates())"
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            temp = float("inf")
            legalActions = gameState.getLegalActions(agentindex)
            if agentindex == numghosts:
                for action in legalActions:
                    temp = min(temp, maxvalue(gameState.generateSuccessor(agentindex, action), depth - 1, numghosts))
            else:
                for action in legalActions:
                    temp = min(temp, minvalue(gameState.generateSuccessor(agentindex, action), depth, agentindex + 1, numghosts))
            return temp
        legalActions = gameState.getLegalActions()
        numghosts = gameState.getNumAgents() - 1
        bestaction = Directions.STOP
        score = -(float("inf"))
        for action in legalActions:
            nextState = gameState.generateSuccessor(0, action)
            prevscore = score
            score = max(score, minvalue(nextState, self.depth, 1, numghosts))
            if score > prevscore:
                bestaction = action
        return bestaction
            
            
            
            
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        
        def maxvalue(gameState,alpha,beta,depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            temp = -(float("inf"))
            legalActions = gameState.getLegalActions(0)
            for action in legalActions:
                nextState = gameState.generateSuccessor(0, action)
                temp = max(temp, minvalue(nextState,alpha,beta,gameState.getNumAgents() -1,depth))
                if temp >= beta:
                    return beta
                alpha = max(alpha, temp)
            return temp
        
        def minvalue(gameState,alpha,beta,agentindex,depth):
            numghosts = gameState.getNumAgents() - 1
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            temp = (float("inf"))
            legalActions = gameState.getLegalActions(agentindex)
            for action in legalActions:
                nextState = gameState.generateSuccessor(agentindex, action)
                if agentindex == numghosts:
                    temp = min(temp, maxvalue(nextState, alpha, beta, depth - 1))
                    if temp <= alpha:
                        return temp
                    beta = min(beta, temp)
                
                else:
                    temp = min(temp, minvalue(nextState, alpha, beta, agentindex + 1, depth))
                    if temp <= alpha:
                        return temp
                    beta = min(beta, temp)
            return temp
        
        legalActions = gameState.getLegalActions(0)
        bestAction = Directions.STOP
        score = -(float("inf"))
        alpha = -(float("inf"))
        beta = float("inf")
    
        for action in legalActions:
            nextState = gameState.generateSuccessor(0,action)
            prevScore = score
            score = max(score, minvalue(nextState, alpha, beta, 1, self.depth))
        
            if score > prevScore:
                bestAction = action
            if score >= beta:
                return bestAction
            alpha = max(alpha,score)
        
        return bestAction
    
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
        return self.getActionHelper(gameState, self.depth, 0)[1]
        util.raiseNotDefined()
    
    def getActionHelper(self, gameState, depth, agentIndex):
        if depth == 0 or gameState.isWin() or gameState.isLose():
            eval_result = self.evaluationFunction(gameState)
            return (eval_result, '')
        else:
            if agentIndex == gameState.getNumAgents() - 1:
                depth -= 1
            if agentIndex == 0:
                maxAlpha = -999999999
            else:
                maxAlpha = 0
            maxAction = ''
            
            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            actions = gameState.getLegalActions(agentIndex)
            for action in actions:
                    result = self.getActionHelper(gameState.generateSuccessor(agentIndex, action), depth, nextAgentIndex)
                    if agentIndex == 0:
                        if result[0] > maxAlpha:
                            maxAlpha = result[0]
                            maxAction = action
                    else:
                        maxAlpha += 1.0/len(actions) * result[0]
                        maxAction = action
            return (maxAlpha, maxAction)
            
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    if currentGameState.isWin():
      return float("inf")
    if currentGameState.isLose():
      return -float("inf")
    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    foodPos = newFood.asList()
    closestfood = float("inf")
    for pos in foodPos:
      thisDistance = util.manhattanDistance(pos, currentGameState.getPacmanPosition())
      if (thisDistance < closestfood):
          closestfood = thisDistance
    numghosts = currentGameState.getNumAgents() - 1
    i = 1
    disttoghost = float("inf")
    while i <= numghosts:
      nextDistance = util.manhattanDistance(currentGameState.getPacmanPosition(), currentGameState.getGhostPosition(i))
      disttoghost = min(disttoghost, nextDistance)
      i += 1
    score += max(disttoghost, 4) * 2
    score -= closestfood * 1.5
    capsulelocations = currentGameState.getCapsules()
    score -= 4 * len(foodPos)
    score -= 3.5 * len(capsulelocations)
    return score

# Abbreviation
better = betterEvaluationFunction

