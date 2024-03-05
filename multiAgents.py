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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Define some constants to influence the evaluation
        FOOD_WEIGHT = 10  # Weight for food
        GHOST_WEIGHT = -100  # Weight for ghosts

        # Calculate the distance to the nearest food
        if newFood:
            min_food_dist = min([manhattanDistance(newPos, food) for food in newFood])
        else:
            min_food_dist = 1  # Set a small distance if there's no food left

        # Check if there are any ghosts nearby
        ghost_dist = min([manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates])
        if ghost_dist == 0:
            ghost_dist = 1

        # Evaluate the state based on the calculated values
        score = successorGameState.getScore()  # Start with the game score
        score += FOOD_WEIGHT / min_food_dist  # Encourage getting closer to food
        score += GHOST_WEIGHT / ghost_dist if ghost_dist < 2 else 0  # Discourage being too close to ghosts

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='10'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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

        def minimax(agentIndex, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            if agentIndex == 0:  # Pacman's turn (maximize)
                maxVal = float('-inf')
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    maxVal = max(maxVal, minimax(1, depth, successorState))
                return maxVal
            else:  # Ghost's turn (minimize)
                minVal = float('inf')
                for action in gameState.getLegalActions(agentIndex):
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    if agentIndex == gameState.getNumAgents() - 1:
                        minVal = min(minVal, minimax(0, depth - 1, successorState))
                    else:
                        minVal = min(minVal, minimax(agentIndex + 1, depth, successorState))
                return minVal

        legalMoves = gameState.getLegalActions(0)
        bestScore = float('-inf')
        bestAction = None
        for action in legalMoves:
            successorState = gameState.generateSuccessor(0, action)
            score = minimax(1, self.depth, successorState)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
            def alpha_beta_pruning(agentIndex, depth, gameState, alpha, beta):
                if gameState.isWin() or gameState.isLose() or depth == 0:
                    return self.evaluationFunction(gameState), None

                if agentIndex == 0:  # Pacman's turn (maximize)
                    maxVal = float('-inf')
                    bestAction = None
                    for action in gameState.getLegalActions(agentIndex):
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        value, _ = alpha_beta_pruning(1, depth, successorState, alpha, beta)
                        if value > maxVal:
                            maxVal = value
                            bestAction = action
                        if maxVal > beta:
                            return maxVal, bestAction
                        alpha = max(alpha, maxVal)
                    return maxVal, bestAction
                else:  # Ghost's turn (minimize)
                    minVal = float('inf')
                    bestAction = None
                    for action in gameState.getLegalActions(agentIndex):
                        successorState = gameState.generateSuccessor(agentIndex, action)
                        if agentIndex == gameState.getNumAgents() - 1:
                            value, _ = alpha_beta_pruning(0, depth - 1, successorState, alpha, beta)
                        else:
                            value, _ = alpha_beta_pruning(agentIndex + 1, depth, successorState, alpha, beta)
                        if value < minVal:
                            minVal = value
                            bestAction = action
                        if minVal < alpha:
                            return minVal, bestAction
                        beta = min(beta, minVal)
                    return minVal, bestAction

            legalMoves = gameState.getLegalActions(0)
            bestScore = float('-inf')
            bestAction = None
            alpha = float('-inf')
            beta = float('inf')

            for action in legalMoves:
                successorState = gameState.generateSuccessor(0, action)
                score, _ = alpha_beta_pruning(1, self.depth, successorState, alpha, beta)
                if score > bestScore:
                    bestScore = score
                    bestAction = action
                if bestScore > beta:
                    return bestAction
                alpha = max(alpha, bestScore)

            return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState: GameState):
        def expectimax(agentIndex, depth, game_state):
            if depth == 0 or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state), None

            if agentIndex == 0:  # Pacman's turn (maximizing)
                max_value = float("-inf")
                best_action = None
                for action in game_state.getLegalActions(agentIndex):
                    #print(game_state.getLegalActions(agentIndex))
                    successor_state = game_state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax(1, depth -1, successor_state)
                    if value >= max_value:
                        max_value = value
                        best_action = action

                return max_value, best_action
            else:  # Ghost's turn (expectation)
                total_value = 0
                num_actions = len(game_state.getLegalActions(agentIndex))
                for action in game_state.getLegalActions(agentIndex):
                    successor_state = game_state.generateSuccessor(agentIndex, action)
                    value, _ = expectimax((agentIndex + 1) % game_state.getNumAgents(), depth, successor_state)
                    total_value += value
                expected_value = total_value / num_actions
                return expected_value, None

        _, best_action = expectimax(0, self.depth, gameState)
        return best_action

def betterEvaluationFunction(currentGameState: GameState):

    util.raiseNotDefined()



# Abbreviation
better = betterEvaluationFunction
