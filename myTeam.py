# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
from copy import deepcopy

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed, **kwargs):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.
    return [myDefendingAttacker(firstIndex, **kwargs), myAttackingDefender(secondIndex)]

##########
# Agents #
##########

class QLearningAgent(CaptureAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)

        if kwargs.get('numTraining') is not None:
            self.numTraining = int(kwargs['numTraining'])
            # Exploration
            self.epsilon = 0.1

            # Learning rate
            self.alpha = 0.1

            # Discount factor for future rewards
            self.discount = 0.9
        else:
            self.numTraining = 0

            # Exploration
            self.epsilon = 0.0

            # Learning rate
            self.alpha = 0.0

            # Discount factor for future rewards
            self.discount = 0.9

        self.episodeSoFar = 0

        self.weights = {
            # Greatly discourage Pacman
            'eaten-by-ghost': -9982.008997143268,
            'dead-end-ahead': -970.2881242681565,
            'stops-moving': -50.02661041131593,

            # Slightly discourage Pacman
            'dist-to-nearest-ghost': 8.049509252106429,

            # Slightly encourage Pacman
            'dist-to-nearest-pallet': -2.5753564250044234,
            'dist-to-nearest-capsule': -5.0,
            # 'dist-to-invader': 0.0
            'dist-to-best-exit-home': -3.01,
            'dist-to-best-entry-home': -1.7071951995339094,

            # Greatly encourage Pacman
            'eats-pallet': 176.90001376709728,
            'eats-capsule': 299.99998535233027,
            'returns-pallet': 469.7209898878529,

            # Attack/defend mode
            'attack-mode': 0.9733899880879124,
            'defense-mode': -1.9919053941378122

            # Bias
            # 'bias': 0

        }

    # Overriding parent's method in captureAgents.py
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.prevGameState = None
        self.prevAction = None
        self.prevActionSequence = []
        self.startEpisode()

    # Start Training episode
    def startEpisode(self):
        self.episodeRewards = 0.0
        if self.numTraining > 0 and self.episodeSoFar == 0:
            print('Starting %d Episodes of Training' % (self.numTraining))

    # End Training episode
    def endEpisode(self):
        self.episodeSoFar += 1
        if self.numTraining > 0:
            self.printStatistics()
        if self.episodeSoFar >= self.numTraining:
            # Stop Training ---> reset epsilon (exploration) and alpha (learning)
            self.epsilon = 0.0
            self.alpha = 0.0

    def chooseAction(self, gameState):
        legalActions = self.getLegalActions(gameState)
        if len(legalActions) == 0:
            return None

        loopAction = self.checkLoopAction()
        if loopAction is not None and loopAction in legalActions:
            legalActions.remove(loopAction)
            chosenAction = random.choice(legalActions)
            self.executeAction(gameState, chosenAction)
            return chosenAction

        if util.flipCoin(self.epsilon):
            chosenAction = random.choice(legalActions)  # Exploration
        else:
            chosenAction = self.getPolicy(gameState)  # Exploitation

        self.executeAction(gameState, chosenAction)

        return chosenAction

    def getLegalActions(self, gameState):
        actions = gameState.getLegalActions(self.index)
        return actions

    def executeAction(self, gameState, action):
        self.prevGameState = gameState
        self.prevAction = action
        self.prevActionSequence.append(action)

    def checkLoopAction(self):
        if len(self.prevActionSequence) < 4:
            return None
        if len(self.prevActionSequence) >= 4 \
                and self.prevActionSequence[-1] == self.prevActionSequence[-3] \
                and self.prevActionSequence[-2] == self.prevActionSequence[-4] \
                and self.prevActionSequence[-1] != self.prevActionSequence[-2] \
                and self.prevActionSequence[-3] != self.prevActionSequence[-4]:
            loopAction = self.prevActionSequence[-2]
            self.prevActionSequence = []
            return loopAction
        else:
            return None

    def getPolicy(self, gameState):
        actions = self.getLegalActions(gameState)
        maxQValue = self.getmaxQValue(gameState)
        bestActions = [action for action in actions if self.getQValue(gameState, action) == maxQValue]

        if len(bestActions) > 0:
            return random.choice(bestActions)
        elif len(actions) > 0:
            return random.choice(actions)
        else:
            return None

    def update(self, gameState, action, nextGameState, reward):
        features = self.getFeatures(gameState, action)
        for feature, value in features.items():
            delta = (reward + self.discount * self.getmaxQValue(nextGameState)) - self.getQValue(gameState, action)
            self.weights[feature] = self.weights[feature] + self.alpha * delta * value

    def updateWeights(self, gameState):
        if not self.prevGameState is None:
            reward = self.getReward(gameState)
            self.update(self.prevGameState, self.prevAction, gameState, reward)

    def getQValue(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights = self.getWeights()
        # Return dot product of the two dictionaries, in which `features` is created using util.Counter()
        return features * weights

    def getmaxQValue(self, gameState):
        actions = self.getLegalActions(gameState)
        if len(actions) > 0:
            return max([self.getQValue(gameState, action) for action in actions])
        return 0.0

    def getWeights(self):
        return self.weights

    def getFeatures(self, gameState, action):
        features = util.Counter()
        return features

    def getReward(self, gameState):
        return gameState.getScore()

    # Overriding method observationFunction() from captureAgents.py
    def observationFunction(self, gameState):
        self.updateWeights(gameState)
        return gameState.makeObservation(self.index)

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        return successor

    def getMyState(self, gameState):
        return gameState.getAgentState(self.index)

    def getOpponentsState(self, gameState):
        # Using getOpponents() method from captureAgents
        return [gameState.getAgentState(i) for i in self.getOpponents(gameState)]

    def final(self, gameState):
        """
          Called by Pacman game at the terminal state
        """
        self.updateWeights(gameState)
        self.endEpisode()

        if self.episodeSoFar == self.numTraining and self.numTraining != 0:
            mes = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (mes,'-' * len(mes)))
            print(self.getWeights())

        print('{}. Score: {}'.format(self.episodeSoFar, self.getScore(gameState)))

    def printStatistics(self):
        with open('allEpisodesWeights_old', 'a') as f1:
            print(self.weights, file=f1)
        with open('lastepisodesWeights', 'w') as f2:
            print(self.weights, file=f2)


class myAttackingAgent(QLearningAgent):
    def registerInitialState(self, gameState):
        QLearningAgent.registerInitialState(self, gameState)
        layout = deepcopy(gameState.data.layout)
        self.layoutWidth = layout.width
        self.layoutHeight = layout.height
        self.maxMazeDist = layout.width * layout.height
        self.walls = gameState.getWalls()
        self.totalTargetPallets = int(len(self.getFood(gameState).asList()) / 2)
        self.myStartPosition = gameState.getAgentState(self.index).getPosition()
        self.nearest1sGhostBelief = None

        # Red area is on the left of the layout. Position (0,0) is at the bottom-left corner of the layout.
        if self.red:
            opponentBorderX = int(self.layoutWidth/2)
            myBorderX = int(self.layoutWidth/ 2) - 1
            myInvasionStartX = opponentBorderX
            myInvasionEndX = self.layoutWidth
        else:
            opponentBorderX = int(self.layoutWidth/2) - 1
            myBorderX = int(self.layoutWidth/2)
            myInvasionStartX = 0
            myInvasionEndX = opponentBorderX

        self.doorPositions = []
        for myBorderY in range(0, self.layoutHeight):
            if not layout.isWall((myBorderX, myBorderY)):
                self.doorPositions.append((myBorderX, myBorderY))

        # List of positions that lead to dead-end situation (dead-end paths)
        self.deadEndPositions = []

        while True:
            prevLen = len(self.deadEndPositions)
            newWalls = deepcopy(layout.walls)
            for x in range(myInvasionStartX, myInvasionEndX):
                for y in range(0, self.layoutHeight):
                    if not layout.walls[x][y] and self.isBlockedByWalls((x, y), layout.walls):
                        self.deadEndPositions.append((x, y))
                        # Set the dead-end position as new wall
                        newWalls[x][y] = True
            layout.walls = newWalls
            # Stop when all dead-end positions have been appended
            if prevLen == len(self.deadEndPositions):
                break

        self.openPositions = []
        for x in range(myInvasionStartX, myInvasionEndX):
            for y in range(0, self.layoutHeight):
                if not layout.walls[x][y] and (x, y) not in self.deadEndPositions:
                    self.openPositions.append((x, y))


    #------------------- Helper Methods-------------------
    def isBlockedByWalls(self, position, walls):
        x, y = position
        wallCount = sum([walls[nx][ny] for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if
                          0 <= nx < walls.width and 0 <= ny < walls.height])
        return wallCount >= 3

    def isDeadEnd(self, pos):
        x, y = pos
        return (int(x), int(y)) in self.deadEndPositions

    def getNearestGhost(self, myNextPosition, opponents, scaredTimer):
        nearestGhost = None
        nearbyActiveGhosts = [g for g in opponents
                              if not g.isPacman and g.getPosition() is not None
                              and g.scaredTimer <= scaredTimer]
        if len(nearbyActiveGhosts) <= 0:
            return None
        else:
            minDistToGhost = 999999999
            for ghost in nearbyActiveGhosts:
                distance = self.getMazeDistance(myNextPosition, ghost.getPosition())
                if distance <= 5 and distance < minDistToGhost:
                    minDistToGhost = distance
                    nearestGhost = ghost.getPosition()
        if nearestGhost:
            return nearestGhost, minDistToGhost     # Returning a tuple
        else:
            return None

    # Get best door position.
    # The further from the opponent ghost and the closer to the nearest pallet, the better.
    def getBestDoorPosition(self, nearestGhost, pallets):
        maxDiff = -999999999
        bestDoorPosition = None
        for door in self.doorPositions:
            minDistDoorToGhost = self.getMazeDistance(door, nearestGhost) if nearestGhost else 0
            minDistDoorToPallet = min([self.getMazeDistance(door, pallet) for pallet in pallets.asList()]) \
                if len(pallets.asList()) > 0 else 0
            if minDistDoorToGhost - minDistDoorToPallet > maxDiff:
                maxDiff = minDistDoorToGhost - minDistDoorToPallet
                bestDoorPosition = door
        return bestDoorPosition

    def getBestPalletPosition(self, myNewPosition, nearestGhost, pallets):
        maxDiff = -999999999
        bestPalletPosition = None
        for pallet in pallets.asList():
            minDistGhostToPallet = self.getMazeDistance(pallet, nearestGhost) if nearestGhost else 0
            minDistToPallet = self.getMazeDistance(pallet, myNewPosition)
            if minDistGhostToPallet - minDistToPallet > maxDiff:
                maxDiff = minDistGhostToPallet - minDistToPallet
                bestPalletPosition = pallet
        return bestPalletPosition

    def getFeatures(self, gameState, action):
        myCurrentState = self.getMyState(gameState)
        myCurrentPosition = myCurrentState.getPosition()   # A tuple (x,y)
        if self.prevGameState:
            myPrevPosition = self.getMyState(self.prevGameState).getPosition()
        else:
            myPrevPosition = None

        opponents = self.getOpponentsState(gameState)       # A list of Opponent objects

        nextGameState = self.getSuccessor(gameState, action)
        myNextState = self.getMyState(nextGameState)
        myNextPosition = myNextState.getPosition()

        timeLeft = int(gameState.data.timeleft)

        pallets = self.getFood(gameState)
        capsules = self.getCapsules(gameState)

        features = util.Counter()

        #features['bias'] = 1

        if action == Directions.STOP:
            features['stops-moving'] = 1

        # Finding the nearest ghost (in our BELIEF) whose scaredTimer <= 1 and the distance between my Agent and that ghost
        # If None, use the most recently observed nearest 1s ghost position
        self.nearest1sGhostBelief, _ = self.getNearestGhost(myNextPosition, opponents, 1) or (self.nearest1sGhostBelief, None)

        #------------------ My Agent is in opponent's area after taking action ------------------
        if myNextState.isPacman:
            # Finding the nearest ghost whose scaredTimer <= 5 and the distance between my Agent and that ghost
            nearest5sGhost, minDistTo5sGhost = self.getNearestGhost(myNextPosition, opponents, 5) or (None, None)

            # Finding the ACTUAL nearest ghost whose scaredTimer <= 1 and the distance between my Agent and that ghost
            nearest1sGhost, minDistTo1sGhost = self.getNearestGhost(myNextPosition, opponents, 1) or (None, None)

            # Finding the best door position.
            # The further from the opponent ghost and the closer to the nearest pallet, the better.
            bestDoorPosition = self.getBestDoorPosition(nearest1sGhost, pallets)

            # My Agent is at home, next to the best exit before taking action
            if not myCurrentState.isPacman and self.getMazeDistance(myCurrentPosition, bestDoorPosition) <= 1:
                features['attack-mode'] = 1

            # Calculating min distance from my agent to door positions after he takes action
            minDistanceToHome = min([self.getMazeDistance(myNextPosition, p) for p in self.doorPositions])

            # There is an active ghost within a distance of 5 whose scaredTimer <= 5
            if nearest5sGhost is not None:
                features['dist-to-nearest-ghost'] = float(minDistTo5sGhost) / (self.maxMazeDist)
                if minDistTo5sGhost <= 1:
                    features['eaten-by-ghost'] = 1

                # If my agent is moving towards a dead-end since he wants to hide from the nearest ghost,
                # reverse dist-to-nearest-ghost feature so he can get out of the dead-end and move towards the ghost
                if self.isDeadEnd(myNextPosition):
                    features['dead-end-ahead'] = 1
                    features['dist-to-nearest-ghost'] = -features['dist-to-nearest-ghost']

                # If there are more than (shortestDistanceHome + 10) moves left,
                # and there is one or more capsules available,
                # encourage my agent to move towards the nearest capsule.
                # This is not a priority if there are not many moves left.
                # My agent needs to move towards the nearest capsule then come back home, and he just can sensor ghosts
                # within 5 distance, hence the threshold (shortestDistanceHome + 10).
                if timeLeft / 4 >= minDistanceToHome + 10 and len(capsules) > 0:
                    # If my Agent is in the capsule position after taking an action, that means he has eaten it
                    if myNextPosition in capsules:
                        features['eats-capsule'] = 1

                    # Finding the nearest capsule
                    minDistanceToCapsule = 999999999
                    nearestCapsule = None
                    for capsule in capsules:
                        distanceToCapsule = self.getMazeDistance(myNextPosition, capsule)
                        if distanceToCapsule < minDistanceToCapsule:
                            minDistanceToCapsule = distanceToCapsule
                            nearestCapsule = capsule
                    if nearestCapsule:
                        features['dist-to-nearest-capsule'] = float(minDistanceToCapsule) / (self.maxMazeDist)

                    # If the agent is moving toward the capsule
                    # and it is on a dead-end path
                    # and minDistGhostToCapsule > minCapsuleDistance,
                    # turn off 'dead-end-ahead' and reverse 'dist-to-nearest-ghost' features for my agent
                    # to reach the capsule and scare the ghost away.
                    if myPrevPosition is not None:
                        minDistGhostToCapsule = self.getMazeDistance(nearest5sGhost, nearestCapsule)
                        lastMinDistToCapsule = self.getMazeDistance(myPrevPosition, nearestCapsule)
                        if lastMinDistToCapsule > minDistanceToCapsule and self.isDeadEnd(nearestCapsule) \
                                and minDistGhostToCapsule > minDistanceToCapsule:
                            features['dead-end-ahead'] = 0
                            features['dist-to-nearest-ghost'] = -features['dist-to-nearest-ghost']

                # If there is no capsule available or there is not many moves left,
                # encourage my agent to go back home to get the score.
                else:
                    distToMyStartPosition = self.getMazeDistance(myNextPosition, self.myStartPosition)
                    features['dist-to-best-entry-home'] = float(distToMyStartPosition) / (self.maxMazeDist)

            # There is no dangerous ghost whose scaredTimer <= 5 nearby (within a distance of 5).
            else:
                # If there are more than (shortestDistanceHome + 10) moves left,
                # and there is one or more pallets available,
                # and my agent is not carrying all the pallets
                # and he (does not carry any pallet OR minDistanceToPallet < minDistanceToHome) after taking action
                # encourage my agent to move towards the nearest pallet.
                # Otherwise, encourage him to go back home to get the score.
                # This is not a priority if there are not many moves left.
                # My agent needs to move towards the nearest pallet then come back home, and he just can sensor ghosts
                # within 5 distance, hence the threshold (shortestDistanceHome + 10).

                if timeLeft / 4 >= minDistanceToHome + 10 and len(pallets.asList()) > 2:
                    # If my Agent is in the pallet position after taking an action, that means he has eaten it
                    x, y = myNextPosition
                    if pallets[int(x)][int(y)]:
                        features['eats-pallet'] = 1

                    # Finding the distance to the nearest pallet position
                    numberOfInvaders = len([o for o in opponents if o.isPacman])
                    scaredOpponents = True if len([o for o in opponents if o.scaredTimer > 5]) > 0 else False
                    targetPallets = []
                    if numberOfInvaders < 2 and scaredOpponents is False:
                        for pallet in pallets.asList():
                            distToNearestOpenPos = min([self.getMazeDistance(pallet, pos) for pos in self.openPositions])
                            if pallet not in self.deadEndPositions or distToNearestOpenPos < 2:
                                targetPallets.append(pallet)
                    else:
                        targetPallets = pallets.asList()

                    if len(targetPallets) == 0: targetPallets = pallets.asList()

                    distToNearestPallet = min([self.getMazeDistance(myNextPosition, pallet) for pallet in targetPallets])

                    if myCurrentState.numCarrying <= self.totalTargetPallets - 2 \
                            and (myNextState.numCarrying == 0 or distToNearestPallet < minDistanceToHome):
                        features['dist-to-nearest-pallet'] = float(distToNearestPallet) / (self.maxMazeDist)
                    else:
                        features['dist-to-best-entry-home'] = float(minDistanceToHome) / (self.maxMazeDist)

                # If there is no pallet available, encourage my agent to go back home to get the score.
                else:
                    distToMyStartPosition = self.getMazeDistance(myNextPosition, self.myStartPosition)
                    features['dist-to-best-entry-home'] = float(distToMyStartPosition) / (self.maxMazeDist)

                # My Agent can only sensor ghosts within a distance of 5, he needs to be cautious by always assuming
                # that the ghost is within a distance of 6
                if features['dist-to-nearest-ghost'] == 0.0:
                    features['dist-to-nearest-ghost'] = float(6) / (self.maxMazeDist)

        #------------------ My Agent is at home after taking action -----------------
        else:
            # self.nearest1sGhostBelief refers to the last possibly observed nearest 1s ghost position

            # Finding the best door position.
            # The further from the opponent ghost and the closer to the nearest pallet, the better.
            bestDoorPosition = self.getBestDoorPosition(self.nearest1sGhostBelief, pallets)

            # Calculate distance to best exit home position.
            features['dist-to-best-exit-home'] = float(self.getMazeDistance(myNextPosition, bestDoorPosition)) \
                                                 / (self.maxMazeDist)

            # My Agent is in opponent's area before taking action, which means he is taking the action to return home.
            if myCurrentState.isPacman:
                # My Agent is eaten by ghost after taking action
                if myNextPosition == self.myStartPosition:
                    features['eaten-by-ghost'] = 1
                # My Agent returns pallet home after taking action
                elif myCurrentState.numCarrying > 0:
                    features['returns-pallet'] = 1

                # My Agent is returning home for no good reason.
                # The weight for this feature should be negative to discourage him from doing so, as an attacker.
                if myNextPosition != self.myStartPosition and myCurrentState.numCarrying == 0 \
                        and (self.nearest1sGhostBelief is None or self.getMazeDistance(myNextPosition, self.nearest1sGhostBelief) > 2):
                    features['defense-mode'] = 1

            # Finding the current closest pacman opponent among those nearby pacman opponents, ie. distance <= 5
            # when my Agent is not scared, ie. scaredTimer <=5
            # nearbyInvaders = [o for o in opponents if o.isPacman and o.getPosition() is not None
            #                   and myCurrentState.scaredTimer <= 5]
            #
            # if len(nearbyInvaders) > 0:
            #     minDistanceToInvader = 999999999
            #     for invader in nearbyInvaders:
            #         distance = self.getMazeDistance(myCurrentPosition, invader.getPosition())
            #         if distance <= 5 and distance < minDistanceToInvader:
            #             minDistanceToInvader = distance
            #     features['dist-to-invader'] = float(minDistanceToInvader) / (self.maxMazeDist)

        # Normalize feature values
        features.divideAll(10)
        return features

    def getReward(self, gameState):
        reward = 0
        myCurrentState = self.getMyState(gameState)
        myCurrentPosition = myCurrentState.getPosition()

        myPrevState = self.getMyState(self.prevGameState)
        myPrevPosition = myPrevState.getPosition()

        prevPallets = self.getFood(self.prevGameState)
        preCapsules = self.getCapsules(self.prevGameState)

        # My Agent was killed
        if myCurrentPosition == self.myStartPosition and myPrevState.isPacman and not myCurrentState.isPacman:
            reward -= 100

        # My Agent returned pallets and got the score
        if myCurrentPosition != self.myStartPosition and myCurrentState.numCarrying == 0 and myPrevState.numCarrying > 0:
            reward += 50 * myPrevState.numCarrying

        # My Agent ate a pallet
        x, y = myCurrentPosition
        if prevPallets[int(x)][int(y)]:
            reward += 5

        # My Agent ate a capsule
        if myCurrentPosition in preCapsules:
            reward += 30

        # My Agent stops moving
        if myCurrentPosition == myPrevPosition:
            reward -= 5

        return reward

class myDefensiveAgent(QLearningAgent):
    def registerInitialState(self, gameState):
        QLearningAgent.registerInitialState(self, gameState)
        self.map = deepcopy(gameState.data.layout)
        self.layoutWidth = int(self.map.width)
        self.layoutHeight = int(self.map.height)
        self.nextPalletToBeGuarded = None
        self.nextPalletToGuard = None
        self.nextCapsuleToGuard = None
        self.weights = self.getWeights()
        self.prevGameState = None
        self.guardPosition = None

        # Red area is on the left of the layout. Position (0,0) is at the bottom-left corner of the layout.
        if self.red:
            self.myEndX = int(self.layoutWidth/ 2) - 1
            self.myStartX = 0
            myBorderX = self.myEndX
        else:
            self.myStartX = int(self.layoutWidth/2)
            self.myEndX = int(self.layoutWidth - 1)
            myBorderX = self.myStartX

        self.doorPositions = []
        for myBorderY in range(0, self.layoutHeight):
            if not self.map.isWall((myBorderX, myBorderY)):
                self.doorPositions.append((myBorderX, myBorderY))


    def chooseAction(self, gameState):
        legalActions = self.getLegalActions(gameState)
        if len(legalActions) == 0:
            return None
        if util.flipCoin(self.epsilon):
            chosenAction = random.choice(legalActions)  # Exploration
        else:
            chosenAction = self.getPolicy(gameState)  # Exploitation

        self.executeAction(gameState)
        return chosenAction

    def executeAction(self, gameState):
        self.prevGameState = gameState

    def getFeatures(self, gameState, action):
        myCurrentState = self.getMyState(gameState)
        opponents = self.getOpponentsState(gameState)
        nextGameState = self.getSuccessor(gameState, action)
        myNextState = self.getMyState(nextGameState)
        myNextPosition = myNextState.getPosition()

        features = util.Counter()

        # My Agent is at home defending
        if not myNextState.isPacman:
            features['defense-mode'] = 1

        if action == Directions.STOP:
            features['stops-moving'] = 1

        nearbyInvaders = [o for o in opponents if o.isPacman and o.getPosition() is not None]
        features['number-of-invaders'] = len(nearbyInvaders)

        capsules = self.getCapsulesYouAreDefending(gameState)
        adjacentDoorCapsules = []
        for door in self.doorPositions:
            for capsule in capsules:
                distDoorToCapsule = self.getMazeDistance(door, capsule)
                if distDoorToCapsule <= 2:
                    adjacentDoorCapsules.append(capsule)

        if len(adjacentDoorCapsules) == 0:
            minDistToDoor = 999999
            for x in range(self.myStartX, self.myEndX + 1):
                for y in range(0, self.layoutHeight):
                    if not self.map.isWall((x, y)):
                        totalDistToDoor = sum([self.getMazeDistance((x, y), door) for door in self.doorPositions])
                        if totalDistToDoor < minDistToDoor:
                            minDistToDoor = totalDistToDoor
                            self.guardPosition = (x, y)
        else:
            minDistToDoorCapsule = 999999
            for x in range(self.myStartX, self.myEndX + 1):
                for y in range(0, self.layoutHeight):
                    if not self.map.isWall((x, y)):
                        avrDistToDoor = float(
                            sum([self.getMazeDistance((x, y), door) for door in self.doorPositions]) \
                            / len(self.doorPositions))
                        avrDistToAdjCapsules = float(
                            sum([self.getMazeDistance((x, y), cap) for cap in adjacentDoorCapsules]) \
                            / len(adjacentDoorCapsules)) if adjacentDoorCapsules else 0
                        if avrDistToDoor + avrDistToAdjCapsules < minDistToDoorCapsule:
                            minDistToDoorCapsule = avrDistToDoor + avrDistToAdjCapsules
                            self.guardPosition = (x, y)

        if len(nearbyInvaders) > 0:
            self.nextPalletToBeGuarded = None
            minDistToInvader = min([self.getMazeDistance(myNextPosition, i.getPosition()) for i in nearbyInvaders])

            # If my agent is <= 3 distance away from the nearest invader and has a scaredTimer >= 1
            # discourage him from moving towards the invader.
            # Otherwise, encourage him to move towards the invader.
            if minDistToInvader <= 3 and myCurrentState.scaredTimer > 0:
                features['dist-to-nearest-invader'] = - minDistToInvader
            else:
                features['dist-to-nearest-invader'] = minDistToInvader

        else:
            if self.prevGameState:
                prevPalletsToBeGuarded = self.getFoodYouAreDefending(self.prevGameState).asList()
                curPalletsToBeGuarded = self.getFoodYouAreDefending(gameState).asList()

                stolenPallets = set(prevPalletsToBeGuarded).difference(curPalletsToBeGuarded)

                minDistToStolenPallets = 999999999
                if len(stolenPallets) > 0:
                    for p in curPalletsToBeGuarded:
                        distToStolenPallets = min([self.getMazeDistance(p, sp) for sp in stolenPallets])
                        if distToStolenPallets < minDistToStolenPallets:
                            minDistToStolenPallets = distToStolenPallets
                            self.nextPalletToBeGuarded = p

            if self.nextPalletToBeGuarded:
                features['dist-to-next-pallet'] = self.getMazeDistance(myNextPosition, self.nextPalletToBeGuarded)
            elif myNextPosition and self.guardPosition:
                features['dist-to-guard-position'] = self.getMazeDistance(myNextPosition, self.guardPosition)

        # If there is no invader nearby, encourage my agent to move towards guardPosition.

        # If my agent reverses its current direction tendency
        if action == Directions.REVERSE[myCurrentState.configuration.direction]:
            features['reverse'] = 1

        return features

    def getWeights(self):
        weights = {
            'defense-mode': 100,
            'stops-moving': -100,
            'dist-to-guard-position': -20,
            'number-of-invaders': -1000,
            'dist-to-nearest-invader': -10,
            'dist-to-next-pallet': -10,
            'reverse': -3
            }
        return weights

class myAttackingDefender(CaptureAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.attacker = myAttackingAgent(*args, **kwargs)
        self.defender = myDefensiveAgent(*args, **kwargs)

    def registerInitialState(self, gameState):
        self.attacker.registerInitialState(gameState)
        self.defender.registerInitialState(gameState)

    def chooseAction(self, gameState):
        myCurrentState = gameState.getAgentState(self.index)

        if myCurrentState.scaredTimer > 2:
            return self.attacker.chooseAction(gameState)
        else:
            return self.defender.chooseAction(gameState)

class myDefendingAttacker(CaptureAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args)
        self.attacker = myAttackingAgent(*args, **kwargs)
        self.defender = myDefensiveAgent(*args, **kwargs)

    def registerInitialState(self, gameState):
        self.attacker.registerInitialState(gameState)
        self.defender.registerInitialState(gameState)
        self.prevGameState = None
        layout = deepcopy(gameState.data.layout)
        self.layoutWidth = int(layout.width)
        self.layoutHeight = int(layout.height)

        # Red area is on the left of the layout. Position (0,0) is at the bottom-left corner of the layout.
        if self.red:
            myEndX = int(self.layoutWidth / 2) - 1
            myStartX = 0
            myBorderX = myEndX
        else:
            myStartX = int(self.layoutWidth / 2)
            myEndX = int(self.layoutWidth - 1)
            myBorderX = myStartX

        self.doorPositions = []
        for myBorderY in range(0, self.layoutHeight):
            if not layout.isWall((myBorderX, myBorderY)):
                self.doorPositions.append((myBorderX, myBorderY))

        # Finding the optimal guardPosition - the position that has the minimum average distance to door positions
        self.guardPosition = None
        self.avrDistGuardToDoor = 0
        if self.prevGameState:
            minAvrDistToDoor = 999999
            for x in range(myStartX, myEndX + 1):
                for y in range(0, self.layoutHeight):
                    if not layout.isWall((x, y)):
                        avrDistToDoor = sum([self.getMazeDistance((x, y), door) for door in self.doorPositions])\
                                          /len(self.doorPositions)
                        if avrDistToDoor < minAvrDistToDoor:
                            minAvrDistToDoor = avrDistToDoor
                            self.guardPosition = (x, y)

            self.avrDistGuardToDoor = sum([self.getMazeDistance(self.guardPosition, door) for door in self.doorPositions])\
                                      /len(self.doorPositions)

    def executeAction(self, gameState):
        self.prevGameState = gameState

    def getTeamMateIndex(self, gameState):
        team = self.getTeam(gameState)
        teamMate = team.remove(self.index)[0]
        return teamMate

    def getTeamMateState(self, gameState):
        teamMateIndex = self.getTeamMateIndex(gameState)
        teamMateState = gameState.getAgentState(teamMateIndex)
        return teamMateState

    def chooseAction(self, gameState):
        if self.prevGameState:
            movesLeft = int(gameState.data.timeleft / 4)
            # Calculating min distance from my agent to door positions after he takes action
            myCurrentState = gameState.getAgentState(self.index)
            curTeamMateState = self.getTeamMateState(gameState)
            myCurrentPosition = myCurrentState.getPosition()
            distToGuardPosition = self.getMazeDistance(self.guardPosition, myCurrentPosition)
            currentOpponents = self.getOpponentsState(gameState)
            prevOpponents = self.getOpponentsState(self.prevGameState)
            currentNumOfInvaders = [o for o in currentOpponents if o.isPacman]
            prevNumOfInvaders = [o for o in prevOpponents if o.isPacman]
            if currentNumOfInvaders > 1 and prevNumOfInvaders > 1 and curTeamMateState.scaredTime == 0 \
                    and movesLeft <= int(distToGuardPosition + self.avrDistGuardToDoor) + 3:
                return self.defender.chooseAction(gameState)
            else:
                return self.attacker.chooseAction(gameState)
        else:
            return self.attacker.chooseAction(gameState)
