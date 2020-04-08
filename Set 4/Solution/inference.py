# inference.py
# ------------
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


import itertools
import random
import busters
import game

from util import manhattanDistance, raiseNotDefined


class DiscreteDistribution(dict):
    """
    A DiscreteDistribution models belief distributions and weight distributions
    over a finite set of discrete keys.
    """
    def __getitem__(self, key):
        self.setdefault(key, 0)
        return dict.__getitem__(self, key)

    def copy(self):
        """
        Return a copy of the distribution.
        """
        return DiscreteDistribution(dict.copy(self))

    def argMax(self):
        """
        Return the key with the highest value.
        """
        if len(self.keys()) == 0:
            return None
        all = list(self.items())
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def total(self):
        """
        Return the sum of values for all keys.
        """
        return float(sum(self.values()))

    def normalize(self):
        """
        Normalize the distribution such that the total value of all keys sums
        to 1. The ratio of values for all keys will remain the same. In the case
        where the total value of the distribution is 0, do nothing.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> dist.normalize()
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0)]
        >>> dist['e'] = 4
        >>> list(sorted(dist.items()))
        [('a', 0.2), ('b', 0.4), ('c', 0.4), ('d', 0.0), ('e', 4)]
        >>> empty = DiscreteDistribution()
        >>> empty.normalize()
        >>> empty
        {}
        """
        "*** YOUR CODE HERE ***"
        
        # Unless total is equal to 0, modify the distributions as required. If zero, return nothing
        total = self.total()

        if total == 0:
            return
        
        else:
            for i in self.keys():
                self[i] = self[i] / total

    def sample(self):
        """
        Draw a random sample from the distribution and return the key, weighted
        by the values associated with each key.

        >>> dist = DiscreteDistribution()
        >>> dist['a'] = 1
        >>> dist['b'] = 2
        >>> dist['c'] = 2
        >>> dist['d'] = 0
        >>> N = 100000.0
        >>> samples = [dist.sample() for _ in range(int(N))]
        >>> round(samples.count('a') * 1.0/N, 1)  # proportion of 'a'
        0.2
        >>> round(samples.count('b') * 1.0/N, 1)
        0.4
        >>> round(samples.count('c') * 1.0/N, 1)
        0.4
        >>> round(samples.count('d') * 1.0/N, 1)
        0.0
        """
        "*** YOUR CODE HERE ***"

        # If the total is not 1, normalize
        if self.total() != 1:
            self.normalize()
        
        # Intitalize a random choice, a list of items, a variable to store distributions and values
        randomChoice = random.random()
        items = sorted(self.items())
        distribution = []
        values = []

        # For each value and distribution in items, append it to their respective lists
        for value, dist in items:
            values.append(value)

            distribution.append(dist)

        # The temporary total value is set to the first element in the distribution list
        tempTotal = distribution[0]

        i = 0

        # While the randomly generated value is greater than the total, increment the cointer i and increment the temp total by the corresponding distribution value
        while randomChoice > tempTotal:
            i += 1
            tempTotal += distribution[i]

        # Return the i-th value
        return values[i]

        # raiseNotDefined()


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    """
    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        """
        Set the ghost agent for later access.
        """
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistributionHelper(self, gameState, pos, index, agent):
        try:
            jail = self.getJailPosition()
            gameState = self.setGhostPosition(gameState, pos, index + 1)
        except TypeError:
            jail = self.getJailPosition(index)
            gameState = self.setGhostPositions(gameState, pos)
        pacmanPosition = gameState.getPacmanPosition()
        ghostPosition = gameState.getGhostPosition(index + 1)  # The position you set
        dist = DiscreteDistribution()
        if pacmanPosition == ghostPosition:  # The ghost has been caught!
            dist[jail] = 1.0
            return dist
        pacmanSuccessorStates = game.Actions.getLegalNeighbors(pacmanPosition, \
                gameState.getWalls())  # Positions Pacman can move to
        if ghostPosition in pacmanSuccessorStates:  # Ghost could get caught
            mult = 1.0 / float(len(pacmanSuccessorStates))
            dist[jail] = mult
        else:
            mult = 0.0
        actionDist = agent.getDistribution(gameState)
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            if successorPosition in pacmanSuccessorStates:  # Ghost could get caught
                denom = float(len(actionDist))
                dist[jail] += prob * (1.0 / denom) * (1.0 - mult)
                dist[successorPosition] = prob * ((denom - 1.0) / denom) * (1.0 - mult)
            else:
                dist[successorPosition] = prob * (1.0 - mult)
        return dist

    def getPositionDistribution(self, gameState, pos, index=None, agent=None):
        """
        Return a distribution over successor positions of the ghost from the
        given gameState. You must first place the ghost in the gameState, using
        setGhostPosition below.
        """
        if index == None:
            index = self.index - 1
        if agent == None:
            agent = self.ghostAgent
        return self.getPositionDistributionHelper(gameState, pos, index, agent)

    def getObservationProb(self, noisyDistance, pacmanPosition, ghostPosition, jailPosition):
        """
        Return the probability P(noisyDistance | pacmanPosition, ghostPosition).
        """
        "*** YOUR CODE HERE ***"
        
        # If the ghost is where the jail is, and there is no noisy distance, return 1
        if ghostPosition == jailPosition:
            
            if noisyDistance == None:

                return 1
            
            # If the noisy distance is not null, return 0
            else:
                return 0
        
        # If only the noisy distance is not null, return 0
        if noisyDistance == None:
            
            return 0
        
        # Calculate the Manhattan distance, and return the observation probability
        realDistance = manhattanDistance(pacmanPosition, ghostPosition)
        return busters.getObservationProbability(noisyDistance, realDistance)

        # raiseNotDefined()

    def setGhostPosition(self, gameState, ghostPosition, index):
        """
        Set the position of the ghost for this inference module to the specified
        position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observe.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[index] = game.AgentState(conf, False)
        return gameState

    def setGhostPositions(self, gameState, ghostPositions):
        """
        Sets the position of all ghosts to the values in ghostPositions.
        """
        for index, pos in enumerate(ghostPositions):
            conf = game.Configuration(pos, game.Directions.STOP)
            gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
        return gameState

    def observe(self, gameState):
        """
        Collect the relevant noisy distance observation and pass it along.
        """
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observeUpdate(obs, gameState)

    def initialize(self, gameState):
        """
        Initialize beliefs to a uniform distribution over all legal positions.
        """
        self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.allPositions = self.legalPositions + [self.getJailPosition()]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        """
        Set the belief state to a uniform prior belief over all positions.
        """
        raise NotImplementedError

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        raise NotImplementedError

    def elapseTime(self, gameState):
        """
        Predict beliefs for the next time step from a gameState.
        """
        raise NotImplementedError

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        raise NotImplementedError


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward algorithm updates to
    compute the exact belief function at each time step.
    """
    def initializeUniformly(self, gameState):
        """
        Begin with a uniform distribution over legal ghost positions (i.e., not
        including the jail position).
        """
        self.beliefs = DiscreteDistribution()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        self.allPositions is a list of the possible ghost positions, including
        the jail position. You should only consider positions that are in
        self.allPositions.

        The update model is not entirely stationary: it may depend on Pacman's
        current position. However, this is not a problem, as Pacman's current
        position is known.
        """
        "*** YOUR CODE HERE ***"

        # Initialise variables for distances, the current position of PacMan, and the current jail position
        distances = DiscreteDistribution()

        currentPosition = gameState.getPacmanPosition()

        jailPosition = self.getJailPosition()

        # For each position, update the correcponding distance to the product of the observation probability and the corresponding belief value
        for i in self.allPositions:

            distances[i] = self.getObservationProb(observation, currentPosition, i, jailPosition) * self.beliefs[i]

        # Assign the new distances value to the beliefs
        self.beliefs = distances
        
        # Normalize the distances
        distances.normalize()
        
        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Predict beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position. However, this is not a problem, as Pacman's
        current position is known.
        """
        "*** YOUR CODE HERE ***"

        # Initialise variables for distances
        distances = DiscreteDistribution()

        # For each position, update the corresponding distances value by adding the product of new probability distribution and the old probability value
        for i in self.allPositions:

            tempPositionDistrib = self.getPositionDistribution(gameState, i)

            tempProbability = self.beliefs[i]

            for j in tempPositionDistrib.keys():

                distances[j] = distances[j] + tempPositionDistrib[j] * tempProbability

        # Update the beliefs with the new distance values
        self.beliefs = distances

        # raiseNotDefined()

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.
    """
    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initialize a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where
        a particle could be located. Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior. Use
        self.particles for the list of particles.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"

        particleCount = self.numParticles

        tempParticles = []
        
        legalPositionsCount = len(self.legalPositions)
        
        # While there are particles, if the number of particles is no more than the number of legal positions, append the corresponding legal positions to the tempParticles list
        while particleCount > 0:
            
            if particleCount <= legalPositionsCount:
                
                for index, item in enumerate(self.legalPositions):
                    if index < particleCount:

                        tempParticles.append(self.legalPositions[index])

                # Upon success, reset the particle count to 0
                particleCount = 0

            # If the particle count is greater than the number of legal positions, append all the legal positions to the tempParticlees, and subtract the number of legal positions from the particle count.
            else:

                for i in self.legalPositions:
                    tempParticles.append(i)
                
                particleCount -= legalPositionsCount


        # Update self.particles with the tempParticles list
        self.particles = tempParticles

        # raiseNotDefined()

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distance to the ghost you are
        tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"
        # Initialise a variable for distance
        distances = DiscreteDistribution()

        # For each particle, increase the corresponding distance by its observation probability
        for i in self.particles:
            distances[i] += self.getObservationProb(observation, gameState.getPacmanPosition(), i, self.getJailPosition())

        # If the particles all receive a non-zero weight, normalize the distances and update beliefs.  
        if distances.total() != 0:

            distances.normalize()
            self.beliefs = distances

            i = 0

            # For each particle, set its distance to the resampled value
            while i < self.numParticles:
                self.particles[i] = distances.sample()
                i += 1
        
        # If the total weight is zero, reinitialize the list of particles
        else:
            self.initializeUniformly(gameState)

        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        "*** YOUR CODE HERE ***"
        
        # List to store particles
        tempParticles = []

        # For each particle, if it is in tempParticles, resample the corresponding tempSample value and update it
        for i in range(self.numParticles):
            particle = self.particles[i]

            if particle in tempParticles:
                self.particles[i] = tempParticles[particle].sample()

            # If not, find the particle in tempParticles, and append the position distribution to it
            else:
                tempPositionDistrib = self.getPositionDistribution(gameState, particle)

                for j in tempParticles:
                    if tempParticles[j] == particle:
                        tempParticles[j] = tempPositionDistrib

                self.particles[i] = tempPositionDistrib.sample()

        # raiseNotDefined()

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution.
        
        This function should return a normalized distribution.
        """
        "*** YOUR CODE HERE ***"

        # Initialise a variable for distances
        distances = DiscreteDistribution()

        # For each particle, increase its respective distance by 1
        for i in self.particles:
            distances[i] += 1

        # Normalize and return the updated distances
        distances.normalize()

        return distances

        # raiseNotDefined()


class JointParticleFilter(ParticleFilter):
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """
    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def initialize(self, gameState, legalPositions):
        """
        Store information about the game, then initialize particles.
        """
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeUniformly(gameState)

    def initializeUniformly(self, gameState):
        """
        Initialize particles to be consistent with a uniform prior. Particles
        should be evenly distributed across positions in order to ensure a
        uniform prior.
        """
        self.particles = []
        "*** YOUR CODE HERE ***"
        # Create a list of permutations, and shuffle them
        permutations = list(itertools.product(self.legalPositions, repeat=self.numGhosts))
        random.shuffle(permutations)

        particleCount = self.numParticles
        
        # While the number of particles exceeds the number of permutations, append the permutations to the particles list, and reduce the number of particles by the number of permutations appended
        while particleCount > len(permutations):

            self.particles.extend(permutations)

            particleCount -= len(permutations)
        
        # Append the permutations up to the particleCount to the particles list
        self.particles.extend(permutations[:particleCount])
        
        # raiseNotDefined()

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observe(self, gameState):
        """
        Resample the set of particles using the likelihood of the noisy
        observations.
        """
        observation = gameState.getNoisyGhostDistances()
        self.observeUpdate(observation, gameState)

    def observeUpdate(self, observation, gameState):
        """
        Update beliefs based on the distance observation and Pacman's position.

        The observation is the noisy Manhattan distances to all ghosts you
        are tracking.

        There is one special case that a correct implementation must handle.
        When all particles receive zero weight, the list of particles should
        be reinitialized by calling initializeUniformly. The total method of
        the DiscreteDistribution may be useful.
        """
        "*** YOUR CODE HERE ***"        
        distances = DiscreteDistribution()
        
        # For each particle, get all the ghosts, and multiply their probabilities by the observation probability. Then, increment the corresponding distances element by the calculated probability
        for i in self.particles:

            probability = 1
            
            for j in range(self.numGhosts):
                
                probability *= self.getObservationProb(observation[j], gameState.getPacmanPosition(), i[j], self.getJailPosition(j))
            
            distances[i] += probability

        # Replace the beliefs with the calculated distances
        self.beliefs = distances

        # If the total of all beliefs is not 0, normalize the beliefs, and re-sample each particle
        if self.beliefs.total() != 0:
            self.beliefs.normalize()

            for i in range(self.numParticles):
                self.particles[i] = self.beliefs.sample()

        # if the total is 0, reinitialize the game state
        else:
            self.initializeUniformly(gameState)

        # raiseNotDefined()

    def elapseTime(self, gameState):
        """
        Sample each particle's next state based on its current state and the
        gameState.
        """
        newParticles = []
        
        # Dictionary to store temporary particles
        tempParticles = {}

        for oldParticle in self.particles:
            
            newParticle = list(oldParticle)  # A list of ghost positions

            # now loop through and update each entry in newParticle...
            "*** YOUR CODE HERE ***"
            
            # List to store ghost positions
            tempGhostPositions = list(oldParticle)
            
            # For each ghost, if the particle's previous position was not near the ghost, update the positions. 
            for i in range(self.numGhosts):

                if (oldParticle, i) not in tempParticles:

                    tempPositionDistrib = self.getPositionDistribution(gameState, tempGhostPositions, i, self.ghostAgents[i])

                    tempParticles[(oldParticle, i)] = tempPositionDistrib

                    newParticle[i] = tempPositionDistrib.sample()

                # If near the ghost, re-sample the corresponding value in tempParticles
                else:
                    newParticle[i] = tempParticles[(oldParticle, i)].sample()
                
            # raiseNotDefined()

            """*** END YOUR CODE HERE ***"""
            newParticles.append(tuple(newParticle))
        self.particles = newParticles


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """
    def initializeUniformly(self, gameState):
        """
        Set the belief state to an initial, prior value.
        """
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observe(self, gameState):
        """
        Update beliefs based on the given distance observation and gameState.
        """
        if self.index == 1:
            jointInference.observe(gameState)

    def elapseTime(self, gameState):
        """
        Predict beliefs for a time step elapsing from a gameState.
        """
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        """
        Return the marginal belief over a particular ghost by summing out the
        others.
        """
        jointDistribution = jointInference.getBeliefDistribution()
        dist = DiscreteDistribution()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist
