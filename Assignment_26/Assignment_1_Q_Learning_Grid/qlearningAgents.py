# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math
import numpy as np

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
        self.q_values = util.Counter()
        """
        Pseudo code for Initialization of the Q-learning agent

        1. Call the parent class constructor with the given arguments
          - This sets up any parameters or initial configurations from the parent class

        2. Initialize the Q-values:
          - Create a data structure (like a dictionary) to store the Q-values for state-action pairs
          - Set all initial Q-values from Counter class from util file
        """
        


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"

        """
        Pseudo code for getting the values of the Q-learning agent
        we are getting the q learning agent values from state and action
        """

        return self.q_values[(state, action)]



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        """
          1. Get all legalactions based on current state
          2. If legalactions is empty then return 0.0
          3. Loop via legalactions and get Q-Learning agent values and find the maximum value from it
        """
        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
          return 0.0
        values = util.Counter()
        for action in legalActions:
            values[action] = self.getQValue(state, action)
        return values[values.argMax()] 

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        """
          1. Get all legalactions based on current state
          2. If not actions then return None
          3. Installize the max_q_value
          3. Lopp via actions and get Q-learning agent value and compare the q_value with max_q_value and if q_value is greater then replace max_q_value with q_value
             and replace best_action with the action 
          4. Return the best action
        """
        actions = self.getLegalActions(state)
        if len(actions)==0:
          return None
        best_action = None
        max_q_value = float('-inf')
        for action in actions:
            q_value = self.getQValue(state, action)
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
        return best_action

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
        """
          1. Get legal action based on current state
          2. Use flipcoin method implemented in util.py which returns True with probability p and False with probability 1-p
          3. If action is True then take random choice from legalactions and if action is False then get use getPolicy method to compute the action from Q values
        """
        legalActions = self.getLegalActions(state)
        action = util.flipCoin(self.epsilon)

        if action:
            return random.choice(legalActions)
        else:
            return self.getPolicy(state)
            
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        """
          Here we will update the Q-Learning func
          The formula for Q-learning update is
          Q(state, action) = (1 - self.alpha ) + self.alpha * ( reward + self.discount * self.getValue(nextState))
          1. First we will find the old_value of q by using self.getQValue(state, action)
          2. Then we will find the new_value of q by putting all the values in Q-Learning method
          3. Once we get the new_value of q then we will update the self.q_values[(state, action )] with the new_value we get in first 2nd step
        """
        old_value = self.getQValue(state, action)
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.discount * self.getValue(nextState))
        self.q_values[(state, action)] = new_value

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
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
