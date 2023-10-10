import numpy as np

from main.Action import Action
from main.Maze import Maze


class QLearning:

    def __init__(self):
        # dictionary (State, <Action, Value>)
        self.q = {}

    def get_q(self, state, action):
        # checks if it can find a value for <s,a> in q and returns it, if not return 0.
        try:
            action_values = self.q[state]
            try:
                value = action_values[action]
                return value
            except KeyError:
                return 0
        except KeyError:
            return 0

    def set_q(self, state, action, value):
        # sets the value of am <s,a> pair to q
        try:
            action_values = self.q[state]
            float_value = float(value)
            action_values[action] = float_value
        except KeyError:
            # no entry known for s, make one and store the action value too
            action_values = {}
            float_value = float(value)
            action_values[action] = float_value
            self.q[state] = action_values

    def get_action_values(self, state, actions):
        # returns the associated action values for all actions in <actions> in that order;
        result = []
        for action in actions:
            result.append(self.get_q(state, action))
        return result

    def update_q(self, state, action, r, state_next, possible_actions, alfa, gamma):
        """
        q_old is the old estimate of action and state, that we will update
        q_max the best estimate of action from the new possible actions
        r - reward
        gamma - discount factor
        alpha - learning rate
        """
        q_old = self.get_q(state, action)

        # Q-values for possible_actions in the next state s':
        qChoices = self.get_action_values(state_next, possible_actions)
        q_max = qChoices[np.argmax(qChoices)]

        """
        Formula for updating q
        """
        q_new = q_old + alfa * (r + (gamma * q_max) - q_old)

        self.set_q(state, action, q_new)
        return

    def printQ(self, maze: Maze):
        for i in range(10):
            for j in range(10):
                state = maze.get_state(j, i)
                actions = [Action("up"), Action("down"), Action("left"), Action("right")]
                print("State (" + str(i) + " " + str(j) + ") ", end="")
                for a in actions:
                    print(self.get_q(state, a), end=" ")
                print()
