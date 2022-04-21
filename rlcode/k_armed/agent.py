import numpy as np
from rlcode.utils import pick_softmax_action
from types import NoneType


class Agent:
    """Base class for k-armed bandit agents. Always picks greedy solution
    This class is not meant to be used directly. Use one of its extensions
    instead.
    """

    def __init__(
        self,
        estimates=None,
        number_of_bandits=10,
        alpha=0.1,
    ):
        if type(estimates) == NoneType:
            self.estimates = np.zeros(number_of_bandits)
        else:
            self.estimates = estimates
        # self.estimates = estimates or np.zeros(number_of_bandits)
        self.number_of_bandits = number_of_bandits
        self.action_counts = np.zeros(number_of_bandits, dtype=int)
        self.pick_at_random = False
        self.alpha = alpha

    def pick_greedy_action(self):
        """picks one action from the possible actions. Either randomly
        (epsilon choice) or whichever is estimated to be best (greedy choice)

        Returns:
            int: number of chosen action
        """
        # if there is a unique best choice, pick that one
        if sum(self.estimates == max(self.estimates)) == 1:
            action = np.argmax(self.estimates)
        else:
            # if there are multiple best choices, pick one of those at random
            possible_actions = np.where(self.estimates == max(self.estimates))[0]
            action = np.random.choice(possible_actions)

        return action

    def update_estimate(self, action, reward):
        """update the reward estimate based on the given reward from the bandit

        Args:
            action (int): action number
            reward (float): reward from bandit
        """
        # update action count for action
        # since these should always be performed together, I put them in
        # the same method
        self.action_counts[action] += 1
        # update sample average
        self.estimates[action] += (1 / self.action_counts[action]) * (
            reward - self.estimates[action]
        )


class EpsilonGreedyAgent(Agent):
    def __init__(self, estimates=None, number_of_bandits=10, epsilon=0.1, alpha=0.1):
        super().__init__(estimates, number_of_bandits, alpha)
        self.epsilon = epsilon

    def roll_epsilon(self):
        """whether or not to pick a random choice"""
        return np.random.default_rng().uniform(0, 1) > 1 - self.epsilon

    def pick_action(self):
        # determine if we are going for epsilon choice this time
        pick_at_random = self.roll_epsilon()
        # epsilon case
        if pick_at_random:
            action = np.random.randint(self.number_of_bandits)
        # greedy case
        else:
            action = self.pick_greedy_action()

        return action


class MovingAverageAgent(Agent):
    def __init__(self, estimates=None, number_of_bandits=10, epsilon=0.1, alpha=0.1):
        super().__init__(estimates, number_of_bandits, alpha)
        self.epsilon = epsilon
        self.alpha = alpha

    def roll_epsilon(self):
        """whether or not to pick a random choice"""
        return np.random.default_rng().uniform(0, 1) > 1 - self.epsilon

    def pick_action(self):
        # determine if we are going for epsilon choice this time
        pick_at_random = self.roll_epsilon()
        # epsilon case
        if pick_at_random:
            action = np.random.randint(self.number_of_bandits)
        # greedy case
        else:
            action = self.pick_greedy_action()

        return action

    def update_estimate(self, action, reward):
        # overwrite the super.update_estimate with a moving average function.
        self.estimates[action] += self.alpha * (reward - self.estimates[action])


class UpperConfidenceBoundAgent(Agent):
    def __init__(self, estimates=None, number_of_bandits=10, alpha=0.1, c=1):
        super().__init__(estimates, number_of_bandits, alpha)
        self.c = c

    def pick_action(self):
        # if any of the actions haven't been tried yet, pick one of the
        # untried actions
        if 0 in self.action_counts:
            possible_actions = np.where(self.action_counts == 0)[0]
            action = np.random.choice(possible_actions)
        else:
            t = sum(self.action_counts)  # total number of timesteps
            values = self.estimates + self.c * np.sqrt(np.log(t) / self.action_counts)
            action = np.argmax(values)
        return action


class GradientAgent(Agent):
    def __init__(
        self,
        estimates=None,
        number_of_bandits=10,
        alpha=0.1,
        preferences=None,
        step_size=0.1,
    ):
        super().__init__(estimates, number_of_bandits, alpha)
        if preferences is not None:
            assert len(preferences) == number_of_bandits
            self.preferences = preferences
        else:
            self.preferences = np.zeros(number_of_bandits)
        self.step_size = step_size

    def update_preference(self, action, reward):
        # probabilities of picking each action
        pi = np.exp(self.preferences) / sum(np.exp(self.preferences))
        for a in range(self.number_of_bandits):
            if a == action:
                self.preferences[action] = self.preferences[action] - self.step_size * (
                    reward - self.estimates[action]
                ) * (1 - pi[a])
            else:
                self.preferences[action] = (
                    self.preferences[action]
                    - self.step_size * (reward - self.estimates[action]) * pi[a]
                )

    def pick_action(self):
        # pick an action according to the softmax probabiliteis
        action = pick_softmax_action(self.preferences)
        return action
