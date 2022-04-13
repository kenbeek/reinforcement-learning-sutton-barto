import numpy as np


class Agent:
    def __init__(
        self,
        estimates=None,
        number_of_bandits=10,
        estimator="sample_average",
        epsilon=0.1,
        alpha=0.1,
    ):
        assert estimator in ["sample_average", "weighted_exponential"]
        self.estimates = estimates or np.zeros(number_of_bandits)
        self.number_of_bandits = number_of_bandits
        self.estimator = estimator
        self.epsilon = epsilon
        self.action_counts = np.zeros(number_of_bandits, dtype=int)
        self.pick_at_random = False
        self.alpha = alpha

    def pick_action(self):
        """picks one action from the possible actions. Either randomly
        (epsilon choice) or whichever is estimated to be best (greedy choice)

        Returns:
            int: number of chosen action
        """
        # determine if we are going for epsilon choice this time
        self.pick_at_random = self.roll_epsilon()
        # epsilon case
        if self.pick_at_random:
            # pick random action
            action = np.random.randint(self.number_of_bandits)
        # greedy case
        else:
            # if there is a unique best choice, pick that one
            if sum(self.estimates == max(self.estimates)) == 1:
                action = np.argmax(self.estimates)
            else:
                # if there are multiple best choices, pick one of those at random
                possible_actions = np.where(self.estimates == max(self.estimates))
                action = np.random.choice(possible_actions)

        return action

    def roll_epsilon(self):
        """whether or not to pick a random choice"""
        self.pick_at_random = np.random.default_rng().uniform(0, 1) > 1 - self.epsilon

    def update_estimate(self, action, reward):
        """update the reward estimate based on the given reward from the bandit

        Args:
            action (int): action number
            reward (float): reward from bandit
        """
        # check type of estimator
        if self.estimator == "sample_average":
            # update sample average
            self.estimates[action] += (
                1 / self.action_counts[action]
            ) * reward - self.estimates[action]
        if self.estimator == "weighted_exponential":
            # update exponentially decaying weighted average
            self.estimates[action] += self.alpha * (reward - self.estimates[action])
        # in both cases, update action count for action
        # since these should always be performed together, I put them in the same
        self.action_counts[action] += 1
