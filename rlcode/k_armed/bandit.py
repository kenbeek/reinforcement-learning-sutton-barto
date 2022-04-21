import numpy as np


class Bandit:
    def __init__(
        self, number_of_bandits: int, initial_state=None, stddev=1, drift=None
    ):
        # Check if an initial state is provided. If not, set to zeros
        if initial_state is not None:
            assert len(initial_state) == number_of_bandits
            if type(initial_state) == np.ndarray:
                self.reward_means = initial_state
            else:
                self.reward_means = np.array(initial_state)
        else:
            self.reward_means = np.zeros(number_of_bandits)
        self.number_of_bandits = number_of_bandits
        self.stddev = stddev
        self.drift = drift

    def provide_reward(self, action):
        # provide a reward fpr a given actiom
        mean = self.reward_means[action]
        reward = np.random.default_rng().normal(loc=mean, scale=self.stddev)
        return reward

    def perform_drift(self):
        # if drift is not set to none, randomly increase or decrease the reward means
        if self.drift is not None:
            self.reward_means += np.random.default_rng().normal(
                loc=0, scale=self.drift, size=self.number_of_bandits
            )
