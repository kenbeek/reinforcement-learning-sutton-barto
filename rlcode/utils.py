from datetime import datetime
import numpy as np


def print_time(input):
    """prints a string with the current time prefixed

    Args:
        input Any: output to be printed
    """
    now = datetime.now()
    print(f"{now}: {input}")


def softmax(preferences: np.array):
    probabilities = np.exp(preferences) / sum(np.exp(preferences))
    probabilities = np.cumsum(probabilities)
    return probabilities


def pick_softmax_action(preferences: np.ndarray):
    # convert preferences to probabilities using softmax
    probabilities = softmax(preferences)
    # instantiate a uniform random variable on the interval [0,1)
    uniform_rv = np.random.default_rng().uniform(0, 1)
    # find the smallest cumulative softmax preference sum that is smaller than the rv
    # This is the action that you choose
    action = np.min(np.where(uniform_rv < probabilities))
    return action
