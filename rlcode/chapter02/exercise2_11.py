from rlcode.k_armed.agent import EpsilonGreedyAgent, UpperConfidenceBoundAgent
from rlcode.k_armed.bandit import Bandit

"""Make a figure analogous to Figure 2.6 for the nonstationary case 
outlined in exercise 2.5. Include the constant step-size epsilon-
greedy algorithm with alpha = 0.1. Use runs of 200,000 steps and, as
a performance measure for each algorithm and parameter setting, use 
the reward over the last 100,000 steps.
"""

# test parameters
EPSILONS = [2 ** (-x) for x in range(2, 8)]
ALPHAS = [2 ** (-x) for x in range(-1, 6)]
CS = [2 ** (-x) for x in range(-2, 5)]
Q_ZEROS = [2 ** (-x) for x in range(-2, 3)]

agent_parameter_mapping = {
    "epsilon-greedy": EPSILONS,
    "gradient bandit": ALPHAS,
    "UCB": CS,
    "optimistic-greedy": Q_ZEROS,
}

# initiate one bandit
# initiate four agents
# let them run for 200.000 steps
# track the average of the last 100.000

bandit = Bandit(number_of_bandits=10, initial_state=None, stddev=1, drift=0.1)
