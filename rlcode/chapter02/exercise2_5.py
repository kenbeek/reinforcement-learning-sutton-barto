import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# local imports
from rlcode.constants import output_dir
from rlcode.bandit import Bandit
from rlcode.agent import Agent

"""
Design and conduct an experiment to demonstrate the difficulties that sample-average 
methods have for nostationary problems. Use a modified version of the 10-armed testbed
in which all the q*(a) start out equal and then take independent random walks 
(say by adding a normally distributed increment wiht mean zero and standard deviation 
0.01 to all the q*(a) on each step). Prepare plots like Figure 2.2 for an action-value 
method using sample averages, incrementally computed, and another action-value method 
using a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and 
longer runs, say of 10.000 steps.
"""


# parameters
STEPS = 10000
AVERAGE_OVER = 2000
REWARD_VARIANCE = 1
RANDOM_WALK_STANDARD_DEVIATION = 0.01
ALPHA = 0.1
EPSILON = 0.1
NUMBER_OF_BANDITS = 10

result_array_stationary = np.zeros(STEPS)
result_array_dynamic = np.zeros(STEPS)

for k in tqdm(range(AVERAGE_OVER)):

    bandit = Bandit(number_of_bandits=10, drift=0.01)
    stationary_agent = Agent(number_of_bandits=10, estimator="sample_average")
    decay_agent = Agent(number_of_bandits=10, estimator="weighted_exponential")
    # track rewards for plotting
    rewards_stationary, rewards_decay = [], []

    for i in tqdm(range(STEPS), leave=False):

        stationary_action = stationary_agent.pick_action()
        stationary_reward = bandit.provide_reward(stationary_action)
        stationary_agent.update_estimate(stationary_action, stationary_reward)

        decay_action = decay_agent.pick_action()
        decay_reward = bandit.provide_reward(decay_action)
        decay_agent.update_estimate(decay_action, decay_reward)

        bandit.perform_drift()
        # update reward means with random walk

        rewards_stationary.append(stationary_reward)
        rewards_decay.append(decay_reward)

    # compute the average over the rewards
    result_array_stationary = result_array_stationary + (1 / (k + 1)) * (
        rewards_stationary - result_array_stationary
    )
    result_array_dynamic = result_array_dynamic + (1 / (k + 1)) * (
        rewards_decay - result_array_dynamic
    )

fig, ax = plt.subplots(figsize=(10, 7))
sns.lineplot(x=[i for i in range(STEPS)], y=result_array_stationary)
sns.lineplot(x=[i for i in range(STEPS)], y=result_array_dynamic)
plt.legend(labels=["Sample average method", "Exponential Decay Method"])
plt.title(
    f"Comparison of sample average vs Exponential decay method, {STEPS} steps, averaged over {AVERAGE_OVER} runs"
)

save_file = output_dir.joinpath(f"exp_vs_samp_{STEPS}_{AVERAGE_OVER}.png")
fig.savefig(save_file)
plt.show()

# (ggplot(rewards) + aes(x=[i for i in range(len(rewards))], y=rewards) + geom_line())
