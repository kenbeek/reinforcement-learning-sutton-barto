import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# local imports
from rl_code.constants import output_dir

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
STEPS = 20000
AVERAGE_OVER = 2000
REWARD_VARIANCE = 1
RANDOM_WALK_STANDARD_DEVIATION = 0.01
ALPHA = 0.1
EPSILON = 0.1
NUMBER_OF_BANDITS = 10

result_array_stationary = np.zeros(STEPS)
result_array_dynamic = np.zeros(STEPS)

for k in tqdm(range(AVERAGE_OVER)):

    # initialize reward means to zero
    reward_means = np.ones(NUMBER_OF_BANDITS)
    # initialize estimates Q(a) to zero
    sample_averages_stationary = np.zeros(NUMBER_OF_BANDITS)
    sample_averages_dynamic = np.zeros(
        NUMBER_OF_BANDITS
    )  # this isn't actually a sample average anymore
    # initialize action counts N(a) to zero
    action_counts_stationary = np.zeros(NUMBER_OF_BANDITS, dtype=int)
    action_counts_dynamic = np.zeros(NUMBER_OF_BANDITS, dtype=int)
    # store all rewards (for plotting)
    rewards_stationary = []
    rewards_dynamic = []

    for i in tqdm(range(STEPS), leave=False):

        # determine if this is a greedy step or an epsilon step
        pick_at_random = np.random.default_rng().uniform(0, 1) > 1 - EPSILON
        # pick an action (either greedy or random)

        # STATIONARY
        # epsilon case
        if pick_at_random:
            action_stationary = np.random.randint(NUMBER_OF_BANDITS)
        # greedy case
        else:
            # check if there is a unique maximum in the sample averages
            # if so, simply use that maximum
            # if not, select randomly from the joint maxima
            if sum(sample_averages_stationary == max(sample_averages_stationary)) == 1:
                action_stationary = np.argmax(sample_averages_stationary)
            else:
                possible_actions = np.where(
                    sample_averages_stationary == max(sample_averages_stationary)
                )[0]
                action_stationary = np.random.choice(possible_actions)

        # get the reward for the chosen action
        action_mean_stationary = reward_means[action_stationary]
        reward_stationary = np.random.default_rng().normal(
            action_mean_stationary, REWARD_VARIANCE
        )
        # update sample average and action count
        action_counts_stationary[action_stationary] += 1
        sample_averages_stationary[action_stationary] = sample_averages_stationary[
            action_stationary
        ] + (1 / action_counts_stationary[action_stationary]) * (
            reward_stationary - sample_averages_stationary[action_stationary]
        )

        # DYNAMIC
        # epsilon case
        if pick_at_random:
            action_dynamic = np.random.randint(NUMBER_OF_BANDITS)

        # greedy case
        else:
            if sum(sample_averages_dynamic == max(sample_averages_dynamic)) == 1:
                action_dynamic = np.argmax(sample_averages_dynamic)
            else:
                possible_actions = np.where(
                    sample_averages_dynamic == max(sample_averages_dynamic)
                )[0]
                action_dynamic = np.random.choice(possible_actions)

        # get the reward for the chose action
        action_mean_dynamic = reward_means[action_dynamic]
        reward_dynamic = np.random.default_rng().normal(
            action_mean_dynamic, REWARD_VARIANCE
        )
        # update the estimate and the action count
        action_counts_dynamic[action_dynamic] += 1
        sample_averages_dynamic[action_dynamic] = sample_averages_dynamic[
            action_dynamic
        ] + ALPHA * (reward_dynamic - sample_averages_dynamic[action_dynamic])

        # update reward means with random walk
        reward_means += np.random.default_rng().normal(
            loc=0, scale=RANDOM_WALK_STANDARD_DEVIATION, size=NUMBER_OF_BANDITS
        )
        rewards_stationary.append(reward_stationary)
        rewards_dynamic.append(reward_dynamic)

    # compute the average over the rewards
    result_array_stationary = result_array_stationary + (1 / (k + 1)) * (
        rewards_stationary - result_array_stationary
    )
    result_array_dynamic = result_array_dynamic + (1 / (k + 1)) * (
        rewards_dynamic - result_array_dynamic
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
