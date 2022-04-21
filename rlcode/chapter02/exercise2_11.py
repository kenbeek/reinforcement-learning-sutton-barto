import numpy as np
import pandas as pd
from rlcode.k_armed.agent import (
    EpsilonGreedyAgent,
    GradientAgent,
    UpperConfidenceBoundAgent,
)
from rlcode.k_armed.bandit import Bandit
from tqdm import tqdm

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
STEPS = 2000  # should be 200k. currently less for debugging purposes

# agent_parameter_mapping = {
#     "epsilon-greedy": EPSILONS,
#     "gradient bandit": ALPHAS,
#     "UCB": CS,
#     "optimistic-greedy": Q_ZEROS,
# }


# initiate agents and bandits
# let them run for 200.000 steps
# track the average of the last 100.000
print(STEPS)

result_frame = pd.DataFrame(columns=["parameter", "parameter_value", "reward"])

for epsilon in tqdm(EPSILONS):
    agent = EpsilonGreedyAgent(epsilon=epsilon)
    bandit = Bandit(number_of_bandits=10, initial_state=None, stddev=1, drift=0.1)
    reward_tracker = 0
    for step in tqdm(range(STEPS), leave=False):
        action = agent.pick_action()
        reward = bandit.provide_reward(action=action)
        agent.update_estimate(action=action, reward=reward)
        bandit.perform_drift()

        if step > np.floor(STEPS / 2):
            reward_tracker += (1 / (step - np.floor(STEPS / 2))) * (
                reward - reward_tracker
            )
    rf = pd.DataFrame(
        {
            "parameter": ["epsilon"],
            "parameter_value": [epsilon],
            "reward": [reward_tracker],
        }
    )
    result_frame = result_frame.append([rf])
    # print(f"epsilon: {epsilon}, reward: {reward_tracker}")


for alpha in tqdm(ALPHAS):
    agent = GradientAgent(step_size=alpha)
    bandit = Bandit(number_of_bandits=10, initial_state=None, stddev=1, drift=0.1)
    reward_tracker = 0
    for step in tqdm(range(STEPS), leave=False):
        action = agent.pick_action()
        reward = bandit.provide_reward(action=action)
        agent.update_estimate(action=action, reward=reward)
        bandit.perform_drift()

        if step > np.floor(STEPS / 2):
            reward_tracker += (1 / (step - np.floor(STEPS / 2))) * (
                reward - reward_tracker
            )
    rf = pd.DataFrame(
        {
            "parameter": ["alpha"],
            "parameter_value": [alpha],
            "reward": [reward_tracker],
        }
    )
    result_frame = result_frame.append([rf])

for c in tqdm(CS):
    agent = UpperConfidenceBoundAgent(c=c)
    bandit = Bandit(number_of_bandits=10, initial_state=None, stddev=1, drift=0.1)
    reward_tracker = 0
    for step in tqdm(range(STEPS), leave=False):
        action = agent.pick_action()
        reward = bandit.provide_reward(action=action)
        agent.update_estimate(action=action, reward=reward)
        bandit.perform_drift()

        if step > np.floor(STEPS / 2):
            reward_tracker += (1 / (step - np.floor(STEPS / 2))) * (
                reward - reward_tracker
            )
    rf = pd.DataFrame(
        {
            "parameter": ["c"],
            "parameter_value": [c],
            "reward": [reward_tracker],
        }
    )
    result_frame = result_frame.append([rf])

for q_zero in tqdm(Q_ZEROS):
    q_zero_array = np.ones(10) * q_zero
    agent = EpsilonGreedyAgent(estimates=q_zero_array, epsilon=0.1)
    bandit = Bandit(number_of_bandits=10, initial_state=None, stddev=1, drift=0.1)
    reward_tracker = 0
    for step in tqdm(range(STEPS), leave=False):
        action = agent.pick_action()
        reward = bandit.provide_reward(action=action)
        agent.update_estimate(action=action, reward=reward)
        bandit.perform_drift()

        if step > np.floor(STEPS / 2):
            reward_tracker += (1 / (step - np.floor(STEPS / 2))) * (
                reward - reward_tracker
            )
    rf = pd.DataFrame(
        {
            "parameter": ["Q_0"],
            "parameter_value": [q_zero],
            "reward": [reward_tracker],
        }
    )
    result_frame = result_frame.append([rf])

print(result_frame)
