import numpy as np
import pandas as pd
from rlcode.k_armed.agent import (
    EpsilonGreedyAgent,
    GradientAgent,
    UpperConfidenceBoundAgent,
)
from rlcode.k_armed.bandit import Bandit
from rlcode.constants import project_dir
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
STEPS = 20000  # should be 200k. currently less for debugging purposes
RUNS = 10


# initiate one bandit and all agents
# in every step, let all agents pick a reward from the same agent
# this is necessary to reduce random differences in drift between the agents
# let them run for 200.000 steps
# track the average of the last 100.000

print(STEPS)


result_frame = pd.DataFrame(columns=["parameter", "parameter_value", "reward"])
for i in tqdm(range(RUNS)):

    bandit = Bandit(number_of_bandits=10, initial_state=None, stddev=1, drift=0.01)

    epsilon_agents = []
    gradient_agents = []
    ucb_agents = []
    optimistic_agents = []

    for epsilon in EPSILONS:
        epsilon_agents.append(EpsilonGreedyAgent(epsilon=epsilon))
    for alpha in ALPHAS:
        gradient_agents.append(GradientAgent(step_size=alpha))
    for c in CS:
        ucb_agents.append(UpperConfidenceBoundAgent(c=c))
    for q_zero in Q_ZEROS:
        q_zero_array = np.ones(10) * q_zero
        optimistic_agents.append(
            EpsilonGreedyAgent(estimates=q_zero_array, epsilon=0.1)
        )

    epsilon_reward_trackers = np.zeros(len(epsilon_agents))
    gradient_reward_trackers = np.zeros(len(gradient_agents))
    ucb_reward_trackers = np.zeros(len(ucb_agents))
    optimistic_reward_trackers = np.zeros(len(optimistic_agents))

    for step in tqdm(range(STEPS), leave=False):
        epsilon_rewards = np.empty(len(epsilon_agents))
        gradient_rewards = np.empty(len(gradient_agents))
        ucb_rewards = np.empty(len(ucb_agents))
        optimistic_rewards = np.empty(len(optimistic_agents))

        for i, agent in enumerate(epsilon_agents):
            action = agent.pick_action()
            reward = bandit.provide_reward(action=action)
            agent.update_estimate(action=action, reward=reward)
            epsilon_rewards[i] = reward

        for i, agent in enumerate(gradient_agents):
            action = agent.pick_action()
            reward = bandit.provide_reward(action=action)
            agent.update_preference(action=action, reward=reward)
            agent.update_estimate(action=action, reward=reward)
            gradient_rewards[i] = reward

        for i, agent in enumerate(ucb_agents):
            action = agent.pick_action()
            reward = bandit.provide_reward(action=action)
            agent.update_estimate(action=action, reward=reward)
            ucb_rewards[i] = reward

        for i, agent in enumerate(optimistic_agents):
            action = agent.pick_action()
            reward = bandit.provide_reward(action=action)
            agent.update_estimate(action=action, reward=reward)
            optimistic_rewards[i] = reward

        bandit.perform_drift()

        if step > np.floor(STEPS / 2):
            epsilon_reward_trackers = epsilon_reward_trackers + (
                1 / (step - np.floor(STEPS / 2))
            ) * (epsilon_rewards - epsilon_reward_trackers)
            gradient_reward_trackers = gradient_reward_trackers + (
                1 / (step - np.floor(STEPS / 2))
            ) * (gradient_rewards - gradient_reward_trackers)
            ucb_reward_trackers = ucb_reward_trackers + (
                1 / (step - np.floor(STEPS / 2))
            ) * (ucb_rewards - ucb_reward_trackers)
            optimistic_reward_trackers = optimistic_reward_trackers + (
                1 / (step - np.floor(STEPS / 2))
            ) * (optimistic_rewards - optimistic_reward_trackers)

    # print(reward_trackers)

    epsilon_result_frame = pd.DataFrame(
        {
            "parameter": "epsilon",
            "parameter_value": EPSILONS,
            "reward": epsilon_reward_trackers,
        }
    )
    gradient_result_frame = pd.DataFrame(
        {
            "parameter": "alpha",
            "parameter_value": ALPHAS,
            "reward": gradient_reward_trackers,
        }
    )
    ucb_result_frame = pd.DataFrame(
        {"parameter": "c", "parameter_value": CS, "reward": ucb_reward_trackers}
    )
    optimistic_result_frame = pd.DataFrame(
        {
            "parameter": "Q_0",
            "parameter_value": Q_ZEROS,
            "reward": optimistic_reward_trackers,
        }
    )
    result_frame = pd.concat(
        [
            result_frame,
            epsilon_result_frame,
            gradient_result_frame,
            ucb_result_frame,
            optimistic_result_frame,
        ]
    )

print(result_frame)

result_frame_save_path = project_dir.joinpath("results").joinpath(
    f"q2_11_result_{RUNS}_{STEPS}.csv"
)
result_frame.to_csv(result_frame_save_path)
