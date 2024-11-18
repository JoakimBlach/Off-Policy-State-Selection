import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# Plot
def plot_results(res):
    # plt.rcParams['text.usetex'] = True
    colors = ["cornflowerblue", "olive", "orange"]

    _, ax = plt.subplots(1)

    # Labels
    labels = []
    for r in res[1:]:
        labels.append(r[1])

    # Behavior Rewards and Confidence intervals
    mean_rewards = []
    ci_rewards = []
    for (i, r) in enumerate(res):
        rewards = []
        if not r[0]:
            continue
        for data in r[0]:
            rewards.append(data["R"])
        cum_sum_rewards = np.cumsum(np.dstack(rewards)[0], axis=0)

        # Mean rewards
        mean_reward = np.mean(cum_sum_rewards, axis=1)
        mean_rewards.append(mean_reward) 

        print(mean_reward[-1])

        # CIs
        ci_05 = np.percentile(cum_sum_rewards, 5, axis=1)
        ci_95 = np.percentile(cum_sum_rewards, 95, axis=1)
        ci_rewards.append(np.vstack((ci_05, ci_95)).T)

    # Behav. Policy
    baseline_R = mean_rewards[0]

    for i, (mean_reward, ci_reward) in enumerate(zip(mean_rewards[1:], ci_rewards[1:])):
            ax.step(
                range(mean_reward.shape[0]),
                mean_reward - baseline_R, 
                label = labels[i],
                color = colors[i],
                linewidth=0.5)
            ax.fill_between(
                range(mean_reward.shape[0]),
                ci_reward[:, 0] - baseline_R,
                ci_reward[:, 1] - baseline_R,
                color = colors[i],
                alpha = .1)

    ax.axhline(0, color = "black", linewidth=0.5)
    ax.set_ylabel("Cumulative Reward")
    ax.set_xlabel("Time")
    ax.legend(loc = "upper right")

    plt.tight_layout()

    plt.show()