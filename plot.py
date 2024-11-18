import pickle
import numpy as np
import matplotlib.pyplot as plt

font = {'fontname':'Helvetica'}

files = ["output/example1_specs.pkl", "output/example2_specs.pkl", "output/example3_specs.pkl", "output/example4_specs.pkl"]
titles = ["(a)", "(b)", "(c)", "(d)"]
# files = ["output/pricing_confounded_specs_output.pkl"]

fig, ax = plt.subplots(1, len(files), figsize=(12, 3))

for k, filename in enumerate(files):
    with open(filename, 'rb') as file:
        res = pickle.load(file)

    colors = ["cornflowerblue", "olive", "orange"]

    # Labels
    labels = []
    for r in res:
        labels.append(r[1])

    print(filename)

    # Behavior Rewards and Confidence intervals
    mean_rewards = []
    ci_rewards = []
    for (i, r) in enumerate(res):

        if not r[0]: # If rewards are empty
            continue

        rewards = []
        for data in r[0]:
            rewards.append(data["R"])

        cum_sum_rewards = np.cumsum(np.dstack(rewards)[0], axis=0)

        # Mean rewards
        mean_reward = np.mean(cum_sum_rewards, axis=1)
        mean_rewards.append(mean_reward) 

        # CIs
        ci_05 = np.percentile(cum_sum_rewards, 5, axis=1)
        ci_95 = np.percentile(cum_sum_rewards, 95, axis=1)
        ci_rewards.append(np.vstack((ci_05, ci_95)).T)

    # Behav. Policy
    baseline_R = mean_rewards[0]

    for i, (mean_reward, ci_reward) in enumerate(zip(mean_rewards[1:], ci_rewards[1:])):
            ax[k].step(
                range(mean_reward.shape[0]),
                mean_reward - baseline_R, 
                label = labels[i],
                color = colors[i],
                linewidth=0.5)
            ax[k].fill_between(
                range(mean_reward.shape[0]),
                ci_reward[:, 0] - baseline_R,
                ci_reward[:, 1] - baseline_R,
                color = colors[i],
                alpha = .1)

    ax[k].axhline(0, color = "black", linewidth=0.5)
    ax[k].set_title(titles[k], **font)

    yabs_max = abs(max(ax[k].get_ylim(), key=abs))
    ax[k].set_ylim(ymin=-yabs_max, ymax=yabs_max)
    # ax[k].set_ylabel("Cumulative Reward")
    # ax[k].set_xlabel("Time")
    # ax[k].legend(loc = "upper right")

fig.text(0, 0.5, "Cumulative Reward", va='center', rotation='vertical', **font)
fig.text(0.5, 0.02, "Time", va='center', **font)

plt.tight_layout()
plt.savefig("pics/MDP_examples.png", bbox_inches='tight')

plt.show()