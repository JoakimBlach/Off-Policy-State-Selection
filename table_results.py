import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

filenames = [
    "output/pricing_specs.pkl",
    "output/pricing_no_markov_0.pkl",
    "output/pricing_no_markov_0.1.pkl",
    "output/pricing_no_markov_0.25.pkl",
    "output/pricing_no_markov_0.5.pkl",
    "output/pricing_no_markov_0.75.pkl",
    "output/pricing_no_markov_0.9.pkl",
    "output/pricing_dyn_back_door_0.pkl",
    "output/pricing_dyn_back_door_1.pkl",
    "output/pricing_dyn_back_door_2.pkl",
    "output/pricing_dyn_back_door_4.pkl",
    "output/pricing_confounded_0.pkl",
    "output/pricing_confounded_1.pkl",
    "output/pricing_confounded_3.pkl",
    "output/pricing_confounded_5.pkl"
]

def percentage(x, y):
    return np.round(((y - x) / x) * 100, 2)

for k, filename in enumerate(filenames):
    with open(filename, 'rb') as file:
        res = pickle.load(file)

    # Behavior Rewards and Confidence intervals
    mean_rewards = []
    std_devs = []
    ci_rewards = []
    for (i, r) in enumerate(res):
        rewards = []
        if not r[0]:
            continue

        for data in r[0]:
            rewards.append(data["R"])

        cum_sum_rewards = np.cumsum(np.vstack(rewards).T, axis=0)

        # Check that it is normal

        # data = (cum_sum_rewards[-1, :] - np.mean(cum_sum_rewards, axis=1)[-1]) / np.std(cum_sum_rewards[-1, :], ddof=1)

        # points = np.linspace(-4, 4, 100)
        # from scipy import stats
        # import seaborn as sns

        # ax = sns.histplot(data, kde=False, stat='density', label='samples')
        # x0, x1 = ax.get_xlim()  # extract the endpoints for the x-axis
        # x_pdf = np.linspace(x0, x1, 100)
        # y_pdf = stats.norm.pdf(x_pdf)

        # ax.plot(x_pdf, y_pdf, 'r', lw=2, label='pdf')                                                   

        # plt.show()

        # Mean rewards
        mean_reward = np.round(np.mean(cum_sum_rewards, axis=1))
        mean_rewards.append(mean_reward)

        # Standard deviations
        std_dev = np.round(np.std(cum_sum_rewards[-1, :], ddof=1), 2)
        std_devs.append(std_dev)

        # CIs
        ci_05 = np.percentile(cum_sum_rewards, 5, axis=1)
        ci_95 = np.percentile(cum_sum_rewards, 95, axis=1)
        ci_rewards.append(np.vstack((ci_05, ci_95)).T)

    # TODO: Something is wrong! Get opposite results for no Markov??


    # Behav. Policy
    baseline_R = mean_rewards[0]

    if filename == "output/pricing_specs.pkl":
        print(fr"""     ${{\gG}}$                                                   &     -     &   {mean_rewards[0][-1]}   &  {mean_rewards[1][-1]}    & {mean_rewards[2][-1]} & {percentage(mean_rewards[0][-1], mean_rewards[1][-1])}   & {percentage(mean_rewards[0][-1], mean_rewards[2][-1])}    \\ """)
        print(fr"""                                                                 &           &   ({std_devs[0]})         &  ({std_devs[1]})          & ({std_devs[2]})       &                                                           &                                                           \\ \addlinespace""")
        continue

    alpha = filename.split("/")[1].split("_")[-1].split(".pkl")[0]
    if filename == "output/pricing_no_markov_0.pkl":
        print(fr"""  $\overset{{\textcolor{{yellow}}{{\longrightarrow}}}}{{\gG}}$   &  {alpha}  &   {mean_rewards[0][-1]}   &  {mean_rewards[1][-1]}    & -                     & {percentage(mean_rewards[0][-1], mean_rewards[1][-1])}   & -                                                         \\ """)
        print(fr"""                                                                 &           &   ({std_devs[0]})         &  ({std_devs[1]})          &                       &                                                           &                                                           \\ \addlinespace""")
    elif filename in [
            "output/pricing_no_markov_0.1.pkl",
            "output/pricing_no_markov_0.5.pkl",
            "output/pricing_no_markov_0.9.pkl",]:
        print(fr"""                                                                 &  {alpha}  &   {mean_rewards[0][-1]}   &  {mean_rewards[1][-1]}    & -                     & {percentage(mean_rewards[0][-1], mean_rewards[1][-1])}   & -                                                         \\ """)
        print(fr"""                                                                 &           &   ({std_devs[0]})         &  ({std_devs[1]})          &                       &                                                           &                                                           \\ \addlinespace""")
    if filename == "output/pricing_dyn_back_door_0.pkl":
        print(fr"""  $\overset{{\textcolor{{green}}{{\longrightarrow}}}}{{\gG}}$    &  {alpha}  &   {mean_rewards[0][-1]}   &  {mean_rewards[1][-1]}    & {mean_rewards[2][-1]} & {percentage(mean_rewards[0][-1], mean_rewards[1][-1])}   & {percentage(mean_rewards[0][-1], mean_rewards[2][-1])}    \\ """)
        print(fr"""                                                                 &           &   ({std_devs[0]})         &  ({std_devs[1]})          & ({std_devs[2]})       &                                                           &                                                           \\ \addlinespace""")
    elif filename in [
            "output/pricing_dyn_back_door_1.pkl",
            "output/pricing_dyn_back_door_2.pkl",
            "output/pricing_dyn_back_door_4.pkl"]:
        print(fr"""                                                                 &  {alpha}  &   {mean_rewards[0][-1]}   &  {mean_rewards[1][-1]}    & {mean_rewards[2][-1]} & {percentage(mean_rewards[0][-1], mean_rewards[1][-1])}   & {percentage(mean_rewards[0][-1], mean_rewards[2][-1])}    \\ """)
        print(fr"""                                                                 &           &   ({std_devs[0]})         &  ({std_devs[1]})          & ({std_devs[2]})       &                                                           &                                                           \\ \addlinespace""")
    if filename == "output/pricing_confounded_0.pkl":
        print(fr"""  $\overset{{\textcolor{{red}}{{\longrightarrow}}}}{{\gG}}$      &  {alpha}  &   {mean_rewards[0][-1]}   &  {mean_rewards[1][-1]}    &  -                    & {percentage(mean_rewards[0][-1], mean_rewards[1][-1])}   & -                                                         \\""")
        print(fr"""                                                                 &           &   ({std_devs[0]})         &  ({std_devs[1]})          &                       &                                                           &                                                           \\ \addlinespace""")
    elif filename in [
            "output/pricing_confounded_1.pkl",
            "output/pricing_confounded_3.pkl",
            "output/pricing_confounded_5.pkl"]:
        print(fr"""                                                                 &  {alpha}  &   {mean_rewards[0][-1]}   &  {mean_rewards[1][-1]}    &  -                    & {percentage(mean_rewards[0][-1], mean_rewards[1][-1])}   & -                                                         \\""")
        print(fr"""                                                                 &           &   ({std_devs[0]})         &  ({std_devs[1]})          &                       &                                                           &                                                           \\ \addlinespace""")
