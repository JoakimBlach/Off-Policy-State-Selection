# We want to compare states and find a setting where
# 1. without the state the behavior (and preferably the fixed), does worse than the
#   behavior
# 2. with the state behavior does better.

# OK! 

def compute_mean_reward(res):
    mean_rewards = []
    for (i, r) in enumerate(res):
        rewards = []
        if not r[0]:
            continue

        for data in r[0]:
            rewards.append(data["R"])

        cum_sum_rewards = np.cumsum(np.dstack(rewards)[0], axis=0)

        # Mean rewards
        mean_reward = np.mean(cum_sum_rewards, axis=1)
        mean_rewards.append(mean_reward[-1])
    return mean_rewards

# Load setting
import os 
import sys
import yaml 
import pickle
import numpy as np

state1 = {
  0: ["M"],
}

specs_covered = []
for spec_file in os.listdir("config/example3_tests"):
    with open(os.path.join("config/example3_tests", spec_file), 'r') as file:
        specs = yaml.safe_load(file)

    if not os.path.isfile(f"output/example3_tests/{spec_file.split(".")[0]}.pkl"):
        continue

    # check if condition 1 is satisfied
    with open(f"output/example3_tests/{spec_file.split(".")[0]}.pkl", "rb") as file:
        res = pickle.load(file)

    mean_rewards = compute_mean_reward(res)

    if mean_rewards[0] > mean_rewards[1]:
        print(mean_rewards)
        print(spec_file, "works!")
        print("\n")
    else:
        continue

    # if specs["state"] != state1:
    #     continue

    # # Check that L1_2 effect on L2 is not zero
    # if specs["variables"]["L2"]["kernel"]["terms"][4]["param"] == 0:
    #     continue

    # del specs['state']  # no longer need this



    # print(f"Satisfied for {spec_file}")
    # # find corresponding spec with state2
    # for spec_file2 in os.listdir("config/pricing_no_markov_tests"):
    #     with open(os.path.join("config/pricing_no_markov_tests", spec_file2), 'r') as file:
    #         specs2 = yaml.safe_load(file)
    
    #     if specs2["state"] != state2:
    #         continue

    #     # Match on all others
    #     del specs2['state']

    #     if specs != specs2:
    #         continue

    #     print("Found match!")

    #     # check if iter is better with state
    #     with open(f"output/pricing_no_markov_tests/{spec_file2.split(".")[0]}.pkl", "rb") as file:
    #         res = pickle.load(file)

    #     mean_rewards = compute_mean_reward(res)

    #     if mean_rewards[0] + 5 < mean_rewards[1]:
    #         print("Successfully found a setting.")
    #         print(f"{spec_file}")
    #         print(f"{spec_file2}")

    #         sys.exit()
    #     else:
    #         continue


# test_folder = "config/pricing_no_markov_tests"
# if not os.path.exists(test_folder):
#     os.makedirs(test_folder)

# # load all specs
# all_current_specs = []
# for spec_name in os.listdir(test_folder):
