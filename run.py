import os
import sys
import yaml
import copy
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt 

from scipy import stats
from scipy.special import expit

sys.path.insert(0,'/Users/joakim.blach.andersen/Documents/PhD/graph_simulator')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import optimize
from scipy.integrate import quad
from dag import DAG_Simulator
from MDP_utils import PolicyIterator
from MDP_plot import plot_results
from MDP_utils import DeterministicPolicy

from functools import wraps
import time

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

def compute_mean(episodes, gamma):

    rewards = np.vstack([episode["R"] for episode in episodes]).T
    gammas = np.array([gamma**t for t in range(rewards.shape[0])])[:, np.newaxis]

    disc_rewards = gammas * rewards

    return np.sum(np.mean(disc_rewards, axis=1))

def modify_specs(specs, state_key, policy):

    new_specs = copy.deepcopy(specs)

    if specs[state_key] is None:
        # for j, (action_name, action_dom) in enumerate(zip(action_names, action_var_domains)):
        for action_name in policy.columns:
            action_var_name = action_name.split("_")[0]

            new_specs['variables'][action_var_name] = {
                'kernel': {
                    'type': 'constant',
                    'value': policy.loc[0, action_name].item(),
                    'terms': None
                },
                'dependencies': None,
                'level_offset': 0.1}

        return new_specs

    # Create proper lag from action perspective
    action_lags = []
    for lag in list(specs["action"].keys())[::-1]:
        action_lags.append(lag)
    action_lag = max(action_lags)
    
    # subtract action-lag from state in specs
    # TODO: This might be buggy!
    state_spec = {}
    for lag in list(specs[state_key].keys())[::-1]:
        state_spec[lag - action_lag] = specs[state_key][lag]

    # Set policy
    for j, action_name in enumerate(policy.columns):
        action_var_name = action_name.split("_")[0]
        action_domain = policy[action_name].unique().tolist()

        # Type
        new_specs['variables'][action_var_name] = {
            'kernel': {
                'type': 'deterministic',
                'domain': action_domain,
                'terms': [
                    {
                        'param': 1,
                        'variables': state_spec
                    }
                ],
            },
            'dependencies':state_spec,
            'level_offset': 0.1}    

        # Fill out tree
        new_specs['variables'][action_var_name]["kernel"]["tree"] = []

        for state in policy.index.drop_duplicates():

            input_ = {"output": policy.loc[state, action_name].item()}

            k = 0
            for lag in sorted(list(state_spec.keys()), reverse=True):
                input_[lag] = {}
                for name in state_spec[lag]:
                    if isinstance(state, tuple):
                        input_[lag][name] = state[k]
                    else:
                        input_[lag][name] = state
                    k += 1

            new_specs['variables'][action_var_name]["kernel"]["tree"].append(input_)

    return new_specs

# @timeit
def sim_dataframe(specs, steps, observed_vars):
    simulator = DAG_Simulator(specs["variables"])

    df = pd.DataFrame(
        simulator.run(steps=steps),
        columns = observed_vars)

    return df

def save_experiment(res, output_path):
    # Save results
    with open(output_path, 'wb') as file:
        pickle.dump(res, file)

def main(args):
    # Load specs
    config_path = os.path.join("config", args.file_name) + ".yaml"
    print(f"Loading config file: {config_path}")
    with open(config_path, 'r') as file:
        specs = yaml.safe_load(file)

    # Test
    if False:
        beta = -0.5
        alpha = 3

        for a in [1, 2, 3]:
            value = 0
            for i in [1, 2, 3, 4]:
                value += a * np.exp(-1 + beta * a + alpha * (a < i))

            print(value)

        print("\n")

        value = 0
        for i in [1, 2, 3, 4, 5]:
            a = i - 1
            if a == 0:
                a = 1
            value += a * np.exp(-1 + beta * a + alpha * (a < i))

        print(value)

        sys.exit()

    # All observed variables
    observed_vars = [var for var in specs["variables"].keys()] # if var[0] != "U"]

    # df = sim_dataframe(
    #     specs=specs,
    #     steps=args.episode_length,
    #     observed_vars=observed_vars)

    # print(df.head(50))
    # # print(np.mean(df["L1"]))

    # # sns.histplot(data=df, x="R", hue="A")
    # # plt.show()

    # sys.exit()

    # Check if output exists
    output_path = os.path.join("output", args.file_name) + ".pkl"
    print(f"checking for {output_path}")
    if os.path.isfile(output_path):
        print(f"Output file {output_path} already exists.")
        return

    # Check if data exists
    data_path = os.path.join("data", args.file_name) + ".pkl"
    print(f"checking for {data_path}")
    if os.path.isfile(data_path) and not args.replace_data:
        print(f"Loading file: {data_path}")
        with open(data_path, 'rb') as file:
            episodes = pickle.load(file)
    else:
        episodes = Parallel(n_jobs=-1)(
            delayed(sim_dataframe)(specs=specs, steps=args.episode_length, observed_vars=observed_vars) 
            for i in tqdm(range(args.num_episodes))
        )

        with open(data_path, 'wb') as file:
            pickle.dump(episodes, file)

    if False:
        # Plot distribution
        df = pd.concat(episodes, axis=0)

        df = pd.concat(
            [
                df.shift(4).rename(columns={c: c + f"_{4}" for c in df.columns}),
                df.shift(3).rename(columns={c: c + f"_{3}" for c in df.columns}),
                df.shift(2).rename(columns={c: c + f"_{2}" for c in df.columns}),
                df.shift(1).rename(columns={c: c + f"_{1}" for c in df.columns}),
                df
            ],
            axis=1)

        df = df.dropna()

        for a in specs['variables']['A2']['domain']:
            print(np.mean(df[df['A2_1'] == a]['L2']))

        print("\n")

        for a in specs['variables']['A2']['domain']:
            print(np.mean(df[df['A2_1'] == a]['R']))

        sns.histplot(data=df, x="L2")
        plt.show()

        sys.exit()


        # df = df[(df["A1_2"] == 2) & (df["L1_2"] == 0) & (df["X_2"] == -1) & (df["A1_1"] == 1) & (df["A2_1"] == 1)]
        print(df[['A1_3', 'A1_1', 'L1', 'R']].head(20))

        print("A1_2 - L2") # immediate reward
        for a in specs['variables']['A2']['domain']:
            print(np.mean(df[df['A2_1'] == a]['L2']))

        print("\n")

        print("A1_2 - R") # immediate reward
        for i, a in enumerate(specs['variables']['A2']['domain']):
            print(np.mean(df[df['A2_1'] == a]['R']))

        sns.histplot(data=df, x="L2")
        plt.show()

        sys.exit()

        print("A1_1 - L1") # immediate reward
        for a in specs['variables']['A1']['domain']:
            print(np.mean(df[df['A1_1'] == a]['L1']))

        print("\n")

        print("A1_1 - R") # immediate reward
        for i, a in enumerate(specs['variables']['A1']['domain']):
            print(np.mean(df[df['A1_1'] == a]['R']))

        sns.histplot(data=df, x="L1")
        plt.show()

        sys.exit()
        print("\n")

        print("A1_3 - R") # immediate reward
        for a in specs['variables']['A1']['domain']:
            print(np.mean(df[df['A1_3'] == a]['R']))

        sys.exit()

        print("A1_3 - L1_2")
        for a in specs['variables']['A1']['domain']:
            print(np.mean(df[df['A1_3'] == a]['L1_2']))

        print("\n")

        print("L1_2 - L2")
        for l in specs['variables']['L1']['domain']:
            print(np.mean(df[df['L1_2'] == l]['L2']))

        print("\n")

        print("A1_3 - L2")
        for a in specs['variables']['A1']['domain']:
            print(np.mean(df[df['A1_3'] == a]['L2']))

        print("\n")

        print("L2 - R")
        for l in specs['variables']['L2']['domain']:
            print(np.mean(df[df['L2'] == l]['R']))

        print("\n")

        print("A1_3 - R")
        for a in specs['variables']['A1']['domain']:
            print(np.mean(df[df['A1_3'] == a]['R']))



        sys.exit()

    # State policy iteration
    policy_iterator = PolicyIterator(
        episodes,
        specs["state"],
        specs["action"],
        specs["reward"],
        specs['variables'])

    # # Regression analysis for A2 policy
    # print(policy_iterator.data_lagged.head())

    # mean_y = policy_iterator\
    #     .data_lagged\
    #     .groupby(['A1_2', 'L1_1'])['D_2'].mean().unstack()

    # # Plot the heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(mean_y, annot=True, cmap="viridis")
    # plt.xlabel("A1_2")
    # plt.ylabel("L1_1")
    # plt.show()

    # D_2 is almost certaintly 1 when:
    # 1) A1_2>=2 & L1_1 > 0 (simple!)

    iter_policy = policy_iterator.policy_iteration()
    print(f"{iter_policy.head(50)=}")

    iter_specs = modify_specs(specs, "state", iter_policy)
    # print(iter_specs['variables']["A"])

    # df = sim_dataframe(
    #     specs=iter_specs,
    #     steps=args.episode_length,
    #     observed_vars=observed_vars)

    # print(df["A"])

    # sys.exit()

    # Sim. data
    iter_episodes = Parallel(n_jobs=-1)(
        delayed(sim_dataframe)(specs=iter_specs, steps=args.episode_length, observed_vars=observed_vars)
        for _ in tqdm(range(args.num_eval_episodes))
    )

    if "correct_state" in specs.keys():
        correct_state_iterator = PolicyIterator(
            episodes,
            specs["correct_state"],
            specs["action"],
            specs["reward"],
            specs['variables'])

        opt_policy = correct_state_iterator.policy_iteration()
        print(f"{opt_policy=}")
    
        opt_specs = modify_specs(specs, "correct_state", opt_policy)

        opt_episodes = Parallel(n_jobs=-1)(
            delayed(sim_dataframe)(specs=opt_specs, steps=args.episode_length, observed_vars=observed_vars)
            for _ in tqdm(range(args.num_eval_episodes))
        )
    else:
        opt_episodes = []

    res = [
        (episodes, r'$\pi^b(s)$'),
        (iter_episodes, r'$\pi^i(s)$'),
        (opt_episodes, r'$\pi^{\ast}(s)$')]

    if args.save_results == "True":
        save_experiment(res, output_path)

    if args.plot == "True":
        plot_results(res)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Graph Simulator.')

    parser.add_argument('-g', '--gamma', default=0.95, type=float)
    parser.add_argument('-n', '--num_episodes', default=10, type=int)
    parser.add_argument('-ne', '--num_eval_episodes', default=1000, type=int)
    parser.add_argument('-T', '--episode_length', default=1000, type=int)
    parser.add_argument('-fn', '--file_name', default="some_file", type=str)
    parser.add_argument('-r', '--replace_data', default=False, type=bool)
    parser.add_argument('-p', '--plot', choices=["True", "False"], default="True")
    parser.add_argument('-s', '--save_results', choices=["True", "False"], default="True")

    args = parser.parse_args()

    main(args)