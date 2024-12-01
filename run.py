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
import graph_simulator
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import optimize
from scipy.integrate import quad
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

def modify_specs(state_specs, dag_specs, state_key, policy):

    new_dag_specs = copy.deepcopy(dag_specs)

    if state_specs[state_key] is None:
        # for j, (action_name, action_dom) in enumerate(zip(action_names, action_var_domains)):
        for action_name in policy.columns:
            action_var_name = action_name.split("_")[0]

            new_dag_specs[action_var_name] = {
                'kernel': {
                    'type': 'constant',
                    'noise': 0,
                    'value': policy.loc[0, action_name].item(),
                    'terms': None
                },
                'dependencies': None}

        return new_dag_specs

    # Create proper lag from action perspective
    action_lags = []
    for lag in list(state_specs["action"].keys())[::-1]:
        action_lags.append(lag)
    action_lag = max(action_lags)
    
    # subtract action-lag from state in specs
    # TODO: This might be buggy!
    state_spec = {}
    for lag in list(state_specs[state_key].keys())[::-1]:
        state_spec[lag - action_lag] = state_specs[state_key][lag]

    # Set policy
    for _, action_name in enumerate(policy.columns):
        action_var_name = action_name.split("_")[0]
        action_domain = policy[action_name].unique().tolist()
        sample_domain = dag_specs[action_var_name]['kernel']['sample_domain']

        # Type
        new_dag_specs[action_var_name] = {
            'kernel': {
                'type': 'linear',
                'noise': 0,
                'lower_bound': min(sample_domain),
                'upper_bound': max(sample_domain),
                'sample_domain': sample_domain,
                'terms': [],
            },
            'dependencies':state_spec}


        # Fill out tree
        for state in policy.index.drop_duplicates():

            term = {}
            term['value'] = 0
            term['variables'] = None

            # Get output
            term['intercept'] = policy.loc[state, action_name].item()

            # Construct indicators
            term['indicators'] = []

            k = 0
            for lag in sorted(list(state_spec.keys()), reverse=True):
                # input_[lag] = {}
                for name in state_spec[lag]:
                    if isinstance(state, tuple):
                        # input_[lag][name] = state[k]
                        term['indicators'].append(
                            {
                                "type": "equal_to",
                                "variable": {lag: name},
                                "threshold": state[k]

                            }
                        )
                    else:
                        # input_[lag][name] = state
                        term['indicators'].append(
                            {
                                "type": "equal_to",
                                "variable": {lag: name},
                                "threshold": state

                            }
                        )
                    k += 1

            new_dag_specs[action_var_name]['kernel']["terms"].append(term)

    return new_dag_specs

# @timeit
def sim_dataframe(config_path, steps):
    simulator = graph_simulator.DagSimulator(config_path)

    df = pd.DataFrame(simulator.run(steps=steps))

    return df

def save_experiment(res, output_path):
    # Save results
    with open(output_path, 'wb') as file:
        pickle.dump(res, file)

def main(args):
    # Load configs
    config_path = os.path.join("config", args.file_name) + ".yaml"
    with open(config_path, 'r') as file:
        dag_specs = yaml.safe_load(file)

    with open("config/states.yaml", 'r') as file:
        all_state_specs = yaml.safe_load(file)
    state_specs = all_state_specs[args.state_specs]

    # import time
    # start_time = time.time()
    # df = sim_dataframe(config_path, args.episode_length)

    # print(time.time() - start_time)

    # sys.exit()

    # Retrieving data
    episodes = []

    data_path = os.path.join("data", args.file_name) + ".pkl"
    if os.path.isfile(data_path):
        print(f"Data file {data_path} already exists.")
        with open(data_path, 'rb') as file:
            episodes += pickle.load(file)
        print(f"Loaded file with {len(episodes)} episodes.")
    if (not os.path.isfile(data_path)) or (args.append_data == "True"):
        print(f"Simulating data.")
        episodes += Parallel(n_jobs=-1)(
            delayed(sim_dataframe)(config_path, args.episode_length)
            for _ in tqdm(range(args.num_episodes))
        )

        print(f"Saving data.")
        with open(data_path, 'wb') as file:
            pickle.dump(episodes, file)

    # df = pd.concat(episodes, axis=0)
    # print(df.tail(20))
    # sys.exit()

    # # Plot histograms for all variables
    # df.hist(bins=10, figsize=(10, 5), layout=(1, len(df.columns)), edgecolor='black')

    # # Show the plot
    # plt.tight_layout()
    # plt.show()

    # print(df.head(5))

    # sys.exit()

    # State policy iteration
    policy_iterator = PolicyIterator(
        episodes,
        state_specs["state"],
        state_specs["action"],
        state_specs["reward"])

    iter_policy = policy_iterator.policy_iteration()
    print(f"{iter_policy.head(50)=}")

    # Save iterated policy specs
    iter_specs = modify_specs(state_specs, dag_specs, "state", iter_policy)
    # print(iter_specs["A"]['kernel'])
    with open('config/iterated_specs.yaml', 'w') as outfile:
        yaml.dump(iter_specs, outfile, default_flow_style=False)

    # Sim. data
    iter_episodes = Parallel(n_jobs=-1)(
        delayed(sim_dataframe)('config/iterated_specs.yaml', steps=args.episode_length)
        for _ in tqdm(range(args.num_eval_episodes))
    )

    if "correct_state" in state_specs.keys():
        policy_iterator.set_mdp_names(
            state_specs["correct_state"],
            state_specs["action"],
            state_specs["reward"])

        opt_policy = policy_iterator.policy_iteration()
        print(f"{opt_policy=}")

        opt_specs = modify_specs(state_specs, dag_specs, "correct_state", opt_policy)
        with open('config/opt_specs.yaml', 'w') as outfile:
            yaml.dump(opt_specs, outfile, default_flow_style=False)

        opt_episodes = Parallel(n_jobs=-1)(
            delayed(sim_dataframe)('config/opt_specs.yaml', args.episode_length)
            for _ in tqdm(range(args.num_eval_episodes))
        )
    else:
        opt_episodes = []

    res = [
        (episodes, r'$\pi^b(s)$'),
        (iter_episodes, r'$\pi^i(s)$'),
        (opt_episodes, r'$\pi^{\ast}(s)$')]

    output_path = os.path.join("output", args.file_name) + ".pkl"
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
    parser.add_argument('-sp', '--state_specs', default="some_file", type=str)
    parser.add_argument('-r', '--append_data', choices=["True", "False"], default="True")
    parser.add_argument('-p', '--plot', choices=["True", "False"], default="True")
    parser.add_argument('-s', '--save_results', choices=["True", "False"], default="True")

    args = parser.parse_args()

    main(args)