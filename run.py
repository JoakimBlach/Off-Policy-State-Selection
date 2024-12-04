import os
import sys
import yaml
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
import graph_simulator
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from MDP_utils import PolicyIterator, plot_results, modify_specs

sys.path.insert(0,'/Users/joakim.blach.andersen/Documents/PhD/graph_simulator')

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
    with open('config/iterated_specs.yaml', 'w') as outfile:
        yaml.dump(iter_specs, outfile, default_flow_style=False)

    # Sim. data
    iter_episodes = Parallel(n_jobs=-1)(
        delayed(sim_dataframe)('config/iterated_specs.yaml', steps=args.episode_length)
        for _ in tqdm(range(args.num_eval_episodes))
    )

    if "correct_state" in state_specs.keys():
        policy_iterator = PolicyIterator(
            episodes, 
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