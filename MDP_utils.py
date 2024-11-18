import sys
import copy
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm
from joblib import Parallel, delayed

class DeterministicPolicy:
    def __init__(self, states, actions, state_names=None, action_names=None):
        self.state_names = state_names
        self.states = states

        self.action_names = action_names
        self.actions = actions

        self.policy_actions = {}
        for state in states:
            self.policy_actions[state] = actions[0]

    def predict(self, state):
        return self.policy_actions[state]

def get_variable_names(name_dict):    
    names = []
    if not name_dict:
        return names

    for lag in list(name_dict.keys())[::-1]:
        for var in name_dict[lag]:
            if lag == 0:
                col_name = var
            else:
                col_name = var + f"_{lag}"

            names.append(col_name)

    return names

class PolicyIterator:
    def __init__(self, data, state, action, reward, specs):
        # Set specs
        self.specs = specs

        # Create state and action columns
        self.state_names = get_variable_names(state)
        self.action_names = get_variable_names(action)
        self.reward_names = get_variable_names(reward)

        self.next_state_names = []
        for name in self.state_names:
            var_name = name.split("_")[0]            
            lag = int(name.split("_")[1]) - 1

            if lag != 0:
                var_name = var_name + "_" + str(lag)
            
            self.next_state_names.append(var_name)

        print(f"{self.state_names=}")
        print(f"{self.action_names=}")
        print(f"{self.reward_names=}")
        print(f"{self.next_state_names=}")
        print("\n")

        # TODO: Reward in state?
        self.reward_in_state = "R" in [i[0:1] for i in self.state_names]
        if self.reward_in_state:
            self.reward_idx = np.max([i for i in range(len(self.state_names)) if self.state_names[i][0:1] == "R"]).item()

        # Get lagged dataframe
        check_names = self.next_state_names + self.state_names + self.action_names
        print(f"{check_names=}")
        self.data_lagged = self.get_lagged_dataframe(data, check_names)
        
        print(self.data_lagged.head(3))
        print("... done")
        print("\n")

    def get_prob_matrix(self):
        print("... computing prob df")
        # Find unique values for each variable
        state_data = self.data_lagged[self.state_names]
        state_data.columns = ["s_" + col_name for col_name in state_data.columns]
        self.prob_df_state_names = list(state_data.columns)

        action_data = self.data_lagged[self.action_names]
        action_data.columns = ["a_" + col_name for col_name in action_data.columns]
        self.prob_df_action_names = list(action_data.columns)

        reward_data = self.data_lagged[self.reward_names]
        reward_data.columns = ["r_" + col_name for col_name in reward_data.columns]
        self.prob_df_reward_names = list(reward_data.columns)

        next_state_data = self.data_lagged[self.next_state_names]
        next_state_data.columns = ["ns_" + col_name for col_name in next_state_data.columns]
        self.prob_df_next_state_names = list(next_state_data.columns)

        # Columns
        state_action_columns = \
            self.prob_df_state_names + \
            self.prob_df_action_names

        all_columns = \
            self.prob_df_state_names + \
            self.prob_df_action_names + \
            self.prob_df_reward_names + \
            self.prob_df_next_state_names

        # Observed data
        observed_data = pd.concat(
            [
                state_data,
                action_data,
                reward_data,
                next_state_data
            ],
            axis=1)

        # Get unique values for each column
        self.state_values = list(
            itertools.product(
                *[state_data[col].unique().tolist() for col in state_data.columns]))        
        self.action_values = list(
            itertools.product(
                *[action_data[col].unique().tolist() for col in action_data.columns]))
        self.reward_values = list(
            itertools.product(
                *[reward_data[col].unique().tolist() for col in reward_data.columns]))
        self.next_state_values = list(
            itertools.product(
                *[next_state_data[col].unique().tolist() for col in next_state_data.columns]))

        print(f"{self.state_values=}")
        print(f"{self.action_values=}")
        print(f"{self.reward_values=}")
        print("\n")

        # Compute the Cartesian product of all unique values across columns
        all_combos = [s + a + r + ss for s, a, r, ss in itertools.product(
            self.state_values,
            self.action_values,
            self.reward_values,
            self.next_state_values)]

        # All columns
        prob_df = pd.DataFrame(all_combos, columns=all_columns) \
            .sort_values(all_columns) \
            .set_index(all_columns)

        # Merge with the original data to count occurrences
        count_data = observed_data.groupby(all_columns).size().to_frame('count')

        # Merge all combinations with count_data to ensure all combinations are represented
        prob_df = prob_df.join(count_data, how="left").fillna(0)

        # NB: THIS IS A HACK TO AVOID TOO MANY REPS
        # If nothing is observed - assume all options are equally likely.
        # prob_df = prob_df.fillna(1)
        # TODO: Delete to be sure about results! 

        # Calculate total counts for each state-action pair
        prob_df["total_count"] = prob_df.groupby(level=state_action_columns)["count"].transform("sum")

        # Calculate probabilities
        prob_df['probability'] = (prob_df['count'] / prob_df['total_count'])

        # Check probabilities
        prob_df['probability_sum'] = prob_df.groupby(state_action_columns)['probability'].transform('sum')

        prob_check = ~np.isclose(prob_df['probability_sum'], 1.0, atol=1e-3)
        if sum(prob_check) != 0:
            raise ValueError(f"No coverage for {sum(prob_check)} observations. \n {prob_df[prob_check]}")

        prob_df = prob_df[['probability']]

        return prob_df

    def get_lagged_dataframe(self, episodes, check_names):
        print(f"... lagging data...")
        transitions = []
        for episode in episodes:

            episode_lagged = episode.copy()

            lag = 1
            while True:
                col_missing = False
                for col in check_names:
                    if col not in episode_lagged.columns:
                        col_missing = True
                        break

                if col_missing:
                    episode_lag = episode \
                        .shift(lag) \
                        .rename(columns={c: c + f"_{lag}" for c in episode.columns})

                    episode_lagged = pd.concat(
                        [episode_lag, episode_lagged],
                        axis=1) \
                    .dropna()

                    lag += 1

                if not col_missing:
                    break

            transitions.append(episode_lagged)

        return pd.concat(transitions)

    # Policy iteration
    def policy_iteration(self, gamma=0.95):

        # Compute transition matrix
        self.prob_df = self.get_prob_matrix()

        print(f"Policy Iteration")
        print("\n")

        # If state is empty
        if not self.state_names:
            policy = pd.DataFrame(
                self.action_values[0],
                columns=self.action_names)

            # Policy iteration with empty state
            action_values = []
            for action in self.action_values:
                value = 0
                for reward in self.reward_values:
                    value += self.prob_df.loc[*[action + reward]] * reward[0]

                action_values.append(value)

            # Map to action
            policy[self.action_names] = self.action_values[np.argmax(action_values).item()]

            return policy

        # Policy
        policy = pd.DataFrame(self.state_values, columns=self.prob_df_state_names)
        policy[self.prob_df_action_names] = self.action_values[0]
        policy = policy.set_index(self.prob_df_state_names)

        # Policy Iteration
        k = 0
        while True:
            print(f"Iteration {k=}")

            # Policy Evaluation
            print(f"    Policy Evaluation")
            v = self.policy_evaluation(policy, gamma)
            # print(f"        values={np.round(v, 2)}")

            # Policy Improvement
            print(f"    Policy Improvement")
            policy, policy_stable = self.policy_improvement(v, policy, gamma)
            # print(f"        {policy=}")
            print("\n")

            if policy_stable:
                print("... converged!")
                break

            k += 1

        # Prepare output
        rename_idx = dict(zip(
            self.prob_df_state_names,
            self.state_names))

        rename_cols = dict(zip(
            self.prob_df_action_names,
            self.action_names))

        policy = policy.rename_axis(index=rename_idx)
        policy = policy.rename(columns=rename_cols)

        return policy

    # Policy Evaluation
    def policy_evaluation(self, policy, gamma):

        # State DF
        state_df = self.prob_df\
            .copy() \
            .reset_index()[self.prob_df_state_names] \
            .drop_duplicates() \
            .set_index(self.prob_df_state_names)

        # State-action DF
        state_action_df = state_df\
            .join(policy)\
            .set_index(self.prob_df_action_names, append=True)

        # Add rewards, next states, and transition probabilities
        state_action_df = state_action_df.join(self.prob_df)

        # State values
        rename_dict = dict(zip(self.prob_df_state_names, self.prob_df_next_state_names))

        state_values = state_df.copy()
        state_values['state_value'] = float(0)
        state_values = state_values.rename_axis(index=rename_dict)

        # Policy Evaluation
        delta = 0

        k = 0
        while True:

            if k % 50 == 0:
                print(f"        Iteration {k}. Delta={np.round(delta, 3).item()}")

            # Join next state values (NB: index in state values is self.prob_df_next_state_names)
            evaluation_df = state_action_df.join(state_values)

            # Compute new state values
            evaluation_df = evaluation_df.reset_index(self.prob_df_reward_names)

            evaluation_df['state_value'] = \
                evaluation_df['probability'] * \
                (
                    evaluation_df[self.prob_df_reward_names[0]] + \
                    gamma * evaluation_df['state_value']                    
                )

            evaluation_df = evaluation_df.drop(self.prob_df_reward_names + ['probability'], axis=1)

            new_state_values = evaluation_df\
                .groupby(level=self.prob_df_state_names)\
                .sum('state_value')\
                .rename_axis(index=rename_dict)

            delta = np.max(np.abs((state_values - new_state_values).to_numpy()))

            if delta < 1e-2:
                break

            # Set new state values
            state_values = new_state_values.copy()

            k += 1

        # State values to dict
        rename_dict = dict(zip(
            self.prob_df_next_state_names,
            self.state_names))

        state_values = state_values\
            .rename_axis(index=rename_dict)

        return state_values

    # Policy Improvement
    def policy_improvement(self, v, policy, gamma):

        # State values
        rename_dict = dict(zip(self.state_names, self.prob_df_next_state_names))

        state_values = v.copy()
        state_values = state_values.rename_axis(index=rename_dict)

        # Get state-action DF
        state_action_df = self.prob_df \
            .copy() \
            .join(state_values)

        state_action_df = state_action_df\
            .reset_index(self.prob_df_reward_names)

        # Compute state-action values
        state_action_df['state_value'] = state_action_df['probability'] * \
        (
            state_action_df[self.prob_df_reward_names[0]] + \
            gamma * state_action_df['state_value']                    
        )

        state_action_df = state_action_df\
            .drop(self.prob_df_reward_names + ['probability'], axis=1)

        state_action_df = state_action_df\
            .groupby(self.prob_df_state_names + self.prob_df_action_names)\
            .sum('state_value')

        # Get max actions
        max_action_idx = state_action_df\
            .groupby(self.prob_df_state_names)['state_value'].idxmax()
        max_state_value_df = state_action_df.loc[max_action_idx]

        # Check if there are diff. with current policy.
        optimal_policy = max_state_value_df\
            .drop(["state_value"], axis=1)\
            .reset_index(self.prob_df_action_names)

        policy_stable = True
        if not optimal_policy.equals(policy):
            policy_stable = False

        # Set policy to optimal in this iteration
        policy = optimal_policy.copy()

        return policy, policy_stable