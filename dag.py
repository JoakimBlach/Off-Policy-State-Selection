import sys
import yaml
import timeit
import itertools
import numpy as np
import networkx as nx

from scipy import stats
from scipy.special import expit

class ConstantKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name
        self.dependencies = None
        self.value = kernel_params["value"]

    def predict(self, observation):
        return self.value

    def sample(self):
        return self.value

class MixedKernel:
    def __init__(self, var_name, kernel_params):

        kernel_mapping = {
            "poisson": PoissonKernel,
            "mixed_poisson": MixedPoissonKernel,
            "binomial": BinomialKernel,
            "uniform": UniformKernel,
            "linear": LinearKernel,
            "deterministic": DeterministicKernel,
            "constant": ConstantKernel,
        }

        self.var_name = var_name
        self.dependencies = None
        self.kernel_params = kernel_params

        self.kernels = []
        for kernel in kernel_params["kernels"]:
            kernel_type = kernel["type"]

            self.kernels.append(
                kernel_mapping[kernel_type](var_name, kernel)
            )

        self.mixed_probs = kernel_params["mixed_probs"]

    def predict(self, observation):

        # sample kernel
        kernel_idx = np.random.choice(
            a = list(range(len(self.kernels))),
            p = self.mixed_probs)

        # get kernel
        kernel = self.kernels[kernel_idx]

        return kernel.predict(observation)

    def sample(self):
        return self.kernels[0].sample()

class DeterministicKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name
        self.dependencies = None
        self.terms = kernel_params["terms"]
        self.tree = kernel_params["tree"]
        self.domain = kernel_params["domain"]

    def predict(self, observation):
        for leaf in self.tree:
            correct_leaf = True

            for lag in set(leaf) - set(["output"]):
                for var in leaf[lag]:
                    obs_var_val = observation[lag][var]
                    leaf_var_val = leaf[lag][var]

                    if obs_var_val != leaf_var_val:
                        correct_leaf = False

            if not correct_leaf:
                continue

            # if observation[0]["DE"] == 1:
            #     print(f"{observation=}")
            #     print(f"{leaf=}")
            #     print(f"{leaf["output"]=}")

            return leaf["output"]

        raise ValueError("No leaf match found.")

    def sample(self):
        return np.random.choice(self.domain)


class NormalKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name

    def pdf(self, x, observation):
        return stats.norm.pdf(x, 0, 1).item()

    def predict(self, observation):
        return stats.norm.rvs(0, 1).item()

    def sample(self):
        return stats.norm.rvs(0, 1).item()

class LinearKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name

        self.sample_domain = kernel_params["sample_domain"]
        self.terms = kernel_params["terms"]

        self.indicator_terms = None
        if "indicator_terms" in kernel_params.keys():
            self.indicator_terms = kernel_params["indicator_terms"]

        self.lower_bound = None
        if "lower_bound" in kernel_params.keys():
            self.lower_bound = kernel_params["lower_bound"]

        self.upper_bound = None
        if "upper_bound" in kernel_params.keys():
            self.upper_bound = kernel_params["upper_bound"]

        self.intercept = 0
        if "intercept" in kernel_params.keys():
            self.intercept = kernel_params["intercept"]

        self.noise_type = None
        if "noise" in kernel_params.keys():
            self.noise_type = kernel_params["noise"]["type"]
            if "prob" in kernel_params["noise"].keys():
                self.noise_prob = kernel_params["noise"]["prob"]
            else:
                self.noise_prob = None

    def predict(self, observation):

        prod = linear_predictor(self.terms, self.intercept, observation)

        prod += indicator_predictor(self.indicator_terms, observation)

        # Add stochastic term
        if self.noise_type == "random":
            coin = stats.binom.rvs(n = 1, p = self.noise_prob)
            if coin:
                return self.sample()

        if self.lower_bound is not None:
            if prod < self.lower_bound:
                return self.lower_bound

        if self.upper_bound is not None:
            if prod > self.upper_bound:
                return self.upper_bound

        return prod

    def sample(self):
        return np.random.choice(self.sample_domain).item()

class UniformKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name
        self.domain = kernel_params["domain"]

        self.probs = None
        if "probs" in kernel_params.keys():
            self.probs = kernel_params["probs"]

    def predict(self, observation):
        value = np.random.choice(self.domain, p=self.probs).item()
        return value

    def sample(self):
        value = np.random.choice(self.domain).item()
        return value

class PoissonKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name
        self.terms = kernel_params["terms"]

        if "indicator_terms" in kernel_params.keys():
            self.indicator_terms = kernel_params["indicator_terms"]
        else:
            self.indicator_terms = []

        # capacity limit
        self.limit_value = None
        self.limit_variables = None
        if "limit" in kernel_params.keys():
            self.limit_value = kernel_params["limit"]["value"]
            self.limit_variables = kernel_params["limit"]["variables"]

        if "intercept" in kernel_params.keys():
            self.intercept = kernel_params["intercept"]
        else:
            self.intercept = 0

        self.noise_type = None
        if "noise" in kernel_params.keys():
            self.noise_type = kernel_params["noise"]["type"]
            if "prob" in kernel_params["noise"].keys():
                self.noise_prob = kernel_params["noise"]["prob"]
            else:
                self.noise_prob = None

    def predict(self, observation):
        mu = self._get_mu(observation)

        # print(f"{observation=}")
        # print(f"{mu=}")

        output = stats.poisson.rvs(mu)

        # Add stochastic term
        if self.noise_type == "random":
            coin = stats.binom.rvs(n = 1, p = self.noise_prob)
            if coin:
                return self.sample()

        if self.limit_value:
            current_value = 0
            if self.limit_variables:
                for lag in self.limit_variables:
                    for variable in self.limit_variables[lag]:
                        current_value += observation[lag][variable]

            if current_value + output > self.limit_value:
                return self.limit_value - current_value

        return output

    def _get_mu(self, observation):

        prod = linear_predictor(self.terms, self.intercept, observation)

        prod += indicator_predictor(self.indicator_terms, observation)

        return np.exp(prod)

    def sample(self):
        if self.limit_value:
            return np.random.choice(range(self.limit_value + 1)).item()
        else:
            return np.random.choice(range(10)).item()

class BinomialKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name

        self.dim = kernel_params["dim"]
        self.offset = 0
        if "offset" in kernel_params.keys():
            self.offset = kernel_params["offset"]
        self.terms = kernel_params["terms"]

        if "intercept" in kernel_params.keys():
            self.intercept = kernel_params["intercept"]
        else:
            self.intercept = 0

        if "noise" in kernel_params.keys():
            self.noise_type = kernel_params["noise"]["type"]
            if "prob" in kernel_params["noise"].keys():
                self.noise_prob = kernel_params["noise"]["prob"]
            else:
                self.noise_prob = None
        else:
            self.noise_type = None

    def pdf(self, x, observation):
        probs = self._get_p(observation)
        return stats.binom.pmf(x, self.dim - 1, probs).item()

    def predict(self, observation):

        # Add stochastic term
        if self.noise_type == "random":
            coin = stats.binom.rvs(n = 1, p = self.noise_prob)
            if coin:
                return self.sample()

        probs = self._get_p(observation)

        output = stats.binom.rvs(n = self.dim - 1, p = probs)

        return self.offset + output

    def _get_p(self, observation):

        if self.terms is None:
            return 0.5

        prod = linear_predictor(self.terms, self.intercept, observation)

        return expit(prod)

    def sample(self):
        output = np.random.choice(range(self.dim)).item()
        return self.offset + output

class MixedPoissonKernel:
    def __init__(self, var_name, kernel_params):
        self.var_name = var_name

        # demand
        self.d_terms = kernel_params["demand"]["terms"]
        self.d_constant = kernel_params["demand"]["constant"]
        self.d_intercept = kernel_params["demand"]["intercept"]            

        self.d_shock_prob = kernel_params["demand"]["shock"]["prob"]
        self.d_shock_value = kernel_params["demand"]["shock"]["value"]

        # success prob
        self.p_terms = kernel_params["success_prob"]["terms"]
        self.p_constant = kernel_params["success_prob"]["constant"]
        self.p_intercept = kernel_params["success_prob"]["intercept"]            

        # capacity limit
        self.limit_value = kernel_params["limit"]["value"]
        self.limit_variables = kernel_params["limit"]["variables"]

        # noise
        if "noise" in kernel_params.keys():
            self.noise_type = kernel_params["noise"]["type"]
            self.noise_prob = kernel_params["noise"]["prob"]

    def predict(self, observation):

        # Get distribution parameters
        d = self._get_expected_demand(observation)
        p = self._get_prob_of_rejection(observation)
        r = p / (1 - p) * d

        lam = stats.gamma.rvs(a = r, scale = (1 - p) / p)

        l = stats.poisson.rvs(lam)

        if self.noise_type == "random":
            if stats.bernoulli.rvs(self.noise_prob):
                l = self.sample()

        # print(f"{d=}, {p=}, {r=}, {l=}")

        remaining_value = self.remaining_value(observation)

        if remaining_value < l:
            return max(remaining_value, 0)

        return l

    def remaining_value(self, observation):
        remaining_value = self.limit_value

        if self.limit_variables is None:
            return remaining_value

        for lag in self.limit_variables.keys():
            for variable in self.limit_variables[lag]:
                remaining_value -= observation[lag][variable]

        return remaining_value
 
    def _get_expected_demand(self, observation):
        if stats.bernoulli.rvs(self.d_shock_prob):
            d = self.d_shock_value
        else:
            prod = linear_predictor(self.d_terms, self.d_intercept, observation)
            d = self.d_constant * np.exp(prod)

        return d

    def _get_prob_of_rejection(self, observation):
        prod = linear_predictor(self.p_terms, self.p_intercept, observation)

        p = 1 / (1 + self.p_constant * np.exp(prod))

        return p
    
    def sample(self):
        return np.random.choice(range(self.limit_value + 1))

def indicator_predictor(indicator_terms, observation):
    if indicator_terms is None:
        return 0

    total_value = 0
    for indicator_term in indicator_terms:

        indicator_value = linear_predictor(
            indicator_term["terms"],
            indicator_term["intercept"],
            observation)
        
        if (indicator_term["type"] == "multiple"):
            checks = []
            for term in indicator_term['indicators']:
                checks.append(check_single_indicator(term, observation))
            
            if all(checks):
                total_value += indicator_value
        else:
            if check_single_indicator(indicator_term, observation):
                total_value += indicator_value

    return total_value

def check_single_indicator(term, observation):
    check_value = get_check_value(term, observation)

    if (term["type"] == "greater_than_value"):
        if check_value > term['threshold']:
            return True

    if (term["type"] == "greater_or_equal_than_value"):
        if check_value >= term['threshold']:
            return True

    if (term["type"] == "smaller_or_equal_than_value"):
        if check_value <= term['threshold']:                
            return True

    if (term["type"] == "greater_than_variable"):
        for lag in term['threshold']:
            for variable in term['threshold'][lag]:
                threshold = observation[lag][variable]

        if check_value > threshold:                
            return True

def get_check_value(term, observation):
    # Get check_variable
    lag = list(term['variable'].keys())[0]
    variable = term['variable'][lag][0]

    check_value = observation[lag][variable]
    
    return check_value

def linear_predictor(terms, intercept, observation):

    prod = 0

    if intercept:
        prod += intercept

    if terms is None:
        return prod

    # print(observation)
    for term in terms:
        value = term["param"]
        for lag in term["variables"].keys():
            for var in term["variables"][lag]:
                # print(f"{lag=}, {var=}")
                value *= observation[lag][var]

        prod += value

    return prod

class DAG_Simulator:

    def __init__(self, specs):
        self.time_step = 0
        self.specs = specs
        self.kernels = {var: self._get_kernel(var, spec["kernel"]) for var, spec in specs.items()}
        self.G = nx.DiGraph()  # Initialize the graph early for reuse across methods

    def _get_kernel(self, var_name, kernel_params):
        kernel_type = kernel_params["type"]
        kernel_mapping = {
            "mixed": MixedKernel,
            "normal": NormalKernel,
            "poisson": PoissonKernel,
            "mixed_poisson": MixedPoissonKernel,
            "binomial": BinomialKernel,
            "uniform": UniformKernel,
            "linear": LinearKernel,
            "deterministic": DeterministicKernel,
            "constant": ConstantKernel,
        }
        return kernel_mapping[kernel_type](var_name, kernel_params)

    def _extend_graph(self):
        for var_name in self.topological_order:
            node_name = f"{var_name}_{self.time_step}"
            self._add_variable(var_name, self.time_step)
            self.G.add_edges_from([(parent_node, node_name) for parent_node in self._get_parent_nodes(var_name, self.time_step)])

    def get_parent_values(self, child_node: str):
        var_name = self.G.nodes[child_node]["var_name"]
        dependencies = self.specs[var_name]["dependencies"]

        if dependencies is None:
            return np.array([])

        parent_values = {}
        for graph_parent in self.G.predecessors(child_node):
            # if var_name == "X":
            #     print(f"    {graph_parent=}")
            # print(f"    {self.G.nodes=}")
            # print(f"    {self.G.nodes[graph_parent]=}")
            # print(f"    {self.G.nodes[graph_parent]["value"]=}")

            # lag = int(child_node.split("_")[1]) - int(graph_parent.split("_")[1])
            parent_lag = self.time_step - self.G.nodes[graph_parent]["time"]
            parent_var_name = self.G.nodes[graph_parent]["var_name"]
            parent_value = self.G.nodes[graph_parent]["value"]

            if not parent_lag in parent_values.keys():
                parent_values[parent_lag] = {}

            parent_values[parent_lag][parent_var_name] = parent_value

        return parent_values

    def _add_variable(self, var_name, time_step):
        node_name = var_name + f"_{time_step}"
        node_offset = self.specs[var_name]["level_offset"]
        self.G.add_node(node_name, level = time_step + node_offset, var_name = var_name, time = time_step)

    def _get_parent_nodes(self, var_name, time_step):
        dependencies = self.specs[var_name]["dependencies"]

        if dependencies is None:
            return []

        parent_nodes = []
        for lag, parent_vars in dependencies.items():
            for parent_var_name in parent_vars:
                t_append = time_step - int(lag)
                if t_append < 0:
                    continue

                parent_name = parent_var_name + f"_{str(t_append)}"

                if parent_name in self.G:
                    parent_nodes.append(parent_name)
                else:
                    raise ValueError(f"Parent {parent_name} of {var_name} at time-step {time_step} is missing in graph!")

        return parent_nodes

    def _set_values(self):
        values = {}

        for var_name, kernel in self.kernels.items():
            node_name = f"{var_name}_{self.time_step}"
            parent_values = self.get_parent_values(node_name)
            node_value = kernel.predict(parent_values)
            self.G.nodes[node_name]['value'] = node_value
            values[var_name] = node_value

            # if var_name == "X":
            #     print(f"    {self.topological_order}")
            #     print(f"    {node_name=}")

            #     print(f"    {var_name=}", f"{parent_values=}")
            #     print(f"    {node_value=}")
            #     print("\n")

            # if var_name == "A1":
            #     print(f"    {var_name=}", f"{parent_values=}")
            #     print(f"    {node_value=}")
            #     print("\n")

            # if var_name == "A2":
            #     print(f"    {var_name=}", f"{parent_values=}")
            #     print(f"    {node_value=}")
            #     print("\n")

            # if var_name == "L1":
            # print(f"    {node_value=}")

            # if var_name == "A":
            #     print(f"{var_name=}", f"{parent_values=}")
            #     print(f"{node_value=}")
                # if node_value != 1:
                #     sys.exit()
            # print(f"    {node_name=}")

        return values
    
    def run(self, steps=100):
        """will use TimeLimit wrapper for truncation..."""
        self._init_graph()

        data = []
        for _ in range(steps):
            # Extend Graph (build all next time-step variables)
            self._extend_graph()

            # Set node values
            data.append(self._set_values())

            # Clean-up graph (remove all non-dependencies)
            self._clean_up_graph()

            # Move one time-step
            self.time_step += 1

        return data

    def _clean_up_graph(self):
        relevant_nodes = set()
        min_required_time_step = self.time_step

        for var_name in self.specs.keys():
            node_name = f"{var_name}_{self.time_step}"
            relevant_nodes.add(node_name)
            for parent_node in self.G.predecessors(node_name):
                parent_time_step = int(parent_node.split("_")[1])
                min_required_time_step = min(min_required_time_step, parent_time_step)
                relevant_nodes.add(parent_node)

        redundant_nodes = [node for node in self.G.nodes if int(node.split("_")[1]) < min_required_time_step and node not in relevant_nodes]
        self.G.remove_nodes_from(redundant_nodes)

    def _init_graph(self, steps=5):
        # NB: Need to do this in two steps, b/c we don't know 
        # topological order yet. 

        # Init values
        for time_step in range(steps):
            for var_name in self.specs.keys():
                node_name = f"{var_name}_{time_step}"
                self._add_variable(var_name, time_step)
                self.G.nodes[node_name]["value"] = self.kernels[var_name].sample()

        # Init edges
        for time_step in range(steps):
            for var_name in self.specs.keys():
                node_name = f"{var_name}_{time_step}"
                # print(f"{node_name=}, {self._get_parent_nodes(var_name, time_step)=}")
                self.G.add_edges_from([(parent_node, node_name) for parent_node in self._get_parent_nodes(var_name, time_step)])

            self.time_step += 1

        # Set topological order
        topological_list = [i for i in list(nx.topological_sort(self.G)) if i.split("_")[1] == str(self.time_step - 1)]
        self.topological_order = [i.split("_")[0] for i in topological_list]

        self.kernels = {key: self.kernels[key] for key in self.topological_order} # sort kernel dict
