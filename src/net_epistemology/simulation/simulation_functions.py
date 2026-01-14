from net_epistemology.utils.imports import *
from net_epistemology.core.agents import BetaAgent, BayesAgent
from net_epistemology.core.model import Model
from net_epistemology.utils.network_utils import *

# from network_randomization import *
from net_epistemology.utils.network_generation import *

G_default = barabasi_albert_directed(100, 5)


def generate_parameters(_, G=G_default):
    unique_id = uuid.uuid4().hex
    # I am not sure what the three lines below are for
    process_seed = int.from_bytes(os.urandom(4), byteorder="little")
    rd.seed(process_seed)

    # Now what all simulations share
    uncertainty = rd.uniform(0.000001, 0.001)
    n_experiments = rd.randint(1000, 10000)

    # now we pick a random number
    p_rewiring = rd.rand()

    # Do randomization
    n_edges = int(p_rewiring * len(G.edges()))
    randomized_network = randomize_network(G, n_edges=n_edges)

    params = {
        "randomized": True,
        "unique_id": unique_id,
        "n_agents": int(len(randomized_network.nodes)),
        "network": randomized_network,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        "p_rewiring": float(p_rewiring),
    }
    stats = network_statistics(randomized_network)
    for stat in stats.keys():
        params[stat] = stats[stat]

    return params


def generate_parameters_fixed(
    _, G=G_default, uncertainty=0.005, n_experiments=50
):  # ,p_rewiring=0):
    unique_id = uuid.uuid4().hex
    # I am not sure what the three lines below are for
    process_seed = int.from_bytes(os.urandom(4), byteorder="little")
    rd.seed(process_seed)

    # Do randomization
    # randomized_network = randomize_network(G, p_rewiring=p_rewiring)

    params = {
        "randomized": True,
        "unique_id": unique_id,
        "n_agents": int(len(G.nodes)),
        "network": G,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        # "p_rewiring": float(p_rewiring),
    }
    stats = network_statistics(G)
    for stat in stats.keys():
        params[stat] = stats[stat]

    return params


def generate_parameters_aggregate(
    G=G_default, uncertainty=0.005, n_experiments=20, p_rewiring=0
):
    unique_id = uuid.uuid4().hex
    # process_seed = int.from_bytes(os.urandom(4), byteorder='little')
    # rd.seed(process_seed)
    # I am not sure what the three lines below are for
    params = {
        "randomized": True,
        "unique_id": unique_id,
        "n_agents": int(len(G.nodes)),
        "network": G,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        "p_rewiring": float(p_rewiring),
        # "parameterization_seed": process_seed,
    }
    stats = network_statistics(G)
    for stat in stats.keys():
        params[stat] = stats[stat]

    return params


def run_simulation_with_params(
    param_dict,
    tolerance=5 * 1e-03,
    tstep_stopping=True,
    seed=420,
    seeded=False,
    number_of_steps=10000,
    show_bar=False,
    agent_type="bayes",
):

    process_seed = int.from_bytes(os.urandom(4), byteorder="little")
    rd.seed(process_seed)

    # Extract the network directly since it's already a NetworkX graph objsect
    my_network = param_dict["network"]
    # Other parameters are directly extracted from the dictionary
    my_model = Model(
        my_network,
        n_experiments=param_dict["n_experiments"],
        uncertainty=param_dict["uncertainty"],
        tolerance=tolerance,
        histories=False,
        sampling_update=False,
        variance_stopping=False,
        tstep_stopping=tstep_stopping,
        directed_network=True,
        seed=seed,
        seeded=False,
        agent_type=agent_type,
    )
    # Run the simulation with predefined steps and show_bar option

    my_model.run_simulation(number_of_steps=number_of_steps, show_bar=show_bar)
    result_dict = {
        key: value
        for key, value in param_dict.items()
        if isinstance(value, (int, float, str, tuple, list, bool))
    }

    result_dict["share_of_correct_agents_at_convergence"] = my_model.conclusion
    result_dict["share_of_core_agents_at_convergence"] = my_model.conclusion_core
    result_dict["convergence_step"] = (
        my_model.n_steps
    )  # takes note of the last reported step
    result_dict["proportion_reached_by_truth"] = my_model.proportion_reached_by_truth
    if "group_id" in param_dict:
        result_dict["group_id"] = param_dict["group_id"]
    if "sim_index" in param_dict:
        result_dict["sim_index"] = param_dict["sim_index"]

    return result_dict
