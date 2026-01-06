from imports import *
from vectorized_model import VectorizedModel
from network_utils import *
from network_generation import *

G_default = barabasi_albert_directed(100, 5)


def run_vectorized_simulation_with_params(
    param_dict,
    tolerance=5 * 1e-03,
    tstep_stopping=True,
    seed=420,
    seeded=False,
    number_of_steps=10000,
    show_bar=False,
    agent_type="beta",
):
    process_seed = int.from_bytes(os.urandom(4), byteorder="little")
    rd.seed(process_seed)

    # Extract the network
    my_network = param_dict["network"]

    my_model = VectorizedModel(
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
        seeded=seeded,
        agent_type=agent_type,
    )

    my_model.run_simulation(number_of_steps=number_of_steps, show_bar=show_bar)

    result_dict = {
        key: value
        for key, value in param_dict.items()
        if isinstance(value, (int, float, str, tuple, list, bool))
    }

    result_dict["share_of_correct_agents_at_convergence"] = my_model.conclusion
    result_dict["share_of_core_agents_at_convergence"] = my_model.conclusion_core
    result_dict["convergence_step"] = my_model.n_steps
    result_dict["proportion_reached_by_truth"] = my_model.proportion_reached_by_truth

    if "group_id" in param_dict:
        result_dict["group_id"] = param_dict["group_id"]
    if "sim_index" in param_dict:
        result_dict["sim_index"] = param_dict["sim_index"]

    return result_dict
