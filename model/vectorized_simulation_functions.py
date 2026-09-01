from model.vectorized_model import VectorizedModel
from utils.imports import *
from utils.network_utils import *


def run_vectorized_simulation_with_params(
    param_dict,
    tolerance=5 * 1e-03,
    tolerance_stopping=True,
    tstep_stopping=False,
    seed=420,
    seeded=False,
    number_of_steps=10000,
    show_bar=False,
    agent_type="beta",
    compute_convergence=False,
    compute_root_analysis=False,
    auc_stopping=False,
    auc_threshold=0.95,
    auc_check_interval=500,
    snapshot_interval=0,
    # Choice-stability (decision-stability) stopping — see VectorizedModel:
    choice_stability_stopping=False,  # stop when every agent's choice is stable for W steps
    choice_stability_window=500,      # the window W (consecutive flip-free steps)
    choice_stability_min_steps=0,     # never stop before this many steps (0 = no floor)
    record_choice_flips=False,        # log (step, truth_share) flips -> result_dict for offline W-sweep
):
    # Per-job seed: prefer the one supplied in param_dict (typically derived
    # from numpy.random.SeedSequence.spawn for a reproducible study); otherwise
    # draw from OS entropy. The seed actually used is logged in result_dict so
    # any single run can be replayed and the whole study can be re-derived from
    # a master seed.
    if param_dict.get("seed") is not None:
        process_seed = int(param_dict["seed"])
    else:
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
        tolerance_stopping=tolerance_stopping,
        tstep_stopping=tstep_stopping,
        seed=seed,
        seeded=seeded,
        agent_type=agent_type,
        compute_convergence=compute_convergence,
        compute_root_analysis=compute_root_analysis,
        snapshot_interval=snapshot_interval,
        choice_stability_stopping=choice_stability_stopping,
        choice_stability_window=choice_stability_window,
        choice_stability_min_steps=choice_stability_min_steps,
        record_choice_flips=record_choice_flips,
    )

    my_model.run_simulation(
        number_of_steps=number_of_steps,
        show_bar=show_bar,
        auc_stopping=auc_stopping,
        auc_threshold=auc_threshold,
        auc_check_interval=auc_check_interval,
    )

    result_dict = {
        key: value
        for key, value in param_dict.items()
        if isinstance(value, (int, float, str, tuple, list, bool))
    }
    result_dict["seed"] = process_seed

    result_dict["share_of_correct_agents_at_convergence"] = my_model.conclusion
    result_dict["share_of_core_agents_at_convergence"] = my_model.conclusion_core
    result_dict["convergence_step"] = my_model.n_steps
    result_dict["proportion_reached_by_truth"] = my_model.proportion_reached_by_truth

    # Add convergence metrics if computed (Beta agent only)
    if compute_convergence and my_model.belief_change_abs is not None:
        result_dict["avg_belief_change_abs_theory_0"] = np.mean(
            my_model.belief_change_abs[:, 0]
        )
        result_dict["avg_belief_change_abs_theory_1"] = np.mean(
            my_model.belief_change_abs[:, 1]
        )
        result_dict["avg_belief_change_abs"] = np.mean(my_model.belief_change_abs)
        result_dict["avg_belief_change_kl_theory_0"] = np.mean(
            my_model.belief_change_kl[:, 0]
        )
        result_dict["avg_belief_change_kl_theory_1"] = np.mean(
            my_model.belief_change_kl[:, 1]
        )
        result_dict["avg_belief_change_kl"] = np.mean(my_model.belief_change_kl)
        # Store full arrays for detailed analysis (optional)
        result_dict["belief_change_abs_per_agent"] = my_model.belief_change_abs
        result_dict["belief_change_kl_per_agent"] = my_model.belief_change_kl

    # Add root analysis if computed
    if compute_root_analysis and my_model.root_analysis is not None:
        ra = my_model.root_analysis
        result_dict["n_roots"] = ra["n_roots"]
        result_dict["weighted_truth_share"] = ra["weighted_truth_share"]
        result_dict["unweighted_truth_share"] = ra["unweighted_truth_share"]
        # Node-level accuracy metrics
        result_dict["node_accuracy"] = ra.get("node_accuracy")
        result_dict["node_auc_roc"] = ra.get("node_auc_roc")
        # Store full root analysis dict for detailed analysis
        result_dict["root_analysis"] = ra

    if snapshot_interval > 0:
        result_dict["snapshots"] = my_model.snapshots

    # Flip history for the offline choice-stability window sweep: a list of
    # (step, truth_share) pairs. All windows W are derived from this single run.
    if record_choice_flips:
        result_dict["choice_flip_history"] = my_model.choice_flip_history

    if "group_id" in param_dict:
        result_dict["group_id"] = param_dict["group_id"]
    if "sim_index" in param_dict:
        result_dict["sim_index"] = param_dict["sim_index"]

    return result_dict
