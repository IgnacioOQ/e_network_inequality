from functools import partial
from multiprocessing import Pool
from pathlib import Path

from model.vectorized_simulation_functions import run_vectorized_simulation_with_params
from utils.imports import np, os, pd, pickle, rd, tqdm, uuid
from utils.network_utils import network_statistics, scatter_plot


_SA_NETWORK_ROWS_CACHE = {}
LEGACY_VARIATION_COLUMNS = ("p_rewiring", "proportion_edges")


def _sa_paths(sa_output_root):
    root = Path(sa_output_root).resolve()
    return root, root / "saved_graphs", root / "tables"


def validate_sa_paths(sa_output_root):
    root, saved_graphs_dir, tables_dir = _sa_paths(sa_output_root)
    if not root.exists():
        raise FileNotFoundError(f"SA output root does not exist: {root}")
    if not saved_graphs_dir.exists():
        raise FileNotFoundError(
            f"SA saved graph directory does not exist: {saved_graphs_dir}"
        )
    if not tables_dir.exists():
        raise FileNotFoundError(f"SA metadata table directory does not exist: {tables_dir}")


def resolve_network_type(network_type, network_type_source_ids):
    key = str(network_type).strip()
    if key not in network_type_source_ids:
        raise KeyError(
            f"Unknown network_type={network_type!r}. "
            f"Known network types: {sorted(network_type_source_ids)}"
        )
    return network_type_source_ids[key]


def _normalize_target_stats(target_stats):
    if target_stats is None:
        return None
    if isinstance(target_stats, str):
        return (target_stats,)
    return tuple(target_stats)


def _source_paths_for_network_type(network_type, sa_output_root, network_type_source_ids):
    _, saved_graphs_dir, tables_dir = _sa_paths(sa_output_root)
    source_id = resolve_network_type(network_type, network_type_source_ids)
    return source_id, saved_graphs_dir / source_id, tables_dir / f"{source_id}_results.parquet"


def available_sa_network_types(sa_output_root, network_type_source_ids=None):
    _, saved_graphs_dir, _ = _sa_paths(sa_output_root)
    return sorted(path.name for path in saved_graphs_dir.iterdir() if path.is_dir())


def has_sa_network_pool(network_type, sa_output_root, network_type_source_ids):
    _, graph_dir, table_path = _source_paths_for_network_type(
        network_type, sa_output_root, network_type_source_ids
    )
    return table_path.exists() and graph_dir.exists() and any(graph_dir.glob("*/*.pkl"))


def _parse_grid_seed(path):
    grid_index = None
    seed = None
    parts = path.stem.split("_")
    if len(parts) >= 4 and parts[0] == "grid" and parts[2] == "seed":
        try:
            grid_index = int(parts[1])
            seed = int(parts[3])
        except ValueError:
            pass
    return grid_index, seed


def _expected_graph_path(row, source_id, saved_graphs_dir):
    target_stat = row.get("target_stat", row.get("sa_target_stat", ""))
    grid_index = row.get("grid_index", row.get("sa_grid_index", None))
    seed = row.get("seed", row.get("sa_seed", None))
    if pd.isna(target_stat) or str(target_stat) == "" or pd.isna(grid_index) or pd.isna(seed):
        return None
    return (
        saved_graphs_dir
        / source_id
        / str(target_stat)
        / f"grid_{int(grid_index):03d}_seed_{int(seed)}.pkl"
    )


def _resolve_row_graph_path(row, source_id, saved_graphs_dir):
    expected = _expected_graph_path(row, source_id, saved_graphs_dir)
    if expected is None:
        raise ValueError(
            "Could not construct graph path from SA metadata row. "
            f"source_id={source_id!r}, row keys={list(row.index)}"
        )
    return expected


def load_sa_network_rows(
    network_type,
    sa_output_root,
    network_type_source_ids,
    target_stats=None,
):
    root, saved_graphs_dir, _ = _sa_paths(sa_output_root)
    source_id, graph_dir, table_path = _source_paths_for_network_type(
        network_type, root, network_type_source_ids
    )
    target_stats = _normalize_target_stats(target_stats)
    cache_key = (str(root), source_id, target_stats)
    if cache_key in _SA_NETWORK_ROWS_CACHE:
        return _SA_NETWORK_ROWS_CACHE[cache_key]

    if not graph_dir.exists():
        raise FileNotFoundError(f"SA saved graph source directory does not exist: {graph_dir}")
    if not table_path.exists():
        raise FileNotFoundError(f"SA metadata table does not exist: {table_path}")

    df = pd.read_parquet(table_path)
    if "feasible" in df.columns:
        df = df[df["feasible"].fillna(False)]
    if "saved_graph_path" in df.columns:
        df = df[df["saved_graph_path"].astype(str).str.len() > 0]
    if target_stats is not None:
        df = df[df["target_stat"].isin(set(target_stats))].copy()
    else:
        df = df.copy()

    if "source_id" not in df.columns:
        df["source_id"] = source_id

    if df.empty:
        raise FileNotFoundError(
            f"No feasible SA graph metadata rows found for network_type={network_type!r} "
            f"(resolved source_id={source_id!r}) in {table_path}"
        )

    graph_paths = []
    missing_paths = []
    for _, row in df.iterrows():
        path = _resolve_row_graph_path(row, source_id, saved_graphs_dir)
        graph_paths.append(str(path))
        if not path.exists():
            missing_paths.append(path)

    if missing_paths:
        examples = "\n".join(str(path) for path in missing_paths[:5])
        raise FileNotFoundError(
            f"{len(missing_paths)} graph paths from {table_path} do not exist. "
            f"First missing paths:\n{examples}"
        )

    df["network_path"] = graph_paths
    df = df.reset_index(drop=True)
    _SA_NETWORK_ROWS_CACHE[cache_key] = df
    return df


def describe_sa_network_pool(network_type, sa_output_root, network_type_source_ids):
    df = load_sa_network_rows(network_type, sa_output_root, network_type_source_ids)
    summary = (
        df.groupby(["source_id", "target_stat"])
        .size()
        .reset_index(name="n_graphs")
        .sort_values(["source_id", "target_stat"])
    )
    print(f"{network_type!r} -> {resolve_network_type(network_type, network_type_source_ids)!r}")
    print(f"{len(df):,} saved graphs available")
    return summary


def _copy_scalar_sa_metadata(row):
    metadata = {}
    for key, value in row.items():
        if key == "network_path":
            continue
        if isinstance(value, (str, bool, int, float, np.bool_, np.integer, np.floating)):
            metadata[f"sa_{key}"] = value.item() if hasattr(value, "item") else value
    return metadata


def _row_has(row, key):
    return key in row and pd.notna(row[key])


def generate_parameters_here(
    _,
    network_type,
    sa_output_root,
    network_type_source_ids,
    parameter_generation_settings,
    target_stats=None,
):
    process_seed = int.from_bytes(os.urandom(4), byteorder="little")
    rd.seed(process_seed)
    uncertainty = rd.uniform(
        parameter_generation_settings["uncertainty_min"],
        parameter_generation_settings["uncertainty_max"],
    )
    n_experiments = rd.randint(
        parameter_generation_settings["n_experiments_min"],
        parameter_generation_settings["n_experiments_max_exclusive"],
    )

    rows = load_sa_network_rows(
        network_type,
        sa_output_root,
        network_type_source_ids,
        target_stats=target_stats,
    )
    row = rows.iloc[int(rd.randint(0, len(rows)))]
    source_id = str(row.get("source_id", resolve_network_type(network_type, network_type_source_ids)))
    network_path = str(row["network_path"])

    result = {
        "randomized": True,
        "unique_id": uuid.uuid4().hex,
        "network_type": str(network_type),
        "source_id": source_id,
        "network_path": network_path,
        "uncertainty": float(uncertainty),
        "n_experiments": int(n_experiments),
        "parameter_random_seed": process_seed,
    }
    result.update(_copy_scalar_sa_metadata(row))

    if _row_has(row, "achieved_degree_gini"):
        result["degree_gini_coefficient"] = float(row["achieved_degree_gini"])
    if _row_has(row, "achieved_clustering"):
        result["approx_average_clustering_coefficient"] = float(row["achieved_clustering"])
    if _row_has(row, "achieved_density") and _row_has(row, "n"):
        result["average_degree"] = float(row["achieved_density"]) * (int(row["n"]) - 1)

    return result


def run_vectorized_simulation_from_path(param_dict, *args, **kwargs):
    params = dict(param_dict)
    graph_path = Path(params["network_path"])
    if not graph_path.exists():
        raise FileNotFoundError(f"Cannot find saved graph: {graph_path}")

    with graph_path.open("rb") as handle:
        network = pickle.load(handle)

    params["network_path"] = str(graph_path)
    params["network"] = network
    params.setdefault("n_agents", int(network.number_of_nodes()))

    stat_keys = {
        "average_degree",
        "degree_gini_coefficient",
        "approx_average_clustering_coefficient",
    }
    if any(key not in params for key in stat_keys):
        for key, value in network_statistics(network).items():
            params.setdefault(key, value)

    return run_vectorized_simulation_with_params(params, *args, **kwargs)


def results_path_for_network_type(dumping_path, network_type, results_name=None):
    name = results_name or str(network_type)
    return Path(dumping_path) / f"{name}_results_sa.csv"


def run_network_type(
    network_type,
    dumping_path,
    num_cores,
    sa_output_root,
    network_type_source_ids,
    parameter_generation_settings,
    simulation_kwargs,
    n_simulations,
    target_stats=None,
    results_name=None,
    combine_results=True,
):
    if not has_sa_network_pool(network_type, sa_output_root, network_type_source_ids):
        print(
            f"No saved SA graph pool found for {network_type!r} "
            f"(resolved source_id={resolve_network_type(network_type, network_type_source_ids)!r})."
        )
        print(f"Available saved source ids: {available_sa_network_types(sa_output_root)}")
        return pd.DataFrame()

    print(f"Running simulations for network_type={network_type!r}")
    print("Generating parameters...")
    generate_params = partial(
        generate_parameters_here,
        network_type=network_type,
        sa_output_root=sa_output_root,
        network_type_source_ids=network_type_source_ids,
        parameter_generation_settings=parameter_generation_settings,
        target_stats=target_stats,
    )

    with Pool(num_cores) as pool:
        param_dict = list(
            tqdm(
                pool.imap_unordered(generate_params, range(n_simulations)),
                total=n_simulations,
                desc="Generating parameters",
            )
        )

    run_simulation_wrapper = partial(
        run_vectorized_simulation_from_path,
        **simulation_kwargs,
    )
    with Pool(num_cores) as pool:
        simulation_results = list(
            tqdm(
                pool.imap_unordered(run_simulation_wrapper, param_dict),
                total=len(param_dict),
                desc="Running simulations",
            )
        )

    results_path = results_path_for_network_type(
        dumping_path, network_type, results_name=results_name
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    new_results_df = pd.DataFrame(simulation_results)

    if results_path.exists() and combine_results:
        existing_results_df = pd.read_csv(results_path)
        combined_results_df = pd.concat([existing_results_df, new_results_df], ignore_index=True)
    else:
        combined_results_df = new_results_df

    combined_results_df = combined_results_df.drop(
        columns=[col for col in LEGACY_VARIATION_COLUMNS if col in combined_results_df.columns]
    )
    combined_results_df.to_csv(results_path, index=False)
    print(f"Saved {len(combined_results_df):,} rows to {results_path}")
    return combined_results_df


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return list(value)


def _plot_selected_variables(df, plot_variables, target_variable):
    selected_columns = list(dict.fromkeys([*plot_variables, target_variable]))
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(
            f"Cannot plot missing columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
    scatter_plot(df[selected_columns], target_variable=target_variable)


def plot_network_type_results(
    network_type,
    dumping_path,
    results_name=None,
    plot_variables=None,
    target_variables=None,
):
    results_path = results_path_for_network_type(
        dumping_path, network_type, results_name=results_name
    )
    combined_results_df = pd.read_csv(results_path)
    print(len(combined_results_df))
    plot_variables = _as_list(plot_variables)
    target_variables = _as_list(target_variables) or [
        "share_of_correct_agents_at_convergence",
        "convergence_step",
    ]

    if plot_variables is None:
        for target_variable in target_variables:
            scatter_plot(combined_results_df, target_variable=target_variable)
    else:
        for target_variable in target_variables:
            _plot_selected_variables(combined_results_df, plot_variables, target_variable)

    return combined_results_df
