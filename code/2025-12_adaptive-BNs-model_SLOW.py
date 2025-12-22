"""
Adaptive Belief Networks Model
version 2025-12-17, Peter Steiglechner, steiglechner@csh.ac.at
"""

import numpy as np
import pandas as pd
import os
import json
import string
from itertools import combinations
from scipy.sparse import csr_matrix
import time
from joblib import Parallel, delayed
import multiprocessing
from numba import jit


# def hebbianV(wij, delb_i, delb_j, eps):
#     """Hebbian learning from consistency/inconsistency of belief change"""
#     return eps * delb_i * delb_j


# REMOVED FOR NOW
# def socialconformity(wij, Wij, mu):
#     """Social conformity on BN edge weights."""
#     return mu * (Wij - wij) if ~np.isnan(Wij) else 0


# @jit(nopython=True)
# def decay(wij, lam):
#     """Decay term for edge weights"""
#     return -lam * wij


def social_energy(belief, social_beliefs, rho):
    """Calculate social energy for a belief value"""
    return -rho * belief * np.sum(social_beliefs)


def personal_energy(
    beliefs,
    dim,
    edge_weights,
    edge_list,
    external_pressure=None,
    external_pressure_strength=0,
):
    """Calculate personal energy for a specific dim."""
    H = 0.0
    for n, edge in enumerate(edge_list):
        if dim in edge:
            H -= edge_weights[n] * beliefs[edge[0]] * beliefs[edge[1]]
    if external_pressure is not None:
        H += -external_pressure_strength * beliefs[external_pressure] * 1
    return H


def glauber_probabilities_withSocial(
    dim,
    agent_beliefs,
    agent_edgeweights,
    edge_list,
    social_beliefs,
    belief_options,
    rho,
    beta,
    external_pressure=(None, 0),
):
    """Compute Glauber transition probabilities for a specific belief."""
    original_value = agent_beliefs[dim]
    curr_external_pressure = external_pressure[0]
    curr_external_pressure_strength = external_pressure[1]

    H0 = personal_energy(
        agent_beliefs,
        dim,
        agent_edgeweights,
        edge_list,
        curr_external_pressure,
        curr_external_pressure_strength,
    )
    H_soc_0 = social_energy(original_value, social_beliefs, rho)

    H = np.zeros(len(belief_options))
    H_soc = np.zeros(len(belief_options))
    for i, opt in enumerate(belief_options):
        agent_beliefs[dim] = opt
        H[i] = personal_energy(
            agent_beliefs,
            dim,
            agent_edgeweights,
            edge_list,
            curr_external_pressure,
            curr_external_pressure_strength,
        )
        H_soc[i] = social_energy(opt, social_beliefs, rho)

    agent_beliefs[dim] = original_value

    delH = (H - H0) + (H_soc - H_soc_0)
    exp_term = 1 / (1 + np.exp(beta * delH))
    return exp_term / np.sum(exp_term)


def initialise_agents_and_network(params):
    """Initialize social network of agents with personal belief networks"""
    np.random.seed(params["seed"])
    belief_dimensions = params["belief_dimensions"]
    n_agents = params["n_agents"]
    focal = params["focal_dim"]

    A = np.triu(
        (np.random.random((n_agents, n_agents)) <= params["link_prob"]).astype(bool),
        k=1,
    )  # adjacency matrix
    A = A + A.T
    A_sparse = csr_matrix(A)
    # neighbours = {
    #     agent: [agent_list[nb] for nb in A_sparse[i].indices]
    #     for i, agent in enumerate(agent_list)
    # }

    # Build agent dictionary
    agent_dict = {}
    edge_list = params["edge_list"]

    belief_options = params["belief_options"]
    n_options = len(belief_options)
    prob_options = np.ones(n_options) / n_options

    for agent in range(n_agents):
        if params["starBN"]:
            pass
            # belief_network = {
            #     (dim1, dim2): (
            #         params["initial_w"] if dim1 == focal or dim2 == focal else 0
            #     )
            #     for dim1, dim2 in edge_list
            # }
            edgeweights = params["initial_w"] * np.array(
                [1 if focal in edge else 0 for edge in edge_list]
            )
        else:
            # belief_network = {
            #     (dim1, dim2): params["initial_w"] for dim1, dim2 in edge_list
            # }
            edgeweights = params["initial_w"] * np.ones(len(edge_list))
        # initialise beliefs
        opinion_vector = np.random.choice(
            belief_options, size=len(belief_dimensions), p=prob_options
        )  # assuming belief_options = [0,1,...9]

        agent_dict[agent] = {
            "id": agent,
            "x": opinion_vector,
            "del_b": np.zeros(len(belief_dimensions)),
            "del_b_past": [np.zeros(len(belief_dimensions))],
            "edgeweights": edgeweights,
            "neighbours": A_sparse[agent].indices,
        }

    return agent_dict


# # VISUALISE EXAMPLE SOCIAL NETWORK
# M = 2
# params = {
#     "n_agents": 100,
#     "belief_dimensions": [0, 1, 2, 3, 4],
#     "link_prob": 0.1,
#     "initial_w": 1,
#     "focal_dim": 0,
#     "seed": 2,
#     "starBN": False,
#     "memory": 3,
#     "belief_options": np.arange(-1, 1.01, 0.1),
# }
# params["edge_list"] = np.array(list(combinations(params["belief_dimensions"], 2)))
# agent_dict = initialise_agents_and_network(
#     params,
# )
# import networkx as nx

# G = nx.from_dict_of_lists(
#     {i: agent_dict[i]["neighbours"] for i in range(params["n_agents"])}
# )


def update_edge_weights(
    agent_edgeweights,
    del_beliefs,
    edge_list,
    eps,
    lam,
):
    """Update all edge weights for an agent."""
    for n, edge in enumerate(edge_list):
        i, j = edge
        delta = (
            # hebbianV(weight, del_beliefs[i], del_beliefs[j], eps)
            eps * del_beliefs[i] * del_beliefs[j]
            # + delta_social
            - lam * agent_edgeweights[n]
        )
        agent_edgeweights[n] += delta
    return agent_edgeweights


def update_beliefs(
    agent_beliefs,
    agent_edgeweights,
    social_beliefs,
    edge_list,
    belief_dimensions,
    focal_dim,
    belief_options,
    rho,
    beta,
    external_pressure=(None, 0),
):
    """Update all belief values for an agent."""
    # belief_dimensions = params["belief_dimensions"]
    # focal_dim = params["focal_dim"]
    # belief_options = params["belief_options"]

    # x = agent["x"]
    # belief_network = agent["BN"]
    agent_prior_beliefs = np.copy(agent_beliefs)
    belief_dimensions_shuffled = list(belief_dimensions)
    np.random.shuffle(belief_dimensions_shuffled)

    for dim in belief_dimensions_shuffled:
        # Get social beliefs for focal dim only
        # if dim == focal_dim and len(neighbours) > 0:
        #     social_beliefs = [agent_dict[ag]["x"][dim] for ag in neighbours]
        # else:
        #     social_beliefs = np.array([])
        # Update belief using Glauber probabilities
        ps = glauber_probabilities_withSocial(
            dim,
            agent_beliefs,  # should this be x (one belief updated afte rthe next) or x_prior (all simultaneous)? DECISION: use x here, not x_prior (see 2025-12-18)
            agent_edgeweights,
            edge_list,
            social_beliefs if dim == focal_dim else np.array([]),
            belief_options,
            rho,
            beta,
            external_pressure=external_pressure,
        )
        agent_beliefs[dim] = np.random.choice(belief_options, p=ps)
    belief_trend = agent_beliefs - agent_prior_beliefs
    return agent_beliefs, belief_trend


# def update_belief_trend(agent, x_prior, params):
#     """Update agent belief_trend."""
#     # belief_dimensions = params["belief_dimensions"]
#     # x = agent["x"]
#     # belief_trend = np.array([x[dim] - x_prior[dim] for dim in belief_dimensions])

#     agent["del_b_past"].append(belief_trend)
#     if len(agent["del_b_past"]) > params["memory"]:
#         # forget oldest del_b vector.
#         agent["del_b_past"] = agent["del_b_past"][1:]
#     new_del_b = np.mean(agent["del_b_past"], axis=0)
#     agent["del_b"] = dict(zip(belief_dimensions, new_del_b))


def take_snapshot(t, agent_dict, params):
    """Take snapshot of current state."""
    edge_list = params["edge_list"]
    belief_dimensions = params["belief_dimensions"]
    focal_dim = params["focal_dim"]
    curr_external_pressure = (
        params["external_pressure"] if t in params["external_event_times"] else None
    )
    curr_external_pressure_strength = (
        params["external_pressure_strength"]
        if t in params["external_event_times"]
        else 0
    )
    agent_ids = sorted(agent_dict.keys())
    snapshot = []
    for ag in agent_ids:
        agent = agent_dict[ag]
        row = [t, agent["id"], "none"]
        row.extend(agent["edgeweights"])
        row.extend(agent["x"])
        row.extend(
            [
                ("" if curr_external_pressure is None else curr_external_pressure),
                curr_external_pressure_strength,
            ]
        )
        row.extend(
            [np.nan, np.nan]
        )  # 1st arg could be filled with list of agent neighbours for t=0, 2nd will be filled with number of social neighbours
        full_personal_energy = np.sum(
            [
                personal_energy(
                    agent["x"],
                    dim,
                    agent["edgeweights"],
                    edge_list,
                    (curr_external_pressure if dim == focal_dim else None),
                    (curr_external_pressure_strength if dim == focal_dim else 0),
                )
                for dim in belief_dimensions
            ]
        )
        focal_personal_energy = personal_energy(
            agent["x"],
            focal_dim,
            agent["edgeweights"],
            edge_list,
            curr_external_pressure,
            curr_external_pressure_strength,
        )
        neighbours = agent["neighbours"]
        if len(neighbours) > 0:
            social_beliefs = [agent_dict[ag]["x"][focal_dim] for ag in neighbours]
        else:
            social_beliefs = np.array([])

        full_social_energy = social_energy(
            agent["x"][focal_dim], social_beliefs, params["rho"]
        )
        row.extend([full_personal_energy, focal_personal_energy, full_social_energy])
        snapshot.append(row)
    return snapshot


def run_simulation(params):
    """Run single simulation with given parameters."""
    np.random.seed(params["seed"])
    edge_list = params["edge_list"]
    belief_dimensions = params["belief_dimensions"]
    belief_options = params["belief_options"]
    focal_dim = params["focal_dim"]

    # Initialize agents
    agent_dict = initialise_agents_and_network(params)
    snapshots = take_snapshot(0, agent_dict, params)
    for ag in range(params["n_agents"]):
        snapshots[ag][-1] = ""  # json.dumps(list(agent_dict[ag].get("neighbours", [])))
        snapshots[ag][-2] = len(agent_dict[ag].get("neighbours", []))

    # Main simulation loop
    T = params["T"]
    time_steps = np.arange(0, T + 1, 1)
    for t in time_steps[1:]:
        agent_ids_shuffled = list(agent_dict.keys())
        np.random.shuffle(agent_ids_shuffled)

        extEvent = (
            (params["external_pressure"], params["external_pressure_strength"])
            if t in params["external_event_times"]
            else (None, 0)
        )

        for agent_id in agent_ids_shuffled:
            agent = agent_dict[agent_id]
            agent_beliefs = agent["x"]
            agent_edgeweights = agent["edgeweights"]

            # social beliefs
            neighbours = agent["neighbours"]
            if len(neighbours) > 0:
                social_beliefs = [agent_dict[nb]["x"][focal_dim] for nb in neighbours]
            else:
                social_beliefs = np.array([])

            # Updates
            agent_beliefs, belief_trend = update_beliefs(
                agent_beliefs,
                agent_edgeweights,
                social_beliefs,
                edge_list,
                belief_dimensions,
                focal_dim,
                belief_options,
                params["rho"],
                params["beta"],
                external_pressure=extEvent,
            )
            agent["del_b_past"].append(belief_trend)
            if len(agent["del_b_past"]) > params["memory"]:
                # forget oldest del_b vector.
                agent["del_b_past"] = agent["del_b_past"][1:]
            agent["del_b"] = np.mean(agent["del_b_past"], axis=0)

            agent["edgeweights"] = update_edge_weights(
                agent_edgeweights,
                agent["del_b"],
                edge_list,
                params["eps"],
                params["lam"],
            )

        if t in params["track_times"] or t == time_steps[-1]:
            snapshots.extend(take_snapshot(t, agent_dict, params))

    # Convert to DataFrame
    columns = (
        ["time", "agent_id", "identity"]
        + params["edgeNames"]
        + params["belief_dimensions"]
        + ["external_pressure", "external_pressure_strength"]
        + ["n_neighbours", "neighbours"]
        + ["fullpersonalEnergy", "focalpersonalEnergy", "socialEnergy"]
    )
    n_agents = len(agent_dict)
    snapshot_dfs = [
        pd.DataFrame(snapshots[i : i + n_agents], columns=columns)
        for i in range(0, len(snapshots), n_agents)
    ]
    snapshot_df = pd.concat(snapshot_dfs, ignore_index=True)
    snap_avgd = []
    for timeperiod in (
        [(0, 0), (0, 9), (90, 99)]
        + [(i, i) for i in range(90, 100)]
        + [(140, 149), (190, 199), (290, 299)]
    ):
        a = (
            snapshot_df.loc[snapshot_df["time"].between(timeperiod[0], timeperiod[1])]
            .groupby("agent_id", as_index=False)
            .mean(numeric_only=True)
        )
        a["avgeraged"] = timeperiod[0] != timeperiod[1]
        snap_avgd.append(a)

    return pd.concat(snap_avgd)


def generate_filename(params, results_folder):
    social_net = f"(p={params['link_prob']})"
    externalevent = f"_ext-{params['external_event_times'][0]}-{params['external_event_times'][-1]}-on-{params['external_pressure']}-strength{params['external_pressure_strength']}"
    return (
        f"{results_folder}adaptiveBN_M-{params['M']}{'star' if params['starBN'] else ''}-{'depolInitial' if params["depolarisation"] else 'randomInitialOps'}_n-{params['n_agents']}-{social_net}"
        f"_eps{params['eps']}-m{params['memory']}"
        f"_lam{params['lam']}_rho{params['rho']:.2f}_beta{params['beta']}_initialW-{params['initial_w']}"
        f"{externalevent}"
        f"_seed{params['seed']}"
    )


def run_one(seed, eps, mu, lam, initial_w, rho, beta, base_params, results_folder):
    params = base_params.copy()
    params.update(
        {
            "eps": eps,
            "mu": mu,
            "lam": lam,
            "initial_w": initial_w,
            "rho": rho,
            "beta": beta,
            "seed": seed,
        }
    )
    # if seed in list(range(0, 10)):
    #     params["track_times"] = np.arange(0, T + 1, 1)
    print(
        "simulate:",
        f"beta: {beta}",
        f"rho: {rho}",
        f"init_w: {initial_w}",
        f"seed: {seed}",
        f"eps: {eps}",
        f"mu: {mu}",
        f"lam: {lam}",
        f"ext pressure on {params['external_pressure']}",
        f"with s={params['external_pressure_strength']}",
    )
    results = run_simulation(params)
    filename = generate_filename(params, results_folder)
    # #### Here calculate metrics ####
    #
    #
    # TODO
    #
    #
    results.to_csv(filename + ".csv")
    return filename


# Main execution
if __name__ == "__main__":
    # Base parameters
    T = 300
    base_params = {
        "n_agents": 100,
        "belief_options": np.linspace(-1, 1, 21),
        "memory": 1,
        "M": 10,
        "starBN": False,
        "depolarisation": False,
        "focal_dim": 0,
        "link_prob": 10 / 100,
        "T": T,
        "track_times": list(range(0, 10))
        + list(range(90, 101))
        + list(range(140, 151))
        + list(range(190, 201))
        + list(range(290, 301)),
        "external_event_times": list(range(100, 150)),
        "external_pressure": 0,
        # "external_pressure_strength": None,
    }

    # Derived parameters
    base_params["belief_dimensions"] = list(range(10))
    # list(string.ascii_lowercase[: base_params["M"]])
    base_params["edge_list"] = np.array(
        list(combinations(base_params["belief_dimensions"], 2))
    )
    base_params["edgeNames"] = [f"{i}_{j}" for i, j in base_params["edge_list"]]

    # Parameter combinations
    eps_val = 1.0  # 1 lower
    mu_val = 0.0
    initial_w = 0.2  # 1 higher
    rho_val = 1.0 / 3.0  # 1 higher
    lam = 0.005  # 1 higher
    beta = 3.0  # 1 lower, 1 higher
    param_combis = [
        [0.0, 0.0, 0.0, initial_w, rho_val, beta],  # fixed
        [eps_val, mu_val, lam, initial_w, rho_val, beta],  # adaptive
        [0.0, 0.0, 0.0, initial_w, rho_val, 1.0],
        [eps_val, mu_val, lam, initial_w, rho_val, 1.0],
        [0.0, 0.0, 0.0, initial_w, rho_val, 5.0],
        [eps_val, mu_val, lam, initial_w, rho_val, 5.0],
        [eps_val, mu_val, 4 * lam, initial_w, rho_val, beta],  # adaptive
        [0.5 * eps_val, mu_val, lam, initial_w, rho_val, beta],  # adaptive
        [2 * eps_val, mu_val, lam, initial_w, rho_val, beta],  # adaptive
        [0.0, 0.0, 0.0, 2 * initial_w, rho_val, beta],  # fixed
        [eps_val, mu_val, lam, 2 * initial_w, rho_val, beta],  # adaptive
        [0.0, 0.0, 0.0, initial_w, 2 * rho_val, beta],  # fixed
        [eps_val, mu_val, lam, initial_w, 2 * rho_val, beta],  # adaptive
    ]
    ext_strengths = [0, 1, 2, 4, 8, 16]  # 6 exter

    # Results folder
    results_folder = "2025-12-16/sims/"
    if not os.path.isdir(results_folder.split("/")[0]):
        os.mkdir(results_folder.split("/")[0])
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # Run experiments
    # for seed in range(20):
    #    for eps, mu, lam, initial_w in param_combis:

    # Run in parallel
    seeds = np.arange(0, 100)
    param_combis_withSeed = [
        param_combi + [seed] for param_combi in param_combis for seed in seeds
    ]
    for extEvent_strength in ext_strengths:
        base_params["external_pressure_strength"] = extEvent_strength
        Parallel(n_jobs=max(1, multiprocessing.cpu_count() - 2))(
            delayed(run_one)(
                seed, eps, mu, lam, initial_w, rho, beta, base_params, results_folder
            )
            for eps, mu, lam, initial_w, rho, beta, seed in param_combis_withSeed
        )
