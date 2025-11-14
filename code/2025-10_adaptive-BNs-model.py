"""
Adaptive Belief Networks Model
version 2025-10-23, Peter Steiglechner, steiglechner@csh.ac.at
updates:
simple random network
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


def hebbianV(wij, delb_i, delb_j, epsV):
    """Hebbian learning from consistency/inconsistency of belief trends"""
    return epsV * delb_i * delb_j


def socialconformity(wij, Wij, mu):
    """Social conformity on BN edge weights."""
    return mu * (Wij - wij) if ~np.isnan(Wij) else 0


def decay(wij, lam):
    """Decay term for edge weights."""
    return -lam * wij


def social_energy(belief, social_beliefs, social_edge_weight):
    """Calculate social energy for a belief value."""
    return np.sum([-social_edge_weight * belief * sb for sb in social_beliefs])


def energy(
    beliefs,
    att,
    atts,
    BN_ag,
    intervention_att=None,
    intervention_strength=None,
    intervention_val=None,
):
    """Calculate personal energy for a specific attribute."""
    H = np.sum(
        [
            -BN_ag[(a1, a2)] * beliefs[a1] * beliefs[a2]
            for a1, a2 in combinations(atts, 2)
            if a1 == att or a2 == att
        ]
    )
    if intervention_att is not None:
        H += -intervention_strength * beliefs[intervention_att] * intervention_val
    return H


def glauber_probabilities_withSocial(
    att,
    options,
    beliefs,
    BN_ag,
    atts,
    social_beliefs,
    social_edge_weight,
    beta_pers=1,
    beta_soc=1,
    intervention_att=None,
    intervention_strength=None,
    intervention_val=None,
):
    """Compute Glauber transition probabilities for a specific attitude."""
    original_value = beliefs[att]
    H0 = energy(
        beliefs,
        att,
        atts,
        BN_ag,
        intervention_att,
        intervention_strength,
        intervention_val,
    )
    H_soc_0 = social_energy(original_value, social_beliefs, social_edge_weight)

    H, H_soc = [], []
    for opt in options:
        beliefs[att] = opt
        H.append(
            energy(
                beliefs,
                att,
                atts,
                BN_ag,
                intervention_att,
                intervention_strength,
                intervention_val,
            )
        )
        H_soc.append(social_energy(opt, social_beliefs, social_edge_weight))

    beliefs[att] = original_value  # restore original value

    delH = beta_pers * (np.array(H) - H0) + beta_soc * (np.array(H_soc) - H_soc_0)
    exp_term = 1 / (1 + np.exp(delH))
    return exp_term / np.sum(exp_term)


def initialise_socialnetwork(agent_list, opinions, params):
    """Initialize social network of agents with specified structure."""
    np.random.seed(params["seed"])
    atts = params["atts"]
    link_prob = params["link_prob"]
    initial_weight = params["initial_w"]
    n_agents = len(agent_list)
    focal = params["focal_att"]

    # Generate adjacency matrix
    A = np.triu((np.random.random((n_agents, n_agents)) <= link_prob).astype(bool), k=1)
    A = A + A.T
    A_sparse = csr_matrix(A)
    neighbours = {
        agent: [agent_list[nb] for nb in A_sparse[i].indices]
        for i, agent in enumerate(agent_list)
    }

    # Build agent dictionary
    agent_dict = {}
    for n, agent in enumerate(agent_list):
        if params["starBN"]:
            belief_network = {
                (att1, att2): initial_weight if att1 == focal or att2 == focal else 0
                for att1, att2 in combinations(atts, 2)
            }
        else:
            belief_network = {
                (att1, att2): initial_weight for att1, att2 in combinations(atts, 2)
            }
        opinion_vector = dict(zip(atts, opinions[n, :]))

        agent_dict[agent] = {
            "id": agent,
            "x": opinion_vector,
            "del_b": dict(zip(atts, np.zeros(len(atts)))),
            "del_b_past": [[0 for _ in atts] for _ in range(params["memory"])],
            # "identity": identity[agent],
            "BN": belief_network,
            "neighbours": neighbours[agent],
        }

    return agent_dict


# agent_ids = list(range(100))
# M = 2
# params = {
#     "atts": ["a", "b"],
#     "clusters": ["A", "B"],
#     "link_prob": 0.8,
#     "initial_w": 1,
#     "focal_att": "a",
#     "seed": 2,
#     "starBN": False,
#     "memory": 3,
# }
# agent_dict = initialise_socialnetwork(
#     agent_ids,
#     {i: "A" if i < 10 else "B" for i in agent_ids},
#     np.random.random(size=(len(agent_ids), M)) * 2 - 1,
#     params,
# )
# import networkx as nx
# G = nx.from_dict_of_lists({i: agent_dict[i]["neighbours"] for i in agent_ids})


def update_edge_weights(agent, agent_dict, edge_list, epsV, mu, lam, dt):
    """Update all edge weights for an agent."""
    belief_network = agent["BN"]
    del_b = agent["del_b"]

    edges = list(edge_list)
    np.random.shuffle(edges)

    for edge in edges:
        i, j = edge
        weight = belief_network[edge]

        # Get social weight from random neighbor
        neighbours = agent.get("neighbours", [])
        if neighbours:
            neighbor = np.random.choice(neighbours)
            social_weight = agent_dict[neighbor]["BN"].get(edge, 0)
        else:
            social_weight = np.nan

        # Calculate weight change
        delta = hebbianV(weight, del_b[i], del_b[j], epsV) + decay(weight, lam)
        if not np.isnan(social_weight):
            delta += socialconformity(weight, social_weight, mu)

        belief_network[edge] = weight + dt * delta


def update_beliefs(
    agent,
    agent_dict,
    atts,
    focal_att,
    belief_options,
    social_edge_weight,
    beta_pers=1,
    intervention=(False, {}),
):
    """Update all belief values for an agent."""
    x = agent["x"]
    belief_network = agent["BN"]
    neighbours = agent.get("neighbours", [])
    interv_att, interv_strength, interv_val = (
        intervention["att"],
        intervention["strength"],
        intervention["val"],
    )
    x_prior = {att: x[att] for att in atts}

    atts_shuffled = list(atts)
    np.random.shuffle(atts_shuffled)

    for att in atts_shuffled:
        # Get social beliefs for focal attribute only
        social_beliefs = (
            [agent_dict[ag]["x"][att] for ag in neighbours] if att == focal_att else []
        )
        # Update belief using Glauber probabilities
        ps = glauber_probabilities_withSocial(
            att,
            belief_options,
            x,  # should this be x_prior???
            belief_network,
            atts,
            social_beliefs,
            social_edge_weight,
            beta_pers=beta_pers,
            beta_soc=1,
            intervention_att=interv_att,
            intervention_strength=interv_strength,
            intervention_val=interv_val,
        )
        x[att] = np.random.choice(belief_options, p=ps)

    return x_prior


def update_belief_trend(agent, x_prior, atts, memory):
    """Update agent belief_trend."""
    x = agent["x"]
    belief_trend = [x[att] - x_prior[att] for att in atts]

    agent["del_b_past"].append(belief_trend)
    if len(agent["del_b_past"]) > memory:
        agent["del_b_past"] = agent["del_b_past"][1:]

    new_del_b = np.array(agent["del_b_past"]).mean(axis=0)
    agent["del_b"] = dict(zip(atts, new_del_b))


def take_snapshot(t, agent_dict, atts, edge_list, intervention):
    """Take snapshot of current state."""
    agent_ids = sorted(agent_dict.keys())
    return [
        [t, agent_dict[ag]["id"], "none"]
        + [agent_dict[ag]["BN"][e] for e in edge_list]
        + [agent_dict[ag]["x"][att] for att in atts]
        + [intervention["att"], intervention["strength"], intervention["val"]]
        + [np.nan]  # will be filled with list of agent neighbours for t=0
        for ag in agent_ids
    ]


def run_simulation(params):
    """Run single simulation with given parameters."""
    np.random.seed(params["seed"])

    # Unpack key parameters
    n, T, dt = params["n"], params["T"], params["dt"]
    atts, edge_list = params["atts"], params["edge_list"]
    belief_options = np.array(params["belief_options"]).flatten()
    epsV, mu, lam = params["epsV"], params["mu"], params["lam"]
    memory, focal_att = params["memory"], params["focal_att"]
    social_edge_weight = params["social_edge_weight"]
    # clusters = params["clusters"]
    intervention_dict = {
        "att": params["intervention_att"],
        "strength": params["intervention_strength"],
        "val": params["intervention_val"],
    }
    no_intervention_dict = {"att": None, "strength": None, "val": None}
    intervention_period = params["intervention_period"]

    # Initialize agents
    agent_ids = list(range(n))

    n_options = len(belief_options)
    prob_options = np.ones(n_options) / n_options
    opinions = np.array(
        [
            np.random.choice(belief_options, size=(len(atts)), p=prob_options)
            for i in agent_ids
        ]
    )

    agent_dict = initialise_socialnetwork(agent_ids, opinions, params)

    # Initial snapshot
    snapshots = take_snapshot(
        0, agent_dict, atts, edge_list, intervention=no_intervention_dict
    )
    for n, ag in enumerate(agent_dict):
        snapshots[n][-1] = json.dumps(agent_dict[ag].get("neighbours", []))

    # Main simulation loop
    time_steps = np.arange(0, T + 1, dt)
    for t in time_steps[1:]:
        agent_ids_shuffled = list(agent_dict.keys())
        np.random.shuffle(agent_ids_shuffled)
        curr_intervention_params = (
            intervention_dict if t in intervention_period else no_intervention_dict
        )

        for agent_id in agent_ids_shuffled:
            agent = agent_dict[agent_id]

            # Update beliefs
            x_prior = update_beliefs(
                agent,
                agent_dict,
                atts,
                focal_att,
                belief_options,
                social_edge_weight,
                beta_pers=params["beta_pers"],
                intervention=curr_intervention_params,
            )

            # Update belief_trend
            update_belief_trend(agent, x_prior, atts, memory)

            # Update edge weights
            update_edge_weights(agent, agent_dict, edge_list, epsV, mu, lam, dt)

        # Take snapshot if needed
        if t in params["track_times"] or t == time_steps[-1]:
            snapshots.extend(
                take_snapshot(
                    t,
                    agent_dict,
                    atts,
                    edge_list,
                    intervention=curr_intervention_params,
                )
            )

    # Convert to DataFrame
    columns = (
        ["time", "agent_id", "identity"]
        + params["edgeNames"]
        + atts
        + ["intervention_att", "intervention_strength", "intervention_val"]
        + ["neighbours"]
    )
    n_agents = len(agent_dict)
    snapshot_dfs = [
        pd.DataFrame(snapshots[i : i + n_agents], columns=columns)
        for i in range(0, len(snapshots), n_agents)
    ]

    return pd.concat(snapshot_dfs, ignore_index=True)


def generate_filename(params, results_folder):
    """Generate filename for results."""
    social_net = f"(p={params['link_prob']})"
    intervention = (
        f"_noIntervention"
        if params["intervention_att"] is None
        else f"_interv{params['intervention_period'][0]}-{params['intervention_period'][-1]}-{params['intervention_att']}-strength{params['intervention_strength']}-value{params['intervention_val']}"
    )
    return (
        f"{results_folder}adaptiveBN_M-{params['M']}{'star' if params['starBN'] else ''}-{'depolInitial' if params["depolarisation"] else 'randInitial'}_n-{params['n']}-{social_net}"
        f"_epsV{params['epsV']}-m{params['memory']}_mu{params['mu']}"
        f"_lam{params['lam']}_rho{params['social_edge_weight']}_beta{params['beta_pers']}_initialW-{params['initial_w']}"
        f"{intervention}"
        f"_seed{params['seed']}"
    )


def run_one(seed, epsV, mu, lam, initial_w, rho, base_params, results_folder):
    params = base_params.copy()
    params.update(
        {
            "epsV": epsV,
            "mu": mu,
            "lam": lam,
            "initial_w": initial_w,
            "seed": seed,
            "social_edge_weight": rho,
        }
    )
    if seed in list(range(0, 20)):
        params["track_times"] = np.arange(0, T + 1, 1)
    print(
        "simulate:",
        epsV,
        mu,
        lam,
        initial_w,
        "seed",
        seed,
        params["intervention_att"],
        params["intervention_strength"],
    )
    results = run_simulation(params)
    filename = generate_filename(params, results_folder)
    results.to_csv(filename + ".csv")
    return filename  # optional: return what was saved


# Main execution
if __name__ == "__main__":
    # Base parameters
    T = 200
    base_params = {
        "n": 100,
        "belief_options": np.linspace(-1, 1, 21),
        "social_edge_weight": None,
        "memory": 1,
        "beta_pers": 3.0,
        "M": 10,
        "starBN": False,
        "depolarisation": False,
        "focal_att": "a",
        # "clusters": ["A", "B"],
        # "withinCluster_link_prob": 0.4,  # approx 0.3 * 100* 99 / 2 links --> 14.85 links per agent
        # "betweenCluster_link_prob": 0.01,  # approx 100*100*0.01/2/ links = 100/100 --> every agent on average one link!
        "link_prob": 10 / 100,
        "T": T,
        "dt": 1,
        "track_times": [0]
        + list(range(90, 101))
        + list(range(140, 151))
        + list(range(190, 201)),
        "intervention_period": [],
        "intervention_att": None,
        "intervention_strength": None,
        "intervention_val": None,
    }

    # Derived parameters
    base_params["atts"] = list(string.ascii_lowercase[: base_params["M"]])
    base_params["edge_list"] = list(combinations(base_params["atts"], 2))
    base_params["edgeNames"] = [f"({i},{j})" for i, j in base_params["edge_list"]]

    # Parameter combinations
    epsV_val = 1.0
    mu_val = 0.0
    initial_w = 0.2
    rho_val = 1.0
    lam = 0.005

    param_combis = [
        # (0.0, 0.0, 0.0, 3*initial_w, rho_val),
        [0.0, 0.0, 0.0, initial_w, rho_val],
        [epsV_val, mu_val, lam, initial_w, rho_val],
        # larger eps
        # (0.6, mu_val, lam, initial_w, rho_val),
        # # smaller eps
        # (0.2, mu_val, lam, initial_w, rho_val),
        # # larger initial_w
        # (0.0, 0.0, 0.0, 0.6, rho_val),
        # (epsV_val, mu_val, lam, 0.6, rho_val),
        # # smaller initial_w
        # (0.0, 0.0, 0.0, 0.2, rho_val),
        # (epsV_val, mu_val, lam, 0.2, rho_val),
        # # smaller rho
        # (0.0, 0.0, 0.0, initial_w, 0.2),
        # (epsV_val, mu_val, lam, initial_w, 0.2),
        # # larger rho
        # (0.0, 0.0, 0.0, initial_w, 0.4),
        # (epsV_val, mu_val, lam, initial_w, 0.4),
    ]
    interventions = [
        # ([], None, None, None),
        (range(100, 150), "a", 2, 1),
        # (range(100, 150), "a", 4, 1),
        (range(100, 150), "a", 8, 1),
        # (range(100, 150), "b", 2, 1),
        # (range(100, 150), "b", 5, 1),
    ]

    # Results folder
    results_folder = "2025-11-02/sims/"
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # Run experiments
    # for seed in range(20):
    #    for epsV, mu, lam, initial_w in param_combis:

    # Run in parallel
    seeds = np.arange(60, 130)
    param_combis_withSeed = [
        param_combi + [seed] for param_combi in param_combis for seed in seeds
    ]
    for period, i_att, i_stren, i_val in interventions:
        base_params.update(
            {
                "intervention_period": period,
                "intervention_att": i_att,
                "intervention_strength": i_stren,
                "intervention_val": i_val,
            }
        )
        Parallel(n_jobs=max(1, multiprocessing.cpu_count() - 2))(
            delayed(run_one)(
                seed, epsV, mu, lam, initial_w, rho, base_params, results_folder
            )
            for epsV, mu, lam, initial_w, rho, seed in param_combis_withSeed
        )
