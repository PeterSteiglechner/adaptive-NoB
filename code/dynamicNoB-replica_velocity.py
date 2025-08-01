# This file contains an extension of the network of beliefs to allow both dynamic node values and dynamic edge weights.
# Note: we disregard here the possibility that social beliefs are not adequately reflected in actual beliefs of others.
# ADD: we use Hebbian learning via activation

# %%
import numpy as np
import pandas as pd
import os

# import pingouin # for partial correlation: pcorr
from itertools import combinations

# import copy
import string
from help_functions import initialise_socialnetwork, glauber_probabilities_withSocial
from help_functions import hebbianV, socialinfluence, decay
import json


#################################
#####  DYNAMIC MODEL   #####
#################################
def update_step(t, agent_id, agentdict, Wij, params, **kwargs):
    epsV, eps, mu, lam, dt = (
        params["epsV"],
        params["eps"],
        params["mu"],
        params["lam"],
        params["dt"],
    )
    socialInfl_type = "copy"  # alternative co-occurence
    # network_type = "observe-neighbours"  # alternative observe-all
    memory = params["memory"]
    focal_att = params["focal_att"]
    atts = params["atts"]
    belief_options = params["belief_options"]
    social_edge_weight = params["social_edge_weight"]

    agent = agentdict[agent_id]
    x = agent["x"]
    v = agent["v"]
    belief_network = agent["BN"]

    order_edges = list(params["edge_list"])
    np.random.shuffle(order_edges)
    for edge in order_edges:
        i, j = edge
        weight_ij = belief_network[edge]

        # Social Influence
        if socialInfl_type == "copy":
            # if network_type == "observe-all":
            #     neighbours = list(agentdict.keys())
            # else:
            neighbours = agent.get("neighbours", [])
            if neighbours:
                sampled_neighbour = np.random.choice(neighbours)
                curr_weight = agentdict[sampled_neighbour]["BN"].get(edge, 0)
            else:
                curr_weight = np.nan
        else:
            social_signal = Wij(agent_id)
            curr_weight = social_signal.loc[i, j] if not social_signal.empty else 0

        # Decay + Hebbian
        delta_beta = hebbianV(weight_ij, v[i], v[j], epsV) + decay(weight_ij, lam)
        if ~np.isnan(curr_weight):
            delta_beta += socialinfluence(weight_ij, curr_weight, mu)

        # Update weight without clipping
        # belief_network[edge] = np.clip(weight_ij + dt * delta_beta, -1, 1)
        belief_network[edge] = weight_ij + dt * delta_beta

    # NODE UPDATING
    x_prior = dict(zip(atts, [x[a] for a in atts]))
    atts_order = list(params["atts"])
    np.random.shuffle(atts_order)
    for att in atts_order:
        if att == focal_att:
            soc_beliefs = [agentdict[ag]["x"][att] for ag in neighbours]
        else:
            soc_beliefs = []
        # options are M points in -1...1
        ps = glauber_probabilities_withSocial(
            att,
            belief_options,
            x,
            belief_network,
            atts,
            soc_beliefs,
            social_edge_weight,
            beta_pers=1,
            beta_soc=1,
        )
        x[att] = np.random.choice(belief_options, p=ps)

    # VELOCITY UPDATING
    agent["velo_past"].append([x[a] - x_prior[a] for a in atts])
    if len(agent["velo_past"]) > memory:
        agent["velo_past"] = agent["velo_past"][1:]
    new_v = np.array(agent["velo_past"]).mean(axis=0)

    # Save updates
    agent["BN"] = belief_network
    agent["x"] = x
    agent["v"] = dict(zip(atts, new_v))

    return agentdict


def snap(t, agent_dict, atts, edge_list):
    agent_ids = sorted(list(agent_dict.keys()))
    return [
        [t, agent_dict[ag]["id"], agent_dict[ag]["identity"]]
        + [agent_dict[ag]["BN"][e] for e in edge_list]
        + [agent_dict[ag]["x"][att] for att in atts]
        + [np.nan]
        for ag in agent_ids
    ]


# %%
#################################
#####  Simulation Run   #####
#################################
def dynSim_NoB(params):
    """
    Run a dynamic simulation of adaptive NoB.

    Opinions are initialised random between -1 and 1.
    All identities are set to none

    Args:
        agent_ids (list): List of agent identifiers.
        atts (list): Names of belief dimensions (opinion attributes).
        params (dict): Configuration parameters including:
            # - "eps" (float): Strength of Hebbian Learning from holding both opinions
            - atts (list):
            - edge_list (list):
            - "epsV" (float): Strength of Hebbian Learning from Activation
            - "mu" (float): Strength of Social Influence
            - "lam" (float): Strength of edge weight regularisation
            - "socInfType" (str): Type of social influence ("correlation", "co-occurence", "copy").
            - "socNetType" (str): Network structure ("observe-all", "observe-neighbours").
            - "parties" (list): Identity groups, excluding "none".
            - "indegree" (float): In-group connection probability.
            - "outdegree" (float): Out-group connection probability.
            - "initial_w" (float): Initial edge weight in belief network.
            - "seed" (int): Random seed for reproducibility.
            - "T" (float): Total time.
            - "dt" (float): Time step.
            - "track_times" (list): Time steps to track results.
            # - "beta_pers" (float): attention to personal dissonance; 1/TempP
            # - "beta_scc" (float): attention for social dissonance; 1/TempS
            - "belief_options" (list): possible belief states
            - "social_edge_weight" (float): fixed edge weight of a social link (for node updating)

    Returns:
        simOut (pd.DataFrame): Final belief network weights and opinions for each agent.
        snapshots (pd.DataFrame): Belief states recorded at specified time steps.
        agent_ids (list): Ordered list of agent IDs (original order).
    """
    np.random.seed(params["seed"])

    atts = params["atts"]
    edge_list = params["edge_list"]
    agent_ids = list(range(0, params["n"]))

    # Initialise opinions randomly in [-1, 1]
    opinions = pd.DataFrame(
        [
            np.random.choice(params["belief_options"], replace=True, size=len(atts))
            for ag in agent_ids
        ],
        index=agent_ids,
        columns=atts,
    )

    # Initialise agent network
    assert (len(agent_ids) % 2) == 0
    group_size = int(len(agent_ids) / 2)
    identity = dict(zip(agent_ids, ["A"] * group_size + ["B"] * group_size))
    agent_dict = initialise_socialnetwork(agent_ids, identity, opinions, params)
    edge_labels = [f"({i},{j})" for i, j in edge_list]
    # original_agent_ids = list(agent_ids)

    print("simulate", end="...")
    time_steps = np.arange(0, params["T"] + 1, step=params["dt"])

    snapshots = [snap(0, agent_dict, atts, edge_list)]
    for n, ag in enumerate(agent_dict):
        snapshots[-1][n][-1] = json.dumps(
            agent_dict[ag].get("neighbours", [])
        )  # store network

    # Main simulation loop (starts at time[1] to skip t=0)
    for t in time_steps[1:]:
        # if t % 10 == 0: print(t, end=", ")

        # Set up social influence matrix if needed
        # socInfType = params["socInfType"]
        # if socInfType in {"co-occurence", "correlation"}:
        #     Wij = get_socialOmega(agent_dict, agent_ids, opinions, atts, params)
        # else:
        #     Wij = None

        np.random.shuffle(agent_ids)
        for ag in agent_ids:
            agent_dict = update_step(t, ag, agent_dict, Wij=None, params=params)

        # Record results at tracked times
        if t in params["track_times"] or t == time_steps[-1]:
            snapshots.append(snap(t, agent_dict, atts, edge_list))
    print("done")

    # Create final output DataFrame
    agent_ids = sorted(list(agent_dict.keys()))
    snaps_df = [
        pd.DataFrame(
            data=snap,
            columns=["time", "agent_id", "identity"]
            + edge_labels
            + atts
            + ["neighbours"],
            index=agent_ids,
        )
        for snap in snapshots
    ]
    df = pd.concat(snaps_df)
    return df


# %%
#################################
#####  MAIN   #####
#################################
if __name__ == "__main__":
    for seed in range(0, 20):
        T = 100
        params = {
            # General Setup
            "n": 100,
            "belief_options": np.linspace(-1, 1, 7),
            "social_edge_weight": 1.0,
            "memory": 3,
            "M": 10,  # number of beliefs
            "focal_att": "a",
            # Init
            "initial_w": None,
            # Edge Dynamics
            "eps": None,
            "epsV": None,
            "mu": None,
            "lam": None,
            # Social Network
            "parties": ["A", "B"],
            "withinClusterP": 0.4,
            "betweenClusterP": 0.01,
            # Belief Dynamics
            # "beta_pers": None,
            # "beta_soc": None,
            # Simulation setup:
            "seed": seed,
            "T": T,
            "dt": 1,
            "track_times": np.arange(0, T + 1, 1),
            # "socNetType":"observe-neighbours",  # observe-neighbours or observe-all
            # "socInfType":None,   # correlation or co-occurence or copy
        }
        params["atts"] = list(string.ascii_lowercase[: params["M"]])
        params["edge_list"] = list(combinations(params["atts"], 2))
        params["edgeNames"] = [f"({i},{j})" for i, j in params["edge_list"]]

        # base scenario:
        # OLD (0.4,0.2,0.05, Temp, "copy", "observe-neighbours") where Temp = 0.01, 0.1, 1, 10
        #
        epsV_val = 0.3
        mu_val = 0.5
        paramCombis = [
            # eps, epsV, mu, lam, initial_w
            (0.0, epsV_val, mu_val, 0.0, 0.8),
            (0.0, epsV_val, mu_val, 0.0, 0.2),
            (0.0, 0.0, 0.0, 0.0, 0.8),
            (0.0, 0.0, 0.0, 0.0, 0.2),
        ]

        resultsfolder = "2025-07_results-dynNoB_velo/"
        if not os.path.isdir(resultsfolder):
            os.mkdir(resultsfolder)
        for eps, epsV, mu, lam, initial_w in paramCombis:
            params["eps"] = eps
            params["epsV"] = epsV
            params["mu"] = mu
            params["lam"] = lam
            params["initial_w"] = initial_w
            # params["socInfType"] = "copy"
            # params["socNetType"] = "observe-neighbours"
            print(eps, epsV, mu, lam, initial_w, seed)

            simOut = dynSim_NoB(params)

            # socNetName = f"{socNetType}"+(f"(Stoch-{len(params['parties'])}-Block-{params['withinClusterP']}-{params['betweenClusterP']})" if "neighbours" in socNetType else "")
            socNetName = f"(Stoch-{len(params['parties'])}-Block-{params['withinClusterP']}-{params['betweenClusterP']})"

            filename = (
                resultsfolder
                + f"dynamicNoB-_M-{params['M']}_n-{params['n']}-"
                + socNetName
                # + f"_beta-p{beta_pers}-s{beta_soc}"
                + f"_epsV{epsV}-m{params['memory']}_eps{eps}_mu{mu}_lam{lam}_rho{params['social_edge_weight']}_initialW-{params['initial_w']}_seed{params['seed']}"
            )

            # Store final results
            simOut.to_csv(filename + ".csv")

            # Store results over time
            # snapshots.to_csv(filename+"_overTime.csv")

# %%
