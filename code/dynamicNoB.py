# This file contains the very simple Network of Beliefs

import numpy as np
import pandas as pd
#import pingouin # for partial correlation: pcorr
from itertools import combinations
#import copy
import string
from help_functions import get_socialOmega, initialise_socialnetwork, glauber_probabilities
from help_functions import hebbian, socialinfluence, decay

#################################
#####  DYNAMIC MODEL   #####
#################################
def update_step(t, agent_id, agentdict, atts, Wij, params, **kwargs):
    eps, mu, lam, dt = params["eps"], params["mu"], params["lam"], params["dt"]
    socialInfl_type = params["socInfType"]
    network_type = params["socNetType"]
    belief_jump, temperature = params["belief_jump"], params["Temp"]

    agent = agentdict[agent_id]
    x = agent["x"]
    belief_network = agent["BN"]

    edgelist = list(belief_network.keys())
    np.random.shuffle(edgelist)

    for edge in edgelist:
        i, j = edge
        weight_ij = belief_network[edge]

        # Decay + Hebbian
        delta_beta = decay(weight_ij, lam) + hebbian(weight_ij, x[i], x[j], eps)

        # Social Influence
        if socialInfl_type == "copy":
            if network_type == "observe-all":
                neighbours = list(agentdict.keys())
            else:
                neighbours = agent.get("neighbours", [])
            
            if neighbours:
                sampled_neighbour = np.random.choice(neighbours)
                curr_weight = agentdict[sampled_neighbour]["BN"].get(edge, 0)
            else:
                curr_weight = np.nan
        else:
            curr_weight = 0 if Wij(agent_id).empty else Wij(agent_id).loc[i, j]

        if ~np.isnan(curr_weight): 
            delta_beta += socialinfluence(weight_ij, curr_weight, mu)

        # Update weight with clipping
        belief_network[edge] = np.clip(weight_ij + dt * delta_beta, -1, 1)

    # NODE UPDATING
    items = list(x.items())
    np.random.shuffle(items)
    for att, b in items:
        # options are b-0.1, b, b+0.1
        options = [max(-1, b - belief_jump), b, min(1, b + belief_jump)]
        ps = glauber_probabilities(att, options, x, belief_network, temperature, atts)
        x[att] = np.random.choice(options, p=ps)
 
    # Save updates
    agent["BN"] = belief_network
    agent["x"] = x

    # Tracking
    # if t in params["track_times"]:
    #     edge_weights = np.array([belief_network[e] for e in edgelist])
    #     coherence_vals = edge_weights * np.array([x[i] * x[j] for i, j in edgelist])
    #     # Node-level coherence
    #     coherence_by_node = {
    #         node: coherence_vals[[node in edge for edge in edgelist]].sum()
    #         for node in atts
    #     }
    #     agent["coherence"] = sum(coherence_by_node.values())
    #     agent["nodeCentrality"] = [
    #         sum(abs(belief_network[e]) for e in edgelist if node in e) for node in atts
    #     ]

    return agentdict

    
#%% 
#################################
#####  Simulation Run   #####
#################################
def dynSim_NoB(agent_ids, atts, params):
    """
    Run a dynamic simulation of adaptive NoB. 

    Opinions are initialised random between -1 and 1. 
    All identities are set to none

    Args:
        agent_ids (list): List of agent identifiers.
        atts (list): Names of belief dimensions (opinion attributes).
        params (dict): Configuration parameters including:
            - "eps" (float): Strength of Hebbian Learning
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
            - "Temp" (float): Temperature for belief updating
            - "belief_jump" (float): step of considered belief change 

    Returns:
        simOut (pd.DataFrame): Final belief network weights and opinions for each agent.
        snapshots (pd.DataFrame): Belief states recorded at specified time steps.
        agent_ids (list): Ordered list of agent IDs (original order).
    """
    np.random.seed(params["seed"])

    # Initialize opinions randomly in [-1, 1]
    opinions = pd.DataFrame(
        [np.random.uniform(-1, 1, size=len(atts)) for agent in agent_ids],
        index=agent_ids, columns=atts
    )
    identity = pd.Series("none", index=agent_ids)

    # Initialize agent network
    agent_dict = initialise_socialnetwork(agent_ids, identity, opinions, params)
    agent_ids = list(agent_dict.keys())
    original_agent_ids = list(agent_ids)  # Ensure a true copy of the initial order

    print("simulate", end="...")
    time_steps = np.arange(0, params["T"], step=params["dt"])
    results_over_time = []

    # Main simulation loop (starts at time[1] to skip t=0)
    for t in time_steps[1:]:
        if t % 10 == 0:
            print(t, end=", ")

        # Set up social influence matrix if needed
        socInfType = params["socInfType"]
        if socInfType in {"co-occurence", "correlation"}:
            Wij = get_socialOmega(agent_dict, agent_ids, opinions, atts, params)
        else:
            Wij = None

        np.random.shuffle(agent_ids)

        for ag in agent_ids:
            agent_dict = update_step(t, ag, agent_dict, atts, Wij, params)

        # Record results at tracked times
        if t in params["track_times"] or t==time_steps[-1]:
            snapshot = [
                [agent_dict[ag]['BN'][e] for e in agent_dict[ag]['BN']] +
                [agent_dict[ag]['x'][att] for att in atts]
                for ag in original_agent_ids
            ]
            results_over_time.append(snapshot)
    print("done")

    # Create final output DataFrame
    edge_labels = [f"({i},{j})" for i, j in agent_dict[original_agent_ids[0]]['BN']]
    snapshots = []
    for t, res in zip(params["track_times"] , results_over_time):
        snap = pd.DataFrame(
            data=res, # edge weights and beliefs (columns) per agent (row)
            columns = edge_labels + atts, index=original_agent_ids)
        snap.loc[original_agent_ids, "agent_weight"] = pd.Series(np.ones(len(original_agent_ids)), index=original_agent_ids)
        snap.loc[original_agent_ids, "identity"] = identity.loc[original_agent_ids]
        snap["t"] = t
        snap = snap.reset_index(names="agent_id")
        snapshots.append(snap)
    snapshots_df = pd.concat(snapshots)

    return snapshots[-1], snapshots_df, original_agent_ids


# %%
#################################
#####  MAIN   #####
#################################
if __name__=="__main__":

    params = {
        "seed":1,
        "n": 100,
        "T":100,
        "dt":1,
        "M": 4, # number of beliefs
        "track_times": np.arange(0,100, 1),
        "socNetType":"observe-neighbours",  # observe-neighbours or observe-all
        "socInfType":None,   # correlation or co-occurence or copy
        "eps":None,
        "mu":None, 
        "lam":None,
        "parties": [], 
        "indegree":10, 
        "outdegree":0,
        "initial_w":0.,
        "belief_jump":0.1,
        "Temp":None,
    }
    atts = list(string.ascii_lowercase[:params["M"]])
    edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
    
    agentlist = list(range(params["n"]))        
    
    # base scenario:
    # (0.4,0.2,0.05, Temp, "copy", "observe-neighbours") where Temp = 0.01, 0.1, 1, 10
    #     
    paramCombis = [
        # eps, mu, lam, Temp, socInfType, socNetType
        (0.4,0.2,0.05, 0.01, "copy", "observe-neighbours"), 
        (0.4,0.2,0.05, 0.1, "copy", "observe-neighbours"), 
        (0.4,0.2,0.05, 1, "copy", "observe-neighbours"),
        (0.4,0.2,0.05, 10, "copy", "observe-neighbours"),
        # EDGES FIXED
        (0.0,0.0,0.00, 0.01, "co-occurence", "observe-neighbours"),
        (0.0,0.0,0.00, 0.1, "co-occurence", "observe-neighbours"),
        (0.0,0.0,0.00, 1, "co-occurence", "observe-neighbours"),
        (0.0,0.0,0.00, 10, "co-occurence", "observe-neighbours"),
        # CO-OCCURENCE
        (0.4,0.2,0.05, 0.01, "co-occurence", "observe-neighbours"),
        (0.4,0.2,0.05, 0.1, "co-occurence", "observe-neighbours"),
        (0.4,0.2,0.05, 1, "co-occurence", "observe-neighbours"),
        (0.4,0.2,0.05, 10, "co-occurence", "observe-neighbours"),
        # LESS HEBBIAN
        (0.05,0.2,0.05, 0.01, "copy", "observe-neighbours"), 
        (0.05,0.2,0.05, 0.1, "copy", "observe-neighbours"), 
        (0.05,0.2,0.05, 1, "copy", "observe-neighbours"),
        (0.05,0.2,0.05, 10, "copy", "observe-neighbours"),
        # MORE SOCIAL
        (0.4,0.4,0.05, 0.01, "copy", "observe-neighbours"), 
        (0.4,0.4,0.05, 0.1, "copy", "observe-neighbours"), 
        (0.4,0.4,0.05, 1, "copy", "observe-neighbours"),
        (0.4,0.4,0.05, 10, "copy", "observe-neighbours"), 
        # NO NETWORK
        (0.4,0.2,0.05, 0.01, "copy", "observe-all"), 
        (0.4,0.2,0.05, 0.1, "copy", "observe-all"), 
        (0.4,0.2,0.05, 1, "copy", "observe-all"),
        (0.4,0.2,0.05, 10, "copy", "observe-all"),
        # NO NETWORK + co-occurence
        (0.4,0.2,0.05, 0.01, "co-occurence", "observe-all"), 
        (0.4,0.2,0.05, 0.1, "co-occurence", "observe-all"), 
        (0.4,0.2,0.05, 1, "co-occurence", "observe-all"),
        (0.4,0.2,0.05, 10, "co-occurence", "observe-all"),
        ]
    
    resultsfolder = "results-dynNoB/"
    
    for eps, mu, lam, Temp, socInfType, socNetType in paramCombis:
        params["eps"] = eps
        params["mu"]  = mu
        params["lam"] = lam
        params["Temp"] = Temp 
        params["socInfType"] = socInfType
        params["socNetType"] = socNetType

        simOut, snapshots, original_agent_ids = dynSim_NoB(agentlist, atts, params)
            
        socNetName = f"{socNetType}"+(f"(Stoch-{len(params['parties'])}-Block-{params['indegree']}-{params['outdegree']})" if "neighbours" in socNetType else "")
        
        filename = resultsfolder+f"dynamicNoB_M-{params['M']}_n-{params['n']}_{socInfType}-"+socNetName+f"_beliefJump-{params['belief_jump']}-T{Temp}"+f"_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{params['seed']}"
          
        # Store final results
        simOut.to_csv(filename+".csv")

        # Store results over time        
        snapshots.to_csv(filename+"_overTime.csv")