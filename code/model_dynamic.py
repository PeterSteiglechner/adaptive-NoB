
#%% 

import numpy as np
import pandas as pd
#import pingouin  # for partial correlation: pcorr
from itertools import combinations
import string
from help_functions import *

countryXdataset = {"gesis":"Germany", "liss":"Netherlands", "autnes":"Austria"}
atts_datasets = {
    "gesis": ["econ", "migr", "assim", "clim", "euro", "femin"],
    "liss": ["euth", "inequ", "migrAss", "eu", "nr_mig", "asyl"],
    "autnes":[]
}
parties = {
    "Germany": ["Linke", "Grüne", "SPD", "FDP", "CDU/CSU", "AfD"],# "none",
    "Netherlands": ["far left", "centre left", "centre", "centre right", "far right"], # "none"
    "Austria": ["Grüne", "SPÖ", "NEOS", "ÖVP", "FPÖ"] # "none"
}


#################################
#####  DYNAMIC MODEL   #####
#################################
def update_step(t, agent_id, agentdict, atts, Wij, params, **kwargs):
    eps, mu, lam, dt = params["eps"], params["mu"], params["lam"], params["dt"]
    socialInfl_type = params["socInfType"]
    network_type = params["socNetType"]

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
            delta_beta += socialinfluence(weight_ij, curr_weight, mu)#
        # Update weight with clipping
        belief_network[edge] = np.clip(weight_ij + dt * delta_beta, -1, 1)

    # HERE NO NODE UPDATING. 

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

def dynSimStaticBeliefs(filepath, wave, agentlist, atts, params):
    """
    Run a dynamic simulation (edges) with initialized agent opinion and identity from panel data.

    Args:
        filepath (str): Path to input data.
        wave (int or list): Wave number or list of wave numbers.
        agentlist (list): Optional list of preselected agent IDs.
        atts (list): Names of belief dimensions (opinion attributes).
        params (dict): Configuration parameters including:
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

    Returns:
        simOut (pd.DataFrame): Final belief network weights and opinions for each agent.
        snapshots (pd.DataFrame): Belief states recorded at specified time steps.
        agent_ids (list): Ordered list of agent IDs (original order).
    """
    np.random.seed(params["seed"])

    # Initialize agents
    agent_ids, weights, opinions, identity = initialise(
        filepath, wave, params["n"], agentlist, atts, params["seed"]
    )
    print(f"Nr of agents: {len(agent_ids)} | Identity counts:\n{identity.value_counts()}")

    # Set up the agent social network
    agent_dict = initialise_socialnetwork(agent_ids, identity, opinions, params)
    agent_ids = list(agent_dict.keys())
    original_agent_ids = list(agent_ids)  # Ensure a true copy of the initial order

    # Set up STATIC social influence matrix if needed
    socInfType = params["socInfType"]
    if socInfType in {"co-occurence", "correlation"}:
        Wij = get_socialOmega(agent_dict, agent_ids, opinions, atts, params)
    else:
        Wij = None

    print("simulate", end="...")
    time_steps = np.arange(0, params["T"], step=params["dt"])
    results_over_time = []

    # Main simulation loop (starts at time[1] to skip t=0)
    for t in time_steps[1:]:
        if t % 10 == 0:
            print(t, end=", ")

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
        snap.loc[original_agent_ids, "agent_weight"] = weights.loc[original_agent_ids]
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
    folder = ""
    inputfolder = folder+"inputdata/" 
    dataset = "liss"
    for dataset in ["liss", "gesis"]:
        country= countryXdataset[dataset]
        params={
            "n": 1000,    # integer or "all"
            "seed":1,
            "T":100,
            "dt":1,
            "track_times": np.arange(0,100, 10),
            "socNetType":"observe-neighbours",  # observe-neighbours or observe-all
            "socInfType":"co-occurence",   # correlation or co-occurence or copy
            "eps":None, #filled later
            "mu":None, 
            "lam":None,
            "parties": parties[country], 
            "indegree":10, 
            "outdegree":0,
            "initial_w":0.,
        }
        atts = atts_datasets[dataset]
        edgeNames = [f"({i},{j})" for i,j in list(combinations(atts, 2))]
        # waves defines a set of waves before and after 2021
        waves = {
            "gesis":[list(range(1,20)), list(range(20,28))], 
            "liss":[
                [string.ascii_lowercase[y-8-1] for y in [16,17,18,19,20]], 
                [string.ascii_lowercase[y-8-1] for y in [21,22,23,24]]
                ], 
            "autnes":[],
        }
        
        filepath = f"{inputfolder}{dataset}-{country.lower()}.csv"
        
        agentlist = None if params["n"]=="all" else determine_agentlist(filepath, waves[dataset], params["n"], atts, params["seed"])

        # base scenario:
        # (0.4,0.2,0.05, Temp, "copy", "observe-neighbours") where Temp = None
        #
        paramCombis = [
            # eps, mu, lam, Temp, socInfType, socNetType
            (0.4,0.2,0.05, None, "copy", "observe-neighbours"), 
            ]
        
        resultsfolder = folder+"results/static_"
        for eps, mu, lam, Temp, socInfType, socNetType in paramCombis:
            params["eps"] = eps
            params["mu"]  = mu
            params["lam"] = lam
            params["socInfType"] = socInfType
            params["socNetType"] = socNetType
            

            for wave in waves[dataset]:
                simOut, snapshots, original_agent_ids = dynSimStaticBeliefs(filepath, wave, agentlist, atts, params)
                
                waveName = wave if type(wave)==int else f"{wave[0]}-{wave[-1]}"
                socNetName = f"{socNetType}"+(f"(Stoch-{len(params['parties'])}-Block-{params['indegree']}-{params['outdegree']})" if "neighbours" in socNetType else "")

                filename = resultsfolder+f"dynamicNoB_fromPanelData_{dataset.lower()}-{country}_{waveName}_n-{params['n']}_{socInfType}-"+socNetName+f"_beliefJump-0-T-NA"+f"_eps{eps}_mu{mu}_lam{lam}_initialW-{params['initial_w']}_seed{params['seed']}"

                # Store final results
                simOut.to_csv(filename+".csv")
                # Store results over time
                snapshots.to_csv(filename+"_overTime.csv")


#%%

