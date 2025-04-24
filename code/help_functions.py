# functions

import numpy as np
import pandas as pd
import pingouin
from itertools import combinations
import copy
import string
from scipy.sparse import csr_matrix

def determine_agentlist(filepath, waves, n, atts, seed):
    """determine set of valid participants in the dataset for all wave regimes 

        determine a list of n participants from a panel dataset that ensures that each selected participant has valid answers for the variables in atts in each element of waves. 

    Args:
        filepath (string): path to the panel dataset
        waves (list): a list of wave regimes that contain either a single wave or a list of waves, e.g. waves = [1,19] or waves = [["a", "b"], ["c", "d"]]
        n (int): selected participants 
        atts (list): list of opinion dimensions used to determine whether the participant gave valid answers
        seed (int): random seed

    Returns:
        list: a list of IDs of participants that have valid responses in each element of waves 
    """
    np.random.seed(seed)
    inds = []
    for w in waves:
        df = pd.read_csv(filepath, sep=",", index_col="ID", low_memory=False)
        if "gesis" in filepath:
            renameAtts = dict(zip( ["1-kpx_1090", "kpx_1130", "1-kpx_1210", "kpx_1290", "kpx_1250", "kpx_1590"],  ["econ", "migr", "assim", "clim", "euro", "femin"]))
            attOriginalNames = [c for c in df.columns if (len(c.split(" "))==3) and (c.split(" ")[2][1:-1] in renameAtts.keys()) and not "Imp:" in c]
            df = df.rename(columns=dict(zip(attOriginalNames, [renameAtts[c.split(" ")[2][1:-1]] for c in attOriginalNames[:]] )))
        
        if type(w)==int or type(w)==str:
            # single wave
            df = df.loc[df.wave==w]
        elif type(w)==list:
            # multiple waves. average per individual (level=0, same index) and take the first entry, which should be the same for all ("the first")
            df = df.loc[df.wave.isin(w)]
            for att in atts:
                df[att] = df.reset_index().groupby("ID")[att].mean()
            df = df.loc[~df.index.duplicated(keep='first')]
        
        ops = df[atts].dropna(how="any", axis="index")
        inds.append(ops.index.to_list())
    common = list(set(inds[0]) & set(inds[1]))
    agentlist = np.random.choice(common, size=n, replace=False)
    assert (len(agentlist)==n)
    return agentlist


def initialise(filepath, wave, n, predefined_agentlist, atts, seed=42):
    """initialise agents from a panel dataset

    if n is "all", we use all valid participants in the wave of the panel data 
    if predefined_agentlist is a list of participant IDs, we use those as agents
    if predefined_agentlist is None and n is an integer, we randomly sample agents from the valid participants

    Args:
        filepath (string): path to the panel dataset
        wave (string, int or list): either single wave or list of waves in the dataset
        n (int): number of agents
        predefined_agentlist (None or list of int): None or list of integers
        atts (list): variables in the dataset 
        seed (int, optional): random state, only needed if we sample. Defaults to 42.

    Returns:
        list: list of participant ids
        pandas.Series: weights for each agent
        pandas.DataFrame: opinion values for each attitude in atts for each agent 
        pandas.Series: identity for each agent
    """

    #prepare dataset 
    data = pd.read_csv(filepath, sep=",", index_col="ID", low_memory=False)
    if "gesis" in filepath:
        renameAtts = dict(zip( ["1-kpx_1090", "kpx_1130", "1-kpx_1210", "kpx_1290", "kpx_1250", "kpx_1590"],  ["econ", "migr", "assim", "clim", "euro", "femin"]))
        attOriginalNames = [c for c in data.columns if (len(c.split(" "))==3) and (c.split(" ")[2][1:-1] in renameAtts.keys()) and not "Imp:" in c]
        
        data = data.rename(columns=dict(zip(attOriginalNames, [renameAtts[c.split(" ")[2][1:-1]] for c in attOriginalNames[:]] )))

    if type(wave)==int or type(wave)==str:
        # single wave
        data = data.loc[data.wave==wave]
    elif type(wave)==list:
        # multiple waves. average per individual (level=0, same index) and take the first entry, which should be the same for all ("the first")
        data = data.loc[data.wave.isin(wave)]
        for att in atts:
            data[att] = data.reset_index().groupby("ID")[att].mean()
        data = data.loc[~data.index.duplicated(keep='first')]
    
    ops = data[atts].dropna(how="any", axis="index")
    
    if not n=="all" and predefined_agentlist is not None:
        ops = ops.loc[predefined_agentlist]
    if not n=="all" and predefined_agentlist is None:
        ops = ops.sample(n, replace=False, random_state=seed)
    agentlist = ops.index.tolist()

    # extract identity and weights
    identity = data.loc[agentlist, "partyIdent"]
    if "ess" in filepath:
        weights = data.loc[agentlist, "anweight"]
    else:
        weights = pd.Series(np.ones(len(ops)), index=ops.index)
    return agentlist, weights, ops, identity



hebbian = lambda wij, xi, xj, eps: eps * (1 - np.sign(xi * xj) * wij) * xi *xj

decay = lambda wij, lam: -lam * wij

socialinfluence = lambda wij, Wij, mu: mu * (Wij - wij) if ~np.isnan(Wij) else 0

#dwij = lambda wij, xi, xj, Wij, eps, mu, lam: hebbian(wij, xi, xj, eps) + socialinfluence(wij, Wij, mu) + decay(wij, lam)


def get_socialOmega(agentdict, agentlist, ops, atts, params):
    """
    calculate the social signal (Wij) from the correlation or co-occurence of opinions

    Args:
        agentdict (dict): contains the attributes (incl opinions, neighbours, ...) of all agents
        agentlist (list): contains indices in the dataframe that define the agents 
        ops (pd.DataFrame): contains opinion values for dimensions (atts) and indices (agentlist)
        atts (list): contains the opinion dimensions
        params (dict): contains parameter settings 

    Returns:
        function: function of agent-ID ag that returns the co-occurence or correlation matrix as seen by agent ag. 
    """
    
    signOp = lambda op: 0 if op==0 else op/abs(op)
    
    if params["socInfType"]=="correlation":
        if params["socNetType"]=="observe-all":
            corrNet = pd.DataFrame( ops.corr()  - np.diag(np.ones(len(atts))),  index=atts, columns=atts)
            return lambda ag: corrNet

    elif params["socInfType"]=="co-occurence":
        if params["socNetType"] == "observe-neighbours": 
            observed_signs = {ag: {att: np.array([signOp(ops.loc[nb, att]) for nb in agentdict[ag]["neighbours"] if nb!= ag]).astype(int) for att in atts} for ag in agentlist}
            W_cooc_ag = {}
            for ag in agentlist:
                if len(agentdict[ag]["neighbours"])==0:
                    W_cooc = [(i,j,np.nan) for i,j in combinations(atts, 2)]
                else:
                    W_cooc= [
                        (i, j, np.mean(observed_signs[ag][i] * observed_signs[ag][j]))
                          for i, j in combinations(atts, 2)
                        ]                      
                W_cooc_ag[ag] = pd.DataFrame(W_cooc, columns=["i", "j", "cooccurence"]).pivot_table(index="i", columns="j", values="cooccurence")
            return lambda ag: W_cooc_ag[ag]
        
        elif params["socNetType"] == "observe-all":
            observed_signs = {att: np.array([signOp(ops.loc[ag, att]) for ag in agentlist]) for att in atts}
            W_cooc = [(i, j, np.mean(observed_signs[i]*observed_signs[j])) for i, j in combinations(atts)]
            W_cooc = pd.DataFrame(W_cooc, columns=["i", "j", "cooccurence"]).pivot_table(index="i", columns="j", values="cooccurence")
            return lambda ag: W_cooc
    



import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from itertools import combinations


def initialise_socialnetwork(agent_list, identity, opinions, params):
    """
    Initialise a social network of agents with specified identity, opinions, and network structure.

    Args:
        agent_list (list): List of agent IDs.
        identity (pd.Series): Identity label of each agent indexed by agent ID.
        opinions (pd.DataFrame): Opinion values (rows: agents, columns: atts).
        params (dict): Configuration parameters:
            - "socInfType": Type of social influence ("correlation", "co-occurence", "copy")
            - "socNetType": Network structure type ("observe-all", "observe-neighbours")
            - "parties": List of identity groups. Excluding "none" (non-partisans).
            - "indegree": Expected in-group connection probability.
            - "outdegree": Expected out-group connection probability.
            - "initial_w": Initial weight for belief network edges.
            - "seed": Random seed for reproducibility.

    Returns:
        dict: Agent-wise dictionary containing atts like opinions, identity, belief network, and neighbours.
    """
    #print("Initialising social network...")
    np.random.seed(params["seed"])
    atts = opinions.columns
    soc_inf_type = params["socInfType"]
    soc_net_type = params["socNetType"]
    parties = params["parties"]
    indegree = params["indegree"]
    outdegree = params["outdegree"]
    initial_weight = params["initial_w"]

    if soc_inf_type == "correlation":
        if soc_net_type != "observe-all":
            raise ValueError("Correlation-based influence requires 'observe-all' network type.")
        neighbours = {agent: [] for agent in agent_list}

    elif soc_inf_type in {"co-occurence", "copy"}:
        if soc_net_type == "observe-all":
            neighbours = {agent: [] for agent in agent_list}

        elif soc_net_type == "observe-neighbours":
            n_agents = len(agent_list)
            group_sizes = [(identity == party).sum() for party in parties]
            n_partisan = sum(group_sizes)

            # Account for none, nan, other group identities
            group_sizes.append((identity == "none").sum() + identity.isna().sum() + (identity == "other").sum())

            assert sum(group_sizes) == n_agents, "Group size mismatch."

            # Generate base adjacency matrix for non partisan agents
            A = (np.random.random((n_agents, n_agents)) <= ((indegree+outdegree) / n_agents)).astype(bool)

            # change to out-group degree for partisan agents
            if n_partisan>0:
                A[:n_partisan, :n_partisan] = (np.random.random((n_partisan, n_partisan)) <= ((outdegree) / n_partisan)).astype(bool)

                # Create in-group ties
                start_idx = 0
                for group_size in group_sizes[:-1]:
                    end_idx = start_idx + group_size
                    A[start_idx:end_idx, start_idx:end_idx] = (
                        np.random.random((group_size, group_size)) <= (indegree / group_size)
                    )
                    start_idx = end_idx

            # Sort agent list for group order
            agent_list = [
                agent for party in parties for agent in agent_list if identity[agent] == party
            ] + [
                agent for agent in agent_list if identity[agent]=="none" or identity[agent] == "other" or pd.isna(identity[agent])
            ]
            
            A_sparse = csr_matrix(np.triu(A, k=1))
            neighbours = {
                agent: [agent_list[nb] for nb in A_sparse[i].indices]
                for i, agent in enumerate(agent_list)
            }

        else:
            raise ValueError(f"Unsupported network type: {soc_net_type}")
    else:
        raise ValueError(f"Unsupported influence type: {soc_inf_type}")

    # Build agent dictionary
    agent_dict = {}
    for agent in agent_list:
        belief_network = {
            (att1, att2): initial_weight for att1, att2 in combinations(atts, 2)
        }
        opinion_vector = opinions.loc[agent].to_dict()

        agent_dict[agent] = {
            "x": opinion_vector,
            "identity": identity[agent],
            "BN": belief_network,
            "coherence": 0,
            "nodeCentrality": [
                sum(w for edge, w in belief_network.items() if attr in edge)
                for attr in atts
            ],
            "neighbours": neighbours[agent],
        }

    return agent_dict

# PLOT THE NETWORK

# for the Netherlands identities
# pcols = dict(zip(["far left", "centre left", "centre", "centre right", "far right", "none", "other", np.nan], ["pink", "red", "darkgreen", "k", "darkblue", "grey", "grey", "grey"]))
# G = nx.from_scipy_sparse_array(A_sparse)
# nx.set_node_attributes(G, identity.loc[agent_list].reset_index()["partyIdent"].map(pcols), "col")

# pos = nx.spring_layout(G)
# nx.draw_networkx_nodes(G, pos, node_size=5, node_color=[mpl.colors.to_rgb(G.nodes[n]["col"]) for n in G.nodes], nodelist= G.nodes())
# nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.2)
# plt.show()


#################################
#####  coherence    #####
#################################

def coherence(df, atts):
    coherence_edges = [df[i]*df[j]*df[f"({i},{j})"] for i,j in list(combinations(atts, 2))]
    df["coherence"] = np.sum(coherence_edges, axis=0)
    return df 

def coherenceOther(bnEdges, opinion):
    """calculate the coherence of a set of opinions given a personal belief network

    Args:
        bnEdges (pd.Series or dict): a dictionary containing the edge weights of the personal belief network for edges "(a,b)" (keys/index). 
        opinion (dict): a dictionary containing the opinion for which to evaluate the coherence

    Returns:
        float: summed coherence of the opinion given the personal belief network
    """
    atts = opinion.keys() if type(opinion)==dict else (opinion.index if type(opinion)==pd.Series else None)
    coherence_edges = [opinion[i]*opinion[j]*bnEdges[f"({i},{j})"] for i,j in list(combinations(atts, 2))]
    return np.sum(coherence_edges, axis=0)


#################################
#####  Probabilities for opinion jump   #####
#################################

def glauber_probabilities(att, options, beliefs, BN_ag, Temp, atts):
    """
    Compute Glauber transition probabilities for a specific attribute.

    Parameters:
    - att: The attribute being updated.
    - options: Possible values for the attribute.
    - beliefs: Current belief states for all attributes.
    - BN_ag: A dictionary with interaction weights between attribute pairs.
    - Temp: Temperature parameter controlling randomness.
    - atts: List of all attributes.

    Returns:
    - ps: Array of transition probabilities for each option.
    """

    def energy(beliefs):
        return np.sum([
            -BN_ag[(a1, a2)] * beliefs[a1] * beliefs[a2]
            for a1, a2 in combinations(atts, 2)
            if a1 == att or a2 == att
        ])

    # Original energy
    H0 = energy(beliefs)

    # Energy for each option
    H = []
    original_value = beliefs[att]  # Save original to restore later
    for opt in options:
        beliefs[att] = opt
        H.append(energy(beliefs))
    beliefs[att] = original_value  # Restore original value

    delH = np.array(H) - H0
    exp_term = np.exp(-delH / Temp)
    ps = exp_term / np.sum(exp_term)
    
    return ps
