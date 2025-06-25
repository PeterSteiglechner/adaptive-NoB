# functions

import numpy as np
import pandas as pd
#import pingouin
from itertools import combinations
import copy
import string
from scipy.sparse import csr_matrix

hebbianV = lambda wij, vi, vj, eps:  eps * vi * vj
# (1 - abs(wij))

hebbian = lambda wij, xi, xj, eps: eps * (1 - np.sign(xi * xj) * wij) * xi *xj

decay = lambda wij, lam: -lam * wij

socialinfluence = lambda wij, Wij, mu: mu * (Wij - wij) if ~np.isnan(Wij) else 0

socialinfluenceMult = lambda wij, Wij, mu: mu * (1 - abs(wij)) * Wij * wij if ~np.isnan(Wij) else 0

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
            W_cooc = [(i, j, np.mean(observed_signs[i]*observed_signs[j])) for i, j in combinations(atts, 2)]
            W_cooc = pd.DataFrame(W_cooc, columns=["i", "j", "cooccurence"]).pivot_table(index="i", columns="j", values="cooccurence")
            return lambda ag: W_cooc
    


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
            - "withinClusterP": in-group link probability.
            - "betweenClusterP": between-group link probability.
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
    withinClusterP = params["withinClusterP"]
    betweenClusterP = params["betweenClusterP"]
    initial_weight = params["initial_w"]
    n_agents = len(agent_list)
    group_sizes = [(identity == party).sum() for party in parties]
    # Generate base adjacency matrix for non partisan agents
    A = (np.random.random((n_agents, n_agents)) <= betweenClusterP).astype(bool)

    # Create in-group ties
    start_idx = 0
    for group_size in group_sizes:
        end_idx = start_idx + group_size
        A[start_idx:end_idx, start_idx:end_idx] = (
            np.random.random((group_size, group_size)) <= withinClusterP
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

    # Build agent dictionary
    agent_dict = {}
    for agent in agent_list:
        belief_network = {
            (att1, att2): initial_weight for att1, att2 in combinations(atts, 2)
        }
        opinion_vector = opinions.loc[agent].to_dict()
        agent_dict[agent] = {
            "x": opinion_vector,
            "v": dict(zip(atts, np.zeros_like(atts))),
            "velo_past":[],
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

# def glauber_probabilities(att, options, beliefs, BN_ag, Temp, atts):
#     """
#     Compute Glauber transition probabilities for a specific attribute.

#     Parameters:
#     - att: The attribute being updated.
#     - options: Possible values for the attribute.
#     - beliefs: Current belief states for all attributes.
#     - BN_ag: A dictionary with interaction weights between attribute pairs.
#     - Temp: Temperature parameter controlling randomness.
#     - atts: List of all attributes.

#     Returns:
#     - ps: Array of transition probabilities for each option.
#     """

#     def energy(beliefs):
#         return np.sum([
#             - BN_ag[(a1, a2)] * beliefs[a1] * beliefs[a2]
#             for a1, a2 in combinations(atts, 2)
#             if a1 == att or a2 == att
#         ])

#     # Original energy
#     H0 = energy(beliefs)

#     # Energy for each option
#     H = []
#     original_value = beliefs[att]  # Save original to restore later
#     for opt in options:
#         beliefs[att] = opt
#         H.append(energy(beliefs))
#     beliefs[att] = original_value  # Restore original value

#     delH = np.array(H) - H0
#     exp_term = 1 / (1 + np.exp(delH / Temp))
#     ps = exp_term / np.sum(exp_term)
#     return ps



def social_energy(belief, social_beliefs, social_edge_weight):
    return np.sum([
        - social_edge_weight * belief * sb for sb in social_beliefs
    ])

def energy(beliefs, att, atts, BN_ag ):
    return np.sum([
        - BN_ag[(a1, a2)] * beliefs[a1] * beliefs[a2]
        for a1, a2 in combinations(atts, 2)
        if a1 == att or a2 == att
    ])


def glauber_probabilities_withSocial(att, options, beliefs, BN_ag, atts, social_beliefs, social_edge_weight, beta_pers=1, beta_soc=1):
    """
    Compute Glauber transition probabilities for a specific attribute.

    Parameters:
    - att: The attribute being updated.
    - options: Possible values for the attribute.
    - beliefs: Current belief states for all attributes.
    - BN_ag: A dictionary with interaction weights between attribute pairs.
    - beta_pers: attention parameter controlling randomness; 1/T to personal dissonance
    - beta_soc: attention parameter controlling randomness; 1/attention to social dissonance
    - atts: List of all attributes.
    - social_beliefs: list of social beliefs on att
    - social_edge_weight: The weight of the social belief.

    Returns:
    - ps: Array of transition probabilities for each option.
    """

    # Original energy
    original_value = beliefs[att]  
    H0 = energy(beliefs, att, atts, BN_ag )
    H_soc_0 = social_energy(original_value, social_beliefs, social_edge_weight)

    # Energy for each option
    H = []
    H_soc = []
    for opt in options:
        beliefs[att] = opt
        H.append(energy(beliefs, att, atts, BN_ag))
        H_soc.append(social_energy(opt, social_beliefs, social_edge_weight))
    beliefs[att] = original_value  # Restore original value

    delH = beta_pers * (np.array(H)  - H0) + beta_soc * (np.array(H_soc) - H_soc_0)
    exp_term = 1 / (1+np.exp(delH))
    ps = exp_term / np.sum(exp_term)
    
    return ps
